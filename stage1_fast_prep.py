#!/usr/bin/env python3
"""Stage 1: Production Preprocessing Pipeline for 2026-hybrid-xai-lung-nodule.

Processes 1176 LIDC-IDRI 3D CT NIfTI patches and metadata into model-ready
tensors, handcrafted tabular features, and scan-grouped splits.
"""

from collections import Counter
import json
import os
import sys
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION & REPRODUCIBILITY
# ==============================================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SOURCE_DIR = os.path.join(BASE_DIR, "data_source")
NIFTI_DIR = os.path.join(DATA_SOURCE_DIR, "LIDC-IDRI")
METADATA_PATH = os.path.join(DATA_SOURCE_DIR, "LIDC-IDRI_1176.npy")

PROCESSED_DIR = os.path.join(BASE_DIR, "lidc_processed")
PATCHES_DIR = os.path.join(PROCESSED_DIR, "patches")

# Spatial parameters (PyTorch layout: Depth, Height, Width)
CROP_D, CROP_H, CROP_W = 32, 32, 32
HU_MIN, HU_MAX = -1000.0, 400.0

# Split allocations
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ==============================================================================
# 1. STRICT VALIDATION & PARITY AUDIT
# ==============================================================================
def validate_inputs():
  """Validates existence, parity, and integrity of raw inputs."""
  print("=" * 70)
  print("STAGE 1: VERIFYING DATASET INTEGRITY & PARITY")
  print("=" * 70)

  if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError(f"Metadata file missing: {METADATA_PATH}")

  if not os.path.exists(NIFTI_DIR):
    raise FileNotFoundError(f"NIfTI directory missing: {NIFTI_DIR}")

  raw_meta = np.load(METADATA_PATH, allow_pickle=True)
  print(f"[OK] Loaded metadata records: {len(raw_meta)}")

  nii_files = [f for f in os.listdir(NIFTI_DIR) if f.endswith(".nii.gz")]
  print(f"[OK] Discovered .nii.gz files: {len(nii_files)}")

  if len(raw_meta) != 1176 or len(nii_files) != 1176:
    raise ValueError(
        f"Strict count parity check failed! Expected 1176 records, got "
        f"Metadata={len(raw_meta)}, NIfTI={len(nii_files)}"
    )

  # Check duplicate filenames in metadata
  meta_filenames = [
      os.path.basename(item["Filename"].strip()) for item in raw_meta
  ]
  filename_counts = Counter(meta_filenames)
  duplicates = [f for f, count in filename_counts.items() if count > 1]
  if duplicates:
    raise ValueError(
        f"Duplicate Filename entries detected in metadata: {duplicates[:5]}"
    )

  # Check 1-to-1 bijection
  meta_set = set(meta_filenames)
  disk_set = set(nii_files)

  missing_on_disk = meta_set - disk_set
  if missing_on_disk:
    raise FileNotFoundError(
        f"{len(missing_on_disk)} metadata records have no file on disk."
        f" Example: {list(missing_on_disk)[:3]}"
    )

  orphan_on_disk = disk_set - meta_set
  if orphan_on_disk:
    raise ValueError(
        f"{len(orphan_on_disk)} files on disk have no metadata record."
        f" Example: {list(orphan_on_disk)[:3]}"
    )

  print(
      "[OK] 1-to-1 bijection verified: All 1176 records match corresponding"
      " NIfTI files.\n"
  )
  return raw_meta


# ==============================================================================
# 2. FEATURE EXTRACTION & VOXEL PREPROCESSING
# ==============================================================================
def extract_handcrafted_features(cube_hu):
  """Computes 10 first-order volumetric intensity statistics from the HU cube."""
  flat = cube_hu.ravel()
  mean_val = float(np.mean(flat))
  std_val = float(np.std(flat))
  min_val = float(np.min(flat))
  max_val = float(np.max(flat))
  median_val = float(np.median(flat))
  q25_val = float(np.percentile(flat, 25))
  q75_val = float(np.percentile(flat, 75))
  iqr_val = q75_val - q25_val
  skew_val = float(skew(flat))
  kurt_val = float(kurtosis(flat))

  total_voxels = float(cube_hu.size)
  # Solid tissue threshold (soft tissue vs lung parenchyma / air)
  solid_fraction = float(np.sum(cube_hu > -400.0) / total_voxels)
  # Hyperdense / Calcification fraction
  hyperdense_fraction = float(np.sum(cube_hu > 100.0) / total_voxels)

  return {
      "feat_mean_hu": mean_val,
      "feat_std_hu": std_val,
      "feat_min_hu": min_val,
      "feat_max_hu": max_val,
      "feat_median_hu": median_val,
      "feat_q25_hu": q25_val,
      "feat_q75_hu": q75_val,
      "feat_iqr_hu": iqr_val,
      "feat_skewness": skew_val,
      "feat_kurtosis": kurt_val,
      "feat_solid_fraction": solid_fraction,
      "feat_hyperdense_fraction": hyperdense_fraction,
  }


# ==============================================================================
# 3. MAIN PREPROCESSING PIPELINE
# ==============================================================================
def run_pipeline():
  raw_meta = validate_inputs()

  # Clean and establish target directories
  os.makedirs(PATCHES_DIR, exist_ok=True)

  # Pre-scan label distribution
  print("=" * 70)
  print("ANALYZING MALIGNANCY SCORES & CONSENSUS")
  print("=" * 70)

  mean_mal_scores = []
  indeterminate_count = 0
  benign_count = 0
  malignant_count = 0

  for item in raw_meta:
    scores = item["Malignancy"]
    m_bar = float(np.mean(scores))
    mean_mal_scores.append(m_bar)
    if m_bar == 3.0:
      indeterminate_count += 1
    elif m_bar < 3.0:
      benign_count += 1
    else:
      malignant_count += 1

  print(f"Total Nodule Annotations: {len(raw_meta)}")
  print(f"  - Benign Candidates      (m_bar < 3.0): {benign_count}")
  print(f"  - Malignant Candidates   (m_bar > 3.0): {malignant_count}")
  print(
      f"  - Indeterminate Excluded (m_bar == 3.0): {indeterminate_count} (Will"
      " be omitted from training)"
  )
  print(f"Total Valid Binary Cases: {benign_count + malignant_count}\n")

  # Process volumes one-by-one (Memory bounded)
  print("=" * 70)
  print("PROCESSING 3D NIFTI VOLUMES (BOUNDED MEMORY)")
  print("=" * 70)

  records = []
  excluded_records = []
  corrupted_files = []

  for idx, item in enumerate(tqdm(raw_meta, desc="Processing patches")):
    filename = os.path.basename(item["Filename"].strip())
    filepath = os.path.join(NIFTI_DIR, filename)

    scores = item["Malignancy"]
    m_bar = float(np.mean(scores))

    # Base identifier payload for traceability
    trace_info = {
        "sample_index": idx,
        "original_filename": filename,
        "series_instance_uid": str(item["SeriesInstanceUID"]).strip(),
        "voxel_coord_x": int(item["VoxelCoordX"]),
        "voxel_coord_y": int(item["VoxelCoordY"]),
        "voxel_coord_z": int(item["VoxelCoordZ"]),
        "malignancy_scores": json.dumps(
            [int(s) for s in scores]
            if isinstance(scores, (list, np.ndarray))
            else [int(scores)]
        ),
        "mean_malignancy": round(m_bar, 4),
        "annotator_count": len(scores)
        if isinstance(scores, (list, np.ndarray))
        else 1,
    }

    # Handle indeterminate exclusion
    if m_bar == 3.0:
      trace_info["exclusion_reason"] = "Indeterminate (mean_malignancy == 3.0)"
      excluded_records.append(trace_info)
      continue

    binary_label = 0 if m_bar < 3.0 else 1

    # Load 3D Volume
    try:
      nii_obj = nib.load(filepath)
      vol = nii_obj.get_fdata(dtype=np.float32)
    except Exception as e:
      corrupted_files.append((filename, str(e)))
      continue

    # Spatial parity check
    if vol.shape != (128, 128, 64):
      raise ValueError(
          f"Unexpected volume shape {vol.shape} in {filename}. Expected (128,"
          " 128, 64)."
      )

    # Transpose nibabel (X, Y, Z) -> PyTorch Depth-first (Z, Y, X)
    # Resulting shape: (64, 128, 128) -> Depth=64, Height=128, Width=128
    vol_zyx = np.transpose(vol, (2, 1, 0))

    # Crop center 32x32x32 ROI:
    # Depth: center 32 is [16:48] of 64
    # Height: center 32 is [48:80] of 128
    # Width: center 32 is [48:80] of 128
    crop_3d = vol_zyx[16:48, 48:80, 48:80]

    if crop_3d.shape != (32, 32, 32):
      raise RuntimeError(f"Crop slice failure on {filename}: {crop_3d.shape}")

    # Compute handcrafted features from raw HU values
    hu_stats = extract_handcrafted_features(crop_3d)

    # Apply HU Windowing and Min-Max Normalization: [-1000, 400] -> [0.0, 1.0]
    clipped = np.clip(crop_3d, HU_MIN, HU_MAX)
    normalized = (clipped - HU_MIN) / (HU_MAX - HU_MIN)
    normalized = normalized.astype(np.float32)

    # Save isolated processed .npy patch
    out_patch_name = f"patch_{idx:04d}.npy"
    out_patch_path = os.path.join(PATCHES_DIR, out_patch_name)
    np.save(out_patch_path, normalized)

    # Consensus metadata features
    diameters = item.get("Diameter", [0.0])
    calcifications = item.get("Calcification", [0.0])
    textures = item.get("Texture", [0.0])

    record = {
        "patch_filename": out_patch_name,
        **trace_info,
        "label": binary_label,
        **hu_stats,
        "meta_mean_diameter": float(np.mean(diameters)),
        "meta_mean_calcification": float(np.mean(calcifications)),
        "meta_mean_texture": float(np.mean(textures)),
    }
    records.append(record)

  if corrupted_files:
    raise RuntimeError(
        f"Encountered corrupted NIfTI files: {corrupted_files}"
    )

  # ==============================================================================
  # 4. LEAKAGE-FREE SCAN-LEVEL GROUPED SPLIT
  # ==============================================================================
  print("\n" + "=" * 70)
  print("PERFORMING SCAN-LEVEL (SERIESINSTANCEUID) GROUPED SPLIT")
  print("=" * 70)

  df = pd.DataFrame(records)

  # Note on grouping:
  # SeriesInstanceUID groups all nodules originating from the exact same CT scan series.
  # This prevents physical CT scanner calibration and parenchymal noise leakage.
  unique_series = df["series_instance_uid"].nunique()
  print(
      f"Identified {unique_series} unique SeriesInstanceUIDs across"
      f" {len(df)} samples."
  )

  # Stage A: Separate Test Set (15%)
  gss_test = GroupShuffleSplit(
      n_splits=1, test_size=TEST_RATIO, random_state=RANDOM_SEED
  )
  train_val_idx, test_idx = next(
      gss_test.split(df, groups=df["series_instance_uid"])
  )

  df_train_val = df.iloc[train_val_idx].copy()
  df_test = df.iloc[test_idx].copy()

  # Stage B: Separate Validation Set (15% of total = ~17.65% of train_val)
  relative_val_ratio = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
  gss_val = GroupShuffleSplit(
      n_splits=1, test_size=relative_val_ratio, random_state=RANDOM_SEED
  )
  train_sub_idx, val_sub_idx = next(
      gss_val.split(df_train_val, groups=df_train_val["series_instance_uid"])
  )

  df_train = df_train_val.iloc[train_sub_idx].copy()
  df_val = df_train_val.iloc[val_sub_idx].copy()

  df_train["split"] = "train"
  df_val["split"] = "val"
  df_test["split"] = "test"

  df_final = pd.concat([df_train, df_val, df_test]).sort_values("sample_index")

  # Audit Group Disjointness (Zero Leakage Check)
  train_series = set(df_train["series_instance_uid"])
  val_series = set(df_val["series_instance_uid"])
  test_series = set(df_test["series_instance_uid"])

  leakage_tv = train_series.intersection(val_series)
  leakage_tt = train_series.intersection(test_series)
  leakage_vt = val_series.intersection(test_series)

  if leakage_tv or leakage_tt or leakage_vt:
    raise RuntimeError(
        f"Critical data leakage detected across splits! TV: {len(leakage_tv)},"
        f" TT: {len(leakage_tt)}, VT: {len(leakage_vt)}"
    )

  print("[PASS] Grouped split is strictly disjoint across series.")
  print(
      f"  - Train: {len(df_train)} samples ({df_train['label'].sum()} Malignant)"
  )
  print(f"  - Val:   {len(df_val)} samples ({df_val['label'].sum()} Malignant)")
  print(
      f"  - Test:  {len(df_test)} samples ({df_test['label'].sum()} Malignant)"
  )

  # ==============================================================================
  # 5. SERIALIZATION & DATASET SUMMARY
  # ==============================================================================
  metadata_csv_path = os.path.join(PROCESSED_DIR, "metadata.csv")
  train_csv_path = os.path.join(PROCESSED_DIR, "train.csv")
  val_csv_path = os.path.join(PROCESSED_DIR, "val.csv")
  test_csv_path = os.path.join(PROCESSED_DIR, "test.csv")
  summary_json_path = os.path.join(PROCESSED_DIR, "dataset_summary.json")

  df_final.to_csv(metadata_csv_path, index=False)
  df_train.to_csv(train_csv_path, index=False)
  df_val.to_csv(val_csv_path, index=False)
  df_test.to_csv(test_csv_path, index=False)

  summary = {
      "dataset_name": "LIDC-IDRI Nodule Subvolume Benchmark",
      "total_input_files": len(raw_meta),
      "total_processed_samples": len(df_final),
      "total_excluded_indeterminate": len(excluded_records),
      "class_distribution": {
          "benign_0": int((df_final["label"] == 0).sum()),
          "malignant_1": int((df_final["label"] == 1).sum()),
          "indeterminate_excluded": len(excluded_records),
      },
      "split_counts": {
          "train": len(df_train),
          "val": len(df_val),
          "test": len(df_test),
      },
      "series_counts": {
          "total_unique_series": unique_series,
          "train_series": len(train_series),
          "val_series": len(val_series),
          "test_series": len(test_series),
      },
      "patch_properties": {
          "spatial_shape": [CROP_D, CROP_H, CROP_W],
          "axis_order": "Depth, Height, Width (PyTorch standard C x D x H x W)",
          "dtype": "float32",
          "windowing": [HU_MIN, HU_MAX],
          "normalization": "min-max scaled to [0.0, 1.0]",
      },
      "grouping_limitation_declaration": (
          "Splits are strictly grouped by SeriesInstanceUID. In LIDC-IDRI,"
          " this prevents scan-level leakage for co-occurring nodules. In"
          " rare cases where one patient underwent multiple distinct scan"
          " series, patient-level cross-scan correlation is possible due to"
          " absence of explicit PatientID tags in this pre-extracted patch"
          " metadata."
      ),
  }

  with open(summary_json_path, "w") as f:
    json.dump(summary, f, indent=4)

  print("\n" + "=" * 70)
  print("STAGE 1 COMPLETED SUCCESSFULLY")
  print("=" * 70)
  print(f"Patches written: {len(df_final)} files in {PATCHES_DIR}")
  print(f"Artifacts saved in: {PROCESSED_DIR}")
  print(f"  ├── metadata.csv\n  ├── train.csv\n  ├── val.csv\n  ├── test.csv")
  print(f"  └── dataset_summary.json")
  print("=" * 70)


if __name__ == "__main__":
  run_pipeline()

