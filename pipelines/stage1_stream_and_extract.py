import io
import os
import shutil
import zipfile
import numpy as np
import pandas as pd
import pylidc as pl
from pylidc.utils import consensus
import requests
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & DIRECTORY SETUP
# ==========================================
TARGET_NODULES = 350  # ~350 valid nodules (sufficient for course project)
OUTPUT_DIR = "./lidc_processed"
NODULE_DIR = os.path.join(OUTPUT_DIR, "nodules")
TEMP_DICOM_DIR = os.path.join(OUTPUT_DIR, "temp_dicom")

os.makedirs(NODULE_DIR, exist_ok=True)
os.makedirs(TEMP_DICOM_DIR, exist_ok=True)

# Automatically configure pylidc configuration file (~/.pylidcrc)
pylidc_config_path = os.path.expanduser("~/.pylidcrc")
with open(pylidc_config_path, "w") as f:
  f.write(f"[dicom]\npath = {os.path.abspath(TEMP_DICOM_DIR)}\nwarn = False\n")


# ==========================================
# 2. TCIA REST API DOWNLOAD FUNCTION
# ==========================================
def download_ct_series(series_instance_uid, target_folder):
  """Streams a single CT series zip from TCIA public REST API and unzips it."""
  url = f"https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID={series_instance_uid}"
  try:
    response = requests.get(url, stream=True, timeout=60)
    if response.status_code == 200:
      with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(target_folder)
      return True
    else:
      return False
  except Exception as e:
    print(f"\n[Warning] Failed to download series {series_instance_uid}: {e}")
    return False


# ==========================================
# 3. 3D ROI EXTRACTION & NORMALIZATION
# ==========================================
def crop_32x32x32(volume, center_zyx):
  """Extracts a 32x32x32 cube centered at (z, y, x).

  Pads with -1000 (air) if bounding box touches volume borders.
  """
  cz, cy, cx = center_zyx
  z_min, z_max = cz - 16, cz + 16
  y_min, y_max = cy - 16, cy + 16
  x_min, x_max = cx - 16, cx + 16

  d, h, w = volume.shape  # volume is (Z, Y, X)

  # Compute slices inside boundaries
  z_start, z_end = max(0, z_min), min(d, z_max)
  y_start, y_end = max(0, y_min), min(h, y_max)
  x_start, x_end = max(0, x_min), min(w, x_max)

  cropped = volume[z_start:z_end, y_start:y_end, x_start:x_end]

  # Pad with -1000 HU (background air) if cropped shape < 32
  pad_z = (max(0, -z_min), max(0, z_max - d))
  pad_y = (max(0, -y_min), max(0, y_max - h))
  pad_x = (max(0, -x_min), max(0, x_max - w))

  cube = np.pad(
      cropped,
      (pad_z, pad_y, pad_x),
      mode="constant",
      constant_values=-1000.0,
  )
  return cube[:32, :32, :32]


# ==========================================
# 4. MAIN STREAMING & PREPROCESSING LOOP
# ==========================================
records = []
nodule_count = 0

# Query all scans registered in pylidc's built-in SQLite catalog
scans = pl.query(pl.Scan).all()
print(
    f"Found {len(scans)} scans in LIDC database. Starting streaming"
    " extraction..."
)

for scan in tqdm(scans, desc="Processing Scans"):
  if nodule_count >= TARGET_NODULES:
    print(f"\n[Done] Reached target of {TARGET_NODULES} nodules.")
    break

  # Step A: Check annotations before downloading to avoid wasted network traffic
  nodule_clusters = scan.cluster_annotations()
  valid_clusters = []

  for cluster in nodule_clusters:
    # Filter 1: Require at least 2 radiologist consensus for clean labels
    if len(cluster) < 2:
      continue

    # Filter 2: Calculate mean malignancy and exclude indeterminate cases (score == 3.0)
    malignancies = [ann.malignancy for ann in cluster]
    mean_mal = float(np.mean(malignancies))
    if mean_mal == 3.0:
      continue

    valid_clusters.append((cluster, mean_mal))

  if not valid_clusters:
    continue  # Skip scan if it contains no usable nodules

  # Step B: Download ONLY this specific CT scan from TCIA
  shutil.rmtree(TEMP_DICOM_DIR, ignore_errors=True)
  os.makedirs(TEMP_DICOM_DIR, exist_ok=True)

  success = download_ct_series(scan.series_instance_uid, TEMP_DICOM_DIR)
  if not success:
    continue

  # Step C: Load 3D Volume into Memory
  try:
    # pylidc loads slices as (Y, X, Z); we transpose to (Z, Y, X) for standard 3D CNN depth-first format
    vol_yxz = scan.to_volume(verbose=False)
    vol_zyx = np.transpose(vol_yxz, (2, 0, 1))
  except Exception as e:
    shutil.rmtree(TEMP_DICOM_DIR, ignore_errors=True)
    continue

  # Step D: Extract Nodule Cubes and Baseline Features
  for cluster, mean_mal in valid_clusters:
    try:
      cmask, cbbox, _ = consensus(cluster, clevel=0.5, verbose=False)

      # Find centroid from consensus bounding box
      # cbbox indices in pylidc are (slice_y, slice_x, slice_z)
      cy = int(0.5 * (cbbox[0].start + cbbox[0].stop))
      cx = int(0.5 * (cbbox[1].start + cbbox[1].stop))
      cz = int(0.5 * (cbbox[2].start + cbbox[2].stop))

      # Crop 32x32x32 cube (Raw Hounsfield Units)
      cube_hu = crop_32x32x32(vol_zyx, (cz, cy, cx))

      # Normalize: HU [-1000, 400] -> [0.0, 1.0]
      cube_clipped = np.clip(cube_hu, -1000.0, 400.0)
      cube_norm = (cube_clipped + 1000.0) / 1400.0
      cube_norm = cube_norm.astype(np.float32)

      # Binary Malignancy Label (0: Benign, 1: Malignant)
      label = 0 if mean_mal < 3.0 else 1

      # Save 3D numpy cube
      nodule_filename = f"nodule_{nodule_count:04d}.npy"
      np.save(os.path.join(NODULE_DIR, nodule_filename), cube_norm)

      # Extract 10 Basic Statistical/Morphological Features for Classical ML Baseline
      feat_mean = float(np.mean(cube_hu))
      feat_std = float(np.std(cube_hu))
      feat_min = float(np.min(cube_hu))
      feat_max = float(np.max(cube_hu))
      feat_median = float(np.median(cube_hu))
      feat_q25 = float(np.percentile(cube_hu, 25))
      feat_q75 = float(np.percentile(cube_hu, 75))
      feat_solid_voxels = int(np.sum(cube_hu > -400.0))
      feat_num_annotators = len(cluster)
      feat_avg_subtlety = float(np.mean([ann.subtlety for ann in cluster]))

      records.append({
          "nodule_id": nodule_filename,
          "patient_id": scan.patient_id,
          "label": label,
          "mean_malignancy": mean_mal,
          "f_mean": feat_mean,
          "f_std": feat_std,
          "f_min": feat_min,
          "f_max": feat_max,
          "f_median": feat_median,
          "f_q25": feat_q25,
          "f_q75": feat_q75,
          "f_solid_voxels": feat_solid_voxels,
          "f_annotators": feat_num_annotators,
          "f_subtlety": feat_avg_subtlety,
      })

      nodule_count += 1
      if nodule_count >= TARGET_NODULES:
        break

    except Exception as e:
      continue

  # Step E: Clean up temporary scan files immediately to free disk space
  shutil.rmtree(TEMP_DICOM_DIR, ignore_errors=True)

# ==========================================
# 5. PATIENT-STRATIFIED SPLIT & CSV EXPORT
# ==========================================
df = pd.DataFrame(records)

# 70% Train, 15% Val, 15% Test strictly split by PatientID (No data leakage)
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_val_idx, test_idx = next(gss_test.split(df, groups=df["patient_id"]))

df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
train_idx, val_idx = next(
    gss_val.split(df_train_val, groups=df_train_val["patient_id"])
)

df.loc[df.index.isin(test_idx), "split"] = "test"
df.loc[
    df.index.isin(train_val_idx[train_idx]), "split"
] = "train"
df.loc[df.index.isin(train_val_idx[val_idx]), "split"] = "val"

metadata_csv = os.path.join(OUTPUT_DIR, "metadata.csv")
df.to_csv(metadata_csv, index=False)

print("\n================ DATASET READY ================")
print(f"Total Nodules Extracted: {len(df)}")
print(f"Class Breakdown: {df['label'].value_counts().to_dict()}")
print(f"Split Breakdown:\n{df['split'].value_counts()}")
print(f"Saved Metadata: {metadata_csv}")
print(f"Saved Nodule Cubes: {NODULE_DIR}")
print("================================================")

