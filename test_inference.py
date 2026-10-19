#!/usr/bin/env python3
"""Sanity test and verification script for 2026-hybrid-xai-lung-nodule.

Verifies checkpoint loading, pipeline forward pass, and test set inference.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

# Setup paths and device
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "lidc_processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "best_3d_resnet18.pth")
TEST_CSV_PATH = os.path.join(PROCESSED_DIR, "test.csv")
PATCHES_DIR = os.path.join(PROCESSED_DIR, "patches")

if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
  DEVICE = torch.device("mps")
else:
  DEVICE = torch.device("cpu")


def build_3d_resnet():
  model = r3d_18(weights=None)
  orig_stem = model.stem[0]
  model.stem[0] = nn.Conv3d(
      in_channels=1,
      out_channels=orig_stem.out_channels,
      kernel_size=orig_stem.kernel_size,
      stride=orig_stem.stride,
      padding=orig_stem.padding,
      bias=False,
  )
  model.fc = nn.Linear(model.fc.in_features, 1)
  return model


def main():
  print("=" * 65)
  print("RUNNING PIPELINE VERIFICATION TEST")
  print("=" * 65)
  print(f"[1/4] Compute Device: {DEVICE}")

  # 1. Check Artifacts
  assert os.path.exists(CHECKPOINT_PATH), (
      f"Checkpoint missing: {CHECKPOINT_PATH}"
  )
  assert os.path.exists(TEST_CSV_PATH), f"Test CSV missing: {TEST_CSV_PATH}"
  print("[2/4] Found model checkpoint and test metadata.")

  # 2. Instantiate & Load Weights
  model = build_3d_resnet().to(DEVICE)
  checkpoint = torch.load(
      CHECKPOINT_PATH, map_location=DEVICE, weights_only=False
  )
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()
  val_auc = checkpoint.get("val_auc", "N/A")
  print(
      f"[3/4] Successfully loaded 3D ResNet-18 weights (Validation ROC-AUC:"
      f" {val_auc if isinstance(val_auc, str) else f'{val_auc:.4f}'})"
  )

  # 3. Shape & Forward Pass Check
  dummy_input = torch.randn(1, 1, 32, 32, 32, device=DEVICE)
  with torch.no_grad():
    dummy_out = model(dummy_input)
    assert dummy_out.shape == (
        1,
        1,
    ), f"Unexpected output shape: {dummy_out.shape}"
  print("      Dummy forward pass (1, 1, 32, 32, 32) -> Output (1, 1) PASSED.")

  # 4. Run Live Sample Inference from Test Set
  print("\n[4/4] Running Live Sample Inferences from Test Set:")
  print("-" * 65)
  print(
      f"{'Patch File':<16} | {'GT':<10} | {'Prediction':<10} |"
      f" {'Prob':<8} | {'Latency':<8}"
  )
  print("-" * 65)

  df_test = pd.read_csv(TEST_CSV_PATH).sample(
      n=min(5, len(pd.read_csv(TEST_CSV_PATH))), random_state=42
  )

  correct_count = 0
  with torch.no_grad():
    for _, row in df_test.iterrows():
      patch_file = row["patch_filename"]
      patch_path = os.path.join(PATCHES_DIR, patch_file)
      gt_label = int(row["label"])

      cube = np.load(patch_path)
      tensor = (
          torch.from_numpy(cube).unsqueeze(0).unsqueeze(0).to(DEVICE)
      )  # (1, 1, 32, 32, 32)

      t0 = time.perf_counter()
      logit = model(tensor)
      prob = torch.sigmoid(logit).item()
      latency_ms = (time.perf_counter() - t0) * 1000.0

      pred_label = 1 if prob >= 0.5 else 0
      gt_str = "Malignant" if gt_label == 1 else "Benign"
      pred_str = "Malignant" if pred_label == 1 else "Benign"

      if pred_label == gt_label:
        correct_count += 1

      print(
          f"{patch_file:<16} | {gt_str:<10} | {pred_str:<10} | {prob:<8.3f} |"
          f" {latency_ms:.1f} ms"
      )

  print("-" * 65)
  print(f"Sample Accuracy: {correct_count}/{len(df_test)} correct.")
  print("=" * 65)
  print("VERIFICATION COMPLETE: System is fully operational.\n")


if __name__ == "__main__":
  main()
