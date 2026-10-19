#!/usr/bin/env python3
"""Downloads open-access test lung CT samples (3D volumes and 2D slices)

into ./custom_test_inputs/ for independent model testing.
"""

import io
import os
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "./custom_test_inputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("DOWNLOADING TEST LUNG CT SAMPLES")
print("=" * 65)

# 1. Download official open benchmark 3D nodules (NoduleMNIST3D archive)
BENCHMARK_URL = (
    "https://zenodo.org/records/6496656/files/nodulemnist3d.npz?download=1"
)
npz_path = os.path.join(OUTPUT_DIR, "nodulemnist3d_temp.npz")

print("[1/3] Downloading public lung nodule benchmark volumes from Zenodo...")
try:
  urllib.request.urlretrieve(BENCHMARK_URL, npz_path)
  data = np.load(npz_path)

  test_images = data["test_images"]  # shape (N, 28, 28, 28)
  test_labels = data["test_labels"].ravel()

  # Find confirmed Benign (0) and Malignant (1) sample indices
  benign_idx = np.where(test_labels == 0)[0][0]
  malignant_idx = np.where(test_labels == 1)[0][0]

  # Pad (28, 28, 28) to our model's required (32, 32, 32)
  for name, idx, label_str in [
      ("sample_benign_3d.npy", benign_idx, "Benign"),
      ("sample_malignant_3d.npy", malignant_idx, "Malignant"),
  ]:
    raw_vol = test_images[idx].astype(np.float32) / 255.0
    vol_32 = np.pad(raw_vol, ((2, 2), (2, 2), (2, 2)), mode="edge")
    save_path = os.path.join(OUTPUT_DIR, name)
    np.save(save_path, vol_32)
    print(f"      Saved {label_str} 3D Volume -> {save_path} (shape: {vol_32.shape})")

  # 2. Extract and save one 2D slice as a PNG to test arbitrary 2D image support
  slice_2d = vol_32[16, :, :]
  png_path = os.path.join(OUTPUT_DIR, "sample_lung_slice_2d.png")
  plt.imsave(png_path, slice_2d, cmap="gray")
  print(f"[2/3] Saved 2D Image Slice -> {png_path}")

  # Clean up temporary archive
  if os.path.exists(npz_path):
    os.remove(npz_path)

except Exception as e:
  print(f"[Warning] Direct benchmark download failed ({e}).")
  print("          Generating synthetic geometric nodule volumes as fallback...")

  # Fallback: Synthetic ellipsoidal nodule simulation
  grid = np.linspace(-1, 1, 32)
  z, y, x = np.meshgrid(grid, grid, grid, indexing="ij")

  # Benign: small smooth spherical lesion
  r_benign = np.sqrt(x**2 + y**2 + z**2)
  benign_vol = (
      np.exp(-(r_benign**2) / (2 * 0.15**2)).astype(np.float32) * 0.8
  )
  np.save(os.path.join(OUTPUT_DIR, "sample_benign_3d.npy"), benign_vol)

  # Malignant: irregular, larger spiculated lesion
  r_mal = np.sqrt(
      (x / 0.35) ** 2
      + (y / 0.25) ** 2
      + (z / 0.25) ** 2
      + 0.1 * np.sin(10 * x) * np.cos(10 * y)
  )
  mal_vol = np.clip(1.0 - r_mal, 0, 1).astype(np.float32)
  np.save(os.path.join(OUTPUT_DIR, "sample_malignant_3d.npy"), mal_vol)

  plt.imsave(
      os.path.join(OUTPUT_DIR, "sample_lung_slice_2d.png"),
      mal_vol[16, :, :],
      cmap="gray",
  )

print("[3/3] Ready! Test files are located in ./custom_test_inputs/")
print("=" * 65)
