#!/usr/bin/env python3
"""Universal inference and explainability testing script for 2026-hybrid-xai-lung-nodule.

Accepts .npy, .nii, .nii.gz, or 2D image (.png, .jpg) inputs, runs 3D ResNet-18,
and outputs prediction metrics and a Grad-CAM visualization.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18

# Device configuration
if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
  DEVICE = torch.device("mps")
else:
  DEVICE = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "results", "best_3d_resnet18.pth")


# ==============================================================================
# 1. MODEL ARCHITECTURE & 3D GRAD-CAM ENGINE
# ==============================================================================
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


class GradCAM3D:

  def __init__(self, model, target_layer):
    self.model = model
    self.target_layer = target_layer
    self.gradients = None
    self.activations = None
    self.target_layer.register_forward_hook(
        lambda m, i, o: setattr(self, "activations", o.detach())
    )
    self.target_layer.register_full_backward_hook(
        lambda m, gi, go: setattr(self, "gradients", go[0].detach())
    )

  def generate(self, input_tensor):
    self.model.eval()
    self.model.zero_grad()
    logit = self.model(input_tensor)
    prob = torch.sigmoid(logit).item()

    logit.backward()
    weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
    cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(
        cam, size=(32, 32, 32), mode="trilinear", align_corners=False
    )

    cam_vol = cam.squeeze().cpu().numpy()
    c_min, c_max = cam_vol.min(), cam_vol.max()
    if c_max - c_min > 1e-8:
      cam_vol = (cam_vol - c_min) / (c_max - c_min)
    else:
      cam_vol = np.zeros_like(cam_vol)

    return cam_vol, prob


# ==============================================================================
# 2. ADAPTIVE INPUT LOADER (SUPPORTS .NPY, .NII, .PNG, .JPG)
# ==============================================================================
def load_and_preprocess_any(file_path):
  """Converts any input file into a normalized (32, 32, 32) float32 array in [0, 1]."""
  ext = file_path.lower()

  # Case A: NumPy binary (.npy)
  if ext.endswith(".npy"):
    arr = np.load(file_path)
    if arr.ndim == 3:
      # Crop or pad to 32x32x32
      d, h, w = arr.shape
      cd, ch, cw = d // 2, h // 2, w // 2
      z_s, z_e = max(0, cd - 16), min(d, cd + 16)
      y_s, y_e = max(0, ch - 16), min(h, ch + 16)
      x_s, x_e = max(0, cw - 16), min(w, cw + 16)
      crop = arr[z_s:z_e, y_s:y_e, x_s:x_e]
      pad_z = (max(0, 16 - cd), max(0, cd + 16 - d))
      pad_y = (max(0, 16 - ch), max(0, ch + 16 - h))
      pad_x = (max(0, 16 - cw), max(0, cw + 16 - w))
      vol = np.pad(crop, (pad_z, pad_y, pad_x), mode="edge")[:32, :32, :32]
    else:
      raise ValueError(f"NumPy array must be 3D, got shape {arr.shape}")

  # Case B: NIfTI 3D CT scan (.nii or .nii.gz)
  elif ext.endswith(".nii") or ext.endswith(".nii.gz"):
    import nibabel as nib

    nii = nib.load(file_path)
    vol_raw = nii.get_fdata(dtype=np.float32)
    # Transpose (X, Y, Z) -> (Z, Y, X)
    if vol_raw.ndim == 3:
      vol_zyx = np.transpose(vol_raw, (2, 1, 0))
    else:
      vol_zyx = vol_raw
    # Window to HU [-1000, 400] and scale to [0, 1]
    vol_clipped = np.clip(vol_zyx, -1000.0, 400.0)
    vol_norm = (vol_clipped + 1000.0) / 1400.0
    # Center crop 32x32x32
    d, h, w = vol_norm.shape
    cd, ch, cw = d // 2, h // 2, w // 2
    vol = vol_norm[
        max(0, cd - 16) : cd + 16,
        max(0, ch - 16) : ch + 16,
        max(0, cw - 16) : cw + 16,
    ]
    if vol.shape != (32, 32, 32):
      vol = np.pad(
          vol,
          (
              (0, max(0, 32 - vol.shape[0])),
              (0, max(0, 32 - vol.shape[1])),
              (0, max(0, 32 - vol.shape[2])),
          ),
          mode="constant",
      )[:32, :32, :32]

  # Case C: 2D image (PNG, JPG, JPEG)
  elif ext.endswith(".png") or ext.endswith(".jpg") or ext.endswith(".jpeg"):
    img = Image.open(file_path).convert("L")  # Grayscale
    img = img.resize((32, 32), Image.Resampling.BILINEAR)
    arr_2d = np.array(img, dtype=np.float32) / 255.0
    # Expand into pseudo-3D volume by repeating slice across the depth axis
    vol = np.repeat(arr_2d[np.newaxis, :, :], 32, axis=0)

  else:
    raise ValueError(f"Unsupported format: {ext}")

  # Ensure strict range and type
  vol = np.clip(vol, 0.0, 1.0).astype(np.float32)
  return vol


# ==============================================================================
# 3. MAIN INFERENCE EXECUTION
# ==============================================================================
def main():
  parser = argparse.ArgumentParser(
      description="Test 3D ResNet-18 on any lung image"
  )
  parser.add_argument(
      "--input",
      type=str,
      default="./custom_test_inputs/sample_malignant_3d.npy",
      help="Path to input image/volume (.npy, .nii.gz, .png, .jpg)",
  )
  args = parser.parse_args()

  if not os.path.exists(args.input):
    print(f"Error: File not found at '{args.input}'")
    print(
        "Tip: Run 'python download_test_samples.py' first to download ready"
        " test files."
    )
    return

  print("=" * 65)
  print(f"TESTING FILE: {os.path.basename(args.input)}")
  print("=" * 65)

  # Load and adapt input
  volume_3d = load_and_preprocess_any(args.input)
  print(f"Input successfully adapted to 3D tensor shape: {volume_3d.shape}")

  # Load model
  model = build_3d_resnet().to(DEVICE)
  checkpoint = torch.load(
      CHECKPOINT_PATH, map_location=DEVICE, weights_only=False
  )
  model.load_state_dict(checkpoint["model_state_dict"])
  print(
      f"Model loaded: {CHECKPOINT_PATH} (Trained ROC-AUC:"
      f" {checkpoint.get('val_auc', 0.0):.4f})"
  )

  # Attach Grad-CAM to layer4
  gradcam = GradCAM3D(model, model.layer4[-1])

  # Forward + Backward Pass
  input_tensor = (
      torch.from_numpy(volume_3d).unsqueeze(0).unsqueeze(0).to(DEVICE)
  )
  cam_volume, prob = gradcam.generate(input_tensor)

  prediction = "MALIGNANT" if prob >= 0.5 else "BENIGN"
  confidence = prob if prob >= 0.5 else (1.0 - prob)

  print("-" * 65)
  print(f"PREDICTED CLASS   : {prediction}")
  print(f"MALIGNANT CHANCE  : {prob * 100:.2f}%")
  print(f"CONFIDENCE LEVEL  : {confidence * 100:.2f}%")
  print("-" * 65)

  # Generate 3-Panel Visual Overlay
  slice_idx = 16  # Center axial slice
  ct_slice = volume_3d[slice_idx, :, :]
  cam_slice = cam_volume[slice_idx, :, :]

  fig, axes = plt.subplots(1, 3, figsize=(12, 4))

  # Panel 1: Original Input Slice
  axes[0].imshow(ct_slice, cmap="gray", vmin=0.0, vmax=1.0)
  axes[0].set_title(
      f"Input CT Slice (Z={slice_idx})", fontsize=11, fontweight="bold"
  )
  axes[0].axis("off")

  # Panel 2: 3D Grad-CAM Activation Heatmap
  axes[1].imshow(cam_slice, cmap="jet", vmin=0.0, vmax=1.0)
  axes[1].set_title("3D Grad-CAM Heatmap", fontsize=11, fontweight="bold")
  axes[1].axis("off")

  # Panel 3: Alpha Overlay
  axes[2].imshow(ct_slice, cmap="gray", vmin=0.0, vmax=1.0)
  axes[2].imshow(cam_slice, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
  color = "darkred" if prediction == "MALIGNANT" else "darkgreen"
  axes[2].set_title(
      f"Prediction: {prediction} ({prob * 100:.1f}%)",
      fontsize=11,
      fontweight="bold",
      color=color,
  )
  axes[2].axis("off")

  plt.tight_layout()
  out_fig = os.path.join(BASE_DIR, "results", "custom_test_result.png")
  plt.savefig(out_fig, dpi=300, bbox_inches="tight")
  plt.close()

  print(f"[Visual Saved] Grad-CAM overlay written to: {out_fig}")
  print("=" * 65)


if __name__ == "__main__":
  main()
