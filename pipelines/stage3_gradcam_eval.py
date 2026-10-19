#!/usr/bin/env python3
"""Stage 3: 3D Grad-CAM Explainability Pipeline for 2026-hybrid-xai-lung-nodule.

Loads results/best_3d_resnet18.pth, extracts 3D Grad-CAM heatmaps from layer4,
and generates multi-case axial slice overlay visualizations on held-out test
nodules.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18

# ==============================================================================
# 1. SETUP & DEVICE
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "lidc_processed")
PATCHES_DIR = os.path.join(PROCESSED_DIR, "patches")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, "best_3d_resnet18.pth")

if torch.cuda.is_available():
  DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
  DEVICE = torch.device("mps")
else:
  DEVICE = torch.device("cpu")

print(f"[Device] Running Stage 3 on: {DEVICE}")

if not os.path.exists(CHECKPOINT_PATH):
  raise FileNotFoundError(
      f"Model checkpoint not found at: {CHECKPOINT_PATH}. Run Stage 2 first!"
  )


# ==============================================================================
# 2. 3D MODEL ARCHITECTURE
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


# ==============================================================================
# 3. 3D GRAD-CAM HOOK ENGINE
# ==============================================================================
class GradCAM3D:

  def __init__(self, model, target_layer):
    self.model = model
    self.target_layer = target_layer
    self.gradients = None
    self.activations = None

    self._register_hooks()

  def _register_hooks(self):

    def forward_hook(module, input_val, output_val):
      self.activations = output_val.detach()

    def backward_hook(module, grad_input, grad_output):
      self.gradients = grad_output[0].detach()

    self.target_layer.register_forward_hook(forward_hook)
    self.target_layer.register_full_backward_hook(backward_hook)

  def generate_heatmap(self, input_tensor):
    """Computes a normalized 32x32x32 Grad-CAM volume for input_tensor (1, 1, 32, 32, 32)."""
    self.model.eval()
    self.model.zero_grad()

    # Forward pass
    output = self.model(input_tensor)
    prob = torch.sigmoid(output).item()

    # Backward pass with respect to logit
    output.backward()

    # Global Average Pooling of gradients across spatial dimensions (Depth, Height, Width)
    weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)

    # Weighted linear combination of activation maps
    cam_3d = torch.sum(weights * self.activations, dim=1, keepdim=True)
    cam_3d = F.relu(cam_3d)

    # Trilinear interpolation back to original input resolution (32, 32, 32)
    cam_upsampled = F.interpolate(
        cam_3d, size=(32, 32, 32), mode="trilinear", align_corners=False
    )
    cam_volume = cam_upsampled.squeeze().cpu().numpy()

    # Min-max normalization of heatmap
    cam_min, cam_max = cam_volume.min(), cam_volume.max()
    if cam_max - cam_min > 1e-8:
      cam_volume = (cam_volume - cam_min) / (cam_max - cam_min)
    else:
      cam_volume = np.zeros_like(cam_volume)

    return cam_volume, prob


# ==============================================================================
# 4. TEST CASE SELECTION & INFERENCE
# ==============================================================================
def run_stage3():
  print("=" * 70)
  print("STAGE 3: 3D GRAD-CAM EXPLAINABILITY EVALUATION")
  print("=" * 70)

  # Load model weights
  model = build_3d_resnet().to(DEVICE)
<<<<<<< HEAD
  checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
=======
  checkpoint = torch.load(
    CHECKPOINT_PATH, map_location=DEVICE, weights_only=False
  )
>>>>>>> c44f2c3910463f7bfca85a3f61b9428fa41af64c
  model.load_state_dict(checkpoint["model_state_dict"])
  print(
      f"[OK] Loaded best 3D ResNet-18 weights (Validation AUC:"
      f" {checkpoint.get('val_auc', 0.0):.4f})"
  )

  # Target layer: last residual block of layer4
  target_conv = model.layer4[-1]
  gradcam = GradCAM3D(model, target_conv)

  # Load test split
  test_csv_path = os.path.join(PROCESSED_DIR, "test.csv")
  df_test = pd.read_csv(test_csv_path)

  # Run inference across test set to identify representative clinical quadrants
  results = []
  for idx, row in df_test.iterrows():
    patch_path = os.path.join(PATCHES_DIR, row["patch_filename"])
    cube = np.load(patch_path)  # (32, 32, 32)
    tensor = (
        torch.from_numpy(cube).unsqueeze(0).unsqueeze(0).to(DEVICE)
    )  # (1, 1, 32, 32, 32)

    with torch.no_grad():
      out = model(tensor)
      prob = torch.sigmoid(out).item()

    pred = 1 if prob >= 0.5 else 0
    gt = int(row["label"])

    results.append({
        "index": idx,
        "patch_filename": row["patch_filename"],
        "series_instance_uid": row["series_instance_uid"],
        "gt": gt,
        "pred": pred,
        "prob": prob,
        "mean_mal": row["mean_malignancy"],
        "diameter": row.get("meta_mean_diameter", 0.0),
    })

  df_res = pd.DataFrame(results)

  # Select 4 representative cases for the evaluation figure:
  # Case 1: True Positive  (Ground Truth = 1, Predicted = 1) -> Confirmed Malignant
  # Case 2: True Negative  (Ground Truth = 0, Predicted = 0) -> Confirmed Benign
  # Case 3: High-Confidence Malignant Nodule
  # Case 4: Borderline / Challenging Nodule
  tp_candidates = df_res[(df_res["gt"] == 1) & (df_res["pred"] == 1)].sort_values(
      "prob", ascending=False
  )
  tn_candidates = df_res[(df_res["gt"] == 0) & (df_res["pred"] == 0)].sort_values(
      "prob", ascending=True
  )

  selected_cases = []
  if not tp_candidates.empty:
    selected_cases.append(
        (tp_candidates.iloc[0], "Case 1: True Positive (Malignant Nodule)")
    )
  if not tn_candidates.empty:
    selected_cases.append(
        (tn_candidates.iloc[0], "Case 2: True Negative (Benign Nodule)")
    )
  if len(tp_candidates) > 1:
    selected_cases.append(
        (tp_candidates.iloc[1], "Case 3: High-Confidence Suspicious Lesion")
    )
  elif not df_res[df_res["gt"] == 1].empty:
    selected_cases.append(
        (df_res[df_res["gt"] == 1].iloc[0], "Case 3: Malignant Sample")
    )

  # Fourth case: False Positive / False Negative if available, otherwise challenging case
  fp_candidates = df_res[(df_res["gt"] == 0) & (df_res["pred"] == 1)]
  fn_candidates = df_res[(df_res["gt"] == 1) & (df_res["pred"] == 0)]
  if not fp_candidates.empty:
    selected_cases.append(
        (fp_candidates.iloc[0], "Case 4: False Positive (Benign Margin Flaw)")
    )
  elif not fn_candidates.empty:
    selected_cases.append(
        (fn_candidates.iloc[0], "Case 4: False Negative (Indeterminate Margin)")
    )
  else:
    # Fallback to another Benign case
    selected_cases.append(
        (tn_candidates.iloc[-1], "Case 4: True Negative (Small Benign Lesion)")
    )

  # ==============================================================================
  # 5. GENERATE GRAD-CAM VISUALIZATION GRID
  # ==============================================================================
  print(f"[Grad-CAM] Generating 4-case XAI panel...")
  fig, axes = plt.subplots(len(selected_cases), 3, figsize=(11, 3.2 * len(selected_cases)))

  # Handle single-row case gracefully
  if len(selected_cases) == 1:
    axes = np.expand_dims(axes, 0)

  col_titles = [
      "Input CT (Central Axial Slice)",
      "3D Grad-CAM Activation Heatmap",
      "Explainability Overlay",
  ]
  for col_idx, title in enumerate(col_titles):
    axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=12)

  xai_records = []

  for row_idx, (case, case_desc) in enumerate(selected_cases):
    patch_path = os.path.join(PATCHES_DIR, case["patch_filename"])
    cube = np.load(patch_path)  # (32, 32, 32)
    tensor = (
        torch.from_numpy(cube).unsqueeze(0).unsqueeze(0).to(DEVICE)
    )  # (1, 1, 32, 32, 32)

    cam_volume, model_prob = gradcam.generate_heatmap(tensor)

    # Extract central axial plane (Slice index 16 along Z-axis)
    slice_idx = 16
    ct_slice = cube[slice_idx, :, :]
    cam_slice = cam_volume[slice_idx, :, :]

    # Plot Column 1: Normalized CT Slice
    ax_ct = axes[row_idx, 0]
    ax_ct.imshow(ct_slice, cmap="gray", vmin=0.0, vmax=1.0)
    gt_str = "Malignant" if case["gt"] == 1 else "Benign"
    ax_ct.set_ylabel(
        f"{case_desc}\nGT: {gt_str} (Score: {case['mean_mal']:.1f})",
        fontsize=9,
        fontweight="semibold",
    )
    ax_ct.set_xticks([])
    ax_ct.set_yticks([])

    # Plot Column 2: Pure Grad-CAM Heatmap
    ax_cam = axes[row_idx, 1]
    im_cam = ax_cam.imshow(cam_slice, cmap="jet", vmin=0.0, vmax=1.0)
    ax_cam.set_xticks([])
    ax_cam.set_yticks([])

    # Plot Column 3: Alpha Blended Overlay
    ax_over = axes[row_idx, 2]
    ax_over.imshow(ct_slice, cmap="gray", vmin=0.0, vmax=1.0)
    ax_over.imshow(cam_slice, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    pred_str = "Malignant" if case["pred"] == 1 else "Benign"
    ax_over.set_xlabel(
        f"Model Prediction: {pred_str} (p = {model_prob:.3f})",
        fontsize=10,
        color="darkred" if case["pred"] == 1 else "darkgreen",
        fontweight="bold",
    )
    ax_over.set_xticks([])
    ax_over.set_yticks([])

    xai_records.append({
        "case_description": case_desc,
        "patch_filename": case["patch_filename"],
        "ground_truth_label": int(case["gt"]),
        "radiologist_mean_malignancy": float(case["mean_mal"]),
        "predicted_probability": round(float(model_prob), 4),
        "predicted_label": int(case["pred"]),
        "central_slice_index": slice_idx,
    })

  plt.tight_layout()
  output_fig_path = os.path.join(RESULTS_DIR, "gradcam_xai_overlay.png")
  plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
  plt.close()

  # Save traceability audit JSON
  audit_json_path = os.path.join(RESULTS_DIR, "xai_cases_audit.json")
  with open(audit_json_path, "w") as f:
    json.dump(xai_records, f, indent=4)

  print("\n" + "=" * 70)
  print("STAGE 3 COMPLETED SUCCESSFULLY")
  print("=" * 70)
  print(f"[Artifact Saved] XAI Visualization Panel : {output_fig_path}")
  print(f"[Artifact Saved] Case Traceability Audit : {audit_json_path}")
  print("=" * 70)


if __name__ == "__main__":
<<<<<<< HEAD
  run_stage3()
=======
  run_stage3()
>>>>>>> c44f2c3910463f7bfca85a3f61b9428fa41af64c
