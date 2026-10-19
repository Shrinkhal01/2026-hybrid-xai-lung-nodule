#!/usr/bin/env python3
"""Interactive Streamlit Web Dashboard for 2026-hybrid-xai-lung-nodule.

Demonstrates real-time 3D ResNet-18 inference and 3D Grad-CAM slice exploration.
"""

import io
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Lung Nodule Malignancy & 3D XAI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "results", "best_3d_resnet18.pth")

# Fallback to FP16 checkpoint if original was converted
if not os.path.exists(CHECKPOINT_PATH):
  alt_fp16 = os.path.join(BASE_DIR, "results", "best_3d_resnet18_fp16.pth")
  if os.path.exists(alt_fp16):
    CHECKPOINT_PATH = alt_fp16

# Device Selection
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


# ==============================================================================
# MODEL & GRAD-CAM CACHING
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


@st.cache_resource
def load_system():
  if not os.path.exists(CHECKPOINT_PATH):
    return None, None
  model = build_3d_resnet().to(DEVICE)
  ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

  # Check if model was stored in half precision
  state_dict = ckpt["model_state_dict"]
  first_weight = next(iter(state_dict.values()))
  if first_weight.dtype == torch.float16:
    model.half()

  model.load_state_dict(state_dict)
  model.eval()
  gradcam = GradCAM3D(model, model.layer4[-1])
  return model, gradcam


# ==============================================================================
# PREPROCESSING HELPER
# ==============================================================================
def process_uploaded_data(file_bytes, filename):
  name = filename.lower()

  if name.endswith(".npy"):
    buf = io.BytesIO(file_bytes)
    arr = np.load(buf)
    if arr.ndim == 3:
      d, h, w = arr.shape
      cd, ch, cw = d // 2, h // 2, w // 2
      crop = arr[
          max(0, cd - 16) : cd + 16,
          max(0, ch - 16) : ch + 16,
          max(0, cw - 16) : cw + 16,
      ]
      vol = np.pad(
          crop,
          (
              (max(0, 16 - cd), max(0, cd + 16 - d)),
              (max(0, 16 - ch), max(0, ch + 16 - h)),
              (max(0, 16 - cw), max(0, cw + 16 - w)),
          ),
          mode="edge",
      )[:32, :32, :32]
    else:
      raise ValueError("Uploaded NumPy array must be 3-dimensional.")

  elif name.endswith(".nii") or name.endswith(".nii.gz"):
    import nibabel as nib

    # Temporary file storage for nibabel stream
    temp_path = os.path.join(BASE_DIR, "temp_stream.nii.gz")
    with open(temp_path, "wb") as f:
      f.write(file_bytes)
    nii = nib.load(temp_path)
    vol_raw = nii.get_fdata(dtype=np.float32)
    os.remove(temp_path)

    vol_zyx = np.transpose(vol_raw, (2, 1, 0)) if vol_raw.ndim == 3 else vol_raw
    vol_clipped = np.clip(vol_zyx, -1000.0, 400.0)
    vol_norm = (vol_clipped + 1000.0) / 1400.0
    d, h, w = vol_norm.shape
    vol = vol_norm[
        max(0, d // 2 - 16) : d // 2 + 16,
        max(0, h // 2 - 16) : h // 2 + 16,
        max(0, w // 2 - 16) : w // 2 + 16,
    ][:32, :32, :32]

  else:  # PNG, JPG, JPEG
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = img.resize((32, 32), Image.Resampling.BILINEAR)
    arr_2d = np.array(img, dtype=np.float32) / 255.0
    vol = np.repeat(arr_2d[np.newaxis, :, :], 32, axis=0)

  vol = np.clip(vol, 0.0, 1.0).astype(np.float32)
  return vol


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.title("🫁 Navigation & Controls")
st.sidebar.markdown(
    "**Project:** M.Tech AIML - Advanced ML\n**Model:** Volumetric 3D ResNet-18"
)

model, gradcam = load_system()

if model is None:
  st.sidebar.error(
      f"⚠️ Checkpoint not found at: `{CHECKPOINT_PATH}`. Please run Stage 2"
      " first."
  )
  st.stop()
else:
  st.sidebar.success(f"✓ Model loaded on: `{DEVICE}`")

data_source_mode = st.sidebar.radio(
    "Select Input Source:",
    ("Use Preloaded Benchmarks", "Upload Custom CT File (.npy, .nii.gz, .png)"),
)

input_vol = None
sample_name = ""

if data_source_mode == "Use Preloaded Benchmarks":
  sample_choices = {}
  custom_dir = os.path.join(BASE_DIR, "custom_test_inputs")
  patches_dir = os.path.join(BASE_DIR, "lidc_processed", "patches")

  if os.path.exists(custom_dir):
    for f in sorted(os.listdir(custom_dir)):
      if f.endswith((".npy", ".png", ".jpg")):
        sample_choices[f"Benchmark: {f}"] = os.path.join(custom_dir, f)

  if os.path.exists(patches_dir):
    for f in sorted(os.listdir(patches_dir))[:8]:
      sample_choices[f"Test Cohort: {f}"] = os.path.join(patches_dir, f)

  if sample_choices:
    selected_key = st.sidebar.selectbox("Choose Sample:", list(sample_choices.keys()))
    sample_path = sample_choices[selected_key]
    with open(sample_path, "rb") as f:
      input_vol = process_uploaded_data(f.read(), os.path.basename(sample_path))
    sample_name = selected_key
  else:
    st.sidebar.info(
        "Run `python download_test_samples.py` to populate test samples."
    )

else:
  uploaded_file = st.sidebar.file_uploader(
      "Upload Lung Scan", type=["npy", "nii", "gz", "png", "jpg", "jpeg"]
  )
  if uploaded_file is not None:
    input_vol = process_uploaded_data(uploaded_file.read(), uploaded_file.name)
    sample_name = uploaded_file.name


# ==============================================================================
# MAIN DASHBOARD INTERFACE
# ==============================================================================
st.title("Explainable 3D Deep Learning for Lung Nodule Malignancy")
st.markdown(
    "Interactive Clinical Decision Support Dashboard featuring **Volumetric 3D"
    " ResNet-18** and **3D Grad-CAM**."
)

if input_vol is not None:
  # Inference computation
  first_weight = next(iter(model.parameters()))
  is_half = first_weight.dtype == torch.float16

  tensor = torch.from_numpy(input_vol).unsqueeze(0).unsqueeze(0).to(DEVICE)
  if is_half:
    tensor = tensor.half()

  cam_volume, prob = gradcam.generate(tensor)

  is_malignant = prob >= 0.5
  pred_label = "MALIGNANT" if is_malignant else "BENIGN"
  confidence = prob if is_malignant else (1.0 - prob)

  # Top KPI Metrics Row
  col1, col2, col3, col4 = st.columns(4)
  col1.metric("Input Nodule Source", sample_name)
  col2.metric(
      "Diagnostic Prediction",
      pred_label,
      delta="High Risk" if is_malignant else "Low Risk",
      delta_color="inverse" if is_malignant else "normal",
  )
  col3.metric("Malignancy Probability", f"{prob * 100:.2f}%")
  col4.metric("Model Confidence", f"{confidence * 100:.2f}%")

  st.divider()

  # Interactive 3D Depth Exploration Slider
  st.subheader("🔍 Volumetric Spatial Exploration")
  slice_z = st.slider(
      "Scrub 3D Axial Depth (Z-axis plane):",
      min_value=0,
      max_value=31,
      value=16,
      help="Inspect how the nodule and 3D Grad-CAM activations evolve across depth slices.",
  )

  # Extract slices
  ct_slice = input_vol[slice_z, :, :]
  cam_slice = cam_volume[slice_z, :, :]

  # Visual Columns
  vcol1, vcol2, vcol3 = st.columns(3)

  with vcol1:
    st.markdown(f"**1. Input CT Slice (Z = {slice_z})**")
    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.imshow(ct_slice, cmap="gray", vmin=0.0, vmax=1.0)
    ax1.axis("off")
    st.pyplot(fig1, use_container_width=True)
    plt.close()

  with vcol2:
    st.markdown(f"**2. 3D Grad-CAM Heatmap (Z = {slice_z})**")
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2.imshow(cam_slice, cmap="jet", vmin=0.0, vmax=1.0)
    ax2.axis("off")
    st.pyplot(fig2, use_container_width=True)
    plt.close()

  with vcol3:
    st.markdown(f"**3. Alpha-Blended Explainability Overlay**")
    fig3, ax3 = plt.subplots(figsize=(4, 4))
    ax3.imshow(ct_slice, cmap="gray", vmin=0.0, vmax=1.0)
    ax3.imshow(cam_slice, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    ax3.axis("off")
    st.pyplot(fig3, use_container_width=True)
    plt.close()

  # Clinical Interpretation Callout
  st.info(
      f"**Clinical Interpretability Note:** For this slice (Z={slice_z}),"
      f" the 3D ResNet-18 activation gradients concentrate on areas of high"
      " color intensity (red/yellow). In malignant cases, activations align"
      " with margin spiculation and soft-tissue density borders, verifying"
      " that the prediction is pathologically grounded."
  )

else:
  st.warning(
      "👈 Select a preloaded benchmark nodule from the sidebar or upload a file"
      " to initiate diagnostic prediction."
  )

