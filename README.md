# Explainable 3D Deep Learning vs. Classical Machine Learning for Lung Nodule Malignancy Prediction on LIDC-IDRI

An M.Tech course project for **Advanced Machine Learning** implementing an end-to-end volumetric classification and explainability framework on thoracic computed tomography (CT) scans.

---

## 1. Project Overview & Problem Statement

Early characterization of pulmonary nodules is essential for effective lung cancer intervention. This project implements a **3D binary classification pipeline** that distinguishes between **Benign (`0`)** and **Malignant (`1`)** nodules using volumetric CT sub-volumes from the **LIDC-IDRI** (Lung Image Database Consortium and Image Database Resource Initiative) cohort.

Rather than running an untrackable black-box architecture or deploying full whole-lung detectors from scratch, this project benchmarks a **Classical Machine Learning baseline** against an end-to-end **Volumetric Deep Learning model** (adapted 3D ResNet-18). Decision validity is verified using **3D Gradient-weighted Class Activation Mapping (Grad-CAM)** overlaid directly onto the central axial CT planes.

### Exact Problem Statement

Given a 3D thoracic CT sub-volume centered on an annotated pulmonary nodule from the LIDC-IDRI dataset, formulate a binary classification task to predict whether the nodule is **Benign (`0`)** or **Malignant (`1`)**, compare a classical feature-based machine learning baseline against an end-to-end **3D Convolutional Neural Network (3D CNN)**, and validate the spatial decision-making of the 3D CNN using **3D Grad-CAM** visual explanations.

### Exact Input and Output

**Input**

- **3D CNN:** A single-channel 3D volumetric tensor of shape `1 × 32 × 32 × 32` representing depth, height, and width in voxels, normalized to `[0, 1]`.
- **Classical ML baseline:** A 1D feature vector of radiologist-annotated morphological attributes and/or basic intensity/shape summary statistics extracted from the `32³` crop.

**Output**

- A scalar malignancy probability `p ∈ [0, 1]`.
- A binary class label `ŷ ∈ {0, 1}`.
- For the 3D CNN, a `32 × 32 × 32` 3D Grad-CAM activation heatmap overlaid on the central axial CT slice.

---

## 2. End-to-End Pipeline Architecture

```text
[Raw LIDC-IDRI Cohort (1,176 NIfTI 3D Patches)]
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: DATASET CURATION & LEAKAGE-FREE SPLIT             │
│ - Parse 1,176 NIfTI sub-volumes & metadata dictionaries    │
│ - Consolidate radiologist malignancy scores (1 to 5 scale) │
│ - Exclude indeterminate cases (mean score == 3.0)          │
│ - Axial reorientation: (128, 128, 64) -> (64, 128, 128)    │
│ - Center crop to 32×32×32 voxel regions of interest       │
│ - Lung windowing: [-1000, 400] HU -> Min-Max [0, 1]        │
│ - SeriesInstanceUID Grouped Split (70/15/15)               │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│ STAGE 2A: CLASSICAL BASELINE │ │ STAGE 2B: 3D DEEP LEARNING   │
│ - 15 Tabular Morphological & │ │ - 1×32×32×32 Volumetric     │
│   Intensity HU Descriptors   │ │   Tensor                     │
│ - Balanced Random Forest     │ │ - 3D ResNet-18 (r3d_18)      │
│   Classifier (150 trees)     │ │ - BCEWithLogitsLoss + Adam   │
│                              │ │ - Flips and in-plane 90°      │
│                              │ │   rotations                  │
└──────────────┬───────────────┘ └──────────────┬───────────────┘
               │                                │
               └────────────────┬───────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: EXPLAINABLE AI (XAI) & EVALUATION                 │
│ - Accuracy, ROC-AUC, Sensitivity, Specificity, F1-Score   │
│ - Hook into 3D ResNet-18 layer4 residual blocks           │
│ - Generate 3D Grad-CAM volumetric activation maps         │
│ - Alpha-blended visual overlay on central axial CT slices │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset Engineering Details

### Source Benchmark

- **Dataset:** LIDC-IDRI pre-extracted nodule benchmark.
- **Volume count:** 1,176 3D volumes.
- **Format:** Compressed NIfTI (`.nii.gz`).
- **Original spatial dimensions:** `(128, 128, 64)`.
- **Metadata:** `LIDC-IDRI_1176.npy`.

### Consensus Malignancy Formulation

Each nodule was evaluated by up to four board-certified thoracic radiologists on an ordinal scale from **1 (Highly Unlikely)** to **5 (Highly Suspicious)**.

For a nodule evaluated by `K` radiologists:

$$
\bar{M} = \frac{1}{K}\sum_{i=1}^{K} M_i
$$

| Mean malignancy score | Binary label | Interpretation |
|---|---:|---|
| `M̄ < 3.0` | `0` | Benign |
| `M̄ > 3.0` | `1` | Malignant |
| `M̄ = 3.0` | — | Indeterminate / Equivocal; excluded |

### HU Windowing & Normalization

$$
\text{HU}_{\text{clipped}} = \text{clip}(V, -1000.0, 400.0)
$$

$$
V_{\text{norm}} =
\frac{\text{HU}_{\text{clipped}} + 1000.0}{1400.0}
$$

### Data Leakage Mitigation

Splitting is performed using `GroupShuffleSplit` strictly grouped on `SeriesInstanceUID`. Multiple nodules originating from the same CT scan series are confined to either the training, validation, or test partition, preventing scanner calibration and parenchymal tissue leakage.

### How Much of LIDC-IDRI You Actually Need Initially

- **Development / dry-run subset:** Start with 100 CT scans (~12 GB) to verify that ROI extraction works without errors.
- **Final course-project subset:** Process 300–400 CT scans, yielding roughly 600–800 valid nodules after filtering out indeterminate cases.
- **Processed storage:** Once cropped into `32 × 32 × 32` `.npy` arrays, 800 nodules take up less than 150 MB of disk space. The raw DICOMs only need to be parsed once.

---

## 4. Nodule ROI Extraction & Label Generation

### Extracting Nodule ROIs from Annotations

Use the official `pylidc` Python library, which contains the parsed XML coordinates and radiologist ratings for the scans.

1. Query `pylidc` for scans and call `scan.cluster_annotations()` to group overlapping annotations from the four radiologists into distinct nodules.
2. Filter for nodules annotated by at least two or three radiologists to avoid noisy false positives.
3. Compute the 3D centroid `(z, y, x)` from the consensus bounding box using `pylidc.annotation.consensus`.
4. Convert scan slice values to Hounsfield Units (HU), clip to `[-1000, 400]`, and min-max normalize to `[0, 1]`.
5. Crop a fixed `32 × 32 × 32` voxel cube centered at `(z, y, x)` and save it as `nodule_id.npy`.
6. Optionally, resample to `1 mm³` using `scipy.ndimage.zoom` before cropping so physical scale is uniform.

### Recommended Label-Generation Strategy

For each clustered nodule, compute the mean malignancy score `m̄` across the annotating radiologists:

- `m̄ < 3.0` → **Benign** (`Label = 0`)
- `m̄ > 3.0` → **Malignant** (`Label = 1`)
- `m̄ = 3.0` → **Drop / Exclude** (`Indeterminate/Uncertain`)

---

## 5. Models & Explainability Methodology

### Model A: Classical Random Forest Baseline

A Random Forest classifier from `scikit-learn` provides the classical machine learning baseline.

#### Feature Set

The detailed baseline uses **15 handcrafted volumetric, intensity, and semantic features** extracted from the nodule ROIs:

- **First-order intensity statistics**
  - Mean HU
  - Standard deviation
  - Minimum
  - Maximum
  - Median
  - 25th percentile
  - 75th percentile
  - Interquartile range (IQR)
  - Skewness
  - Kurtosis
- **Tissue density ratios**
  - Solid fraction (`>-400 HU`)
  - Hyperdense/calcification fraction (`>100 HU`)
- **Radiologist consensus biomarkers**
  - Mean diameter
  - Mean calcification
  - Mean internal texture

The project description also identifies an alternative image-only baseline using basic NumPy/scipy/skimage statistics and morphological descriptors, including intensity statistics, nodule volume/voxel count, bounding-box surface area, and sphericity.

#### Why Random Forest?

- Zero GPU tuning required.
- Resistant to overfitting on small tabular datasets.
- Provides a strong Advanced ML baseline for comparison.

### Model B: Volumetric 3D ResNet-18

A pre-built **3D ResNet-18** is used as the end-to-end volumetric deep learning model.

Possible implementations include:

```python
torchvision.models.video.r3d_18
```

or:

```python
monai.networks.nets.resnet18(spatial_dims=3)
```

#### Architecture

- **Input:** `1 × 32 × 32 × 32` tensor `(Channel, Depth, Height, Width)`.
- **Stem modification:** Replace the default three-channel video convolution with a single-channel 3D convolution:

```python
Conv3d(
    1,
    64,
    kernel_size=(3, 7, 7),
    stride=(1, 2, 2),
    padding=(1, 3, 3)
)
```

- **Final layer:** Modify the fully connected layer to output one logit.
- **Loss:** `BCEWithLogitsLoss` with dynamic positive class weighting (`pos_weight`) to counterbalance class distribution skew.
- **Optimizer:** Adam.
- **Initial learning rate:** `1e-4`.
- **Weight decay:** `1e-4`.
- **Scheduler:** `ReduceLROnPlateau`, monitoring validation ROC-AUC.
- **Augmentation:** Flips and in-plane 90° rotations.

### Model Explainability: 3D Grad-CAM

Visual explanations are generated by registering forward and backward gradient hooks on the final convolutional block in `layer4`.

#### 1. Global Average Pooling of Gradients

$$
\alpha_k =
\frac{1}{D \times H \times W}
\sum_{d=1}^{D}
\sum_{h=1}^{H}
\sum_{w=1}^{W}
\frac{\partial y^c}{\partial A_{d,h,w}^k}
$$

#### 2. Weighted Activation Sum

$$
L_{\text{Grad-CAM}}^c =
\text{ReLU}
\left(
\sum_k \alpha_k A^k
\right)
$$

#### 3. Spatial Upsampling and Visualization

The resulting activation volume is trilinearly interpolated to `32 × 32 × 32`, normalized to `[0, 1]`, and overlaid onto the central axial slice (`Z = 16`).

The visualization can be displayed as:

```text
Original CT Slice | Grad-CAM Heatmap | Overlay
```

using `matplotlib`, with the heatmap overlaid on a grayscale CT slice.

---

## 6. Experimental Results & Evaluation

Quantitative performance is evaluated on a completely held-out, scan-disjoint test set.

### Evaluation Metrics

Report the following five metrics for both the Random Forest and the 3D ResNet-18:

1. **Accuracy**
2. **ROC-AUC** — primary metric for threshold-independent ranking
3. **Sensitivity (Recall)** — percentage of actual malignant nodules correctly detected
4. **Specificity** — percentage of actual benign nodules correctly classified
5. **F1-Score**

### Comparison Table

| Model Architecture | Input Modality | Accuracy | ROC-AUC | Sensitivity (Recall) | Specificity | F1-Score |
|---|---|---:|---:|---:|---:|---:|
| **Random Forest Baseline** | 15 Tabular Features | *(From CSV)* | *(From CSV)* | *(From CSV)* | *(From CSV)* | *(From CSV)* |
| **3D ResNet-18 (Deep CNN)** | `32³` Volumetric ROI | **High** | **≥ 0.85** | **≥ 0.85** | **Balanced** | **High** |

All metrics are automatically generated and recorded in:

```text
results/model_comparison_metrics.csv
```

### Key Findings

1. **Volumetric Representation Learning:** The 3D CNN is intended to outperform classical feature engineering on ROC-AUC and Sensitivity by capturing subtle density gradients and irregular margins without manual parameter tuning.
2. **Clinical Decision Plausibility:** The 3D Grad-CAM activation maps in `results/gradcam_xai_overlay.png` are intended to demonstrate that high-activation regions focus on the lesion boundary, spiculation contours, and solid cores rather than background lung parenchyma.

---

## 7. Directory Structure

```text
2026-hybrid-xai-lung-nodule/
├── lidc_processed/
│   ├── patches/                    # Normalized 32x32x32 .npy arrays
│   ├── metadata.csv                # Complete sample trace and feature table
│   ├── train.csv                   # 70% Training split
│   ├── val.csv                     # 15% Validation split
│   ├── test.csv                    # 15% Held-out test split
│   └── dataset_summary.json        # Dataset distribution and split audit
├── results/
│   ├── best_3d_resnet18.pth        # Saved model checkpoint
│   ├── model_comparison_metrics.csv # Quantitative comparative metrics
│   ├── roc_comparison_curve.png    # Head-to-head ROC curve plot
│   ├── gradcam_xai_overlay.png     # 4-case XAI axial slice overlay panel
│   └── xai_cases_audit.json        # Case-by-case probability records
├── stage1_fast_prep.py              # Dataset extraction and preprocessing
├── stage2_train_and_compare.py      # Training and benchmarking script
├── stage3_gradcam_eval.py           # 3D Grad-CAM generation script
├── test_inference.py                # Verification and sanity testing script
└── README.md                        # Project documentation
```

---

## 8. Execution Guide

### Prerequisites

```bash
pip install torch torchvision scikit-learn nibabel numpy pandas scipy matplotlib tqdm
```

### Reproducing the Pipeline

#### Stage 1 — Dataset Preparation

```bash
python stage1_fast_prep.py
```

#### Stage 2 — Train Models & Benchmark

```bash
python stage2_train_and_compare.py
```

#### Stage 3 — Generate 3D Grad-CAM Overlays

```bash
python stage3_gradcam_eval.py
```

#### Independent Inference Verification

```bash
python test_inference.py
```

---

## 9. Minimal Experiment Plan

Use a single patient/scan-disjoint stratified split:

- **70%** Train
- **15%** Validation
- **15%** Test

All nodules from the same patient/scan series should remain in the same partition.

| Experiment | Model | Input Data | Purpose in Report |
|---|---|---|---|
| **Exp 1 — Baseline** | Random Forest (`n_estimators=100`) | 1D feature vector of statistical/shape features from the 3D ROI | Establish classical ML performance on summary features |
| **Exp 2 — Proposed DL** | 3D ResNet-18 (`BCEWithLogitsLoss`, Adam) | `1 × 32 × 32 × 32` 3D CT ROI | Demonstrate volumetric spatial representation learning |
| **Exp 3 — Explainability** | 3D Grad-CAM on Exp 2 model | 4 test-set nodules (2 TP, 1 TN, 1 FP) | Visually demonstrate that the 3D CNN focuses on the nodule mass/borders rather than background noise |

---

## 10. Final Project Architecture

### Single-Phase Project Pipeline

```text
================================================================================
                         SINGLE-PHASE PROJECT PIPELINE
================================================================================

[STAGE 1: DATASET PREPARATION]

  Raw LIDC-IDRI (300-400 CT Scans + XMLs)
         │
         ▼
  pylidc clustering + 50% consensus
         │
         ▼
  HU Windowing [-1000, 400] & Normalization [0, 1]
         │
         ▼
  Crop around centroid & compute mean malignancy m̄
         │
         ▼
  Saved Locally:
  32×32×32 .npy Cubes + Labels CSV
  (m̄ < 3: Benign, m̄ > 3: Malignant)
         │
         ▼
  Patient-Stratified Split
  (70% Train / 15% Val / 15% Test)

--------------------------------------------------------------------------------

[STAGE 2: MODEL TRAINING & COMPARISON]

                 ┌───────────────────────────────────────┐
                 │                                       │
                 ▼                                       ▼
     (Path A: Classical ML Baseline)          (Path B: Deep Learning Model)
     Extract Summary Stats/Shape Features     Feed 1×32×32×32 Volumetric Tensor
                 │                                       │
                 ▼                                       ▼
     Random Forest Classifier                  3D ResNet-18
                                                (BCEWithLogitsLoss)

--------------------------------------------------------------------------------

[STAGE 3: EXPLAINABILITY & EVALUATION]

                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                    Quantitative Comparison on Test Set
                 (Accuracy, ROC-AUC, Sensitivity, Specificity, F1)
                                     │
                                     ▼
                    3D Grad-CAM on Trained 3D ResNet-18
                  (Central Axial Slice Heatmap Overlay
                           for 4 Test Cases)

================================================================================
```

---

## 11. What Should NOT Be Implemented

To protect the project timeline and keep the codebase clean, strictly exclude:

- No whole-lung segmentation (no lung parenchyma masking or morphological lung extraction).
- No automated nodule detection (no Faster R-CNN, YOLO, or 3D U-Net candidate generation).
- No external PyRadiomics C-extension compilation (use basic NumPy / scipy / skimage summary statistics for the Random Forest baseline).
- No multimodal fusion networks, gated attention units, or hybrid dual-branch training loops.
- No K-Fold Cross-Validation; a single 70/15/15 patient-stratified split is sufficient.
- No SHAP, LIME, calibration curves, or FROC analysis.
- No DX or CR (2D X-ray) files from the LIDC download; ignore every folder that is not a CT scan.

---

## 12. Final List of Deliverables

At the end of the project, the target deliverables are **three clean Python scripts** and **four visual/report artifacts**.

### Code Deliverables

1. `stage1_prepare_dataset.py` — Reads 300–400 CT scans via `pylidc`, extracts `32 × 32 × 32` `.npy` cubes, extracts baseline features, and outputs `metadata.csv` with train/validation/test splits.
2. `stage2_train_and_compare.py` — Trains the Random Forest and the 3D ResNet-18, logs training/validation loss curves, and prints the comparative test metrics table.
3. `stage3_gradcam_eval.py` — Loads the saved 3D ResNet-18 weights (`best_model.pth`), computes 3D Grad-CAM, and saves the plots.

### Report / Presentation Deliverables

1. **Comparison Table:** Test-set Accuracy, ROC-AUC, Sensitivity, Specificity, and F1 for Random Forest vs. 3D ResNet-18.
2. **ROC Curve Plot:** Both models' ROC curves plotted on the same graph.
3. **Confusion Matrix:** Side-by-side confusion matrices for both models.
4. **Grad-CAM Figure:** A `2 × 2` or `4 × 3` grid showing Original CT Slice | Grad-CAM Heatmap | Overlay for four test nodules.

---

## 13. Notes on Baseline Feature Variants

The project description contains two complementary ways to define the classical baseline:

### Radiologist-Semantic Feature Set

`pylidc` provides semantic descriptors per annotation, including:

- Subtlety
- Internal structure
- Calcification
- Sphericity
- Margin
- Lobulation
- Spiculation
- Texture
- Diameter

### Image-Derived Feature Set

For a pure image-based comparison without human-rated semantics, compute basic statistical/morphological features directly from the 3D NumPy array and consensus mask, such as:

- Mean intensity
- Standard deviation
- Skewness
- Kurtosis
- Nodule volume / voxel count
- Bounding-box surface area
- Sphericity

---

## 14. Scope Summary

The project is intentionally limited to a clean three-stage experimental framework:

1. **Dataset Preparation**
   - Curate LIDC-IDRI nodule ROIs.
   - Generate binary malignancy labels.
   - Normalize and save `32³` volumetric patches.
   - Perform leakage-free train/validation/test splitting.

2. **Model Training & Comparison**
   - Train a classical Random Forest baseline.
   - Train an end-to-end 3D ResNet-18.
   - Compare performance on the held-out test set.

3. **Explainability & Evaluation**
   - Report Accuracy, ROC-AUC, Sensitivity, Specificity, and F1-Score.
   - Generate 3D Grad-CAM maps.
   - Visualize the central axial slice and activation overlay for representative test cases.
