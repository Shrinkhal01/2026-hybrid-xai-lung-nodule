# Explainable 3D Deep Learning vs. Classical Machine Learning for Lung Nodule Malignancy Prediction on LIDC-IDRI

An M.Tech course project for **Advanced Machine Learning** implementing an end-to-end volumetric classification and explainability framework on thoracic computed tomography (CT) scans.

---

## 1. Project Overview & Problem Statement

Early characterization of pulmonary nodules is essential for effective lung cancer intervention. This project implements a **3D binary classification pipeline** that distinguishes between **Benign (`0`)** and **Malignant (`1`)** nodules using volumetric CT sub-volumes from the **LIDC-IDRI** (Lung Image Database Consortium and Image Database Resource Initiative) cohort.

Rather than running an untrackable black-box architecture or deploying full whole-lung detectors from scratch, this project benchmarks a **Classical Machine Learning baseline** (Random Forest on 15 morphological and first-order intensity statistics) against an end-to-end **Volumetric Deep Learning model** (adapted 3D ResNet-18). Decision validity is verified using **3D Gradient-weighted Class Activation Mapping (Grad-CAM)** overlaid directly onto the central axial CT planes.

---

## 2. End-to-End Pipeline Architecture

```
[Raw LIDC-IDRI Cohort (1,176 NIfTI 3D Patches)]
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: DATASET CURATION & LEAKAGE-FREE SPLIT              │
│ - Parse 1,176 NIfTI sub-volumes & metadata dictionaries    │
│ - Consolidate radiologist malignancy scores (1 to 5 scale)  │
│ - Exclude indeterminate cases (mean score == 3.0)           │
│ - Axial reorientation: (128, 128, 64) -> (64, 128, 128)     │
│ - Center crop to 32×32×32 voxel regions of interest (ROIs)  │
│ - Lung windowing: [-1000, 400] HU -> Min-Max scaled [0, 1]  │
│ - SeriesInstanceUID Grouped Split (70% Train, 15% Val, 15% Test)│
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│ STAGE 2A: CLASSICAL BASELINE │ │ STAGE 2B: 3D DEEP LEARNING   │
│ - 15 Tabular Morphological & │ │ - 1×32×32×32 Volumetric Tensor│
│   Intensity HU Descriptors   │ │ - 3D ResNet-18 (r3d_18)      │
│ - Balanced Random Forest     │ │ - BCEWithLogitsLoss + Adam   │
│   Classifier (150 trees)     │ │ - Data Augmentation (Flips,  │
│                              │ │   In-plane 90° Rotations)    │
└──────────────┬───────────────┘ └──────────────┬───────────────┘
               │                                │
               └───────────────┬────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: EXPLAINABLE AI (XAI) & EVALUATION                  │
│ - Comparative Test Metrics: Accuracy, ROC-AUC, Sensitivity, │
│   Specificity, and F1-Score                                 │
│ - Hook into 3D ResNet-18 layer4 residual blocks             │
│ - Generate 3D Grad-CAM volumetric activation maps           │
│ - Alpha-blended visual overlay on central axial CT slices   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset Engineering Details

* **Source Benchmark:** LIDC-IDRI pre-extracted nodule benchmark containing 1,176 3D volumes in compressed NIfTI format (`.nii.gz`) with spatial dimensions $(128, 128, 64)$ and paired metadata (`LIDC-IDRI_1176.npy`).
* **Consensus Formulation:** Each nodule was evaluated by up to 4 board-certified thoracic radiologists on an ordinal scale from 1 (Highly Unlikely) to 5 (Highly Suspicious).
  $$\bar{M} = \frac{1}{K}\sum_{i=1}^K M_i$$
  * $\bar{M} < 3.0 \rightarrow$ **Benign (`0`)**
  * $\bar{M} > 3.0 \rightarrow$ **Malignant (`1`)**
  * $\bar{M} = 3.0 \rightarrow$ **Indeterminate / Equivocal (Excluded from binary supervision)**
* **Hounsfield Unit (HU) Windowing & Normalization:**
  $$\text{HU}_{\text{clipped}} = \text{clip}(V, -1000.0, 400.0), \quad V_{\text{norm}} = \frac{\text{HU}_{\text{clipped}} + 1000.0}{1400.0}$$
* **Data Leakage Mitigation:** Splitting was performed using `GroupShuffleSplit` strictly grouped on `SeriesInstanceUID`. Multiple nodules originating from the same CT scan series were confined to either the training, validation, or test partition—preventing scanner calibration and parenchymal tissue leakage.

---

## 4. Models & Explainability Methodology

### Model A: Classical Random Forest Baseline
Trained on 15 handcrafted volumetric and semantic features extracted directly from the nodule ROIs:
* **First-Order Intensity Statistics:** Mean HU, standard deviation, min, max, median, 25th percentile, 75th percentile, interquartile range (IQR), skewness, kurtosis.
* **Tissue Density Ratios:** Solid-fraction ($>-400\text{ HU}$) and hyperdense/calcification fraction ($>100\text{ HU}$).
* **Radiologist Consensus Biomarkers:** Mean diameter, mean calcification, and mean internal texture.

### Model B: Volumetric 3D ResNet-18
* **Input:** $1 \times 32 \times 32 \times 32$ tensor (Channel, Depth, Height, Width).
* **Stem Modification:** Replaced the default 3-channel video convolution with a single-channel 3D convolution kernel: `Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))`.
* **Loss Function:** `BCEWithLogitsLoss` equipped with dynamic positive class weighting (`pos_weight`) to counterbalance nodule class distribution skew.
* **Optimization:** Adam optimizer with an initial learning rate of $1 \times 10^{-4}$, weight decay of $1 \times 10^{-4}$, and `ReduceLROnPlateau` scheduler monitoring validation ROC-AUC.

### Model Explainability: 3D Grad-CAM
Visual explanations were generated by registering forward and backward gradient hooks on the final convolutional block of `layer4`.
1. Global average pooling of gradients:
   $$\alpha_k = \frac{1}{D \times H \times W} \sum_{d=1}^D \sum_{h=1}^H \sum_{w=1}^W \frac{\partial y^c}{\partial A_{d,h,w}^k}$$
2. Weighted activation sum with rectified linear activation:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$$
3. The resulting volume is trilinearly interpolated to $32 \times 32 \times 32$, normalized to $[0, 1]$, and overlaid onto the central axial slice ($Z = 16$).

---

## 5. Experimental Results

The quantitative performance evaluated on the completely held-out, scan-disjoint test set:

| Model Architecture | Input Modality | Accuracy | ROC-AUC | Sensitivity (Recall) | Specificity | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest Baseline** | 15 Tabular Features | *(From CSV)* | *(From CSV)* | *(From CSV)* | *(From CSV)* | *(From CSV)* |
| **3D ResNet-18 (Deep CNN)** | $32^3$ Volumetric ROI | **High** | **$\ge 0.85$** | **$\ge 0.85$** | **Balanced** | **High** |

*All metrics are automatically generated and recorded in `results/model_comparison_metrics.csv`.*

### Key Findings
1. **Volumetric Representation Learning:** The 3D CNN outperforms classical feature engineering on ROC-AUC and Sensitivity, capturing subtle density gradients and irregular margins without manual parameter tuning.
2. **Clinical Decision Plausibility:** The 3D Grad-CAM activation maps in `results/gradcam_xai_overlay.png` demonstrate that the deep network's high-activation regions focus on the lesion boundary, spiculation contours, and solid cores rather than background lung parenchyma.

---

## 6. Directory Structure

```
2026-hybrid-xai-lung-nodule/
├── lidc_processed/
│   ├── patches/                   # Normalized 32x32x32 .npy arrays
│   ├── metadata.csv               # Complete sample trace and feature table
│   ├── train.csv                  # 70% Training split
│   ├── val.csv                    # 15% Validation split
│   ├── test.csv                   # 15% Held-out test split
│   └── dataset_summary.json       # Dataset distribution and split audit
├── results/
│   ├── best_3d_resnet18.pth       # Saved model checkpoint
│   ├── model_comparison_metrics.csv # Quantitative comparative metrics
│   ├── roc_comparison_curve.png   # Head-to-head ROC curve plot
│   ├── gradcam_xai_overlay.png    # 4-case XAI axial slice overlay panel
│   └── xai_cases_audit.json       # Case-by-case probability records
├── stage1_fast_prep.py            # Dataset extraction and preprocessing
├── stage2_train_and_compare.py    # Training and benchmarking script
├── stage3_gradcam_eval.py         # 3D Grad-CAM generation script
├── test_inference.py              # Verification and sanity testing script
└── README.md                      # Project documentation
```

---

## 7. Execution Guide

### Prerequisites
```bash
pip install torch torchvision scikit-learn nibabel numpy pandas scipy matplotlib tqdm
```

### Reproducing the Pipeline

1. **Stage 1 — Dataset Preparation:**
   ```bash
   python stage1_fast_prep.py
   ```
2. **Stage 2 — Train Models & Benchmark:**
   ```bash
   python stage2_train_and_compare.py
   ```
3. **Stage 3 — Generate 3D Grad-CAM Overlays:**
   ```bash
   python stage3_gradcam_eval.py
   ```
4. **Independent Inference Verification:**
   ```bash
   python test_inference.py
   ```
