# Explainable 3D Deep Learning vs. Classical Machine Learning for Lung Nodule Malignancy Prediction on LIDC-IDRI

## Exact Problem Statement

- Given a 3D thoracic CT sub-volume centered on an annotated pulmonary nodule from the LIDC-IDRI dataset, formulate a binary classification task to predict whether the nodule is Benign (0) or Malignant (1), compare a classical feature-based machine learning baseline against an end-to-end 3D Convolutional Neural Network (3D CNN), and validate the spatial decision-making of the 3D CNN using 3D Grad-CAM visual explanations.

## Exact Input and Output

- Input:
  - For the 3D CNN: A single-channel 3D volumetric tensor of shape 1 \times 32 \times 32 \times 32 (representing depth, height, and width in voxels, normalized to [0, 1]).
  - For the Classical ML Baseline: A 1D feature vector of 9 radiologist-annotated morphological attributes (or basic intensity/shape summary stats extracted from the 32^3 crop).

- Output:
  - A scalar malignancy probability p \in [0, 1] and binary class label \hat{y} \in \{0, 1\}.
  - For the 3D CNN: A 32 \times 32 \times 32 3D Grad-CAM activation heatmap overlaid on the central axial CT slice.

## How Much of LIDC-IDRI You Actually Need Initially

- Development / Dry-Run Subset: Start with 100 CT scans (~12 GB) to verify your ROI extraction script works without errors.
- Final Course Project Subset: Process 300 to 400 CT scans (which yields roughly 600 to 800 valid nodules after filtering out indeterminate cases).
- Once cropped into 32 \times 32 \times 32 .npy arrays, 800 nodules take up less than 150 MB of disk space. You only parse the raw DICOMs once, save the tiny .npy dataset locally on your GPU server, and never touch the 133 GB folder again.

## How to Extract Nodule ROIs from the Annotations

Use the official pylidc Python library, which already contains the parsed XML coordinates and radiologist ratings for every scan—meaning you only need to point pylidc to the CT DICOM folders:

1. Query pylidc for scans and call scan.cluster_annotations() to group overlapping annotations from the 4 radiologists into distinct nodules.
2. Filter for nodules annotated by at least 2 or 3 radiologists to avoid noisy false positives.
3. Compute the 3D centroid (z, y, x) from the consensus bounding box (pylidc.annotation.consensus).
4. Convert the scan slice values to Hounsfield Units (HU), clip to the standard lung window [-1000, 400], and min-max normalize to [0, 1].
5. Crop a fixed 32 \times 32 \times 32 voxel cube centered at (z, y, x) and save it as nodule_id.npy. (Optional minimal step: resample to 1\text{ mm}^3 via scipy.ndimage.zoom before cropping so physical scale is uniform).

## Recommended Label-Generation Strategy for Malignancy

Each radiologist rates malignancy on an ordinal scale from 1 (Highly Unlikely) to 5 (Highly Suspicious). For each clustered nodule, compute the mean malignancy score \bar{m} across the annotating radiologists:

- \bar{m} < 3.0 \rightarrow Benign (Label = 0)
- \bar{m} > 3.0 \rightarrow Malignant (Label = 1)
- \bar{m} = 3.0 \rightarrow Drop / Exclude. (Score 3 means "Indeterminate/Uncertain." Dropping it is standard practice in LIDC-IDRI literature and makes binary classification much cleaner).

## One Classical ML Baseline

Random Forest Classifier (from scikit-learn).

- Why: Zero GPU tuning required, resistant to overfitting on small tabular datasets, and serves as a rock-solid Advanced ML baseline.
- Features to feed it (No extra library needed): pylidc already provides 8 semantic descriptors per annotation (subtlety, internalstructure, calcification, sphericity, margin, lobulation, spiculation, texture) plus diameter. Alternatively, if you want a pure image-based comparison without human-rated semantics, compute 10 basic statistical/morphological numbers directly from the 3D NumPy array + consensus mask (mean intensity, standard deviation, skewness, kurtosis, nodule volume/voxel count, bounding box surface area, and sphericity).

## One Deep-Learning Model

Pre-built 3D ResNet-18 (torchvision.models.video.r3d_18 or monai.networks.nets.resnet18(spatial_dims=3)).

- Why: You do not have to write a custom CNN from scratch. Modify only the first convolutional layer to accept 1 input channel (CT grayscale instead of 3-channel RGB) and the final fully connected layer (fc) to output 1 logit (for BCEWithLogitsLoss).
- On your college GPU, training r3d_18 on ~700 nodules of size 32^3 for 30–40 epochs will finish in under 5 minutes.

## One XAI Technique

3D Grad-CAM (displayed on the central 2D axial slice).

- How: Hook into the final convolutional block (layer4) of your trained 3D ResNet-18 to generate a 32 \times 32 \times 32 class activation volume for test nodules. Extract the middle slice (index 16 along the z-axis) of both the CT cube and the Grad-CAM volume, and plot them side-by-side using matplotlib (cmap='jet' overlaid on cmap='gray').

## Exact Evaluation Metrics

Report the following 5 numbers on the held-out Test Set for both the Random Forest and the 3D ResNet-18:

1. Accuracy
2. ROC-AUC (Primary metric for threshold-independent ranking)
3. Sensitivity (Recall) (Crucial in medicine: percentage of actual malignant nodules caught)
4. Specificity (Percentage of actual benign nodules correctly cleared)
5. F1-Score

11. Minimal Experiment Plan

Split your extracted nodules once using a Patient-Level Stratified Split: 70% Train / 15% Validation / 15% Test (ensuring all nodules from a single patient stay in the same split).

Experiment ID	Model	Input Data	Purpose in Report
Exp 1 (Baseline)	Random Forest (n_estimators=100)	1D Feature Vector (10 statistical/shape features from the 3D ROI)	Establishes classical ML performance on summary features.
Exp 2 (Proposed DL)	3D ResNet-18 (BCEWithLogitsLoss, Adam)	1 \times 32 \times 32 \times 32 3D CT ROI	Demonstrates the advantage of volumetric spatial representation learning.
Exp 3 (Explainability)	3D Grad-CAM on Exp 2 model	4 Test Set Nodules (2 True Positives, 1 True Negative, 1 False Positive)	Visually proves the 3D CNN focuses on the nodule mass/borders rather than background noise.

12. Final Project Architecture (1 Phase, 3 Internal Stages)

================================================================================
                        SINGLE-PHASE PROJECT PIPELINE
================================================================================

[STAGE 1: DATASET PREPARATION]
  Raw LIDC-IDRI (300-400 CT Scans + XMLs)
         │
         ▼  (pylidc clustering + 50% consensus)
  HU Windowing [-1000, 400] & Normalization [0, 1]
         │
         ▼  (Crop around centroid & compute mean malignancy m̄)
  Saved Locally: 32×32×32 .npy Cubes + Labels CSV (m̄ < 3: Benign, m̄ > 3: Malignant)
         │
         ▼  (Patient-Stratified Split: 70% Train / 15% Val / 15% Test)

--------------------------------------------------------------------------------

[STAGE 2: MODEL TRAINING & COMPARISON]
         ├───────────────────────────────────────┐
         ▼                                       ▼
  (Path A: Classical ML Baseline)         (Path B: Deep Learning Model)
  Extract 10 Summary Stats/Shape Features Feed 1×32×32×32 Volumetric Tensor
         │                                       │
         ▼                                       ▼
  Random Forest Classifier                3D ResNet-18 (BCEWithLogitsLoss)

--------------------------------------------------------------------------------

[STAGE 3: EXPLAINABILITY & EVALUATION]
         │                                       │
         └───────────────┬───────────────────────┘
                         ▼
          Quantitative Comparison on Test Set
      (Accuracy, ROC-AUC, Sensitivity, Specificity, F1)
                         │
                         ▼
          3D Grad-CAM on Trained 3D ResNet-18
      (Central Axial Slice Heatmap Overlay for 4 Test Cases)
================================================================================


13. What Should NOT Be Implemented

To protect your timeline and keep the codebase clean, strictly exclude all of the following:

⚬ No whole-lung segmentation (no lung parenchyma masking or morphological lung extraction).
⚬ No automated nodule detection (no Faster R-CNN, YOLO, or 3D U-Net candidate generation).
⚬ No external PyRadiomics C-extension compilation (use basic NumPy / scipy / skimage summary stats for the Random Forest baseline).
⚬ No multimodal fusion networks, gated attention units, or hybrid dual-branch training loops.
⚬ No K-Fold Cross-Validation (a single 70/15/15 patient-stratified split is sufficient).
⚬ No SHAP, LIME, calibration curves, or FROC analysis.
⚬ No DX or CR (2D X-ray) files from the LIDC download—ignore every folder that isn't a CT scan.

14. Final List of Deliverables

At the end of this project, you will have 3 clean Python scripts and 4 visual artifacts for your report and slides:

Code Deliverables:

1. stage1_prepare_dataset.py — Reads 300–400 CT scans via pylidc, extracts 32 \times 32 \times 32 .npy cubes, extracts baseline features, and outputs metadata.csv with train/val/test splits.
2. stage2_train_and_compare.py — Trains the Random Forest and the 3D ResNet-18, logs training/validation loss curves, and prints the comparative test metrics table.
3. stage3_gradcam_eval.py — Loads the saved 3D ResNet-18 weights (best_model.pth), computes 3D Grad-CAM, and saves the plots.

Report / Presentation Deliverables:

1. 1 Comparison Table: Test set Accuracy, ROC-AUC, Sensitivity, Specificity, and F1 for Random Forest vs. 3D ResNet-18.
2. 1 ROC Curve Plot: Both models' ROC curves plotted on the same graph.
3. 1 Confusion Matrix: Side-by-side confusion matrices for both models.
4. 1 Grad-CAM Figure: A 2 \times 2 or 4 \times 3 grid showing Original CT Slice | Grad-CAM Heatmap | Overlay for 4 test nodules.