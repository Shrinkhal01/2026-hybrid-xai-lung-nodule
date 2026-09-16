# Hybrid XAI Lung Nodule Analysis

- Out of your 1,608 total nodules, exactly 366 of them had an indeterminate score of 3.0.
```
1,608 - 366 = 1,242
```
- This is why your Sanity Check cell printed out:Total valid nodules for binary classification: 1242

- The line that instructs the AI to drop the ambiguous Class -1 (Indeterminate) nodules so it can learn a clear, binary "Benign vs. Malignant" classification is : 
```
self.manifest = self.manifest[self.manifest['malignancy_class'] != -1].reset_index(drop=True)
```

- These 1,608 nodules are grouped into three classes based on the 4 radiologists' average malignancy scores:
```
Class 0 (Benign): Score < 3.0
Class 1 (Malignant): Score > 3.0
Class -1 (Indeterminate): Score exactly = 3.0 (borderline cases where the radiologists were evenly split).
```

---

## Model Performance & Evaluation Metrics

The final ensemble model was evaluated across **1,242 consensus-labeled nodules** using 5-fold cross-validation.

### Performance Summary Table

| Metric | Result | Description |
| :--- | :---: | :--- |
| **Overall Accuracy** | **86.00%** | Proportion of all nodules correctly classified as benign or malignant. |
| **Ensemble ROC-AUC** | **0.9042** | Diagnostic discrimination capability across all classification thresholds. |
| **Calibrated Threshold** | **0.5783** | Optimal decision boundary derived via Youden's J statistic. |
| **Malignant (Precision / Recall / F1)** | **0.77 / 0.76 / 0.77** | Solid detection of cancerous nodules despite dataset class imbalance. |
| **Benign (Precision / Recall / F1)** | **0.89 / 0.90 / 0.90** | High specificity and precision for non-malignant scans. |

---

### Detailed Metric Breakdown

#### 1. Overall Accuracy (86.00%)
* **Definition:** The ratio of correct predictions (both True Benign and True Malignant) to the total number of evaluated samples (1,242 nodules).
* **Clinical Significance:** 86% of patient scans receive an accurate automated diagnosis.

#### 2. Ensemble ROC-AUC (0.9042)
* **Definition:** The **Area Under the Receiver Operating Characteristic Curve**. It measures the model's ability to rank a randomly chosen malignant nodule higher than a randomly chosen benign nodule across every possible decision threshold.
* **Clinical Significance:** A score above **0.90** indicates excellent diagnostic discrimination, confirming that the ensemble generalizes well across unseen patient folds.

#### 3. Calibrated Decision Threshold (0.5783 via Youden's J)
* **Definition:** Rather than using an arbitrary 50% ($0.50$) default cutoff, the optimal threshold was mathematically calculated using **Youden's Index** ($J = \text{Sensitivity} + \text{Specificity} - 1$).
* **Clinical Significance:** Calibrates the consensus probability cutoff to maximize sensitivity (catching cancer) while minimizing false-positive alarms (reducing unnecessary invasive biopsies).

#### 4. Malignant (Cancerous) Detection
* **Precision (0.77):** When the model flags a nodule as malignant, it is correct **77%** of the time.
* **Recall / Sensitivity (0.76):** The model successfully identifies **76%** of all actual malignant nodules in the cohort.
* **F1-Score (0.77):** The harmonic mean of precision and recall ($\frac{2 \cdot P \cdot R}{P + R}$), providing a balanced benchmark for the minority malignant class.

#### 5. Benign (Non-Cancerous) Detection
* **Precision (0.89):** High reliability—**89%** of nodules predicted as benign are truly non-cancerous.
* **Recall / Specificity (0.90):** Successfully detects **90%** of all benign nodules, preventing misdiagnoses.
* **F1-Score (0.90):** Strong overall harmonic balance for the benign class.

---

### Multimodal Ablation Study

To evaluate the contribution of each branch, an ablation study was conducted:

| Model Configuration | ROC-AUC | Clinical Interpretation |
| :--- | :---: | :--- |
| **3D CNN Only** | `0.6901` | Spatial 3D image features alone struggle due to visual complexity and patch variance. |
| **Tabular MLP Only** | `0.8985` | Strong baseline using handcrafted radiologist features (subtlety, sphericity, margin, etc.). |
| **Hybrid Gated Fusion** | `0.9042` | **Best performance.** Multimodal gating dynamically weights visual + radiomic cues for superior classification. |
# 🫁 Hybrid Explainable AI: Lung Nodule Malignancy Detection

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)
![Medical Imaging](https://img.shields.io/badge/Medical_Imaging-SimpleITK-blue?style=for-the-badge)
<p align="center">
  <img src="hybrid-xai-lung-nodule.jpg" alt="Hybrid XAI Lung Nodule" width="600">
</p>

An end-to-end, multimodal machine learning pipeline designed to classify lung nodules as Benign or Malignant using the LIDC-IDRI dataset. This project goes beyond standard image classification by implementing a **Hybrid Gated Fusion Model** that dynamically merges 3D CT scan volumes with handcrafted radiomic features. 

To ensure clinical safety and transparency, the pipeline includes an **Explainable AI (XAI)** framework via 3D Grad-CAM to visualize the network's diagnostic reasoning.

*(Insert a screenshot of your Gradio Dashboard with the 3D Heatmaps here)*

---

## 📑 Table of Contents
1. [Project Highlights](#-project-highlights)
2. [Dataset & Preprocessing](#-dataset--preprocessing)
3. [Model Architecture](#-model-architecture)
4. [Explainable AI (XAI)](#-explainable-ai-xai)
5. [Performance & Metrics](#-performance--metrics)
6. [Modality Ablation Study](#-modality-ablation-study)
7. [Repository Structure & Usage](#-repository-structure--usage)

---

## 🚀 Project Highlights
* **Multimodal Learning:** Simultaneously processes unstructured 3D volumetric images and structured tabular clinical data.
* **Leakage-Free Cross-Validation:** Utilizes `GroupKFold` split by Patient ID to ensure robust, real-world generalization without data leakage.
* **Calibrated Decision Boundaries:** Replaces default probability thresholds with optimal cutoffs derived mathematically via Youden's J Statistic.
* **Live Diagnostic Dashboard:** Production-ready inference pipeline wrapped in an interactive web application via Gradio.

---

## 📊 Dataset & Preprocessing
**Source:** [LIDC-IDRI (The Cancer Imaging Archive)](https://www.cancerimagingarchive.net/collection/lidc-idri/)

The original dataset contains 1,608 expert-annotated nodules. To train a clear binary classifier, ambiguous borderline cases were removed:
* **Class 0 (Benign):** Consensus Score < 3.0
* **Class 1 (Malignant):** Consensus Score > 3.0
* **Class -1 (Indeterminate):** Consensus Score exactly = 3.0 (366 borderline cases dropped).

**Total valid nodules for binary classification: 1,242**

**Preprocessing Pipeline (`phase1preprocess.ipynb`):**
1. Resampled all CT volumes to an isotropic 1.0mm spacing.
2. Applied Hounsfield Unit (HU) lung windowing (-1000 to 400).
3. Extracted uniform $64 \times 64 \times 64$ 3D spatial tensors around nodule centroids.

---

## 🧠 Model Architecture
The network relies on a custom **Hybrid Gated Fusion** approach (`phase2.ipynb` & `phase4_full_scale.ipynb`):
1. **Visual Branch:** A 3D Convolutional Neural Network (CNN) extracts spatial morphological features directly from the voxel patches.
2. **Clinical Branch:** A Multi-Layer Perceptron (MLP) processes 5 radiomic features evaluated by radiologists (Subtlety, Sphericity, Margin, Spiculation, Texture).
3. **Gated Fusion Layer:** A dynamic sigmoid gating mechanism merges the 128 visual features and 32 clinical features, allowing the network to suppress noisy image data if clinical indicators are stronger, or vice versa.

---

## 🔍 Explainable AI (XAI)
Deep learning in healthcare requires transparency. This project implements **3D Grad-CAM (Gradient-weighted Class Activation Mapping)** attached to the final convolutional layer of the CNN branch (`phase3_evaluation.ipynb`). 

During inference, it backpropagates the target malignancy score to generate spatial heatmaps. These maps are overlaid onto the Axial, Coronal, and Sagittal slices of the original CT scan, explicitly highlighting the anatomical structures that triggered the model's diagnosis.

---

## 📈 Performance & Metrics
The final ensemble model was evaluated across the 1,242 consensus-labeled nodules using 5-fold cross-validation.

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Overall Accuracy** | 86.00% | Proportion of all nodules correctly classified. |
| **Ensemble ROC-AUC** | 0.9042 | Diagnostic discrimination capability across all thresholds. |
| **Calibrated Threshold** | 0.5783 | Optimal decision boundary derived via Youden's J statistic. |
| **Benign (Precision/Recall/F1)** | 0.89 / 0.90 / 0.90 | High specificity and precision for non-malignant scans. |
| **Malignant (Precision/Recall/F1)** | 0.77 / 0.76 / 0.77 | Solid detection of cancerous nodules despite dataset class imbalance. |

### Metric Insights
* **Ensemble ROC-AUC (0.9042):** Indicates excellent diagnostic discrimination, confirming that the ensemble generalizes well across unseen patient folds.
* **Calibrated Decision Threshold (0.5783):** Rather than using an arbitrary 50% default, Youden's Index ($J = \text{Sensitivity} + \text{Specificity} - 1$) calibrates the consensus probability cutoff to maximize sensitivity (catching cancer) while minimizing false-positive alarms.

---

## 🔬 Modality Ablation Study
To evaluate the contribution of each network branch, an ablation study was conducted by zeroing out specific input tensors during inference.

| Model Configuration | ROC-AUC | Clinical Interpretation |
| :--- | :--- | :--- |
| **3D CNN Only** | 0.6901 | Spatial 3D image features alone struggle due to visual complexity and patch variance. |
| **Tabular MLP Only** | 0.8985 | Strong baseline using handcrafted radiologist features (subtlety, sphericity, margin, etc.). |
| **Hybrid Gated Fusion** | **0.9042** | Best performance. Multimodal gating dynamically weights visual + radiomic cues for superior classification. |

---

## 💻 Repository Structure & Usage

This project was developed entirely within Google Colab to leverage cloud GPUs. The pipeline is split into sequential Jupyter Notebooks:

1. **`downloader.ipynb`**: Connects to TCIA to securely download raw LIDC-IDRI DICOM scans directly to Google Drive.
2. **`phase1preprocess.ipynb`**: Handles medical image resampling, HU windowing, and 3D patch extraction using `SimpleITK`.
3. **`phase2.ipynb`**: Prototypes the PyTorch `HybridGatedFusionModel` architecture and prevents data leakage using `GroupKFold`.
4. **`phase3_evaluation.ipynb`**: Calculates ensemble ROC-AUC, calibrates the threshold with Youden's J, and implements 3D Grad-CAM.
5. **`phase4_full_scale.ipynb`**: Full cohort training utilizing 3D spatial augmentations and Automatic Mixed Precision (AMP).
6. **`final_inference_pipeline.ipynb`**: The live, interactive AI diagnostic dashboard.

### How to Run the Live Dashboard
To test the pre-trained ensemble weights and generate your own XAI heatmaps:
1. Open the [`final_inference_pipeline.ipynb`](https://colab.research.google.com/drive/1GyK3uTITmAruTdwpRWFbhkgJPry00yHU#scrollTo=R4l6n5rNrDE5) file in Google Colab.
2. Ensure your Google Drive is mounted, and the notebook has access to the exported `.pt` model weights and the `manifest.csv`.
3. Select **Runtime > Run all** from the top menu.
4. Scroll to the final cell. You can interact with the Gradio dashboard directly within the Colab output window, or click the generated public URL (e.g., `https://xxxxx.gradio.live`) to open it in a full-screen browser tab.
