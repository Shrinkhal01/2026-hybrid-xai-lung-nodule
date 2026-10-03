# 🖥️ Presentation Slide Deck Summary (10 Slides)
## Presentation File: [`Hybrid_XAI_Lung_Nodule_Presentation.pptx`](file:///Users/shrinkhals/Shrinkhal-Github/Projects/2026-hybrid-xai-lung-nodule/Hybrid_XAI_Lung_Nodule_Presentation.pptx)
**Format:** 16:9 Widescreen (13.333" x 7.5")  
**Theme:** Modern Clinical AI Dark Palette (Navy `#0B132B`, Slate `#1C2541`, Cyan `#00F5D4`, Teal `#48CAE4`, Gold `#F7B801`, Text `#FFFFFF`)  
**Total Slides:** Exactly 10 Slides  
**Target Speaking Time:** ~9–11 minutes  

---

### Slide 1: Title & Key Highlights
* **Header Tag:** `RESEARCH & ENGINEERING DEFENSE`
* **Title:** Hybrid Explainable AI Framework for Pulmonary Nodule Malignancy Detection
* **Subtitle:** An End-to-End Multimodal Deep Learning System Merging 3D CT Volumetric Tensors with Handcrafted Clinical Radiomics via Dynamic Gated Fusion and 3D Grad-CAM Explainability
* **Key Stat Badges:**
  * **0.9042** — Ensemble ROC-AUC
  * **86.00%** — Overall Accuracy
  * **0.5783** — Youden's J Cutoff
  * **1,242** — Consensus Nodules

---

### Slide 2: Clinical Motivation & The Healthcare Dilemma
* **Header Tag:** `CLINICAL CONTEXT`
* **Title:** Clinical Motivation & The Healthcare Dilemma
* **Subtitle:** Lung cancer causes 1.8M annual deaths worldwide; diagnostic uncertainty and opaque AI models create clinical bottlenecks.
* **Content Cards:**
  1. **The High Clinical Stakes:** Leading cancer killer worldwide; Stage I detection gives >60% 5-year survival vs <10% for late detection; visual subtlety of 3–30mm nodules.
  2. **Radiologist Uncertainty:** High inter-observer disagreement on borderline nodules; 20–30% of surgical biopsies turn out benign; cognitive fatigue in thin-slice CT reviews.
  3. **The AI 'Black-Box' Trap:** Uninterpretable deep networks create clinical distrust; our solution merges 3D image voxels with radiomics and multi-planar 3D Grad-CAM heatmaps.

---

### Slide 3: Dataset Architecture & Ground Truth Consensus
* **Header Tag:** `DATASET & GROUND TRUTH`
* **Title:** LIDC-IDRI Dataset: 4-Radiologist Consensus Ground Truth
* **Subtitle:** Establishing a rigorous binary classification benchmark across 1,010 thoracic CT scan subjects.
* **Content Cards:**
  * **The LIDC-IDRI Cohort:** 1,010 thoracic CT patients across 1,018 series; ~133 GB raw DICOM slices; 4 independent thoracic radiologists; 1,608 unique nodules on a 1–5 scale.
  * **Consensus Ground Truth Partitioning:** Mean rating across radiologists; **dropped 366 indeterminate nodules** (score == 3.0); leaves **exactly 1,242 consensus nodules** (709 benign, 533 malignant) to eliminate label noise.

---

### Slide 4: End-to-End Pipeline & Medical Preprocessing (Phases 0 & 1)
* **Header Tag:** `PHASES 0 & 1 PIPELINE`
* **Title:** Automated Cloud Ingestion & Medical 3D Preprocessing
* **Subtitle:** Translating 133 GB of heterogeneous clinical DICOM scans into standardized, AI-ready 3D tensors.
* **Content Cards:**
  1. **Phase 0 Cloud Streaming:** Direct cloud-to-cloud byte streaming from TCIA REST API into Google Drive (`downloader.ipynb`); resume-safe checkpointing.
  2. **Isotropic 1.0mm Resampling:** SimpleITK resamples all volumes to universal 1.0 x 1.0 x 1.0 mm physical spacing to eliminate scanner slice-thickness bias.
  3. **HU Windowing & 64³ Patches:** Lung windowing `[-1000, 400]` HU isolates pulmonary tissue; min-max normalized to `[0, 1]`; cropped $64 \times 64 \times 64$ subvolume around nodule centroids.
  4. **5 Radiomic Features & Manifest:** Subtlety, Sphericity, Margin, Spiculation, and Texture extracted via `pylidc`; saved as individual `.pt` tensors with master `manifest.csv`.

---

### Slide 5: Hybrid Gated Fusion Architecture (Phase 2)
* **Header Tag:** `MODEL ARCHITECTURE`
* **Title:** Phase 2: Dual-Branch Network & Dynamic Gated Fusion
* **Subtitle:** A dual-stream architecture adaptively modulating spatial 3D convolutions with clinical radiomics.
* **Content Cards:**
  * **Dual-Branch Network Structure:**
    * 3D CNN (Visual): Ingests (B, 1, 64, 64, 64) -> 4 cascaded Conv3D blocks (16, 32, 64, 128) + ReLU + MaxPool3D(2) -> Flatten -> Linear(256) -> Linear(128) feature embedding.
    * Tabular MLP (Clinical): Ingests (B, 5) vector -> Linear(5, 16) -> ReLU -> Linear(16, 32) -> ReLU -> 32-d radiomic embedding.
  * **Dynamic Sigmoid Gating Mechanism:**
    * Concatenation: $\mathbf{z} = [\mathbf{f}_{\text{img}} \,\|\, \mathbf{f}_{\text{tab}}] \in \mathbb{R}^{160}$
    * Gating Vector: $\mathbf{g} = \sigma(\mathbf{W}_g \mathbf{z} + \mathbf{b}_g) \in (0, 1)^{160}$
    * Hadamard Fusion: $\mathbf{z}_{gated} = \mathbf{z} \odot \mathbf{g}$
    * Suppresses noisy image channels if CT is motion-blurred, preventing visual noise from dominating the classification head.

---

### Slide 6: Leakage-Free Training & Scaled Optimization (Phases 2 & 4)
* **Header Tag:** `TRAINING & OPTIMIZATION`
* **Title:** Leakage-Free Validation & Full-Scale GPU Training
* **Subtitle:** Ensuring rigorous patient-level generalizability and efficient mixed-precision GPU optimization.
* **Content Cards:**
  1. **Patient-Level GroupKFold:** Enforced `GroupKFold(n_splits=5)` grouped strictly by `patient_id`; 0% patient identity overlap between train and validation folds.
  2. **Data Hygiene & Imbalance:** `StandardScaler` fit strictly on training fold indices before transforming validation folds; loss weighted inversely by class frequencies.
  3. **Automatic Mixed Precision (AMP):** `torch.cuda.amp.autocast()` + `GradScaler()` cuts VRAM by 50% and doubles throughput; `torchio` 3D spatial augmentations (RandomAffine, RandomFlip) prevent slice memorization.

---

### Slide 7: Ensemble Soft Voting & Youden's J Calibration (Phase 3)
* **Header Tag:** `EVALUATION & CALIBRATION`
* **Title:** Phase 3: Ensemble Soft Voting & Decision Calibration
* **Subtitle:** Aggregating 5 cross-validation fold models with mathematically calibrated threshold optimization.
* **Content Cards:**
  * **5-Fold Committee Consensus:** Averages predicted probabilities across all 5 fold models simultaneously ($P = \frac{1}{5} \sum Softmax(M_k(x))$), cancelling out individual fold variance.
  * **Decision Cutoff Calibration (Youden's J):** Maximizes $J = \text{TPR} - \text{FPR}$, identifying **0.5783 (57.83%)** as the optimal cutoff. Balances cancer sensitivity (76%) with 90% benign specificity, eliminating unnecessary biopsies.

---

### Slide 8: Explainable AI: 3D Grad-CAM Transparency (Phase 3)
* **Header Tag:** `EXPLAINABLE AI (XAI)`
* **Title:** Phase 3: 3D Grad-CAM Voxel-Level Transparency
* **Subtitle:** Visualizing the network's internal diagnostic reasoning across Axial, Coronal, and Sagittal planes.
* **Content Cards:**
  * **3D Grad-CAM Engine:** Backward hooks capture gradients on conv block 4; volumetric global average pooling computes channel weights; trilinear interpolation resamples map back to $64^3$.
  * **Tri-Planar Clinical Verification:** Overlaid simultaneously onto Axial, Coronal, and Sagittal CT views; proves model focuses on irregular nodule borders rather than ribs or noise; fulfills EU AI Act & FDA SaMD auditability mandates.

---

### Slide 9: Performance Results, Ablation Study & Benchmarks
* **Header Tag:** `EXPERIMENTAL RESULTS`
* **Title:** Model Performance, Ablation Study & Benchmarks
* **Subtitle:** Comprehensive validation across 1,242 consensus nodules under 5-fold cross-validation.
* **Performance Table:**
  * Overall Accuracy: **86.00%** (at 0.5783 cutoff)
  * Ensemble ROC-AUC: **0.9042**
  * Benign (Precision / Recall / F1): **0.89 / 0.90 / 0.90**
  * Malignant (Precision / Recall / F1): **0.77 / 0.76 / 0.77**
* **Ablation Matrix & Literature Benchmarks:**
  * 3D CNN Only: 0.6901 ROC-AUC (-0.2141 drop)
  * Tabular MLP Only: 0.8985 ROC-AUC
  * **Hybrid Gated Fusion:** **0.9042 ROC-AUC** (Best Overall)
  * Literature: Traditional SVM (0.850), Standard 3D CNN (0.930), Our Hybrid Model (0.9042 with 3D XAI), NoduleX SOTA (0.971).

---

### Slide 10: Live Gradio Dashboard, Clinical Impact & Conclusion
* **Header Tag:** `DEPLOYMENT & CONCLUSION`
* **Title:** Live Gradio Dashboard, Clinical Impact & Conclusion
* **Subtitle:** Translating deep learning research into a point-of-care clinical tool (`final_inference_pipeline.ipynb`).
* **Content Cards:**
  * **Live Production Web Dashboard:** Dropdown nodule selector; 5 interactive radiomic sliders; sub-2-second ensemble inference; diagnostic readout + calibrated cutoff comparison; synchronized Axial, Coronal, and Sagittal Grad-CAM overlays.
  * **Clinical Impact & Core Conclusions:**
    ✔ 89% benign precision and 90% specificity reduce unnecessary invasive biopsies.  
    ✔ Operates as an interactive human-in-the-loop second reader (CADx).  
    ✔ Automated cloud-to-cloud pipeline processing 133 GB DICOMs.  
    ✔ Dynamic Gated Fusion surpassing standalone modalities (0.9042 AUC).  
    ✔ Zero patient leakage via strict 5-Fold GroupKFold validation.  
    *Open for Questions & Defense.*
