# 🫁 CT Scan Data Acquisition & Preprocessing Pipeline (Phase 0 & Phase 1)
> **Master 50 Q&A Study Guide for Offline Phone & Markdown Note Reading**

---

## 📊 Summary Overview Table

| Pipeline Stage | Key Technology / Tool | Output / Parameter Standard |
| :--- | :--- | :--- |
| **Phase 0 Acquisition** | `tcia_utils` (NBIA REST API) | 1,018 CT series (1,010 patients), ~133 GB raw DICOM |
| **Isotropic Resampling** | `SimpleITK` Linear Resampler | Universal (1.0, 1.0, 1.0) mm physical voxel spacing |
| **Lung Windowing** | [-1000, 400] HU Min-Max Scaling | Clipped HU mapped cleanly into [0.0, 1.0] |
| **Patch Extraction** | 3D Physical Centroid Crop | 64 x 64 x 64 PyTorch tensors shape `(1, 64, 64, 64)` `.pt` |
| **Consensus Labels** | `pylidc` Annotation Clustering | 1,608 nodules -> 1,242 consensus targets (`manifest.csv`) |

---

## 📌 Phase 0: Raw Data Acquisition (downloader.ipynb)

### Q1: What is the main objective of Phase 0 in the pipeline?
**Answer:** The main objective of Phase 0 is to programmatically query and stream raw computed tomography (CT) scans from The Cancer Imaging Archive (TCIA) public dataset (LIDC-IDRI collection) directly into persistent cloud storage.

### Q2: What is tcia_utils and why is it used instead of manual web downloads?
**Answer:** tcia_utils is the official Python client wrapper for the National Biomedical Imaging Archive (NBIA) REST API. It allows programmatically searching, filtering, downloading, and automatically unzipping DICOM series without needing manual browser downloads or .tcia manifest files.

### Q3: What is NBIA and how does tcia_utils interact with it?
**Answer:** NBIA (National Biomedical Imaging Archive) is the underlying REST API repository for TCIA. tcia_utils sends HTTPS request queries to NBIA endpoints to retrieve series metadata, slice counts, and series instance UIDs.

### Q4: Why is Google Drive mounted at /content/drive during Colab execution?
**Answer:** Google Colab virtual machines are ephemeral (temporary). Storing ~133 GB of raw CT scans on Colab's local container disk would result in complete data loss as soon as the session disconnects or times out.

### Q5: What happens to local files on Colab if the runtime disconnects?
**Answer:** All files written to Colab's local disk (/content/) are permanently erased upon container disconnect or timeout. Only files saved to mounted Google Drive or external persistent volumes survive.

### Q6: Why is modality='CT' explicitly passed when querying nbia.getSeries?
**Answer:** The LIDC-IDRI collection contains heterogeneous medical imaging modalities, including chest X-rays, scout scans, and localizers. Setting modality='CT' filters out non-CT series, returning exactly 1,018 series across 1,010 patients.

### Q7: How many CT series and unique patients are returned by the LIDC-IDRI query?
**Answer:** The query returns exactly 1,018 CT series across 1,010 unique patient cases.

### Q8: What is the approximate total storage footprint of the raw LIDC-IDRI dataset?
**Answer:** The uncompressed raw DICOM dataset occupies approximately 133 GB of disk space.

### Q9: How does resume-safe streaming work in nbia.downloadSeries()?
**Answer:** tcia_utils inspects the destination directory on disk before downloading each series. If a folder for a given patient/series UID already exists and contains unzipped files, it logs a warning and automatically skips to the next missing series.

### Q10: What directory layout is created on Google Drive after Phase 0 completes?
**Answer:** The downloaded files are saved as: raw_data/<LIDC-IDRI-PatientID>/<SeriesInstanceUID>/*.dcm, where each folder contains 2D axial slice DICOM files.

---

## 📌 Medical Imaging Challenges & Preprocessing Rationale

### Q11: What is voxel resolution heterogeneity and why is it a problem for 3D CNNs?
**Answer:** CT scanners vary widely in slice thickness (0.6 mm to 3.0 mm) and in-plane pixel spacing (~0.5 mm to 0.8 mm). A 10 mm nodule would occupy 10 slices on a high-resolution scan but only 3 slices on a coarse scan, distorting 3D spatial convolutions.

### Q12: Why cannot full-size CT volumes (512x512x400) be directly fed into neural networks?
**Answer:** A full 3D CT scan matrix contains over 100 million voxels, mostly composed of irrelevant anatomical structures (air, ribs, bed, heart). Processing full volumes directly would cause extreme GPU memory overflow (OOM) and massive class imbalance.

### Q13: What is the range of raw CT voxel values and what unit represents them?
**Answer:** Raw CT voxels represent tissue radiodensity measured in Hounsfield Units (HU), ranging from -2000 HU (air) up to +3000 HU (dense bone / metal implants).

### Q14: What causes inter-observer variance in clinical CT annotations?
**Answer:** Different radiologists have varying thresholds for nodule boundaries and malignancy risk. In LIDC-IDRI, up to 4 independent radiologists reviewed each scan, producing overlapping but non-identical 3D contour markings and malignancy scores.

### Q15: What are the two primary output assets produced at the end of Phase 1?
**Answer:** 1. Individual 3D image patch tensors saved as .pt files inside processed_patches/patches/. 2. A master tabular metadata registry saved as processed_patches/manifest.csv.

### Q16: What is the role of pydicom and SimpleITK in medical image processing?
**Answer:** pydicom parses raw DICOM metadata tags (slice thickness, pixel spacing, patient orientation). SimpleITK provides high-performance C++ backend routines for 3D spatial transforms, isotropic resampling, and image manipulation.

---

## 📌 Isotropic Voxel Resampling (SimpleITK)

### Q17: What is an isotropic volume in 3D medical imaging?
**Answer:** An isotropic volume is a 3D grid where each physical voxel has identical dimensions along all three axes (dx = dy = dz), meaning 1 voxel represents a perfect 1 mm x 1 mm x 1 mm cube.

### Q18: Why are raw CT slice thicknesses non-isotropic?
**Answer:** Raw CT scans are typically acquired with fine in-plane resolution (e.g., 0.7 mm x 0.7 mm) but coarser z-axis slice steps (e.g., 2.5 mm slice thickness) to reduce radiation dose and scan acquisition time.

### Q19: What target spacing is selected for resampling in Phase 1?
**Answer:** A universal target spacing of (1.0 mm, 1.0 mm, 1.0 mm) physical voxel dimensions is applied across all patient volumes.

### Q20: What is the mathematical formula for computing target volume dimensions?
**Answer:** N_target = round( N_original * (Spacing_original / Spacing_target) ), computed independently for x, y, and z axes.

### Q21: What interpolation method is used for resampling CT image volumes in SimpleITK?
**Answer:** Linear interpolation (sitk.sitkLinear) is used via SimpleITK.ResampleImageFilter to smoothly compute physical intensity values at new grid coordinates.

### Q22: How does isotropic resampling affect physical distance calculations?
**Answer:** Because every voxel is standardized to 1 mm^3, spatial distance in voxel indices directly equals physical distance in millimeters, allowing invariant 3D geometric filters.

### Q23: Why is linear interpolation preferred over nearest-neighbor for continuous voxel intensities?
**Answer:** Nearest-neighbor interpolation creates artificial stair-step boundary artifacts, whereas linear interpolation preserves smooth continuous radiodensity gradients across tissue transitions.

### Q24: What happens to the voxel grid length along the z-axis after resampling a 2.5 mm slice thickness volume to 1.0 mm?
**Answer:** The number of z-axis slices increases by a factor of 2.5 (e.g., 100 slices become 250 slices), refining spatial continuity along the axial dimension.

---

## 📌 Hounsfield Units, Lung Windowing, & Normalization

### Q25: What is a Hounsfield Unit (HU) and how is it calculated?
**Answer:** HU measures tissue radiodensity relative to distilled water (0 HU) and vacuum/air (-1000 HU): HU = 1000 * (mu_tissue - mu_water) / (mu_water - mu_air).

### Q26: What are typical HU values for air, water, lung tissue, soft tissue, and bone?
**Answer:** Air: -1000 HU; Water: 0 HU; Lung Parenchyma: -900 to -500 HU; Soft Muscle Tissue: +20 to +40 HU; Dense Bone: +400 to +1000+ HU.

### Q27: What is CT windowing and why is it essential for lung analysis?
**Answer:** Windowing truncates radiodensity outside a target clinical range, discarding non-relevant anatomical structures (like external air or dense bone) to maximize dynamic contrast in soft lung tissue.

### Q28: What upper and lower HU bounds define the Lung Window applied in Phase 1?
**Answer:** The Lung Window clamps raw HU values between a lower bound of -1000 HU and an upper bound of +400 HU (a window width of 1400 HU).

### Q29: What mathematical formula normalizes windowed HU values into [0.0, 1.0]?
**Answer:** I_norm = ( clip(I_raw, -1000, 400) - (-1000) ) / (400 - (-1000)) = ( clip(I_raw, -1000, 400) + 1000 ) / 1400.

### Q30: What happens to voxels with raw values below -1000 HU or above +400 HU?
**Answer:** Voxels below -1000 HU are clipped to 0.0 (pure black), and voxels above +400 HU are clipped to 1.0 (pure white).

### Q31: Why is min-max scaling to [0.0, 1.0] necessary before neural network input?
**Answer:** Unbounded raw HU values (-1000 to +3000) would cause exploding gradients and numerical instability during Backpropagation in PyTorch neural networks.

### Q32: How does lung windowing help the model distinguish soft nodule tissue?
**Answer:** By removing high-density bone signals and low-density external air, the dynamic range of the tensor is dedicated entirely to differentiating ground-glass vs solid nodule tissue.

---

## 📌 3D Subvolume Patch Extraction & Tensor Storage

### Q33: What subvolume dimensions are used when cropping 3D nodule patches?
**Answer:** Each nodule is cropped into a 64 x 64 x 64 voxel 3D subvolume.

### Q34: How is the center of the cropped 3D patch determined for each nodule?
**Answer:** The patch center is set to the exact physical (z, y, x) centroid computed from radiologist annotation contours in the resampled 1.0 mm volume.

### Q35: How are coordinate transformations performed between original and resampled volumes?
**Answer:** Original voxel indices are converted to physical millimeter positions using the DICOM origin and spacing matrix, then mapped to new integer index positions in the resampled grid.

### Q36: What happens when a nodule centroid is located near the boundary of the CT volume?
**Answer:** The extract_3d_patch function automatically applies zero-padding along any axis where the 64 x 64 x 64 bounding box extends outside the scan volume bounds.

### Q37: What PyTorch tensor format and shape is saved for each extracted nodule patch?
**Answer:** Patches are stored as float32 PyTorch tensors with shape (1, 64, 64, 64) representing (channels, depth, height, width).

### Q38: Where are individual .pt tensor files stored on disk?
**Answer:** They are stored as individual files under: processed_patches/patches/<nodule_id>.pt.

---

## 📌 Radiologist Consensus & pylidc Configuration

### Q39: What is pylidc and what primary functions does it provide?
**Answer:** pylidc is an open-source Python library developed specifically for the LIDC-IDRI dataset to query scans, cluster radiologist XML annotations, and compute consensus statistics.

### Q40: How is ~/.pylidcrc configured to link patient IDs to raw DICOM folders?
**Answer:** A configuration file ~/.pylidcrc is created containing: [pylidc]
path = /path/to/raw_data
warn = False. pylidc builds a local SQLite index mapping patient IDs to folders.

### Q41: What function in pylidc clusters overlapping 3D markings from multiple radiologists?
**Answer:** scan.cluster_annotations() clusters 3D spatial contour overlaps from up to 4 independent reviewing radiologists into distinct nodule entities.

### Q42: How is the consensus malignancy rating computed across reviewing radiologists?
**Answer:** Radiologist malignancy ratings (scale 1 to 5) for a given nodule cluster are averaged: Consensus_Score = (1/K) * sum(rating_i).

### Q43: What threshold values determine binary class assignments?
**Answer:** Score < 3.0 -> Class 0 (Benign); Score > 3.0 -> Class 1 (Malignant); Score == 3.0 -> Class -1 (Indeterminate).

### Q44: What are 'indeterminate' nodules (Class -1) and how are they handled in Phase 2?
**Answer:** Indeterminate nodules have a average score of exactly 3.0 (borderline/disagreed cases). They are preserved in manifest.csv for completeness but dropped during Phase 2 training to avoid noisy labels.

### Q45: Why are indeterminate cases preserved in manifest.csv even if dropped during training?
**Answer:** Preserving all 1,608 extracted nodules in manifest.csv ensures complete auditability and allows secondary experiments (e.g., three-class classification or uncertainty estimation).

---

## 📌 Radiomic Feature Mining & Master Registry (manifest.csv)

### Q46: What 5 morphological radiomic features are extracted from radiologist annotations?
**Answer:** 1. Subtlety, 2. Sphericity, 3. Margin, 4. Spiculation, and 5. Texture.

### Q47: Define the Subtlety biomarker and its clinical scoring scale.
**Answer:** Subtlety measures visual contrast and difficulty of detection on a 1 (extremely subtle) to 5 (obvious) scale.

### Q48: Define Sphericity, Margin, Spiculation, and Texture.
**Answer:** Sphericity: roundness (1=linear to 5=perfect sphere); Margin: border clarity (1=ill-defined to 5=sharp); Spiculation: starburst spikes (1=smooth to 5=highly spiculated/malignant); Texture: internal density (1=ground glass to 5=solid).

### Q49: What information is recorded in each row of manifest.csv?
**Answer:** nodule_id, patient_id, series_instance_uid, patch_path, centroid_z, centroid_y, centroid_x, consensus_malignancy, malignancy_class, subtlety, sphericity, margin, spiculation, and texture.

### Q50: How many total unique nodules are extracted into manifest.csv and how many consensus nodules remain for Phase 2 training?
**Answer:** A total of 1,608 unique nodules are logged in manifest.csv. After removing the 366 indeterminate (Class -1) cases in Phase 2, exactly 1,242 consensus nodules remain for training.

---

