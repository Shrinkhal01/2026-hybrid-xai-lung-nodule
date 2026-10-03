## NOTES
- This project is a comprehensive end-to-end deep learning pipeline for lung nodule malignancy classification, utilizing the LIDC-IDRI dataset.
- The goal is to accurately classify lung nodules as either benign or malignant by combining 3D medical imaging (CT scans) with tabular clinical features. 
- The project is structured into four distinct phases :
1. **Phase 1: Data Acquisition & Preprocessing**

    > **Downloading:** The downloader.ipynb script securely fetches the raw LIDC-IDRI CT scan series directly from The Cancer Imaging Archive (TCIA) into your Google Drive.
    > **Preprocessing:** The phase1preprocess.ipynb notebook handles the heavy lifting of medical image processing. It converts raw DICOM files into 3D volumes, resamples them to an isotropic resolution, and applies "lung windowing" (clipping Hounsfield Units) to isolate lung tissue.
    > **Extraction:** It uses the pylidc library to parse expert annotations, extracts targeted 64x64x64 3D patches around the nodules, and gathers tabular features (subtlety, sphericity, margin, spiculation, texture) to create a structured dataset (manifest.csv).

2. **Phase 2: Model Architecture & Prototyping**

    > **Hybrid Architecture:** In phase2.ipynb, a Hybrid Gated Fusion Model is built. It uses a two-pronged approach:
            > **3D CNN branch:** Processes the spatial 3D image patches.
            > **Tabular MLP branch:** Processes the clinical metadata.
    > **Gated Fusion:** The features from both branches are concatenated and passed through a gating mechanism to dynamically weight the importance of the image versus the tabular data before making a final classification.
    > **Validation:** The model is prototyped using a 5-fold GroupKFold cross-validation strategy to ensure no data leakage occurs at the patient level.

3. **Phase 3: Evaluation & Explainability (XAI)**

    > **Ensemble Inference:** phase3_evaluation.ipynb evaluates the trained models from all 5 folds together, calculating an ensemble ROC-AUC score and using Youden's J Statistic to find the mathematically optimal probability threshold.
    > **Explainable AI (XAI):** It features a custom 3D Grad-CAM implementation to generate heatmaps over the CT scans. This visualizes exactly where the model is looking when it predicts a nodule is malignant, ensuring clinical trust.
    > **Ablation Study:** A modality ablation study is conducted (zeroing out images, then zeroing out tabular data) to scientifically prove that the hybrid combination outperforms using either data source on its own.

4. **Phase 4: Full-Scale Optimization**

    > **Optimized Training:** The phase4_full_scale.ipynb notebook scales the training loop to the entire dataset (over 1,200 nodules). It implements Automatic Mixed Precision (AMP) to drastically reduce memory consumption and speed up training on the GPU.
    > **Benchmarking:** Finally, it compares the model's performance against established literature (like Traditional SVMs, Standard 3D CNNs, and SOTA models like NoduleX), tracking its trajectory to highly competitive AUC scores.


## INFORMATION FILE WISE

### 1. downloader.ipynb
- *Core Libraries in the Setup Phase*
    1. tcia_utils
        - TCIA = The Cancer Imaging Archive
        - Instead of manually downloading ZIP files from a browser, this library allows you to programmatically query and download the exact CT scans you need.
        - Import its nbia (National Biomedical Imaging Archive) module. 
        - This gives you access to functions like ```nbia.getSeries``` to find the files and ```nbia.downloadSeries``` to pull them down.
###    2. google.colab.drive
        - Google Colab's native utility for interacting with your personal Google Drive.
        - Colab runtimes are temporary (ephemeral). 
        - If you downloaded 80GB of CT scans directly to the Colab instance, they would be deleted as soon as your session closed.
        - By mounting the drive (```drive.mount```), you route the tcia_utils download directly into your persistent Google Drive storage.
###    3. os & sys
        - Built-in Python libraries for interacting with the operating system.
        - The script uses o```s.makedirs(destination_path, exist_ok=True)``` to safely construct the folder tree (Lung_Nodule_Project/raw_data) inside your Google Drive
        - If the folders already exist, ```exist_ok=True``` prevents the script from crashing and throwing an error.
- *The Data Retrieval Execution*
    1. nbia.getSeries(...)
        - Before downloading anything, the script asks the TCIA database for a specific list of files.
        - The Parameters:
            - ```collection="LIDC-IDRI"```: Tells the API exactly which dataset to look at (The Lung Image Database Consortium).
            - ```modality="CT"```: 
                - This is a crucial filter. 
                - Medical datasets often contain mixed types of imaging (like X-Rays, MRIs, and CTs). 
                - This ensures you are only fetching 3D Computed Tomography scans, ignoring everything else.
                - It returns a lightweight list of metadata/identifiers (finding 1,018 series in your case).
    2. nbia.downloadSeries(...)
        - This takes the list of 1,018 identifiers you just generated and begins sequentially downloading and extracting the raw DICOM files directly into your Google Drive path.
        - If you look closely at the output log in your notebook, you'll see a lot of lines saying ```WARNING: ... already downloaded and unzipped.```
        - This means the tcia_utils library is smart! If your Colab session crashes or times out (which happens a lot with 80GB downloads), running this cell again won't duplicate your data; it just skips what you already have and picks up where it left off.

    - The ```series_data``` variable stores a list of unique identifiers (Series UIDs) for every single CT scan in that collection.
    - Two main purposes:
        1. **Verification:** It allows you to count *exactly how many scans you found* (which is why len(series_data) returns 1,018 in your output) before committing to a massive download.
        2. **Execution:** In the very next step, you feed this exact variable into nbia.downloadSeries(series_data=series_data, ...). You are essentially handing the downloader the shopping list and saying, "*Go fetch the actual files for every single ID on this list.*"


2. phase1preprocess.ipynb
- *Bridge between raw medical data and deep learning model.*
- *Raw CT scans (DICOM files) are bulky, inconsistently sized, and contain a lot of irrelevant information (like air, bones, and the scanner bed).*
- *This file takes the raw data downloaded in your previous step and:*
    1. **Standardizes it:** *Resamples* the 3D CT volumes to a uniform physical spacing.
    2. **Filters it:** Applies a "lung window" to isolate lung tissue and normalizes the pixel values.
    3. **Extracts:** Finds the specific lung nodules using the LIDC-IDRI annotations and crops out precise 64x64x64 3D patches around them.
    4. **Saves:** Converts these patches into PyTorch tensors (.pt files) for training and generates a master manifest.csv file that logs all the metadata (malignancy ratings, coordinates, etc.).

- Cell 1 & Setup: Environment Preparation
    - first command is ```!pip install -q pydicom SimpleITK pylidc torch h5py pandas matplotlib tqdm```
    - ```pydicom and SimpleITK:``` Used for reading and manipulating the raw medical imaging formats.
    - ```pylidc:``` A specialized library designed specifically for querying the LIDC-IDRI dataset annotations.
    - ```torch:``` The deep learning framework that will eventually consume your processed data.
    
- Cell 2 Importing libraries and colab utils
    - Mounting of google Drive to access the database

- Cell 3 Setting of dataset paths
    - ```raw_data_path = '/content/drive/MyDrive/Lung_Nodule_Project/raw_data/'```
    - ```processed_data_path = '/content/drive/MyDrive/Lung_Nodule_Project/processed_data/'```
    - ```manifest_path = '/content/drive/MyDrive/Lung_Nodule_Project/manifest.csv'```
    - making of colab raw data dir and output dir 
    - making of a patches directory

- Cell 4 Configuration of the ~/.pylidcrc to enable pylidc scan queries
    - first find home dir of current environemnt and sets /.pylidcrc as the target file path
    - 
