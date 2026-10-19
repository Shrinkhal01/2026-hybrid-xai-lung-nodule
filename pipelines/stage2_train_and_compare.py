
#!/usr/bin/env python3
"""
Stage 2: Model Training and Comparative Evaluation
for 2026-hybrid-xai-lung-nodule.

Compares:
    1. Random Forest Baseline
    2. 3D ResNet-18 Deep Learning Model

Evaluation is performed on the disjoint test split.

Outputs:
    - Comparative metrics
    - ROC comparison curve
    - Saved best 3D ResNet-18 model weights
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r3d_18
from tqdm import tqdm


# ==============================================================================
# 1. CONFIGURATION & REPRODUCIBILITY
# ==============================================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ==============================================================================
# 2. PATH CONFIGURATION
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "lidc_processed"
)

PATCHES_DIR = os.path.join(
    PROCESSED_DIR,
    "patches"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ==============================================================================
# 3. DEVICE SELECTION
# ==============================================================================

# CUDA -> MPS -> CPU

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

else:
    DEVICE = torch.device("cpu")


print(f"[Device] Using compute backend: {DEVICE}")


# ==============================================================================
# 4. TRAINING HYPERPARAMETERS
# ==============================================================================

BATCH_SIZE = 16

NUM_EPOCHS = 35

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4


# ==============================================================================
# 5. RANDOM FOREST FEATURE COLUMNS
# ==============================================================================

FEATURE_COLS = [
    "feat_mean_hu",
    "feat_std_hu",
    "feat_min_hu",
    "feat_max_hu",
    "feat_median_hu",
    "feat_q25_hu",
    "feat_q75_hu",
    "feat_iqr_hu",
    "feat_skewness",
    "feat_kurtosis",
    "feat_solid_fraction",
    "feat_hyperdense_fraction",
    "meta_mean_diameter",
    "meta_mean_calcification",
    "meta_mean_texture",
]


# ==============================================================================
# 6. PYTORCH 3D DATASET
# ==============================================================================

class Nodule3DDataset(Dataset):
    """
    Dataset for loading preprocessed 32x32x32 volumetric nodule patches.
    """

    def __init__(
        self,
        df,
        patches_dir,
        augment=False
    ):
        self.df = df.reset_index(drop=True)
        self.patches_dir = patches_dir
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        patch_path = os.path.join(
            self.patches_dir,
            row["patch_filename"]
        )

        if not os.path.exists(patch_path):
            raise FileNotFoundError(
                f"Patch file not found: {patch_path}"
            )

        cube = np.load(
            patch_path
        )

        # Expected shape:
        # (32, 32, 32)

        if cube.shape != (32, 32, 32):
            raise ValueError(
                f"Unexpected patch shape for "
                f"{patch_path}: {cube.shape}. "
                f"Expected (32, 32, 32)."
            )

        # ------------------------------------------------------------------
        # Data augmentation
        # ------------------------------------------------------------------

        if self.augment:

            # Depth flip
            if np.random.rand() > 0.5:
                cube = np.flip(
                    cube,
                    axis=0
                )

            # Height flip
            if np.random.rand() > 0.5:
                cube = np.flip(
                    cube,
                    axis=1
                )

            # Width flip
            if np.random.rand() > 0.5:
                cube = np.flip(
                    cube,
                    axis=2
                )

            # Random 90-degree in-plane rotation
            k = np.random.choice(
                [0, 1, 2, 3]
            )

            cube = np.rot90(
                cube,
                k,
                axes=(1, 2)
            )

        # ------------------------------------------------------------------
        # Ensure contiguous float32 NumPy array
        # ------------------------------------------------------------------

        cube = np.ascontiguousarray(
            cube,
            dtype=np.float32
        )

        # ------------------------------------------------------------------
        # Convert:
        #
        # (D, H, W)
        #
        # -> (1, D, H, W)
        # ------------------------------------------------------------------

        tensor = torch.from_numpy(
            cube
        ).unsqueeze(0)

        # Explicit float32 label
        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32
        )

        return tensor, label


# ==============================================================================
# 7. 3D RESNET-18 ARCHITECTURE
# ==============================================================================

def build_3d_resnet():

    """
    Constructs a 3D ResNet-18 adapted for:

        Input:
            1-channel CT volume

        Output:
            1 binary-classification logit
    """

    model = r3d_18(
        weights=None
    )

    # ------------------------------------------------------------------
    # Replace original 3-channel video input with 1-channel CT input
    # ------------------------------------------------------------------

    orig_stem = model.stem[0]

    model.stem[0] = nn.Conv3d(
        in_channels=1,
        out_channels=orig_stem.out_channels,
        kernel_size=orig_stem.kernel_size,
        stride=orig_stem.stride,
        padding=orig_stem.padding,
        bias=False,
    )

    # ------------------------------------------------------------------
    # Replace final classification layer
    # ------------------------------------------------------------------

    num_feats = model.fc.in_features

    model.fc = nn.Linear(
        num_feats,
        1
    )

    return model


# ==============================================================================
# 8. CLINICAL EVALUATION METRICS
# ==============================================================================

def compute_clinical_metrics(
    y_true,
    y_probs,
    threshold=0.5
):

    y_preds = (
        y_probs >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        y_preds,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(
        y_true,
        y_preds
    )

    if len(np.unique(y_true)) > 1:

        roc_auc = roc_auc_score(
            y_true,
            y_probs
        )

    else:

        roc_auc = 0.5

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    precision = precision_score(
        y_true,
        y_preds,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_preds,
        zero_division=0
    )

    return {
        "Accuracy": accuracy,
        "ROC-AUC": roc_auc,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1-Score": f1,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


# ==============================================================================
# 9. MAIN TRAINING PIPELINE
# ==============================================================================

def main():

    print("=" * 70)
    print("STAGE 2: TRAINING & COMPARATIVE EVALUATION")
    print("=" * 70)

    # ==================================================================
    # LOAD DATA SPLITS
    # ==================================================================

    train_csv = os.path.join(
        PROCESSED_DIR,
        "train.csv"
    )

    val_csv = os.path.join(
        PROCESSED_DIR,
        "val.csv"
    )

    test_csv = os.path.join(
        PROCESSED_DIR,
        "test.csv"
    )

    if not os.path.exists(train_csv):
        raise FileNotFoundError(
            f"Training split not found: {train_csv}"
        )

    if not os.path.exists(val_csv):
        raise FileNotFoundError(
            f"Validation split not found: {val_csv}"
        )

    if not os.path.exists(test_csv):
        raise FileNotFoundError(
            f"Test split not found: {test_csv}"
        )

    df_train = pd.read_csv(
        train_csv
    )

    df_val = pd.read_csv(
        val_csv
    )

    df_test = pd.read_csv(
        test_csv
    )

    print(
        f"Data Splits: "
        f"Train={len(df_train)}, "
        f"Val={len(df_val)}, "
        f"Test={len(df_test)}"
    )

    # ==================================================================
    # CLASS DISTRIBUTION
    # ==================================================================

    n_benign = int(
        (df_train["label"] == 0).sum()
    )

    n_malignant = int(
        (df_train["label"] == 1).sum()
    )

    pos_weight_val = (
        float(n_benign)
        / float(max(1, n_malignant))
    )

    print(
        f"Train Class Distribution: "
        f"Benign={n_benign}, "
        f"Malignant={n_malignant} "
        f"(pos_weight={pos_weight_val:.2f})\n"
    )

    # ==================================================================
    # EXPERIMENT 1: RANDOM FOREST
    # ==================================================================

    print("-" * 70)
    print("1. TRAINING RANDOM FOREST BASELINE")
    print("-" * 70)

    # Check required feature columns

    missing_features = [
        col
        for col in FEATURE_COLS
        if col not in df_train.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing Random Forest feature columns: "
            + ", ".join(missing_features)
        )

    X_train = df_train[
        FEATURE_COLS
    ].values

    y_train = df_train[
        "label"
    ].values

    X_test = df_test[
        FEATURE_COLS
    ].values

    y_test = df_test[
        "label"
    ].values

    # Random Forest

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=RANDOM_SEED,
        class_weight="balanced"
    )

    rf.fit(
        X_train,
        y_train
    )

    rf_test_probs = rf.predict_proba(
        X_test
    )[:, 1]

    rf_metrics = compute_clinical_metrics(
        y_test,
        rf_test_probs
    )

    print(
        "[Done] Random Forest Test Metrics:"
    )

    for k, v in rf_metrics.items():

        if isinstance(v, float):

            print(
                f"  {k:22s}: {v:.4f}"
            )

        else:

            print(
                f"  {k:22s}: {v}"
            )

    # ==================================================================
    # EXPERIMENT 2: 3D RESNET-18
    # ==================================================================

    print(
        "\n"
        + "-"
        * 70
    )

    print(
        "2. TRAINING 3D RESNET-18 "
        "ON 32x32x32 VOLUMETRIC PATCHES"
    )

    print("-" * 70)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    train_dataset = Nodule3DDataset(
        df_train,
        PATCHES_DIR,
        augment=True
    )

    val_dataset = Nodule3DDataset(
        df_val,
        PATCHES_DIR,
        augment=False
    )

    test_dataset = Nodule3DDataset(
        df_test,
        PATCHES_DIR,
        augment=False
    )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model = build_3d_resnet().to(
        DEVICE
    )

    # ------------------------------------------------------------------
    # IMPORTANT MPS FIX
    #
    # Explicitly force pos_weight to float32.
    #
    # Without dtype=torch.float32, NumPy/Python numerical values can
    # cause PyTorch to create a float64 tensor, which MPS does not
    # support for this operation.
    # ------------------------------------------------------------------

    pos_weight = torch.tensor(
        [pos_weight_val],
        dtype=torch.float32,
        device=DEVICE
    )

    print(
        f"[Loss] pos_weight={pos_weight.item():.4f} "
        f"| dtype={pos_weight.dtype} "
        f"| device={pos_weight.device}"
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ------------------------------------------------------------------
    # Learning-rate scheduler
    # ------------------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5
    )

    # ------------------------------------------------------------------
    # Best model tracking
    # ------------------------------------------------------------------

    best_val_auc = 0.0

    best_weights_path = os.path.join(
        RESULTS_DIR,
        "best_3d_resnet18.pth"
    )

    # ==================================================================
    # TRAINING LOOP
    # ==================================================================

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        # --------------------------------------------------------------
        # TRAIN
        # --------------------------------------------------------------

        model.train()

        running_loss = 0.0

        for inputs, labels in train_loader:

            inputs = inputs.to(
                DEVICE,
                dtype=torch.float32
            )

            labels = labels.to(
                DEVICE,
                dtype=torch.float32
            ).unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(
                inputs
            )

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * inputs.size(0)
            )

        train_loss = (
            running_loss
            / len(train_dataset)
        )

        # --------------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------------

        model.eval()

        val_loss = 0.0

        val_probs_list = []

        val_labels_list = []

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(
                    DEVICE,
                    dtype=torch.float32
                )

                labels = labels.to(
                    DEVICE,
                    dtype=torch.float32
                ).unsqueeze(1)

                outputs = model(
                    inputs
                )

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += (
                    loss.item()
                    * inputs.size(0)
                )

                probs = (
                    torch.sigmoid(
                        outputs
                    )
                    .cpu()
                    .numpy()
                    .ravel()
                )

                val_probs_list.extend(
                    probs
                )

                val_labels_list.extend(
                    labels
                    .cpu()
                    .numpy()
                    .ravel()
                )

        val_loss /= len(
            val_dataset
        )

        if len(
            np.unique(
                val_labels_list
            )
        ) > 1:

            val_auc = roc_auc_score(
                val_labels_list,
                val_probs_list
            )

        else:

            val_auc = 0.5

        # --------------------------------------------------------------
        # Scheduler
        # --------------------------------------------------------------

        scheduler.step(
            val_auc
        )

        # --------------------------------------------------------------
        # Save best model
        # --------------------------------------------------------------

        if val_auc > best_val_auc:

            best_val_auc = val_auc

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_auc": val_auc,
                },
                best_weights_path
            )

            save_flag = " [*SAVED BEST]"

        else:

            save_flag = ""

        # --------------------------------------------------------------
        # Logging
        # --------------------------------------------------------------

        if (
            epoch % 5 == 0
            or epoch == NUM_EPOCHS
            or save_flag
        ):

            current_lr = optimizer.param_groups[0][
                "lr"
            ]

            print(
                f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] "
                f"| Train Loss: {train_loss:.4f} "
                f"| Val Loss: {val_loss:.4f} "
                f"| Val ROC-AUC: {val_auc:.4f} "
                f"| LR: {current_lr:.2e}"
                f"{save_flag}"
            )

    # ==================================================================
    # TEST SET EVALUATION
    # ==================================================================

    print(
        "\n"
        + "-"
        * 70
    )

    print(
        "EVALUATING BEST 3D RESNET-18 "
        f"ON HELD-OUT TEST SET "
        f"({best_weights_path})"
    )

    print("-" * 70)

    if not os.path.exists(
        best_weights_path
    ):

        raise FileNotFoundError(
            "Best ResNet checkpoint was not created."
        )

    checkpoint = torch.load(
        best_weights_path,
<<<<<<< HEAD
        map_location=DEVICE
=======
        map_location=DEVICE,
        weights_only=False
>>>>>>> c44f2c3910463f7bfca85a3f61b9428fa41af64c
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    resnet_test_probs = []

    resnet_test_targets = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(
                DEVICE,
                dtype=torch.float32
            )

            outputs = model(
                inputs
            )

            probs = (
                torch.sigmoid(
                    outputs
                )
                .cpu()
                .numpy()
                .ravel()
            )

            resnet_test_probs.extend(
                probs
            )

            resnet_test_targets.extend(
                labels
                .numpy()
                .ravel()
            )

    resnet_test_probs = np.array(
        resnet_test_probs
    )

    resnet_test_targets = np.array(
        resnet_test_targets
    )

    resnet_metrics = compute_clinical_metrics(
        resnet_test_targets,
        resnet_test_probs
    )

    print(
        "[Done] 3D ResNet-18 Test Metrics:"
    )

    for k, v in resnet_metrics.items():

        if isinstance(v, float):

            print(
                f"  {k:22s}: {v:.4f}"
            )

        else:

            print(
                f"  {k:22s}: {v}"
            )

    # ==================================================================
    # EXPORT COMPARISON TABLE
    # ==================================================================

    comparison_df = pd.DataFrame(
        [
            {
                "Model":
                    "Random Forest Baseline",
                **rf_metrics
            },
            {
                "Model":
                    "3D ResNet-18 (Deep Learning)",
                **resnet_metrics
            },
        ]
    )

    comparison_csv = os.path.join(
        RESULTS_DIR,
        "model_comparison_metrics.csv"
    )

    comparison_df.to_csv(
        comparison_csv,
        index=False
    )

    # ==================================================================
    # PRINT HEAD-TO-HEAD COMPARISON
    # ==================================================================

    print(
        "\n"
        + "="
        * 70
    )

    print(
        "HEAD-TO-HEAD PERFORMANCE COMPARISON"
    )

    print(
        "="
        * 70
    )

    print(
        comparison_df[
            [
                "Model",
                "Accuracy",
                "ROC-AUC",
                "Sensitivity (Recall)",
                "Specificity",
                "F1-Score",
            ]
        ].to_string(
            index=False
        )
    )

    # ==================================================================
    # ROC CURVES
    # ==================================================================

    fpr_rf, tpr_rf, _ = roc_curve(
        y_test,
        rf_test_probs
    )

    fpr_resnet, tpr_resnet, _ = roc_curve(
        resnet_test_targets,
        resnet_test_probs
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr_rf,
        tpr_rf,
        label=(
            f"Random Forest "
            f"(AUC = {rf_metrics['ROC-AUC']:.3f})"
        ),
        linewidth=2
    )

    plt.plot(
        fpr_resnet,
        tpr_resnet,
        label=(
            f"3D ResNet-18 "
            f"(AUC = {resnet_metrics['ROC-AUC']:.3f})"
        ),
        linewidth=2
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        alpha=0.6,
        label="Random Guess"
    )

    plt.xlabel(
        "False Positive Rate (1 - Specificity)",
        fontsize=11
    )

    plt.ylabel(
        "True Positive Rate (Sensitivity / Recall)",
        fontsize=11
    )

    plt.title(
        "ROC Curve: Classical ML vs. 3D Deep Learning",
        fontsize=12
    )

    plt.legend(
        loc="lower right",
        fontsize=10
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    roc_plot_path = os.path.join(
        RESULTS_DIR,
        "roc_comparison_curve.png"
    )

    plt.savefig(
        roc_plot_path,
        dpi=300
    )

    plt.close()

    # ==================================================================
    # FINAL OUTPUT
    # ==================================================================

    print(
        f"\n[Saved] Metrics Table : "
        f"{comparison_csv}"
    )

    print(
        f"[Saved] ROC Curve Plot: "
        f"{roc_plot_path}"
    )

    print(
        f"[Saved] Best Model    : "
        f"{best_weights_path}"
    )

    print(
        "="
        * 70
    )


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
