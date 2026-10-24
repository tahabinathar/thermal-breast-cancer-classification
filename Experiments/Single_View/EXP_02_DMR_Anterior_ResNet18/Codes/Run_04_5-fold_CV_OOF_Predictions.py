import os
import csv
import json
import gc
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from preprocessing import (
    calculate_temperature_range,
    get_image_paths,
    preprocess_image,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = (
    r"D:\HITEC\STUDENT\NEW PROJECT"
    r"\Breast_Thermography\Dataset\DMR_IR_Anterior_View_ROI"
)

RESULTS_DIR = (
    r"D:\HITEC\STUDENT\NEW PROJECT"
    r"\Breast_Thermography\Experiments"
    r"\EXP_02_DMR_Anterior_ResNet18\Results"
)

EXPERIMENT_NAME = "EXP02_DMR_Anterior_ResNet18"

RUN_NAME = "Run_04_5-fold_CV_OOF_Predictions"

RUN_RESULTS_DIR = os.path.join(
    RESULTS_DIR,
    RUN_NAME,
    "results"
)

os.makedirs(
    RUN_RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
FIXED_EPOCH = 32
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3
N_FOLDS = 5
NUM_WORKERS = 0

CLASSES = {
    "Healthy": 0,
    "Sick": 1,
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("\nExperiment:", EXPERIMENT_NAME)
print("Run:", RUN_NAME)
print("Seed:", SEED)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Fixed epoch:", FIXED_EPOCH)
print("Learning rate:", LEARNING_RATE)
print("Dropout rate:", DROPOUT_RATE)
print("Number of folds:", N_FOLDS)
print("Training duration: Fixed")
print("Output: OOF predictions")
print("Threshold: NOT APPLIED")
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

print("Test set: NOT USED")


# ============================================================
# COLLECT DATASET PATHS FOR CROSS-VALIDATION
# ============================================================

all_paths = []
all_labels = []

for class_name, class_label in CLASSES.items():

    train_paths = get_image_paths(
        DATA_ROOT,
        "Train",
        class_name
    )

    validation_paths = get_image_paths(
        DATA_ROOT,
        "Validation",
        class_name
    )

    print(
        f"Train - {class_name}: "
        f"{len(train_paths)} images"
    )

    print(
        f"Validation - {class_name}: "
        f"{len(validation_paths)} images"
    )

    class_paths = train_paths + validation_paths

    all_paths.extend(class_paths)

    all_labels.extend(
        [class_label] * len(class_paths)
    )


all_paths = np.asarray(
    all_paths
)

all_labels = np.asarray(
    all_labels,
    dtype=np.int64
)

print("\nPooled dataset for cross-validation:")
print("Total images:", len(all_paths))
print("Healthy:", np.sum(all_labels == 0))
print("Sick:", np.sum(all_labels == 1))

print("\nTest set: NOT LOADED OR USED")


# ============================================================
# DATA AUGMENTATION
# ============================================================

def create_training_transform():

    return transforms.Compose(
        [
            transforms.ToPILImage(),

            transforms.RandomRotation(
                18
            ),

            transforms.RandomAffine(
                degrees=0,
                scale=(0.95, 1.05)
            ),

            transforms.ToTensor(),

            transforms.Lambda(
                lambda x: x.repeat(3, 1, 1)
            ),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            ),
        ]
    )


def create_validation_transform():

    return transforms.Compose(
        [
            transforms.ToPILImage(),

            transforms.ToTensor(),

            transforms.Lambda(
                lambda x: x.repeat(3, 1, 1)
            ),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            ),
        ]
    )


# ============================================================
# DATASET
# ============================================================

class ThermalDataset(Dataset):

    def __init__(
        self,
        paths,
        labels,
        global_min,
        global_max,
        transform=None
    ):

        self.paths = paths
        self.labels = labels
        self.global_min = global_min
        self.global_max = global_max
        self.transform = transform

    def __len__(self):

        return len(self.paths)

    def __getitem__(
        self,
        index
    ):

        image = preprocess_image(
            self.paths[index],
            self.global_min,
            self.global_max,
            IMAGE_SIZE
        )

        if self.transform is not None:

            image = self.transform(
                image
            )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return image, label


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model():

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    # Freeze entire pretrained backbone

    for parameter in model.parameters():

        parameter.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(
            512,
            64
        ),

        nn.ReLU(),

        nn.Dropout(
            DROPOUT_RATE
        ),

        nn.Linear(
            64,
            1
        )
    )

    return model.to(
        DEVICE
    )


# ============================================================
# FROZEN BACKBONE MODE
# ============================================================

def set_frozen_backbone_mode(
    model
):

    model.train()

    for name, module in model.named_children():

        if name != "fc":

            module.eval()

    model.fc.train()


# ============================================================
# OOF STORAGE
# ============================================================

oof_probabilities = np.full(
    len(all_paths),
    np.nan,
    dtype=np.float32
)

oof_fold = np.full(
    len(all_paths),
    -1,
    dtype=np.int32
)

fold_summaries = []


# ============================================================
# STRATIFIED 5-FOLD CROSS-VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)


for fold_number, (
    train_indices,
    val_indices
) in enumerate(
    skf.split(
        all_paths,
        all_labels
    ),
    start=1
):

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"FOLD {fold_number}/{N_FOLDS}"
    )

    print(
        f"{'=' * 60}"
    )


    # --------------------------------------------------------
    # Fold paths and labels
    # --------------------------------------------------------

    fold_train_paths = all_paths[
        train_indices
    ]

    fold_val_paths = all_paths[
        val_indices
    ]

    y_train = all_labels[
        train_indices
    ]

    y_val = all_labels[
        val_indices
    ]


    print(
        "\nTraining images:",
        len(fold_train_paths)
    )

    print(
        "Validation images:",
        len(fold_val_paths)
    )

    print(
        "Training Healthy:",
        np.sum(y_train == 0)
    )

    print(
        "Training Sick:",
        np.sum(y_train == 1)
    )

    print(
        "Validation Healthy:",
        np.sum(y_val == 0)
    )

    print(
        "Validation Sick:",
        np.sum(y_val == 1)
    )


    # --------------------------------------------------------
    # Fold-specific temperature range
    # --------------------------------------------------------

    (
        fold_global_min,
        fold_global_max
    ) = calculate_temperature_range(
        fold_train_paths
    )

    print(
        "\nFold temperature range:"
    )

    print(
        "Min:",
        fold_global_min
    )

    print(
        "Max:",
        fold_global_max
    )


    # --------------------------------------------------------
    # Fold-specific transforms
    # --------------------------------------------------------

    train_transform = create_training_transform()
    val_transform = create_validation_transform()


    # --------------------------------------------------------
    # Fold datasets
    # --------------------------------------------------------

    train_dataset = ThermalDataset(
        fold_train_paths,
        y_train,
        fold_global_min,
        fold_global_max,
        train_transform
    )

    val_dataset = ThermalDataset(
        fold_val_paths,
        y_val,
        fold_global_min,
        fold_global_max,
        val_transform
    )


    # --------------------------------------------------------
    # Fold data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )


    # --------------------------------------------------------
    # Fold-specific class weights
    # --------------------------------------------------------

    n_total = len(
        y_train
    )

    n_healthy = np.sum(
        y_train == 0
    )

    n_sick = np.sum(
        y_train == 1
    )

    healthy_weight = (
        n_total /
        (2 * n_healthy)
    )

    sick_weight = (
        n_total /
        (2 * n_sick)
    )

    class_weights = {
        0: float(
            healthy_weight
        ),
        1: float(
            sick_weight
        )
    }

    print(
        "\nClass weights:"
    )

    print(
        class_weights
    )


    # --------------------------------------------------------
    # Reproducibility for this fold
    # --------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            SEED
        )

        torch.cuda.manual_seed_all(
            SEED
        )


    # --------------------------------------------------------
    # Build fresh model
    # --------------------------------------------------------

    model = build_model()

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "\nTrainable parameters:",
        trainable_parameters
    )


    # --------------------------------------------------------
    # Optimizer and loss
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.BCEWithLogitsLoss(
        reduction="none"
    )


    # --------------------------------------------------------
    # Fixed-duration training
    # --------------------------------------------------------

    print(
        f"\nTraining for exactly "
        f"{FIXED_EPOCH} epochs..."
    )

    training_losses = []

    for epoch in range(
        FIXED_EPOCH
    ):

        set_frozen_backbone_mode(
            model
        )

        running_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad()

            logits = model(
                images
            ).reshape(-1)

            losses = criterion(
                logits,
                labels
            )

            sample_weights = torch.where(
                labels == 0,
                torch.tensor(
                    class_weights[0],
                    device=DEVICE
                ),
                torch.tensor(
                    class_weights[1],
                    device=DEVICE
                )
            )

            weighted_loss = (
                losses *
                sample_weights
            ).mean()

            weighted_loss.backward()

            optimizer.step()

            batch_size = (
                labels.size(0)
            )

            running_loss += (
                weighted_loss.item() *
                batch_size
            )

            total_samples += (
                batch_size
            )

        epoch_loss = (
            running_loss /
            total_samples
        )

        training_losses.append(
            float(epoch_loss)
        )

        print(
            f"Epoch "
            f"{epoch + 1:3d}/{FIXED_EPOCH}"
            f" - loss: "
            f"{epoch_loss:.6f}"
        )


    # --------------------------------------------------------
    # Generate OOF predictions
    # --------------------------------------------------------

    print(
        "\nGenerating OOF predictions..."
    )

    model.eval()

    fold_predictions = []

    with torch.no_grad():

        for images, _ in val_loader:

            images = images.to(
                DEVICE
            )

            logits = model(
                images
            ).reshape(-1)

            probabilities = torch.sigmoid(
                logits
            )

            fold_predictions.extend(
                probabilities.cpu().numpy()
            )

    fold_predictions = np.asarray(
        fold_predictions,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Store OOF predictions
    # --------------------------------------------------------

    oof_probabilities[
        val_indices
    ] = fold_predictions

    oof_fold[
        val_indices
    ] = fold_number


    # --------------------------------------------------------
    # Fold OOF metrics
    # --------------------------------------------------------

    fold_auc = roc_auc_score(
        y_val,
        fold_predictions
    )

    fold_pr_auc = average_precision_score(
        y_val,
        fold_predictions
    )

    print(
        "\nFold OOF ROC-AUC:",
        fold_auc
    )

    print(
        "Fold OOF PR-AUC:",
        fold_pr_auc
    )


    # --------------------------------------------------------
    # Save fold predictions
    # --------------------------------------------------------

    fold_prediction_path = os.path.join(
        RUN_RESULTS_DIR,
        f"fold_{fold_number}_predictions.csv"
    )

    with open(
        fold_prediction_path,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            [
                "index",
                "filepath",
                "true_label",
                "oof_probability",
            ]
        )

        for local_index, global_index in enumerate(
            val_indices
        ):

            writer.writerow(
                [
                    int(
                        global_index
                    ),
                    all_paths[
                        global_index
                    ],
                    int(
                        y_val[
                            local_index
                        ]
                    ),
                    float(
                        fold_predictions[
                            local_index
                        ]
                    ),
                ]
            )


    # --------------------------------------------------------
    # Save fold temperature range
    # --------------------------------------------------------

    temperature_range_path = os.path.join(
        RUN_RESULTS_DIR,
        f"fold_{fold_number}_temperature_range.json"
    )

    temperature_range_summary = {
        "temperature_min":
            float(
                fold_global_min
            ),

        "temperature_max":
            float(
                fold_global_max
            ),

        "computed_from":
            "fold_training_subset_only",

        "validation_used_for_range":
            False,

        "test_used_for_range":
            False,
    }

    with open(
        temperature_range_path,
        "w"
    ) as json_file:

        json.dump(
            temperature_range_summary,
            json_file,
            indent=4
        )


    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = os.path.join(
        RUN_RESULTS_DIR,
        f"fold_{fold_number}_training_history.json"
    )

    history_dict = {
        "train_loss":
            training_losses
    }

    with open(
        history_path,
        "w"
    ) as json_file:

        json.dump(
            history_dict,
            json_file,
            indent=4
        )


    # --------------------------------------------------------
    # Save fold summary
    # --------------------------------------------------------

    fold_summary = {

        "fold":
            fold_number,

        "epochs":
            int(
                FIXED_EPOCH
            ),

        "learning_rate":
            LEARNING_RATE,

        "dropout_rate":
            DROPOUT_RATE,

        "trainable_parameters":
            int(
                trainable_parameters
            ),

        "n_train":
            int(
                len(train_indices)
            ),

        "n_validation":
            int(
                len(val_indices)
            ),

        "train_healthy":
            int(
                np.sum(
                    y_train == 0
                )
            ),

        "train_sick":
            int(
                np.sum(
                    y_train == 1
                )
            ),

        "validation_healthy":
            int(
                np.sum(
                    y_val == 0
                )
            ),

        "validation_sick":
            int(
                np.sum(
                    y_val == 1
                )
            ),

        "class_weights":
            class_weights,

        "temperature_min":
            float(
                fold_global_min
            ),

        "temperature_max":
            float(
                fold_global_max
            ),

        "roc_auc":
            float(
                fold_auc
            ),

        "pr_auc":
            float(
                fold_pr_auc
            ),
    }

    fold_summaries.append(
        fold_summary
    )


    # --------------------------------------------------------
    # Save fold summary JSON
    # --------------------------------------------------------

    fold_summary_path = os.path.join(
        RUN_RESULTS_DIR,
        f"fold_{fold_number}_summary.json"
    )

    with open(
        fold_summary_path,
        "w"
    ) as json_file:

        json.dump(
            fold_summary,
            json_file,
            indent=4
        )


    # --------------------------------------------------------
    # Clean up fold
    # --------------------------------------------------------

    del model
    del train_dataset
    del val_dataset
    del train_loader
    del val_loader
    del y_train
    del y_val
    del fold_predictions

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# VERIFY OOF COVERAGE
# ============================================================

print(
    "\nOOF COVERAGE CHECK"
)

missing_predictions = np.isnan(
    oof_probabilities
)

missing_folds = (
    oof_fold == -1
)

print(
    "Total development images:",
    len(all_paths)
)

print(
    "OOF predictions generated:",
    np.sum(
        ~missing_predictions
    )
)

print(
    "Missing OOF predictions:",
    np.sum(
        missing_predictions
    )
)

print(
    "Missing fold assignments:",
    np.sum(
        missing_folds
    )
)

if np.any(
    missing_predictions
):

    raise RuntimeError(
        "Some development images do not have OOF predictions."
    )

if np.any(
    missing_folds
):

    raise RuntimeError(
        "Some development images do not have fold assignments."
    )


# ============================================================
# SAVE COMBINED OOF PREDICTIONS
# ============================================================

oof_prediction_path = os.path.join(
    RUN_RESULTS_DIR,
    "oof_predictions.csv"
)

with open(
    oof_prediction_path,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(
        csv_file
    )

    writer.writerow(
        [
            "index",
            "filepath",
            "true_label",
            "true_class",
            "oof_fold",
            "oof_probability",
        ]
    )

    for index in range(
        len(all_paths)
    ):

        true_label = int(
            all_labels[
                index
            ]
        )

        true_class = (
            "Healthy"
            if true_label == 0
            else "Sick"
        )

        writer.writerow(
            [
                int(index),

                all_paths[
                    index
                ],

                true_label,

                true_class,

                int(
                    oof_fold[
                        index
                    ]
                ),

                float(
                    oof_probabilities[
                        index
                    ]
                ),
            ]
        )


# ============================================================
# OVERALL OOF METRICS
# ============================================================

overall_oof_auc = roc_auc_score(
    all_labels,
    oof_probabilities
)

overall_oof_pr_auc = average_precision_score(
    all_labels,
    oof_probabilities
)

print(
    "\nOverall OOF ROC-AUC:",
    overall_oof_auc
)

print(
    "Overall OOF PR-AUC:",
    overall_oof_pr_auc
)


# ============================================================
# SAVE FOLD SUMMARY CSV
# ============================================================

fold_summary_csv_path = os.path.join(
    RUN_RESULTS_DIR,
    "fold_summary.csv"
)

with open(
    fold_summary_csv_path,
    "w",
    newline=""
) as csv_file:

    fieldnames = [
        "fold",
        "epochs",
        "learning_rate",
        "dropout_rate",
        "trainable_parameters",
        "n_train",
        "n_validation",
        "train_healthy",
        "train_sick",
        "validation_healthy",
        "validation_sick",
        "healthy_class_weight",
        "sick_class_weight",
        "temperature_min",
        "temperature_max",
        "roc_auc",
        "pr_auc",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for summary in fold_summaries:

        writer.writerow(
            {
                "fold":
                    summary["fold"],

                "epochs":
                    summary["epochs"],

                "learning_rate":
                    summary["learning_rate"],

                "dropout_rate":
                    summary["dropout_rate"],

                "trainable_parameters":
                    summary["trainable_parameters"],

                "n_train":
                    summary["n_train"],

                "n_validation":
                    summary["n_validation"],

                "train_healthy":
                    summary["train_healthy"],

                "train_sick":
                    summary["train_sick"],

                "validation_healthy":
                    summary["validation_healthy"],

                "validation_sick":
                    summary["validation_sick"],

                "healthy_class_weight":
                    summary["class_weights"][0],

                "sick_class_weight":
                    summary["class_weights"][1],

                "temperature_min":
                    summary["temperature_min"],

                "temperature_max":
                    summary["temperature_max"],

                "roc_auc":
                    summary["roc_auc"],

                "pr_auc":
                    summary["pr_auc"],
            }
        )


# ============================================================
# SAVE RUN SUMMARY
# ============================================================

run_summary = {

    "experiment":
        EXPERIMENT_NAME,

    "run":
        RUN_NAME,

    "seed":
        SEED,

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        FIXED_EPOCH,

    "learning_rate":
        LEARNING_RATE,

    "dropout_rate":
        DROPOUT_RATE,

    "n_folds":
        N_FOLDS,

    "optimizer":
        "Adam",

    "model":
        "ResNet18",

    "pretrained":
        "ImageNet1K_V1",

    "backbone_frozen":
        True,

    "classifier":
        "Linear(512,64) -> ReLU -> Dropout(0.3) -> Linear(64,1)",

    "trainable_parameters":
        32897,

    "development_images":
        int(
            len(all_paths)
        ),

    "healthy_images":
        int(
            np.sum(
                all_labels == 0
            )
        ),

    "sick_images":
        int(
            np.sum(
                all_labels == 1
            )
        ),

    "temperature_normalization":
        "Fold-specific training-only temperature range",

    "validation_used_for_temperature_range":
        False,

    "test_used_for_temperature_range":
        False,

    "overall_oof_roc_auc":
        float(
            overall_oof_auc
        ),

    "overall_oof_pr_auc":
        float(
            overall_oof_pr_auc
        ),

    "threshold_applied":
        False,

    "threshold_selection_performed":
        False,

    "test_loaded":
        False,

    "test_evaluated":
        False,

    "fold_results":
        fold_summaries,
}


run_summary_path = os.path.join(
    RUN_RESULTS_DIR,
    "run_summary.json"
)

with open(
    run_summary_path,
    "w"
) as json_file:

    json.dump(
        run_summary,
        json_file,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\nRUN 04 COMPLETE"
)

print(
    "Development images:",
    len(all_paths)
)

print(
    "OOF predictions:",
    np.sum(
        ~missing_predictions
    )
)

print(
    "Folds:",
    N_FOLDS
)

print(
    "Fixed epoch:",
    FIXED_EPOCH
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Dropout:",
    DROPOUT_RATE
)

print(
    "Overall OOF ROC-AUC:",
    overall_oof_auc
)

print(
    "Overall OOF PR-AUC:",
    overall_oof_pr_auc
)

print(
    "\nOOF predictions saved to:"
)

print(
    oof_prediction_path
)

print(
    "\nTest set: NOT USED"
)

print(
    "Threshold selection: NOT PERFORMED"
)