import os
import csv
import json
import gc

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models
from torchvision.transforms import (
    ToPILImage,
    ToTensor,
    Normalize,
    RandomRotation,
    RandomAffine,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)

from preprocessing import (
    calculate_temperature_range,
    get_image_paths,
    preprocess_image,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 10


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# PATHS
# ============================================================

DATA_ROOT = (
    r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography"
    r"\Dataset\DMR_IR_Anterior_View_ROI"
)

RESULTS_DIR = (
    r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography"
    r"\Experiments\EXP_02_DMR_Anterior_ResNet18\Results"
)

EXPERIMENT_NAME = "EXP02_DMR_Anterior_ResNet18"
RUN_NAME = "Run_03_Fixed_Epoch_Selection"

RUN02_DIR = os.path.join(
    RESULTS_DIR,
    "Run_02_5-fold_CV_Training_Dynamics",
)

RUN03_RESULTS_DIR = os.path.join(
    RESULTS_DIR,
    RUN_NAME,
    "results"
)

os.makedirs(
    RUN03_RESULTS_DIR,
    exist_ok=True,
)


# ============================================================
# RUN SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100

LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3

N_FOLDS = 5
THRESHOLD = 0.5

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
print("Maximum epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Dropout rate:", DROPOUT_RATE)
print("Number of folds:", N_FOLDS)
print("Threshold:", THRESHOLD)
print("Device:", DEVICE)
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
        class_name,
    )

    validation_paths = get_image_paths(
        DATA_ROOT,
        "Validation",
        class_name,
    )

    print(
        f"Train - {class_name}: "
        f"{len(train_paths)} images"
    )

    print(
        f"Validation - {class_name}: "
        f"{len(validation_paths)} images"
    )

    class_paths = (
        train_paths +
        validation_paths
    )

    all_paths.extend(class_paths)

    all_labels.extend(
        [class_label] * len(class_paths)
    )


all_paths = np.asarray(all_paths)

all_labels = np.asarray(
    all_labels,
    dtype=np.int32,
)


print("\nPooled dataset for cross-validation:")
print("Total images:", len(all_paths))
print(
    "Healthy:",
    np.sum(all_labels == CLASSES["Healthy"]),
)
print(
    "Sick:",
    np.sum(all_labels == CLASSES["Sick"]),
)

print("\nTest set: NOT LOADED OR USED")


# ============================================================
# LOAD RUN02 TRAINING HISTORIES
# ============================================================

def load_run02_histories():

    fold_histories = []

    for fold_number in range(
        1,
        N_FOLDS + 1,
    ):

        history_path = os.path.join(
            RUN02_DIR,
            f"fold_{fold_number}",
            "results",
            "training_history.json",
        )

        if not os.path.isfile(history_path):
            raise FileNotFoundError(
                f"\nExpected Run02 history file not found:\n"
                f"{history_path}\n\n"
                f"Check that RUN02_DIR points to the corrected "
                f"Run02 results directory."
            )

        with open(
            history_path,
            "r",
            encoding="utf-8",
        ) as f:
            history = json.load(f)

        if "val_auc" not in history:
            raise KeyError(
                f"'val_auc' not found in:\n"
                f"{history_path}"
            )

        val_auc = history["val_auc"]

        if len(val_auc) != EPOCHS:
            raise ValueError(
                f"Expected {EPOCHS} validation AUC values "
                f"in:\n{history_path}\n"
                f"Found {len(val_auc)}."
            )

        fold_histories.append(
            [
                float(value)
                for value in val_auc
            ]
        )

        print(
            f"Fold {fold_number}: "
            f"{len(val_auc)} epochs recorded"
        )

    return fold_histories


# ============================================================
# DERIVE CANDIDATE EPOCHS FROM RUN02
# ============================================================

print()
print("RUN02 TRAINING-DYNAMICS SUMMARY")

fold_histories = load_run02_histories()


epochs = np.arange(
    1,
    EPOCHS + 1,
)


# ------------------------------------------------------------
# Aggregate mean-AUC optimum
# ------------------------------------------------------------

mean_auc_by_epoch = np.mean(
    np.asarray(fold_histories),
    axis=0,
)

mean_auc_best_index = int(
    np.argmax(mean_auc_by_epoch)
)

mean_auc_best_epoch = int(
    epochs[mean_auc_best_index]
)

mean_auc_best_value = float(
    mean_auc_by_epoch[mean_auc_best_index]
)


# ------------------------------------------------------------
# Individual fold optima
# ------------------------------------------------------------

fold_best_epochs = {}
fold_best_auc = {}

for fold_number in range(
    1,
    N_FOLDS + 1,
):

    fold_auc = np.asarray(
        fold_histories[fold_number - 1]
    )

    best_index = int(
        np.argmax(fold_auc)
    )

    fold_best_epochs[fold_number] = int(
        epochs[best_index]
    )

    fold_best_auc[fold_number] = float(
        fold_auc[best_index]
    )


print()
print(
    "Aggregate mean-AUC optimum:",
    f"epoch {mean_auc_best_epoch}",
    f"(AUC = {mean_auc_best_value:.6f})",
)

print()
print("Individual fold optima:")

for fold_number in range(
    1,
    N_FOLDS + 1,
):

    print(
        f"  Fold {fold_number}: "
        f"epoch {fold_best_epochs[fold_number]} "
        f"(AUC = {fold_best_auc[fold_number]:.6f})"
    )


# ------------------------------------------------------------
# Candidate epochs
# ------------------------------------------------------------
#
# Candidates consist only of:
#
#   - individual best epoch from each fold
#   - aggregate mean-AUC best epoch
#   - full 100-epoch training duration
#
# No arbitrary intermediate epochs are introduced.
# ------------------------------------------------------------

candidate_epochs = []

candidate_epochs.extend(
    fold_best_epochs.values()
)

candidate_epochs.append(
    mean_auc_best_epoch
)

candidate_epochs.append(
    EPOCHS
)

candidate_epochs = sorted(
    set(
        epoch
        for epoch in candidate_epochs
        if 1 <= epoch <= EPOCHS
    )
)


print()
print("FINAL FIXED-EPOCH CANDIDATES")

for index, epoch in enumerate(
    candidate_epochs,
    start=1,
):

    print(
        f"  Candidate {index}: "
        f"Epoch {epoch}"
    )

print()
print(
    f"Total candidates: "
    f"{len(candidate_epochs)}"
)

print()
print("Candidate set is now FIXED.")
print(
    "Run 03 5-fold CV will evaluate only these epochs."
)


# ============================================================
# SAVE CANDIDATE INFORMATION
# ============================================================

candidate_info = {

    "experiment_name":
        EXPERIMENT_NAME,

    "run_name":
        RUN_NAME,

    "source_run":
        "Run_02_5-fold_CV_Training_Dynamics",

    "source_directory":
        RUN02_DIR,

    "aggregate_mean_auc_best_epoch":
        mean_auc_best_epoch,

    "aggregate_mean_auc_best_value":
        mean_auc_best_value,

    "fold_best_epochs": {
        str(fold): int(epoch)
        for fold, epoch
        in fold_best_epochs.items()
    },

    "fold_best_auc": {
        str(fold): float(value)
        for fold, value
        in fold_best_auc.items()
    },

    "full_training_epoch":
        EPOCHS,

    "candidate_epochs":
        candidate_epochs,

    "selection_basis": (
        "Candidates were derived from Run02 training "
        "dynamics using the individual fold optima, "
        "the aggregate mean-AUC optimum and the full "
        "100-epoch training duration. No arbitrary "
        "intermediate epochs were introduced."
    ),
}

candidate_info_path = os.path.join(
    RUN03_RESULTS_DIR,
    "candidate_epochs.json",
)

with open(
    candidate_info_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        candidate_info,
        f,
        indent=4,
    )


# ============================================================
# CREATE FIXED STRATIFIED CV SPLITS
# ============================================================
#
# The SAME five folds are used for every candidate epoch.
# This makes the candidate comparison fair.
# ============================================================

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED,
)

cv_splits = list(
    skf.split(
        all_paths,
        all_labels,
    )
)


# ============================================================
# PREPROCESSING
# ============================================================

def prepare_image(
    filepath,
    temperature_min,
    temperature_max,
    training,
):

    image = preprocess_image(
        filepath,
        temperature_min,
        temperature_max,
        IMAGE_SIZE,
    )

    if image.ndim == 3:
        image = image.squeeze()

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    image = ToPILImage()(image)

    if training:

        image = RandomRotation(
            degrees=18,
        )(image)

        image = RandomAffine(
            degrees=0,
            scale=(0.95, 1.05),
        )(image)

    image = ToTensor()(image)

    image = image.repeat(
        3,
        1,
        1,
    )

    image = Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    )(image)

    return image


# ============================================================
# MODEL
# ============================================================

def build_resnet18():

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )

    # Freeze the complete ResNet-18 backbone.
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(
            512,
            64,
        ),

        nn.ReLU(),

        nn.Dropout(
            DROPOUT_RATE
        ),

        nn.Linear(
            64,
            1,
        ),
    )

    return model


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    y_train,
):

    healthy_count = np.sum(
        y_train == CLASSES["Healthy"]
    )

    sick_count = np.sum(
        y_train == CLASSES["Sick"]
    )

    total = len(y_train)

    weight_healthy = (
        total /
        (
            2 * healthy_count
        )
    )

    weight_sick = (
        total /
        (
            2 * sick_count
        )
    )

    return {
        0: float(weight_healthy),
        1: float(weight_sick),
    }


# ============================================================
# DATA LOADER
# ============================================================

class ThermalDataset(
    torch.utils.data.Dataset
):

    def __init__(
        self,
        paths,
        labels,
        temperature_min,
        temperature_max,
        training,
    ):

        self.paths = paths
        self.labels = labels
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.training = training

    def __len__(self):

        return len(self.paths)

    def __getitem__(
        self,
        index,
    ):

        image = prepare_image(
            self.paths[index],
            self.temperature_min,
            self.temperature_max,
            self.training,
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32,
        )

        return image, label


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y_true,
    y_prob,
    threshold=THRESHOLD,
):

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

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

    return {

        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        ),

        "average_precision": float(
            average_precision_score(
                y_true,
                y_prob,
            )
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "sensitivity": float(
            sensitivity
        ),

        "specificity": float(
            specificity
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),

        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_one_fold(
    candidate_epoch,
    fold_number,
    train_indices,
    val_indices,
):

    print()
    print(
        f"Candidate epoch {candidate_epoch} "
        f"| Fold {fold_number}/{N_FOLDS}"
    )

    # --------------------------------------------------------
    # Reset random seeds before every model
    # --------------------------------------------------------

    torch.cuda.empty_cache()

    set_seed(SEED)

    # --------------------------------------------------------
    # Paths and labels
    # --------------------------------------------------------

    train_paths = all_paths[
        train_indices
    ]

    train_labels = all_labels[
        train_indices
    ]

    val_paths = all_paths[
        val_indices
    ]

    val_labels = all_labels[
        val_indices
    ]

    # --------------------------------------------------------
    # Fold-specific temperature range
    # --------------------------------------------------------
    #
    # Only fold training images are used to calculate the
    # normalization range.
    # --------------------------------------------------------

    print(
        "Calculating fold-specific "
        "temperature range..."
    )

    train_global_min, train_global_max = (
        calculate_temperature_range(
            train_paths
        )
    )

    print(
        f"Temperature range: "
        f"{train_global_min:.6f} "
        f"to "
        f"{train_global_max:.6f}"
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = calculate_class_weights(
        train_labels
    )

    print(
        f"Class weight - Healthy: "
        f"{class_weights[0]:.6f}"
    )

    print(
        f"Class weight - Sick: "
        f"{class_weights[1]:.6f}"
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = ThermalDataset(
        train_paths,
        train_labels,
        train_global_min,
        train_global_max,
        training=True,
    )

    val_dataset = ThermalDataset(
        val_paths,
        val_labels,
        train_global_min,
        train_global_max,
        training=False,
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_resnet18()

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Trainable parameters
    # --------------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Training loss
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss(
        reduction="none"
    )

    # --------------------------------------------------------
    # Fixed-epoch training
    # --------------------------------------------------------
    #
    # No early stopping.
    # No best-epoch checkpoint.
    #
    # The candidate itself is the fixed training epoch.
    # --------------------------------------------------------

    for epoch in range(
        1,
        candidate_epoch + 1,
    ):

        model.train()

        # Keep the frozen ResNet backbone in evaluation mode
        # so its BatchNorm running statistics do not change.
        for child in model.children():
            if child is not model.fc:
                child.eval()

        running_loss = 0.0
        sample_count = 0

        for images, labels in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad()

            logits = model(
                images
            ).squeeze(1)

            losses = criterion(
                logits,
                labels,
            )

            sample_weights = torch.where(
                labels == 0,
                torch.tensor(
                    class_weights[0],
                    device=DEVICE,
                ),
                torch.tensor(
                    class_weights[1],
                    device=DEVICE,
                ),
            )

            loss = (
                losses * sample_weights
            ).mean()

            loss.backward()

            optimizer.step()

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            sample_count += batch_size

        epoch_loss = (
            running_loss /
            sample_count
        )

        if (
            epoch == 1 or
            epoch == candidate_epoch
        ):
            print(
                f"  Epoch {epoch}/{candidate_epoch} "
                f"| Loss: {epoch_loss:.6f}"
            )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    model.eval()

    validation_probabilities = []

    with torch.no_grad():

        for images, _ in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            logits = model(
                images
            ).squeeze(1)

            probabilities = torch.sigmoid(
                logits
            )

            validation_probabilities.extend(
                probabilities.cpu().numpy()
            )

    val_prob = np.asarray(
        validation_probabilities,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        val_labels,
        val_prob,
        THRESHOLD,
    )

    print(
        f"Validation ROC-AUC: "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"Validation Accuracy: "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Validation Sensitivity: "
        f"{metrics['sensitivity']:.6f}"
    )

    print(
        f"Validation Specificity: "
        f"{metrics['specificity']:.6f}"
    )

    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    candidate_dir = os.path.join(
        RUN03_RESULTS_DIR,
        f"epoch_{candidate_epoch}",
    )

    fold_dir = os.path.join(
        candidate_dir,
        f"fold_{fold_number}",
    )

    os.makedirs(
        fold_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        fold_dir,
        "resnet18_frozen_fixed_epoch.pth",
    )

    torch.save(
        model.state_dict(),
        model_path,
    )

    # --------------------------------------------------------
    # Save validation predictions
    # --------------------------------------------------------

    predictions_path = os.path.join(
        fold_dir,
        "validation_predictions.csv",
    )

    with open(
        predictions_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "filepath",
                "true_label",
                "predicted_probability",
                "predicted_label",
            ]
        )

        for filepath, true_label, probability in zip(
            val_paths,
            val_labels,
            val_prob,
        ):

            predicted_label = int(
                probability >= THRESHOLD
            )

            writer.writerow(
                [
                    filepath,
                    int(true_label),
                    float(probability),
                    predicted_label,
                ]
            )

    # --------------------------------------------------------
    # Save fold result
    # --------------------------------------------------------

    fold_result = {

        "candidate_epoch":
            int(candidate_epoch),

        "fold":
            int(fold_number),

        "train_count":
            int(len(train_paths)),

        "validation_count":
            int(len(val_paths)),

        "train_healthy":
            int(np.sum(train_labels == 0)),

        "train_sick":
            int(np.sum(train_labels == 1)),

        "validation_healthy":
            int(np.sum(val_labels == 0)),

        "validation_sick":
            int(np.sum(val_labels == 1)),

        "temperature_range": {
            "global_min":
                float(train_global_min),

            "global_max":
                float(train_global_max),
        },

        "class_weights": {
            "Healthy":
                float(class_weights[0]),

            "Sick":
                float(class_weights[1]),
        },

        "learning_rate":
            float(LEARNING_RATE),

        "dropout_rate":
            float(DROPOUT_RATE),

        "batch_size":
            int(BATCH_SIZE),

        "fixed_epoch":
            int(candidate_epoch),

        "threshold":
            float(THRESHOLD),

        "seed":
            int(SEED),

        "metrics":
            metrics,
    }

    fold_result_path = os.path.join(
        fold_dir,
        "fold_result.json",
    )

    with open(
        fold_result_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            fold_result,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # Clean up
    # --------------------------------------------------------

    del model
    del train_loader
    del val_loader
    del train_dataset
    del val_dataset
    del val_prob

    torch.cuda.empty_cache()
    gc.collect()

    return fold_result


# ============================================================
# RUN 5-FOLD CV FOR EVERY CANDIDATE
# ============================================================

all_candidate_results = []

for candidate_index, candidate_epoch in enumerate(
    candidate_epochs,
    start=1,
):

    print()
    print()
    print(
        f"CANDIDATE {candidate_index}/"
        f"{len(candidate_epochs)}"
    )
    print(
        f"FIXED EPOCH = {candidate_epoch}"
    )

    fold_results = []

    for fold_number, (
        train_indices,
        val_indices,
    ) in enumerate(
        cv_splits,
        start=1,
    ):

        fold_result = train_one_fold(
            candidate_epoch,
            fold_number,
            train_indices,
            val_indices,
        )

        fold_results.append(
            fold_result
        )

    # ========================================================
    # AGGREGATE CANDIDATE RESULTS
    # ========================================================

    roc_auc_values = [
        result["metrics"]["roc_auc"]
        for result in fold_results
    ]

    accuracy_values = [
        result["metrics"]["accuracy"]
        for result in fold_results
    ]

    balanced_accuracy_values = [
        result["metrics"]["balanced_accuracy"]
        for result in fold_results
    ]

    sensitivity_values = [
        result["metrics"]["sensitivity"]
        for result in fold_results
    ]

    specificity_values = [
        result["metrics"]["specificity"]
        for result in fold_results
    ]

    precision_values = [
        result["metrics"]["precision"]
        for result in fold_results
    ]

    f1_values = [
        result["metrics"]["f1"]
        for result in fold_results
    ]

    mcc_values = [
        result["metrics"]["mcc"]
        for result in fold_results
    ]

    candidate_result = {

        "candidate_epoch":
            int(candidate_epoch),

        "mean_roc_auc":
            float(np.mean(roc_auc_values)),

        "std_roc_auc":
            float(np.std(roc_auc_values)),

        "mean_accuracy":
            float(np.mean(accuracy_values)),

        "std_accuracy":
            float(np.std(accuracy_values)),

        "mean_balanced_accuracy":
            float(
                np.mean(
                    balanced_accuracy_values
                )
            ),

        "std_balanced_accuracy":
            float(
                np.std(
                    balanced_accuracy_values
                )
            ),

        "mean_sensitivity":
            float(
                np.mean(
                    sensitivity_values
                )
            ),

        "std_sensitivity":
            float(
                np.std(
                    sensitivity_values
                )
            ),

        "mean_specificity":
            float(
                np.mean(
                    specificity_values
                )
            ),

        "std_specificity":
            float(
                np.std(
                    specificity_values
                )
            ),

        "mean_precision":
            float(
                np.mean(
                    precision_values
                )
            ),

        "std_precision":
            float(
                np.std(
                    precision_values
                )
            ),

        "mean_f1":
            float(
                np.mean(
                    f1_values
                )
            ),

        "std_f1":
            float(
                np.std(
                    f1_values
                )
            ),

        "mean_mcc":
            float(
                np.mean(
                    mcc_values
                )
            ),

        "std_mcc":
            float(
                np.std(
                    mcc_values
                )
            ),

        "fold_results":
            fold_results,
    }

    all_candidate_results.append(
        candidate_result
    )

    # --------------------------------------------------------
    # Print candidate summary
    # --------------------------------------------------------

    print()
    print(
        f"Candidate epoch {candidate_epoch} summary:"
    )

    print(
        f"  Mean ROC-AUC: "
        f"{candidate_result['mean_roc_auc']:.6f} "
        f"± "
        f"{candidate_result['std_roc_auc']:.6f}"
    )

    print(
        f"  Mean Accuracy: "
        f"{candidate_result['mean_accuracy']:.6f}"
    )

    print(
        f"  Mean Sensitivity: "
        f"{candidate_result['mean_sensitivity']:.6f}"
    )

    print(
        f"  Mean Specificity: "
        f"{candidate_result['mean_specificity']:.6f}"
    )

    print(
        f"  Mean F1: "
        f"{candidate_result['mean_f1']:.6f}"
    )


# ============================================================
# SAVE COMPLETE CANDIDATE COMPARISON
# ============================================================

comparison_json_path = os.path.join(
    RUN03_RESULTS_DIR,
    "candidate_epoch_comparison.json",
)

with open(
    comparison_json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        all_candidate_results,
        f,
        indent=4,
    )


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

comparison_csv_path = os.path.join(
    RUN03_RESULTS_DIR,
    "candidate_epoch_comparison.csv",
)

with open(
    comparison_csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "candidate_epoch",

            "mean_roc_auc",
            "std_roc_auc",

            "mean_accuracy",
            "std_accuracy",

            "mean_balanced_accuracy",
            "std_balanced_accuracy",

            "mean_sensitivity",
            "std_sensitivity",

            "mean_specificity",
            "std_specificity",

            "mean_precision",
            "std_precision",

            "mean_f1",
            "std_f1",

            "mean_mcc",
            "std_mcc",
        ]
    )

    for result in all_candidate_results:

        writer.writerow(
            [
                result["candidate_epoch"],

                result["mean_roc_auc"],
                result["std_roc_auc"],

                result["mean_accuracy"],
                result["std_accuracy"],

                result["mean_balanced_accuracy"],
                result["std_balanced_accuracy"],

                result["mean_sensitivity"],
                result["std_sensitivity"],

                result["mean_specificity"],
                result["std_specificity"],

                result["mean_precision"],
                result["std_precision"],

                result["mean_f1"],
                result["std_f1"],

                result["mean_mcc"],
                result["std_mcc"],
            ]
        )


# ============================================================
# SELECT FINAL FIXED EPOCH
# ============================================================
#
# Primary criterion:
# Highest mean 5-fold validation ROC-AUC.
#
# Tie-breaker:
# Lower epoch.
#
# The test set is not involved.
# ============================================================

selected_epoch_result = max(
    all_candidate_results,
    key=lambda result: (
        result["mean_roc_auc"],
        -result["candidate_epoch"],
    ),
)

selected_epoch = int(
    selected_epoch_result[
        "candidate_epoch"
    ]
)

selected_mean_auc = float(
    selected_epoch_result[
        "mean_roc_auc"
    ]
)

selected_std_auc = float(
    selected_epoch_result[
        "std_roc_auc"
    ]
)


# ============================================================
# PRINT FINAL RESULT
# ============================================================

print()
print("RUN 03 FINAL RESULT")

print()
print("Candidate epochs:")
print(candidate_epochs)

print()

print(
    f"Selected fixed epoch: "
    f"{selected_epoch}"
)

print(
    f"Mean 5-fold validation ROC-AUC: "
    f"{selected_mean_auc:.6f}"
)

print(
    f"Std 5-fold validation ROC-AUC: "
    f"{selected_std_auc:.6f}"
)

print()

print(
    "Selection criterion:"
)

print(
    "Highest mean 5-fold validation ROC-AUC"
)

print()

print(
    "Tie-breaker:"
)

print(
    "Lower epoch"
)

print()

print(
    "Test set:"
)

print(
    "NOT USED"
)


# ============================================================
# SAVE RUN SUMMARY
# ============================================================

run_summary = {

    "experiment_name":
        EXPERIMENT_NAME,

    "run_name":
        RUN_NAME,

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "maximum_epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "dropout_rate":
        DROPOUT_RATE,

    "n_folds":
        N_FOLDS,

    "threshold":
        THRESHOLD,

    "seed":
        SEED,

    "device":
        str(DEVICE),

    "development_dataset": {
        "total":
            int(len(all_paths)),

        "healthy":
            int(
                np.sum(
                    all_labels == 0
                )
            ),

        "sick":
            int(
                np.sum(
                    all_labels == 1
                )
            ),
    },

    "test_set_used":
        False,

    "candidate_epochs":
        candidate_epochs,

    "selected_fixed_epoch":
        selected_epoch,

    "selected_mean_roc_auc":
        selected_mean_auc,

    "selected_std_roc_auc":
        selected_std_auc,

    "selection_criterion":
        "Highest mean 5-fold validation ROC-AUC",

    "tie_breaker":
        "Lower epoch",

    "source_run":
        "Run_02_5-fold_CV_Training_Dynamics",

    "run02_mean_auc_best_epoch":
        mean_auc_best_epoch,

    "run02_mean_auc_best_value":
        mean_auc_best_value,

    "run02_fold_best_epochs": {
        str(fold): int(epoch)
        for fold, epoch
        in fold_best_epochs.items()
    },

    "normalization_protocol":
        "Fold-training-only temperature normalization",

    "retraining_performed":
        True,
}


run_summary_path = os.path.join(
    RUN03_RESULTS_DIR,
    "run_summary.json",
)

with open(
    run_summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        run_summary,
        f,
        indent=4,
    )

# ============================================================
# COMPLETION
# ============================================================

print()
print(
    "Run 03 completed successfully."
)

print()
print(
    "Results saved to:"
)

print(
    RUN03_RESULTS_DIR
)