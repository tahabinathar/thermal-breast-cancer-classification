import os
import json
import csv
import random
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    roc_curve
)

from sklearn.model_selection import StratifiedKFold

from torchvision import transforms

from torchvision import models

from torch.utils.data import Dataset


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

RUN_NAME = "Run_02_5-fold_CV_Training_Dynamics"

RUN_DIR = os.path.join(
    RESULTS_DIR,
    RUN_NAME
)

RESULTS_OUTPUT_DIR = os.path.join(
    RUN_DIR,
    "results"
)

CHECKPOINT_DIR = os.path.join(
    RUN_DIR,
    "checkpoints"
)

PLOT_DIR = os.path.join(
    RUN_DIR,
    "plots"
)

os.makedirs(
    RESULTS_OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


# ============================================================
# RUN SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 16

EPOCHS = 100

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.3

THRESHOLD = 0.5

NUM_CLASSES = 2

NUM_FOLDS = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
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
print("Number of folds:", NUM_FOLDS)
print("Threshold:", THRESHOLD)
print("Device:", DEVICE)
print("Test set: NOT USED") 

# ============================================================
# DATASET SETTINGS
# ============================================================

CLASS_NAMES = [
    "Healthy",
    "Sick"
]

CLASS_TO_LABEL = {
    "Healthy": 0,
    "Sick": 1
}


# ============================================================
# LOAD PREPROCESSING FUNCTIONS
# ============================================================

from preprocessing import (
    get_image_paths,
    calculate_temperature_range,
    preprocess_image
)


# ============================================================
# LOAD TRAIN + VALIDATION DATA
# ============================================================

all_image_paths = []

all_labels = []

print()

print(
    "Loading Train and Validation splits..."
)

for split in [
    "Train",
    "Validation"
]:

    for class_name in CLASS_NAMES:

        label = CLASS_TO_LABEL[
            class_name
        ]

        image_paths = get_image_paths(
            DATA_ROOT,
            split,
            class_name
        )

        print(
            f"{split} - "
            f"{class_name}: "
            f"{len(image_paths)} images"
        )

        for filepath in image_paths:

            all_image_paths.append(
                filepath
            )

            all_labels.append(
                label
            )


all_image_paths = np.asarray(
    all_image_paths
)

all_labels = np.asarray(
    all_labels,
    dtype=np.int64
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

total_healthy = np.sum(
    all_labels == 0
)

total_sick = np.sum(
    all_labels == 1
)

print()

print(
    "Pooled Train + Validation distribution:"
)

print(
    f"Healthy: {total_healthy}"
)

print(
    f"Sick: {total_sick}"
)

print(
    f"Total: {len(all_labels)}"
)

print()

print(
    "Test set was NOT loaded."
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),

        transforms.RandomRotation(
            degrees=18
        ),

        transforms.RandomAffine(
            degrees=0,
            scale=(0.95, 1.05)
        ),

        transforms.ToTensor(),

        transforms.Lambda(
            lambda x: x.repeat(
                3,
                1,
                1
            )
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
        )
    ]
)


validation_transform = transforms.Compose(
    [
        transforms.ToPILImage(),

        transforms.ToTensor(),

        transforms.Lambda(
            lambda x: x.repeat(
                3,
                1,
                1
            )
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
        )
    ]
)


# ============================================================
# DATASET
# ============================================================

class ThermalDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels,
        global_min,
        global_max,
        transform=None
    ):

        self.image_paths = image_paths

        self.labels = labels

        self.global_min = global_min

        self.global_max = global_max

        self.transform = transform

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(
        self,
        index
    ):

        image = preprocess_image(
            self.image_paths[index],
            self.global_min,
            self.global_max,
            IMAGE_SIZE
        )

        image = image.squeeze(
            axis=-1
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
# STRATIFIED K-FOLD CROSS-VALIDATION
# ============================================================

cross_validator = StratifiedKFold(
    n_splits=NUM_FOLDS,
    shuffle=True,
    random_state=SEED
)


fold_results = []

fold_histories = []

fold_temperature_ranges = []


print()

print(
    "5-FOLD CROSS-VALIDATION"
)


# ============================================================
# FOLD LOOP
# ============================================================

for fold, (
    train_indices,
    validation_indices
) in enumerate(
    cross_validator.split(
        all_image_paths,
        all_labels
    ),
    start=1
):

    print()

    print(
        f"FOLD {fold}/{NUM_FOLDS}"
    )

    # ========================================================
    # RESET SEED
    # ========================================================

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(SEED)

        torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


    # ========================================================
    # FOLD PATHS
    # ========================================================

    FOLD_DIR = os.path.join(
        RUN_DIR,
        f"fold_{fold}"
    )

    FOLD_RESULTS_OUTPUT_DIR = os.path.join(
        FOLD_DIR,
        "results"
    )

    FOLD_CHECKPOINT_DIR = os.path.join(
        FOLD_DIR,
        "checkpoints"
    )

    FOLD_PLOT_DIR = os.path.join(
        FOLD_DIR,
        "plots"
    )

    os.makedirs(
        FOLD_RESULTS_OUTPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        FOLD_CHECKPOINT_DIR,
        exist_ok=True
    )

    os.makedirs(
        FOLD_PLOT_DIR,
        exist_ok=True
    )


    # ========================================================
    # FOLD DATA
    # ========================================================

    fold_train_paths = all_image_paths[
        train_indices
    ]

    fold_train_labels = all_labels[
        train_indices
    ]

    fold_validation_paths = all_image_paths[
        validation_indices
    ]

    fold_validation_labels = all_labels[
        validation_indices
    ]


    fold_train_healthy = np.sum(
        fold_train_labels == 0
    )

    fold_train_sick = np.sum(
        fold_train_labels == 1
    )

    fold_validation_healthy = np.sum(
        fold_validation_labels == 0
    )

    fold_validation_sick = np.sum(
        fold_validation_labels == 1
    )


    print()

    print(
        "Training distribution:"
    )

    print(
        f"Healthy: {fold_train_healthy}"
    )

    print(
        f"Sick: {fold_train_sick}"
    )

    print()

    print(
        "Validation distribution:"
    )

    print(
        f"Healthy: {fold_validation_healthy}"
    )

    print(
        f"Sick: {fold_validation_sick}"
    )


    # ========================================================
    # FOLD TEMPERATURE RANGE
    # ========================================================

    GLOBAL_MIN, GLOBAL_MAX = (
        calculate_temperature_range(
            fold_train_paths
        )
    )


    print()

    print(
        "Temperature range calculated "
        "from fold Training data:"
    )

    print(
        f"Minimum: {GLOBAL_MIN:.6f}"
    )

    print(
        f"Maximum: {GLOBAL_MAX:.6f}"
    )


    fold_temperature_ranges.append(
        {
            "fold": fold,
            "minimum": float(GLOBAL_MIN),
            "maximum": float(GLOBAL_MAX)
        }
    )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    classes = np.array(
        [0, 1]
    )

    class_weight_values = (
        __import__(
            "sklearn.utils.class_weight",
            fromlist=["compute_class_weight"]
        ).compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=fold_train_labels
        )
    )

    class_weights = {

        0: float(
            class_weight_values[0]
        ),

        1: float(
            class_weight_values[1]
        )

    }


    print()

    print(
        "Class weights:"
    )

    print(
        f"Healthy: {class_weights[0]:.6f}"
    )

    print(
        f"Sick: {class_weights[1]:.6f}"
    )


    # ========================================================
    # DATASET
    # ========================================================

    train_dataset = ThermalDataset(
        fold_train_paths,
        fold_train_labels,
        GLOBAL_MIN,
        GLOBAL_MAX,
        transform=train_transform
    )

    validation_dataset = ThermalDataset(
        fold_validation_paths,
        fold_validation_labels,
        GLOBAL_MIN,
        GLOBAL_MAX,
        transform=validation_transform
    )


    # ========================================================
    # DATA LOADER
    # ========================================================

    train_generator = torch.Generator()

    train_generator.manual_seed(
        SEED
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=train_generator
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    # ========================================================
    # BUILD RESNET-18 MODEL
    # ========================================================

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )


    for parameter in model.parameters():

        parameter.requires_grad = False


    num_features = model.fc.in_features

    model.fc = nn.Sequential(

        nn.Linear(
            num_features,
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


    model = model.to(
        DEVICE
    )


    print()

    print(
        "ResNet-18 backbone frozen."
    )

    print(
        "Trainable parameters:"
    )

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            print(
                name
            )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE
    )


    # ========================================================
    # MODEL CHECKPOINT
    # ========================================================

    BEST_MODEL_PATH = os.path.join(
        FOLD_CHECKPOINT_DIR,
        "resnet18_frozen_best.pth"
    )

    best_validation_auc = -np.inf

    best_epoch = 0


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    history = {

        "loss": [],

        "accuracy": [],

        "auc": [],

        "val_loss": [],

        "val_accuracy": [],

        "val_auc": [],

        "val_pr_auc": []

    }


    print()

    print(
        "TRAINING"
    )


    for epoch in range(
        EPOCHS
    ):


        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        model.train()


        for module in model.children():

            if module is not model.fc:

                module.eval()


        train_loss_total = 0.0

        train_predictions = []

        train_targets = []


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
            ).squeeze(
                dim=1
            )


            sample_weights = torch.where(

                labels == 0,

                torch.full_like(
                    labels,
                    class_weights[0]
                ),

                torch.full_like(
                    labels,
                    class_weights[1]
                )

            )


            loss = (
                nn.functional
                .binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    weight=sample_weights
                )
            )


            loss.backward()

            optimizer.step()


            train_loss_total += (

                loss.item()
                * images.size(0)

            )


            probabilities = torch.sigmoid(
                logits
            )


            train_predictions.extend(

                probabilities.detach()
                .cpu()
                .numpy()

            )


            train_targets.extend(

                labels.detach()
                .cpu()
                .numpy()

            )


        train_loss = (

            train_loss_total
            / len(train_dataset)

        )


        train_predictions = np.asarray(
            train_predictions
        )

        train_targets = np.asarray(
            train_targets
        )


        train_binary = (

            train_predictions
            >= THRESHOLD

        ).astype(int)


        train_accuracy = np.mean(

            train_binary
            == train_targets

        )


        train_auc = roc_auc_score(

            train_targets,
            train_predictions

        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()


        val_loss_total = 0.0

        val_predictions = []

        val_targets = []


        with torch.no_grad():

            for images, labels in validation_loader:

                images = images.to(
                    DEVICE
                )

                labels = labels.to(
                    DEVICE
                )


                logits = model(
                    images
                ).squeeze(
                    dim=1
                )


                sample_weights = torch.where(

                    labels == 0,

                    torch.full_like(
                        labels,
                        class_weights[0]
                    ),

                    torch.full_like(
                        labels,
                        class_weights[1]
                    )

                )


                loss = (

                    nn.functional
                    .binary_cross_entropy_with_logits(
                        logits,
                        labels,
                        weight=sample_weights
                    )

                )


                val_loss_total += (

                    loss.item()
                    * images.size(0)

                )


                probabilities = torch.sigmoid(
                    logits
                )


                val_predictions.extend(

                    probabilities
                    .cpu()
                    .numpy()

                )


                val_targets.extend(

                    labels
                    .cpu()
                    .numpy()

                )


        val_loss = (

            val_loss_total
            / len(validation_dataset)

        )


        val_predictions = np.asarray(
            val_predictions
        )

        val_targets = np.asarray(
            val_targets
        )


        val_binary = (

            val_predictions
            >= THRESHOLD

        ).astype(int)


        val_accuracy = np.mean(

            val_binary
            == val_targets

        )


        val_auc = roc_auc_score(

            val_targets,
            val_predictions

        )


        val_pr_auc = average_precision_score(

            val_targets,
            val_predictions

        )


        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history["loss"].append(
            float(train_loss)
        )

        history["accuracy"].append(
            float(train_accuracy)
        )

        history["auc"].append(
            float(train_auc)
        )

        history["val_loss"].append(
            float(val_loss)
        )

        history["val_accuracy"].append(
            float(val_accuracy)
        )

        history["val_auc"].append(
            float(val_auc)
        )

        history["val_pr_auc"].append(
            float(val_pr_auc)
        )


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_auc > best_validation_auc:

            best_validation_auc = val_auc

            best_epoch = epoch + 1

            torch.save(

                model.state_dict(),

                BEST_MODEL_PATH

            )


        print(

            f"Fold {fold} | "

            f"Epoch {epoch + 1:03d}/{EPOCHS} | "

            f"Loss: {train_loss:.4f} | "

            f"Accuracy: {train_accuracy:.4f} | "

            f"AUC: {train_auc:.4f} | "

            f"Val Loss: {val_loss:.4f} | "

            f"Val Accuracy: {val_accuracy:.4f} | "

            f"Val AUC: {val_auc:.4f}"

        )


    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    history_path = os.path.join(

        FOLD_RESULTS_OUTPUT_DIR,

        "training_history.json"

    )


    with open(

        history_path,

        "w"

    ) as f:

        json.dump(

            history,

            f,

            indent=4

        )


    fold_histories.append(
        history
    )


    # ========================================================
    # FIND BEST VALIDATION AUC EPOCH
    # ========================================================

    best_epoch_index = int(

        np.argmax(

            history["val_auc"]

        )

    )


    best_epoch = (

        best_epoch_index + 1

    )


    best_validation_auc = float(

        history["val_auc"][

            best_epoch_index

        ]

    )


    print()

    print(
        "BEST VALIDATION AUC"
    )

    print(

        f"Best epoch: {best_epoch}"

    )

    print(

        f"Best validation AUC: "

        f"{best_validation_auc:.4f}"

    )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    model.load_state_dict(

        torch.load(

            BEST_MODEL_PATH,

            map_location=DEVICE

        )

    )


    model = model.to(
        DEVICE
    )

    model.eval()


    # ========================================================
    # VALIDATION PROBABILITIES
    # ========================================================

    validation_probabilities = []

    validation_labels = []


    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(
                DEVICE
            )


            logits = model(
                images
            ).squeeze(
                dim=1
            )


            probabilities = torch.sigmoid(
                logits
            )


            validation_probabilities.extend(

                probabilities
                .cpu()
                .numpy()

            )


            validation_labels.extend(

                labels.numpy()

            )


    validation_probabilities = np.asarray(

        validation_probabilities

    )


    validation_labels = np.asarray(

        validation_labels

    )


    # ========================================================
    # VALIDATION ROC-AUC
    # ========================================================

    validation_roc_auc = roc_auc_score(

        validation_labels,

        validation_probabilities

    )


    validation_pr_auc = average_precision_score(

        validation_labels,

        validation_probabilities

    )


    print()

    print(

        f"Validation ROC-AUC: "

        f"{validation_roc_auc:.4f}"

    )


    print(

        f"Validation PR-AUC: "

        f"{validation_pr_auc:.4f}"

    )


    # ========================================================
    # VALIDATION CLASSIFICATION METRICS
    # ========================================================

    validation_predictions = (

        validation_probabilities

        >= THRESHOLD

    ).astype(int)


    tn, fp, fn, tp = confusion_matrix(

        validation_labels,

        validation_predictions,

        labels=[0, 1]

    ).ravel()


    validation_accuracy = accuracy_score(

        validation_labels,

        validation_predictions

    )


    validation_balanced_accuracy = (

        balanced_accuracy_score(

            validation_labels,

            validation_predictions

        )

    )


    validation_sensitivity = recall_score(

        validation_labels,

        validation_predictions,

        zero_division=0

    )


    validation_specificity = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else 0.0

    )


    validation_precision = precision_score(

        validation_labels,

        validation_predictions,

        zero_division=0

    )


    validation_npv = (

        tn / (tn + fn)

        if (tn + fn) > 0

        else 0.0

    )


    validation_f1 = f1_score(

        validation_labels,

        validation_predictions,

        zero_division=0

    )


    validation_mcc = matthews_corrcoef(

        validation_labels,

        validation_predictions

    )


    validation_metrics = {

        "roc_auc":
            float(validation_roc_auc),

        "pr_auc":
            float(validation_pr_auc),

        "threshold":
            float(THRESHOLD),

        "accuracy":
            float(validation_accuracy),

        "balanced_accuracy":
            float(validation_balanced_accuracy),

        "sensitivity":
            float(validation_sensitivity),

        "specificity":
            float(validation_specificity),

        "precision":
            float(validation_precision),

        "npv":
            float(validation_npv),

        "f1":
            float(validation_f1),

        "mcc":
            float(validation_mcc),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp)

    }


    print()

    print(
        "VALIDATION METRICS"
    )

    print(

        f"ROC-AUC:            "
        f"{validation_roc_auc:.4f}"

    )

    print(

        f"PR-AUC:             "
        f"{validation_pr_auc:.4f}"

    )

    print(

        f"Threshold:          "
        f"{THRESHOLD:.4f}"

    )

    print(

        f"Accuracy:           "
        f"{validation_accuracy:.4f}"

    )

    print(

        f"Balanced Accuracy:  "
        f"{validation_balanced_accuracy:.4f}"

    )

    print(

        f"Sensitivity:        "
        f"{validation_sensitivity:.4f}"

    )

    print(

        f"Specificity:        "
        f"{validation_specificity:.4f}"

    )

    print(

        f"Precision:          "
        f"{validation_precision:.4f}"

    )

    print(

        f"NPV:                "
        f"{validation_npv:.4f}"

    )

    print(

        f"F1:                 "
        f"{validation_f1:.4f}"

    )

    print(

        f"MCC:                "
        f"{validation_mcc:.4f}"

    )

    print()

    print(

        f"TN={tn}, FP={fp}, "
        f"FN={fn}, TP={tp}"

    )


    # ========================================================
    # SAVE VALIDATION METRICS
    # ========================================================

    validation_metrics_path = os.path.join(

        FOLD_RESULTS_OUTPUT_DIR,

        "validation_metrics.json"

    )


    with open(

        validation_metrics_path,

        "w"

    ) as f:

        json.dump(

            validation_metrics,

            f,

            indent=4

        )


    # ========================================================
    # SAVE ROC DATA
    # ========================================================

    fpr, tpr, roc_thresholds = roc_curve(

        validation_labels,

        validation_probabilities

    )


    roc_data = {

        "fpr": [

            float(value)

            for value in fpr

        ],

        "tpr": [

            float(value)

            for value in tpr

        ],

        "thresholds": [

            float(value)

            for value in roc_thresholds

        ],

        "roc_auc":

            float(validation_roc_auc)

    }


    roc_data_path = os.path.join(

        FOLD_RESULTS_OUTPUT_DIR,

        "validation_roc_data.json"

    )


    with open(

        roc_data_path,

        "w"

    ) as f:

        json.dump(

            roc_data,

            f,

            indent=4

        )


    # ========================================================
    # SAVE FOLD SUMMARY
    # ========================================================

    fold_summary = {

        "fold":
            fold,

        "seed":
            SEED,

        "train_samples":
            int(len(fold_train_labels)),

        "validation_samples":
            int(len(fold_validation_labels)),

        "train_healthy":
            int(fold_train_healthy),

        "train_sick":
            int(fold_train_sick),

        "validation_healthy":
            int(fold_validation_healthy),

        "validation_sick":
            int(fold_validation_sick),

        "temperature_min":
            float(GLOBAL_MIN),

        "temperature_max":
            float(GLOBAL_MAX),

        "healthy_class_weight":
            float(class_weights[0]),

        "sick_class_weight":
            float(class_weights[1]),

        "best_epoch":
            int(best_epoch),

        "epoch_1_validation_auc":
            float(history["val_auc"][0]),

        "best_validation_auc":
            float(best_validation_auc),

        "epoch_100_validation_auc":
            float(history["val_auc"][-1]),

        "validation_roc_auc":
            float(validation_roc_auc),

        "validation_pr_auc":
            float(validation_pr_auc),

        "threshold":
            float(THRESHOLD),

        "accuracy":
            float(validation_accuracy),

        "balanced_accuracy":
            float(validation_balanced_accuracy),

        "sensitivity":
            float(validation_sensitivity),

        "specificity":
            float(validation_specificity),

        "precision":
            float(validation_precision),

        "npv":
            float(validation_npv),

        "f1":
            float(validation_f1),

        "mcc":
            float(validation_mcc),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp)

    }


    fold_results.append(
        fold_summary
    )


    fold_summary_path = os.path.join(

        FOLD_RESULTS_OUTPUT_DIR,

        "fold_summary.json"

    )


    with open(

        fold_summary_path,

        "w"

    ) as f:

        json.dump(

            fold_summary,

            f,

            indent=4

        )


    # ========================================================
    # SAVE VALIDATION PREDICTIONS
    # ========================================================

    validation_predictions_path = os.path.join(

        FOLD_RESULTS_OUTPUT_DIR,

        "validation_predictions.csv"

    )


    with open(

        validation_predictions_path,

        "w",

        newline=""

    ) as f:

        writer = csv.writer(f)

        writer.writerow(

            [

                "filepath",

                "true_label",

                "predicted_probability",

                "predicted_label"

            ]

        )


        for filepath, label, probability, prediction in zip(

            fold_validation_paths,

            validation_labels,

            validation_probabilities,

            validation_predictions

        ):

            writer.writerow(

                [

                    filepath,

                    int(label),

                    float(probability),

                    int(prediction)

                ]

            )


    # ========================================================
    # PLOTS
    # ========================================================

    epochs_range = range(

        1,

        EPOCHS + 1

    )


    plt_imported = False

    import matplotlib.pyplot as plt

    plt_imported = True


    # --------------------------------------------------------
    # Training and Validation Loss
    # --------------------------------------------------------

    plt.figure()

    plt.plot(

        epochs_range,

        history["loss"],

        label="Training Loss"

    )

    plt.plot(

        epochs_range,

        history["val_loss"],

        label="Validation Loss"

    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        f"Fold {fold} Training and Validation Loss"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FOLD_PLOT_DIR,

            "training_validation_loss.png"

        ),

        dpi=300

    )

    plt.close()


    # --------------------------------------------------------
    # Training and Validation Accuracy
    # --------------------------------------------------------

    plt.figure()

    plt.plot(

        epochs_range,

        history["accuracy"],

        label="Training Accuracy"

    )

    plt.plot(

        epochs_range,

        history["val_accuracy"],

        label="Validation Accuracy"

    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        f"Fold {fold} Training and Validation Accuracy"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FOLD_PLOT_DIR,

            "training_validation_accuracy.png"

        ),

        dpi=300

    )

    plt.close()


    # --------------------------------------------------------
    # Training and Validation AUC
    # --------------------------------------------------------

    plt.figure()

    plt.plot(

        epochs_range,

        history["auc"],

        label="Training AUC"

    )

    plt.plot(

        epochs_range,

        history["val_auc"],

        label="Validation AUC"

    )

    plt.axvline(

        best_epoch,

        linestyle="--",

        label="Best Epoch"

    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "ROC-AUC"
    )

    plt.title(
        f"Fold {fold} Training and Validation AUC"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FOLD_PLOT_DIR,

            "training_validation_auc.png"

        ),

        dpi=300

    )

    plt.close()


    # --------------------------------------------------------
    # Validation ROC Curve
    # --------------------------------------------------------

    plt.figure()

    plt.plot(

        fpr,

        tpr,

        label=(

            f"ResNet-18 "

            f"(AUC = {validation_roc_auc:.4f})"

        )

    )

    plt.plot(

        [0, 1],

        [0, 1],

        linestyle="--"

    )

    plt.xlabel(

        "False Positive Rate"

    )

    plt.ylabel(

        "True Positive Rate"

    )

    plt.title(

        f"Fold {fold} Validation ROC Curve"

    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FOLD_PLOT_DIR,

            "validation_roc_curve.png"

        ),

        dpi=300

    )

    plt.close()


# ============================================================
# SAVE FOLD TEMPERATURE RANGES
# ============================================================

fold_temperature_ranges_path = os.path.join(

    RESULTS_OUTPUT_DIR,

    "fold_temperature_ranges.json"

)


with open(

    fold_temperature_ranges_path,

    "w"

) as f:

    json.dump(

        fold_temperature_ranges,

        f,

        indent=4

    )


# ============================================================
# SAVE FOLD SUMMARY
# ============================================================

fold_summary_path = os.path.join(

    RESULTS_OUTPUT_DIR,

    "fold_summary.csv"

)


with open(

    fold_summary_path,

    "w",

    newline=""

) as f:

    writer = csv.DictWriter(

        f,

        fieldnames=list(

            fold_results[0].keys()

        )

    )

    writer.writeheader()

    writer.writerows(
        fold_results
    )


# ============================================================
# EPOCH-WISE CROSS-VALIDATION SUMMARY
# ============================================================

epoch_wise_results = []


for epoch in range(
    EPOCHS
):

    loss_values = [

        history["loss"][epoch]

        for history in fold_histories

    ]

    val_loss_values = [

        history["val_loss"][epoch]

        for history in fold_histories

    ]

    accuracy_values = [

        history["accuracy"][epoch]

        for history in fold_histories

    ]

    val_accuracy_values = [

        history["val_accuracy"][epoch]

        for history in fold_histories

    ]

    auc_values = [

        history["auc"][epoch]

        for history in fold_histories

    ]

    val_auc_values = [

        history["val_auc"][epoch]

        for history in fold_histories

    ]


    epoch_wise_results.append(

        {

            "epoch":
                epoch + 1,

            "loss_mean":
                float(np.mean(loss_values)),

            "loss_std":
                float(np.std(loss_values)),

            "val_loss_mean":
                float(np.mean(val_loss_values)),

            "val_loss_std":
                float(np.std(val_loss_values)),

            "accuracy_mean":
                float(np.mean(accuracy_values)),

            "accuracy_std":
                float(np.std(accuracy_values)),

            "val_accuracy_mean":
                float(np.mean(val_accuracy_values)),

            "val_accuracy_std":
                float(np.std(val_accuracy_values)),

            "auc_mean":
                float(np.mean(auc_values)),

            "auc_std":
                float(np.std(auc_values)),

            "val_auc_mean":
                float(np.mean(val_auc_values)),

            "val_auc_std":
                float(np.std(val_auc_values))

        }

    )


# ============================================================
# BEST MEAN VALIDATION AUC EPOCH
# ============================================================

mean_validation_auc = np.asarray(

    [

        result["val_auc_mean"]

        for result in epoch_wise_results

    ]

)


best_mean_validation_auc_index = int(

    np.argmax(
        mean_validation_auc
    )

)


best_mean_validation_auc_epoch = (

    best_mean_validation_auc_index + 1

)


best_mean_validation_auc = float(

    mean_validation_auc[

        best_mean_validation_auc_index

    ]

)


print()

print(
    "BEST MEAN VALIDATION AUC"
)

print(

    f"Best epoch: "
    f"{best_mean_validation_auc_epoch}"

)

print(

    f"Best mean validation AUC: "
    f"{best_mean_validation_auc:.4f}"

)


# ============================================================
# SAVE EPOCH-WISE CROSS-VALIDATION SUMMARY
# ============================================================

epoch_wise_results_path = os.path.join(

    RESULTS_OUTPUT_DIR,

    "epoch_wise_cv_summary.csv"

)


with open(

    epoch_wise_results_path,

    "w",

    newline=""

) as f:

    writer = csv.DictWriter(

        f,

        fieldnames=list(

            epoch_wise_results[0].keys()

        )

    )

    writer.writeheader()

    writer.writerows(

        epoch_wise_results

    )


# ============================================================
# CROSS-VALIDATION METRICS
# ============================================================

metric_names = [

    "epoch_1_validation_auc",

    "best_validation_auc",

    "best_epoch",

    "epoch_100_validation_auc",

    "validation_roc_auc",

    "validation_pr_auc",

    "accuracy",

    "balanced_accuracy",

    "sensitivity",

    "specificity",

    "precision",

    "npv",

    "f1",

    "mcc"

]


cross_validation_metrics = {}


for metric_name in metric_names:

    values = np.asarray(

        [

            result[metric_name]

            for result in fold_results

        ],

        dtype=np.float64

    )


    cross_validation_metrics[metric_name] = {

        "mean":
            float(np.mean(values)),

        "std":
            float(np.std(values))

    }


# ============================================================
# SAVE CROSS-VALIDATION SUMMARY
# ============================================================

cv_summary = {

    "experiment_name":
        EXPERIMENT_NAME,

    "run_name":
        RUN_NAME,

    "seed":
        SEED,

    "seed_reset_each_fold":
        True,

    "num_folds":
        NUM_FOLDS,

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "dropout_rate":
        DROPOUT_RATE,

    "threshold":
        THRESHOLD,

    "pooled_train_validation_samples":
        int(len(all_labels)),

    "healthy_samples":
        int(total_healthy),

    "sick_samples":
        int(total_sick),

    "test_loaded":
        False,

    "test_evaluated":
        False,

    "test_used_during_training":
        False,

    "best_mean_validation_auc_epoch":
        int(best_mean_validation_auc_epoch),

    "best_mean_validation_auc":
        float(best_mean_validation_auc),

    "metrics":
        cross_validation_metrics

}


cv_summary_path = os.path.join(

    RESULTS_OUTPUT_DIR,

    "cv_summary.json"

)


with open(

    cv_summary_path,

    "w"

) as f:

    json.dump(

        cv_summary,

        f,

        indent=4

    )


# ============================================================
# CROSS-VALIDATION PLOTS
# ============================================================

import matplotlib.pyplot as plt


epochs_range = range(

    1,

    EPOCHS + 1

)


# ------------------------------------------------------------
# Mean Validation AUC
# ------------------------------------------------------------

val_auc_mean = np.asarray(

    [

        result["val_auc_mean"]

        for result in epoch_wise_results

    ]

)


val_auc_std = np.asarray(

    [

        result["val_auc_std"]

        for result in epoch_wise_results

    ]

)


plt.figure()

plt.plot(

    epochs_range,

    val_auc_mean,

    label="Mean Validation AUC"

)

plt.fill_between(

    epochs_range,

    val_auc_mean - val_auc_std,

    val_auc_mean + val_auc_std,

    alpha=0.2

)

plt.axvline(

    best_mean_validation_auc_epoch,

    linestyle="--",

    label="Best Mean Epoch"

)

plt.xlabel(

    "Epoch"

)

plt.ylabel(

    "ROC-AUC"

)

plt.title(

    "5-Fold Cross-Validation Validation AUC"

)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(

    os.path.join(

        PLOT_DIR,

        "cv_validation_auc_mean_sd.png"

    ),

    dpi=300

)

plt.close()


# ------------------------------------------------------------
# Mean Loss
# ------------------------------------------------------------

loss_mean = np.asarray(

    [

        result["loss_mean"]

        for result in epoch_wise_results

    ]

)


loss_std = np.asarray(

    [

        result["loss_std"]

        for result in epoch_wise_results

    ]

)


val_loss_mean = np.asarray(

    [

        result["val_loss_mean"]

        for result in epoch_wise_results

    ]

)


val_loss_std = np.asarray(

    [

        result["val_loss_std"]

        for result in epoch_wise_results

    ]

)


plt.figure()

plt.plot(

    epochs_range,

    loss_mean,

    label="Mean Training Loss"

)

plt.plot(

    epochs_range,

    val_loss_mean,

    label="Mean Validation Loss"

)

plt.fill_between(

    epochs_range,

    loss_mean - loss_std,

    loss_mean + loss_std,

    alpha=0.2

)

plt.fill_between(

    epochs_range,

    val_loss_mean - val_loss_std,

    val_loss_mean + val_loss_std,

    alpha=0.2

)

plt.xlabel(

    "Epoch"

)

plt.ylabel(

    "Loss"

)

plt.title(

    "5-Fold Cross-Validation Loss"

)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(

    os.path.join(

        PLOT_DIR,

        "cv_loss_mean_sd.png"

    ),

    dpi=300

)

plt.close()


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

training_summary = {

    "experiment_name":
        EXPERIMENT_NAME,

    "run_name":
        RUN_NAME,

    "seed":
        SEED,

    "seed_reset_each_fold":
        True,

    "device":
        str(DEVICE),

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "dropout_rate":
        DROPOUT_RATE,

    "threshold":
        THRESHOLD,

    "num_folds":
        NUM_FOLDS,

    "pooled_train_validation_samples":
        int(len(all_labels)),

    "healthy_samples":
        int(total_healthy),

    "sick_samples":
        int(total_sick),

    "cross_validation":
        "5-fold StratifiedKFold",

    "cross_validation_random_state":
        SEED,

    "temperature_range_source":
        "Fold Training split only",

    "backbone":
        "ResNet-18",

    "pretrained_weights":
        "ImageNet1K_V1",

    "backbone_frozen":
        True,

    "test_loaded":
        False,

    "test_evaluated":
        False,

    "test_used_during_training":
        False,

    "best_mean_validation_auc_epoch":
        int(best_mean_validation_auc_epoch),

    "best_mean_validation_auc":
        float(best_mean_validation_auc)

}


training_summary_path = os.path.join(

    RESULTS_OUTPUT_DIR,

    "training_summary.json"

)


with open(

    training_summary_path,

    "w"

) as f:

    json.dump(

        training_summary,

        f,

        indent=4

    )


# ============================================================
# SAVE MODEL SUMMARY
# ============================================================

model_summary_path = os.path.join(

    RESULTS_OUTPUT_DIR,

    "model_summary.txt"

)


model_summary_model = models.resnet18(

    weights=models.ResNet18_Weights.IMAGENET1K_V1

)


for parameter in model_summary_model.parameters():

    parameter.requires_grad = False


num_features = model_summary_model.fc.in_features


model_summary_model.fc = nn.Sequential(

    nn.Linear(
        num_features,
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


total_parameters = 0

total_trainable = 0


for parameter in model_summary_model.parameters():

    total_parameters += parameter.numel()

    if parameter.requires_grad:

        total_trainable += parameter.numel()


with open(

    model_summary_path,

    "w"

) as f:

    f.write(

        f"Total parameters: "
        f"{total_parameters}\n"

    )

    f.write(

        f"Trainable parameters: "
        f"{total_trainable}\n"

    )

    f.write(

        f"Frozen parameters: "
        f"{total_parameters - total_trainable}\n"

    )


# ============================================================
# FINAL TRAINING INFORMATION
# ============================================================

print()

print(
    "FINAL TRAINING INFORMATION"
)

print(

    f"Experiment: "
    f"{EXPERIMENT_NAME}"

)

print(

    f"Run: "
    f"{RUN_NAME}"

)

print(

    f"Seed: "
    f"{SEED}"

)

print(

    "Seed reset to 10 "
    "at the beginning of every fold."

)

print(

    f"Best mean validation AUC epoch: "
    f"{best_mean_validation_auc_epoch}"

)

print(

    f"Best mean validation AUC: "
    f"{best_mean_validation_auc:.4f}"

)

print()

print(

    "Test set was NOT loaded."

)

print(

    "Test set was NOT evaluated."

)

print(

    "Test set was NOT used during training."

)

print()

print(

    f"Results saved to:\n"
    f"{RESULTS_OUTPUT_DIR}"

)

print()

print(

    f"Plots saved to:\n"
    f"{PLOT_DIR}"

)

print()

print(
    "EXP02 Run02 completed."
)