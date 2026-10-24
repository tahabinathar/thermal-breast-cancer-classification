import os
import json
import random
import csv

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

from sklearn.utils.class_weight import compute_class_weight

from torchvision import transforms
from torchvision import models

from torch.utils.data import Dataset

import matplotlib.pyplot as plt

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 10

os.environ["PYTHONHASHSEED"] = str(SEED)

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
    r"\EXP_03_DMR_Anterior_ResNet18_Partial_FineTuning\Results"
)

EXPERIMENT_NAME = (
    "EXP03_DMR_Anterior_ResNet18_Partial_FineTuning"
)

RUN_NAME = (
    "Run_01_Partial_FineTuning_Baseline"
)

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

CLASSIFIER_LEARNING_RATE = 0.001
BACKBONE_LEARNING_RATE = 0.0001

DROPOUT_RATE = 0.3

# Used only for diagnostic validation metrics.
# This is NOT the final decision threshold.
DIAGNOSTIC_THRESHOLD = 0.5

NUM_WORKERS = 0

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
print(
    "Classifier learning rate:",
    CLASSIFIER_LEARNING_RATE
)
print(
    "Backbone learning rate:",
    BACKBONE_LEARNING_RATE
)
print("Dropout rate:", DROPOUT_RATE)
print(
    "Diagnostic threshold:",
    DIAGNOSTIC_THRESHOLD
)
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
# LOAD TRAINING IMAGE PATHS
# ============================================================

train_paths = []
train_labels = []

for class_name in CLASS_NAMES:

    label = CLASS_TO_LABEL[
        class_name
    ]

    class_paths = get_image_paths(
        DATA_ROOT,
        "Train",
        class_name
    )

    train_paths.extend(
        class_paths
    )

    train_labels.extend(
        [label] * len(class_paths)
    )


# ============================================================
# LOAD VALIDATION IMAGE PATHS
# ============================================================

validation_paths = []
validation_labels = []

for class_name in CLASS_NAMES:

    label = CLASS_TO_LABEL[
        class_name
    ]

    class_paths = get_image_paths(
        DATA_ROOT,
        "Validation",
        class_name
    )

    validation_paths.extend(
        class_paths
    )

    validation_labels.extend(
        [label] * len(class_paths)
    )


# ============================================================
# SORT DATA
# ============================================================

train_pairs = sorted(
    zip(
        train_paths,
        train_labels
    ),
    key=lambda x: x[0]
)

train_paths = [
    x[0]
    for x in train_pairs
]

train_labels = [
    x[1]
    for x in train_pairs
]


validation_pairs = sorted(
    zip(
        validation_paths,
        validation_labels
    ),
    key=lambda x: x[0]
)

validation_paths = [
    x[0]
    for x in validation_pairs
]

validation_labels = [
    x[1]
    for x in validation_pairs
]


# ============================================================
# DATASET SUMMARY
# ============================================================

print()
print("DATASET")

print(
    f"Training images: "
    f"{len(train_paths)}"
)

print(
    f"Validation images: "
    f"{len(validation_paths)}"
)

print(
    f"Training Healthy: "
    f"{np.sum(np.array(train_labels) == 0)}"
)

print(
    f"Training Sick: "
    f"{np.sum(np.array(train_labels) == 1)}"
)

print(
    f"Validation Healthy: "
    f"{np.sum(np.array(validation_labels) == 0)}"
)

print(
    f"Validation Sick: "
    f"{np.sum(np.array(validation_labels) == 1)}"
)


# ============================================================
# CALCULATE GLOBAL TEMPERATURE RANGE
# ============================================================

GLOBAL_MIN, GLOBAL_MAX = (
    calculate_temperature_range(
        train_paths
    )
)

print()
print(
    "Global temperature range calculated "
    "from Train split:"
)

print(
    f"Minimum: {GLOBAL_MIN:.6f}"
)

print(
    f"Maximum: {GLOBAL_MAX:.6f}"
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


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
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
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
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ]
)


# ============================================================
# DATASET
# ============================================================

class ThermalDataset(Dataset):

    def __init__(
        self,
        filepaths,
        labels,
        transform=None
    ):

        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):

        return len(
            self.filepaths
        )

    def __getitem__(
        self,
        index
    ):

        filepath = self.filepaths[
            index
        ]

        label = self.labels[
            index
        ]

        image = preprocess_image(
            filepath,
            GLOBAL_MIN,
            GLOBAL_MAX,
            IMAGE_SIZE
        )

        image = np.squeeze(
            image,
            axis=-1
        )

        if self.transform is not None:

            image = self.transform(
                image
            )

        label = torch.tensor(
            label,
            dtype=torch.float32
        )

        return image, label


train_dataset = ThermalDataset(
    train_paths,
    train_labels,
    transform=train_transform
)

validation_dataset = ThermalDataset(
    validation_paths,
    validation_labels,
    transform=validation_transform
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.array(
    [0, 1]
)

class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=np.asarray(
        train_labels
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
print("Class weights:")

print(
    f"Healthy: {class_weights[0]:.6f}"
)

print(
    f"Sick: {class_weights[1]:.6f}"
)


# ============================================================
# BUILD RESNET-18 MODEL
# ============================================================

model = models.resnet18(
    weights=models.ResNet18_Weights.IMAGENET1K_V1
)


# ============================================================
# FREEZE ENTIRE BACKBONE
# ============================================================

for parameter in model.parameters():

    parameter.requires_grad = False


# ============================================================
# PARTIAL FINE-TUNING
# ============================================================

for parameter in model.layer4.parameters():

    parameter.requires_grad = True


# ============================================================
# CLASSIFIER
# ============================================================

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


# Ensure classifier is trainable.

for parameter in model.fc.parameters():

    parameter.requires_grad = True


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(
    DEVICE
)


# ============================================================
# PARAMETER SUMMARY
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

frozen_parameters = (
    total_parameters
    - trainable_parameters
)

print()
print("MODEL")

print(
    "Architecture: ResNet-18"
)

print(
    "Pretrained weights: "
    "ImageNet1K_V1"
)

print(
    "Fine-tuning strategy: "
    "Partial"
)

print(
    "Frozen modules: "
    "conv1, bn1, layer1, layer2, layer3"
)

print(
    "Trainable modules: "
    "layer4, fc"
)

print(
    f"Total parameters: "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

print(
    f"Frozen parameters: "
    f"{frozen_parameters:,}"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    [
        {
            "params":
                model.layer4.parameters(),
            "lr":
                BACKBONE_LEARNING_RATE
        },

        {
            "params":
                model.fc.parameters(),
            "lr":
                CLASSIFIER_LEARNING_RATE
        }
    ]
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    reduction="none"
)


# ============================================================
# SAMPLE WEIGHTS
# ============================================================

def calculate_sample_weights(
    labels
):

    weights = torch.ones_like(
        labels,
        dtype=torch.float32
    )

    weights[
        labels == 0
    ] = class_weights[0]

    weights[
        labels == 1
    ] = class_weights[1]

    return weights


# ============================================================
# MODEL CHECKPOINT
# ============================================================

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "resnet18_partial_finetuning_best.pth"
)

best_validation_auc = -np.inf
best_epoch = 0


# ============================================================
# TRAIN MODEL
# ============================================================

history = {
    "loss": [],
    "accuracy": [],
    "auc": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_auc": []
}

print()
print("TRAINING")


for epoch in range(
    EPOCHS
):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    # Keep frozen layers in evaluation mode.
    # This prevents their BatchNorm running statistics
    # from changing.

    model.conv1.eval()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()

    train_loss_total = 0.0

    train_predictions = []
    train_targets = []

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = model(
            images
        ).squeeze(
            dim=1
        )

        probabilities = torch.sigmoid(
            logits
        )

        loss_per_sample = criterion(
            logits,
            labels
        )

        sample_weights = (
            calculate_sample_weights(
                labels
            )
        )

        loss = (
            loss_per_sample
            * sample_weights
        ).mean()

        loss.backward()

        optimizer.step()

        train_loss_total += (
            loss.item()
            * images.size(0)
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
        >= DIAGNOSTIC_THRESHOLD
    ).astype(int)

    train_accuracy = accuracy_score(
        train_targets,
        train_binary
    )

    train_auc = roc_auc_score(
        train_targets,
        train_predictions
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0

    val_predictions = []
    val_targets = []

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(
                images
            ).squeeze(
                dim=1
            )

            probabilities = torch.sigmoid(
                logits
            )

            loss_per_sample = criterion(
                logits,
                labels
            )

            sample_weights = (
                calculate_sample_weights(
                    labels
                )
            )

            loss = (
                loss_per_sample
                * sample_weights
            ).mean()

            val_loss_total += (
                loss.item()
                * images.size(0)
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
        >= DIAGNOSTIC_THRESHOLD
    ).astype(int)

    val_accuracy = accuracy_score(
        val_targets,
        val_binary
    )

    val_auc = roc_auc_score(
        val_targets,
        val_predictions
    )


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_auc > best_validation_auc:

        best_validation_auc = val_auc
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )


    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Accuracy: {train_accuracy:.4f} | "
        f"AUC: {train_auc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f} | "
        f"Val AUC: {val_auc:.4f}"
    )


# ============================================================
# TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    RESULTS_OUTPUT_DIR,
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


# ============================================================
# FIND BEST VALIDATION AUC EPOCH
# ============================================================

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
print("BEST VALIDATION AUC")

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation AUC: "
    f"{best_validation_auc:.4f}"
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

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


# ============================================================
# VALIDATION PROBABILITIES
# ============================================================

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


# ============================================================
# VALIDATION ROC-AUC
# ============================================================

validation_roc_auc = roc_auc_score(
    validation_labels,
    validation_probabilities
)

print()
print(
    f"Validation ROC-AUC: "
    f"{validation_roc_auc:.4f}"
)


# ============================================================
# VALIDATION CLASSIFICATION METRICS
# ============================================================

validation_predictions = (
    validation_probabilities
    >= DIAGNOSTIC_THRESHOLD
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

validation_pr_auc = (
    average_precision_score(
        validation_labels,
        validation_probabilities
    )
)


validation_metrics = {

    "roc_auc": float(
        validation_roc_auc
    ),

    "pr_auc": float(
        validation_pr_auc
    ),

    "threshold": float(
        DIAGNOSTIC_THRESHOLD
    ),

    "accuracy": float(
        validation_accuracy
    ),

    "balanced_accuracy": float(
        validation_balanced_accuracy
    ),

    "sensitivity": float(
        validation_sensitivity
    ),

    "specificity": float(
        validation_specificity
    ),

    "precision": float(
        validation_precision
    ),

    "npv": float(
        validation_npv
    ),

    "f1": float(
        validation_f1
    ),

    "mcc": float(
        validation_mcc
    ),

    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp)
}


print()
print("VALIDATION METRICS")

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
    f"{DIAGNOSTIC_THRESHOLD:.4f}"
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


# ============================================================
# SAVE VALIDATION METRICS
# ============================================================

validation_metrics_path = os.path.join(
    RESULTS_OUTPUT_DIR,
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


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

validation_predictions_path = os.path.join(
    RESULTS_OUTPUT_DIR,
    "validation_predictions.csv"
)

with open(
    validation_predictions_path,
    "w",
    newline=""
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow(
        [
            "filename",
            "true_label",
            "true_class",
            "probability_sick",
            "predicted_label",
            "predicted_class",
            "correct"
        ]
    )

    for (
        filepath,
        true_label,
        probability,
        prediction
    ) in zip(
        validation_paths,
        validation_labels,
        validation_probabilities,
        validation_predictions
    ):

        writer.writerow(
            [
                os.path.basename(
                    filepath
                ),

                int(
                    true_label
                ),

                (
                    "Sick"
                    if true_label == 1
                    else "Healthy"
                ),

                float(
                    probability
                ),

                int(
                    prediction
                ),

                (
                    "Sick"
                    if prediction == 1
                    else "Healthy"
                ),

                int(
                    true_label == prediction
                )
            ]
        )


# ============================================================
# SAVE ROC DATA
# ============================================================

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

    "roc_auc": float(
        validation_roc_auc
    )
}


roc_data_path = os.path.join(
    RESULTS_OUTPUT_DIR,
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


# ============================================================
# SAVE TEMPERATURE RANGE
# ============================================================

temperature_range = {

    "source":
        "Train split only",

    "temperature_min":
        float(GLOBAL_MIN),

    "temperature_max":
        float(GLOBAL_MAX),

    "train_images_used":
        int(len(train_paths))
}

temperature_range_path = os.path.join(
    RESULTS_OUTPUT_DIR,
    "temperature_range.json"
)

with open(
    temperature_range_path,
    "w"
) as f:

    json.dump(
        temperature_range,
        f,
        indent=4
    )


# ============================================================
# PLOTS
# ============================================================

epochs_range = range(
    1,
    EPOCHS + 1
)


# ------------------------------------------------------------
# Training and Validation Loss
# ------------------------------------------------------------

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
    "Training and Validation Loss"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "training_validation_loss.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Training and Validation Accuracy
# ------------------------------------------------------------

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
    "Training and Validation Accuracy"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "training_validation_accuracy.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Training and Validation AUC
# ------------------------------------------------------------

plt.figure()

plt.plot(
    epochs_range,
    history["auc"],
    label="Training ROC-AUC"
)

plt.plot(
    epochs_range,
    history["val_auc"],
    label="Validation ROC-AUC"
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
    "Training and Validation ROC-AUC"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "training_validation_auc.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Validation ROC Curve
# ------------------------------------------------------------

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
    "Validation ROC Curve"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "validation_roc_curve.png"
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

    "device":
        str(DEVICE),

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "classifier_learning_rate":
        CLASSIFIER_LEARNING_RATE,

    "backbone_learning_rate":
        BACKBONE_LEARNING_RATE,

    "dropout_rate":
        DROPOUT_RATE,

    "diagnostic_threshold":
        DIAGNOSTIC_THRESHOLD,

    "global_temperature_min":
        float(GLOBAL_MIN),

    "global_temperature_max":
        float(GLOBAL_MAX),

    "temperature_range_source":
        "Train split only",

    "train_samples":
        int(len(train_labels)),

    "validation_samples":
        int(len(validation_labels)),

    "best_epoch":
        int(best_epoch),

    "best_validation_auc":
        float(best_validation_auc),

    "final_validation_roc_auc":
        float(validation_roc_auc),

    "final_validation_pr_auc":
        float(validation_pr_auc),

    "backbone":
        "ResNet-18",

    "pretrained_weights":
        "ImageNet1K_V1",

    "fine_tuning_strategy":
        "Partial",

    "frozen_modules": [
        "conv1",
        "bn1",
        "layer1",
        "layer2",
        "layer3"
    ],

    "trainable_modules": [
        "layer4",
        "fc"
    ],

    "total_parameters":
        int(total_parameters),

    "trainable_parameters":
        int(trainable_parameters),

    "frozen_parameters":
        int(frozen_parameters),

    "test_loaded":
        False,

    "test_evaluated":
        False,

    "test_used_during_training":
        False
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

with open(
    model_summary_path,
    "w"
) as f:

    f.write(
        f"Architecture: ResNet-18\n"
    )

    f.write(
        f"Pretrained weights: "
        f"ImageNet1K_V1\n"
    )

    f.write(
        f"Fine-tuning strategy: "
        f"Partial\n"
    )

    f.write(
        f"Frozen modules: "
        f"conv1, bn1, layer1, layer2, layer3\n"
    )

    f.write(
        f"Trainable modules: "
        f"layer4, fc\n"
    )

    f.write(
        f"Total parameters: "
        f"{total_parameters}\n"
    )

    f.write(
        f"Trainable parameters: "
        f"{trainable_parameters}\n"
    )

    f.write(
        f"Frozen parameters: "
        f"{frozen_parameters}\n"
    )


# ============================================================
# FINAL TRAINING INFORMATION
# ============================================================

print()
print("FINAL TRAINING INFORMATION")

print(
    f"Experiment: {EXPERIMENT_NAME}"
)

print(
    f"Run: {RUN_NAME}"
)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation AUC: "
    f"{best_validation_auc:.4f}"
)

print(
    f"Validation ROC-AUC: "
    f"{validation_roc_auc:.4f}"
)

print(
    f"Validation PR-AUC: "
    f"{validation_pr_auc:.4f}"
)

print(
    f"Diagnostic threshold: "
    f"{DIAGNOSTIC_THRESHOLD:.4f}"
)

print()

print(
    "Temperature range calculated "
    "from Train split only."
)

print(
    "Partial fine-tuning: "
    "layer4 and fc trainable."
)

print(
    "Frozen modules: "
    "conv1, bn1, layer1, layer2, layer3."
)

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
    f"Best model saved to:\n"
    f"{BEST_MODEL_PATH}"
)

print()

print(
    f"Plots saved to:\n"
    f"{PLOT_DIR}"
)

print()

print(
    f"Results saved to:\n"
    f"{RESULTS_OUTPUT_DIR}"
)

print()

print(
    "EXP03 Run01 completed."
)