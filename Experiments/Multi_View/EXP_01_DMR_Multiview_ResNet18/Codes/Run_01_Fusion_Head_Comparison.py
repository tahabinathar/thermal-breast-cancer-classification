import os
import re
import json
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

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

from preprocessing import (
    calculate_temperature_range,
    preprocess_image
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 10
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
THRESHOLD = 0.5

# Select one:
# "baseline" = 1536 -> 64 -> 1, dropout 0.3, no weight decay
# "linear"   = 1536 -> 1, L2 weight decay 0.001
# "small"    = 1536 -> 16 -> 1, dropout 0.5, weight decay 0.001

FUSION_HEAD = "small"

EXPERIMENT_NAME = "EXP_01_DMR_Multiview_ResNet18"

DATA_ROOT = (
    r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography"
    r"\Dataset\DMR_IR_Multi_View_ROI"
)

RESULTS_DIR = (
    r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography"
    r"\Experiments\Multi_View\EXP_01_DMR_Multiview_ResNet18"
    r"\Results\Run_01_Fusion_Head_Comparison"
)

HEAD_CONFIGS = {
    "baseline": {
        "run_name": "Run_01_A_Baseline",
        "classifier_name": "1536 -> 64 -> 1",
        "hidden_dim": 64,
        "dropout": 0.3,
        "weight_decay": 0.0
    },
    "linear": {
        "run_name": "Run_01_B_Linear_Fusion",
        "classifier_name": "1536 -> 1",
        "hidden_dim": None,
        "dropout": 0.0,
        "weight_decay": 1e-3
    },
    "small": {
        "run_name": "Run_01_C_Small_Regularized",
        "classifier_name": "1536 -> 16 -> 1",
        "hidden_dim": 16,
        "dropout": 0.5,
        "weight_decay": 1e-3
    }
}

if FUSION_HEAD not in HEAD_CONFIGS:
    raise ValueError(
        f"Invalid FUSION_HEAD={FUSION_HEAD!r}. "
        f"Choose one of {list(HEAD_CONFIGS)}."
    )

CONFIG = HEAD_CONFIGS[FUSION_HEAD]

RUN_NAME = CONFIG["run_name"]
DROPOUT_RATE = CONFIG["dropout"]
WEIGHT_DECAY = CONFIG["weight_decay"]

RUN_DIR = os.path.join(RESULTS_DIR, RUN_NAME)
RESULTS_OUTPUT_DIR = os.path.join(RUN_DIR, "results")
CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
PLOT_DIR = os.path.join(RUN_DIR, "plots")

CLASS_NAMES = ["Healthy", "Sick"]
CLASS_TO_LABEL = {"Healthy": 0, "Sick": 1}

VIEW_FOLDERS = {
    "anterior": "Anterior_ROI",
    "left_oblique": "Left_Oblique_ROI",
    "right_oblique": "Right_Oblique_ROI"
}

VIEW_NAMES = list(VIEW_FOLDERS.keys())

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# INITIALIZATION AND OUTPUT FOLDERS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

for directory in (
    RESULTS_OUTPUT_DIR,
    CHECKPOINT_DIR,
    PLOT_DIR
):
    os.makedirs(directory, exist_ok=True)


# ============================================================
# PATIENT GROUPING
# ============================================================

def get_patient_id(filepath):
    filename = os.path.basename(filepath)

    match = re.search(
        r"patient(\d+)",
        filename,
        re.IGNORECASE
    )

    if match is None:
        raise ValueError(
            f"Could not extract patient ID from filename: {filename}"
        )

    return match.group(1)


def build_patient_groups(split):
    patient_groups = {}

    for class_name in CLASS_NAMES:
        label = CLASS_TO_LABEL[class_name]

        for view_name, view_folder in VIEW_FOLDERS.items():
            view_dir = os.path.join(
                DATA_ROOT,
                split,
                class_name,
                view_folder
            )

            if not os.path.isdir(view_dir):
                raise FileNotFoundError(
                    f"Required directory not found: {view_dir}"
                )

            for filename in sorted(os.listdir(view_dir)):
                if not filename.lower().endswith(
                    (".tif", ".tiff")
                ):
                    continue

                filepath = os.path.join(view_dir, filename)
                patient_id = get_patient_id(filepath)

                if patient_id not in patient_groups:
                    patient_groups[patient_id] = {
                        "label": label,
                        "class_name": class_name,
                        "views": {}
                    }

                patient_data = patient_groups[patient_id]

                if patient_data["label"] != label:
                    raise ValueError(
                        f"Patient {patient_id} appears in multiple "
                        f"classes within {split}."
                    )

                if view_name in patient_data["views"]:
                    raise ValueError(
                        f"Duplicate {view_name} image for patient "
                        f"{patient_id} in {split}."
                    )

                patient_data["views"][view_name] = filepath

    return patient_groups


def get_complete_patients(patient_groups):
    return {
        patient_id: patient_data
        for patient_id, patient_data in patient_groups.items()
        if all(
            view_name in patient_data["views"]
            for view_name in VIEW_NAMES
        )
    }


def report_patient_completeness(split, patient_groups):
    complete = get_complete_patients(patient_groups)

    incomplete = {
        patient_id: sorted(
            set(VIEW_NAMES) - set(patient_data["views"])
        )
        for patient_id, patient_data in patient_groups.items()
        if patient_id not in complete
    }

    print(f"\n{split} patient grouping:")
    print(f"  Patients discovered: {len(patient_groups)}")
    print(f"  Complete patients:   {len(complete)}")
    print(f"  Incomplete patients: {len(incomplete)}")

    if incomplete:
        print("  Excluded patients with missing views:")

        for patient_id, missing_views in sorted(
            incomplete.items(),
            key=lambda item: int(item[0])
        ):
            print(
                f"    Patient {patient_id}: "
                f"{', '.join(missing_views)}"
            )

    return complete


train_groups_all = build_patient_groups("Train")
val_groups_all = build_patient_groups("Validation")

train_groups = report_patient_completeness(
    "Train",
    train_groups_all
)

val_groups = report_patient_completeness(
    "Validation",
    val_groups_all
)

train_patient_ids = sorted(train_groups.keys(), key=int)
val_patient_ids = sorted(val_groups.keys(), key=int)

if not train_patient_ids:
    raise ValueError(
        "No complete training patients with all three views were found."
    )

if not val_patient_ids:
    raise ValueError(
        "No complete validation patients with all three views were found."
    )

overlapping_patients = set(train_patient_ids) & set(val_patient_ids)

if overlapping_patients:
    raise ValueError(
        "Patient leakage detected between Train and Validation: "
        + ", ".join(sorted(overlapping_patients, key=int))
    )

print("\nPatient split check:")
print(f"  Complete training patients:   {len(train_patient_ids)}")
print(f"  Complete validation patients: {len(val_patient_ids)}")
print(f"  Train/Validation overlap:      {len(overlapping_patients)}")


# ============================================================
# TEMPERATURE RANGE: TRAINING IMAGES ONLY
# ============================================================

train_paths = [
    train_groups[patient_id]["views"][view_name]
    for patient_id in train_patient_ids
    for view_name in VIEW_NAMES
]

GLOBAL_MIN, GLOBAL_MAX = calculate_temperature_range(train_paths)

if (
    not np.isfinite(GLOBAL_MIN)
    or not np.isfinite(GLOBAL_MAX)
    or GLOBAL_MAX <= GLOBAL_MIN
):
    raise ValueError(
        f"Invalid temperature range: {GLOBAL_MIN} to {GLOBAL_MAX}"
    )

print("\nTemperature normalization:")
print(f"  Training images used: {len(train_paths)}")
print(f"  Global minimum:       {GLOBAL_MIN:.8f}")
print(f"  Global maximum:       {GLOBAL_MAX:.8f}")
print("  Range source: training images only")
print("  Validation used for range calculation: No")
print("  Test used for range calculation: No")


# ============================================================
# LOAD AND PREPROCESS IMAGES
# ============================================================

def load_split(split, patient_groups, patient_ids):
    images = []
    labels = []

    for patient_id in patient_ids:
        patient_data = patient_groups[patient_id]
        patient_views = []

        for view_name in VIEW_NAMES:
            filepath = patient_data["views"][view_name]

            image = preprocess_image(
                filepath,
                GLOBAL_MIN,
                GLOBAL_MAX,
                IMAGE_SIZE
            )

            image = np.asarray(image, dtype=np.float32)

            if image.ndim == 2:
                image = image[..., np.newaxis]

            if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 1):
                raise ValueError(
                    f"Unexpected image shape {image.shape}: {filepath}"
                )

            if not np.isfinite(image).all():
                raise ValueError(
                    f"NaN or infinite values found in {filepath}"
                )

            if image.min() < -1e-6 or image.max() > 1.000001:
                raise ValueError(
                    f"Image values outside [0, 1] in {filepath}: "
                    f"min={image.min()}, max={image.max()}"
                )

            patient_views.append(image)

        images.append(np.stack(patient_views, axis=0))
        labels.append(patient_data["label"])

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    print(f"\n{split} data:")
    print(f"  Image array shape: {X.shape}")
    print(f"  Label array shape: {y.shape}")

    return X, y


X_train, y_train = load_split(
    "Train",
    train_groups,
    train_patient_ids
)

X_val, y_val = load_split(
    "Validation",
    val_groups,
    val_patient_ids
)


# ============================================================
# CLASS DISTRIBUTIONS
# ============================================================

def print_class_distribution(split_name, labels):
    counts = {
        name: int(np.sum(labels == label))
        for name, label in CLASS_TO_LABEL.items()
    }

    total = len(labels)

    print(f"\n{split_name} class distribution:")
    print(f"  Healthy: {counts['Healthy']}")
    print(f"  Sick:    {counts['Sick']}")
    print(f"  Total:   {total}")

    if total:
        print(
            f"  Healthy percentage: "
            f"{100 * counts['Healthy'] / total:.2f}%"
        )
        print(
            f"  Sick percentage: "
            f"{100 * counts['Sick'] / total:.2f}%"
        )

    return counts


train_class_counts = print_class_distribution("Training", y_train)
val_class_counts = print_class_distribution("Validation", y_val)

if len(np.unique(y_train)) != 2:
    raise ValueError("Training data must contain both classes.")

if len(np.unique(y_val)) != 2:
    raise ValueError(
        "Validation data must contain both classes to calculate ROC-AUC."
    )


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(degrees=18),
    transforms.RandomAffine(
        degrees=0,
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# MULTI-VIEW DATASET
# ============================================================

class MultiViewDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        views = []

        for view_index in range(len(VIEW_NAMES)):
            image = self.images[index, view_index]

            if image.ndim == 3 and image.shape[-1] == 1:
                image = image[..., 0]

            image = np.clip(image, 0.0, 1.0)
            image = np.round(image * 255.0).astype(np.uint8)

            views.append(self.transform(image))

        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return (*views, label)


train_dataset = MultiViewDataset(
    X_train, y_train, train_transform
)

val_dataset = MultiViewDataset(
    X_val, y_val, val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_weight_values = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train
)

class_weights = torch.tensor(
    class_weight_values,
    dtype=torch.float32,
    device=DEVICE
)

print("\nClass weights:")
print(f"  Healthy (0): {class_weights[0].item():.6f}")
print(f"  Sick (1):    {class_weights[1].item():.6f}")


# ============================================================
# MULTI-VIEW RESNET-18
# ============================================================

class MultiViewResNet18(nn.Module):
    def __init__(self, head_config):
        super().__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        fused_dim = num_features * len(VIEW_NAMES)

        if FUSION_HEAD == "linear":
            self.classifier = nn.Linear(fused_dim, 1)

        else:
            hidden_dim = head_config["hidden_dim"]
            dropout = head_config["dropout"]

            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, anterior, left_oblique, right_oblique):
        anterior_features = self.backbone(anterior)
        left_features = self.backbone(left_oblique)
        right_features = self.backbone(right_oblique)

        fused_features = torch.cat(
            [
                anterior_features,
                left_features,
                right_features
            ],
            dim=1
        )

        return self.classifier(fused_features).squeeze(1)


model = MultiViewResNet18(CONFIG).to(DEVICE)

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "resnet18_multiview_best.pth"
)


# ============================================================
# MODEL SUMMARY
# ============================================================

total_params = sum(p.numel() for p in model.parameters())

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

frozen_params = total_params - trainable_params

print("\nMULTI-VIEW RESNET-18 TRAINING")
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Run:        {RUN_NAME}")
print(f"Fusion head: {FUSION_HEAD}")
print(f"Device:     {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU:        {torch.cuda.get_device_name(0)}")

print(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Maximum epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Dropout: {DROPOUT_RATE}")
print(f"Weight decay: {WEIGHT_DECAY}")
print(f"Classification threshold: {THRESHOLD}")
print(f"Views: {', '.join(VIEW_NAMES)}")
print("Backbone: Shared ImageNet-pretrained ResNet-18 (frozen)")
print("Fusion: Feature concatenation")
print(f"Classifier: {CONFIG['classifier_name']}")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters:    {frozen_params:,}")
print(f"Checkpoint: {checkpoint_path}")
print(f"Results:    {RESULTS_OUTPUT_DIR}")
print(f"Plots:      {PLOT_DIR}")


# ============================================================
# LOSS FUNCTION
# ============================================================

def calculate_weighted_training_loss(logits, labels):
    sample_weights = class_weights[labels.long()]

    return nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        weight=sample_weights
    )


# ============================================================
# TRAINING AND VALIDATION
# ============================================================

history = {
    "loss": [],
    "accuracy": [],
    "auc": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_auc": []
}

best_val_auc = -1.0
best_epoch = 0

for epoch in range(EPOCHS):
    model.train()

    # Keep the frozen backbone and its BatchNorm layers in eval mode.
    model.backbone.eval()
    model.classifier.train()

    train_loss_total = 0.0
    train_probs = []
    train_targets = []

    for anterior, left, right, labels in train_loader:
        anterior = anterior.to(DEVICE)
        left = left.to(DEVICE)
        right = right.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(anterior, left, right)
        loss = calculate_weighted_training_loss(logits, labels)

        loss.backward()
        optimizer.step()

        train_loss_total += loss.item() * labels.size(0)

        train_probs.extend(
            torch.sigmoid(logits).detach().cpu().numpy()
        )
        train_targets.extend(labels.detach().cpu().numpy())

    train_loss = train_loss_total / len(train_dataset)
    train_probs = np.asarray(train_probs)
    train_targets = np.asarray(train_targets)

    train_preds = (train_probs >= THRESHOLD).astype(int)

    train_accuracy = accuracy_score(train_targets, train_preds)
    train_auc = roc_auc_score(train_targets, train_probs)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0
    val_probs = []
    val_targets = []

    with torch.no_grad():
        for anterior, left, right, labels in val_loader:
            anterior = anterior.to(DEVICE)
            left = left.to(DEVICE)
            right = right.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(anterior, left, right)

            # Unweighted validation loss.
            val_loss = nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels
            )

            val_loss_total += val_loss.item() * labels.size(0)

            val_probs.extend(
                torch.sigmoid(logits).cpu().numpy()
            )
            val_targets.extend(labels.cpu().numpy())

    val_loss = val_loss_total / len(val_dataset)
    val_probs = np.asarray(val_probs)
    val_targets = np.asarray(val_targets)

    val_preds = (val_probs >= THRESHOLD).astype(int)

    val_accuracy = accuracy_score(val_targets, val_preds)
    val_auc = roc_auc_score(val_targets, val_probs)

    history["loss"].append(float(train_loss))
    history["accuracy"].append(float(train_accuracy))
    history["auc"].append(float(train_auc))
    history["val_loss"].append(float(val_loss))
    history["val_accuracy"].append(float(val_accuracy))
    history["val_auc"].append(float(val_auc))

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"Loss {train_loss:.4f} | "
        f"Acc {train_accuracy:.4f} | "
        f"AUC {train_auc:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_accuracy:.4f} | "
        f"Val AUC {val_auc:.4f}"
    )


# ============================================================
# LOAD BEST CHECKPOINT AND EVALUATE VALIDATION
# ============================================================

if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(
        f"Best-model checkpoint was not created: {checkpoint_path}"
    )

model.load_state_dict(
    torch.load(checkpoint_path, map_location=DEVICE)
)

model.eval()

val_probs = []
val_targets = []

with torch.no_grad():
    for anterior, left, right, labels in val_loader:
        logits = model(
            anterior.to(DEVICE),
            left.to(DEVICE),
            right.to(DEVICE)
        )

        val_probs.extend(torch.sigmoid(logits).cpu().numpy())
        val_targets.extend(labels.numpy())

val_probs = np.asarray(val_probs)
val_targets = np.asarray(val_targets).astype(int)
val_preds = (val_probs >= THRESHOLD).astype(int)

tn, fp, fn, tp = confusion_matrix(
    val_targets,
    val_preds,
    labels=[0, 1]
).ravel()

val_auc = roc_auc_score(val_targets, val_probs)
val_pr_auc = average_precision_score(val_targets, val_probs)

metrics = {
    "fusion_head": FUSION_HEAD,
    "classifier": CONFIG["classifier_name"],
    "roc_auc": float(val_auc),
    "pr_auc": float(val_pr_auc),
    "threshold": THRESHOLD,
    "accuracy": float(accuracy_score(val_targets, val_preds)),
    "balanced_accuracy": float(
        balanced_accuracy_score(val_targets, val_preds)
    ),
    "sensitivity": float(
        recall_score(val_targets, val_preds, zero_division=0)
    ),
    "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
    "precision": float(
        precision_score(val_targets, val_preds, zero_division=0)
    ),
    "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
    "f1": float(
        f1_score(val_targets, val_preds, zero_division=0)
    ),
    "mcc": float(matthews_corrcoef(val_targets, val_preds)),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp)
}


# ============================================================
# SAVE TRAINING HISTORY AND VALIDATION METRICS
# ============================================================

def save_json(filepath, content):
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=4)


save_json(
    os.path.join(RESULTS_OUTPUT_DIR, "training_history.json"),
    history
)

save_json(
    os.path.join(RESULTS_OUTPUT_DIR, "validation_metrics.json"),
    metrics
)


# ============================================================
# SAVE VALIDATION ROC DATA
# ============================================================

fpr, tpr, roc_thresholds = roc_curve(val_targets, val_probs)

roc_data = {
    "fpr": fpr.tolist(),
    "tpr": tpr.tolist(),
    "thresholds": roc_thresholds.tolist(),
    "roc_auc": float(val_auc)
}

save_json(
    os.path.join(RESULTS_OUTPUT_DIR, "validation_roc_data.json"),
    roc_data
)


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

validation_predictions = {
    "patient_ids": val_patient_ids,
    "true_labels": val_targets.tolist(),
    "predicted_probabilities": val_probs.tolist(),
    "predicted_labels": val_preds.tolist(),
    "threshold": THRESHOLD
}

save_json(
    os.path.join(
        RESULTS_OUTPUT_DIR,
        "validation_predictions.json"
    ),
    validation_predictions
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

summary = {
    "experiment_name": EXPERIMENT_NAME,
    "run_name": RUN_NAME,
    "fusion_head": FUSION_HEAD,
    "seed": SEED,
    "backbone": "ImageNet-pretrained ResNet-18",
    "backbone_frozen": True,
    "views": VIEW_NAMES,
    "fusion": "Feature concatenation",
    "feature_dimension_per_view": 512,
    "fused_feature_dimension": 1536,
    "classifier": CONFIG["classifier_name"],
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "maximum_epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "dropout": DROPOUT_RATE,
    "weight_decay": WEIGHT_DECAY,
    "threshold": THRESHOLD,
    "global_temperature_min": float(GLOBAL_MIN),
    "global_temperature_max": float(GLOBAL_MAX),
    "temperature_range_source": "Training images only",
    "train_patient_count": len(train_patient_ids),
    "validation_patient_count": len(val_patient_ids),
    "train_class_counts": train_class_counts,
    "validation_class_counts": val_class_counts,
    "class_weights": {
        "Healthy": float(class_weights[0].item()),
        "Sick": float(class_weights[1].item())
    },
    "total_parameters": int(total_params),
    "trainable_parameters": int(trainable_params),
    "frozen_parameters": int(frozen_params),
    "best_epoch": int(best_epoch),
    "best_validation_auc": float(best_val_auc),
    "validation_metrics": metrics,
    "test_loaded": False,
    "test_evaluated": False
}

save_json(
    os.path.join(RESULTS_OUTPUT_DIR, "training_summary.json"),
    summary
)


# ============================================================
# TRAINING PLOTS
# ============================================================

epoch_numbers = range(1, len(history["loss"]) + 1)

plot_configurations = [
    (
        "loss",
        "Training and Validation Loss",
        "Loss",
        "training_validation_loss.png"
    ),
    (
        "accuracy",
        "Training and Validation Accuracy",
        "Accuracy",
        "training_validation_accuracy.png"
    ),
    (
        "auc",
        "Training and Validation ROC-AUC",
        "ROC-AUC",
        "training_validation_auc.png"
    )
]

for key, title, ylabel, filename in plot_configurations:
    plt.figure(figsize=(9, 6))

    plt.plot(
        epoch_numbers,
        history[key],
        label="Training"
    )

    plt.plot(
        epoch_numbers,
        history[f"val_{key}"],
        label="Validation"
    )

    if key == "auc":
        plt.axvline(
            best_epoch,
            linestyle="--",
            label=f"Best Epoch: {best_epoch}"
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{title} - {FUSION_HEAD}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOT_DIR, filename),
        dpi=300
    )

    plt.close()


# ============================================================
# VALIDATION ROC CURVE
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"{CONFIG['classifier_name']} (AUC = {val_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Validation ROC Curve - {FUSION_HEAD}")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(PLOT_DIR, "validation_roc_curve.png"),
    dpi=300
)

plt.close()


# ============================================================
# MODEL SUMMARY FILE
# ============================================================

with open(
    os.path.join(RESULTS_OUTPUT_DIR, "model_summary.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(f"Experiment: {EXPERIMENT_NAME}\n")
    file.write(f"Run: {RUN_NAME}\n")
    file.write(f"Fusion head: {FUSION_HEAD}\n")
    file.write(f"Classifier: {CONFIG['classifier_name']}\n")
    file.write(f"Total parameters: {total_params}\n")
    file.write(f"Trainable parameters: {trainable_params}\n")
    file.write(f"Frozen parameters: {frozen_params}\n")
    file.write(
        "Backbone: Shared frozen ImageNet-pretrained ResNet-18\n"
    )
    file.write(
        "Views: Anterior, Left-oblique, Right-oblique\n"
    )
    file.write("Fusion: Feature concatenation\n")
    file.write(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}\n")
    file.write(f"Batch size: {BATCH_SIZE}\n")
    file.write(f"Maximum epochs: {EPOCHS}\n")
    file.write(f"Learning rate: {LEARNING_RATE}\n")
    file.write(f"Dropout: {DROPOUT_RATE}\n")
    file.write(f"Weight decay: {WEIGHT_DECAY}\n")
    file.write(f"Temperature minimum: {GLOBAL_MIN:.8f}\n")
    file.write(f"Temperature maximum: {GLOBAL_MAX:.8f}\n")
    file.write(f"Best epoch: {best_epoch}\n")
    file.write(f"Best validation ROC-AUC: {best_val_auc:.6f}\n")


# ============================================================
# FINAL TERMINAL REPORT
# ============================================================

print("\nTRAINING COMPLETED")

print(f"Experiment:           {EXPERIMENT_NAME}")
print(f"Fusion head:          {FUSION_HEAD}")
print(f"Classifier:            {CONFIG['classifier_name']}")
print(f"Weight decay:          {WEIGHT_DECAY}")
print(f"Best epoch:            {best_epoch}")
print(f"Best validation AUC:   {best_val_auc:.4f}")
print(f"Validation ROC-AUC:    {val_auc:.4f}")
print(f"Validation PR-AUC:     {val_pr_auc:.4f}")
print(f"Threshold:             {THRESHOLD:.2f}")

print("\nValidation metrics:")
print(f"  Accuracy:           {metrics['accuracy']:.4f}")
print(f"  Balanced accuracy:  {metrics['balanced_accuracy']:.4f}")
print(f"  Sensitivity:        {metrics['sensitivity']:.4f}")
print(f"  Specificity:        {metrics['specificity']:.4f}")
print(f"  Precision:          {metrics['precision']:.4f}")
print(f"  NPV:                {metrics['npv']:.4f}")
print(f"  F1-score:           {metrics['f1']:.4f}")
print(f"  MCC:                {metrics['mcc']:.4f}")

print("\nConfusion matrix:")
print("                 Predicted")
print("                Healthy  Sick")
print(f"Actual Healthy    {tn:4d}   {fp:4d}")
print(f"Actual Sick       {fn:4d}   {tp:4d}")

print("\nSaved outputs:")
print(f"  Best model:       {checkpoint_path}")
print(f"  Results:          {RESULTS_OUTPUT_DIR}")
print(f"  Plots:            {PLOT_DIR}")

print("\nTest set was not loaded or evaluated.")