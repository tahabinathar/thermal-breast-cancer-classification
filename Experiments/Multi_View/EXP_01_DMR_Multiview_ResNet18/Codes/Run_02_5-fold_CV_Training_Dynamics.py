import os
import re
import json
import random
import csv
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_curve
)
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import calculate_temperature_range, preprocess_image


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 10
N_SPLITS = 5

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
THRESHOLD = 0.5

FUSION_HEAD = "small"  # "baseline", "linear", or "small"

EXPERIMENT_NAME = "EXP_01_DMR_Multiview_ResNet18"

DATA_ROOT = (
    r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography\Dataset\DMR_IR_Multi_View_ROI"
)

RESULTS_DIR = (
    r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography\Experiments\Multi_View\EXP_01_DMR_Multiview_ResNet18"
    r"\Results\Run_02_5-fold_CV_Training_Dynamics"
)

# Train and Validation are pooled for cross-validation.
# The separate Test directory is never accessed.

DEVELOPMENT_SPLITS = ("Train", "Validation")

VIEW_FOLDERS = {
    "anterior": "Anterior_ROI",
    "left_oblique": "Left_Oblique_ROI",
    "right_oblique": "Right_Oblique_ROI"
}

VIEW_NAMES = list(VIEW_FOLDERS)
CLASS_NAMES = ["Healthy", "Sick"]
CLASS_TO_LABEL = {"Healthy": 0, "Sick": 1}

HEAD_CONFIGS = {
    "baseline": {
        "run_name": "Run_02_A_Baseline",
        "classifier_name": "1536 -> 64 -> 1",
        "hidden_dim": 64,
        "dropout": 0.3,
        "weight_decay": 0.0
    },
    "linear": {
        "run_name": "Run_02_B_Linear_Fusion",
        "classifier_name": "1536 -> 1",
        "hidden_dim": None,
        "dropout": 0.0,
        "weight_decay": 1e-3
    },
    "small": {
        "run_name": "Run_02_C_Small_Regularized",
        "classifier_name": "1536 -> 16 -> 1",
        "hidden_dim": 16,
        "dropout": 0.5,
        "weight_decay": 1e-3
    }
}

if FUSION_HEAD not in HEAD_CONFIGS:
    raise ValueError(f"Unknown fusion head: {FUSION_HEAD}")

HEAD_CONFIG = HEAD_CONFIGS[FUSION_HEAD]
RUN_NAME = HEAD_CONFIG["run_name"]
RUN_DIR = os.path.join(RESULTS_DIR, RUN_NAME)

FOLD_RESULTS_DIR = os.path.join(RUN_DIR, "fold_results")
CHECKPOINTS_DIR = os.path.join(RUN_DIR, "checkpoints")
PLOTS_DIR = os.path.join(RUN_DIR, "plots")

for directory in (FOLD_RESULTS_DIR, CHECKPOINTS_DIR, PLOTS_DIR):
    os.makedirs(directory, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Run:        {RUN_NAME}")
print(f"Fusion head: {FUSION_HEAD}")
print(f"Device:     {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU:        {torch.cuda.get_device_name(0)}")

print(f"CV:         {N_SPLITS}-fold patient-level stratified")
print(f"Epochs:     {EPOCHS}")
print("Test set:   Not loaded or evaluated")


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 3. BUILD PATIENT GROUPS
# ============================================================

def get_patient_id(filepath):
    match = re.search(
        r"patient(\d+)",
        os.path.basename(filepath),
        flags=re.IGNORECASE
    )

    if match is None:
        raise ValueError(f"Cannot extract patient ID: {filepath}")

    return match.group(1)


def build_patient_groups():
    patients = {}
    patient_sources = {}

    for split in DEVELOPMENT_SPLITS:
        for class_name in CLASS_NAMES:
            label = CLASS_TO_LABEL[class_name]

            for view_name, folder_name in VIEW_FOLDERS.items():
                view_dir = os.path.join(
                    DATA_ROOT, split, class_name, folder_name
                )

                if not os.path.isdir(view_dir):
                    raise FileNotFoundError(view_dir)

                for filename in sorted(os.listdir(view_dir)):
                    if not filename.lower().endswith((".tif", ".tiff")):
                        continue

                    filepath = os.path.join(view_dir, filename)
                    patient_id = get_patient_id(filepath)

                    if patient_id not in patients:
                        patients[patient_id] = {
                            "label": label,
                            "class_name": class_name,
                            "views": {}
                        }
                        patient_sources[patient_id] = set()

                    patient = patients[patient_id]

                    if patient["label"] != label:
                        raise ValueError(
                            f"Patient {patient_id} occurs in both classes."
                        )

                    patient_sources[patient_id].add(split)

                    if len(patient_sources[patient_id]) > 1:
                        raise ValueError(
                            f"Patient {patient_id} occurs in both "
                            "Train and Validation."
                        )

                    if view_name in patient["views"]:
                        raise ValueError(
                            f"Duplicate {view_name} image for patient "
                            f"{patient_id}."
                        )

                    patient["views"][view_name] = filepath

    complete_patients = {}
    incomplete_patients = {}

    for patient_id, patient in patients.items():
        missing = [
            view for view in VIEW_NAMES
            if view not in patient["views"]
        ]

        if missing:
            incomplete_patients[patient_id] = missing
        else:
            complete_patients[patient_id] = patient

    complete_patients = dict(
        sorted(complete_patients.items(), key=lambda item: int(item[0]))
    )

    labels = [p["label"] for p in complete_patients.values()]
    counts = Counter(labels)

    print("\nPOOLED DEVELOPMENT DATA")
    print(f"Patients found:      {len(patients)}")
    print(f"Complete patients:   {len(complete_patients)}")
    print(f"Incomplete excluded: {len(incomplete_patients)}")
    print(f"Healthy:             {counts[0]}")
    print(f"Sick:                {counts[1]}")

    if not complete_patients or len(counts) != 2:
        raise RuntimeError(
            "Complete patients from both classes are required."
        )

    if incomplete_patients:
        print("Excluded patients with missing views:")
        for patient_id, missing in incomplete_patients.items():
            print(f"  {patient_id}: {missing}")

    return complete_patients


# ============================================================
# 4. TRANSFORMS AND DATASET
# ============================================================

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(degrees=18),
    transforms.RandomAffine(degrees=0, scale=(0.95, 1.05)),
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


class MultiViewDataset(Dataset):
    def __init__(self, X, y, transform):
        self.X = X
        self.y = np.asarray(y, dtype=np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        views = []

        for image in self.X[index]:
            if image.ndim == 3 and image.shape[-1] == 1:
                image = image[:, :, 0]

            image = np.asarray(image, dtype=np.float32)
            views.append(self.transform(image))

        label = torch.tensor(self.y[index], dtype=torch.float32)

        return (*views, label)


# ============================================================
# 5. FOLD-SPECIFIC PREPROCESSING
# ============================================================

def load_fold_data(patient_groups, train_ids, val_ids):
    train_paths = [
        patient_groups[pid]["views"][view]
        for pid in train_ids
        for view in VIEW_NAMES
    ]

    # Normalization range is calculated from training images only.
    temp_min, temp_max = calculate_temperature_range(train_paths)

    if temp_max <= temp_min:
        raise ValueError(
            f"Invalid temperature range: {temp_min}, {temp_max}"
        )

    def process(patient_ids):
        X, y = [], []

        for patient_id in patient_ids:
            patient = patient_groups[patient_id]
            views = []

            for view_name in VIEW_NAMES:
                filepath = patient["views"][view_name]
                image = preprocess_image(
                    filepath, temp_min, temp_max, IMAGE_SIZE
                )
                image = np.asarray(image, dtype=np.float32)

                if image.shape == (IMAGE_SIZE, IMAGE_SIZE):
                    image = image[..., np.newaxis]

                expected_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)

                if image.shape != expected_shape:
                    raise ValueError(
                        f"Unexpected shape {image.shape}: {filepath}"
                    )

                if not np.isfinite(image).all():
                    raise ValueError(f"Non-finite image values: {filepath}")

                views.append(image)

            X.append(np.stack(views))
            y.append(patient["label"])

        return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.int64)

    X_train, y_train = process(train_ids)
    X_val, y_val = process(val_ids)

    return X_train, y_train, X_val, y_val, float(temp_min), float(temp_max)


# ============================================================
# 6. MODEL
# ============================================================

class MultiViewResNet18(nn.Module):
    def __init__(self, head_config):
        super().__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.backbone.fc = nn.Identity()

        fused_dim = 512 * len(VIEW_NAMES)

        if head_config["hidden_dim"] is None:
            self.classifier = nn.Linear(fused_dim, 1)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, head_config["hidden_dim"]),
                nn.ReLU(),
                nn.Dropout(head_config["dropout"]),
                nn.Linear(head_config["hidden_dim"], 1)
            )

    def forward(self, anterior, left_oblique, right_oblique):
        features = [
            self.backbone(anterior),
            self.backbone(left_oblique),
            self.backbone(right_oblique)
        ]

        fused = torch.cat(features, dim=1)
        return self.classifier(fused).squeeze(1)


def get_class_weights(y_train):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train
    )

    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def weighted_bce_loss(logits, labels, class_weights):
    sample_weights = class_weights[labels.long()]

    return nn.functional.binary_cross_entropy_with_logits(
        logits, labels, weight=sample_weights
    )


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate_model(model, loader, class_weights):
    model.eval()

    labels_all, probabilities_all = [], []
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for anterior, left, right, labels in loader:
            anterior = anterior.to(DEVICE)
            left = left.to(DEVICE)
            right = right.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(anterior, left, right)
            loss = weighted_bce_loss(logits, labels, class_weights)
            probabilities = torch.sigmoid(logits)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            labels_all.extend(labels.cpu().numpy().astype(int))
            probabilities_all.extend(probabilities.cpu().numpy())

    y_true = np.asarray(labels_all, dtype=int)
    y_prob = np.asarray(probabilities_all, dtype=float)
    y_pred = (y_prob >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    has_both_classes = len(np.unique(y_true)) == 2

    return {
        "loss": float(total_loss / total_samples),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob))
            if has_both_classes else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_prob))
            if has_both_classes else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "labels": y_true,
        "probabilities": y_prob,
        "predictions": y_pred
    }


def training_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= THRESHOLD).astype(int)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities))
            if len(np.unique(labels)) == 2 else float("nan")
    }


# ============================================================
# 8. TRAIN ONE FOLD
# ============================================================

def train_one_fold(fold_number, patient_groups, train_ids, val_ids):
    fold_name = f"Fold_{fold_number:02d}"
    fold_dir = os.path.join(FOLD_RESULTS_DIR, fold_name)
    checkpoint_dir = os.path.join(CHECKPOINTS_DIR, fold_name)

    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed = SEED
    set_seed(SEED)

    print(f"\n{fold_name}: {FUSION_HEAD}\n")

    with open(
        os.path.join(fold_dir, "fold_patient_split.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump({
            "fold": fold_number,
            "seed": seed,
            "training_patient_ids": train_ids,
            "validation_patient_ids": val_ids,
            "training_patients": len(train_ids),
            "validation_patients": len(val_ids),
            "test_loaded": False,
            "test_evaluated": False
        }, file, indent=4)

    (
        X_train, y_train, X_val, y_val, temp_min, temp_max
    ) = load_fold_data(patient_groups, train_ids, val_ids)

    print(
        f"Patients: train={len(y_train)}, validation={len(y_val)} | "
        f"Train H/S={sum(y_train == 0)}/{sum(y_train == 1)} | "
        f"Val H/S={sum(y_val == 0)}/{sum(y_val == 1)}"
    )
    print(f"Temperature range: {temp_min:.4f}–{temp_max:.4f} °C")

    train_dataset = MultiViewDataset(X_train, y_train, train_transform)
    val_dataset = MultiViewDataset(X_val, y_val, val_transform)

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    model = MultiViewResNet18(HEAD_CONFIG).to(DEVICE)
    class_weights = get_class_weights(y_train)

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters())

    print(f"Parameters trainable/total: {trainable:,}/{total:,}")
    print(
        f"Class weights H/S: "
        f"{class_weights[0].item():.4f}/{class_weights[1].item():.4f}"
    )

    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
        weight_decay=HEAD_CONFIG["weight_decay"]
    )

    history = {
        "loss": [],
        "accuracy": [],
        "auc": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_auc": [],
        "val_average_precision": []
    }

    # Train all epochs by design. Evaluate the final epoch.
    for epoch in range(1, EPOCHS + 1):
        model.train()
        model.backbone.eval()
        model.classifier.train()

        epoch_loss, n_samples = 0.0, 0
        train_labels, train_probs = [], []

        for anterior, left, right, labels in train_loader:
            anterior = anterior.to(DEVICE)
            left = left.to(DEVICE)
            right = right.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(anterior, left, right)
            loss = weighted_bce_loss(logits, labels, class_weights)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size
            n_samples += batch_size

            train_labels.extend(labels.detach().cpu().numpy().astype(int))
            train_probs.extend(torch.sigmoid(logits.detach()).cpu().numpy())

        train_loss = epoch_loss / n_samples
        train_stats = training_metrics(train_labels, train_probs)

        # One validation evaluation per epoch.
        val_stats = evaluate_model(model, val_loader, class_weights)

        history["loss"].append(float(train_loss))
        history["accuracy"].append(train_stats["accuracy"])
        history["auc"].append(train_stats["auc"])
        history["val_loss"].append(val_stats["loss"])
        history["val_accuracy"].append(val_stats["accuracy"])
        history["val_auc"].append(val_stats["auc"])
        history["val_average_precision"].append(
            val_stats["average_precision"]
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Loss {train_loss:.4f} | "
            f"Train AUC {train_stats['auc']:.4f} | "
            f"Val Loss {val_stats['loss']:.4f} | "
            f"Val Acc {val_stats['accuracy']:.4f} | "
            f"Val AUC {val_stats['auc']:.4f}"
        )

    # val_stats is already the final-epoch evaluation.
    with open(
        os.path.join(fold_dir, "training_history.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(history, file, indent=4, allow_nan=True)

    metric_names = [
        "loss", "accuracy", "auc", "average_precision",
        "balanced_accuracy", "precision", "recall", "specificity",
        "f1", "mcc", "tn", "fp", "fn", "tp"
    ]

    final_metrics = {key: val_stats[key] for key in metric_names}

    fold_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "run_name": RUN_NAME,
        "fusion_head": FUSION_HEAD,
        "classifier": HEAD_CONFIG["classifier_name"],
        "fold": fold_number,
        "epochs_trained": EPOCHS,
        "seed": seed,
        "training_patients": len(train_ids),
        "validation_patients": len(val_ids),
        "temperature_min": temp_min,
        "temperature_max": temp_max,
        "threshold": THRESHOLD,
        "final_epoch_validation": final_metrics,
        "test_loaded": False,
        "test_evaluated": False
    }

    with open(
        os.path.join(fold_dir, "fold_summary.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(fold_summary, file, indent=4, allow_nan=True)

    torch.save({
        "experiment_name": EXPERIMENT_NAME,
        "run_name": RUN_NAME,
        "fusion_head": FUSION_HEAD,
        "fold": fold_number,
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "head_config": HEAD_CONFIG,
        "seed": seed,
        "temperature_min": temp_min,
        "temperature_max": temp_max,
        "image_size": IMAGE_SIZE,
        "threshold": THRESHOLD,
        "training_patient_ids": train_ids,
        "validation_patient_ids": val_ids
    }, os.path.join(checkpoint_dir, "final_epoch.pth"))

    # Training curves
    epochs_axis = np.arange(1, EPOCHS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, train_key, val_key, title in [
        (axes[0], "loss", "val_loss", "Loss"),
        (axes[1], "accuracy", "val_accuracy", "Accuracy"),
        (axes[2], "auc", "val_auc", "ROC-AUC")
    ]:
        ax.plot(epochs_axis, history[train_key], label="Train")
        ax.plot(epochs_axis, history[val_key], label="Validation")
        ax.set(title=title, xlabel="Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{RUN_NAME} - {fold_name}")
    fig.tight_layout()
    fig.savefig(
        os.path.join(PLOTS_DIR, f"{fold_name}_training_curves.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Final-epoch validation ROC curve
    y_true = val_stats["labels"]
    y_prob = val_stats["probabilities"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {val_stats['auc']:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"{RUN_NAME} - {fold_name} Validation ROC"
    )
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(PLOTS_DIR, f"{fold_name}_validation_roc.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(
        f"{fold_name} final validation: "
        f"AUC={val_stats['auc']:.4f}, "
        f"AP={val_stats['average_precision']:.4f}, "
        f"Balanced Acc={val_stats['balanced_accuracy']:.4f}"
    )

    return fold_summary


# ============================================================
# 9. RUN FIVE-FOLD CROSS-VALIDATION
# ============================================================

def main():
    patient_groups = build_patient_groups()
    patient_ids = list(patient_groups)

    labels = np.asarray(
        [patient_groups[pid]["label"] for pid in patient_ids],
        dtype=int
    )

    class_counts = Counter(labels.tolist())

    if min(class_counts.values()) < N_SPLITS:
        raise ValueError(
            f"Each class needs at least {N_SPLITS} patients."
        )

    print(
        f"\nDevelopment patients: {len(patient_ids)} | "
        f"Healthy: {class_counts[0]} | Sick: {class_counts[1]}"
    )

    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    fold_summaries = []

    for fold_number, (train_idx, val_idx) in enumerate(
        splitter.split(np.zeros(len(patient_ids)), labels),
        start=1
    ):
        train_ids = [patient_ids[i] for i in train_idx]
        val_ids = [patient_ids[i] for i in val_idx]

        # Patient-level leakage check.
        if set(train_ids) & set(val_ids):
            raise RuntimeError(f"Patient overlap in Fold {fold_number}")

        fold_summaries.append(
            train_one_fold(
                fold_number, patient_groups, train_ids, val_ids
            )
        )

    # Summarize final-epoch metrics across folds.
    metric_names = [
        "loss", "accuracy", "auc", "average_precision",
        "balanced_accuracy", "precision", "recall", "specificity",
        "f1", "mcc"
    ]

    cv_metrics = {}

    for metric in metric_names:
        values = np.asarray([
            fold["final_epoch_validation"][metric]
            for fold in fold_summaries
        ], dtype=float)

        valid = values[np.isfinite(values)]

        cv_metrics[metric] = {
            "fold_values": values.tolist(),
            "mean": float(np.mean(valid)) if len(valid) else None,
            "std": float(np.std(valid, ddof=1))
                if len(valid) > 1 else 0.0 if len(valid) == 1 else None
        }

    run_summary = {
        "experiment_name": EXPERIMENT_NAME,
        "run_name": RUN_NAME,
        "fusion_head": FUSION_HEAD,
        "classifier": HEAD_CONFIG["classifier_name"],
        "head_config": HEAD_CONFIG,
        "development_splits": list(DEVELOPMENT_SPLITS),
        "n_development_patients": len(patient_ids),
        "n_healthy": int((labels == 0).sum()),
        "n_sick": int((labels == 1).sum()),
        "n_splits": N_SPLITS,
        "random_state": SEED,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "threshold": THRESHOLD,
        "optimizer": "Adam",
        "backbone": "Frozen ImageNet-pretrained ResNet-18",
        "fusion": "Concatenation of three 512-dimensional features",
        "metrics_final_epoch_cv": cv_metrics,
        "fold_summaries": fold_summaries,
        "test_loaded": False,
        "test_evaluated": False
    }

    with open(
        os.path.join(RUN_DIR, "run_summary.json"),
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(run_summary, file, indent=4, allow_nan=True)

    csv_path = os.path.join(RUN_DIR, "cross_validation_results.csv")
    csv_fields = [
        "fold", "training_patients", "validation_patients",
        *metric_names, "tn", "fp", "fn", "tp"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()

        for fold in fold_summaries:
            metrics = fold["final_epoch_validation"]
            writer.writerow({
                "fold": fold["fold"],
                "training_patients": fold["training_patients"],
                "validation_patients": fold["validation_patients"],
                **{key: metrics[key] for key in metric_names},
                **{key: metrics[key] for key in ("tn", "fp", "fn", "tp")}
            })

    print(f"RUN COMPLETE: {RUN_NAME}")
    print("Final-epoch 5-fold CV results (mean ± SD)")

    for metric, result in cv_metrics.items():
        if result["mean"] is not None:
            print(
                f"{metric:20s}: "
                f"{result['mean']:.4f} ± {result['std']:.4f}"
            )

    print(f"\nSummary: {os.path.join(RUN_DIR, 'run_summary.json')}")
    print(f"CSV:     {csv_path}")
    print("Separate Test set was not loaded or evaluated.")

if __name__ == "__main__":
    main()