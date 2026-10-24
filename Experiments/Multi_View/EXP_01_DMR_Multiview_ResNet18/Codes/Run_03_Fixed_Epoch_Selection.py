import os
import re
import csv
import json
import gc
import random

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
    preprocess_image,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 10
IMAGE_SIZE = 224
BATCH_SIZE = 16
MAX_EPOCHS = 100
N_FOLDS = 5
THRESHOLD = 0.5

VIEWS = [
    "Anterior_ROI",
    "Left_Oblique_ROI",
    "Right_Oblique_ROI",
]

CLASSES = {
    "Healthy": 0,
    "Sick": 1,
}

IMAGE_EXTENSIONS = {".tif", ".tiff"}

EXPERIMENT_NAME = "EXP_01_DMR_Multiview_ResNet18"

EXPERIMENT_DIR = (
    r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography"
    r"\Experiments\Multi_View\EXP_01_DMR_Multiview_ResNet18"
)

DATA_ROOT = (
    r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography"
    r"\Dataset\DMR_IR_Multi_View_ROI"
)

RESULTS_DIR = os.path.join(
    EXPERIMENT_DIR,
    "Results",
)

RUN02_DIR = os.path.join(
    RESULTS_DIR,
    "Run_02_5-fold_CV_Training_Dynamics",
)

RUN02_BASELINE_FOLD_RESULTS_DIR = os.path.join(
    RUN02_DIR,
    "Run_02_A_Baseline",
    "fold_results",
)

RUN03_DIR = os.path.join(
    RESULTS_DIR,
    "Run_03_Fixed_Epoch_Selection",
)

os.makedirs(RUN03_DIR, exist_ok=True)

HEAD_CONFIGS = {
    "baseline": {
        "run_name": "Run_03_A_Baseline",
        "hidden_dim": 64,
        "dropout": 0.3,
        "weight_decay": 0.0,
        "learning_rate": 1e-3,
    },
    "linear": {
        "run_name": "Run_03_B_Linear_Fusion",
        "hidden_dim": None,
        "dropout": 0.0,
        "weight_decay": 1e-3,
        "learning_rate": 1e-3,
    },
    "small": {
        "run_name": "Run_03_C_Small_Regularized",
        "hidden_dim": 16,
        "dropout": 0.5,
        "weight_decay": 1e-3,
        "learning_rate": 1e-3,
    },
}

METRICS = [
    "roc_auc",
    "average_precision",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "mcc",
]

for config in HEAD_CONFIGS.values():
    head_dir = os.path.join(
        RUN03_DIR,
        config["run_name"],
    )

    for subfolder in [
        "fold_results",
        "checkpoints",
        "plots",
    ]:
        os.makedirs(
            os.path.join(head_dir, subfolder),
            exist_ok=True,
        )

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Experiment: {EXPERIMENT_NAME}")
print("Run: Run_03_Fixed_Epoch_Selection")
print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Dataset: {DATA_ROOT}")
print(f"Run_02: {RUN02_DIR}")
print(f"Run_03 output: {RUN03_DIR}")


# ============================================================
# 2. DISCOVER COMPLETE DEVELOPMENT PATIENTS
#    Uses Train + Validation only. Does not load Test.
# ============================================================

def extract_patient_id(filepath):
    filename = os.path.basename(filepath)

    match = re.search(
        r"patient[_\-\s]*0*(\d+)",
        filename,
        re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            f"Cannot extract patient ID from filename: {filepath}"
        )

    return str(int(match.group(1)))


patient_records = {}

for split in ("Train", "Validation"):
    for class_name, label in CLASSES.items():
        for view in VIEWS:
            folder = os.path.join(
                DATA_ROOT,
                split,
                class_name,
                view,
            )

            if not os.path.isdir(folder):
                raise FileNotFoundError(
                    f"Dataset folder not found: {folder}"
                )

            for filename in sorted(os.listdir(folder)):
                filepath = os.path.join(folder, filename)

                if not os.path.isfile(filepath):
                    continue

                if (
                    os.path.splitext(filename)[1].lower()
                    not in IMAGE_EXTENSIONS
                ):
                    continue

                patient_id = extract_patient_id(filepath)

                if patient_id not in patient_records:
                    patient_records[patient_id] = {
                        "patient_id": patient_id,
                        "label": label,
                        "split": split,
                        "paths": {},
                    }

                record = patient_records[patient_id]

                if (
                    record["label"] != label
                    or record["split"] != split
                ):
                    raise ValueError(
                        f"Inconsistent class or original split "
                        f"for patient {patient_id}"
                    )

                if view in record["paths"]:
                    raise ValueError(
                        f"Multiple images found for patient "
                        f"{patient_id}, view {view}"
                    )

                record["paths"][view] = filepath


# Keep only patients with all three views.
development_records = sorted(
    [
        record
        for record in patient_records.values()
        if all(view in record["paths"] for view in VIEWS)
    ],
    key=lambda record: int(record["patient_id"]),
)

patient_ids = np.asarray(
    [
        record["patient_id"]
        for record in development_records
    ],
    dtype=str,
)

labels = np.asarray(
    [
        record["label"]
        for record in development_records
    ],
    dtype=np.int64,
)

n_healthy = int(np.sum(labels == 0))
n_sick = int(np.sum(labels == 1))

print(
    f"Complete development patients: {len(development_records)}"
)
print(f"Healthy: {n_healthy}")
print(f"Sick: {n_sick}")

if (
    len(development_records) != 165
    or n_healthy != 101
    or n_sick != 64
):
    raise ValueError(
        "Expected 165 complete development patients "
        "(101 Healthy, 64 Sick). Check the dataset, "
        "patient ID extraction and complete-view filtering."
    )


# ============================================================
# 3. LOAD RUN_02 TRAINING HISTORIES
# ============================================================

def identify_head(filepath):
    path = filepath.lower()

    if "baseline" in path:
        return "baseline"

    if "linear_fusion" in path:
        return "linear"

    if "small_regularized" in path:
        return "small"

    return None


def identify_fold(filepath):
    match = re.search(
        r"fold[_\-\s]*(\d+)",
        filepath,
        re.IGNORECASE,
    )

    return int(match.group(1)) if match else None


def load_run02_histories():
    histories = {
        head: {}
        for head in HEAD_CONFIGS
    }

    if not os.path.isdir(RUN02_DIR):
        raise FileNotFoundError(
            f"Run_02 directory not found: {RUN02_DIR}"
        )

    for root, _, files in os.walk(RUN02_DIR):
        if "training_history.json" not in files:
            continue

        filepath = os.path.join(
            root,
            "training_history.json",
        )

        head = identify_head(filepath)
        fold = identify_fold(filepath)

        if head is None:
            continue

        if fold not in range(1, N_FOLDS + 1):
            continue

        if fold in histories[head]:
            raise ValueError(
                f"Multiple histories found for {head}, fold {fold}"
            )

        with open(
            filepath,
            "r",
            encoding="utf-8",
        ) as file:
            history = json.load(file)

        if "val_auc" not in history:
            raise KeyError(
                f"'val_auc' not found in {filepath}"
            )

        val_auc = np.asarray(
            history["val_auc"],
            dtype=float,
        )

        if len(val_auc) != MAX_EPOCHS:
            raise ValueError(
                f"Expected {MAX_EPOCHS} validation AUC values "
                f"in {filepath}; found {len(val_auc)}"
            )

        if not np.all(np.isfinite(val_auc)):
            raise ValueError(
                f"Non-finite validation AUC found in {filepath}"
            )

        histories[head][fold] = val_auc

        print(
            f"Loaded Run_02 history: {head}, fold {fold}"
        )

    expected_folds = set(range(1, N_FOLDS + 1))

    for head in HEAD_CONFIGS:
        actual_folds = set(histories[head])

        if actual_folds != expected_folds:
            raise FileNotFoundError(
                f"Run_02 histories missing or unidentified for "
                f"{head}. Found folds: {sorted(actual_folds)}. "
                f"Check the history filenames and directory structure."
            )

    return histories


run02_histories = load_run02_histories()


# ============================================================
# 4. DERIVE CANDIDATE EPOCHS FROM RUN_02
# ============================================================

candidate_information = {}

for head, fold_histories in run02_histories.items():

    auc_matrix = np.stack(
        [
            fold_histories[fold]
            for fold in range(1, N_FOLDS + 1)
        ],
        axis=0,
    )

    mean_auc_by_epoch = auc_matrix.mean(axis=0)

    aggregate_best_epoch = int(
        np.argmax(mean_auc_by_epoch) + 1
    )

    fold_best_epochs = {
        str(fold): int(
            np.argmax(fold_histories[fold]) + 1
        )
        for fold in range(1, N_FOLDS + 1)
    }

    candidates = sorted(
        set(
            list(fold_best_epochs.values())
            + [aggregate_best_epoch, MAX_EPOCHS]
        )
    )

    candidate_information[head] = {
        "aggregate_best_epoch": aggregate_best_epoch,
        "aggregate_best_mean_auc": float(
            mean_auc_by_epoch[aggregate_best_epoch - 1]
        ),
        "fold_best_epochs": fold_best_epochs,
        "candidate_epochs": candidates,
    }

    print(
        f"{head} candidate epochs: {candidates}"
    )


with open(
    os.path.join(
        RUN03_DIR,
        "candidate_epochs_by_head.json",
    ),
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        candidate_information,
        file,
        indent=2,
    )


# ============================================================
# 5. LOAD THE FIVE PATIENT SPLITS FROM RUN_02
# ============================================================

def get_manifest_patient_ids(fold_data, fold_number):
    train_ids = fold_data.get(
        "train_patient_ids",
        fold_data.get("training_patient_ids"),
    )

    val_ids = fold_data.get(
        "validation_patient_ids",
    )

    if train_ids is None or val_ids is None:
        raise ValueError(
            f"Training or validation IDs missing "
            f"from Run_02 Fold {fold_number}"
        )

    train_ids = [str(pid) for pid in train_ids]
    val_ids = [str(pid) for pid in val_ids]

    return train_ids, val_ids


available_ids = set(patient_ids.tolist())

cv_splits = []
fold_manifest = {}
validation_occurrences = {
    pid: 0
    for pid in patient_ids.tolist()
}

for fold_number in range(1, N_FOLDS + 1):

    fold_dir = os.path.join(
        RUN02_BASELINE_FOLD_RESULTS_DIR,
        f"Fold_{fold_number:02d}",
    )

    manifest_path = os.path.join(
        fold_dir,
        "fold_patient_split.json",
    )

    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            "Run_02 fold manifest not found:\n"
            f"{manifest_path}"
        )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as file:
        fold_data = json.load(file)

    train_ids, val_ids = get_manifest_patient_ids(
        fold_data,
        fold_number,
    )

    train_set = set(train_ids)
    val_set = set(val_ids)

    if len(train_ids) != len(train_set):
        raise ValueError(
            f"Duplicate training IDs in Fold {fold_number}"
        )

    if len(val_ids) != len(val_set):
        raise ValueError(
            f"Duplicate validation IDs in Fold {fold_number}"
        )

    overlap = train_set.intersection(val_set)

    if overlap:
        raise ValueError(
            f"Patient leakage in Fold {fold_number}: "
            f"{sorted(overlap)}"
        )

    fold_ids = train_set.union(val_set)

    if fold_ids != available_ids:
        missing = available_ids - fold_ids
        unexpected = fold_ids - available_ids

        raise ValueError(
            f"Run_02 Fold {fold_number} does not match the "
            f"current development dataset. "
            f"Missing IDs: {sorted(missing)}; "
            f"unexpected IDs: {sorted(unexpected)}"
        )

    # Convert the saved IDs to indices in development_records.
    # This preserves the exact patient membership from Run_02.
    train_idx = np.asarray(
        [
            i
            for i, pid in enumerate(patient_ids)
            if pid in train_set
        ],
        dtype=int,
    )

    val_idx = np.asarray(
        [
            i
            for i, pid in enumerate(patient_ids)
            if pid in val_set
        ],
        dtype=int,
    )

    if len(train_idx) != len(train_ids):
        raise ValueError(
            f"Training index count mismatch in Fold {fold_number}"
        )

    if len(val_idx) != len(val_ids):
        raise ValueError(
            f"Validation index count mismatch in Fold {fold_number}"
        )

    cv_splits.append(
        (train_idx, val_idx)
    )

    for pid in val_ids:
        validation_occurrences[pid] += 1

    # Preserve any metadata present in the original manifest,
    # while normalizing patient ID field names.
    fold_manifest[str(fold_number)] = {
        "fold": fold_number,
        "seed": fold_data.get("seed", SEED),
        "train_patient_ids": train_ids,
        "validation_patient_ids": val_ids,
        "training_patients": len(train_ids),
        "validation_patients": len(val_ids),
        "test_loaded": False,
        "test_evaluated": False,
        "source_manifest": os.path.relpath(
            manifest_path,
            RUN02_DIR,
        ),
    }

    print(
        f"Loaded Run_02 Fold {fold_number}: "
        f"{len(train_idx)} training patients, "
        f"{len(val_idx)} validation patients"
    )


# Every development patient must be in validation exactly once.
incorrect_validation_counts = {
    pid: count
    for pid, count in validation_occurrences.items()
    if count != 1
}

if incorrect_validation_counts:
    raise ValueError(
        "The Run_02 folds do not form a valid five-fold partition. "
        "Each development patient must appear in validation once. "
        f"Incorrect counts: {incorrect_validation_counts}"
    )

run03_manifest_path = os.path.join(
    RUN03_DIR,
    "fold_patient_split.json",
)

with open(
    run03_manifest_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        fold_manifest,
        file,
        indent=2,
    )

print(
    "Run_03 will reuse the five patient splits from Run_02."
)
print(
    f"Combined fold manifest saved to: {run03_manifest_path}"
)


# ============================================================
# 6. PREPROCESSING
# ============================================================

normalize = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)

rotation = RandomRotation(
    degrees=18,
)

affine = RandomAffine(
    degrees=0,
    scale=(0.95, 1.05),
)


def load_base_image(filepath, tmin, tmax):
    image = preprocess_image(
        filepath,
        tmin,
        tmax,
        IMAGE_SIZE,
    )

    image = np.asarray(
        image,
        dtype=np.float32,
    ).squeeze()

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return torch.from_numpy(
        image.copy()
    ).unsqueeze(0)


def build_fold_cache(train_records, val_records):
    # Temperature range is calculated using training images only.
    train_paths = [
        record["paths"][view]
        for record in train_records
        for view in VIEWS
    ]

    tmin, tmax = calculate_temperature_range(
        train_paths
    )

    image_cache = {}

    for record in train_records + val_records:
        for view in VIEWS:
            filepath = record["paths"][view]

            if filepath not in image_cache:
                image_cache[filepath] = load_base_image(
                    filepath,
                    tmin,
                    tmax,
                )

    return image_cache, float(tmin), float(tmax)


class MultiviewDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        records,
        image_cache,
        training,
    ):
        self.records = records
        self.image_cache = image_cache
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        view_tensors = []

        for view in VIEWS:
            image = ToPILImage()(
                self.image_cache[record["paths"][view]]
            )

            if self.training:
                image = rotation(image)
                image = affine(image)

            tensor = ToTensor()(image)

            # Convert grayscale input to three channels for ResNet-18.
            tensor = tensor.repeat(3, 1, 1)
            tensor = normalize(tensor)

            view_tensors.append(tensor)

        images = torch.stack(
            view_tensors,
            dim=0,
        )

        label = torch.tensor(
            record["label"],
            dtype=torch.float32,
        )

        return (
            images,
            label,
            record["patient_id"],
        )


# ============================================================
# 7. MODEL: FROZEN RESNET-18 + THREE-VIEW FEATURE FUSION
# ============================================================

class MultiviewResNet18(nn.Module):

    def __init__(self, head_name):
        super().__init__()

        config = HEAD_CONFIGS[head_name]

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        backbone.fc = nn.Identity()

        for parameter in backbone.parameters():
            parameter.requires_grad = False

        self.backbone = backbone

        if head_name == "linear":
            self.classifier = nn.Linear(
                1536,
                1,
            )

        else:
            self.classifier = nn.Sequential(
                nn.Linear(
                    1536,
                    config["hidden_dim"],
                ),
                nn.ReLU(),
                nn.Dropout(
                    config["dropout"],
                ),
                nn.Linear(
                    config["hidden_dim"],
                    1,
                ),
            )

    def train(self, mode=True):
        super().train(mode)

        # Keep the frozen backbone in evaluation mode.
        self.backbone.eval()

        return self

    def forward(self, images):
        # Input shape: [batch, views, channels, height, width]
        batch_size = images.shape[0]

        images = images.reshape(
            batch_size * len(VIEWS),
            images.shape[2],
            images.shape[3],
            images.shape[4],
        )

        with torch.no_grad():
            features = self.backbone(images)

        # Three 512-dimensional feature vectors -> 1536 features.
        features = features.reshape(
            batch_size,
            1536,
        )

        logits = self.classifier(features)

        return logits.squeeze(1)


# ============================================================
# 8. METRICS
# ============================================================

def get_class_weights(train_labels):
    healthy = int(np.sum(train_labels == 0))
    sick = int(np.sum(train_labels == 1))
    total = len(train_labels)

    if healthy == 0 or sick == 0:
        raise ValueError(
            "Both classes must be present in the training fold."
        )

    return {
        0: total / (2.0 * healthy),
        1: total / (2.0 * sick),
    }


def calculate_metrics(y_true, y_prob):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_prob = np.asarray(
        y_prob,
        dtype=float,
    )

    y_pred = (
        y_prob >= THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "roc_auc": float(
            roc_auc_score(y_true, y_prob)
        ),
        "average_precision": float(
            average_precision_score(y_true, y_prob)
        ),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "sensitivity": float(
            tp / (tp + fn) if (tp + fn) else 0.0
        ),
        "specificity": float(
            tn / (tn + fp) if (tn + fp) else 0.0
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
# 9. TRAIN AND EVALUATE ONE FOLD AT A FIXED EPOCH
# ============================================================

def train_one_fold(
    head_name,
    epoch_count,
    fold_number,
    train_idx,
    val_idx,
):
    set_seed(SEED)

    train_records = [
        development_records[i]
        for i in train_idx
    ]

    val_records = [
        development_records[i]
        for i in val_idx
    ]

    train_labels = np.asarray(
        [
            record["label"]
            for record in train_records
        ],
        dtype=int,
    )

    val_labels = np.asarray(
        [
            record["label"]
            for record in val_records
        ],
        dtype=int,
    )

    print(
        f"Training {head_name} | "
        f"Candidate epoch {epoch_count} | "
        f"Fold {fold_number}/{N_FOLDS}"
    )

    image_cache, tmin, tmax = build_fold_cache(
        train_records,
        val_records,
    )

    class_weights = get_class_weights(
        train_labels
    )

    train_dataset = MultiviewDataset(
        train_records,
        image_cache,
        training=True,
    )

    val_dataset = MultiviewDataset(
        val_records,
        image_cache,
        training=False,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = MultiviewResNet18(
        head_name
    ).to(DEVICE)

    config = HEAD_CONFIGS[head_name]

    optimizer = optim.Adam(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    criterion = nn.BCEWithLogitsLoss(
        reduction="none",
    )

    # Retrain from scratch for the specified candidate epoch count.
    for epoch in range(1, epoch_count + 1):
        model.train()

        for images, batch_labels, _ in train_loader:
            images = images.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)

            losses = criterion(
                logits,
                batch_labels,
            )

            sample_weights = torch.where(
                batch_labels == 0,
                torch.full_like(
                    batch_labels,
                    class_weights[0],
                ),
                torch.full_like(
                    batch_labels,
                    class_weights[1],
                ),
            )

            loss = (
                losses * sample_weights
            ).mean()

            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch == epoch_count:
            print(
                f"  Epoch {epoch}/{epoch_count}"
            )

    # Validation evaluation.
    model.eval()

    probabilities = []
    predicted_patient_ids = []

    with torch.no_grad():
        for images, _, batch_patient_ids in val_loader:
            images = images.to(DEVICE)

            logits = model(images)

            batch_probabilities = torch.sigmoid(
                logits
            ).cpu().numpy()

            probabilities.extend(
                batch_probabilities.tolist()
            )

            predicted_patient_ids.extend(
                [str(pid) for pid in batch_patient_ids]
            )

    val_prob = np.asarray(
        probabilities,
        dtype=float,
    )

    metrics = calculate_metrics(
        val_labels,
        val_prob,
    )

    print(
        f"  ROC-AUC: {metrics['roc_auc']:.4f} | "
        f"AP: {metrics['average_precision']:.4f}"
    )

    # Per-head output directories match Run_02's organization.
    head_dir = os.path.join(
        RUN03_DIR,
        HEAD_CONFIGS[head_name]["run_name"],
    )

    epoch_fold = os.path.join(
        f"epoch_{epoch_count}",
        f"Fold_{fold_number:02d}",
    )

    output_dir = os.path.join(
        head_dir,
        "fold_results",
        epoch_fold,
    )

    checkpoint_dir = os.path.join(
        head_dir,
        "checkpoints",
        epoch_fold,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "multiview_resnet18.pth",
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    records_by_id = {
        record["patient_id"]: record
        for record in val_records
    }

    predictions_path = os.path.join(
        output_dir,
        "validation_predictions.csv",
    )

    with open(
        predictions_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "patient_id",
            "true_label",
            "predicted_probability",
            "predicted_label",
            "anterior_path",
            "left_oblique_path",
            "right_oblique_path",
        ])

        for pid, true_label, probability in zip(
            predicted_patient_ids,
            val_labels,
            val_prob,
        ):
            record = records_by_id[pid]

            writer.writerow([
                pid,
                int(true_label),
                float(probability),
                int(probability >= THRESHOLD),
                record["paths"]["Anterior_ROI"],
                record["paths"]["Left_Oblique_ROI"],
                record["paths"]["Right_Oblique_ROI"],
            ])

    result = {
        "head": head_name,
        "candidate_epoch": int(epoch_count),
        "fold": int(fold_number),
        "train_patient_count": len(train_records),
        "validation_patient_count": len(val_records),
        "temperature_range": {
            "min": tmin,
            "max": tmax,
        },
        "class_weights": {
            "Healthy": float(class_weights[0]),
            "Sick": float(class_weights[1]),
        },
        "threshold": THRESHOLD,
        "seed": SEED,
        "metrics": metrics,
        "checkpoint": os.path.relpath(
            checkpoint_path,
            RUN03_DIR,
        ),
        "test_set_used": False,
    }

    result_path = os.path.join(
        output_dir,
        "fold_result.json",
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    del model
    del optimizer
    del train_loader
    del val_loader
    del train_dataset
    del val_dataset
    del image_cache

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    return result


# ============================================================
# 10. EVALUATE ALL CANDIDATE EPOCHS
# ============================================================

all_results = []

for head_name in HEAD_CONFIGS:

    candidates = candidate_information[
        head_name
    ]["candidate_epochs"]

    print(f"\nHead: {head_name}")
    print(f"Candidate epochs: {candidates}")

    for epoch_count in candidates:

        fold_results = []

        for fold_number, (train_idx, val_idx) in enumerate(
            cv_splits,
            start=1,
        ):
            result = train_one_fold(
                head_name=head_name,
                epoch_count=epoch_count,
                fold_number=fold_number,
                train_idx=train_idx,
                val_idx=val_idx,
            )

            fold_results.append(result)

        summary = {
            "head": head_name,
            "candidate_epoch": int(epoch_count),
            "n_folds": len(fold_results),
            "fold_results": fold_results,
        }

        for metric in METRICS:
            values = np.asarray(
                [
                    result["metrics"][metric]
                    for result in fold_results
                ],
                dtype=float,
            )

            summary[f"mean_{metric}"] = float(
                values.mean()
            )

            summary[f"std_{metric}"] = float(
                values.std(ddof=0)
            )

        all_results.append(summary)

        print(
            f"Epoch {epoch_count}: "
            f"ROC-AUC = {summary['mean_roc_auc']:.4f} "
            f"+/- {summary['std_roc_auc']:.4f}"
        )


# ============================================================
# 11. SAVE CANDIDATE EPOCH COMPARISON
# ============================================================

comparison_json_path = os.path.join(
    RUN03_DIR,
    "candidate_epoch_comparison.json",
)

with open(
    comparison_json_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        all_results,
        file,
        indent=2,
    )

comparison_csv_path = os.path.join(
    RUN03_DIR,
    "candidate_epoch_comparison.csv",
)

columns = [
    "head",
    "candidate_epoch",
]

for metric in METRICS:
    columns.extend([
        f"mean_{metric}",
        f"std_{metric}",
    ])

with open(
    comparison_csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=columns,
    )

    writer.writeheader()

    for result in all_results:
        row = {
            "head": result["head"],
            "candidate_epoch": result["candidate_epoch"],
        }

        for metric in METRICS:
            row[f"mean_{metric}"] = result[
                f"mean_{metric}"
            ]

            row[f"std_{metric}"] = result[
                f"std_{metric}"
            ]

        writer.writerow(row)


# ============================================================
# 12. SELECT FIXED EPOCH FOR EACH HEAD
#     Highest mean 5-fold validation ROC-AUC.
#     Ties are resolved in favour of the earlier epoch.
# ============================================================

selected_results = {}

for head_name in HEAD_CONFIGS:

    head_candidates = [
        result
        for result in all_results
        if result["head"] == head_name
    ]

    if not head_candidates:
        raise RuntimeError(
            f"No candidate results found for {head_name}"
        )

    selected = max(
        head_candidates,
        key=lambda result: (
            result["mean_roc_auc"],
            -result["candidate_epoch"],
        ),
    )

    selected_results[head_name] = {
        "selected_epoch": int(
            selected["candidate_epoch"]
        ),
        "mean_roc_auc": selected["mean_roc_auc"],
        "std_roc_auc": selected["std_roc_auc"],
        "mean_average_precision": selected[
            "mean_average_precision"
        ],
        "std_average_precision": selected[
            "std_average_precision"
        ],
        "mean_accuracy": selected["mean_accuracy"],
        "mean_balanced_accuracy": selected[
            "mean_balanced_accuracy"
        ],
        "mean_sensitivity": selected["mean_sensitivity"],
        "mean_specificity": selected["mean_specificity"],
        "mean_f1": selected["mean_f1"],
        "mean_mcc": selected["mean_mcc"],
    }

print("\nSelected fixed epochs:")

for head_name, result in selected_results.items():
    print(
        f"{head_name}: "
        f"epoch {result['selected_epoch']} | "
        f"ROC-AUC = {result['mean_roc_auc']:.4f} "
        f"+/- {result['std_roc_auc']:.4f}"
    )


# ============================================================
# 13. SAVE RUN SUMMARY
# ============================================================

run_summary = {
    "experiment": EXPERIMENT_NAME,
    "run": "Run_03_Fixed_Epoch_Selection",
    "development_patients": len(development_records),
    "healthy_patients": n_healthy,
    "sick_patients": n_sick,
    "views": VIEWS,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "maximum_epochs": MAX_EPOCHS,
    "n_folds": N_FOLDS,
    "threshold": THRESHOLD,
    "seed": SEED,
    "backbone": "ResNet-18 ImageNet1K_V1",
    "backbone_frozen": True,
    "features_per_view": 512,
    "fusion_features": 1536,
    "head_configs": HEAD_CONFIGS,
    "candidate_information": candidate_information,
    "selected_results_by_head": selected_results,
    "selection_criterion": (
        "Highest mean 5-fold validation ROC-AUC"
    ),
    "tie_breaker": "Earlier epoch",
    "fold_manifest_source": (
        "Run_02_A_Baseline/fold_results/Fold_XX/"
        "fold_patient_split.json"
    ),
    "combined_fold_manifest": os.path.basename(
        run03_manifest_path
    ),
    "test_set_used": False,
    "final_test_evaluation_performed": False,
}

summary_path = os.path.join(
    RUN03_DIR,
    "run_summary.json",
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        run_summary,
        file,
        indent=2,
    )

print("\nRun_03 completed.")
print(f"Results saved to: {RUN03_DIR}")
print("The independent test set was not evaluated.")
