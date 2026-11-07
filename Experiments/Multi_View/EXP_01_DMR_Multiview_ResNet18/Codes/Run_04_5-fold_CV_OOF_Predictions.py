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
    roc_auc_score,
    average_precision_score,
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
FIXED_EPOCH = 82
N_FOLDS = 5
NUM_WORKERS = 0

LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 0.0

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
RUN_NAME = "Run_04_5-fold_CV_OOF_Predictions"

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

RUN04_DIR = os.path.join(
    RESULTS_DIR,
    RUN_NAME,
)

RUN04_FOLD_RESULTS_DIR = os.path.join(
    RUN04_DIR,
    "fold_results",
)

RUN04_CHECKPOINTS_DIR = os.path.join(
    RUN04_DIR,
    "checkpoints",
)

for folder in [
    RUN04_DIR,
    RUN04_FOLD_RESULTS_DIR,
    RUN04_CHECKPOINTS_DIR,
]:
    os.makedirs(folder, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

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
print(f"Run: {RUN_NAME}")
print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Dataset: {DATA_ROOT}")
print(f"Run_02: {RUN02_DIR}")
print(f"Run_03: {RUN03_DIR}")
print(f"Run_04 output: {RUN04_DIR}")
print(f"Fixed epoch: {FIXED_EPOCH}")
print("Fusion head: A - Baseline")
print("Threshold selection: NOT PERFORMED")
print("Independent test set: NOT USED")


# ============================================================
# 3. DISCOVER COMPLETE DEVELOPMENT PATIENTS
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
    f"\nComplete development patients: {len(development_records)}"
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
# 4. LOAD THE FIVE PATIENT SPLITS FROM RUN_02
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

    train_ids = [str(int(pid)) for pid in train_ids]
    val_ids = [str(int(pid)) for pid in val_ids]

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
            f"Run_02 Fold {fold_number} does not match "
            f"the current development dataset. "
            f"Missing IDs: {sorted(missing)}; "
            f"Unexpected IDs: {sorted(unexpected)}"
        )

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

    if len(train_idx) != 132 or len(val_idx) != 33:
        raise ValueError(
            f"Unexpected fold sizes in Fold {fold_number}: "
            f"{len(train_idx)} training, "
            f"{len(val_idx)} validation."
        )

    cv_splits.append(
        (train_idx, val_idx)
    )

    for pid in val_ids:
        validation_occurrences[pid] += 1

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

run04_manifest_path = os.path.join(
    RUN04_DIR,
    "fold_patient_split.json",
)

with open(
    run04_manifest_path,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        fold_manifest,
        file,
        indent=2,
    )

print(
    "\nRun_04 will reuse the five patient splits from Run_02."
)


# ============================================================
# 5. PREPROCESSING
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

            # Convert grayscale input to three channels.
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
# 6. MODEL: FROZEN RESNET-18 + BASELINE FUSION HEAD
# ============================================================

class MultiviewResNet18(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        backbone.fc = nn.Identity()

        for parameter in backbone.parameters():
            parameter.requires_grad = False

        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Linear(1536, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, 1),
        )

    def train(self, mode=True):
        super().train(mode)

        # Keep the frozen backbone in evaluation mode.
        self.backbone.eval()

        return self

    def forward(self, images):
        # Input: [batch, views, channels, height, width]
        batch_size = images.shape[0]

        images = images.reshape(
            batch_size * len(VIEWS),
            images.shape[2],
            images.shape[3],
            images.shape[4],
        )

        with torch.no_grad():
            features = self.backbone(images)

        # Three 512-dimensional vectors -> 1536 features.
        features = features.reshape(
            batch_size,
            1536,
        )

        logits = self.classifier(features)

        return logits.squeeze(1)


# ============================================================
# 7. CLASS WEIGHTS
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


# ============================================================
# 8. OOF STORAGE
# ============================================================

oof_probabilities = np.full(
    len(development_records),
    np.nan,
    dtype=np.float64,
)

oof_fold = np.full(
    len(development_records),
    -1,
    dtype=np.int32,
)

fold_summaries = []


# ============================================================
# 9. TRAIN FIVE FOLDS AND GENERATE OOF PREDICTIONS
# ============================================================

for fold_number, (train_idx, val_idx) in enumerate(
    cv_splits,
    start=1,
):

    print(f"\n{'=' * 60}")
    print(f"FOLD {fold_number}/{N_FOLDS}")
    print(f"{'=' * 60}")

    # Reset seed for each fold.
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

    print(f"Training patients: {len(train_records)}")
    print(f"Validation patients: {len(val_records)}")
    print(f"Training Healthy: {np.sum(train_labels == 0)}")
    print(f"Training Sick: {np.sum(train_labels == 1)}")
    print(f"Validation Healthy: {np.sum(val_labels == 0)}")
    print(f"Validation Sick: {np.sum(val_labels == 1)}")

    # --------------------------------------------------------
    # Fold-specific temperature normalization
    # --------------------------------------------------------

    image_cache, tmin, tmax = build_fold_cache(
        train_records,
        val_records,
    )

    print(f"Temperature min: {tmin}")
    print(f"Temperature max: {tmax}")

    # --------------------------------------------------------
    # Fold-specific class weights
    # --------------------------------------------------------

    class_weights = get_class_weights(
        train_labels
    )

    print(f"Class weights: {class_weights}")

    # --------------------------------------------------------
    # Datasets and data loaders
    # --------------------------------------------------------

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
        num_workers=NUM_WORKERS,
        generator=generator,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # --------------------------------------------------------
    # Fresh model initialization
    # --------------------------------------------------------

    model = MultiviewResNet18().to(DEVICE)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable parameters: {trainable_parameters}")

    optimizer = optim.Adam(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.BCEWithLogitsLoss(
        reduction="none",
    )

    # --------------------------------------------------------
    # Fixed-duration training
    # --------------------------------------------------------

    training_losses = []

    print(
        f"\nTraining Baseline for exactly "
        f"{FIXED_EPOCH} epochs..."
    )

    for epoch in range(1, FIXED_EPOCH + 1):
        model.train()

        running_loss = 0.0
        total_samples = 0

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

            batch_size = batch_labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        epoch_loss = (
            running_loss / total_samples
        )

        training_losses.append(
            float(epoch_loss)
        )

        print(
            f"Epoch {epoch:3d}/{FIXED_EPOCH} "
            f"- loss: {epoch_loss:.6f}"
        )

    # --------------------------------------------------------
    # Generate held-out validation predictions
    # --------------------------------------------------------

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
        dtype=np.float64,
    )

    expected_patient_ids = [
        record["patient_id"]
        for record in val_records
    ]

    if predicted_patient_ids != expected_patient_ids:
        raise RuntimeError(
            f"Validation prediction order mismatch "
            f"in Fold {fold_number}"
        )

    if len(val_prob) != len(val_idx):
        raise RuntimeError(
            f"Prediction count mismatch in Fold {fold_number}"
        )

    if not np.all(np.isfinite(val_prob)):
        raise RuntimeError(
            f"Non-finite probabilities in Fold {fold_number}"
        )

    # --------------------------------------------------------
    # Store OOF predictions
    # --------------------------------------------------------

    if np.any(np.isfinite(oof_probabilities[val_idx])):
        raise RuntimeError(
            f"Duplicate OOF predictions in Fold {fold_number}"
        )

    oof_probabilities[val_idx] = val_prob
    oof_fold[val_idx] = fold_number

    # --------------------------------------------------------
    # Fold-level threshold-independent metrics
    # --------------------------------------------------------

    fold_auc = float(
        roc_auc_score(
            val_labels,
            val_prob,
        )
    )

    fold_ap = float(
        average_precision_score(
            val_labels,
            val_prob,
        )
    )

    print(f"\nFold ROC-AUC: {fold_auc:.4f}")
    print(f"Fold Average Precision: {fold_ap:.4f}")

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    fold_output_dir = os.path.join(
        RUN04_FOLD_RESULTS_DIR,
        f"Fold_{fold_number:02d}",
    )

    checkpoint_dir = os.path.join(
        RUN04_CHECKPOINTS_DIR,
        f"Fold_{fold_number:02d}",
    )

    os.makedirs(
        fold_output_dir,
        exist_ok=True,
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "multiview_resnet18.pth",
    )

    torch.save(
        model.state_dict(),
        checkpoint_path,
    )

    # --------------------------------------------------------
    # Save fold predictions
    # --------------------------------------------------------

    predictions_path = os.path.join(
        fold_output_dir,
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
            "anterior_path",
            "left_oblique_path",
            "right_oblique_path",
        ])

        for record, probability in zip(
            val_records,
            val_prob,
        ):
            writer.writerow([
                record["patient_id"],
                int(record["label"]),
                float(probability),
                record["paths"]["Anterior_ROI"],
                record["paths"]["Left_Oblique_ROI"],
                record["paths"]["Right_Oblique_ROI"],
            ])

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = os.path.join(
        fold_output_dir,
        "training_history.json",
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "train_loss": training_losses,
            },
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Save fold summary
    # --------------------------------------------------------

    fold_summary = {
        "fold": fold_number,
        "fixed_epoch": FIXED_EPOCH,
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT_RATE,
        "weight_decay": WEIGHT_DECAY,
        "trainable_parameters": trainable_parameters,
        "train_patient_count": len(train_records),
        "validation_patient_count": len(val_records),
        "train_healthy": int(np.sum(train_labels == 0)),
        "train_sick": int(np.sum(train_labels == 1)),
        "validation_healthy": int(np.sum(val_labels == 0)),
        "validation_sick": int(np.sum(val_labels == 1)),
        "class_weights": {
            "Healthy": float(class_weights[0]),
            "Sick": float(class_weights[1]),
        },
        "temperature_range": {
            "min": tmin,
            "max": tmax,
        },
        "roc_auc": fold_auc,
        "average_precision": fold_ap,
        "checkpoint": os.path.relpath(
            checkpoint_path,
            RUN04_DIR,
        ),
        "test_set_used": False,
    }

    fold_summaries.append(
        fold_summary
    )

    fold_summary_path = os.path.join(
        fold_output_dir,
        "fold_result.json",
    )

    with open(
        fold_summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            fold_summary,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Clean up fold
    # --------------------------------------------------------

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


# ============================================================
# 10. VERIFY OOF COVERAGE
# ============================================================

missing_predictions = np.isnan(
    oof_probabilities
)

missing_folds = (
    oof_fold == -1
)

print("\nOOF COVERAGE CHECK")
print(f"Development patients: {len(development_records)}")
print(f"OOF predictions: {np.sum(~missing_predictions)}")
print(f"Missing predictions: {np.sum(missing_predictions)}")
print(f"Missing fold assignments: {np.sum(missing_folds)}")

if np.any(missing_predictions):
    raise RuntimeError(
        "Some development patients do not have OOF predictions."
    )

if np.any(missing_folds):
    raise RuntimeError(
        "Some development patients do not have fold assignments."
    )

if len(np.unique(patient_ids)) != len(development_records):
    raise RuntimeError(
        "Duplicate patient IDs found in development dataset."
    )


# ============================================================
# 11. SAVE COMBINED OOF PREDICTIONS
# ============================================================

oof_prediction_path = os.path.join(
    RUN04_DIR,
    "oof_predictions.csv",
)

with open(
    oof_prediction_path,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "patient_id",
        "true_label",
        "true_class",
        "oof_fold",
        "oof_probability",
        "anterior_path",
        "left_oblique_path",
        "right_oblique_path",
    ])

    for index, record in enumerate(development_records):
        true_label = int(record["label"])

        true_class = (
            "Healthy"
            if true_label == 0
            else "Sick"
        )

        writer.writerow([
            record["patient_id"],
            true_label,
            true_class,
            int(oof_fold[index]),
            float(oof_probabilities[index]),
            record["paths"]["Anterior_ROI"],
            record["paths"]["Left_Oblique_ROI"],
            record["paths"]["Right_Oblique_ROI"],
        ])


# ============================================================
# 12. OVERALL OOF METRICS
# ============================================================

overall_oof_auc = float(
    roc_auc_score(
        labels,
        oof_probabilities,
    )
)

overall_oof_ap = float(
    average_precision_score(
        labels,
        oof_probabilities,
    )
)

print(f"\nOverall OOF ROC-AUC: {overall_oof_auc:.4f}")
print(f"Overall OOF Average Precision: {overall_oof_ap:.4f}")


# ============================================================
# 13. SAVE FOLD SUMMARY CSV
# ============================================================

fold_summary_csv_path = os.path.join(
    RUN04_DIR,
    "fold_summary.csv",
)

columns = [
    "fold",
    "fixed_epoch",
    "train_patient_count",
    "validation_patient_count",
    "train_healthy",
    "train_sick",
    "validation_healthy",
    "validation_sick",
    "temperature_min",
    "temperature_max",
    "roc_auc",
    "average_precision",
]

with open(
    fold_summary_csv_path,
    "w",
    newline="",
    encoding="utf-8",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=columns,
    )

    writer.writeheader()

    for result in fold_summaries:
        writer.writerow({
            "fold": result["fold"],
            "fixed_epoch": result["fixed_epoch"],
            "train_patient_count": result["train_patient_count"],
            "validation_patient_count": result["validation_patient_count"],
            "train_healthy": result["train_healthy"],
            "train_sick": result["train_sick"],
            "validation_healthy": result["validation_healthy"],
            "validation_sick": result["validation_sick"],
            "temperature_min": result["temperature_range"]["min"],
            "temperature_max": result["temperature_range"]["max"],
            "roc_auc": result["roc_auc"],
            "average_precision": result["average_precision"],
        })


# ============================================================
# 14. SAVE RUN SUMMARY
# ============================================================

fold_auc_values = np.asarray(
    [
        result["roc_auc"]
        for result in fold_summaries
    ],
    dtype=float,
)

fold_ap_values = np.asarray(
    [
        result["average_precision"]
        for result in fold_summaries
    ],
    dtype=float,
)

run_summary = {
    "experiment": EXPERIMENT_NAME,
    "run": RUN_NAME,
    "fusion_head": "A_Baseline",
    "fixed_epoch": FIXED_EPOCH,
    "fixed_epoch_source": "Run_03_Fixed_Epoch_Selection",
    "development_patients": len(development_records),
    "healthy_patients": n_healthy,
    "sick_patients": n_sick,
    "views": VIEWS,
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "n_folds": N_FOLDS,
    "seed": SEED,
    "optimizer": "Adam",
    "learning_rate": LEARNING_RATE,
    "dropout": DROPOUT_RATE,
    "weight_decay": WEIGHT_DECAY,
    "backbone": "ResNet-18 ImageNet1K_V1",
    "backbone_frozen": True,
    "features_per_view": 512,
    "fusion_features": 1536,
    "classifier": "1536 -> 64 -> ReLU -> Dropout(0.3) -> 1",
    "trainable_parameters": 98497,
    "temperature_normalization": (
        "Fold-specific training-only temperature range"
    ),
    "mean_fold_roc_auc": float(fold_auc_values.mean()),
    "std_fold_roc_auc": float(fold_auc_values.std(ddof=0)),
    "mean_fold_average_precision": float(fold_ap_values.mean()),
    "std_fold_average_precision": float(fold_ap_values.std(ddof=0)),
    "overall_oof_roc_auc": overall_oof_auc,
    "overall_oof_average_precision": overall_oof_ap,
    "oof_predictions": len(development_records),
    "threshold_applied": False,
    "threshold_selection_performed": False,
    "fold_manifest_source": (
        "Run_02_A_Baseline/fold_results/Fold_XX/"
        "fold_patient_split.json"
    ),
    "test_loaded": False,
    "test_evaluated": False,
    "fold_results": fold_summaries,
}

summary_path = os.path.join(
    RUN04_DIR,
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


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\nRun_04 completed.")
print(f"Development patients: {len(development_records)}")
print(f"OOF predictions: {len(oof_probabilities)}")
print(f"Fixed epoch: {FIXED_EPOCH}")
print(f"Mean fold ROC-AUC: {fold_auc_values.mean():.4f}")
print(f"Mean fold AP: {fold_ap_values.mean():.4f}")
print(f"Overall OOF ROC-AUC: {overall_oof_auc:.4f}")
print(f"Overall OOF AP: {overall_oof_ap:.4f}")
print(f"OOF predictions saved to: {oof_prediction_path}")
print(f"Results saved to: {RUN04_DIR}")
print("Threshold selection: NOT PERFORMED")
print("Independent test set: NOT USED")
