import os
import csv
import json
import random
import re

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from preprocessing import (
    calculate_temperature_range,
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

DATA_ROOT = Path(
    r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography"
    r"\Dataset\DMR_IR_Multi_View_ROI"
)

RESULTS_DIR = Path(
    r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography"
    r"\Experiments\Multi_View\EXP_01_DMR_Multiview_ResNet18"
    r"\Results"
)

EXPERIMENT_NAME = "EXP_01_DMR_Multiview_ResNet18"

RUN_NAME = "Run_06_Final_Training"

RUN_DIR = RESULTS_DIR / RUN_NAME

RESULTS_OUTPUT_DIR = RUN_DIR / "results"
MODEL_DIR = RUN_DIR / "model"
PLOT_DIR = RUN_DIR / "plots"

RUN05_THRESHOLD_FILE = (
    RESULTS_DIR
    / "Run_05_OOF_Threshold_Selection"
    / "results"
    / "selected_threshold.json"
)

for directory in (
    RESULTS_OUTPUT_DIR,
    MODEL_DIR,
    PLOT_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# RUN SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 82

LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 0.0

NUM_WORKERS = 0

VIEWS = (
    "Anterior_ROI",
    "Left_Oblique_ROI",
    "Right_Oblique_ROI",
)

CLASSES = {
    "Healthy": 0,
    "Sick": 1,
}

IMAGE_EXTENSIONS = {
    ".tif",
    ".tiff",
}

EXPECTED_PATIENTS = 165
EXPECTED_HEALTHY = 101
EXPECTED_SICK = 64


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD SELECTED THRESHOLD
# ============================================================

if not RUN05_THRESHOLD_FILE.is_file():
    raise FileNotFoundError(
        f"Run_05 threshold file not found:\n"
        f"{RUN05_THRESHOLD_FILE}"
    )

with open(
    RUN05_THRESHOLD_FILE,
    "r",
    encoding="utf-8",
) as f:
    threshold_data = json.load(f)

if threshold_data.get("experiment") != EXPERIMENT_NAME:
    raise ValueError(
        "Run_05 threshold belongs to another experiment."
    )

if threshold_data.get("fixed_epoch") != EPOCHS:
    raise ValueError(
        "Run_05 fixed epoch does not match final training."
    )

SELECTED_THRESHOLD = float(
    threshold_data["selected_threshold"]
)

if not 0.0 <= SELECTED_THRESHOLD <= 1.0:
    raise ValueError(
        "Selected threshold must be between 0 and 1."
    )


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("\nExperiment:", EXPERIMENT_NAME)
print("Run:", RUN_NAME)
print("Seed:", SEED)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)
print("Dropout:", DROPOUT_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

print("Fusion head: A - Baseline")
print("Training data: Train + Validation")
print("Development patients:", EXPECTED_PATIENTS)
print("Input views:", VIEWS)
print("Test set: NOT USED")
print("Selected threshold:", SELECTED_THRESHOLD)


# ============================================================
# DATASET PATHS
# ============================================================

DEVELOPMENT_SPLITS = (
    "Train",
    "Validation",
)


# ============================================================
# PATIENT ID EXTRACTION
# ============================================================

def patient_id_from_filename(filepath):

    match = re.search(
        r"patient[_\-\s]*0*(\d+)",
        Path(filepath).name,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"Cannot extract patient ID: {filepath}"
        )

    return str(int(match.group(1)))


# ============================================================
# COLLECT DEVELOPMENT PATIENTS
# ============================================================

def collect_development_patients():

    records = {}

    for split in DEVELOPMENT_SPLITS:

        for class_name, label in CLASSES.items():

            for view in VIEWS:

                folder = (
                    DATA_ROOT
                    / split
                    / class_name
                    / view
                )

                if not folder.is_dir():
                    raise FileNotFoundError(
                        f"Directory not found:\n{folder}"
                    )

                for filepath in sorted(folder.iterdir()):

                    if not filepath.is_file():
                        continue

                    if (
                        filepath.suffix.lower()
                        not in IMAGE_EXTENSIONS
                    ):
                        continue

                    patient_id = (
                        patient_id_from_filename(filepath)
                    )

                    if patient_id not in records:

                        records[patient_id] = {
                            "patient_id": patient_id,
                            "label": label,
                            "split": split,
                            "paths": {},
                        }

                    record = records[patient_id]

                    if (
                        record["label"] != label
                        or record["split"] != split
                    ):
                        raise ValueError(
                            "Conflicting class or split "
                            f"for patient {patient_id}"
                        )

                    if view in record["paths"]:
                        raise ValueError(
                            f"Duplicate {view} "
                            f"for patient {patient_id}"
                        )

                    record["paths"][view] = str(filepath)

    incomplete = [
        record["patient_id"]
        for record in records.values()
        if not all(
            view in record["paths"]
            for view in VIEWS
        )
    ]

    if incomplete:
        raise ValueError(
            "Patients with missing views: "
            f"{incomplete}"
        )

    complete = sorted(
        records.values(),
        key=lambda record: int(record["patient_id"]),
    )

    return complete


# ============================================================
# LOAD DEVELOPMENT DATA
# ============================================================

development_records = (
    collect_development_patients()
)

pool_labels = [
    record["label"]
    for record in development_records
]

pool_paths = [
    record["paths"][view]
    for record in development_records
    for view in VIEWS
]

print("\nDevelopment data:")

for split in DEVELOPMENT_SPLITS:

    for class_name, label in CLASSES.items():

        count = sum(
            record["split"] == split
            and record["label"] == label
            for record in development_records
        )

        print(
            f"{split} {class_name}: {count}"
        )

print(
    "\nTotal Healthy:",
    pool_labels.count(0),
)

print(
    "Total Sick:",
    pool_labels.count(1),
)

print(
    "Total patients:",
    len(development_records),
)

print(
    "Total view images:",
    len(pool_paths),
)


# ============================================================
# VERIFY EXPECTED DEVELOPMENT SET
# ============================================================

if len(development_records) != EXPECTED_PATIENTS:
    raise ValueError(
        f"Expected {EXPECTED_PATIENTS} patients, "
        f"found {len(development_records)}"
    )

if pool_labels.count(0) != EXPECTED_HEALTHY:
    raise ValueError(
        f"Expected {EXPECTED_HEALTHY} Healthy patients."
    )

if pool_labels.count(1) != EXPECTED_SICK:
    raise ValueError(
        f"Expected {EXPECTED_SICK} Sick patients."
    )

if len(pool_paths) != EXPECTED_PATIENTS * len(VIEWS):
    raise ValueError(
        "Unexpected number of multi-view images."
    )

if len(set(pool_paths)) != len(pool_paths):
    raise ValueError(
        "Duplicate image paths detected."
    )


# ============================================================
# CALCULATE DEVELOPMENT TEMPERATURE RANGE
#
# Same preprocessing function used in Run_04.
# Uses all 495 development-view images.
# ============================================================

print(
    "\nCalculating development-only "
    "temperature range..."
)

global_min, global_max = (
    calculate_temperature_range(pool_paths)
)

global_min = float(global_min)
global_max = float(global_max)

if (
    not np.isfinite(global_min)
    or not np.isfinite(global_max)
    or global_max <= global_min
):
    raise ValueError(
        "Invalid development temperature range."
    )

print("Global minimum:", global_min)
print("Global maximum:", global_max)


# ============================================================
# SAVE TEMPERATURE RANGE
# ============================================================

temperature_range = {
    "global_min": global_min,
    "global_max": global_max,
    "computed_from": (
        "All 165 development patients "
        "(Train + Validation)"
    ),
    "n_patients": len(development_records),
    "n_files": len(pool_paths),
    "views": list(VIEWS),
}

TEMPERATURE_RANGE_PATH = (
    RESULTS_OUTPUT_DIR
    / "temperature_range.json"
)

with open(
    TEMPERATURE_RANGE_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        temperature_range,
        f,
        indent=4,
    )


# ============================================================
# PREPROCESS DEVELOPMENT IMAGES
#
# Uses the same preprocessing.py as Run_04.
# ============================================================

print(
    "\nPreprocessing development images..."
)

image_cache = {}

for filepath in pool_paths:

    image = preprocess_image(
        filepath,
        global_min,
        global_max,
        IMAGE_SIZE,
    )

    image = np.asarray(
        image,
        dtype=np.float32,
    ).squeeze()

    if image.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(
            f"Unexpected image shape: "
            f"{image.shape}\n{filepath}"
        )

    if not np.all(np.isfinite(image)):
        raise ValueError(
            f"Non-finite image values: {filepath}"
        )

    image_cache[filepath] = (
        torch.from_numpy(
            np.clip(
                image,
                0.0,
                1.0,
            ).copy()
        ).unsqueeze(0)
    )

print(
    "Preprocessed images:",
    len(image_cache),
)


# ============================================================
# TRAINING TRANSFORM
#
# Same augmentation strategy used in Run_04.
# ============================================================

train_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.RandomRotation(
        degrees=18,
    ),

    transforms.RandomAffine(
        degrees=0,
        scale=(0.95, 1.05),
    ),

    transforms.ToTensor(),

    transforms.Lambda(
        lambda x: x.repeat(
            3,
            1,
            1,
        )
    ),

    transforms.Normalize(
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
    ),
])


# ============================================================
# DATASET
# ============================================================

class MultiViewDataset(Dataset):

    def __init__(
        self,
        records,
        cache,
        transform,
    ):

        self.records = records
        self.cache = cache
        self.transform = transform

    def __len__(self):

        return len(self.records)

    def __getitem__(self, idx):

        record = self.records[idx]

        images = torch.stack([
            self.transform(
                self.cache[record["paths"][view]]
            )
            for view in VIEWS
        ])

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
# CREATE DEVELOPMENT DATASET
# ============================================================

train_dataset = MultiViewDataset(
    records=development_records,
    cache=image_cache,
    transform=train_transform,
)

generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    generator=generator,
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print(
    "\nDevelopment class distribution:"
)

for class_name, class_id in CLASSES.items():

    count = np.sum(
        np.asarray(pool_labels) == class_id
    )

    print(
        f"{class_name}: {count}"
    )


# ============================================================
# CLASS WEIGHTS
#
# Same sample-weighted BCE approach as Run_04.
# ============================================================

n_patients = len(pool_labels)

CLASS_WEIGHTS = {
    0: n_patients / (2 * pool_labels.count(0)),
    1: n_patients / (2 * pool_labels.count(1)),
}

weight_healthy = float(CLASS_WEIGHTS[0])
weight_sick = float(CLASS_WEIGHTS[1])

print("\nClass weights:")
print(CLASS_WEIGHTS)


# ============================================================
# MODEL
# ============================================================

class MultiViewResNet18(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = models.resnet18(
            weights=(
                models.ResNet18_Weights.IMAGENET1K_V1
            )
        )

        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(1536, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, 1),
        )

    def train(self, mode=True):

        super().train(mode)

        # Keep frozen backbone in evaluation mode.
        self.backbone.eval()

        return self

    def forward(self, images):

        batch = images.shape[0]

        with torch.no_grad():

            features = self.backbone(
                images.reshape(
                    batch * 3,
                    3,
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                )
            )

        fused_features = features.reshape(
            batch,
            1536,
        )

        logits = self.classifier(
            fused_features
        )

        return logits.squeeze(1)


# ============================================================
# BUILD FINAL MODEL
# ============================================================

print(
    "\nBuilding ImageNet-pretrained "
    "multi-view ResNet18..."
)

model = MultiViewResNet18().to(DEVICE)


# ============================================================
# PARAMETER COUNT
# ============================================================

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "\nTotal parameters:",
    f"{total_params:,}",
)

print(
    "Trainable parameters:",
    f"{trainable_params:,}",
)

if trainable_params != 98433:
    raise ValueError(
        "Unexpected Baseline classifier parameter count."
    )


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    reduction="none",
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\nModel:")
print(model)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {
    "epoch": [],
    "train_loss": [],
    "learning_rate": [],
}


# ============================================================
# FINAL TRAINING
# ============================================================

print(
    "\nSTARTING FINAL TRAINING"
)

print(
    "Device:",
    DEVICE,
)

print(
    "Training data:",
    len(train_dataset),
    "development patients",
)

print(
    "Epochs:",
    EPOCHS,
)


for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    n_samples = 0

    for images, labels, _ in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad()

        logits = model(images)

        sample_weights = torch.where(
            labels == 0,
            torch.full_like(
                labels,
                weight_healthy,
            ),
            torch.full_like(
                labels,
                weight_sick,
            ),
        )

        sample_losses = criterion(
            logits,
            labels,
        )

        loss = (
            sample_losses * sample_weights
        ).mean()

        loss.backward()
        optimizer.step()

        batch_size_actual = images.size(0)

        running_loss += (
            loss.item() * batch_size_actual
        )

        n_samples += batch_size_actual

    epoch_loss = (
        running_loss / n_samples
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    history["epoch"].append(epoch)

    history["train_loss"].append(
        float(epoch_loss)
    )

    history["learning_rate"].append(
        float(current_lr)
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"| Train Loss: {epoch_loss:.6f} "
        f"| LR: {current_lr:.6f}"
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

HISTORY_PATH = (
    RESULTS_OUTPUT_DIR
    / "training_history.json"
)

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        history,
        f,
        indent=4,
    )


# ============================================================
# SAVE FINAL MODEL CHECKPOINT
# ============================================================

MODEL_PATH = (
    MODEL_DIR
    / f"final_model_epoch{EPOCHS}.pth"
)

torch.save(
    {
        "experiment": EXPERIMENT_NAME,
        "run": RUN_NAME,
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "dropout": DROPOUT_RATE,
        "weight_decay": WEIGHT_DECAY,
        "image_size": IMAGE_SIZE,
        "threshold": SELECTED_THRESHOLD,
        "seed": SEED,
        "architecture": (
            "Shared frozen ImageNet-pretrained "
            "ResNet18 with three-view feature "
            "concatenation and 1536-64-1 classifier"
        ),
        "pretrained_weights": "ImageNet1K_V1",
        "views": list(VIEWS),
        "feature_dimension_per_view": 512,
        "fusion_dimension": 1536,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "development_patients": len(
            development_records
        ),
        "healthy_patients": pool_labels.count(0),
        "sick_patients": pool_labels.count(1),
        "temperature_range": {
            "global_min": global_min,
            "global_max": global_max,
        },
        "class_weights": {
            "healthy": weight_healthy,
            "sick": weight_sick,
        },
        "test_set_used": False,
    },
    MODEL_PATH,
)

print("\nFinal model saved:")
print(MODEL_PATH)


# ============================================================
# SAVE DEVELOPMENT PATIENT MANIFEST
# ============================================================

MANIFEST_PATH = (
    RESULTS_OUTPUT_DIR
    / "development_manifest.csv"
)

manifest_columns = [
    "patient_id",
    "split",
    "label",
    "class",
    "anterior_filepath",
    "left_oblique_filepath",
    "right_oblique_filepath",
]

with open(
    MANIFEST_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=manifest_columns,
    )

    writer.writeheader()

    for record in development_records:

        writer.writerow({
            "patient_id": record["patient_id"],
            "split": record["split"],
            "label": record["label"],
            "class": (
                "Healthy"
                if record["label"] == 0
                else "Sick"
            ),
            "anterior_filepath": (
                record["paths"]["Anterior_ROI"]
            ),
            "left_oblique_filepath": (
                record["paths"]["Left_Oblique_ROI"]
            ),
            "right_oblique_filepath": (
                record["paths"]["Right_Oblique_ROI"]
            ),
        })

print(
    "Development manifest saved:",
    MANIFEST_PATH,
)


# ============================================================
# SAVE RUN CONFIGURATION
# ============================================================

RUN_CONFIG = {
    "experiment": EXPERIMENT_NAME,
    "run": RUN_NAME,
    "seed": SEED,
    "device": str(DEVICE),
    "data_root": str(DATA_ROOT),
    "development_splits": list(DEVELOPMENT_SPLITS),
    "development_patients": len(
        development_records
    ),
    "healthy_patients": pool_labels.count(0),
    "sick_patients": pool_labels.count(1),
    "development_images": len(pool_paths),
    "test_used": False,
    "image_size": IMAGE_SIZE,
    "input_channels_per_view": 3,
    "input_views": list(VIEWS),
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "optimizer": "Adam",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "dropout_rate": DROPOUT_RATE,
    "architecture": "ResNet18",
    "pretrained_weights": "ImageNet1K_V1",
    "backbone_frozen": True,
    "shared_backbone": True,
    "feature_dimension_per_view": 512,
    "fusion_method": "concatenation",
    "fusion_dimension": 1536,
    "classifier": [
        "Linear(1536,64)",
        "ReLU",
        "Dropout(0.3)",
        "Linear(64,1)",
    ],
    "trainable_parameters": int(
        trainable_params
    ),
    "total_parameters": int(
        total_params
    ),
    "augmentation": {
        "random_rotation_degrees": 18,
        "random_affine_scale": [
            0.95,
            1.05,
        ],
        "horizontal_flip": False,
    },
    "class_weighting": True,
    "class_weights": {
        "healthy": weight_healthy,
        "sick": weight_sick,
    },
    "loss": "BCEWithLogitsLoss",
    "loss_reduction": (
        "Per-sample weighted mean"
    ),
    "early_stopping": False,
    "checkpointing": False,
    "deterministic": True,
    "model_selection": False,
    "fixed_epoch": EPOCHS,
    "epoch_source": (
        "Run_03_Fixed_Epoch_Selection"
    ),
    "temperature_normalization": {
        "method": "global_min_max",
        "range_source": "development_data",
        "global_min": global_min,
        "global_max": global_max,
    },
    "threshold": SELECTED_THRESHOLD,
    "threshold_source": (
        "Run_05_OOF_Threshold_Selection"
    ),
    "threshold_selection_method": (
        "Constraint-based OOF threshold selection"
    ),
    "test_evaluation": False,
}

CONFIG_PATH = (
    RESULTS_OUTPUT_DIR
    / "run_config.json"
)

with open(
    CONFIG_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        RUN_CONFIG,
        f,
        indent=4,
    )


# ============================================================
# PLOT TRAINING LOSS
# ============================================================

epochs_axis = history["epoch"]

plt.figure(
    figsize=(8, 5),
)

plt.plot(
    epochs_axis,
    history["train_loss"],
    label="Training Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "EXP_01 Multi-View Run_06 - Training Loss"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

LOSS_PLOT_PATH = (
    PLOT_DIR
    / "training_loss.png"
)

plt.savefig(
    LOSS_PLOT_PATH,
    dpi=300,
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\nRUN COMPLETE")

print(
    "Device:",
    DEVICE,
)

print(
    "\nFinal training epochs:",
    EPOCHS,
)

print(
    "Development patients:",
    len(development_records),
)

print(
    "Healthy:",
    pool_labels.count(0),
)

print(
    "Sick:",
    pool_labels.count(1),
)

print(
    "Development images:",
    len(pool_paths),
)

print(
    "\nSelected threshold:",
    SELECTED_THRESHOLD,
)

print(
    "\nResults saved to:",
    RUN_DIR,
)

print("\nSaved files:")

print(
    "Model:",
    MODEL_PATH,
)

print(
    "Temperature range:",
    TEMPERATURE_RANGE_PATH,
)

print(
    "Training history:",
    HISTORY_PATH,
)

print(
    "Configuration:",
    CONFIG_PATH,
)

print(
    "Development manifest:",
    MANIFEST_PATH,
)

print(
    "Training loss plot:",
    LOSS_PLOT_PATH,
)

print(
    "\nTest set was NOT loaded or evaluated."
)

print(
    "Final model is the model state after epoch",
    EPOCHS,
)