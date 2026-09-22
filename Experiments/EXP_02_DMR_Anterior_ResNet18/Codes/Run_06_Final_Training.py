import os

import json

import random

import numpy as np

import pandas as pd

from PIL import Image

import torch

import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

from torchvision.models import ResNet18_Weights

from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt


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
    r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography"
    r"\Dataset\DMR_IR_Anterior_View_ROI"
)

RESULTS_DIR = (
    r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography"
    r"\Experiments\EXP_02_DMR_Anterior_ResNet18"
    r"\Results"
)

EXPERIMENT_NAME = "EXP02_DMR_Anterior_ResNet18"

RUN_NAME = "Run_06_Final_Training"

RUN_DIR = os.path.join(
    RESULTS_DIR,
    RUN_NAME
)

RESULTS_OUTPUT_DIR = os.path.join(
    RUN_DIR,
    "results"
)

MODEL_DIR = os.path.join(
    RUN_DIR,
    "model"
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
    MODEL_DIR,
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

EPOCHS = 32

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.3

SELECTED_THRESHOLD = 0.192174

NUM_WORKERS = 0

CLASSES = {
    "Healthy": 0,
    "Sick": 1,
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print("Training data: Train + Validation")

print("Development images: 197")

print("Test set: NOT USED")

print("Selected threshold:", SELECTED_THRESHOLD)


# ============================================================
# DATASET PATHS
# ============================================================

TRAIN_HEALTHY = os.path.join(
    DATA_ROOT,
    "Train",
    "Healthy"
)

TRAIN_SICK = os.path.join(
    DATA_ROOT,
    "Train",
    "Sick"
)

VAL_HEALTHY = os.path.join(
    DATA_ROOT,
    "Validation",
    "Healthy"
)

VAL_SICK = os.path.join(
    DATA_ROOT,
    "Validation",
    "Sick"
)


# ============================================================
# COLLECT IMAGE FILES
# ============================================================

def collect_files(folder):

    files = []

    if not os.path.isdir(folder):

        raise FileNotFoundError(
            f"Directory not found:\n{folder}"
        )

    for filename in sorted(
        os.listdir(folder)
    ):

        filepath = os.path.join(
            folder,
            filename
        )

        if not os.path.isfile(filepath):
            continue

        if filename.lower().endswith(
            (".tif", ".tiff")
        ):

            files.append(filepath)

    return files


# ============================================================
# LOAD DEVELOPMENT FILE PATHS
# ============================================================

train_healthy_paths = collect_files(
    TRAIN_HEALTHY
)

train_sick_paths = collect_files(
    TRAIN_SICK
)

val_healthy_paths = collect_files(
    VAL_HEALTHY
)

val_sick_paths = collect_files(
    VAL_SICK
)


# ============================================================
# DEVELOPMENT DATA
#
# Development = Train + Validation
#
# Test is intentionally NOT referenced anywhere.
# ============================================================

pool_paths = (
    train_healthy_paths
    + train_sick_paths
    + val_healthy_paths
    + val_sick_paths
)

pool_labels = (
    [0] * len(train_healthy_paths)
    + [1] * len(train_sick_paths)
    + [0] * len(val_healthy_paths)
    + [1] * len(val_sick_paths)
)


print("\nDevelopment data:")

print(
    f"Train Healthy:      "
    f"{len(train_healthy_paths)}"
)

print(
    f"Train Sick:         "
    f"{len(train_sick_paths)}"
)

print(
    f"Validation Healthy: "
    f"{len(val_healthy_paths)}"
)

print(
    f"Validation Sick:    "
    f"{len(val_sick_paths)}"
)

print()

print(
    f"Total Healthy: "
    f"{pool_labels.count(0)}"
)

print(
    f"Total Sick:    "
    f"{pool_labels.count(1)}"
)

print(
    f"Total images:  "
    f"{len(pool_paths)}"
)


# ============================================================
# VERIFY EXPECTED DEVELOPMENT SET
# ============================================================

assert len(pool_paths) == 197, (
    f"Expected 197 development images, "
    f"found {len(pool_paths)}"
)

assert pool_labels.count(0) == 125, (
    f"Expected 125 Healthy images, "
    f"found {pool_labels.count(0)}"
)

assert pool_labels.count(1) == 72, (
    f"Expected 72 Sick images, "
    f"found {pool_labels.count(1)}"
)


# ============================================================
# LOAD RAW THERMAL IMAGE
# ============================================================

def load_thermal_image(filepath):

    image = Image.open(filepath)

    arr = np.array(
        image,
        dtype=np.float32
    )

    return arr


# ============================================================
# CALCULATE DEVELOPMENT TEMPERATURE RANGE
#
# Calculated using all 197 development images.
# ============================================================

def calculate_temperature_range(
    filepaths
):

    global_min = np.inf

    global_max = -np.inf

    for filepath in filepaths:

        arr = load_thermal_image(
            filepath
        )

        valid_pixels = arr[arr > 0]

        if len(valid_pixels) == 0:
            continue

        current_min = valid_pixels.min()

        current_max = valid_pixels.max()

        global_min = min(
            global_min,
            current_min
        )

        global_max = max(
            global_max,
            current_max
        )

    return (
        float(global_min),
        float(global_max)
    )


# ============================================================
# TEMPERATURE NORMALIZATION RANGE
# ============================================================

print(
    "\nCalculating development-only "
    "temperature range..."
)

global_min, global_max = (
    calculate_temperature_range(
        pool_paths
    )
)

print(
    "Global minimum:",
    global_min
)

print(
    "Global maximum:",
    global_max
)


# ============================================================
# SAVE TEMPERATURE RANGE
# ============================================================

temperature_range = {

    "global_min":
        global_min,

    "global_max":
        global_max,

    "computed_from":
        "All 197 development images "
        "(Train + Validation)",

    "n_files":
        len(pool_paths),
}

TEMPERATURE_RANGE_PATH = os.path.join(
    RESULTS_OUTPUT_DIR,
    "temperature_range.json"
)

with open(
    TEMPERATURE_RANGE_PATH,
    "w"
) as f:

    json.dump(
        temperature_range,
        f,
        indent=4
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_temperature(
    arr,
    global_min,
    global_max
):

    valid_mask = arr > 0

    normalized = np.zeros_like(
        arr,
        dtype=np.float32
    )

    normalized[valid_mask] = (
        (
            arr[valid_mask]
            - global_min
        )
        /
        (
            global_max
            - global_min
        )
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    return normalized


# ============================================================
# PAD TO SQUARE
# ============================================================

def pad_to_square(arr):

    h, w = arr.shape

    size = max(h, w)

    pad_h = size - h

    pad_w = size - w

    top = pad_h // 2

    bottom = pad_h - top

    left = pad_w // 2

    right = pad_w - left

    padded = np.pad(
        arr,
        (
            (top, bottom),
            (left, right)
        ),
        mode="constant",
        constant_values=0
    )

    return padded


# ============================================================
# DATASET
# ============================================================

class ThermalDataset(Dataset):

    def __init__(
        self,
        filepaths,
        labels,
        global_min,
        global_max,
        transform=None
    ):

        self.filepaths = filepaths

        self.labels = labels

        self.global_min = global_min

        self.global_max = global_max

        self.transform = transform


    def __len__(self):

        return len(self.filepaths)


    def __getitem__(self, idx):

        filepath = self.filepaths[idx]

        label = self.labels[idx]

        # Load raw thermal image

        arr = load_thermal_image(
            filepath
        )

        # Normalize using
        # development-only range

        arr = normalize_temperature(
            arr,
            self.global_min,
            self.global_max
        )

        # Pad to square

        arr = pad_to_square(arr)

        # Convert to PIL

        image = Image.fromarray(
            (arr * 255).astype(
                np.uint8
            )
        )

        # Apply transforms

        if self.transform is not None:

            image = self.transform(
                image
            )

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.float32
            )
        )


# ============================================================
# TRAINING TRANSFORM
#
# Same augmentation strategy used in Run 04.
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.RandomRotation(
        degrees=18
    ),

    transforms.RandomAffine(
        degrees=0,
        scale=(0.95, 1.05)
    ),

    transforms.ToTensor(),

    # Convert 1-channel thermal image
    # to 3 channels

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
])


# ============================================================
# CREATE DEVELOPMENT DATASET
# ============================================================

train_dataset = ThermalDataset(

    filepaths=pool_paths,

    labels=pool_labels,

    global_min=global_min,

    global_max=global_max,

    transform=train_transform
)


train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    pin_memory=torch.cuda.is_available()
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print(
    "\nDevelopment class distribution:"
)

for class_name, class_id in CLASSES.items():

    count = np.sum(
        np.array(pool_labels)
        == class_id
    )

    print(
        f"{class_name}: {count}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_ids = np.array(
    [0, 1]
)

class_weights = compute_class_weight(

    class_weight="balanced",

    classes=class_ids,

    y=np.array(
        pool_labels
    )
)

CLASS_WEIGHTS = {

    int(class_id):
        float(weight)

    for class_id, weight
    in zip(
        class_ids,
        class_weights
    )
}

weight_healthy = CLASS_WEIGHTS[0]

weight_sick = CLASS_WEIGHTS[1]


print("\nClass weights:")

print(
    CLASS_WEIGHTS
)


# ============================================================
# MODEL
# ============================================================

print(
    "\nBuilding ImageNet-pretrained "
    "ResNet18..."
)

weights = ResNet18_Weights.IMAGENET1K_V1

model = models.resnet18(
    weights=weights
)


# ============================================================
# FREEZE RESNET18 BACKBONE
# ============================================================

for param in model.parameters():

    param.requires_grad = False


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

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


model = model.to(
    DEVICE
)


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
    f"{total_params:,}"
)

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)


# ============================================================
# LOSS FUNCTION
# ============================================================

pos_weight = torch.tensor(

    [
        weight_sick
        / weight_healthy
    ],

    dtype=torch.float32,

    device=DEVICE
)


criterion = nn.BCEWithLogitsLoss(

    pos_weight=pos_weight
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    filter(
        lambda p:
        p.requires_grad,

        model.parameters()
    ),

    lr=LEARNING_RATE
)


# ============================================================
# KEEP FROZEN BACKBONE IN EVAL MODE
#
# Prevent frozen BatchNorm layers from
# updating running statistics.
# ============================================================

def set_frozen_backbone_eval(
    model
):

    model.train()

    model.bn1.eval()

    model.layer1.eval()

    model.layer2.eval()

    model.layer3.eval()

    model.layer4.eval()


# ============================================================
# MODEL SUMMARY
# ============================================================

print(
    "\nModel:"
)

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
    DEVICE
)

print(
    "Training data:",
    len(train_dataset),
    "development images"
)

print(
    "Epochs:",
    EPOCHS
)


for epoch in range(
    1,
    EPOCHS + 1
):

    set_frozen_backbone_eval(
        model
    )

    running_loss = 0.0

    n_samples = 0


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
        ).squeeze(1)


        loss = criterion(
            logits,
            labels
        )


        loss.backward()

        optimizer.step()


        batch_size_actual = (
            images.size(0)
        )

        running_loss += (
            loss.item()
            * batch_size_actual
        )

        n_samples += (
            batch_size_actual
        )


    epoch_loss = (
        running_loss
        / n_samples
    )


    current_lr = (
        optimizer.param_groups[0]
        ["lr"]
    )


    history["epoch"].append(
        epoch
    )

    history["train_loss"].append(
        epoch_loss
    )

    history["learning_rate"].append(
        current_lr
    )


    print(
        f"Epoch {epoch:02d}/{EPOCHS} "
        f"| Train Loss: "
        f"{epoch_loss:.6f} "
        f"| LR: "
        f"{current_lr:.6f}"
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

HISTORY_PATH = os.path.join(

    RESULTS_OUTPUT_DIR,

    "training_history.json"
)

with open(
    HISTORY_PATH,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# SAVE FINAL MODEL CHECKPOINT
# ============================================================

MODEL_PATH = os.path.join(

    MODEL_DIR,

    "final_model_epoch32.pth"
)


torch.save(

    {

        "epoch":
            EPOCHS,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "learning_rate":
            LEARNING_RATE,

        "batch_size":
            BATCH_SIZE,

        "dropout":
            DROPOUT_RATE,

        "image_size":
            IMAGE_SIZE,

        "threshold":
            SELECTED_THRESHOLD,

        "seed":
            SEED,

        "architecture":
            (
                "ImageNet-pretrained ResNet18 "
                "with frozen backbone and "
                "512-64-1 classifier"
            ),

        "pretrained_weights":
            "ImageNet1K_V1",

        "trainable_parameters":
            trainable_params,

        "total_parameters":
            total_params,

        "development_samples":
            len(pool_paths),

        "healthy_samples":
            pool_labels.count(0),

        "sick_samples":
            pool_labels.count(1),

        "temperature_range":
            {
                "global_min":
                    global_min,

                "global_max":
                    global_max
            },

        "class_weights":
            {
                "healthy":
                    weight_healthy,

                "sick":
                    weight_sick
            }

    },

    MODEL_PATH
)


print(
    "\nFinal model saved:"
)

print(
    MODEL_PATH
)


# ============================================================
# SAVE DEVELOPMENT FILE LIST
# ============================================================

development_manifest = pd.DataFrame({

    "filepath":
        pool_paths,

    "label":
        pool_labels,

    "class":
        [
            "Healthy"
            if y == 0
            else "Sick"

            for y in pool_labels
        ]
})


MANIFEST_PATH = os.path.join(

    RESULTS_OUTPUT_DIR,

    "development_manifest.csv"
)


development_manifest.to_csv(

    MANIFEST_PATH,

    index=False
)


print(
    "Development manifest saved:"
)

print(
    MANIFEST_PATH
)


# ============================================================
# SAVE RUN CONFIGURATION
# ============================================================

RUN_CONFIG = {

    "experiment":
        EXPERIMENT_NAME,

    "run":
        RUN_NAME,

    "seed":
        SEED,

    "device":
        str(DEVICE),

    "data_root":
        DATA_ROOT,

    "development_splits":
        [
            "Train",
            "Validation"
        ],

    "train_images":
        int(
            len(train_healthy_paths)
            + len(train_sick_paths)
        ),

    "validation_images":
        int(
            len(val_healthy_paths)
            + len(val_sick_paths)
        ),

    "development_images":
        int(
            len(pool_paths)
        ),

    "healthy_images":
        int(
            pool_labels.count(0)
        ),

    "sick_images":
        int(
            pool_labels.count(1)
        ),

    "test_used":
        False,

    "image_size":
        IMAGE_SIZE,

    "input_channels":
        3,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "optimizer":
        "Adam",

    "learning_rate":
        LEARNING_RATE,

    "dropout_rate":
        DROPOUT_RATE,

    "architecture":
        "ResNet18",

    "pretrained_weights":
        "ImageNet1K_V1",

    "backbone_frozen":
        True,

    "classifier":
        [
            "Linear(512,64)",
            "ReLU",
            "Dropout(0.3)",
            "Linear(64,1)"
        ],

    "trainable_parameters":
        int(trainable_params),

    "augmentation":
        {
            "resize":
                [
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ],

            "random_rotation_degrees":
                18,

            "random_affine_scale":
                [
                    0.95,
                    1.05
                ],

            "horizontal_flip":
                False
        },

    "class_weighting":
        True,

    "class_weights":
        CLASS_WEIGHTS,

    "loss":
        "BCEWithLogitsLoss",

    "pos_weight":
        float(
            weight_sick
            / weight_healthy
        ),

    "early_stopping":
        False,

    "checkpointing":
        False,

    "deterministic":
        True,

    "model_selection":
        False,

    "fixed_epoch":
        EPOCHS,

    "temperature_normalization":
        {
            "method":
                "global_min_max",

            "range_source":
                "development_data",

            "global_min":
                global_min,

            "global_max":
                global_max
        },

    "threshold":
        SELECTED_THRESHOLD,

    "threshold_source":
        "Run 05 OOF threshold selection",

    "test_evaluation":
        False
}


CONFIG_PATH = os.path.join(

    RESULTS_OUTPUT_DIR,

    "run_config.json"
)


with open(
    CONFIG_PATH,
    "w"
) as f:

    json.dump(
        RUN_CONFIG,
        f,
        indent=4
    )


# ============================================================
# PLOT TRAINING LOSS
# ============================================================

epochs_axis = range(

    1,

    len(
        history["train_loss"]
    ) + 1
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(

    epochs_axis,

    history["train_loss"],

    label="Training Loss"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "EXP02 Run 06 - Training Loss"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


LOSS_PLOT_PATH = os.path.join(

    PLOT_DIR,

    "training_loss.png"
)


plt.savefig(

    LOSS_PLOT_PATH,

    dpi=300
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\nRUN COMPLETE"
)

print(
    "Device:",
    DEVICE
)

print(
    "\nFinal training epochs:",
    EPOCHS
)

print(
    "Development images:",
    len(pool_paths)
)

print(
    "Healthy:",
    pool_labels.count(0)
)

print(
    "Sick:",
    pool_labels.count(1)
)

print(
    "\nSelected threshold:",
    SELECTED_THRESHOLD
)

print(
    "\nResults saved to:"
)

print(
    RUN_DIR
)

print(
    "\nSaved files:"
)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Temperature range:",
    TEMPERATURE_RANGE_PATH
)

print(
    "Training history:",
    HISTORY_PATH
)

print(
    "Configuration:",
    CONFIG_PATH
)

print(
    "Development manifest:",
    MANIFEST_PATH
)

print(
    "\nTest set was NOT loaded or evaluated."
)

print(
    "Final model is the model state after epoch",
    EPOCHS
)