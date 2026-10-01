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

from torch.utils.data import (
    Dataset,
    DataLoader
)


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
    r"\EXP_03_DMR_Anterior_ResNet18_Partial_FineTuning"
    r"\Results"
)

EXPERIMENT_NAME = (
    "EXP03_DMR_Anterior_ResNet18_Partial_FineTuning"
)

RUN_NAME = (
    "Run_02_5-fold_CV_Training_Dynamics"
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

BACKBONE_LR = 1e-4

CLASSIFIER_LR = 1e-3

DROPOUT_RATE = 0.3

DIAGNOSTIC_THRESHOLD = 0.5

NUM_CLASSES = 2

NUM_FOLDS = 5

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
    "Layer4 learning rate:",
    BACKBONE_LR
)

print(
    "Classifier learning rate:",
    CLASSIFIER_LR
)

print(
    "Dropout rate:",
    DROPOUT_RATE
)

print(
    "Number of folds:",
    NUM_FOLDS
)

print(
    "Diagnostic threshold:",
    DIAGNOSTIC_THRESHOLD
)

print("Device:", DEVICE)

print(
    "Test set: NOT USED"
)


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

        folder = (
            os.path.join(
                DATA_ROOT,
                split,
                class_name
            )
        )

        if not os.path.isdir(folder):

            raise FileNotFoundError(
                f"Directory not found: {folder}"
            )

        image_paths = []

        for filename in sorted(
            os.listdir(folder)
        ):

            filepath = os.path.join(
                folder,
                filename
            )

            if os.path.isfile(filepath):

                if filename.lower().endswith(
                    (
                        ".tif",
                        ".tiff"
                    )
                ):

                    image_paths.append(
                        filepath
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
# VERIFY DEVELOPMENT DATASET
# ============================================================

expected_healthy = 125

expected_sick = 72

expected_total = 197

actual_healthy = np.sum(
    all_labels == 0
)

actual_sick = np.sum(
    all_labels == 1
)

actual_total = len(
    all_labels
)

if actual_healthy != expected_healthy:

    raise ValueError(
        "Unexpected Healthy count. "
        f"Expected {expected_healthy}, "
        f"found {actual_healthy}."
    )

if actual_sick != expected_sick:

    raise ValueError(
        "Unexpected Sick count. "
        f"Expected {expected_sick}, "
        f"found {actual_sick}."
    )

if actual_total != expected_total:

    raise ValueError(
        "Unexpected development dataset size. "
        f"Expected {expected_total}, "
        f"found {actual_total}."
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
            filepath=self.image_paths[index],
            global_min=self.global_min,
            global_max=self.global_max,
            image_size=IMAGE_SIZE
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

epoch_train_loss_by_fold = []

epoch_val_loss_by_fold = []

epoch_train_auc_by_fold = []

epoch_val_auc_by_fold = []


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
        f"Fold_{fold}"
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
    # SAVE FOLD SPLIT
    # ========================================================

    fold_split_path = os.path.join(
        FOLD_RESULTS_OUTPUT_DIR,
        "fold_split.csv"
    )

    with open(
        fold_split_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "filename",
                "filepath",
                "split",
                "label",
                "class"
            ]
        )

        for filepath, label in zip(
            fold_train_paths,
            fold_train_labels
        ):

            writer.writerow(
                [
                    os.path.basename(filepath),
                    filepath,
                    "train",
                    int(label),
                    "Healthy"
                    if label == 0
                    else "Sick"
                ]
            )

        for filepath, label in zip(
            fold_validation_paths,
            fold_validation_labels
        ):

            writer.writerow(
                [
                    os.path.basename(filepath),
                    filepath,
                    "validation",
                    int(label),
                    "Healthy"
                    if label == 0
                    else "Sick"
                ]
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


    fold_temperature_range_path = os.path.join(
        FOLD_RESULTS_OUTPUT_DIR,
        "temperature_range.json"
    )

    with open(
        fold_temperature_range_path,
        "w"
    ) as f:

        json.dump(
            {
                "fold": fold,
                "source": "training_split_only",
                "minimum": float(GLOBAL_MIN),
                "maximum": float(GLOBAL_MAX)
            },
            f,
            indent=4
        )


    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    total_train_samples = len(
        fold_train_labels
    )

    healthy_weight = (
        total_train_samples
        /
        (
            2.0
            *
            fold_train_healthy
        )
    )

    sick_weight = (
        total_train_samples
        /
        (
            2.0
            *
            fold_train_sick
        )
    )

    class_weights = {

        0: float(
            healthy_weight
        ),

        1: float(
            sick_weight
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


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=train_generator,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )


    # ========================================================
    # RESET SEED BEFORE MODEL INITIALIZATION
    # ========================================================

    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(SEED)

        torch.cuda.manual_seed_all(SEED)


    # ========================================================
    # BUILD RESNET-18 MODEL
    # ========================================================

    model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )


    # Freeze entire pretrained network first.

    for parameter in model.parameters():

        parameter.requires_grad = False


    # Unfreeze layer4 for partial fine-tuning.

    for parameter in model.layer4.parameters():

        parameter.requires_grad = True


    # Replace classification head.

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


    # Classification head is trainable.

    for parameter in model.fc.parameters():

        parameter.requires_grad = True


    model = model.to(
        DEVICE
    )


    # ========================================================
    # MODEL PARAMETER COUNTS
    # ========================================================

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
        -
        trainable_parameters
    )


    if fold == 1:

        print()

        print(
            "MODEL PARAMETERS"
        )

        print(
            f"Total parameters: "
            f"{total_parameters}"
        )

        print(
            f"Trainable parameters: "
            f"{trainable_parameters}"
        )

        print(
            f"Frozen parameters: "
            f"{frozen_parameters}"
        )

        print()

        print(
            "Frozen blocks:"
        )

        print(
            "conv1"
        )

        print(
            "bn1"
        )

        print(
            "layer1"
        )

        print(
            "layer2"
        )

        print(
            "layer3"
        )

        print()

        print(
            "Trainable blocks:"
        )

        print(
            "layer4"
        )

        print(
            "fc"
        )


    # ========================================================
    # TRAINING MODE HELPER
    # ========================================================

    def set_training_mode(model):

        model.train()

        model.conv1.eval()

        model.bn1.eval()

        model.layer1.eval()

        model.layer2.eval()

        model.layer3.eval()


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.Adam(

        [

            {
                "params":
                    model.layer4.parameters(),

                "lr":
                    BACKBONE_LR
            },

            {
                "params":
                    model.fc.parameters(),

                "lr":
                    CLASSIFIER_LR
            }

        ]

    )


    # ========================================================
    # MODEL CHECKPOINT
    # ========================================================

    BEST_MODEL_PATH = os.path.join(
        FOLD_CHECKPOINT_DIR,
        "best_model.pth"
    )

    best_validation_auc = -np.inf

    best_epoch = 0


    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    history = {

        "epoch": [],

        "train_loss": [],

        "train_auc": [],

        "val_loss": [],

        "val_auc": []

    }


    print()

    print(
        "TRAINING"
    )


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    for epoch in range(
        EPOCHS
    ):

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        set_training_mode(
            model
        )

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
                    weight=sample_weights,
                    reduction="mean"
                )
            )


            loss.backward()

            optimizer.step()


            train_loss_total += (
                loss.item()
                *
                images.size(0)
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
            /
            len(train_dataset)

        )


        train_predictions = np.asarray(
            train_predictions
        )

        train_targets = np.asarray(
            train_targets
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


                # Validation loss is unweighted.

                loss = (
                    nn.functional
                    .binary_cross_entropy_with_logits(
                        logits,
                        labels,
                        reduction="mean"
                    )
                )


                val_loss_total += (
                    loss.item()
                    *
                    images.size(0)
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
            /
            len(validation_dataset)

        )


        val_predictions = np.asarray(
            val_predictions
        )

        val_targets = np.asarray(
            val_targets
        )


        val_auc = roc_auc_score(

            val_targets,
            val_predictions

        )


        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history["epoch"].append(
            epoch + 1
        )

        history["train_loss"].append(
            float(train_loss)
        )

        history["train_auc"].append(
            float(train_auc)
        )

        history["val_loss"].append(
            float(val_loss)
        )

        history["val_auc"].append(
            float(val_auc)
        )


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_auc > best_validation_auc:

            best_validation_auc = val_auc

            best_epoch = epoch + 1

            torch.save(

                {
                    "experiment":
                        EXPERIMENT_NAME,

                    "run":
                        RUN_NAME,

                    "fold":
                        fold,

                    "best_epoch":
                        best_epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "temperature_min":
                        float(GLOBAL_MIN),

                    "temperature_max":
                        float(GLOBAL_MAX),

                    "class_weights":
                        class_weights,

                    "backbone_lr":
                        BACKBONE_LR,

                    "classifier_lr":
                        CLASSIFIER_LR,

                    "dropout":
                        DROPOUT_RATE,

                    "diagnostic_threshold":
                        DIAGNOSTIC_THRESHOLD,

                    "seed":
                        SEED

                },

                BEST_MODEL_PATH

            )


        print(

            f"Fold {fold} | "

            f"Epoch {epoch + 1:03d}/{EPOCHS} | "

            f"Train Loss: {train_loss:.4f} | "

            f"Train AUC: {train_auc:.4f} | "

            f"Val Loss: {val_loss:.4f} | "

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
        best_epoch_index
        +
        1
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

        f"Best epoch: "
        f"{best_epoch}"

    )

    print(

        f"Best validation AUC: "
        f"{best_validation_auc:.4f}"

    )


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    checkpoint = torch.load(

        BEST_MODEL_PATH,

        map_location=DEVICE

    )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

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
    # CHECK BEST MODEL AUC
    # ========================================================

    validation_roc_auc = roc_auc_score(

        validation_labels,

        validation_probabilities

    )


    if not np.isclose(

        validation_roc_auc,

        best_validation_auc,

        atol=1e-8

    ):

        raise RuntimeError(

            "Reloaded best-model validation AUC "
            "does not match the recorded best "
            "validation AUC."

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


    validation_metrics = {

        "roc_auc":
            float(validation_roc_auc),

        "pr_auc":
            float(validation_pr_auc),

        "threshold":
            float(DIAGNOSTIC_THRESHOLD),

        "threshold_status":
            "diagnostic_only",

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

        f"Diagnostic threshold:"
        f" {DIAGNOSTIC_THRESHOLD:.4f}"

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

        "diagnostic_threshold":
            float(DIAGNOSTIC_THRESHOLD),

        "threshold_status":
            "diagnostic_only",

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

                "filename",

                "filepath",

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
            label,
            probability,
            prediction
        ) in zip(

            fold_validation_paths,

            validation_labels,

            validation_probabilities,

            validation_predictions

        ):

            writer.writerow(

                [

                    os.path.basename(
                        filepath
                    ),

                    filepath,

                    int(label),

                    "Healthy"
                    if label == 0
                    else "Sick",

                    float(
                        probability
                    ),

                    int(
                        prediction
                    ),

                    "Healthy"
                    if prediction == 0
                    else "Sick",

                    int(
                        prediction == label
                    )

                ]

            )


    # ========================================================
    # STORE EPOCH-WISE RESULTS
    # ========================================================

    epoch_train_loss_by_fold.append(

        history["train_loss"]

    )

    epoch_val_loss_by_fold.append(

        history["val_loss"]

    )

    epoch_train_auc_by_fold.append(

        history["train_auc"]

    )

    epoch_val_auc_by_fold.append(

        history["val_auc"]

    )


    # ========================================================
    # FOLD PLOTS
    # ========================================================

    import matplotlib.pyplot as plt


    epochs_range = range(
        1,
        EPOCHS + 1
    )


    # --------------------------------------------------------
    # Training and Validation Loss
    # --------------------------------------------------------

    plt.figure()

    plt.plot(

        epochs_range,

        history["train_loss"],

        label="Training Loss"

    )

    plt.plot(

        epochs_range,

        history["val_loss"],

        label="Validation Loss"

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
    # Training and Validation AUC
    # --------------------------------------------------------

    plt.figure()

    plt.plot(

        epochs_range,

        history["train_auc"],

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

    train_loss_values = np.asarray(

        [

            history[epoch]

            for history in epoch_train_loss_by_fold

        ],

        dtype=np.float64

    )

    val_loss_values = np.asarray(

        [

            history[epoch]

            for history in epoch_val_loss_by_fold

        ],

        dtype=np.float64

    )

    train_auc_values = np.asarray(

        [

            history[epoch]

            for history in epoch_train_auc_by_fold

        ],

        dtype=np.float64

    )

    val_auc_values = np.asarray(

        [

            history[epoch]

            for history in epoch_val_auc_by_fold

        ],

        dtype=np.float64

    )


    epoch_wise_results.append(

        {

            "epoch":
                epoch + 1,

            "train_loss_mean":
                float(
                    np.mean(
                        train_loss_values
                    )
                ),

            "train_loss_std":
                float(
                    np.std(
                        train_loss_values,
                        ddof=1
                    )
                ),

            "val_loss_mean":
                float(
                    np.mean(
                        val_loss_values
                    )
                ),

            "val_loss_std":
                float(
                    np.std(
                        val_loss_values,
                        ddof=1
                    )
                ),

            "train_auc_mean":
                float(
                    np.mean(
                        train_auc_values
                    )
                ),

            "train_auc_std":
                float(
                    np.std(
                        train_auc_values,
                        ddof=1
                    )
                ),

            "val_auc_mean":
                float(
                    np.mean(
                        val_auc_values
                    )
                ),

            "val_auc_std":
                float(
                    np.std(
                        val_auc_values,
                        ddof=1
                    )
                ),

            "fold_1_val_auc":
                float(
                    val_auc_values[0]
                ),

            "fold_2_val_auc":
                float(
                    val_auc_values[1]
                ),

            "fold_3_val_auc":
                float(
                    val_auc_values[2]
                ),

            "fold_4_val_auc":
                float(
                    val_auc_values[3]
                ),

            "fold_5_val_auc":
                float(
                    val_auc_values[4]
                )

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

    best_mean_validation_auc_index
    +
    1

)


best_mean_validation_auc = float(

    mean_validation_auc[
        best_mean_validation_auc_index
    ]

)


best_mean_validation_auc_std = float(

    epoch_wise_results[
        best_mean_validation_auc_index
    ][
        "val_auc_std"
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

print(

    f"Validation AUC SD: "
    f"{best_mean_validation_auc_std:.4f}"

)


# ============================================================
# INDIVIDUAL FOLD BEST RESULTS
# ============================================================

individual_best_auc = np.asarray(

    [

        result["best_validation_auc"]

        for result in fold_results

    ],

    dtype=np.float64

)

individual_best_epochs = np.asarray(

    [

        result["best_epoch"]

        for result in fold_results

    ],

    dtype=np.float64

)


individual_best_auc_mean = float(

    np.mean(
        individual_best_auc
    )

)

individual_best_auc_std = float(

    np.std(
        individual_best_auc,
        ddof=1
    )

)

individual_best_epoch_mean = float(

    np.mean(
        individual_best_epochs
    )

)

individual_best_epoch_std = float(

    np.std(
        individual_best_epochs,
        ddof=1
    )

)


# ============================================================
# SAVE EPOCH-WISE CROSS-VALIDATION SUMMARY
# ============================================================

epoch_wise_results_path = os.path.join(

    RESULTS_OUTPUT_DIR,

    "aggregate_epoch_metrics.csv"

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

    "best_validation_auc",

    "best_epoch",

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
            float(
                np.mean(values)
            ),

        "std":
            float(
                np.std(
                    values,
                    ddof=1
                )
            )

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

    "layer4_learning_rate":
        BACKBONE_LR,

    "classifier_learning_rate":
        CLASSIFIER_LR,

    "dropout_rate":
        DROPOUT_RATE,

    "diagnostic_threshold":
        DIAGNOSTIC_THRESHOLD,

    "threshold_status":
        "diagnostic_only",

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

    "backbone":
        "ResNet-18",

    "pretrained_weights":
        "ImageNet1K_V1",

    "fine_tuning_strategy":
        "Partial fine-tuning",

    "frozen_layers": [

        "conv1",
        "bn1",
        "layer1",
        "layer2",
        "layer3"

    ],

    "trainable_layers": [

        "layer4",
        "fc"

    ],

    "temperature_range_source":
        "Training split of each fold only",

    "class_weight_source":
        "Training labels of each fold only",

    "model_selection_metric":
        "Validation ROC-AUC",

    "best_aggregate_epoch":
        int(
            best_mean_validation_auc_epoch
        ),

    "best_aggregate_mean_validation_auc":
        float(
            best_mean_validation_auc
        ),

    "best_aggregate_validation_auc_std":
        float(
            best_mean_validation_auc_std
        ),

    "individual_best_auc_mean":
        float(
            individual_best_auc_mean
        ),

    "individual_best_auc_std":
        float(
            individual_best_auc_std
        ),

    "individual_best_epoch_mean":
        float(
            individual_best_epoch_mean
        ),

    "individual_best_epoch_std":
        float(
            individual_best_epoch_std
        ),

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

    label="Best Aggregate Epoch"

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

        "cv_mean_validation_auc.png"

    ),

    dpi=300

)

plt.close()


# ------------------------------------------------------------
# Mean Training and Validation Loss
# ------------------------------------------------------------

train_loss_mean = np.asarray(

    [

        result["train_loss_mean"]

        for result in epoch_wise_results

    ]

)


train_loss_std = np.asarray(

    [

        result["train_loss_std"]

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

    train_loss_mean,

    label="Mean Training Loss"

)

plt.plot(

    epochs_range,

    val_loss_mean,

    label="Mean Validation Loss"

)

plt.fill_between(

    epochs_range,

    train_loss_mean - train_loss_std,

    train_loss_mean + train_loss_std,

    alpha=0.2

)

plt.fill_between(

    epochs_range,

    val_loss_mean - val_loss_std,

    val_loss_mean + val_loss_std,

    alpha=0.2

)

plt.axvline(

    best_mean_validation_auc_epoch,

    linestyle="--",

    label="Best Aggregate Epoch"

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

        "cv_mean_loss.png"

    ),

    dpi=300

)

plt.close()


# ------------------------------------------------------------
# Mean Training and Validation AUC
# ------------------------------------------------------------

train_auc_mean = np.asarray(

    [

        result["train_auc_mean"]

        for result in epoch_wise_results

    ]

)


train_auc_std = np.asarray(

    [

        result["train_auc_std"]

        for result in epoch_wise_results

    ]

)


plt.figure()

plt.plot(

    epochs_range,

    train_auc_mean,

    label="Mean Training AUC"

)

plt.plot(

    epochs_range,

    val_auc_mean,

    label="Mean Validation AUC"

)

plt.fill_between(

    epochs_range,

    train_auc_mean - train_auc_std,

    train_auc_mean + train_auc_std,

    alpha=0.2

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

    label="Best Aggregate Epoch"

)

plt.xlabel(

    "Epoch"

)

plt.ylabel(

    "ROC-AUC"

)

plt.title(

    "5-Fold Cross-Validation Training and Validation AUC"

)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(

    os.path.join(

        PLOT_DIR,

        "cv_mean_auc.png"

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

    "purpose":
        (
            "Evaluate complete training dynamics "
            "of partially fine-tuned ResNet-18 "
            "using 5-fold stratified cross-validation."
        ),

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

    "layer4_learning_rate":
        BACKBONE_LR,

    "classifier_learning_rate":
        CLASSIFIER_LR,

    "dropout_rate":
        DROPOUT_RATE,

    "diagnostic_threshold":
        DIAGNOSTIC_THRESHOLD,

    "threshold_status":
        "diagnostic_only",

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
        "Training split of each fold only",

    "class_weight_source":
        "Training labels of each fold only",

    "backbone":
        "ResNet-18",

    "pretrained_weights":
        "ImageNet1K_V1",

    "fine_tuning_strategy":
        "Partial fine-tuning",

    "frozen_layers": [

        "conv1",
        "bn1",
        "layer1",
        "layer2",
        "layer3"

    ],

    "trainable_layers": [

        "layer4",
        "fc"

    ],

    "model_selection_metric":
        "Validation ROC-AUC",

    "best_aggregate_epoch":
        int(
            best_mean_validation_auc_epoch
        ),

    "best_aggregate_mean_validation_auc":
        float(
            best_mean_validation_auc
        ),

    "best_aggregate_validation_auc_std":
        float(
            best_mean_validation_auc_std
        ),

    "individual_best_auc_mean":
        float(
            individual_best_auc_mean
        ),

    "individual_best_auc_std":
        float(
            individual_best_auc_std
        ),

    "individual_best_epoch_mean":
        float(
            individual_best_epoch_mean
        ),

    "individual_best_epoch_std":
        float(
            individual_best_epoch_std
        ),

    "run03_role":
        (
            "Use epoch-wise cross-validation results "
            "to determine the fixed training epoch."
        ),

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


model_summary_model = models.resnet18(

    weights=models.ResNet18_Weights.IMAGENET1K_V1

)


for parameter in model_summary_model.parameters():

    parameter.requires_grad = False


for parameter in model_summary_model.layer4.parameters():

    parameter.requires_grad = True


num_features = (
    model_summary_model.fc.in_features
)


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


for parameter in model_summary_model.fc.parameters():

    parameter.requires_grad = True


model_total_parameters = sum(

    parameter.numel()

    for parameter
    in model_summary_model.parameters()

)


model_trainable_parameters = sum(

    parameter.numel()

    for parameter
    in model_summary_model.parameters()

    if parameter.requires_grad

)


model_frozen_parameters = (

    model_total_parameters
    -
    model_trainable_parameters

)


with open(

    model_summary_path,

    "w"

) as f:

    f.write(

        "Architecture: ResNet-18\n"

    )

    f.write(

        "Pretrained weights: "
        "ImageNet1K_V1\n"

    )

    f.write(

        "Fine-tuning strategy: "
        "Partial fine-tuning\n"

    )

    f.write(

        "Frozen layers: "
        "conv1, bn1, layer1, layer2, layer3\n"

    )

    f.write(

        "Trainable layers: "
        "layer4, fc\n"

    )

    f.write(

        f"Layer4 learning rate: "
        f"{BACKBONE_LR}\n"

    )

    f.write(

        f"Classifier learning rate: "
        f"{CLASSIFIER_LR}\n"

    )

    f.write(

        f"Dropout: "
        f"{DROPOUT_RATE}\n"

    )

    f.write(

        f"Total parameters: "
        f"{model_total_parameters}\n"

    )

    f.write(

        f"Trainable parameters: "
        f"{model_trainable_parameters}\n"

    )

    f.write(

        f"Frozen parameters: "
        f"{model_frozen_parameters}\n"

    )


# ============================================================
# SAVE RUN SUMMARY
# ============================================================

run_summary = {

    "experiment":
        EXPERIMENT_NAME,

    "run":
        RUN_NAME,

    "purpose":
        (
            "5-fold cross-validation training dynamics "
            "for partially fine-tuned ImageNet-pretrained "
            "ResNet-18."
        ),

    "dataset":
        "DMR_IR_Anterior_View_ROI",

    "development_dataset":
        "Original Train + Original Validation",

    "development_samples":
        int(len(all_labels)),

    "healthy_samples":
        int(total_healthy),

    "sick_samples":
        int(total_sick),

    "test_used":
        False,

    "model":
        "ResNet-18",

    "pretrained_weights":
        "ImageNet1K_V1",

    "fine_tuning_strategy":
        "Partial fine-tuning",

    "frozen_layers": [

        "conv1",
        "bn1",
        "layer1",
        "layer2",
        "layer3"

    ],

    "trainable_layers": [

        "layer4",
        "fc"

    ],

    "total_parameters":
        int(model_total_parameters),

    "trainable_parameters":
        int(model_trainable_parameters),

    "frozen_parameters":
        int(model_frozen_parameters),

    "layer4_learning_rate":
        BACKBONE_LR,

    "classifier_learning_rate":
        CLASSIFIER_LR,

    "dropout":
        DROPOUT_RATE,

    "batch_size":
        BATCH_SIZE,

    "maximum_epochs":
        EPOCHS,

    "number_of_folds":
        NUM_FOLDS,

    "seed":
        SEED,

    "seed_reset_each_fold":
        True,

    "augmentation":
        {
            "rotation_degrees":
                18,

            "scale_range":
                [
                    0.95,
                    1.05
                ],

            "horizontal_flip":
                False
        },

    "temperature_normalization":
        "Fold-specific training-only range",

    "class_weighting":
        "Fold-specific training-label weights",

    "model_selection":
        "Validation ROC-AUC",

    "diagnostic_threshold":
        DIAGNOSTIC_THRESHOLD,

    "threshold_status":
        "Diagnostic only. Final decision threshold not selected.",

    "best_aggregate_epoch":
        int(
            best_mean_validation_auc_epoch
        ),

    "best_aggregate_mean_validation_auc":
        float(
            best_mean_validation_auc
        ),

    "best_aggregate_validation_auc_std":
        float(
            best_mean_validation_auc_std
        ),

    "individual_best_auc_mean":
        float(
            individual_best_auc_mean
        ),

    "individual_best_auc_std":
        float(
            individual_best_auc_std
        ),

    "individual_best_epoch_mean":
        float(
            individual_best_epoch_mean
        ),

    "individual_best_epoch_std":
        float(
            individual_best_epoch_std
        ),

    "run03_role":
        (
            "Epoch-wise cross-validation results "
            "will be used to determine the fixed "
            "training epoch."
        )

}


run_summary_path = os.path.join(

    RESULTS_OUTPUT_DIR,

    "run_summary.json"

)


with open(

    run_summary_path,

    "w"

) as f:

    json.dump(

        run_summary,

        f,

        indent=4

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

print()

print(

    "Fine-tuning strategy: "
    "Partial fine-tuning"

)

print(

    "Frozen: "
    "conv1, bn1, layer1, layer2, layer3"

)

print(

    "Trainable: "
    "layer4, fc"

)

print()

print(

    f"Layer4 learning rate: "
    f"{BACKBONE_LR}"

)

print(

    f"Classifier learning rate: "
    f"{CLASSIFIER_LR}"

)

print()

print(

    f"Best aggregate epoch: "
    f"{best_mean_validation_auc_epoch}"

)

print(

    f"Best aggregate mean validation AUC: "
    f"{best_mean_validation_auc:.4f}"

)

print(

    f"Validation AUC SD at aggregate best epoch: "
    f"{best_mean_validation_auc_std:.4f}"

)

print()

print(

    f"Mean individual best-fold AUC: "
    f"{individual_best_auc_mean:.4f}"

)

print(

    f"SD individual best-fold AUC: "
    f"{individual_best_auc_std:.4f}"

)

print()

print(

    f"Mean individual best epoch: "
    f"{individual_best_epoch_mean:.2f}"

)

print(

    f"SD individual best epoch: "
    f"{individual_best_epoch_std:.2f}"

)

print()

print(

    "Diagnostic threshold: "
    f"{DIAGNOSTIC_THRESHOLD}"

)

print(

    "Final decision threshold selected: NO"

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
    "EXP03 Run02 completed."
)