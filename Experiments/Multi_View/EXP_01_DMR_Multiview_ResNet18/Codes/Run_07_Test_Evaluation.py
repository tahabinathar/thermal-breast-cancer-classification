import os
import csv
import json
import random
import re

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
)

import matplotlib.pyplot as plt

from preprocessing import preprocess_image


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(
    True,
    warn_only=True,
)

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

RUN_NAME = "Run_07_Final_Test_Evaluation"

RUN_DIR = RESULTS_DIR / RUN_NAME

RESULTS_OUTPUT_DIR = RUN_DIR / "results"
PLOT_DIR = RUN_DIR / "plots"

FINAL_TRAINING_RUN = "Run_06_Final_Training"

MODEL_PATH = (
    RESULTS_DIR
    / FINAL_TRAINING_RUN
    / "model"
    / "final_model_epoch82.pth"
)

FINAL_CONFIG_PATH = (
    RESULTS_DIR
    / FINAL_TRAINING_RUN
    / "results"
    / "run_config.json"
)

DEVELOPMENT_MANIFEST_PATH = (
    RESULTS_DIR
    / FINAL_TRAINING_RUN
    / "results"
    / "development_manifest.csv"
)

THRESHOLD_PATH = (
    RESULTS_DIR
    / "Run_05_OOF_Threshold_Selection"
    / "results"
    / "selected_threshold.json"
)

for directory in (
    RESULTS_OUTPUT_DIR,
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
DROPOUT_RATE = 0.3

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


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("\nExperiment:", EXPERIMENT_NAME)
print("Run:", RUN_NAME)
print("Seed:", SEED)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Fusion head: A - Baseline")
print("Input views:", VIEWS)
print("Model source:", FINAL_TRAINING_RUN)
print("Threshold source: Run_05_OOF_Threshold_Selection")
print("Device:", DEVICE)
print("Test set: USED FOR FINAL EVALUATION ONLY")


# ============================================================
# LOAD FINAL TRAINING CONFIGURATION
# ============================================================

print(
    "\nLoading final training configuration..."
)

with open(
    FINAL_CONFIG_PATH,
    "r",
    encoding="utf-8",
) as f:
    FINAL_CONFIG = json.load(f)

if FINAL_CONFIG["experiment"] != EXPERIMENT_NAME:
    raise ValueError(
        "Final training experiment mismatch."
    )

if int(FINAL_CONFIG["epochs"]) != 82:
    raise ValueError(
        "Expected final model trained for 82 epochs."
    )

if tuple(FINAL_CONFIG["input_views"]) != VIEWS:
    raise ValueError(
        "Input view order differs from final training."
    )

if int(FINAL_CONFIG["image_size"]) != IMAGE_SIZE:
    raise ValueError(
        "Image size differs from final training."
    )

global_min = float(
    FINAL_CONFIG["temperature_normalization"]["global_min"]
)

global_max = float(
    FINAL_CONFIG["temperature_normalization"]["global_max"]
)

if not (
    np.isfinite(global_min)
    and np.isfinite(global_max)
    and global_max > global_min
):
    raise ValueError(
        "Invalid saved temperature normalization range."
    )

print("\nTemperature normalization range:")
print("Global minimum:", global_min)
print("Global maximum:", global_max)


# ============================================================
# LOAD FINAL CLASSIFICATION THRESHOLD
# ============================================================

print("\nLoading Run 05 threshold...")

with open(
    THRESHOLD_PATH,
    "r",
    encoding="utf-8",
) as f:
    THRESHOLD_DATA = json.load(f)

if THRESHOLD_DATA["experiment"] != EXPERIMENT_NAME:
    raise ValueError(
        "Threshold experiment mismatch."
    )

THRESHOLD = float(
    THRESHOLD_DATA["selected_threshold"]
)

if not 0.0 <= THRESHOLD <= 1.0:
    raise ValueError(
        "Invalid classification threshold."
    )

if not np.isclose(
    float(FINAL_CONFIG["threshold"]),
    THRESHOLD,
    rtol=0,
    atol=1e-12,
):
    raise ValueError(
        "Run 05 threshold differs from final training."
    )

print(
    "Classification threshold:",
    THRESHOLD,
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
# LOAD TEST PATIENTS
# ============================================================

def load_test_patients():

    records = {}

    for class_name, label in CLASSES.items():

        for view in VIEWS:

            directory = (
                DATA_ROOT
                / "Test"
                / class_name
                / view
            )

            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Directory not found:\n{directory}"
                )

            view_count = 0

            for filepath in sorted(directory.iterdir()):

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
                        "paths": {},
                    }

                record = records[patient_id]

                if record["label"] != label:
                    raise ValueError(
                        "Conflicting class labels for "
                        f"patient {patient_id}"
                    )

                if view in record["paths"]:
                    raise ValueError(
                        f"Duplicate {view} image for "
                        f"patient {patient_id}"
                    )

                record["paths"][view] = str(filepath)

                view_count += 1

            print(
                f"Test | {class_name} | "
                f"{view}: {view_count} images"
            )

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
            "Test patients with missing views: "
            f"{incomplete}"
        )

    complete = sorted(
        records.values(),
        key=lambda record: int(record["patient_id"]),
    )

    return complete


# ============================================================
# LOAD DEVELOPMENT MANIFEST FOR LEAKAGE CHECK
# ============================================================

def load_development_ids():

    if not DEVELOPMENT_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Development manifest not found:\n"
            f"{DEVELOPMENT_MANIFEST_PATH}"
        )

    development_ids = set()

    with open(
        DEVELOPMENT_MANIFEST_PATH,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        if (
            reader.fieldnames is None
            or "patient_id" not in reader.fieldnames
        ):
            raise ValueError(
                "Development manifest missing patient_id."
            )

        for row in reader:

            patient_id = str(
                int(row["patient_id"])
            )

            if patient_id in development_ids:
                raise ValueError(
                    "Duplicate patient in development manifest."
                )

            development_ids.add(patient_id)

    return development_ids


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

test_records = load_test_patients()

development_ids = load_development_ids()

test_ids = {
    record["patient_id"]
    for record in test_records
}

overlap = development_ids.intersection(test_ids)

if overlap:
    raise ValueError(
        "Patient leakage between development and test: "
        f"{sorted(overlap, key=int)}"
    )

print(
    "\nPatient-level leakage check passed."
)

y_test = np.asarray(
    [
        record["label"]
        for record in test_records
    ],
    dtype=np.int32,
)

actual_healthy = int(
    np.sum(y_test == 0)
)

actual_sick = int(
    np.sum(y_test == 1)
)

if len(test_records) == 0:
    raise ValueError(
        "No complete test patients found."
    )

if actual_healthy == 0 or actual_sick == 0:
    raise ValueError(
        "Both classes are required for ROC-AUC."
    )

print("\nTest class distribution:")
print("Healthy:", actual_healthy)
print("Sick:", actual_sick)
print("Total patients:", len(test_records))
print("Total view images:", len(test_records) * 3)


# ============================================================
# TEST PREPROCESSING
#
# Reuses preprocessing.py from training.
# No test-derived normalization parameters.
# No random augmentation.
# ============================================================

test_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Lambda(
        lambda x: x.repeat(3, 1, 1)
    ),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# TEST DATASET
# ============================================================

class MultiViewTestDataset(Dataset):

    def __init__(
        self,
        records,
        global_min,
        global_max,
        transform,
    ):

        self.records = records
        self.global_min = global_min
        self.global_max = global_max
        self.transform = transform

    def __len__(self):

        return len(self.records)

    def __getitem__(self, idx):

        record = self.records[idx]

        view_images = []

        for view in VIEWS:

            filepath = record["paths"][view]

            image = preprocess_image(
                filepath,
                self.global_min,
                self.global_max,
                IMAGE_SIZE,
            )

            image = np.asarray(
                image,
                dtype=np.float32,
            ).squeeze()

            if image.shape != (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ):
                raise ValueError(
                    f"Unexpected image shape: "
                    f"{image.shape}"
                )

            if not np.all(np.isfinite(image)):
                raise ValueError(
                    f"Invalid image values: {filepath}"
                )

            image = torch.from_numpy(
                np.clip(
                    image,
                    0.0,
                    1.0,
                ).copy()
            ).unsqueeze(0)

            image = self.transform(image)

            view_images.append(image)

        images = torch.stack(view_images)

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
# CREATE TEST DATALOADER
# ============================================================

test_dataset = MultiViewTestDataset(
    records=test_records,
    global_min=global_min,
    global_max=global_max,
    transform=test_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)


# ============================================================
# MODEL ARCHITECTURE
#
# Must match Run_07_Final_Training.
# ============================================================

class MultiViewResNet18(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = models.resnet18(
            weights=None,
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

    def forward(self, images):

        batch = images.shape[0]

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
# LOAD FINAL TRAINED MODEL
# ============================================================

print("\nLoading final trained model...")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Final checkpoint not found:\n{MODEL_PATH}"
    )

model = MultiViewResNet18()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False,
)

if int(checkpoint["epoch"]) != 82:
    raise ValueError(
        "Checkpoint epoch mismatch."
    )

if not np.isclose(
    float(checkpoint["threshold"]),
    THRESHOLD,
    rtol=0,
    atol=1e-12,
):
    raise ValueError(
        "Checkpoint threshold mismatch."
    )

checkpoint_range = checkpoint["temperature_range"]

if not np.isclose(
    float(checkpoint_range["global_min"]),
    global_min,
    rtol=0,
    atol=1e-8,
):
    raise ValueError(
        "Checkpoint temperature minimum mismatch."
    )

if not np.isclose(
    float(checkpoint_range["global_max"]),
    global_max,
    rtol=0,
    atol=1e-8,
):
    raise ValueError(
        "Checkpoint temperature maximum mismatch."
    )

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model = model.to(DEVICE)
model.eval()

print("Final model loaded successfully.")
print("Model:", MODEL_PATH)
print("Training epochs:", checkpoint["epoch"])


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

test_probabilities = []
prediction_ids = []

with torch.inference_mode():

    for images, labels, patient_ids in test_loader:

        images = images.to(DEVICE)

        logits = model(images)

        probabilities = torch.sigmoid(
            logits
        )

        test_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

        prediction_ids.extend(
            list(patient_ids)
        )

test_probabilities = np.asarray(
    test_probabilities,
    dtype=np.float64,
)

if prediction_ids != [
    record["patient_id"]
    for record in test_records
]:
    raise ValueError(
        "Test prediction order mismatch."
    )

if len(test_probabilities) != len(y_test):
    raise ValueError(
        "Prediction count mismatch."
    )

if not np.all(np.isfinite(test_probabilities)):
    raise ValueError(
        "Non-finite test probabilities detected."
    )

test_predictions = (
    test_probabilities >= THRESHOLD
).astype(np.int32)


# ============================================================
# TEST METRICS
# ============================================================

test_auc = roc_auc_score(
    y_test,
    test_probabilities,
)

test_pr_auc = average_precision_score(
    y_test,
    test_probabilities,
)

accuracy = accuracy_score(
    y_test,
    test_predictions,
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    test_predictions,
)

sensitivity = recall_score(
    y_test,
    test_predictions,
    pos_label=1,
    zero_division=0,
)

specificity = recall_score(
    y_test,
    test_predictions,
    pos_label=0,
    zero_division=0,
)

precision = precision_score(
    y_test,
    test_predictions,
    pos_label=1,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    test_predictions,
    pos_label=1,
    zero_division=0,
)

mcc = matthews_corrcoef(
    y_test,
    test_predictions,
)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
)

tn, fp, fn, tp = [
    int(value)
    for value in cm.ravel()
]


# ============================================================
# PRINT METRICS
# ============================================================

print("\nFINAL TEST RESULTS")

print(f"ROC-AUC:             {test_auc:.4f}")
print(f"Average Precision:   {test_pr_auc:.4f}")
print(f"Accuracy:            {accuracy:.4f}")
print(f"Balanced Accuracy:   {balanced_accuracy:.4f}")
print(f"Sensitivity:         {sensitivity:.4f}")
print(f"Specificity:         {specificity:.4f}")
print(f"Precision:           {precision:.4f}")
print(f"F1-score:            {f1:.4f}")
print(f"MCC:                 {mcc:.4f}")
print(f"Threshold:           {THRESHOLD:.6f}")

print("\nConfusion Matrix:")
print(cm)

print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_dict = {
    "experiment": EXPERIMENT_NAME,
    "run": RUN_NAME,
    "device": str(DEVICE),
    "seed": SEED,
    "model_source": FINAL_TRAINING_RUN,
    "threshold_source": (
        "Run_05_OOF_Threshold_Selection"
    ),
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "input_views": list(VIEWS),
    "threshold": float(THRESHOLD),
    "temperature_global_min": global_min,
    "temperature_global_max": global_max,
    "test_patients": int(len(y_test)),
    "test_images": int(len(y_test) * 3),
    "test_healthy": actual_healthy,
    "test_sick": actual_sick,
    "roc_auc": float(test_auc),
    "average_precision": float(test_pr_auc),
    "accuracy": float(accuracy),
    "balanced_accuracy": float(balanced_accuracy),
    "sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "precision": float(precision),
    "f1_score": float(f1),
    "matthews_correlation_coefficient": float(mcc),
    "true_negative": tn,
    "false_positive": fp,
    "false_negative": fn,
    "true_positive": tp,
}

METRICS_PATH = (
    RESULTS_OUTPUT_DIR
    / "test_metrics.json"
)

with open(
    METRICS_PATH,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metrics_dict,
        f,
        indent=4,
    )


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

PREDICTIONS_PATH = (
    RESULTS_OUTPUT_DIR
    / "test_predictions.csv"
)

with open(
    PREDICTIONS_PATH,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "patient_id",
        "anterior_filepath",
        "left_oblique_filepath",
        "right_oblique_filepath",
        "true_label",
        "true_class",
        "predicted_probability",
        "predicted_label",
        "predicted_class",
    ])

    for record, probability, prediction in zip(
        test_records,
        test_probabilities,
        test_predictions,
    ):

        true_label = record["label"]

        writer.writerow([
            record["patient_id"],
            record["paths"]["Anterior_ROI"],
            record["paths"]["Left_Oblique_ROI"],
            record["paths"]["Right_Oblique_ROI"],
            int(true_label),
            (
                "Sick"
                if true_label == 1
                else "Healthy"
            ),
            float(probability),
            int(prediction),
            (
                "Sick"
                if prediction == 1
                else "Healthy"
            ),
        ])


# ============================================================
# SAVE RUN CONFIGURATION
# ============================================================

RUN_CONFIG = {
    "experiment": EXPERIMENT_NAME,
    "run": RUN_NAME,
    "device": str(DEVICE),
    "seed": SEED,
    "data_root": str(DATA_ROOT),
    "test_split": "Test",
    "test_used": True,
    "test_patients": int(len(y_test)),
    "test_images": int(len(y_test) * 3),
    "test_healthy": actual_healthy,
    "test_sick": actual_sick,
    "image_size": IMAGE_SIZE,
    "input_shape_per_patient": [
        3,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ],
    "batch_size": BATCH_SIZE,
    "model_source": FINAL_TRAINING_RUN,
    "model_path": str(MODEL_PATH),
    "training_epochs": int(
        FINAL_CONFIG["epochs"]
    ),
    "learning_rate": float(
        FINAL_CONFIG["learning_rate"]
    ),
    "dropout_rate": float(
        FINAL_CONFIG["dropout_rate"]
    ),
    "threshold": float(THRESHOLD),
    "threshold_source": (
        "Run_05_OOF_Threshold_Selection"
    ),
    "temperature_normalization": {
        "method": "global_min_max",
        "range_source": (
            "Run_07_development_data"
        ),
        "global_min": global_min,
        "global_max": global_max,
    },
    "model_architecture": {
        "backbone": "ResNet18",
        "weights": "IMAGENET1K_V1",
        "backbone_frozen": True,
        "shared_backbone": True,
        "fusion_method": "concatenation",
        "fusion_dimension": 1536,
        "classifier": [
            "Linear(1536,64)",
            "ReLU",
            "Dropout(0.3)",
            "Linear(64,1)",
        ],
    },
    "test_evaluation": True,
    "retraining": False,
    "model_selection": False,
    "threshold_optimization": False,
    "patient_overlap_with_development": False,
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
# FINAL TEST ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    y_test,
    test_probabilities,
)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {test_auc:.4f}",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "EXP_01 Multi-View Run_07 - Test ROC Curve"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

ROC_PLOT_PATH = (
    PLOT_DIR
    / "test_roc_curve.png"
)

plt.savefig(
    ROC_PLOT_PATH,
    dpi=300,
)

plt.close()


# ============================================================
# FINAL TEST PRECISION-RECALL CURVE
# ============================================================

pr_precision, pr_recall, _ = (
    precision_recall_curve(
        y_test,
        test_probabilities,
    )
)

plt.figure(figsize=(7, 6))

plt.plot(
    pr_recall,
    pr_precision,
    label=(
        f"Average Precision = "
        f"{test_pr_auc:.4f}"
    ),
)

plt.axhline(
    actual_sick / len(y_test),
    linestyle="--",
    label="Class prevalence",
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "EXP_01 Multi-View Run_07 - Test PR Curve"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

PR_PLOT_PATH = (
    PLOT_DIR
    / "test_pr_curve.png"
)

plt.savefig(
    PR_PLOT_PATH,
    dpi=300,
)

plt.close()


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title(
    "EXP_01 Multi-View Run_07 - Test Confusion Matrix"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    [0, 1],
    ["Healthy", "Sick"],
)

plt.yticks(
    [0, 1],
    ["Healthy", "Sick"],
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
        )

plt.colorbar()
plt.tight_layout()

CM_PLOT_PATH = (
    PLOT_DIR
    / "test_confusion_matrix.png"
)

plt.savefig(
    CM_PLOT_PATH,
    dpi=300,
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\nRUN COMPLETE")

print("Device:", DEVICE)
print("Final test patients:", len(y_test))
print("Final test view images:", len(y_test) * 3)

print(
    "Test ROC-AUC:",
    f"{test_auc:.4f}",
)

print(
    "Test Average Precision:",
    f"{test_pr_auc:.4f}",
)

print(
    "Test Accuracy:",
    f"{accuracy:.4f}",
)

print(
    "Test Balanced Accuracy:",
    f"{balanced_accuracy:.4f}",
)

print(
    "Test Sensitivity:",
    f"{sensitivity:.4f}",
)

print(
    "Test Specificity:",
    f"{specificity:.4f}",
)

print(
    "Test Precision:",
    f"{precision:.4f}",
)

print(
    "Test F1-score:",
    f"{f1:.4f}",
)

print(
    "Test MCC:",
    f"{mcc:.4f}",
)

print("\nResults saved to:")
print(RUN_DIR)

print(
    "\nRun_07 used the final model from Run_06."
)

print(
    "No retraining, model selection, "
    "or threshold optimization was performed."
)

print(
    "The Test set was used only for final evaluation."
)