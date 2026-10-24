import os

import csv
import json

import numpy as np
import torch
import torch.nn as nn

from PIL import Image

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
    auc,
)

import matplotlib.pyplot as plt

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 10

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(
    True,
    warn_only=True
)


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

EXPERIMENT_NAME = (
    "EXP02_DMR_Anterior_ResNet18"
)

RUN_NAME = (
    "Run_07_Final_Test_Evaluation"
)

RUN_DIR = os.path.join(
    RESULTS_DIR,
    RUN_NAME
)

RESULTS_OUTPUT_DIR = os.path.join(
    RUN_DIR,
    "results"
)

PLOT_DIR = os.path.join(
    RUN_DIR,
    "plots"
)

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "Run_06_Final_Training",
    "model",
    "final_model_epoch32.pth"
)

RUN_06_CONFIG_PATH = os.path.join(
    RESULTS_DIR,
    "Run_06_Final_Training",
    "results",
    "run_config.json"
)

RUN_05_THRESHOLD_PATH = os.path.join(
    RESULTS_DIR,
    "Run_05_OOF_Threshold_Selection",
    "results",
    "selected_threshold.json"
)

os.makedirs(
    RESULTS_OUTPUT_DIR,
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

CLASSES = {
    "Healthy": 0,
    "Sick": 1,
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cpu"
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print(
    "\nExperiment:",
    EXPERIMENT_NAME
)

print(
    "Run:",
    RUN_NAME
)

print(
    "Seed:",
    SEED
)

print(
    "Image size:",
    IMAGE_SIZE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Model source:",
    "Run_06_Final_Training"
)

print(
    "Threshold source:",
    "Run_05_OOF_Threshold_Selection"
)

print(
    "Device:",
    DEVICE
)

print(
    "Test set: USED FOR FINAL EVALUATION ONLY"
)


# ============================================================
# LOAD RUN 06 CONFIGURATION
# ============================================================

print(
    "\nLoading Run 06 configuration..."
)

with open(
    RUN_06_CONFIG_PATH,
    "r"
) as f:

    RUN_06_CONFIG = json.load(f)


global_min = float(
    RUN_06_CONFIG[
        "temperature_normalization"
    ][
        "global_min"
    ]
)

global_max = float(
    RUN_06_CONFIG[
        "temperature_normalization"
    ][
        "global_max"
    ]
)

print(
    "\nTemperature normalization range:"
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
# LOAD FINAL CLASSIFICATION THRESHOLD
# ============================================================

print(
    "\nLoading Run 05 threshold..."
)

with open(
    RUN_05_THRESHOLD_PATH,
    "r"
) as f:

    RUN_05_THRESHOLD = json.load(f)


THRESHOLD = float(
    RUN_05_THRESHOLD[
        "selected_threshold"
    ]
)

print(
    "Classification threshold:",
    THRESHOLD
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_raw_thermal_image(path):

    image = Image.open(path)

    image = np.asarray(
        image,
        dtype=np.float32
    )

    return image


def normalize_temperature(
    image,
    global_min,
    global_max,
):

    image = image.copy()

    valid_pixels = (
        image > 0
    )

    image[valid_pixels] = (
        image[valid_pixels]
        - global_min
    ) / (
        global_max
        - global_min
    )

    image[valid_pixels] = np.clip(
        image[valid_pixels],
        0.0,
        1.0
    )

    image[~valid_pixels] = 0.0

    return image


def pad_to_square(image):

    height, width = image.shape

    size = max(
        height,
        width
    )

    padded = np.zeros(
        (size, size),
        dtype=np.float32
    )

    top = (
        size - height
    ) // 2

    left = (
        size - width
    ) // 2

    padded[
        top:top + height,
        left:left + width
    ] = image

    return padded


def preprocess_image(
    path,
    global_min,
    global_max,
    image_size,
):

    image = load_raw_thermal_image(
        path
    )

    image = normalize_temperature(
        image,
        global_min,
        global_max,
    )

    image = pad_to_square(
        image
    )

    image = (
        image * 255.0
    ).astype(
        np.uint8
    )

    image = Image.fromarray(
        image,
        mode="L"
    )

    image = image.resize(
        (
            image_size,
            image_size
        ),
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    image = torch.from_numpy(
        image
    ).unsqueeze(
        0
    )

    image = image.repeat(
        3,
        1,
        1
    )

    mean = torch.tensor(
        [
            0.485,
            0.456,
            0.406,
        ],
        dtype=torch.float32
    ).view(
        3,
        1,
        1
    )

    std = torch.tensor(
        [
            0.229,
            0.224,
            0.225,
        ],
        dtype=torch.float32
    ).view(
        3,
        1,
        1
    )

    image = (
        image - mean
    ) / std

    return image


# ============================================================
# LOAD TEST SPLIT
# ============================================================

def get_image_paths(
    data_root,
    split_name,
    class_name,
):

    directory = os.path.join(
        data_root,
        split_name,
        class_name
    )

    extensions = (
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
    )

    paths = []

    for filename in sorted(
        os.listdir(directory)
    ):

        if filename.lower().endswith(
            extensions
        ):

            paths.append(
                os.path.join(
                    directory,
                    filename
                )
            )

    return paths


def load_split(
    split_name
):

    images = []

    labels = []

    filepaths = []

    for class_name, class_id in CLASSES.items():

        paths = get_image_paths(
            DATA_ROOT,
            split_name,
            class_name
        )

        print(
            f"{split_name} | "
            f"{class_name}: "
            f"{len(paths)} images"
        )

        for path in paths:

            image = preprocess_image(
                path,
                global_min=global_min,
                global_max=global_max,
                image_size=IMAGE_SIZE,
            )

            images.append(
                image
            )

            labels.append(
                class_id
            )

            filepaths.append(
                path
            )

    images = torch.stack(
        images
    )

    labels = np.asarray(
        labels,
        dtype=np.int32
    )

    return (
        images,
        labels,
        filepaths
    )


# ============================================================
# LOAD FINAL TRAINED MODEL
# ============================================================

# ============================================================
# LOAD FINAL TRAINED MODEL
# ============================================================

print(
    "\nLoading final trained model..."
)

model = resnet18(
    weights=ResNet18_Weights.IMAGENET1K_V1
)

model.fc = nn.Sequential(
    nn.Linear(
        512,
        64
    ),

    nn.ReLU(),

    nn.Dropout(
        p=0.3
    ),

    nn.Linear(
        64,
        1
    )
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# Run 06 saves a complete checkpoint dictionary.
state_dict = checkpoint[
    "model_state_dict"
]


model.load_state_dict(
    state_dict
)

model = model.to(
    DEVICE
)

model.eval()


print(
    "Final model loaded successfully."
)

print(
    "Model:",
    MODEL_PATH
)


# ============================================================
# LOAD TEST DATA
# ============================================================

print(
    "\nLoading test data..."
)

X_test, y_test, test_paths = load_split(
    "Test"
)

print(
    "\nDataset shapes:"
)

print(
    "X_test:",
    tuple(X_test.shape)
)

print(
    "y_test:",
    y_test.shape
)


# ============================================================
# TEST CLASS DISTRIBUTION
# ============================================================

print(
    "\nTest class distribution:"
)

for class_name, class_id in CLASSES.items():

    count = np.sum(
        y_test == class_id
    )

    print(
        f"{class_name}: {count}"
    )


# ============================================================
# TEST DATA CHECK
# ============================================================

expected_test_images = 37

if len(y_test) != expected_test_images:

    raise ValueError(
        "Unexpected test-set size. "
        f"Expected {expected_test_images}, "
        f"but found {len(y_test)}."
    )


expected_healthy = 24

expected_sick = 13

actual_healthy = int(
    np.sum(
        y_test == 0
    )
)

actual_sick = int(
    np.sum(
        y_test == 1
    )
)

if actual_healthy != expected_healthy:

    raise ValueError(
        "Unexpected number of Healthy test images. "
        f"Expected {expected_healthy}, "
        f"but found {actual_healthy}."
    )

if actual_sick != expected_sick:

    raise ValueError(
        "Unexpected number of Sick test images. "
        f"Expected {expected_sick}, "
        f"but found {actual_sick}."
    )


print(
    "\nTest-set integrity check passed."
)

print(
    "Total test images:",
    len(y_test)
)

print(
    "Healthy:",
    actual_healthy
)

print(
    "Sick:",
    actual_sick
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print(
    "\nGenerating test predictions..."
)

test_probabilities = []

with torch.no_grad():

    for start in range(
        0,
        len(X_test),
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            len(X_test)
        )

        batch = X_test[
            start:end
        ].to(
            DEVICE
        )

        logits = model(
            batch
        ).view(
            -1
        )

        probabilities = torch.sigmoid(
            logits
        )

        test_probabilities.extend(
            probabilities.cpu().numpy()
        )


test_probabilities = np.asarray(
    test_probabilities,
    dtype=np.float64
)


test_predictions = (
    test_probabilities >= THRESHOLD
).astype(
    np.int32
)


# ============================================================
# TEST METRICS
# ============================================================

test_auc = roc_auc_score(
    y_test,
    test_probabilities
)

accuracy = accuracy_score(
    y_test,
    test_predictions
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    test_predictions
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
    test_predictions
)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# PRINT METRICS
# ============================================================

print(
    "\nFINAL TEST RESULTS"
)

print(
    f"ROC-AUC:             {test_auc:.4f}"
)

print(
    f"Accuracy:            {accuracy:.4f}"
)

print(
    f"Balanced Accuracy:   {balanced_accuracy:.4f}"
)

print(
    f"Sensitivity:         {sensitivity:.4f}"
)

print(
    f"Specificity:         {specificity:.4f}"
)

print(
    f"Precision:           {precision:.4f}"
)

print(
    f"F1-score:            {f1:.4f}"
)

print(
    f"MCC:                 {mcc:.4f}"
)

print(
    f"Threshold:           {THRESHOLD:.6f}"
)

print(
    "\nConfusion Matrix:"
)

print(
    cm
)

print(
    "\nTN:",
    tn
)

print(
    "FP:",
    fp
)

print(
    "FN:",
    fn
)

print(
    "TP:",
    tp
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_dict = {

    "experiment":
        EXPERIMENT_NAME,

    "run":
        RUN_NAME,

    "device":
        "cpu",

    "seed":
        SEED,

    "model_source":
        "Run_06_Final_Training",

    "threshold_source":
        "Run_05_OOF_Threshold_Selection",

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "threshold":
        THRESHOLD,

    "temperature_global_min":
        global_min,

    "temperature_global_max":
        global_max,

    "test_images":
        int(len(y_test)),

    "test_healthy":
        int(actual_healthy),

    "test_sick":
        int(actual_sick),

    "roc_auc":
        float(test_auc),

    "accuracy":
        float(accuracy),

    "balanced_accuracy":
        float(balanced_accuracy),

    "sensitivity":
        float(sensitivity),

    "specificity":
        float(specificity),

    "precision":
        float(precision),

    "f1_score":
        float(f1),

    "matthews_correlation_coefficient":
        float(mcc),

    "true_negative":
        int(tn),

    "false_positive":
        int(fp),

    "false_negative":
        int(fn),

    "true_positive":
        int(tp),

}

METRICS_PATH = os.path.join(
    RESULTS_OUTPUT_DIR,
    "test_metrics.json"
)

with open(
    METRICS_PATH,
    "w"
) as f:

    json.dump(
        metrics_dict,
        f,
        indent=4,
    )


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

PREDICTIONS_PATH = os.path.join(
    RESULTS_OUTPUT_DIR,
    "test_predictions.csv"
)

with open(
    PREDICTIONS_PATH,
    "w",
    newline=""
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow(
        [
            "filepath",
            "true_label",
            "true_class",
            "predicted_probability",
            "predicted_label",
            "predicted_class",
        ]
    )

    for (
        path,
        true_label,
        probability,
        prediction
    ) in zip(
        test_paths,
        y_test,
        test_probabilities,
        test_predictions,
    ):

        writer.writerow(
            [
                path,

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
            ]
        )


# ============================================================
# SAVE RUN CONFIGURATION
# ============================================================

RUN_CONFIG = {

    "experiment":
        EXPERIMENT_NAME,

    "run":
        RUN_NAME,

    "device":
        "cpu",

    "seed":
        SEED,

    "data_root":
        DATA_ROOT,

    "test_split":
        "Test",

    "test_used":
        True,

    "test_images":
        int(len(y_test)),

    "test_healthy":
        int(actual_healthy),

    "test_sick":
        int(actual_sick),

    "image_size":
        IMAGE_SIZE,

    "input_shape":
        [
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        ],

    "batch_size":
        BATCH_SIZE,

    "model_source":
        "Run_06_Final_Training",

    "model_path":
        MODEL_PATH,

    "training_epochs":
        int(
            RUN_06_CONFIG[
                "epochs"
            ]
        ),

    "learning_rate":
        float(
            RUN_06_CONFIG[
                "learning_rate"
            ]
        ),

    "dropout_rate":
        float(
            RUN_06_CONFIG[
                "dropout_rate"
            ]
        ),

    "threshold":
        THRESHOLD,

    "threshold_source":
        "Run_05_OOF_Threshold_Selection",

    "temperature_normalization":
        {
            "method":
                "global_min_max",

            "range_source":
                "Run_06_development_data",

            "global_min":
                global_min,

            "global_max":
                global_max,
        },

    "model_architecture":
        {
            "backbone":
                "ResNet18",

            "weights":
                "IMAGENET1K_V1",

            "backbone_frozen":
                True,

            "classifier":
                [
                    "Linear(512,64)",
                    "ReLU",
                    "Dropout(0.3)",
                    "Linear(64,1)"
                ],

        },

    "test_evaluation":
        True,

    "retraining":
        False,

    "model_selection":
        False,

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
        indent=4,
    )


# ============================================================
# FINAL TEST ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    test_probabilities
)

roc_auc_value = auc(
    fpr,
    tpr
)

plt.figure(
    figsize=(7, 6)
)

plt.plot(
    fpr,
    tpr,
    label=(
        f"ROC-AUC = "
        f"{roc_auc_value:.4f}"
    ),
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "EXP02 Run 07 - Test ROC Curve"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

ROC_PLOT_PATH = os.path.join(
    PLOT_DIR,
    "test_roc_curve.png"
)

plt.savefig(
    ROC_PLOT_PATH,
    dpi=300,
)

plt.close()


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(
    cm
)

plt.title(
    "EXP02 Run 07 - Test Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    [0, 1],
    ["Healthy", "Sick"]
)

plt.yticks(
    [0, 1],
    ["Healthy", "Sick"]
)

for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
        )

plt.colorbar()

plt.tight_layout()

CM_PLOT_PATH = os.path.join(
    PLOT_DIR,
    "test_confusion_matrix.png"
)

plt.savefig(
    CM_PLOT_PATH,
    dpi=300,
)

plt.close()


# ============================================================
# FINAL DEVICE CHECK
# ============================================================

print(
    "\nRUN COMPLETE"
)

print(
    "Device: CPU"
)

print(
    "\nFinal test images:",
    len(y_test)
)

print(
    "Test ROC-AUC:",
    f"{test_auc:.4f}"
)

print(
    "Test Accuracy:",
    f"{accuracy:.4f}"
)

print(
    "Test Balanced Accuracy:",
    f"{balanced_accuracy:.4f}"
)

print(
    "Test Sensitivity:",
    f"{sensitivity:.4f}"
)

print(
    "Test Specificity:",
    f"{specificity:.4f}"
)

print(
    "Test Precision:",
    f"{precision:.4f}"
)

print(
    "Test F1-score:",
    f"{f1:.4f}"
)

print(
    "Test MCC:",
    f"{mcc:.4f}"
)

print(
    "\nResults saved to:"
)

print(
    RUN_DIR
)

print(
    "\nRun 07 used the final model from Run 06."
)

print(
    "No retraining or model selection was performed."
)

print(
    "The Test set was used only for final evaluation."
)