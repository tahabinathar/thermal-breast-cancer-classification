import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import json

import numpy as np
import tensorflow as tf

from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)

from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt

from preprocessing import calculate_global_temperature_range, get_image_paths, preprocess_image


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 10

keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# ============================================================
# PATHS
# ============================================================

DATA_ROOT = (
    r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography\Dataset\DMR_IR_Anterior_View_ROI"
)

RESULTS_DIR = (
    r"D:\HITEC\STUDENT\NEW PROJECT\Breast_Thermography\Experiments\EXP_01_DMR_Anterior_Baseline_CNN\Results"
)

EXPERIMENT_NAME = "EXP01_DMR_Anterior_Baseline"
RUN_NAME = "Run_01_Baseline_CNN"

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

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.5
THRESHOLD = 0.5

CLASSES = {
    "Healthy": 0,
    "Sick": 1,
}


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
print("Threshold:", THRESHOLD)
print("Device: CPU")
print("Test set: NOT USED")


# ============================================================
# CALCULATE AND NORMALIZE TEMPERATURE RANGE (TRAINING)
# ============================================================


train_paths = []

for class_name in CLASSES:
    train_paths.extend(
        get_image_paths(
            DATA_ROOT,
            "Train",
            class_name
        )
    )

global_min, global_max = calculate_global_temperature_range(
    train_paths
)

print("\nTemperature normalization range (Training):")
print("Global minimum:", global_min)
print("Global maximum:", global_max)

# ============================================================
# LOAD SPLIT
# ============================================================

def load_split(split_name):

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
            f"{class_name}: {len(paths)} images"
        )

        for path in paths:

            image = preprocess_image(
                path,
                global_min=global_min,
                global_max=global_max,
                image_size=IMAGE_SIZE,
            )

            images.append(image)
            labels.append(class_id)
            filepaths.append(path)

    images = np.asarray(
        images,
        dtype=np.float32
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
# LOAD TRAINING AND VALIDATION DATA
# ============================================================

print("\nLoading training data...")

X_train, y_train, train_paths = load_split(
    "Train"
)

print("\nLoading validation data...")

X_val, y_val, val_paths = load_split(
    "Validation"
)

print("\nDataset shapes:")

print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "y_val:",
    y_val.shape
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nTraining class distribution:")

for class_name, class_id in CLASSES.items():

    count = np.sum(
        y_train == class_id
    )

    print(
        f"{class_name}: {count}"
    )


print("\nValidation class distribution:")

for class_name, class_id in CLASSES.items():

    count = np.sum(
        y_val == class_id
    )

    print(
        f"{class_name}: {count}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_ids = np.unique(
    y_train
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=class_ids,
    y=y_train,
)

CLASS_WEIGHTS = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        class_ids,
        weights
    )
}

print("\nClass weights:")
print(
    CLASS_WEIGHTS
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

def create_data_augmentation():

    return keras.Sequential(
        [
            layers.RandomRotation(
                factor=0.05
            ),

            layers.RandomZoom(
                height_factor=0.05,
                width_factor=0.05
            ),
        ],
        name="data_augmentation",
    )


# ============================================================
# MODEL
# ============================================================

def build_model():

    model = keras.Sequential(
        [
            layers.Input(
                shape=(
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    1
                )
            ),

            create_data_augmentation(),

            layers.Conv2D(
                32,
                (2, 2),
                activation="relu",
                padding="same",
            ),

            layers.Conv2D(
                32,
                (2, 2),
                activation="relu",
                padding="same",
            ),

            layers.MaxPooling2D(),

            layers.Conv2D(
                64,
                (2, 2),
                activation="relu",
                padding="same",
            ),

            layers.MaxPooling2D(),

            layers.Conv2D(
                128,
                (2, 2),
                activation="relu",
                padding="same",
            ),

            layers.MaxPooling2D(),

            layers.GlobalAveragePooling2D(),

            layers.Dense(
                64,
                activation="relu",
            ),

            layers.Dropout(
                DROPOUT_RATE
            ),

            layers.Dense(
                1,
                activation="sigmoid",
            ),
        ],
        name="Run_01_Baseline_CNN",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy",

            keras.metrics.AUC(
                name="auc"
            ),
        ],
    )

    return model


# ============================================================
# MODEL SUMMARY
# ============================================================

model = build_model()

print("\nModel summary:")

model.summary()


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.keras"
)

checkpoint = keras.callbacks.ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1,
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("Device: CPU")
print("=" * 60)

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=CLASS_WEIGHTS,

    callbacks=[
        checkpoint
    ],

    verbose=1,
)


# ============================================================
# HISTORY
# ============================================================

history_dict = {
    key: [
        float(value)
        for value in values
    ]

    for key, values
    in history.history.items()
}

HISTORY_PATH = os.path.join(
    RESULTS_OUTPUT_DIR,
    "training_history.json"
)

with open(
    HISTORY_PATH,
    "w"
) as f:

    json.dump(
        history_dict,
        f,
        indent=4,
    )


# ============================================================
# BEST EPOCH
# ============================================================

val_auc_history = np.asarray(
    history.history["val_auc"]
)

best_epoch_index = int(
    np.argmax(
        val_auc_history
    )
)

best_epoch = (
    best_epoch_index + 1
)

best_val_auc = float(
    val_auc_history[
        best_epoch_index
    ]
)

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation AUC:",
    best_val_auc
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

best_model = keras.models.load_model(
    CHECKPOINT_PATH
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

val_probabilities = (
    best_model.predict(
        X_val,
        batch_size=BATCH_SIZE,
        verbose=1,
    )
    .ravel()
)

val_predictions = (
    val_probabilities >= THRESHOLD
).astype(
    np.int32
)


# ============================================================
# VALIDATION METRICS
# ============================================================

val_auc = roc_auc_score(
    y_val,
    val_probabilities
)

accuracy = accuracy_score(
    y_val,
    val_predictions
)

sensitivity = recall_score(
    y_val,
    val_predictions,
    pos_label=1,
    zero_division=0,
)

specificity = recall_score(
    y_val,
    val_predictions,
    pos_label=0,
    zero_division=0,
)

precision = precision_score(
    y_val,
    val_predictions,
    pos_label=1,
    zero_division=0,
)

f1 = f1_score(
    y_val,
    val_predictions,
    pos_label=1,
    zero_division=0,
)

cm = confusion_matrix(
    y_val,
    val_predictions,
    labels=[0, 1],
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# PRINT METRICS
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(
    f"ROC-AUC:     {val_auc:.4f}"
)

print(
    f"Accuracy:    {accuracy:.4f}"
)

print(
    f"Sensitivity: {sensitivity:.4f}"
)

print(
    f"Specificity: {specificity:.4f}"
)

print(
    f"Precision:   {precision:.4f}"
)

print(
    f"F1-score:    {f1:.4f}"
)

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

    "experiment":
        EXPERIMENT_NAME,

    "run":
        RUN_NAME,

    "device":
        "cpu",

    "seed":
        SEED,

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "dropout_rate":
        DROPOUT_RATE,

    "threshold":
        THRESHOLD,

    "best_epoch":
        best_epoch,

    "best_validation_auc_during_training":
        best_val_auc,

    "validation_auc":
        float(val_auc),

    "accuracy":
        float(accuracy),

    "sensitivity":
        float(sensitivity),

    "specificity":
        float(specificity),

    "precision":
        float(precision),

    "f1_score":
        float(f1),

    "true_negative":
        int(tn),

    "false_positive":
        int(fp),

    "false_negative":
        int(fn),

    "true_positive":
        int(tp),

    "temperature_global_min":
        global_min,

    "temperature_global_max":
        global_max,

    "train_images":
        int(len(y_train)),

    "validation_images":
        int(len(y_val)),
}

METRICS_PATH = os.path.join(
    RESULTS_OUTPUT_DIR,
    "validation_metrics.json"
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
# SAVE VALIDATION PREDICTIONS
# ============================================================

PREDICTIONS_PATH = os.path.join(
    RESULTS_OUTPUT_DIR,
    "validation_predictions.csv"
)

with open(
    PREDICTIONS_PATH,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

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
        val_paths,
        y_val,
        val_probabilities,
        val_predictions,
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

    "train_split":
        "Train",

    "validation_split":
        "Validation",

    "test_used":
        False,

    "image_size":
        IMAGE_SIZE,

    "input_shape":
        [
            IMAGE_SIZE,
            IMAGE_SIZE,
            1
        ],

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

    "augmentation":
        {
            "random_rotation": 0.05,
            "random_zoom": 0.05,
            "horizontal_flip": False,
        },

    "class_weighting":
        True,

    "class_weights":
        CLASS_WEIGHTS,

    "early_stopping":
        False,

    "deterministic":
        True,

    "model_selection_metric":
        "validation_auc",

    "checkpoint_metric":
        "val_auc",

    "threshold":
        THRESHOLD,

    "temperature_normalization":
        {
            "method":
                "global_min_max",

            "range_source":
                "training_only",

            "global_min":
                global_min,

            "global_max":
                global_max,
        },

    "best_epoch":
        best_epoch,

    "best_validation_auc":
        best_val_auc,
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
# PLOT TRAINING LOSS
# ============================================================

epochs_axis = range(
    1,
    len(
        history.history["loss"]
    ) + 1
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_axis,
    history.history["loss"],
    label="Training Loss",
)

plt.plot(
    epochs_axis,
    history.history["val_loss"],
    label="Validation Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "EXP01 Run 01 - Training and Validation Loss"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

LOSS_PLOT_PATH = os.path.join(
    PLOT_DIR,
    "training_validation_loss.png"
)

plt.savefig(
    LOSS_PLOT_PATH,
    dpi=300,
)

plt.close()


# ============================================================
# PLOT TRAINING AUC
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs_axis,
    history.history["auc"],
    label="Training AUC",
)

plt.plot(
    epochs_axis,
    history.history["val_auc"],
    label="Validation AUC",
)

plt.axvline(
    best_epoch,
    linestyle="--",
    label=f"Best Epoch ({best_epoch})",
)

plt.xlabel("Epoch")
plt.ylabel("ROC-AUC")

plt.title(
    "EXP01 Run 01 - Training and Validation AUC"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

AUC_PLOT_PATH = os.path.join(
    PLOT_DIR,
    "training_validation_auc.png"
)

plt.savefig(
    AUC_PLOT_PATH,
    dpi=300,
)

plt.close()


# ============================================================
# VALIDATION ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_val,
    val_probabilities
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
    label=f"ROC-AUC = {roc_auc_value:.4f}",
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
    "EXP01 Run 01 - Validation ROC Curve"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

ROC_PLOT_PATH = os.path.join(
    PLOT_DIR,
    "validation_roc_curve.png"
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

plt.imshow(cm)

plt.title(
    "EXP01 Run 01 - Validation Confusion Matrix"
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
    "validation_confusion_matrix.png"
)

plt.savefig(
    CM_PLOT_PATH,
    dpi=300,
)

plt.close()


# ============================================================
# FINAL DEVICE CHECK
# ============================================================

print("RUN COMPLETE")

print("Device: CPU")

print(
    "\nBest epoch:",
    best_epoch
)

print(
    "Best validation AUC:",
    f"{best_val_auc:.4f}"
)

print(
    "\nResults saved to:"
)

print(
    RUN_DIR
)

print(
    "\nTest set was NOT loaded or evaluated."
)