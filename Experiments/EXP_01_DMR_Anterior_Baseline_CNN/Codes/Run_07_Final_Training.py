import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv

import json

import numpy as np

import tensorflow as tf

from tensorflow import keras  # type: ignore

from tensorflow.keras import layers  # type: ignore

from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt

from preprocessing import (
    calculate_global_temperature_range,
    get_image_paths,
    preprocess_image,
)


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

RUN_NAME = "Run_07_Final_Training"

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

IMAGE_SIZE = 128

BATCH_SIZE = 16

EPOCHS = 89

LEARNING_RATE = 1e-3

DROPOUT_RATE = 0.3

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

print("Device: CPU")

print("Training data: Train + Validation")

print("Development images: 197")

print("Test set: NOT USED")


# ============================================================
# CALCULATE AND NORMALIZE TEMPERATURE RANGE (DEVELOPMENT)
# ============================================================

development_paths = []

for split_name in ["Train", "Validation"]:

    for class_name in CLASSES:

        development_paths.extend(
            get_image_paths(
                DATA_ROOT,
                split_name,
                class_name
            )
        )


global_min, global_max = calculate_global_temperature_range(
    development_paths
)

print("\nTemperature normalization range (Development):")

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


# ============================================================
# COMBINE DEVELOPMENT DATA
# ============================================================

X_development = np.concatenate(
    [
        X_train,
        X_val
    ],
    axis=0
)

y_development = np.concatenate(
    [
        y_train,
        y_val
    ],
    axis=0
)

development_paths = (
    train_paths +
    val_paths
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

print(
    "X_development:",
    X_development.shape
)

print(
    "y_development:",
    y_development.shape
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nDevelopment class distribution:")

for class_name, class_id in CLASSES.items():

    count = np.sum(
        y_development == class_id
    )

    print(
        f"{class_name}: {count}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_ids = np.unique(
    y_development
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=class_ids,
    y=y_development,
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
# TRAINING
# ============================================================

print("STARTING FINAL TRAINING")

print("Device: CPU")

print("Training data: 197 development images")

print("Epochs:", EPOCHS)


history = model.fit(

    X_development,

    y_development,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=CLASS_WEIGHTS,

    shuffle=True,

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
# SAVE FINAL MODEL
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_model.keras"
)

model.save(
    MODEL_PATH
)

WEIGHTS_PATH = os.path.join(
    MODEL_DIR,
    "final_model.weights.h5"
)

model.save_weights(
    WEIGHTS_PATH
)

print("\nFinal model saved:")

print(
    MODEL_PATH
)

print(
    WEIGHTS_PATH
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

    "development_splits":
        [
            "Train",
            "Validation"
        ],

    "train_images":
        int(len(y_train)),

    "validation_images":
        int(len(y_val)),

    "development_images":
        int(len(y_development)),

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
                global_max,
        },

    "threshold":
        None,

    "test_evaluation":
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

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "EXP01 Run 07 - Training Loss"
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

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "ROC-AUC"
)

plt.title(
    "EXP01 Run 07 - Training AUC"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

AUC_PLOT_PATH = os.path.join(
    PLOT_DIR,
    "training_auc.png"
)

plt.savefig(
    AUC_PLOT_PATH,
    dpi=300,
)

plt.close()


# ============================================================
# FINAL DEVICE CHECK
# ============================================================

print("RUN COMPLETE")

print("Device: CPU")

print(
    "\nFinal training epochs:",
    EPOCHS
)

print(
    "Development images:",
    len(y_development)
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

print(
    "Final model is the model state after epoch",
    EPOCHS
)