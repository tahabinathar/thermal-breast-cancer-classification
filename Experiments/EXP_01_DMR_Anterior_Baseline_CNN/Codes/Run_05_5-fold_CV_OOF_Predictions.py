import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import json
import gc

import numpy as np
import tensorflow as tf

from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

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

RUN_NAME = "Run_05_5-fold_CV_OOF_Predictions"

RUN_DIR = os.path.join(
    RESULTS_DIR,
    RUN_NAME
)

RESULTS_OUTPUT_DIR = os.path.join(
    RUN_DIR,
    "results"
)

os.makedirs(
    RESULTS_OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

IMAGE_SIZE = 128

BATCH_SIZE = 16

FIXED_EPOCH = 89

LEARNING_RATE = 1e-3

DROPOUT_RATE = 0.3

N_FOLDS = 5

CLASSES = {
    "Healthy": 0,
    "Sick": 1,
}

RUN_DESCRIPTION = (
    "Generate out-of-fold (OOF) predictions for the final EXP01 "
    "CNN configuration using stratified 5-fold cross-validation "
    "on the pooled development dataset. The learning rate selected "
    "in Run 02, dropout rate selected in Run 03, and fixed training "
    "epoch selected in Run 04 are held constant. Each fold is trained "
    "for exactly 89 epochs without validation-based checkpoint "
    "selection. The test set is not used, and no classification "
    "threshold is applied in this run."
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("\nExperiment:", EXPERIMENT_NAME)
print("Run:", RUN_NAME)
print("Seed:", SEED)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Fixed epoch:", FIXED_EPOCH)
print("Learning rate:", LEARNING_RATE)
print("Dropout rate:", DROPOUT_RATE)
print("Number of folds:", N_FOLDS)
print("Training duration: Fixed")
print("Output: OOF predictions")
print("Threshold: NOT APPLIED")
print("Device: CPU")
print("Test set: NOT USED")


# ============================================================
# COLLECT DATASET PATHS FOR CROSS-VALIDATION
# ============================================================

all_paths = []
all_labels = []

for class_name, class_label in CLASSES.items():

    train_paths = get_image_paths(
        DATA_ROOT,
        "Train",
        class_name
    )

    validation_paths = get_image_paths(
        DATA_ROOT,
        "Validation",
        class_name
    )

    print(
        f"Train - {class_name}: "
        f"{len(train_paths)} images"
    )

    print(
        f"Validation - {class_name}: "
        f"{len(validation_paths)} images"
    )

    class_paths = train_paths + validation_paths

    all_paths.extend(class_paths)

    all_labels.extend(
        [class_label] * len(class_paths)
    )


all_paths = np.asarray(all_paths)

all_labels = np.asarray(
    all_labels,
    dtype=np.int32
)


print("\nPooled dataset for cross-validation:")
print("Total images:", len(all_paths))
print("Healthy:", np.sum(all_labels == 0))
print("Sick:", np.sum(all_labels == 1))
print("\nTest set: NOT LOADED OR USED")


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
                width_factor=0.05,
            ),
        ],
        name="data_augmentation",
    )


# ============================================================
# MODEL BUILDER
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
                activation="relu"
            ),

            layers.Dropout(
                DROPOUT_RATE
            ),

            layers.Dense(
                1,
                activation="sigmoid"
            ),
        ]
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
# OOF STORAGE
# ============================================================

oof_predictions = np.full(
    len(all_paths),
    np.nan,
    dtype=np.float32
)

oof_fold = np.full(
    len(all_paths),
    -1,
    dtype=np.int32
)

fold_summaries = []


# ============================================================
# STRATIFIED 5-FOLD CROSS-VALIDATION
# ============================================================

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)


for fold_number, (
    train_indices,
    val_indices
) in enumerate(
    skf.split(
        all_paths,
        all_labels
    ),
    start=1
):


    print(
        f"FOLD {fold_number}/{N_FOLDS}"
    )


    # --------------------------------------------------------
    # Fold paths and labels
    # --------------------------------------------------------

    fold_train_paths = all_paths[
        train_indices
    ]

    fold_val_paths = all_paths[
        val_indices
    ]

    y_train = all_labels[
        train_indices
    ]

    y_val = all_labels[
        val_indices
    ]


    print(
        "\nTraining images:",
        len(fold_train_paths)
    )

    print(
        "Validation images:",
        len(fold_val_paths)
    )

    print(
        "Training Healthy:",
        np.sum(y_train == 0)
    )

    print(
        "Training Sick:",
        np.sum(y_train == 1)
    )

    print(
        "Validation Healthy:",
        np.sum(y_val == 0)
    )

    print(
        "Validation Sick:",
        np.sum(y_val == 1)
    )


    # --------------------------------------------------------
    # Fold-specific temperature range
    # --------------------------------------------------------

    (
        fold_global_min,
        fold_global_max
    ) = calculate_global_temperature_range(
        fold_train_paths
    )


    print(
        "\nFold temperature range:"
    )

    print(
        "Min:",
        fold_global_min
    )

    print(
        "Max:",
        fold_global_max
    )


    # --------------------------------------------------------
    # Preprocess fold training data
    # --------------------------------------------------------

    X_train = np.stack(
        [
            preprocess_image(
                filepath,
                fold_global_min,
                fold_global_max,
                IMAGE_SIZE
            )

            for filepath in fold_train_paths
        ]
    )


    # --------------------------------------------------------
    # Preprocess fold validation data
    # --------------------------------------------------------

    X_val = np.stack(
        [
            preprocess_image(
                filepath,
                fold_global_min,
                fold_global_max,
                IMAGE_SIZE
            )

            for filepath in fold_val_paths
        ]
    )


    print(
        "\nX_train shape:",
        X_train.shape
    )

    print(
        "X_val shape:",
        X_val.shape
    )


    # --------------------------------------------------------
    # Fold-specific class weights
    # --------------------------------------------------------

    classes = np.array(
        [0, 1]
    )

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(class_label): float(weight)

        for class_label, weight
        in zip(
            classes,
            class_weights_array
        )
    }


    print(
        "\nClass weights:"
    )

    print(
        class_weights
    )


    # --------------------------------------------------------
    # Reproducibility for this fold
    # --------------------------------------------------------

    keras.backend.clear_session()

    gc.collect()

    keras.utils.set_random_seed(
        SEED
    )


    # --------------------------------------------------------
    # Build fresh model
    # --------------------------------------------------------

    model = build_model()


    # --------------------------------------------------------
    # Train for exactly the selected fixed epoch
    # --------------------------------------------------------

    print(
        f"\nTraining for exactly "
        f"{FIXED_EPOCH} epochs..."
    )

    history = model.fit(
        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=FIXED_EPOCH,

        batch_size=BATCH_SIZE,

        class_weight=class_weights,

        verbose=1,
    )


    # --------------------------------------------------------
    # Generate OOF predictions
    # --------------------------------------------------------

    print(
        "\nGenerating OOF predictions..."
    )

    fold_predictions = (
        model.predict(
            X_val,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        .reshape(-1)
        .astype(np.float32)
    )


    # --------------------------------------------------------
    # Store OOF predictions
    # --------------------------------------------------------

    oof_predictions[
        val_indices
    ] = fold_predictions

    oof_fold[
        val_indices
    ] = fold_number


    # --------------------------------------------------------
    # Verify fold OOF predictions
    # --------------------------------------------------------

    fold_auc = roc_auc_score(
        y_val,
        fold_predictions
    )


    print(
        "\nFold OOF ROC-AUC:",
        fold_auc
    )


    # --------------------------------------------------------
    # Save fold predictions
    # --------------------------------------------------------

    fold_prediction_path = os.path.join(
        RESULTS_OUTPUT_DIR,
        f"fold_{fold_number}_predictions.csv"
    )


    with open(
        fold_prediction_path,
        "w",
        newline=""
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow(
            [
                "index",
                "filepath",
                "true_label",
                "oof_probability",
            ]
        )


        for local_index, global_index in enumerate(
            val_indices
        ):

            writer.writerow(
                [
                    int(global_index),

                    all_paths[
                        global_index
                    ],

                    int(
                        y_val[
                            local_index
                        ]
                    ),

                    float(
                        fold_predictions[
                            local_index
                        ]
                    ),
                ]
            )


    # --------------------------------------------------------
    # Save fold summary
    # --------------------------------------------------------

    fold_summary = {

        "fold":
            fold_number,

        "n_train":
            int(
                len(train_indices)
            ),

        "n_validation":
            int(
                len(val_indices)
            ),

        "train_healthy":
            int(
                np.sum(
                    y_train == 0
                )
            ),

        "train_sick":
            int(
                np.sum(
                    y_train == 1
                )
            ),

        "validation_healthy":
            int(
                np.sum(
                    y_val == 0
                )
            ),

        "validation_sick":
            int(
                np.sum(
                    y_val == 1
                )
            ),

        "temperature_min":
            float(
                fold_global_min
            ),

        "temperature_max":
            float(
                fold_global_max
            ),

        "class_weights":
            class_weights,

        "fixed_epoch":
            int(
                FIXED_EPOCH
            ),

        "oof_roc_auc":
            float(
                fold_auc
            ),
    }


    fold_summaries.append(
        fold_summary
    )


    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = os.path.join(
        RESULTS_OUTPUT_DIR,
        f"fold_{fold_number}_training_history.json"
    )


    history_dict = {
        key: [
            float(value)
            for value in values
        ]

        for key, values
        in history.history.items()
    }


    with open(
        history_path,
        "w"
    ) as json_file:

        json.dump(
            history_dict,
            json_file,
            indent=4
        )


    # --------------------------------------------------------
    # Clean up fold
    # --------------------------------------------------------

    del model
    del X_train
    del X_val
    del y_train
    del y_val
    del history
    del fold_predictions

    tf.keras.backend.clear_session()

    gc.collect()


# ============================================================
# VERIFY OOF COVERAGE
# ============================================================

print(
    "OOF COVERAGE CHECK"
)

missing_predictions = np.isnan(
    oof_predictions
)

missing_folds = (
    oof_fold == -1
)


print(
    "Total development images:",
    len(all_paths)
)

print(
    "OOF predictions generated:",
    np.sum(
        ~missing_predictions
    )
)

print(
    "Missing OOF predictions:",
    np.sum(
        missing_predictions
    )
)

print(
    "Missing fold assignments:",
    np.sum(
        missing_folds
    )
)


if np.any(
    missing_predictions
):

    raise RuntimeError(
        "Some development images do not have OOF predictions."
    )


if np.any(
    missing_folds
):

    raise RuntimeError(
        "Some development images do not have fold assignments."
    )


# ============================================================
# SAVE COMBINED OOF PREDICTIONS
# ============================================================

oof_prediction_path = os.path.join(
    RESULTS_OUTPUT_DIR,
    "oof_predictions.csv"
)


with open(
    oof_prediction_path,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(
        csv_file
    )

    writer.writerow(
        [
            "index",
            "filepath",
            "true_label",
            "oof_fold",
            "oof_probability",
        ]
    )


    for index in range(
        len(all_paths)
    ):

        writer.writerow(
            [
                int(index),

                all_paths[
                    index
                ],

                int(
                    all_labels[
                        index
                    ]
                ),

                int(
                    oof_fold[
                        index
                    ]
                ),

                float(
                    oof_predictions[
                        index
                    ]
                ),
            ]
        )


# ============================================================
# OVERALL OOF ROC-AUC
# ============================================================

overall_oof_auc = roc_auc_score(
    all_labels,
    oof_predictions
)


print(
    "\nOverall OOF ROC-AUC:",
    overall_oof_auc
)


# ============================================================
# SAVE OOF SUMMARY
# ============================================================

oof_summary = {

    "experiment_name":
        EXPERIMENT_NAME,

    "run_name":
        RUN_NAME,

    "run_description":
        RUN_DESCRIPTION,

    "seed":
        SEED,

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "fixed_epoch":
        FIXED_EPOCH,

    "optimizer":
        "Adam",

    "learning_rate":
        LEARNING_RATE,

    "dropout_rate":
        DROPOUT_RATE,

    "n_folds":
        N_FOLDS,

    "development_images":
        int(
            len(all_paths)
        ),

    "healthy_images":
        int(
            np.sum(
                all_labels == 0
            )
        ),

    "sick_images":
        int(
            np.sum(
                all_labels == 1
            )
        ),

    "overall_oof_roc_auc":
        float(
            overall_oof_auc
        ),

    "test_set_used":
        False,

    "threshold_applied":
        False,

    "folds":
        fold_summaries,
}


oof_summary_path = os.path.join(
    RESULTS_OUTPUT_DIR,
    "oof_summary.json"
)


with open(
    oof_summary_path,
    "w"
) as json_file:

    json.dump(
        oof_summary,
        json_file,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "RUN 05 COMPLETE"
)


print(
    "Development images:",
    len(all_paths)
)

print(
    "OOF predictions:",
    np.sum(
        ~missing_predictions
    )
)

print(
    "Folds:",
    N_FOLDS
)

print(
    "Fixed epoch:",
    FIXED_EPOCH
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Dropout:",
    DROPOUT_RATE
)

print(
    "Overall OOF ROC-AUC:",
    overall_oof_auc
)

print(
    "\nOOF predictions saved to:"
)

print(
    oof_prediction_path
)

print(
    "\nTest set: NOT USED"
)

print(
    "Threshold selection: NOT PERFORMED"
)