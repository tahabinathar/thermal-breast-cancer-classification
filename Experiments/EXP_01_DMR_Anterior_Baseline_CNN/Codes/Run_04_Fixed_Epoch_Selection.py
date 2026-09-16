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
RUN_NAME = "Run_04_Fixed_Epoch_Selection"

RUN03_RESULTS_DIR = os.path.join(
    RESULTS_DIR,
    "Run_03_5-fold_CV_Dropout_Sweep",
    "results",
)

RUN03_DROPOUT_DIR = os.path.join(
    RUN03_RESULTS_DIR,
    "dropout_0.3",
)

RUN03_EPOCH_CSV = os.path.join(
    RUN03_DROPOUT_DIR,
    "epoch_wise_val_auc.csv",
)

RUN04_RESULTS_DIR = os.path.join(
    RESULTS_DIR,
    RUN_NAME,
    "results",
)

os.makedirs(RUN04_RESULTS_DIR, exist_ok=True)


# ============================================================
# RUN SETTINGS
# ============================================================

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 100

LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3

N_FOLDS = 5
THRESHOLD = 0.5

SEED = 10

CLASSES = {
    "Healthy": 0,
    "Sick": 1,
}

RUN_DESCRIPTION = (
    "Fixed-epoch selection using 5-fold stratified cross-validation "
    "with learning rate fixed at 1e-3 and dropout fixed at 0.3. "
    "Candidate epochs are derived automatically from the Run 03 "
    "epoch-wise validation AUC results and evaluated independently "
    "across all five folds. The final fixed epoch is selected using "
    "the highest mean 5-fold validation ROC-AUC."
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
print("Dropout rate:", DROPOUT_RATE)
print("Number of folds:", N_FOLDS)
print("Threshold:", THRESHOLD)
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
# LOAD RUN 03 EPOCH-WISE AUC CSV
# ============================================================

def load_epoch_wise_auc(csv_path):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Run 03 epoch-wise AUC CSV not found:\n{csv_path}"
        )

    epochs = []
    mean_auc = []
    fold_auc = {
        fold: []
        for fold in range(1, N_FOLDS + 1)
    }

    with open(csv_path, "r", newline="") as f:

        reader = csv.DictReader(f)

        required_columns = {
            "epoch",
            "mean_val_auc",
        }

        missing = required_columns - set(reader.fieldnames)

        if missing:
            raise KeyError(
                f"Missing required columns in Run 03 CSV: {missing}"
            )

        for fold in range(1, N_FOLDS + 1):

            possible_columns = [
                f"fold_{fold}_val_auc",
                f"fold_{fold}_auc",
            ]

            if not any(
                column in reader.fieldnames
                for column in possible_columns
            ):
                raise KeyError(
                    f"Could not find AUC column for fold {fold}. "
                    f"Expected one of: {possible_columns}"
                )

        for row in reader:

            epoch = int(row["epoch"])

            epochs.append(epoch)
            mean_auc.append(
                float(row["mean_val_auc"])
            )

            for fold in range(1, N_FOLDS + 1):

                if f"fold_{fold}_val_auc" in row:
                    value = row[
                        f"fold_{fold}_val_auc"
                    ]
                else:
                    value = row[
                        f"fold_{fold}_auc"
                    ]

                fold_auc[fold].append(
                    float(value)
                )

    epochs = np.asarray(epochs, dtype=int)
    mean_auc = np.asarray(mean_auc, dtype=np.float64)

    for fold in fold_auc:
        fold_auc[fold] = np.asarray(
            fold_auc[fold],
            dtype=np.float64,
        )

    return epochs, mean_auc, fold_auc


# ============================================================
# DETERMINE CANDIDATE EPOCHS FROM RUN 03
# ============================================================

print("READING RUN 03 TRAINING DYNAMICS")

print()
print(f"Run 03 CSV:")
print(RUN03_EPOCH_CSV)

epochs, mean_auc, fold_auc = load_epoch_wise_auc(
    RUN03_EPOCH_CSV
)


# ------------------------------------------------------------
# Validate epoch range
# ------------------------------------------------------------

if len(epochs) != EPOCHS:
    raise ValueError(
        f"Expected {EPOCHS} epochs in Run 03 CSV, "
        f"but found {len(epochs)}."
    )

expected_epochs = np.arange(
    1,
    EPOCHS + 1,
)

if not np.array_equal(
    epochs,
    expected_epochs,
):
    raise ValueError(
        "Run 03 epoch sequence is not "
        f"1 to {EPOCHS}."
    )


# ------------------------------------------------------------
# Aggregate mean-AUC optimum
# ------------------------------------------------------------

mean_auc_best_index = int(
    np.argmax(mean_auc)
)

mean_auc_best_epoch = int(
    epochs[mean_auc_best_index]
)

mean_auc_best_value = float(
    mean_auc[mean_auc_best_index]
)


# ------------------------------------------------------------
# Individual fold optima
# ------------------------------------------------------------

fold_best_epochs = {}
fold_best_auc = {}

for fold in range(1, N_FOLDS + 1):

    best_index = int(
        np.argmax(fold_auc[fold])
    )

    best_epoch = int(
        epochs[best_index]
    )

    best_value = float(
        fold_auc[fold][best_index]
    )

    fold_best_epochs[fold] = best_epoch
    fold_best_auc[fold] = best_value



# ============================================================
# 7. DEFINE CANDIDATE EPOCHS FROM RUN 03
# ============================================================
#
# Candidate epochs are derived entirely from the observed
# Run 03 training dynamics:
#
# - Individual best epoch from each fold
# - Aggregate mean-AUC best epoch
# - Full 100-epoch training duration
# ============================================================

candidate_epochs = []

# Individual fold optima
candidate_epochs.extend(
    fold_best_epochs.values()
)

# Aggregate mean-AUC optimum
candidate_epochs.append(
    mean_auc_best_epoch
)

# Full training duration
candidate_epochs.append(
    EPOCHS
)

# Remove duplicates and sort
candidate_epochs = sorted(
    set(
        epoch
        for epoch in candidate_epochs
        if 1 <= epoch <= EPOCHS
    )
)


# ============================================================
# PRINT CANDIDATE SELECTION
# ============================================================

print("RUN 03 TRAINING-DYNAMICS SUMMARY")

print()
print(
    f"Aggregate mean-AUC optimum:"
    f" epoch {mean_auc_best_epoch}"
    f"  (AUC = {mean_auc_best_value:.6f})"
)

print()
print("Individual fold optima:")

for fold in range(1, N_FOLDS + 1):

    print(
        f"  Fold {fold}: "
        f"epoch {fold_best_epochs[fold]} "
        f"(AUC = {fold_best_auc[fold]:.6f})"
    )

print()
print("FINAL FIXED-EPOCH CANDIDATES")


print()

for i, epoch in enumerate(
    candidate_epochs,
    start=1,
):
    print(
        f"  Candidate {i}: Epoch {epoch}"
    )

print()
print(
    f"Total candidates: {len(candidate_epochs)}"
)

print()
print(
    "Candidate set is now FIXED."
)

print(
    "Run 04 5-fold CV will begin using only these epochs."
)

print()


# ============================================================
# SAVE CANDIDATE INFORMATION
# ============================================================

candidate_info = {
    "run_name": RUN_NAME,
    "source_run": "Run_03_5-fold_CV_Dropout_Sweep",
    "source_csv": RUN03_EPOCH_CSV,

    "aggregate_mean_auc_best_epoch":
        mean_auc_best_epoch,

    "aggregate_mean_auc_best_value":
        mean_auc_best_value,

    "fold_best_epochs":
        {
            str(k): int(v)
            for k, v in fold_best_epochs.items()
        },

    "fold_best_auc":
        {
            str(k): float(v)
            for k, v in fold_best_auc.items()
        },

    "full_training_epoch":
        EPOCHS,

    "candidate_epochs":
        candidate_epochs,

    "selection_basis": (
        "Candidates were derived from Run 03 training "
        "dynamics using the individual fold optima, the "
        "aggregate mean-AUC optimum, and the full 100-epoch "
        "training duration. No arbitrary intermediate epochs "
        "were introduced."
    ),
}


candidate_info_path = os.path.join(
    RUN04_RESULTS_DIR,
    "candidate_epochs.json",
)

with open(
    candidate_info_path,
    "w",
) as f:

    json.dump(
        candidate_info,
        f,
        indent=4,
    )


# ============================================================
# CREATE FIXED STRATIFIED CV SPLITS
# ============================================================
#
# The SAME five folds are used for every candidate epoch.
# This makes the candidate comparison fair.
# ============================================================

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED,
)

cv_splits = list(
    skf.split(
        all_paths,
        all_labels,
    )
)


# ============================================================
# MODEL
# ============================================================

def build_baseline_cnn():

    model = keras.Sequential(
        [
            layers.Input(
                shape=(
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    1,
                )
            ),

            layers.RandomRotation(
                factor=0.05
            ),

            layers.RandomZoom(
                height_factor=0.05,
                width_factor=0.05,
            ),

            layers.Conv2D(
                32,
                (2, 2),
                padding="same",
                activation="relu",
            ),

            layers.Conv2D(
                32,
                (2, 2),
                padding="same",
                activation="relu",
            ),

            layers.MaxPooling2D(),

            layers.Conv2D(
                64,
                (2, 2),
                padding="same",
                activation="relu",
            ),

            layers.MaxPooling2D(),

            layers.Conv2D(
                128,
                (2, 2),
                padding="same",
                activation="relu",
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
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(
                name="auc"
            )
        ],
    )

    return model


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y_true,
    y_prob,
    threshold=THRESHOLD,
):

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    metrics = {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_prob,
            )
        ),

        "average_precision": float(
            average_precision_score(
                y_true,
                y_prob,
            )
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "sensitivity": float(
            sensitivity
        ),

        "specificity": float(
            specificity
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

    return metrics


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(y_train):

    healthy_count = np.sum(
        y_train == CLASSES["Healthy"]
    )

    sick_count = np.sum(
        y_train == CLASSES["Sick"]
    )

    total = len(y_train)

    weight_healthy = (
        total /
        (
            2 * healthy_count
        )
    )

    weight_sick = (
        total /
        (
            2 * sick_count
        )
    )

    return {
        0: float(weight_healthy),
        1: float(weight_sick),
    }


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_one_fold(
    candidate_epoch,
    fold_number,
    train_indices,
    val_indices,
):

    print()
    print(
        f"Candidate epoch {candidate_epoch} "
        f"| Fold {fold_number}/{N_FOLDS}"
    )


    # --------------------------------------------------------
    # Paths and labels
    # --------------------------------------------------------

    train_paths = all_paths[
        train_indices
    ]

    train_labels = all_labels[
        train_indices
    ]

    val_paths = all_paths[
        val_indices
    ]

    val_labels = all_labels[
        val_indices
    ]


    # --------------------------------------------------------
    # Fold-specific temperature range
    # --------------------------------------------------------
    # IMPORTANT:
    # Only fold training images are used to calculate the
    # normalization range.
    # --------------------------------------------------------

    print(
        "Calculating fold-specific "
        "temperature range..."
    )

    train_global_min, train_global_max = (
        calculate_global_temperature_range(
            train_paths
        )
    )

    print(
        f"Temperature range: "
        f"{train_global_min:.6f} "
        f"to "
        f"{train_global_max:.6f}"
    )


    # --------------------------------------------------------
    # Preprocess training images
    # --------------------------------------------------------

    X_train = np.asarray(
        [
            preprocess_image(
                filepath,
                train_global_min,
                train_global_max,
                IMAGE_SIZE,
            )
            for filepath in train_paths
        ],
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # Preprocess validation images
    # --------------------------------------------------------

    X_val = np.asarray(
        [
            preprocess_image(
                filepath,
                train_global_min,
                train_global_max,
                IMAGE_SIZE,
            )
            for filepath in val_paths
        ],
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    class_weights = (
        calculate_class_weights(
            train_labels
        )
    )

    print(
        f"Class weight - Healthy: "
        f"{class_weights[0]:.6f}"
    )

    print(
        f"Class weight - Sick: "
        f"{class_weights[1]:.6f}"
    )


    # --------------------------------------------------------
    # Reset random seeds before each model
    # --------------------------------------------------------

    tf.keras.backend.clear_session()

    np.random.seed(
        SEED
    )

    tf.random.set_seed(
        SEED
    )


    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_baseline_cnn()


    # --------------------------------------------------------
    # Train exactly to candidate epoch
    # --------------------------------------------------------
    #
    # No early stopping.
    # No best-epoch checkpoint.
    #
    # The candidate itself is the fixed training epoch.
    # --------------------------------------------------------

    history = model.fit(
        X_train,
        train_labels,
        validation_data=(
            X_val,
            val_labels,
        ),
        epochs=candidate_epoch,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        verbose=0,
    )


    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    val_prob = (
        model.predict(
            X_val,
            batch_size=BATCH_SIZE,
            verbose=0,
        )
        .ravel()
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        val_labels,
        val_prob,
        THRESHOLD,
    )


    print(
        f"Validation ROC-AUC: "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"Validation Accuracy: "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Validation Sensitivity: "
        f"{metrics['sensitivity']:.6f}"
    )

    print(
        f"Validation Specificity: "
        f"{metrics['specificity']:.6f}"
    )


    # ========================================================
    # SAVE FOLD RESULTS
    # ========================================================

    candidate_dir = os.path.join(
        RUN04_RESULTS_DIR,
        f"epoch_{candidate_epoch}",
    )

    fold_dir = os.path.join(
        candidate_dir,
        f"fold_{fold_number}",
    )

    os.makedirs(
        fold_dir,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        fold_dir,
        "model.keras",
    )

    model.save(
        model_path
    )


    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_path = os.path.join(
        fold_dir,
        "validation_predictions.csv",
    )

    with open(
        predictions_path,
        "w",
        newline="",
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "filepath",
                "true_label",
                "predicted_probability",
                "predicted_label",
            ]
        )

        for filepath, true_label, probability in zip(
            val_paths,
            val_labels,
            val_prob,
        ):

            predicted_label = int(
                probability >= THRESHOLD
            )

            writer.writerow(
                [
                    filepath,
                    int(true_label),
                    float(probability),
                    predicted_label,
                ]
            )


    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = os.path.join(
        fold_dir,
        "training_history.json",
    )

    history_dict = {
        key: [
            float(value)
            for value in values
        ]
        for key, values in history.history.items()
    }

    with open(
        history_path,
        "w",
    ) as f:

        json.dump(
            history_dict,
            f,
            indent=4,
        )


    # --------------------------------------------------------
    # Save fold metrics
    # --------------------------------------------------------

    fold_result = {
        "candidate_epoch": int(
            candidate_epoch
        ),

        "fold": int(
            fold_number
        ),

        "train_count": int(
            len(train_paths)
        ),

        "validation_count": int(
            len(val_paths)
        ),

        "train_healthy": int(
            np.sum(
                train_labels == 0
            )
        ),

        "train_sick": int(
            np.sum(
                train_labels == 1
            )
        ),

        "validation_healthy": int(
            np.sum(
                val_labels == 0
            )
        ),

        "validation_sick": int(
            np.sum(
                val_labels == 1
            )
        ),

        "temperature_range": {
            "global_min": float(
                train_global_min
            ),
            "global_max": float(
                train_global_max
            ),
        },

        "class_weights": {
            "Healthy": float(
                class_weights[0]
            ),
            "Sick": float(
                class_weights[1]
            ),
        },

        "learning_rate": float(
            LEARNING_RATE
        ),

        "dropout_rate": float(
            DROPOUT_RATE
        ),

        "batch_size": int(
            BATCH_SIZE
        ),

        "fixed_epoch": int(
            candidate_epoch
        ),

        "threshold": float(
            THRESHOLD
        ),

        "metrics": metrics,
    }


    fold_result_path = os.path.join(
        fold_dir,
        "fold_result.json",
    )

    with open(
        fold_result_path,
        "w",
    ) as f:

        json.dump(
            fold_result,
            f,
            indent=4,
        )


    # --------------------------------------------------------
    # Clean up
    # --------------------------------------------------------

    del model
    del X_train
    del X_val
    del val_prob
    del history

    tf.keras.backend.clear_session()
    gc.collect()


    return fold_result


# ============================================================
# 16. RUN 5-FOLD CV FOR EVERY CANDIDATE
# ============================================================

all_candidate_results = []


for candidate_index, candidate_epoch in enumerate(
    candidate_epochs,
    start=1,
):

    print()
    print()

    print(
        f"CANDIDATE {candidate_index}/{len(candidate_epochs)}"
    )

    print(
        f"FIXED EPOCH = {candidate_epoch}"
    )


    fold_results = []


    for fold_number, (
        train_indices,
        val_indices,
    ) in enumerate(
        cv_splits,
        start=1,
    ):

        fold_result = train_one_fold(
            candidate_epoch,
            fold_number,
            train_indices,
            val_indices,
        )

        fold_results.append(
            fold_result
        )


    # ========================================================
    # AGGREGATE CANDIDATE RESULTS
    # ========================================================

    roc_auc_values = [
        result["metrics"]["roc_auc"]
        for result in fold_results
    ]

    accuracy_values = [
        result["metrics"]["accuracy"]
        for result in fold_results
    ]

    balanced_accuracy_values = [
        result["metrics"]["balanced_accuracy"]
        for result in fold_results
    ]

    sensitivity_values = [
        result["metrics"]["sensitivity"]
        for result in fold_results
    ]

    specificity_values = [
        result["metrics"]["specificity"]
        for result in fold_results
    ]

    precision_values = [
        result["metrics"]["precision"]
        for result in fold_results
    ]

    f1_values = [
        result["metrics"]["f1"]
        for result in fold_results
    ]

    mcc_values = [
        result["metrics"]["mcc"]
        for result in fold_results
    ]


    candidate_result = {

        "candidate_epoch": int(
            candidate_epoch
        ),

        "mean_roc_auc": float(
            np.mean(
                roc_auc_values
            )
        ),

        "std_roc_auc": float(
            np.std(
                roc_auc_values
            )
        ),

        "mean_accuracy": float(
            np.mean(
                accuracy_values
            )
        ),

        "std_accuracy": float(
            np.std(
                accuracy_values
            )
        ),

        "mean_balanced_accuracy": float(
            np.mean(
                balanced_accuracy_values
            )
        ),

        "std_balanced_accuracy": float(
            np.std(
                balanced_accuracy_values
            )
        ),

        "mean_sensitivity": float(
            np.mean(
                sensitivity_values
            )
        ),

        "std_sensitivity": float(
            np.std(
                sensitivity_values
            )
        ),

        "mean_specificity": float(
            np.mean(
                specificity_values
            )
        ),

        "std_specificity": float(
            np.std(
                specificity_values
            )
        ),

        "mean_precision": float(
            np.mean(
                precision_values
            )
        ),

        "std_precision": float(
            np.std(
                precision_values
            )
        ),

        "mean_f1": float(
            np.mean(
                f1_values
            )
        ),

        "std_f1": float(
            np.std(
                f1_values
            )
        ),

        "mean_mcc": float(
            np.mean(
                mcc_values
            )
        ),

        "std_mcc": float(
            np.std(
                mcc_values
            )
        ),

        "fold_results": fold_results,
    }


    all_candidate_results.append(
        candidate_result
    )


    # --------------------------------------------------------
    # Print candidate summary
    # --------------------------------------------------------

    print()
    print(
        f"Candidate epoch {candidate_epoch} summary:"
    )

    print(
        f"  Mean ROC-AUC: "
        f"{candidate_result['mean_roc_auc']:.6f} "
        f"± "
        f"{candidate_result['std_roc_auc']:.6f}"
    )

    print(
        f"  Mean Accuracy: "
        f"{candidate_result['mean_accuracy']:.6f}"
    )

    print(
        f"  Mean Sensitivity: "
        f"{candidate_result['mean_sensitivity']:.6f}"
    )

    print(
        f"  Mean Specificity: "
        f"{candidate_result['mean_specificity']:.6f}"
    )

    print(
        f"  Mean F1: "
        f"{candidate_result['mean_f1']:.6f}"
    )


# ============================================================
# SAVE COMPLETE CANDIDATE COMPARISON
# ============================================================

comparison_json_path = os.path.join(
    RUN04_RESULTS_DIR,
    "candidate_epoch_comparison.json",
)

with open(
    comparison_json_path,
    "w",
) as f:

    json.dump(
        all_candidate_results,
        f,
        indent=4,
    )


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

comparison_csv_path = os.path.join(
    RUN04_RESULTS_DIR,
    "candidate_epoch_comparison.csv",
)

with open(
    comparison_csv_path,
    "w",
    newline="",
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "candidate_epoch",

            "mean_roc_auc",
            "std_roc_auc",

            "mean_accuracy",
            "std_accuracy",

            "mean_balanced_accuracy",
            "std_balanced_accuracy",

            "mean_sensitivity",
            "std_sensitivity",

            "mean_specificity",
            "std_specificity",

            "mean_precision",
            "std_precision",

            "mean_f1",
            "std_f1",

            "mean_mcc",
            "std_mcc",
        ]
    )

    for result in all_candidate_results:

        writer.writerow(
            [
                result["candidate_epoch"],

                result["mean_roc_auc"],
                result["std_roc_auc"],

                result["mean_accuracy"],
                result["std_accuracy"],

                result["mean_balanced_accuracy"],
                result["std_balanced_accuracy"],

                result["mean_sensitivity"],
                result["std_sensitivity"],

                result["mean_specificity"],
                result["std_specificity"],

                result["mean_precision"],
                result["std_precision"],

                result["mean_f1"],
                result["std_f1"],

                result["mean_mcc"],
                result["std_mcc"],
            ]
        )


# ============================================================
# SELECT FINAL FIXED EPOCH
# ============================================================
#
# Primary criterion:
# Highest mean 5-fold validation ROC-AUC.
#
# Tie-breaker:
# Lower epoch.
#
# The test set is not involved.
# ============================================================

selected_epoch_result = max(
    all_candidate_results,
    key=lambda result: (
        result["mean_roc_auc"],
        -result["candidate_epoch"],
    ),
)

selected_epoch = int(
    selected_epoch_result[
        "candidate_epoch"
    ]
)

selected_mean_auc = float(
    selected_epoch_result[
        "mean_roc_auc"
    ]
)

selected_std_auc = float(
    selected_epoch_result[
        "std_roc_auc"
    ]
)


# ============================================================
# PRINT FINAL RESULT
# ============================================================

print()
print()

print(
    "RUN 04 FINAL RESULT"
)

print()

print(
    "Candidate epochs:"
)

print(
    candidate_epochs
)

print()

print(
    f"Selected fixed epoch: "
    f"{selected_epoch}"
)

print(
    f"Mean 5-fold validation ROC-AUC: "
    f"{selected_mean_auc:.6f}"
)

print(
    f"Std 5-fold validation ROC-AUC: "
    f"{selected_std_auc:.6f}"
)

print()

print(
    "Selection criterion:"
)

print(
    "Highest mean 5-fold validation ROC-AUC"
)

print()

print(
    "Test set:"
)

print(
    "NOT USED"
)

print()

# ============================================================
# 21. SAVE RUN SUMMARY
# ============================================================

run_summary = {

    "experiment_name":
        EXPERIMENT_NAME,

    "run_name":
        RUN_NAME,

    "run_description":
        RUN_DESCRIPTION,

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "maximum_epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "dropout_rate":
        DROPOUT_RATE,

    "n_folds":
        N_FOLDS,

    "threshold":
        THRESHOLD,

    "seed":
        SEED,

    "development_dataset": {
        "total": int(
            len(all_paths)
        ),
        "healthy": int(
            np.sum(
                all_labels == 0
            )
        ),
        "sick": int(
            np.sum(
                all_labels == 1
            )
        ),
    },

    "test_set_used":
        False,

    "candidate_epochs":
        candidate_epochs,

    "selected_fixed_epoch":
        selected_epoch,

    "selected_mean_roc_auc":
        selected_mean_auc,

    "selected_std_roc_auc":
        selected_std_auc,

    "selection_criterion":
        "Highest mean 5-fold validation ROC-AUC",

    "tie_breaker":
        "Lower epoch",

    "run03_source_csv":
        RUN03_EPOCH_CSV,

    "run03_mean_auc_best_epoch":
        mean_auc_best_epoch,

    "run03_mean_auc_best_value":
        mean_auc_best_value,

    "run03_fold_best_epochs":
        {
            str(k): int(v)
            for k, v in fold_best_epochs.items()
        },
}


run_summary_path = os.path.join(
    RUN04_RESULTS_DIR,
    "run_summary.json",
)

with open(
    run_summary_path,
    "w",
) as f:

    json.dump(
        run_summary,
        f,
        indent=4,
    )


print()
print(
    "Run 04 completed successfully."
)

print()
print(
    f"Results saved to:"
)

print(
    RUN04_RESULTS_DIR
)