import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import json

import numpy as np
import tensorflow as tf

from tensorflow import keras  # type: ignore
from tensorflow.keras import layers  # type: ignore

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

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
RUN_NAME = "Run_03_5-fold_CV_Dropout_Sweep"

RUN_DIR = os.path.join(
    RESULTS_DIR,
    RUN_NAME,
)

RESULTS_OUTPUT_DIR = os.path.join(
    RUN_DIR,
    "results",
)

CHECKPOINT_DIR = os.path.join(
    RUN_DIR,
    "checkpoints",
)

PLOT_DIR = os.path.join(
    RUN_DIR,
    "plots",
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

# Selected in Run 02 and fixed for this experiment
LEARNING_RATE = 1e-3

DROPOUT_RATES = [0.3, 0.5, 0.7]

N_FOLDS = 5

THRESHOLD = 0.5

CLASSES = {
    "Healthy": 0,
    "Sick": 1,
}

RUN_DESCRIPTION = (
    "Dropout ablation using 5-fold stratified cross-validation "
    "with learning rate fixed at 1e-3. No early stopping."
    "Each fold trains for the full configured epoch count, with the" 
    "best epoch selected post-hoc using validation ROC-AUC."
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
print("Dropout rates:", DROPOUT_RATES)
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
# DATA AUGMENTATION + MODEL BUILDER
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


def build_model(dropout_rate):

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
                dropout_rate
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
            "accuracy",
            keras.metrics.AUC(
                name="auc"
            ),
        ],
    )

    return model


# ============================================================
# 5-FOLD CROSS-VALIDATION FOR EACH DROPOUT RATE
# ============================================================

all_dropout_summaries = []

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)


for dropout_rate in DROPOUT_RATES:

    dropout_name = (
        f"dropout_{dropout_rate}"
    )

    dropout_results_dir = os.path.join(
        RESULTS_OUTPUT_DIR,
        dropout_name
    )

    dropout_checkpoint_dir = os.path.join(
        CHECKPOINT_DIR,
        dropout_name
    )

    dropout_plot_dir = os.path.join(
        PLOT_DIR,
        dropout_name
    )

    os.makedirs(
        dropout_results_dir,
        exist_ok=True
    )

    os.makedirs(
        dropout_checkpoint_dir,
        exist_ok=True
    )

    os.makedirs(
        dropout_plot_dir,
        exist_ok=True
    )

    fold_results = []

    print(
        "\nDropout rate:",
        dropout_rate
    )


    # ========================================================
    # 5 FOLDS
    # ========================================================

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(
            all_paths,
            all_labels
        ),
        start=1
    ):

        print(
            f"\nFold {fold}/{N_FOLDS}"
        )


        # ----------------------------------------------------
        # Fold directories
        # ----------------------------------------------------

        fold_results_dir = os.path.join(
            dropout_results_dir,
            f"fold_{fold}"
        )

        fold_checkpoint_dir = os.path.join(
            dropout_checkpoint_dir,
            f"fold_{fold}"
        )

        fold_plot_dir = os.path.join(
            dropout_plot_dir,
            f"fold_{fold}"
        )

        os.makedirs(
            fold_results_dir,
            exist_ok=True
        )

        os.makedirs(
            fold_checkpoint_dir,
            exist_ok=True
        )

        os.makedirs(
            fold_plot_dir,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Fold train / validation data
        # ----------------------------------------------------

        fold_train_paths = (
            all_paths[train_idx]
        )

        fold_train_labels = (
            all_labels[train_idx]
        )

        fold_val_paths = (
            all_paths[val_idx]
        )

        fold_val_labels = (
            all_labels[val_idx]
        )


        # ----------------------------------------------------
        # Fold-specific temperature range
        # ----------------------------------------------------

        fold_min, fold_max = (
            calculate_global_temperature_range(
                fold_train_paths
            )
        )

        print("Temperature range:")
        print("Minimum:", fold_min)
        print("Maximum:", fold_max)


        # ----------------------------------------------------
        # Preprocess training images
        # ----------------------------------------------------

        X_train = np.asarray([

            preprocess_image(
                filepath,
                fold_min,
                fold_max,
                IMAGE_SIZE
            )

            for filepath in fold_train_paths

        ])


        # ----------------------------------------------------
        # Preprocess validation images
        # ----------------------------------------------------

        X_val = np.asarray([

            preprocess_image(
                filepath,
                fold_min,
                fold_max,
                IMAGE_SIZE
            )

            for filepath in fold_val_paths

        ])


        y_train = fold_train_labels

        y_val = fold_val_labels


        # ----------------------------------------------------
        # Class weights
        # ----------------------------------------------------

        class_weights_array = (
            compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_train),
                y=y_train
            )
        )

        class_weights = {

            int(class_label): float(weight)

            for class_label, weight in zip(
                np.unique(y_train),
                class_weights_array
            )

        }

        print(
            "Class weights:",
            class_weights
        )


        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        keras.utils.set_random_seed(
            SEED
        )


        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------

        model = build_model(
            dropout_rate
        )


        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        checkpoint_path = os.path.join(
            fold_checkpoint_dir,
            "best_model.keras"
        )

        checkpoint = (
            keras.callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor="val_auc",
                mode="max",
                save_best_only=True,
                verbose=1
            )
        )


        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        history = model.fit(

            X_train,
            y_train,

            validation_data=(
                X_val,
                y_val
            ),

            epochs=EPOCHS,

            batch_size=BATCH_SIZE,

            class_weight=class_weights,

            callbacks=[
                checkpoint
            ],

            verbose=1
        )


        # ----------------------------------------------------
        # Best epoch based on validation AUC
        # ----------------------------------------------------

        best_epoch = int(

            np.argmax(
                history.history["val_auc"]
            ) + 1

        )

        best_val_auc = float(

            np.max(
                history.history["val_auc"]
            )

        )

        print(
            "\nBest epoch:",
            best_epoch
        )

        print(
            "Best validation AUC:",
            best_val_auc
        )


        # ----------------------------------------------------
        # Load best model
        # ----------------------------------------------------

        model = keras.models.load_model(
            checkpoint_path
        )


        # ----------------------------------------------------
        # Validation predictions
        # ----------------------------------------------------

        y_prob = model.predict(

            X_val,

            batch_size=BATCH_SIZE,

            verbose=0

        ).ravel()


        y_pred = (
            y_prob >= THRESHOLD
        ).astype(int)


        # ----------------------------------------------------
        # Validation metrics
        # ----------------------------------------------------

        tn, fp, fn, tp = (
            confusion_matrix(
                y_val,
                y_pred,
                labels=[0, 1]
            ).ravel()
        )


        accuracy = accuracy_score(
            y_val,
            y_pred
        )

        precision = precision_score(
            y_val,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            y_pred,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_val,
            y_prob
        )

        sensitivity = recall

        specificity = (

            tn / (tn + fp)

            if (tn + fp) > 0

            else 0.0

        )


        print("\nFold metrics:")

        print(
            "Accuracy:",
            accuracy
        )

        print(
            "Sensitivity:",
            sensitivity
        )

        print(
            "Specificity:",
            specificity
        )

        print(
            "Precision:",
            precision
        )

        print(
            "F1:",
            f1
        )

        print(
            "ROC-AUC:",
            roc_auc
        )


        # ----------------------------------------------------
        # Store fold results
        # ----------------------------------------------------

        fold_result = {

            "fold": fold,

            "dropout_rate": dropout_rate,

            "learning_rate": LEARNING_RATE,

            "best_epoch": best_epoch,

            "best_val_auc": best_val_auc,

            "accuracy": float(
                accuracy
            ),

            "sensitivity": float(
                sensitivity
            ),

            "specificity": float(
                specificity
            ),

            "precision": float(
                precision
            ),

            "f1": float(
                f1
            ),

            "roc_auc": float(
                roc_auc
            ),

            "tn": int(tn),

            "fp": int(fp),

            "fn": int(fn),

            "tp": int(tp),

            "temperature_min": float(
                fold_min
            ),

            "temperature_max": float(
                fold_max
            ),

        }

        fold_results.append(
            fold_result
        )


        # ----------------------------------------------------
        # Save fold metrics
        # ----------------------------------------------------

        with open(

            os.path.join(
                fold_results_dir,
                "fold_metrics.json"
            ),

            "w"

        ) as f:

            json.dump(

                fold_result,

                f,

                indent=4

            )


        # ----------------------------------------------------
        # Save training history
        # ----------------------------------------------------

        with open(

            os.path.join(
                fold_results_dir,
                "training_history.json"
            ),

            "w"

        ) as f:

            json.dump(

                history.history,

                f,

                indent=4

            )


        # ----------------------------------------------------
        # Save ROC data
        # ----------------------------------------------------

        fpr, tpr, roc_thresholds = (
            roc_curve(
                y_val,
                y_prob
            )
        )

        roc_data = {

            "fpr": fpr.tolist(),

            "tpr": tpr.tolist(),

            "thresholds": (
                roc_thresholds.tolist()
            ),

            "roc_auc": float(
                roc_auc
            ),

        }

        with open(

            os.path.join(
                fold_results_dir,
                "roc_data.json"
            ),

            "w"

        ) as f:

            json.dump(

                roc_data,

                f,

                indent=4

            )


        # ----------------------------------------------------
        # Save loss plot
        # ----------------------------------------------------

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            history.history["loss"],
            label="Training Loss"
        )

        plt.plot(
            history.history["val_loss"],
            label="Validation Loss"
        )

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.title(

            f"Loss - Dropout {dropout_rate}, "
            f"Fold {fold}"

        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(

            os.path.join(
                fold_plot_dir,
                "loss.png"
            ),

            dpi=300

        )

        plt.close()


        # ----------------------------------------------------
        # Save AUC plot
        # ----------------------------------------------------

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            history.history["auc"],
            label="Training AUC"
        )

        plt.plot(
            history.history["val_auc"],
            label="Validation AUC"
        )

        plt.xlabel("Epoch")

        plt.ylabel("AUC")

        plt.title(

            f"AUC - Dropout {dropout_rate}, "
            f"Fold {fold}"

        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(

            os.path.join(
                fold_plot_dir,
                "auc.png"
            ),

            dpi=300

        )

        plt.close()


    # ========================================================
    # AGGREGATE 5 FOLDS FOR CURRENT DROPOUT RATE
    # ========================================================

    dropout_summary = {

        "dropout_rate": dropout_rate,

        "learning_rate": LEARNING_RATE,

        "mean_auc": float(

            np.mean([

                result["roc_auc"]

                for result in fold_results

            ])

        ),

        "std_auc": float(

            np.std([

                result["roc_auc"]

                for result in fold_results

            ])

        ),

        "mean_accuracy": float(

            np.mean([

                result["accuracy"]

                for result in fold_results

            ])

        ),

        "mean_sensitivity": float(

            np.mean([

                result["sensitivity"]

                for result in fold_results

            ])

        ),

        "mean_specificity": float(

            np.mean([

                result["specificity"]

                for result in fold_results

            ])

        ),

        "mean_precision": float(

            np.mean([

                result["precision"]

                for result in fold_results

            ])

        ),

        "mean_f1_score": float(

            np.mean([

                result["f1"]

                for result in fold_results

            ])

        ),

    }


    all_dropout_summaries.append(
        dropout_summary
    )


    # --------------------------------------------------------
    # Save current dropout summary
    # --------------------------------------------------------

    with open(

        os.path.join(
            dropout_results_dir,
            "cv_summary.json"
        ),

        "w"

    ) as f:

        json.dump(

            dropout_summary,

            f,

            indent=4

        )


    print("\nDropout summary:")

    print(

        "Mean AUC:",

        dropout_summary["mean_auc"]

    )

    print(

        "Std AUC:",

        dropout_summary["std_auc"]

    )


# ============================================================
# SAVE DROPOUT COMPARISON
# ============================================================

with open(

    os.path.join(
        RESULTS_OUTPUT_DIR,
        "cv_dropout_comparison.json"
    ),

    "w"

) as f:

    json.dump(

        all_dropout_summaries,

        f,

        indent=4

    )


comparison_csv = os.path.join(

    RESULTS_OUTPUT_DIR,
    "cv_dropout_comparison.csv"

)


with open(

    comparison_csv,

    "w",

    newline=""

) as f:

    writer = csv.writer(f)

    writer.writerow([

        "dropout_rate",

        "learning_rate",

        "mean_auc",

        "std_auc",

        "mean_accuracy",

        "mean_sensitivity",

        "mean_specificity",

        "mean_precision",

        "mean_f1_score",

    ])


    for summary in all_dropout_summaries:

        writer.writerow([

            summary["dropout_rate"],

            summary["learning_rate"],

            summary["mean_auc"],

            summary["std_auc"],

            summary["mean_accuracy"],

            summary["mean_sensitivity"],

            summary["mean_specificity"],

            summary["mean_precision"],

            summary["mean_f1_score"],

        ])


# ============================================================
# COMPARE ALL DROPOUT RATES
# ============================================================

print(
    "\nDROPOUT RATE COMPARISON "
    "(5-FOLD CV MEANS)"
)

all_dropout_summaries_sorted = sorted(

    all_dropout_summaries,

    key=lambda x: x["mean_auc"],

    reverse=True

)


print(

    "\nRank | Dropout | Mean AUC (±std) | "
    "Mean Acc | Mean Sens | Mean Spec | Mean F1"

)


for rank, summary in enumerate(

    all_dropout_summaries_sorted,

    start=1

):

    print(

        f"{rank:4d} | "

        f"{summary['dropout_rate']:.1f}      | "

        f"{summary['mean_auc']:.4f} "

        f"(±{summary['std_auc']:.4f}) | "

        f"{summary['mean_accuracy']:.4f}   | "

        f"{summary['mean_sensitivity']:.4f}    | "

        f"{summary['mean_specificity']:.4f}    | "

        f"{summary['mean_f1_score']:.4f}"

    )


# ============================================================
# SELECT BEST DROPOUT RATE
# ============================================================

best_dropout_summary = (
    all_dropout_summaries_sorted[0]
)

best_dropout_rate = (
    best_dropout_summary["dropout_rate"]
)


print(

    f"\nBest dropout rate "
    f"(by mean CV AUC): "
    f"{best_dropout_rate:.1f}"

)

print(

    f"Mean AUC: "
    f"{best_dropout_summary['mean_auc']:.4f} "
    f"± {best_dropout_summary['std_auc']:.4f}"

)


# ============================================================
# EPOCH-WISE VALIDATION AUC FOR WINNING DROPOUT
# ============================================================
#
# The winning dropout configuration is carried forward
# to Run 04.
#
# The five fold-specific training histories are used to
# calculate mean and standard deviation of validation AUC
# for every epoch.
#
# These epoch-wise statistics are used by Run 04 to identify
# candidate fixed epochs.
# ============================================================

best_dropout_name = (
    f"dropout_{best_dropout_rate}"
)

best_dropout_results_dir = os.path.join(

    RESULTS_OUTPUT_DIR,

    best_dropout_name

)


epoch_history = []


for fold in range(1, N_FOLDS + 1):

    history_path = os.path.join(

        best_dropout_results_dir,

        f"fold_{fold}",

        "training_history.json"

    )


    if not os.path.exists(history_path):

        raise FileNotFoundError(

            f"Training history not found: "
            f"{history_path}"

        )


    with open(

        history_path,

        "r"

    ) as f:

        fold_history = json.load(f)


    if "val_auc" not in fold_history:

        raise KeyError(

            f"'val_auc' not found in "
            f"{history_path}"

        )


    val_auc = np.asarray(

        fold_history["val_auc"],

        dtype=np.float64

    )


    if len(val_auc) != EPOCHS:

        raise ValueError(

            f"Expected {EPOCHS} validation AUC "
            f"values in {history_path}, "
            f"but found {len(val_auc)}."

        )


    epoch_history.append(
        val_auc
    )


epoch_history = np.asarray(
    epoch_history,
    dtype=np.float64
)


# ------------------------------------------------------------
# Calculate mean and standard deviation for each epoch
# ------------------------------------------------------------

epoch_mean_val_auc = np.mean(
    epoch_history,
    axis=0
)

epoch_std_val_auc = np.std(
    epoch_history,
    axis=0
)


# ------------------------------------------------------------
# Save epoch-wise CSV
# ------------------------------------------------------------

epoch_auc_csv = os.path.join(

    best_dropout_results_dir,

    "epoch_wise_val_auc.csv"

)


with open(

    epoch_auc_csv,

    "w",

    newline=""

) as f:

    writer = csv.writer(f)

    writer.writerow([

        "epoch",

        "mean_val_auc",

        "std_val_auc",

        "fold_1_val_auc",

        "fold_2_val_auc",

        "fold_3_val_auc",

        "fold_4_val_auc",

        "fold_5_val_auc",

    ])


    for epoch_idx in range(EPOCHS):

        writer.writerow([

            epoch_idx + 1,

            float(
                epoch_mean_val_auc[epoch_idx]
            ),

            float(
                epoch_std_val_auc[epoch_idx]
            ),

            float(
                epoch_history[0, epoch_idx]
            ),

            float(
                epoch_history[1, epoch_idx]
            ),

            float(
                epoch_history[2, epoch_idx]
            ),

            float(
                epoch_history[3, epoch_idx]
            ),

            float(
                epoch_history[4, epoch_idx]
            ),

        ])


# ------------------------------------------------------------
# Save epoch-wise statistics as JSON
# ------------------------------------------------------------

epoch_auc_json = os.path.join(

    best_dropout_results_dir,

    "epoch_wise_val_auc.json"

)


epoch_auc_data = {

    "dropout_rate": float(
        best_dropout_rate
    ),

    "learning_rate": float(
        LEARNING_RATE
    ),

    "number_of_folds": N_FOLDS,

    "epochs": EPOCHS,

    "selection_metric": (
        "Validation ROC-AUC"
    ),

    "data": [

        {

            "epoch": epoch_idx + 1,

            "mean_val_auc": float(
                epoch_mean_val_auc[epoch_idx]
            ),

            "std_val_auc": float(
                epoch_std_val_auc[epoch_idx]
            ),

            "fold_1_val_auc": float(
                epoch_history[0, epoch_idx]
            ),

            "fold_2_val_auc": float(
                epoch_history[1, epoch_idx]
            ),

            "fold_3_val_auc": float(
                epoch_history[2, epoch_idx]
            ),

            "fold_4_val_auc": float(
                epoch_history[3, epoch_idx]
            ),

            "fold_5_val_auc": float(
                epoch_history[4, epoch_idx]
            ),

        }

        for epoch_idx in range(EPOCHS)

    ],

}


with open(

    epoch_auc_json,

    "w"

) as f:

    json.dump(

        epoch_auc_data,

        f,

        indent=4

    )


# ------------------------------------------------------------
# Save mean validation AUC curve
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 5)
)

epochs_axis = np.arange(
    1,
    EPOCHS + 1
)

plt.plot(

    epochs_axis,

    epoch_mean_val_auc,

    label="Mean Validation AUC"

)

plt.fill_between(

    epochs_axis,

    epoch_mean_val_auc - epoch_std_val_auc,

    epoch_mean_val_auc + epoch_std_val_auc,

    alpha=0.2,

    label="±1 SD"

)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Validation AUC"
)

plt.title(

    f"Mean 5-Fold Validation AUC "
    f"- Dropout {best_dropout_rate}"

)

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(

        best_dropout_results_dir,

        "epoch_wise_val_auc.png"

    ),

    dpi=300

)

plt.close()


# ------------------------------------------------------------
# Print aggregate epoch-wise optimum
# ------------------------------------------------------------

mean_auc_best_epoch = int(

    np.argmax(
        epoch_mean_val_auc
    ) + 1

)

mean_auc_best_value = float(

    np.max(
        epoch_mean_val_auc
    )

)


print(
    "\nWinning dropout epoch-wise analysis:"
)

print(
    "Dropout:",
    best_dropout_rate
)

print(
    "Mean-AUC optimum epoch:",
    mean_auc_best_epoch
)

print(
    "Mean validation AUC:",
    mean_auc_best_value
)

print(
    "Epoch-wise statistics saved to:",
    best_dropout_results_dir
)


# ============================================================
# FINAL RUN SUMMARY
# ============================================================

final_summary = {

    "experiment": EXPERIMENT_NAME,

    "run": RUN_NAME,

    "seed": SEED,

    "image_size": IMAGE_SIZE,

    "batch_size": BATCH_SIZE,

    "epochs": EPOCHS,

    "learning_rate": LEARNING_RATE,

    "number_of_folds": N_FOLDS,

    "dropout_rates_tested": DROPOUT_RATES,

    "best_dropout_rate": best_dropout_rate,

    "selection_metric": (
        "Mean 5-fold validation ROC-AUC"
    ),

    "epoch_wise_analysis": {

        "performed_for": (
            f"dropout_{best_dropout_rate}"
        ),

        "mean_auc_best_epoch": (
            mean_auc_best_epoch
        ),

        "mean_auc_best_value": (
            mean_auc_best_value
        ),

        "output_csv": (
            os.path.join(
                best_dropout_name,
                "epoch_wise_val_auc.csv"
            )
        ),

        "output_json": (
            os.path.join(
                best_dropout_name,
                "epoch_wise_val_auc.json"
            )
        ),

        "output_plot": (
            os.path.join(
                best_dropout_name,
                "epoch_wise_val_auc.png"
            )
        ),

    },

    "threshold": THRESHOLD,

    "test_set_used": False,

}


with open(

    os.path.join(
        RESULTS_OUTPUT_DIR,
        "run_summary.json"
    ),

    "w"

) as f:

    json.dump(

        final_summary,

        f,

        indent=4

    )


print(
    "\nRUN COMPLETE"
)

print(

    "Best dropout rate:",

    f"{best_dropout_rate:.1f}"

)

print(

    "Selection metric: "
    "Mean 5-fold validation ROC-AUC"

)

print(

    "Mean-AUC optimum epoch:",

    mean_auc_best_epoch

)

print(
    "Test set: NOT USED"
)

print(

    "Results saved to:",

    RUN_DIR

)