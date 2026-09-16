# EXP01 Run 05: 5-Fold Cross-Validation OOF Predictions

## 1. Objective

Generate out-of-fold (OOF) predictions for the final EXP01 baseline CNN using stratified 5-fold cross-validation on the pooled development dataset.

The learning rate selected in Run 02 (1e-3), the dropout rate selected in Run 03 (0.3), and the fixed training epoch selected in Run 04 (89) were fixed.

Each fold was trained for exactly 89 epochs, and predictions were generated from the final model state at epoch 89 for the corresponding held-out fold.

The test set was not used.

**---**

## 2. Methodology

```text
RUN 05

│
├── Development Dataset
│   ├── Train + Validation = 197 images
│   ├── Healthy = 125
│   ├── Sick = 72
│   └── Test = NOT USED
│
├── Fixed Configuration
│   ├── Learning rate = 1e-3
│   ├── Dropout rate = 0.3
│   └── Fixed epoch = 89
│
├── 5-Fold Stratified CV
│   ├── Fold 1
│   ├── Fold 2
│   ├── Fold 3
│   ├── Fold 4
│   └── Fold 5
│
├── Within Each Fold
│   ├── Fold-specific temperature range
│   ├── Preprocess train/validation images
│   ├── Calculate class weights from fold training data
│   ├── Train baseline CNN for exactly 89 epochs
│   ├── No early stopping
│   ├── No best-epoch checkpoint
│   ├── Generate predictions for held-out fold
│   └── Store OOF probabilities
│
├── OOF Verification
│   ├── Total development images = 197
│   ├── OOF predictions generated = 197
│   ├── Missing predictions = 0
│   └── Missing fold assignments = 0
│
├── OOF Evaluation
│   └── Overall OOF ROC-AUC = 0.880111
│
└── OOF Predictions
    └── Complete 197-image OOF prediction set
```

**---**

## 3. Configuration

| Parameter                 | Value                                |
| ------------------------- | ------------------------------------ |
| Input size                | 128 × 128 × 1                        |
| Batch size                | 16                                   |
| Fixed epoch               | 89                                   |
| Optimizer                 | Adam                                 |
| Learning rate             | 1e-3                                 |
| Dropout rate              | 0.3                                  |
| Augmentation              | Rotation 0.05, Zoom 0.05             |
| Folds                     | 5                                    |
| Seed                      | 10                                   |
| Device                    | CPU                                  |
| Class weighting           | Fold-specific balanced weights       |
| Temperature normalization | Fold-specific training-derived range |
| Threshold                 | Not applied                          |
| Test set                  | Not used                             |
| Output                    | OOF predictions                      |

**---**

## 4. Fixed-Epoch OOF Generation

The fixed training epoch used in Run 05 was selected in Run 04.

Run 04 evaluated the candidate epochs:

**12, 70, 89, 95, 100**

and selected **epoch 89** based on the highest mean 5-fold validation ROC-AUC.

Therefore, the training duration was fixed at 89 epochs for every fold in Run 05.

For each fold:

* A new baseline CNN was initialized.

* The learning rate remained fixed at 1e-3.

* The dropout rate remained fixed at 0.3.

* The model was trained for exactly 89 epochs.

* No early stopping was used.

* No best-validation-AUC checkpoint was used.

* Predictions were generated using the model state after epoch 89.

* The predictions were assigned to the corresponding held-out fold as OOF predictions.

The same five stratified CV splits were used across the folds.

The test set was not loaded or used.

**---**

## 5. Preprocessing

For each fold, the temperature range was calculated using **only the fold's training images**. The same range was then used to preprocess the corresponding validation images.

```text
Thermal image

    ↓

Temperature normalization

    ↓

Pad to square

    ↓

Resize to 128×128

    ↓

Add single channel
```

Class weights were also calculated separately for each fold using only the fold's training labels.

The held-out validation images did not contribute to the temperature-range or class-weight calculations.

**---**

## 6. OOF Prediction Generation

Each of the five cross-validation folds was used once as a held-out validation fold.

For each fold:

* The model was trained using only the fold training data.

* Training continued for exactly 89 epochs.

* The final model state after epoch 89 was retained.

* Predictions were generated for the held-out fold.

* The predictions were stored as continuous probabilities.

No classification threshold was applied.

The same process was repeated for all five folds, resulting in one OOF prediction for every image in the 197-image development dataset.

The OOF predictions were then combined into a single prediction file.

**---**

## 7. Results

The OOF predictions generated from all five folds were combined and evaluated across the complete development dataset.

### OOF Coverage

| Check                     | Result |
| ------------------------- | -----: |
| Total development images  |    197 |
| OOF predictions generated |    197 |
| Missing OOF predictions   |      0 |
| Missing fold assignments  |      0 |

All 197 development images received exactly one OOF prediction.

### Overall OOF ROC-AUC

**0.880111**

The overall OOF ROC-AUC calculated across all 197 OOF predictions was:

**0.8801111111**

This represents the ROC-AUC of the final fixed EXP01 CNN configuration when evaluated through out-of-fold predictions on the development dataset.

No threshold was applied when calculating the OOF ROC-AUC.

**### Threshold Selection**

Classification threshold selection was not performed in Run 05.

The OOF probabilities were retained for use in the subsequent threshold-selection stage.

**---**

## 8. Final Run 05 Decision

| Parameter                 | Value             |
| ------------------------- | ----------------- |
| Development dataset       | 197 images        |
| Cross-validation          | 5-fold stratified |
| Fixed learning rate       | 1e-3              |
| Fixed dropout rate        | 0.3               |
| Fixed epoch               | **89**            |
| OOF predictions generated | **197 / 197**     |
| Missing OOF predictions   | **0**             |
| Missing fold assignments  | **0**             |
| Overall OOF ROC-AUC       | **0.880111**      |
| Threshold selection       | **Not performed** |
| Test set                  | **Not used**      |

Run 05 successfully generated a complete set of OOF predictions for all 197 development images using the fixed EXP01 configuration and the fixed training duration of 89 epochs.

The resulting OOF probabilities will be carried forward to Run 06 for classification-threshold selection.

**---**

## 9. Output Structure

```text
Run_05_5-fold_CV_OOF_Predictions/

└── results/

    ├── fold_1_predictions.csv
    ├── fold_2_predictions.csv
    ├── fold_3_predictions.csv
    ├── fold_4_predictions.csv
    ├── fold_5_predictions.csv

    ├── fold_1_training_history.json
    ├── fold_2_training_history.json
    ├── fold_3_training_history.json
    ├── fold_4_training_history.json
    └── fold_5_training_history.json

    ├── oof_predictions.csv
    └── oof_summary.json
```

Each fold prediction file contains the image index, filepath, true label, and OOF probability.

The combined `oof_predictions.csv` contains the complete set of 197 OOF predictions.

The `oof_summary.json` file contains the run configuration, dataset information, fold-level summaries, and overall OOF ROC-AUC.

**---**

## 10. Final Summary

* Development images: **197**
* Cross-validation: **5-fold stratified**
* Fixed learning rate: **1e-3**
* Fixed dropout rate: **0.3**
* Fixed training epoch: **89**
* Total folds: **5**
* OOF predictions generated: **197 / 197**
* Missing OOF predictions: **0**
* Missing fold assignments: **0**
* Overall OOF ROC-AUC: **0.880111**
* Threshold selection: **Not performed**
* Test set: **Not used**

The EXP01 baseline CNN was successfully used to generate OOF predictions for all **197 development images** at the fixed training duration of **89 epochs**.

The complete OOF prediction set achieved an overall ROC-AUC of **0.880111**.

These OOF probabilities will be used in **Run 06 for classification-threshold selection**.
