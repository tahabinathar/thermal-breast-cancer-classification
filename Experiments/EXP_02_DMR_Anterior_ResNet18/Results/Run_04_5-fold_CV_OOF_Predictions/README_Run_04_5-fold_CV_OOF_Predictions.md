# EXP02 Run 04: 5-Fold Cross-Validation OOF Predictions

## 1. Objective

Generate out-of-fold (OOF) predictions for the final EXP02 frozen-backbone ResNet-18 configuration using stratified 5-fold cross-validation on the pooled development dataset.

The learning rate selected in the previous EXP02 runs (1e-3), the dropout rate (0.3), and the fixed training epoch selected in Run 03 (32) were fixed.

Each fold was trained for exactly 32 epochs, and predictions were generated from the final model state at epoch 32 for the corresponding held-out fold.

The test set was not loaded or used.

**---**

## 2. Methodology

```text
RUN 04

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
│   └── Fixed epoch = 32
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
│   ├── Reset random seed to 10
│   ├── Train frozen-backbone ResNet-18 for exactly 32 epochs
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
│   ├── Overall OOF ROC-AUC = 0.892667
│   └── Overall OOF PR-AUC = 0.832326
│
└── OOF Predictions
    └── Complete 197-image OOF prediction set
```

**---**

## 3. Configuration

| Parameter                 | Value                                               |
| ------------------------- | --------------------------------------------------- |
| Model                     | ResNet-18                                           |
| Pretrained weights        | ImageNet-1K V1                                      |
| Input size                | 224 × 224 × 1                                       |
| Model input               | Grayscale repeated to 3 channels                    |
| Batch size                | 16                                                  |
| Fixed epoch               | 32                                                  |
| Optimizer                 | Adam                                                |
| Learning rate             | 1e-3                                                |
| Dropout rate              | 0.3                                                 |
| Trainable parameters      | 32,897                                              |
| Backbone                  | Frozen                                              |
| Classifier                | Linear(512,64) → ReLU → Dropout(0.3) → Linear(64,1) |
| Augmentation              | Rotation 18°, affine scale 0.95–1.05                |
| Folds                     | 5                                                   |
| Seed                      | 10                                                  |
| Device                    | CUDA if available                                   |
| Class weighting           | Fold-specific balanced weights                      |
| Loss                      | Weighted BCE with logits                            |
| Temperature normalization | Fold-specific training-derived range                |
| Threshold                 | Not applied                                         |
| Test set                  | Not used                                            |
| Output                    | OOF predictions                                     |

**---**

## 4. Fixed-Epoch OOF Generation

The fixed training epoch used in Run 04 was selected in Run 03.

Run 03 evaluated the candidate epochs:

**23, 32, 67, 68, 89, 100**

and selected **epoch 32** based on the highest mean 5-fold validation ROC-AUC.

The mean validation ROC-AUC at epoch 32 was **0.912229 ± 0.045715**.

Therefore, the training duration was fixed at 32 epochs for every fold in Run 04.

For each fold:

* A new ImageNet-pretrained ResNet-18 was initialized.
* The entire ResNet-18 backbone was frozen.
* Only the classifier head was trainable.
* The learning rate remained fixed at 1e-3.
* The dropout rate remained fixed at 0.3.
* The model was trained for exactly 32 epochs.
* The random seed was reset to 10 before fold training.
* No early stopping was used.
* No best-validation-AUC checkpoint was used.
* Predictions were generated using the model state after epoch 32.
* The predictions were assigned to the corresponding held-out fold as OOF predictions.

The same five stratified CV splits were used across the run.

The test set was not loaded or used.

**---**

## 5. Preprocessing

For each fold, the temperature range was calculated using **only the fold's training images**. The same range was then used to preprocess the corresponding validation images.

```text
Thermal image

    ↓

Fold-specific temperature normalization

    ↓

Resize to 224×224

    ↓

Single-channel thermal image

    ↓

Repeat grayscale channel to 3 channels

    ↓

ImageNet normalization
```

Training augmentation consisted of:

* Random rotation up to 18°
* Random affine scaling between 0.95 and 1.05
* No horizontal flipping

Validation images were processed without augmentation.

Class weights were also calculated separately for each fold using only the fold's training labels.

The held-out validation images did not contribute to the temperature-range or class-weight calculations.

**---**

## 6. OOF Prediction Generation

Each of the five cross-validation folds was used once as a held-out validation fold.

For each fold:

* The model was trained using only the fold training data.
* Training continued for exactly 32 epochs.
* The final model state after epoch 32 was retained.
* Predictions were generated for the held-out fold.
* The predictions were stored as continuous probabilities.

No classification threshold was applied.

The same process was repeated for all five folds, resulting in one OOF prediction for every image in the 197-image development dataset.

The OOF predictions were then combined into a single prediction file.

**---**

## 7. Results

The OOF predictions generated from all five folds were combined and evaluated across the complete development dataset.

### Fold-Level Results

| Fold | Epochs | Learning Rate | Dropout Rate | Trainable Parameters | n_train | n_validation | Train Healthy | Train Sick | Validation Healthy | Validation Sick | Healthy Class Weight | Sick Class Weight | Temperature Min | Temperature Max |  ROC-AUC |   PR-AUC |
| ---: | -----: | ------------: | -----------: | -------------------: | ------: | -----------: | ------------: | ---------: | -----------------: | --------------: | -------------------: | ----------------: | --------------: | --------------: | -------: | -------: |
|    1 |     32 |         0.001 |          0.3 |               32,897 |     157 |           40 |           100 |         57 |                 25 |              15 |                0.785 |          1.377193 |        22.00000 |        36.81262 | 0.930667 | 0.949768 |
|    2 |     32 |         0.001 |          0.3 |               32,897 |     157 |           40 |           100 |         57 |                 25 |              15 |                0.785 |          1.377193 |        22.00000 |        36.78000 | 0.893333 | 0.805030 |
|    3 |     32 |         0.001 |          0.3 |               32,897 |     158 |           39 |           100 |         58 |                 25 |              14 |                0.790 |          1.362069 |        22.22035 |        36.81262 | 0.988571 | 0.981548 |
|    4 |     32 |         0.001 |          0.3 |               32,897 |     158 |           39 |           100 |         58 |                 25 |              14 |                0.790 |          1.362069 |        22.00000 |        36.81262 | 0.897143 | 0.836683 |
|    5 |     32 |         0.001 |          0.3 |               32,897 |     158 |           39 |           100 |         58 |                 25 |              14 |                0.790 |          1.362069 |        22.00000 |        36.81262 | 0.851429 | 0.822636 |

### OOF Coverage

| Check                     | Result |
| ------------------------- | -----: |
| Total development images  |    197 |
| OOF predictions generated |    197 |
| Missing OOF predictions   |      0 |
| Missing fold assignments  |      0 |

All 197 development images received exactly one OOF prediction.

### Overall OOF ROC-AUC

**0.892667**

The overall OOF ROC-AUC calculated across all 197 OOF predictions was:

**0.8926666667**

No classification threshold was applied when calculating the OOF ROC-AUC.

### Overall OOF PR-AUC

**0.832326**

The overall OOF PR-AUC calculated across all 197 OOF predictions was:

**0.8323261991**

No classification threshold was applied when calculating the OOF PR-AUC.

### Threshold Selection

Classification threshold selection was not performed in Run 04.

The OOF probabilities were retained for use in the subsequent threshold-selection stage.

**---**

## 8. Final Run 04 Decision

| Parameter                 | Value             |
| ------------------------- | ----------------- |
| Development dataset       | 197 images        |
| Cross-validation          | 5-fold stratified |
| Fixed learning rate       | 1e-3              |
| Fixed dropout rate        | 0.3               |
| Fixed epoch               | **32**            |
| Trainable parameters      | **32,897**        |
| OOF predictions generated | **197 / 197**     |
| Missing OOF predictions   | **0**             |
| Missing fold assignments  | **0**             |
| Overall OOF ROC-AUC       | **0.892667**      |
| Overall OOF PR-AUC        | **0.832326**      |
| Threshold selection       | **Not performed** |
| Test set                  | **Not used**      |

Run 04 successfully generated a complete set of OOF predictions for all 197 development images using the fixed EXP02 ResNet-18 configuration and the fixed training duration of 32 epochs.

The resulting OOF probabilities will be carried forward to the subsequent classification-threshold selection stage.

**---**

## 9. Output Structure

```text
Run_04_5-fold_CV_OOF_Predictions/

├── README.md

└── results/

    ├── fold_1_predictions.csv
    ├── fold_2_predictions.csv
    ├── fold_3_predictions.csv
    ├── fold_4_predictions.csv
    ├── fold_5_predictions.csv

    ├── fold_1_temperature_range.json
    ├── fold_2_temperature_range.json
    ├── fold_3_temperature_range.json
    ├── fold_4_temperature_range.json
    └── fold_5_temperature_range.json

    ├── fold_1_training_history.json
    ├── fold_2_training_history.json
    ├── fold_3_training_history.json
    ├── fold_4_training_history.json
    └── fold_5_training_history.json

    ├── fold_1_summary.json
    ├── fold_2_summary.json
    ├── fold_3_summary.json
    ├── fold_4_summary.json
    └── fold_5_summary.json

    ├── oof_predictions.csv
    ├── fold_summary.csv
    └── run_summary.json
```

Each fold prediction file contains the image index, filepath, true label, and OOF probability.

The combined `oof_predictions.csv` contains the complete set of 197 OOF predictions.

The `fold_summary.csv` contains the run configuration and fold-level performance summaries.

The `run_summary.json` contains the overall run configuration, dataset information, fold-level summaries, OOF coverage, and overall OOF metrics.

**---**

## 10. Final Summary

* Development images: **197**
* Healthy images: **125**
* Sick images: **72**
* Cross-validation: **5-fold stratified**
* Model: **ImageNet-pretrained ResNet-18**
* Backbone: **Frozen**
* Trainable parameters: **32,897**
* Fixed learning rate: **1e-3**
* Fixed dropout rate: **0.3**
* Fixed training epoch: **32**
* Total folds: **5**
* OOF predictions generated: **197 / 197**
* Missing OOF predictions: **0**
* Missing fold assignments: **0**
* Overall OOF ROC-AUC: **0.892667**
* Overall OOF PR-AUC: **0.832326**
* Threshold selection: **Not performed**
* Test set: **Not used**

The EXP02 frozen-backbone ResNet-18 was successfully used to generate OOF predictions for all **197 development images** at the fixed training duration of **32 epochs**.

The complete OOF prediction set achieved an overall ROC-AUC of **0.892667** and an overall PR-AUC of **0.832326**.

These OOF probabilities will be used in the subsequent **classification-threshold selection stage**.