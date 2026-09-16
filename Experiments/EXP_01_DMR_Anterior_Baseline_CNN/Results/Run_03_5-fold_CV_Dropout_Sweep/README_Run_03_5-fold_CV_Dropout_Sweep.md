# EXP01 Run 03: 5-Fold CV Dropout Sweep

## 1. Objective

Select the dropout rate for the EXP01 baseline CNN using stratified 5-fold cross-validation on the pooled development dataset.

The learning rate selected in Run 02 (1e-3) was fixed.

The test set was not used.

---

## 2. Methodology

```text
RUN 03
│
├── Development Dataset
│   ├── Train + Validation = 197 images
│   ├── Healthy = 125
│   ├── Sick = 72
│   └── Test = NOT USED
│
├── 5-Fold Stratified CV
│   ├── Dropout = 0.3 → 5 folds
│   ├── Dropout = 0.5 → 5 folds
│   └── Dropout = 0.7 → 5 folds
│
├── Within Each Fold
│   ├── Fold-specific temperature range
│   ├── Preprocess train/validation images
│   ├── Calculate class weights from fold training data
│   ├── Train baseline CNN for 100 epochs
│   ├── Save best validation-AUC checkpoint
│   ├── Select best epoch
│   └── Calculate validation metrics
│
├── Aggregate Results
│   ├── Mean ± SD ROC-AUC
│   ├── Accuracy
│   ├── Sensitivity
│   ├── Specificity
│   └── F1-score
│
├── Select Dropout Rate
│   └── Highest mean 5-fold validation ROC-AUC
│
└── Selected Dropout Training Dynamics
    ├── Load five fold training histories
    ├── Calculate mean validation AUC for each epoch
    ├── Calculate SD validation AUC for each epoch
    └── Save epoch-wise results and plot
```

---

## 3. Configuration

| Parameter        | Value                    |
| ---------------- | ------------------------ |
| Input size       | 128 × 128 × 1            |
| Batch size       | 16                       |
| Epochs           | 100                      |
| Optimizer        | Adam                     |
| Learning rate    | 1e-3                     |
| Dropout rates    | 0.3, 0.5, 0.7            |
| Augmentation     | Rotation 0.05, Zoom 0.05 |
| Folds            | 5                        |
| Threshold        | 0.5                      |
| Seed             | 10                       |
| Device           | CPU                      |
| Selection metric | Mean validation ROC-AUC  |

---

## 4. Preprocessing

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

---

## 5. Results

| Dropout Rate |       Mean AUC ± SD | Accuracy | Sensitivity | Specificity |     F1 |
| ------------ | ------------------: | -------: | ----------: | ----------: | -----: |
| **0.3**      | **0.8987 ± 0.0506** |   0.7455 |      0.8867 |      0.6640 | 0.7138 |
| 0.5          |     0.8976 ± 0.0501 |   0.7558 |      0.9152 |      0.6640 | 0.7320 |
| 0.7          |     0.8954 ± 0.0506 |   0.7301 |      0.9438 |      0.6080 | 0.7219 |

### Selected Dropout Rate

**0.3**

The dropout rate was selected because it achieved the highest mean 5-fold validation ROC-AUC.

The other metrics were not used for dropout-rate selection.

---

## 6. Epoch-wise Validation AUC for Selected Dropout

After selecting dropout = 0.3, the validation AUC from each of the five fold training histories was aggregated epoch-wise.

For each epoch, the mean and standard deviation of validation ROC-AUC across the five folds were calculated.

The five fold training histories were not retrained or modified during this analysis.

The epoch-wise results were saved under:

```text
Results/
└── results/
    └── dropout_0.3/
        ├── epoch_wise_val_auc.csv
        ├── epoch_wise_val_auc.json
        └── epoch_wise_val_auc.png
```

These results provide the training dynamics used to define candidate fixed epochs for the subsequent fixed-epoch selection run.

---

## 7. Output Structure

```text
Run_03_5-fold_CV_Dropout_Sweep/

├── README.md

├── results/
│   ├── dropout_0.3/
│   │   ├── fold_1/
│   │   ├── fold_2/
│   │   ├── fold_3/
│   │   ├── fold_4/
│   │   ├── fold_5/
│   │   ├── epoch_wise_val_auc.csv
│   │   ├── epoch_wise_val_auc.json
│   │   └── epoch_wise_val_auc.png
│   │
│   ├── dropout_0.5/
│   └── dropout_0.7/
│
├── checkpoints/
└── plots/
```

Each dropout rate contains results for all five folds, including metrics, training history, ROC data, checkpoints, and plots.

---

## 8. Final Summary

* Development images: **197**
* Cross-validation: **5-fold stratified**
* Dropout rates tested: **3**
* Total model trainings: **15**
* Epochs per training: **100**
* Fixed learning rate: **1e-3**
* Selected dropout rate: **0.3**
* Mean CV ROC-AUC: **0.8987 ± 0.0506**
* Epoch-wise validation AUC: **Calculated across all 5 folds for the selected dropout**
* Test set: **Not used**
