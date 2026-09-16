# EXP01 Run 02: 5-Fold CV Learning-Rate Sweep

## 1. Objective

Select the learning rate for the EXP01 baseline CNN using stratified 5-fold cross-validation on the pooled development dataset.

The test set was not used.

---

## 2. Methodology

```text
RUN 02
│
├── Development Dataset
│   ├── Train + Validation = 197 images
│   ├── Healthy = 125
│   ├── Sick = 72
│   └── Test = NOT USED
│
├── 5-Fold Stratified CV
│   ├── LR = 1e-3 → 5 folds
│   ├── LR = 5e-4 → 5 folds
│   └── LR = 1e-4 → 5 folds
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
└── Select Learning Rate
    └── Highest mean 5-fold validation ROC-AUC
```

---

## 3. Configuration

| Parameter        | Value                    |
| ---------------- | ------------------------ |
| Input size       | 128 × 128 × 1            |
| Batch size       | 16                       |
| Epochs           | 100                      |
| Optimizer        | Adam                     |
| Learning rates   | 1e-3, 5e-4, 1e-4         |
| Dropout          | 0.5                      |
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

| Learning Rate |       Mean AUC ± SD | Accuracy | Sensitivity | Specificity |     F1 |
| ------------- | ------------------: | -------: | ----------: | ----------: | -----: |
| **1e-3**      | **0.8976 ± 0.0501** |   0.7558 |      0.9152 |      0.6640 | 0.7320 |
| 5e-4          |     0.8915 ± 0.0475 |   0.7710 |      0.9438 |      0.6720 | 0.7514 |
| 1e-4          |     0.8746 ± 0.0431 |   0.7917 |      0.8610 |      0.7520 | 0.7497 |

### Selected Learning Rate

**1e-3**

The learning rate was selected because it achieved the highest mean 5-fold validation ROC-AUC.

The other metrics were not used for learning-rate selection.

---

## 6. Output Structure

```text
Run_02_5-fold_CV_LR_Sweep/
├── README.md
├── results/
├── checkpoints/
└── plots/
```

Each learning rate contains results for all five folds, including metrics, training history, ROC data, checkpoints, and plots.

---

## 7. Final Summary

* Development images: **197**
* Cross-validation: **5-fold stratified**
* Learning rates tested: **3**
* Total model trainings: **15**
* Epochs per training: **100**
* Selected learning rate: **1e-3**
* Mean CV ROC-AUC: **0.8976 ± 0.0501**
* Test set: **Not used**