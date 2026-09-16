# EXP01 Run 04: Fixed-Epoch Selection

## 1. Objective

Select a single fixed training epoch for the EXP01 baseline CNN using stratified 5-fold cross-validation on the pooled development dataset.

The learning rate selected in Run 02 (1e-3) and the dropout rate selected in Run 03 (0.3) were fixed.

Candidate epochs were derived from the Run 03 training dynamics and evaluated independently using 5-fold cross-validation.

The test set was not used.

---

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
├── Run 03 Training Dynamics
│   ├── Fold 1 best epoch = 70
│   ├── Fold 2 best epoch = 100
│   ├── Fold 3 best epoch = 95
│   ├── Fold 4 best epoch = 89
│   ├── Fold 5 best epoch = 12
│   └── Mean-AUC optimum = 89
│
├── Candidate Epochs
│   └── 12, 70, 89, 95, 100
│
├── 5-Fold Stratified CV
│   ├── Epoch = 12  → 5 folds
│   ├── Epoch = 70  → 5 folds
│   ├── Epoch = 89  → 5 folds
│   ├── Epoch = 95  → 5 folds
│   └── Epoch = 100 → 5 folds
│
├── Within Each Fold
│   ├── Fold-specific temperature range
│   ├── Preprocess train/validation images
│   ├── Calculate class weights from fold training data
│   ├── Train baseline CNN for the fixed candidate epoch
│   ├── No early stopping
│   ├── No best-epoch checkpoint
│   └── Calculate validation metrics
│
├── Aggregate Results
│   ├── Mean ± SD ROC-AUC
│   ├── Accuracy
│   ├── Sensitivity
│   ├── Specificity
│   └── F1-score
│
├── Select Fixed Epoch
│   └── Highest mean 5-fold validation ROC-AUC
│
└── Selected Fixed Epoch
    └── 89
```

---

## 3. Configuration

| Parameter        | Value                    |
| ---------------- | ------------------------ |
| Input size       | 128 × 128 × 1            |
| Batch size       | 16                       |
| Maximum epochs   | 100                      |
| Optimizer        | Adam                     |
| Learning rate    | 1e-3                     |
| Dropout rate     | 0.3                      |
| Candidate epochs | 12, 70, 89, 95, 100      |
| Augmentation     | Rotation 0.05, Zoom 0.05 |
| Folds            | 5                        |
| Threshold        | 0.5                      |
| Seed             | 10                       |
| Device           | CPU                      |
| Selection metric | Mean validation ROC-AUC  |

---

## 4. Candidate Epoch Selection

Candidate epochs were derived from the observed Run 03 epoch-wise validation AUC results.

The individual best epochs from the five Run 03 folds were:

| Fold   | Best Epoch | Best Validation AUC |
| ------ | ---------: | ------------------: |
| Fold 1 |         70 |            0.917333 |
| Fold 2 |        100 |            0.930667 |
| Fold 3 |         95 |            0.938571 |
| Fold 4 |         89 |            0.910000 |
| Fold 5 |         12 |            0.797143 |

The aggregate mean validation AUC across the five folds reached its maximum at **epoch 89**, with a mean AUC of **0.891409**.

The full 100-epoch training duration was also retained as a candidate reference.

After removing duplicate epochs, the final candidate set was:

**12, 70, 89, 95, 100**

No arbitrary intermediate epochs were introduced.

---

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

---

## 6. Fixed-Epoch Cross-Validation

Each candidate epoch was evaluated independently using the same five stratified folds.

For each candidate:

* A new baseline CNN was initialized for every fold.
* The learning rate remained fixed at 1e-3.
* The dropout rate remained fixed at 0.3.
* The model was trained for exactly the candidate number of epochs.
* No early stopping was used.
* No best-validation-AUC checkpoint was used.
* Validation predictions were generated after the specified fixed epoch.
* Validation metrics were calculated for each fold.

The same five CV splits were used for every candidate epoch to ensure a fair comparison between training durations.

The test set was not loaded or used.

---

## 7. Results

The Run 04 candidate comparison evaluated five fixed training durations. The primary selection criterion was the highest mean 5-fold validation ROC-AUC.

| Candidate Epoch |       Mean ROC-AUC ± SD |
| --------------: | ----------------------: |
|              12 |     0.869829 ± 0.039429 |
|              70 |     0.881676 ± 0.066347 |
|          **89** | **0.888038 ± 0.059973** |
|              95 |     0.883390 ± 0.059924 |
|             100 |     0.887429 ± 0.061187 |

Epoch 89 achieved the highest mean 5-fold validation ROC-AUC among all candidate epochs.

### Additional Validation Metrics

| Candidate Epoch |     Accuracy | Balanced Accuracy |  Sensitivity |  Specificity |    Precision |           F1 |          MCC |
| --------------: | -----------: | ----------------: | -----------: | -----------: | -----------: | -----------: | -----------: |
|              12 |     0.656282 |          0.706000 |     0.900000 |     0.512000 |     0.544758 |     0.666967 |     0.407795 |
|              70 |     0.745256 |          0.787905 |     0.943810 |     0.632000 |     0.608032 |     0.736218 |     0.564167 |
|          **89** | **0.724872** |      **0.771905** | **0.943810** | **0.600000** | **0.584067** | **0.718397** | **0.538259** |
|              95 |     0.735000 |          0.779905 |     0.943810 |     0.616000 |     0.596377 |     0.727174 |     0.552874 |
|             100 |     0.719615 |          0.767905 |     0.943810 |     0.592000 |     0.583345 |     0.716903 |     0.531561 |

These additional metrics are reported for comparison but were not used to select the fixed epoch.

### Selected Fixed Epoch

**89**

Epoch 89 was selected because it achieved the highest mean 5-fold validation ROC-AUC among the candidate epochs.

**Mean 5-fold validation ROC-AUC: 0.888038 ± 0.059973**

The other validation metrics were not used for fixed-epoch selection.

---

## 8. Final Run 04 Decision

| Parameter                | Value                                  |
| ------------------------ | -------------------------------------- |
| Candidate epochs         | 12, 70, 89, 95, 100                    |
| **Selected fixed epoch** | **89**                                 |
| **Mean CV ROC-AUC**      | **0.888038 ± 0.059973**                |
| Selection criterion      | Highest mean 5-fold validation ROC-AUC |
| Test set                 | **Not used**                           |

The selected fixed epoch of **89** will be carried forward to the subsequent EXP01 runs.

---

## 9. Output Structure

```text
Run_04_Fixed_Epoch_Selection/

├── README.md

└── results/

    ├── candidate_epochs.json
    │
    ├── candidate_epoch_comparison.json
    ├── candidate_epoch_comparison.csv
    ├── run_summary.json
    │
    ├── epoch_12/
    │   ├── fold_1/
    │   ├── fold_2/
    │   ├── fold_3/
    │   ├── fold_4/
    │   └── fold_5/
    │
    ├── epoch_70/
    │   ├── fold_1/
    │   ├── fold_2/
    │   ├── fold_3/
    │   ├── fold_4/
    │   └── fold_5/
    │
    ├── epoch_89/
    │   ├── fold_1/
    │   ├── fold_2/
    │   ├── fold_3/
    │   ├── fold_4/
    │   └── fold_5/
    │
    ├── epoch_95/
    │   ├── fold_1/
    │   ├── fold_2/
    │   ├── fold_3/
    │   ├── fold_4/
    │   └── fold_5/
    │
    └── epoch_100/
        ├── fold_1/
        ├── fold_2/
        ├── fold_3/
        ├── fold_4/
        └── fold_5/
```

Each fold contains the trained model, validation predictions, training history, fold metrics, temperature range, and class weights.

---

## 10. Final Summary

* Development images: **197**
* Cross-validation: **5-fold stratified**
* Candidate epochs tested: **5**
* Candidate epochs: **12, 70, 89, 95, 100**
* Total model trainings: **25**
* Maximum training duration: **100 epochs**
* Fixed learning rate: **1e-3**
* Fixed dropout rate: **0.3**
* Selected fixed epoch: **89**
* Mean CV ROC-AUC: **0.888038 ± 0.059973**
* Selection criterion: **Highest mean 5-fold validation ROC-AUC**
* Test set: **Not used**

The selected fixed training duration for the EXP01 baseline CNN is **89 epochs**. This value will be used in the subsequent OOF prediction and threshold-selection stages.