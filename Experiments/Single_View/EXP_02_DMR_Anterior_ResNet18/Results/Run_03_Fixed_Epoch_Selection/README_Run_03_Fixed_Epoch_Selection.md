# EXP02 Run 03: Fixed-Epoch Selection

## 1. Objective

Select a single fixed training epoch for the EXP02 frozen-backbone ResNet-18 model using stratified 5-fold cross-validation on the pooled development dataset.

The ResNet-18 architecture, ImageNet-1K pretrained weights, frozen backbone, dropout rate of 0.3 and learning rate of 1e-3 were fixed based on the EXP02 configuration.

Candidate epochs were derived from the Run 02 training dynamics and evaluated independently using 5-fold cross-validation.

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

├── Run 02 Training Dynamics

│   ├── Fold 1 best epoch = 68

│   ├── Fold 2 best epoch = 89

│   ├── Fold 3 best epoch = 67

│   ├── Fold 4 best epoch = 32

│   ├── Fold 5 best epoch = 23

│   └── Mean-AUC optimum = 32

│

├── Candidate Epochs

│   └── 23, 32, 67, 68, 89, 100

│

├── 5-Fold Stratified CV

│   ├── Epoch = 23  → 5 folds

│   ├── Epoch = 32  → 5 folds

│   ├── Epoch = 67  → 5 folds

│   ├── Epoch = 68  → 5 folds

│   ├── Epoch = 89  → 5 folds

│   └── Epoch = 100 → 5 folds

│

├── Within Each Fold

│   ├── Fold-specific temperature range

│   ├── Preprocess train/validation images

│   ├── Calculate class weights from fold training data

│   ├── Initialize ImageNet-pretrained ResNet-18

│   ├── Freeze ResNet-18 backbone

│   ├── Train classification head for fixed candidate epoch

│   ├── No early stopping

│   ├── No best-epoch checkpoint

│   └── Calculate validation metrics

│

├── Aggregate Results

│   ├── Mean ± SD ROC-AUC

│   ├── Accuracy

│   ├── Balanced Accuracy

│   ├── Sensitivity

│   ├── Specificity

│   ├── Precision

│   ├── F1-score

│   └── MCC

│

├── Select Fixed Epoch

│   └── Highest mean 5-fold validation ROC-AUC

│

└── Selected Fixed Epoch

    └── 32
```

---

## 3. Configuration

| Parameter           | Value                         |
| ------------------- | ----------------------------- |
| Model               | ResNet-18                     |
| Pretrained weights  | ImageNet-1K V1                |
| Input size          | 224 × 224 × 1                 |
| ResNet input        | 224 × 224 × 3                 |
| Backbone            | Frozen                        |
| Classification head | 512 → 64 → 1                  |
| Batch size          | 16                            |
| Maximum epochs      | 100                           |
| Optimizer           | Adam                          |
| Learning rate       | 1e-3                          |
| Dropout rate        | 0.3                           |
| Candidate epochs    | 23, 32, 67, 68, 89, 100       |
| Augmentation        | Rotation 18°, scale 0.95–1.05 |
| Folds               | 5                             |
| Threshold           | 0.5                           |
| Seed                | 10                            |
| Device              | CUDA                          |
| Selection metric    | Mean validation ROC-AUC       |

---

## 4. Candidate Epoch Selection

Candidate epochs were derived from the observed Run 02 epoch-wise validation AUC results.

The individual best epochs from the five Run 02 folds were:

| Fold   | Best Epoch | Best Validation AUC |
| ------ | ---------: | ------------------: |
| Fold 1 |         68 |            0.941333 |
| Fold 2 |         89 |            0.925333 |
| Fold 3 |         67 |            0.988571 |
| Fold 4 |         32 |            0.911429 |
| Fold 5 |         23 |            0.880000 |

The aggregate mean validation AUC across the five folds reached its maximum at **epoch 32**, with a mean AUC of **0.917800**.

The full 100-epoch training duration was also retained as a candidate reference.

After removing duplicate epochs, the final candidate set was:

**23, 32, 67, 68, 89, 100**

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

Resize to 224×224

    ↓

Add single channel

    ↓

Repeat grayscale channel to 3 channels

    ↓

ImageNet normalization
```

Training augmentation was applied only to the training images and consisted of random rotation up to 18° and random scaling between 0.95 and 1.05. No horizontal flipping was used.

Class weights were also calculated separately for each fold using only the fold's training labels.

The ResNet-18 backbone was kept frozen during training. Frozen backbone layers were maintained in evaluation mode to prevent changes to BatchNorm running statistics, while the classification head remained in training mode.

---

## 6. Fixed-Epoch Cross-Validation

Each candidate epoch was evaluated independently using the same five stratified folds.

For each candidate:

* A new ImageNet-pretrained ResNet-18 model was initialized for every fold.
* The ResNet-18 backbone remained frozen.
* The classification head was trained using a learning rate of 1e-3.
* The dropout rate remained fixed at 0.3.
* The model was trained for exactly the candidate number of epochs.
* No early stopping was used.
* No best-validation-AUC checkpoint was used for epoch selection.
* Validation predictions were generated after the specified fixed epoch.
* Validation metrics were calculated for each fold.

The same five CV splits were used for every candidate epoch to ensure a consistent comparison between training durations.

The test set was not loaded or used.

A total of **30 model trainings** were performed, corresponding to six candidate epochs evaluated across five folds.

---

## 7. Results

The Run 03 candidate comparison evaluated six fixed training durations. The primary selection criterion was the highest mean 5-fold validation ROC-AUC.

| Candidate Epoch |       Mean ROC-AUC ± SD |
| --------------: | ----------------------: |
|              23 |     0.906667 ± 0.049784 |
|          **32** | **0.912229 ± 0.045715** |
|              67 |     0.909524 ± 0.054601 |
|              68 |     0.908419 ± 0.055412 |
|              89 |     0.910210 ± 0.056134 |
|             100 |     0.904990 ± 0.060510 |

Epoch 32 achieved the highest mean 5-fold validation ROC-AUC among all candidate epochs.

### Additional Validation Metrics

| Candidate Epoch |     Accuracy | Balanced Accuracy | Sensitivity | Specificity | Precision |       F1 |      MCC |
| --------------: | -----------: | ----------------: | ----------: | ----------: | --------: | -------: | -------: |
|              23 |     0.842564 |          0.812857 |    0.705714 |    0.920000 |  0.866667 | 0.755876 | 0.667297 |
|          **32** | **0.852821** |          0.829810 |    0.747619 |    0.912000 |  0.860094 | 0.775036 | 0.692022 |
|              67 |     0.862949 |          0.847238 |    0.790476 |    0.904000 |  0.835233 | 0.803956 | 0.708051 |
|              68 |     0.857821 |          0.834762 |    0.749524 |    0.920000 |  0.856120 | 0.787845 | 0.696784 |
|              89 |     0.868077 |          0.848571 |    0.777143 |    0.920000 |  0.853252 | 0.806934 | 0.715505 |
|             100 |     0.847564 |          0.820952 |    0.721905 |    0.920000 |  0.879048 | 0.760977 | 0.684045 |

These additional metrics are reported for comparison but were not used to select the fixed epoch.

### Selected Fixed Epoch

**32**

Epoch 32 was selected because it achieved the highest mean 5-fold validation ROC-AUC among the candidate epochs.

**Mean 5-fold validation ROC-AUC: 0.912229 ± 0.045715**

The other validation metrics were not used for fixed-epoch selection.

---

## 8. Final Run 03 Decision

| Parameter                | Value                                  |
| ------------------------ | -------------------------------------- |
| Candidate epochs         | 23, 32, 67, 68, 89, 100                |
| **Selected fixed epoch** | **32**                                 |
| **Mean CV ROC-AUC**      | **0.912229 ± 0.045715**                |
| Selection criterion      | Highest mean 5-fold validation ROC-AUC |
| Test set                 | **Not used**                           |
| Total model trainings    | **30**                                 |

The selected fixed epoch of **32** will be carried forward to the subsequent EXP02 final model training and evaluation procedure.

---

## 9. Output Structure

```text
Run_03_Fixed_Epoch_Selection/

├── README.md

└── results/

    ├── candidate_epochs.json

    ├── candidate_epoch_comparison.json

    ├── candidate_epoch_comparison.csv

    ├── run_summary.json

    ├── epoch_23/

    │   ├── fold_1/

    │   ├── fold_2/

    │   ├── fold_3/

    │   ├── fold_4/

    │   └── fold_5/

    ├── epoch_32/

    │   ├── fold_1/

    │   ├── fold_2/

    │   ├── fold_3/

    │   ├── fold_4/

    │   └── fold_5/

    ├── epoch_67/

    │   ├── fold_1/

    │   ├── fold_2/

    │   ├── fold_3/

    │   ├── fold_4/

    │   └── fold_5/

    ├── epoch_68/

    │   ├── fold_1/

    │   ├── fold_2/

    │   ├── fold_3/

    │   ├── fold_4/

    │   └── fold_5/

    ├── epoch_89/

    │   ├── fold_1/

    │   ├── fold_2/

    │   ├── fold_3/

    │   ├── fold_4/

    │   └── fold_5/

    └── epoch_100/

        ├── fold_1/

        ├── fold_2/

        ├── fold_3/

        ├── fold_4/

        └── fold_5/
```

Each fold contains the trained model, validation predictions and fold-level results, including the training configuration, temperature range and class weights.

---

## 10. Final Summary

* Development images: **197**
* Healthy images: **125**
* Sick images: **72**
* Cross-validation: **5-fold stratified**
* Candidate epochs tested: **6**
* Candidate epochs: **23, 32, 67, 68, 89, 100**
* Total model trainings: **30**
* Maximum training duration: **100 epochs**
* Fixed learning rate: **1e-3**
* Fixed dropout rate: **0.3**
* Model: **ImageNet-pretrained ResNet-18 with frozen backbone**
* Selected fixed epoch: **32**
* Mean CV ROC-AUC: **0.912229 ± 0.045715**
* Selection criterion: **Highest mean 5-fold validation ROC-AUC**
* Test set: **Not used**

The selected fixed training duration for the EXP02 frozen-backbone ResNet-18 model is **32 epochs**. This value will be used in the subsequent EXP02 final model training and evaluation stage.