# EXP03 Run 03: Fixed-Epoch Selection

## 1. Objective

Determine a fixed training epoch for the EXP03 partial fine-tuning ResNet-18 model using stratified 5-fold cross-validation on the pooled development dataset.

The candidate epochs were derived from the EXP03 Run 02 training dynamics.

During Run 03, each candidate epoch was evaluated using the same five stratified cross-validation folds. The ResNet-18 Layer4 block and classification head were trainable, while the earlier backbone layers remained frozen.

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
│   ├── Load five fold training histories
│   ├── Calculate mean validation ROC-AUC
│   ├── Identify individual fold best epochs
│   └── Identify aggregate mean-AUC best epoch
│
├── Fixed-Epoch Candidate Set
│   ├── Fold 1 best epoch = 10
│   ├── Fold 2 best epoch = 12
│   ├── Fold 3 best epoch = 70
│   ├── Fold 4 best epoch = 26
│   ├── Fold 5 best epoch = 5
│   ├── Aggregate mean-AUC best epoch = 26
│   └── Full training duration = 100
│
├── Final Candidate Epochs
│   └── 5, 10, 12, 26, 70, 100
│
├── Fixed 5-Fold Stratified CV
│   └── Same five folds used for every candidate epoch
│
├── Within Each Fold
│   ├── Reset seed to 10
│   ├── Calculate fold-specific temperature range
│   ├── Preprocess train/validation images
│   ├── Calculate class weights from fold training data
│   ├── Initialize ImageNet-pretrained ResNet-18
│   ├── Freeze backbone layers before Layer4
│   ├── Train Layer4
│   ├── Train classification head
│   ├── Use Layer4 learning rate = 1e-4
│   ├── Use classifier learning rate = 1e-3
│   ├── Train to the candidate fixed epoch
│   └── Calculate validation metrics
│
├── Candidate Comparison
│   ├── Mean validation ROC-AUC
│   ├── SD validation ROC-AUC
│   ├── Mean accuracy
│   ├── Mean balanced accuracy
│   ├── Mean sensitivity
│   ├── Mean specificity
│   ├── Mean precision
│   ├── Mean F1-score
│   └── Mean MCC
│
└── Final Fixed Epoch Selection
    ├── Primary criterion
    │   └── Highest mean 5-fold validation ROC-AUC
    │
    └── Tie-breaker
        └── Lower epoch
```

---

## 3. Configuration

| Parameter                | Value                            |
| ------------------------ | -------------------------------- |
| Model                    | ResNet-18                        |
| Pretrained weights       | ImageNet-1K V1                   |
| Thermal image size       | 224 × 224 × 1                    |
| ResNet-18 input          | 224 × 224 × 3                    |
| Channel conversion       | Grayscale repeated to 3 channels |
| Backbone                 | Partially frozen                 |
| Trainable backbone       | Layer4                           |
| Classification head      | 512 → 64 → 1                     |
| Dropout                  | 0.3                              |
| Batch size               | 16                               |
| Maximum epochs           | 100                              |
| Layer4 learning rate     | 1e-4                             |
| Classifier learning rate | 1e-3                             |
| Optimizer                | Adam                             |
| Augmentation             | Rotation 18°, scale 0.95–1.05    |
| Folds                    | 5                                |
| Threshold                | 0.5                              |
| Seed                     | 10                               |
| Seed reset               | At beginning of every fold/model |
| Device                   | CUDA                             |
| Selection metric         | Mean 5-fold validation ROC-AUC   |
| Test set                 | Not used                         |

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

Resize to 224 × 224

    ↓

Convert to tensor

    ↓

Repeat grayscale channel to 3 channels

    ↓

ImageNet normalization
```

Data augmentation was applied only to the training images.

The validation images were processed without augmentation.

Class weights were calculated separately for each fold using only the fold training data and were applied to the binary cross-entropy loss during training.

The same five stratified cross-validation splits were used for every candidate epoch so that the candidate epochs were evaluated on identical training and validation partitions.

---

## 5. Run 02 Candidate Epoch Derivation

The candidate epochs were derived from the EXP03 Run 02 training dynamics.

The five Run 02 validation histories contained 100 epochs each.

The aggregate mean validation ROC-AUC reached its highest value at **epoch 26**, with a mean validation ROC-AUC of **0.932914**.

The individual fold optima were:

| Fold | Best Epoch | Best Validation AUC |
| ---: | ---------: | ------------------: |
|    1 |         10 |            0.941333 |
|    2 |         12 |            0.952000 |
|    3 |         70 |            0.991429 |
|    4 |         26 |            0.951429 |
|    5 |          5 |            0.960000 |

The candidate set was therefore defined as:

| Candidate | Fixed Epoch |
| --------: | ----------: |
|         1 |           5 |
|         2 |          10 |
|         3 |          12 |
|         4 |          26 |
|         5 |          70 |
|         6 |         100 |

The candidate set was fixed before the Run 03 candidate comparison.

No arbitrary intermediate epochs were introduced.

---

## 6. Results

Each candidate epoch was independently trained across the same five stratified folds.

| Fixed Epoch | Mean ROC-AUC | SD ROC-AUC | Mean Accuracy | Mean Sensitivity | Mean Specificity |  Mean F1 |
| ----------: | -----------: | ---------: | ------------: | ---------------: | ---------------: | -------: |
|           5 |     0.920571 |   0.034775 |      0.847949 |         0.843810 |         0.848000 | 0.802373 |
|          10 | **0.932114** |   0.025803 |      0.847436 |         0.649524 |         0.960000 | 0.752019 |
|          12 |     0.923314 |   0.034431 |      0.862692 |         0.790476 |         0.904000 | 0.806799 |
|          26 |     0.930629 |   0.013967 |      0.862949 |         0.805714 |         0.896000 | 0.811431 |
|          70 |     0.915619 |   0.024693 |      0.802179 |         0.775238 |         0.816000 | 0.738574 |
|         100 |     0.926324 |   0.011275 |      0.862821 |         0.818095 |         0.888000 | 0.813286 |

Additional candidate-level metrics were also calculated and saved, including balanced accuracy, precision and MCC.

### Candidate 1: Epoch 5

Mean validation ROC-AUC was **0.920571 ± 0.034775**.

The mean accuracy was **0.847949**, with mean sensitivity of **0.843810** and mean specificity of **0.848000**.

### Candidate 2: Epoch 10

Mean validation ROC-AUC was **0.932114 ± 0.025803**.

The mean accuracy was **0.847436**, with mean sensitivity of **0.649524** and mean specificity of **0.960000**.

This candidate produced the highest mean 5-fold validation ROC-AUC among all evaluated fixed epochs.

### Candidate 3: Epoch 12

Mean validation ROC-AUC was **0.923314 ± 0.034431**.

The mean accuracy was **0.862692**, with mean sensitivity of **0.790476** and mean specificity of **0.904000**.

### Candidate 4: Epoch 26

Mean validation ROC-AUC was **0.930629 ± 0.013967**.

The mean accuracy was **0.862949**, with mean sensitivity of **0.805714** and mean specificity of **0.896000**.

### Candidate 5: Epoch 70

Mean validation ROC-AUC was **0.915619 ± 0.024693**.

The mean accuracy was **0.802179**, with mean sensitivity of **0.775238** and mean specificity of **0.816000**.

### Candidate 6: Epoch 100

Mean validation ROC-AUC was **0.926324 ± 0.011275**.

The mean accuracy was **0.862821**, with mean sensitivity of **0.818095** and mean specificity of **0.888000**.

---

## 7. Fixed Epoch Selection

The fixed epoch was selected using the following predefined criterion:

**Primary criterion:** Highest mean 5-fold validation ROC-AUC.

**Tie-breaker:** Lower training epoch.

The candidate comparison produced the following mean validation ROC-AUC values:

```text
Epoch 5    → 0.920571
Epoch 10   → 0.932114  ← highest
Epoch 12   → 0.923314
Epoch 26   → 0.930629
Epoch 70   → 0.915619
Epoch 100  → 0.926324
```

Therefore:

* **Selected fixed epoch:** **10**
* **Mean 5-fold validation ROC-AUC:** **0.932114**
* **Standard deviation:** **0.025803**

Epoch 10 was selected because it produced the highest mean validation ROC-AUC across the five folds.

The test set was not involved in candidate generation, candidate comparison or fixed-epoch selection.

---

## 8. Fold-Level Results for Selected Epoch

The selected fixed epoch was **epoch 10**.

The individual fold results were:

| Fold | Train Samples | Validation Samples |  ROC-AUC | Accuracy | Sensitivity | Specificity |
| ---: | ------------: | -----------------: | -------: | -------: | ----------: | ----------: |
|    1 |           157 |                 40 | 0.930667 | 0.900000 |    0.733333 |    1.000000 |
|    2 |           157 |                 40 | 0.941333 | 0.850000 |    0.800000 |    0.880000 |
|    3 |           158 |                 39 | 0.971429 | 0.820513 |    0.500000 |    1.000000 |
|    4 |           158 |                 39 | 0.891429 | 0.820513 |    0.642857 |    0.920000 |
|    5 |           158 |                 39 | 0.925714 | 0.846154 |    0.571429 |    1.000000 |

The corresponding mean values were:

| Metric      |         Mean |
| ----------- | -----------: |
| ROC-AUC     | **0.932114** |
| Accuracy    |     0.847436 |
| Sensitivity |     0.649524 |
| Specificity |     0.960000 |
| F1-score    |     0.752019 |

The selected epoch therefore reflects the cross-validation ROC-AUC criterion rather than optimization of accuracy, sensitivity, specificity or F1-score.

---

## 9. Fold-Specific Training Configuration

The temperature ranges and class weights were calculated independently for each fold.

| Fold | Temperature Min | Temperature Max | Healthy Weight | Sick Weight |
| ---: | --------------: | --------------: | -------------: | ----------: |
|    1 |       22.000000 |       36.812618 |       0.785000 |    1.377193 |
|    2 |       22.000000 |       36.779999 |       0.785000 |    1.377193 |
|    3 |       22.220350 |       36.812618 |       0.790000 |    1.362069 |
|    4 |       22.000000 |       36.812618 |       0.790000 |    1.362069 |
|    5 |       22.000000 |       36.812618 |       0.790000 |    1.362069 |

This fold-specific preprocessing protocol prevented the validation images from contributing to the temperature normalization range.

---

## 10. Output Structure

```text
Run_03_Fixed_Epoch_Selection/

├── results/
│   │
│   ├── candidate_epochs.json
│   ├── candidate_epoch_comparison.json
│   ├── candidate_epoch_comparison.csv
│   ├── run_summary.json
│   │
│   ├── epoch_5/
│   │   ├── fold_1/
│   │   ├── fold_2/
│   │   ├── fold_3/
│   │   ├── fold_4/
│   │   └── fold_5/
│   │
│   ├── epoch_10/
│   │   ├── fold_1/
│   │   ├── fold_2/
│   │   ├── fold_3/
│   │   ├── fold_4/
│   │   └── fold_5/
│   │
│   ├── epoch_12/
│   │   ├── fold_1/
│   │   ├── fold_2/
│   │   ├── fold_3/
│   │   ├── fold_4/
│   │   └── fold_5/
│   │
│   ├── epoch_26/
│   │   ├── fold_1/
│   │   ├── fold_2/
│   │   ├── fold_3/
│   │   ├── fold_4/
│   │   └── fold_5/
│   │
│   ├── epoch_70/
│   │   ├── fold_1/
│   │   ├── fold_2/
│   │   ├── fold_3/
│   │   ├── fold_4/
│   │   └── fold_5/
│   │
│   └── epoch_100/
│       ├── fold_1/
│       ├── fold_2/
│       ├── fold_3/
│       ├── fold_4/
│       └── fold_5/
│
└── README.md
```

Each candidate epoch contains five fold-specific results.

Each fold stores the trained model, validation predictions and fold-level metrics.

The candidate-level comparison is stored in both CSV and JSON formats.

---

## 11. Final Summary

* **Experiment:** EXP03_DMR_Anterior_ResNet18_Partial_FineTuning
* **Run:** Run 03 Fixed Epoch Selection
* **Development images:** 197
* **Healthy images:** 125
* **Sick images:** 72
* **Cross-validation:** 5-fold stratified
* **Candidate epochs:** 5, 10, 12, 26, 70, 100
* **Total candidate model trainings:** 30
* **Folds per candidate:** 5
* **ResNet-18 backbone:** Partially frozen
* **Trainable backbone:** Layer4
* **Classification head:** 512 → 64 → 1
* **Layer4 learning rate:** 1e-4
* **Classifier learning rate:** 1e-3
* **Classification head dropout:** 0.3
* **Batch size:** 16
* **Maximum training duration:** 100 epochs
* **Seed:** 10
* **Seed reset:** At the beginning of every fold/model
* **Normalization:** Fold-training-only temperature normalization
* **Selection criterion:** Highest mean 5-fold validation ROC-AUC
* **Selected fixed epoch:** **10**
* **Selected mean validation ROC-AUC:** **0.932114**
* **Selected ROC-AUC SD:** **0.025803**
* **Test set:** Not used

The experiment was completed successfully. Six predefined fixed training epochs were evaluated using identical five-fold stratified cross-validation splits. The fixed training epoch selected using the highest mean 5-fold validation ROC-AUC was **epoch 10**. The test set was not loaded, evaluated or used during candidate generation, preprocessing, training or fixed-epoch selection.