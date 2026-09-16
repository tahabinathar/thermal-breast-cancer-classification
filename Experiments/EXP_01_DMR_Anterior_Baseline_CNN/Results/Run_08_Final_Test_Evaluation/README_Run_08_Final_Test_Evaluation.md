# EXP01_DMR_Anterior_Baseline

## Run 08: Final Test Evaluation

### 1. Objective

The objective of Run 08 was to perform the final evaluation of the trained EXP01 baseline CNN on the completely untouched independent Test dataset.

The final model trained in Run 07 was loaded without any further training or modification. The Test dataset was then preprocessed using the temperature normalization parameters calculated from the complete 197-image development dataset during Run 07.

The final classification threshold of 0.586, selected from the development-set OOF predictions in Run 06, was applied to convert the model's predicted probabilities into binary Healthy and Sick predictions.

The final Test dataset consisted of:

* Total = 37 images

* Healthy = 24

* Sick = 13

The Test dataset was not used in any previous experiment, including learning-rate selection, dropout selection, fixed-epoch selection, OOF prediction generation, threshold selection, or final model training.

No model configuration, threshold, or preprocessing parameter was modified based on the Test results.

---

### 2. Methodology

```text
RUN 08

│

├── Final Model

│   ├── Loaded from Run 07
│   ├── Trained on 197 development images
│   ├── Learning rate = 1e-3
│   ├── Dropout = 0.3
│   └── Final state after epoch 89

│

├── Test Dataset

│   ├── Healthy = 24 images
│   ├── Sick = 13 images
│   ├── Total = 37 images
│   └── Previously untouched

│

├── Temperature Normalization

│   ├── Range loaded from Run 07
│   ├── Calculated using development images only
│   ├── Global minimum
│   └── Global maximum

│

├── Model Prediction

│   ├── Test images preprocessed
│   ├── Final CNN loaded
│   └── Sigmoid probabilities generated

│

├── Classification

│   ├── Fixed threshold = 0.586
│   ├── Probability < 0.586 → Healthy
│   └── Probability ≥ 0.586 → Sick

│

└── Final Evaluation

    ├── ROC-AUC
    ├── Accuracy
    ├── Balanced Accuracy
    ├── Sensitivity
    ├── Specificity
    ├── Precision
    ├── F1-score
    ├── MCC
    └── Confusion Matrix
```

The final model from Run 07 was loaded directly without retraining.

The temperature normalization range stored during Run 07 was used to preprocess the Test images. The normalization range was not recalculated from the Test dataset.

The model generated a continuous sigmoid probability for each Test image.

ROC-AUC was calculated directly from these predicted probabilities.

The fixed threshold of 0.586 selected in Run 06 was then applied to generate binary predictions.

No threshold optimization was performed during Run 08.

No model selection was performed during Run 08.

---

### 3. Configuration

| Parameter                  | Value                          |
| -------------------------- | ------------------------------ |
| Experiment                 | EXP01_DMR_Anterior_Baseline    |
| Input dataset              | Test dataset                   |
| Test dataset               | 37 images                      |
| Healthy                    | 24                             |
| Sick                       | 13                             |
| Development dataset        | 197 images                     |
| Final model source         | Run 07                         |
| Input size                 | 128 × 128 × 1                  |
| Training epochs            | 89                             |
| Learning rate              | 1e-3                           |
| Dropout rate               | 0.3                            |
| Optimizer                  | Adam                           |
| Batch size during training | 16                             |
| Class weighting            | Balanced during training       |
| Random rotation            | 0.05 during training           |
| Random zoom                | 0.05 during training           |
| Horizontal flip            | False                          |
| Classification threshold   | **0.586**                      |
| Threshold source           | Run 06 OOF Threshold Selection |
| Temperature range source   | Run 07 Final Training          |
| Model training             | Not performed                  |
| Model selection            | None                           |
| Threshold selection        | None                           |
| Test evaluation            | Final                          |
| Random seed                | 10                             |
| Device                     | CPU                            |

The classification threshold of 0.586 was carried forward from Run 06 and was fixed before the Test evaluation.

The Test results were not used to modify the threshold.

---

### 4. Test Dataset Preparation

The independent Test dataset was loaded separately from the development dataset.

The Test dataset contained:

* Healthy = 24 images

* Sick = 13 images

* Total = 37 images

The resulting class distribution was:

```text
Test Dataset

Healthy = 24

Sick    = 13

Total   = 37
```

The Test dataset was accessed for the first time during Run 08.

No Test images were included in the development dataset used in Runs 02–07.

No Test labels were used for model training or configuration selection.

The Test dataset therefore remained completely isolated from the model-development process until this final evaluation.

---

### 5. Preprocessing

Each Test image was processed using the same preprocessing pipeline defined in `preprocessing.py`.

The preprocessing sequence was:

```text
Raw thermal image

        ↓

Temperature normalization

        ↓

Pad to square

        ↓

Resize to 128 × 128

        ↓

Add single-channel dimension

        ↓

Final tensor: 128 × 128 × 1
```

Temperature normalization used the global minimum and maximum calculated from the complete 197-image development dataset during Run 07.

The temperature range was loaded from the saved Run 07 `run_config.json`.

The normalization range was therefore not calculated from the Test images.

This ensured that the Test dataset did not contribute to the estimation of preprocessing parameters.

The processed Test images were stored as `float32` arrays.

No augmentation was applied during Test evaluation.

The Test images were passed directly through the final trained model to obtain prediction probabilities.

---

### 6. Final Model Evaluation

The final CNN saved during Run 07 was loaded from the saved model file.

The model represents the state after exactly 89 epochs of training on all 197 development images.

The model architecture and learned weights were not modified during Run 08.

For each Test image, the model generated a sigmoid output representing the predicted probability of the Sick class.

Two forms of evaluation were performed.

#### Probability-based evaluation

ROC-AUC was calculated using the continuous predicted probabilities.

This metric was calculated before applying the classification threshold and therefore evaluates the model's ability to distinguish between Healthy and Sick cases across possible thresholds.

#### Threshold-based evaluation

The fixed threshold selected in Run 06 was applied:

```text
Probability < 0.586  → Healthy

Probability ≥ 0.586 → Sick
```

The resulting binary predictions were used to calculate:

* Accuracy

* Balanced Accuracy

* Sensitivity

* Specificity

* Precision

* F1-score

* Matthews Correlation Coefficient

* Confusion Matrix

The threshold was not optimized using the Test dataset.

---

### 7. Results

The final evaluation was performed on all 37 independent Test images.

| Metric            |     Result |
| ----------------- | ---------: |
| Test images       |         37 |
| Healthy           |         24 |
| Sick              |         13 |
| ROC-AUC           | **0.8686** |
| Accuracy          |     0.6757 |
| Balanced Accuracy |     0.7324 |
| Sensitivity       | **0.9231** |
| Specificity       |     0.5417 |
| Precision         |     0.5217 |
| F1-score          |     0.6667 |
| MCC               |     0.4575 |
| Threshold         |      0.586 |

The final confusion matrix was:

```text
                 Predicted
                 Healthy   Sick
Actual Healthy      13       11
Actual Sick          1       12
```

Therefore:

```text
TN = 13
FP = 11
FN = 1
TP = 12
```

The model correctly identified 12 of the 13 Sick images.

This resulted in a sensitivity of:

```text
Sensitivity = 12 / 13 = 0.9231
```

The model correctly identified 13 of the 24 Healthy images.

This resulted in a specificity of:

```text
Specificity = 13 / 24 = 0.5417
```

The final Test ROC-AUC was:

```text
ROC-AUC = 0.8686
```

The Test evaluation was performed using the fixed threshold of 0.586 selected previously from the development-set OOF predictions.

---

### 8. Final Run 08 Decision

Run 08 provides the final independent evaluation of the EXP01 baseline CNN on the previously untouched Test dataset.

The final model was not retrained, modified, or selected during Test evaluation.

The classification threshold was not optimized using Test-set performance.

The final results are:

| Parameter                | Value      |
| ------------------------ | ---------- |
| Test dataset             | 37 images  |
| Healthy                  | 24         |
| Sick                     | 13         |
| Final model              | Run 07     |
| Training epoch           | **89**     |
| Learning rate            | 1e-3       |
| Dropout                  | 0.3        |
| Classification threshold | **0.586**  |
| **Test ROC-AUC**         | **0.8686** |
| Test Accuracy            | 0.6757     |
| Test Balanced Accuracy   | 0.7324     |
| Test Sensitivity         | **0.9231** |
| Test Specificity         | 0.5417     |
| Test Precision           | 0.5217     |
| Test F1-score            | 0.6667     |
| Test MCC                 | 0.4575     |

The final reported performance of the EXP01 baseline CNN on the independent Test dataset is therefore:

**Test ROC-AUC = 0.8686**

Using the fixed threshold of 0.586, the model achieved a sensitivity of **92.31%** and a specificity of **54.17%**.

The Test results were not used to modify any model or preprocessing configuration.

The Test dataset therefore served exclusively as the final held-out evaluation set.

---

### 9. Output Structure

```text
Run_08_Final_Test_Evaluation/

├── results/

│   ├── test_metrics.json

│   ├── test_predictions.csv

│   └── run_config.json

└── plots/

    ├── test_roc_curve.png

    └── test_confusion_matrix.png
```

#### `test_metrics.json`

Contains the final Test-set evaluation results, including:

* ROC-AUC

* Accuracy

* Balanced Accuracy

* Sensitivity

* Specificity

* Precision

* F1-score

* MCC

* Confusion matrix

* TN

* FP

* FN

* TP

* Classification threshold

#### `test_predictions.csv`

Contains the prediction results for each Test image, including the image identifier, true label, predicted probability, and binary prediction generated using the fixed threshold of 0.586.

#### `run_config.json`

Stores the Run 08 evaluation configuration, including:

* Experiment and run names

* Dataset information

* Test class distribution

* Final model source

* Input dimensions

* Training configuration

* Classification threshold

* Threshold source

* Temperature normalization source

* Test-set evaluation settings

* Device information

* Test-set usage status

#### Plots

Two final Test evaluation plots were generated:

1. **Test ROC curve**

2. **Test confusion matrix**

The ROC curve visualizes the classification performance across different probability thresholds.

The confusion matrix visualizes the final binary predictions obtained using the fixed threshold of 0.586.

---

### 10. Final Summary

* Experiment: **EXP01_DMR_Anterior_Baseline**

* Run: **Run 08 Final Test Evaluation**

* Development images: **197**

* Test images: **37**

* Healthy Test images: **24**

* Sick Test images: **13**

* Input size: **128 × 128 × 1**

* Final model source: **Run 07**

* Training epochs: **89**

* Learning rate: **1e-3**

* Dropout: **0.3**

* Optimizer: **Adam**

* Classification threshold: **0.586**

* Threshold source: **Run 06 OOF Threshold Selection**

* Temperature normalization source: **Run 07 Final Training**

* **Test ROC-AUC: 0.8686**

* Test Accuracy: **0.6757**

* Test Balanced Accuracy: **0.7324**

* Test Sensitivity: **0.9231**

* Test Specificity: **0.5417**

* Test Precision: **0.5217**

* Test F1-score: **0.6667**

* Test MCC: **0.4575**

* TN: **13**

* FP: **11**

* FN: **1**

* TP: **12**

Run 08 successfully completed the final independent evaluation of the EXP01 baseline CNN using the previously untouched 37-image Test dataset.

The final model from Run 07 was evaluated without retraining or modification. Test images were normalized using the temperature range calculated from the 197-image development dataset, and the fixed classification threshold of **0.586** selected in Run 06 was used for binary classification.

The final Test ROC-AUC was **0.8686**.

At the fixed threshold of **0.586**, the model achieved **92.31% sensitivity**, **54.17% specificity**, **67.57% accuracy**, and an **F1-score of 0.6667**.

The Test dataset was used exclusively for this final evaluation and was not used for model selection, threshold selection, preprocessing-range estimation, or training.