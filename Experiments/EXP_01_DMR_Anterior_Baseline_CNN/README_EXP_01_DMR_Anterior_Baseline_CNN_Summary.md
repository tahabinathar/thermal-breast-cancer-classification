# EXP01: DMR Anterior Baseline CNN

## 1. Overview

### Experiment Name

**EXP01_DMR_Anterior_Baseline**

### Objective

EXP01 establishes and evaluates a baseline CNN for binary classification of anterior-view breast thermal images from the DMR-IR dataset.

The experiment consists of eight sequential runs. Each run determines or applies one component of the final modeling configuration, progressing from the initial baseline model through hyperparameter selection, fixed-epoch selection, out-of-fold prediction generation, threshold selection, final model training, and independent test evaluation.

### Development and Test Framework

Run 01 used the original Training and Validation sets separately. From Run 02 onward, these sets were combined into a 197-image Development set.

The 37-image Test set remained completely untouched until Run 08.

| Split               |  Images | Healthy |   Sick |
| ------------------- | ------: | ------: | -----: |
| Original Training   |     163 |     103 |     60 |
| Original Validation |      34 |      22 |     12 |
| Development         |     197 |     125 |     72 |
| Test                |      37 |      24 |     13 |
| **Total**           | **234** | **149** | **85** |

**Labels:**

* Healthy = `0`
* Sick = `1`

The Development set consists of the original Training and Validation sets:

`163 + 34 = 197 images`

The independent Test set contains:

`24 Healthy + 13 Sick = 37 images`

---

## 2. ROI Pixel Value Inspection

This inspection was performed on the cropped **DMR-IR Dataset regions of interest (ROIs)** to identify potentially abnormal pixel values.

A threshold of **50** was used to flag potentially abnormal pixels across the Training, Validation, and Test splits.

### Inspection Results

| Split      | Healthy |   Sick |
| ---------- | ------: | -----: |
| Train      |     103 |     60 |
| Validation |      22 |     12 |
| Test       |      24 |     13 |
| **Total**  | **149** | **85** |

**Result:** No abnormal pixel values were detected in any image.

```text
flagged_files: []
```

All **234 ROI images** passed the inspection without any flagged pixel values.

This was a dataset quality-control step only. It did not involve model training or model selection.

---

## 3. Preprocessing Check

This check verifies the preprocessing pipeline used for the EXP01 baseline CNN on the DMR-IR Anterior View ROI dataset.

### 3.1 Preprocessing Pipeline

Each thermal image is processed as follows:

**1. Load thermal image**

The `.tif` image is loaded as a 2D floating-point array.

**2. Calculate temperature range**

The minimum and maximum valid temperature values (`> 0`) are calculated from the images used for the relevant preprocessing stage.

**3. Temperature normalization**

Valid temperature values are normalized to `[0, 1]` using:

```text
normalized = (temperature - global_min) /
             (global_max - global_min)
```

Invalid/background pixels (`≤ 0`) remain zero.

**4. Zero padding**

Non-square images are padded with zeros to produce a square image. The original image is centered within the padded array.

**5. Resize**

The square image is resized to `128 × 128` pixels using bilinear interpolation.

**6. Add channel dimension**

A single channel dimension is added, producing:

```text
128 × 128 × 1
```

**7. Output format**

The final array is stored as `float32`.

### 3.2 Verification

The standalone preprocessing check verifies that:

* all images load successfully;
* all images are 2D;
* valid temperature values are present;
* the temperature range can be calculated;
* normalized values remain within `[0, 1]`;
* the final output shape is `128 × 128 × 1`;
* the final output dtype is `float32`.

### 3.3 Result

All **234 images** passed the preprocessing checks with no failed files.

The detailed verification results are stored in:

```text
preprocessing_check.json
```

> **Note:** This check verifies the preprocessing implementation. The temperature range used during actual model development is calculated from the appropriate training/development data and then applied to the corresponding validation or held-out data. The independent Test set is not used to calculate normalization parameters.

---

## 4. Preprocessing Used in EXP01

The preprocessing pipeline used for model input is:

```text
Raw thermal image
        ↓
Temperature normalization
        ↓
Pad to square
        ↓
Resize to 128 × 128
        ↓
Add single channel
        ↓
Final tensor: 128 × 128 × 1
```

For cross-validation runs, the temperature range is calculated separately within each fold using only the fold's training images. The resulting range is then applied to both the training and held-out portions of that fold.

For final training in Run 07, the temperature range is calculated using all 197 Development images.

The Test set is not used to calculate normalization parameters.

---

## 5. Baseline CNN Architecture

The baseline CNN architecture was established in Run 01. The dropout rate was subsequently tuned from `0.5` to the final value of `0.3` in Run 03. The remaining architecture was kept fixed.

```text
Input: 128 × 128 × 1
        ↓
Random Rotation (factor = 0.05)
        ↓
Random Zoom (factor = 0.05)
        ↓
Conv2D: 32 filters, 2 × 2, ReLU, same padding
        ↓
Conv2D: 32 filters, 2 × 2, ReLU, same padding
        ↓
MaxPooling2D
        ↓
Conv2D: 64 filters, 2 × 2, ReLU, same padding
        ↓
MaxPooling2D
        ↓
Conv2D: 128 filters, 2 × 2, ReLU, same padding
        ↓
MaxPooling2D
        ↓
GlobalAveragePooling2D
        ↓
Dense: 64 units, ReLU
        ↓
Dropout: 0.3
        ↓
Dense: 1 unit, Sigmoid
```

### Common Training Configuration

| Parameter       | Setting              |
| --------------- | -------------------- |
| Optimizer       | Adam                 |
| Batch size      | 16                   |
| Loss            | Binary Cross-Entropy |
| Class weighting | Balanced             |
| Random seed     | 10                   |
| Input           | `128 × 128 × 1`      |
| Final dropout   | 0.3                  |
| Hardware        | CPU                  |

---

# 6. Run 01: Baseline CNN

## Objective

To establish an initial CNN baseline using the original Training and Validation sets.

The Test set was not used.

## Configuration

| Parameter       | Setting                 |
| --------------- | ----------------------- |
| Training data   | 163 images              |
| Validation data | 34 images               |
| Learning rate   | 0.001                   |
| Dropout         | 0.5                     |
| Maximum epochs  | 100                     |
| Early stopping  | None                    |
| Model selection | Best validation ROC-AUC |
| Test set        | Not used                |

## Result

The best model was obtained at **epoch 41**.

| Metric      | Validation Result |
| ----------- | ----------------: |
| ROC-AUC     |        **0.9318** |
| Accuracy    |            79.41% |
| Sensitivity |           100.00% |
| Specificity |            68.18% |
| Precision   |            63.16% |
| F1-score    |            77.42% |

### Confusion Matrix

|                    | Predicted Healthy | Predicted Sick |
| ------------------ | ----------------: | -------------: |
| **Actual Healthy** |                15 |              7 |
| **Actual Sick**    |                 0 |             12 |

All 12 Sick validation images were correctly detected, while 7 Healthy images were classified as Sick.

---

# 7. Run 02: Learning Rate Sweep

## Objective

To select the learning rate using stratified 5-fold cross-validation on the 197-image Development set.

Dropout was fixed at `0.5`.

## Learning Rates Tested

* `0.001`
* `0.0005`
* `0.0001`

## Results

| Learning Rate |       Mean AUC ± SD | Accuracy | Sensitivity | Specificity |     F1 |
| ------------: | ------------------: | -------: | ----------: | ----------: | -----: |
|     **0.001** | **0.8976 ± 0.0501** |   0.7558 |      0.9152 |      0.6640 | 0.7320 |
|        0.0005 |     0.8915 ± 0.0475 |   0.7710 |      0.9438 |      0.6720 | 0.7514 |
|        0.0001 |     0.8746 ± 0.0431 |   0.7917 |      0.8610 |      0.7520 | 0.7497 |

## Decision

**Selected learning rate: `0.001`**

The learning rate of `0.001` produced the highest mean 5-fold cross-validation ROC-AUC.

---

# 8. Run 03: Dropout Sweep

## Objective

To select the dropout rate while keeping the learning rate fixed at `0.001`.

## Dropout Values Tested

* `0.3`
* `0.5`
* `0.7`

## Results

| Dropout |       Mean AUC ± SD | Accuracy | Sensitivity | Specificity |     F1 |
| ------: | ------------------: | -------: | ----------: | ----------: | -----: |
| **0.3** | **0.8987 ± 0.0506** |   0.7455 |      0.8867 |      0.6640 | 0.7138 |
|     0.5 |     0.8976 ± 0.0501 |   0.7558 |      0.9152 |      0.6640 | 0.7320 |
|     0.7 |     0.8954 ± 0.0506 |   0.7301 |      0.9438 |      0.6080 | 0.7219 |

## Decision

**Selected dropout: `0.3`**

The dropout rate of `0.3` produced the highest mean 5-fold cross-validation ROC-AUC.

Epoch-wise validation AUC was also logged for the selected configuration and used to determine candidate epochs for Run 04.

### Fold-Best Epochs

| Fold   | Best Epoch |
| ------ | ---------: |
| Fold 1 |         70 |
| Fold 2 |        100 |
| Fold 3 |         95 |
| Fold 4 |         89 |
| Fold 5 |         12 |

These produced the candidate epoch set:

```text
12, 70, 89, 95, 100
```

---

# 9. Run 04: Fixed Epoch Selection

## Objective

To select one fixed training epoch after the learning rate and dropout had been finalized.

### Configuration

* Learning rate = `0.001`
* Dropout = `0.3`
* Stratified 5-fold cross-validation

## Results

|  Epoch |   Mean ROC-AUC ± SD | Accuracy | Sensitivity | Specificity |
| -----: | ------------------: | -------: | ----------: | ----------: |
|     12 |     0.8698 ± 0.0394 |   0.6563 |      0.9000 |      0.5120 |
|     70 |     0.8817 ± 0.0663 |   0.7453 |      0.9438 |      0.6320 |
| **89** | **0.8880 ± 0.0600** |   0.7249 |      0.9438 |      0.6000 |
|     95 |     0.8834 ± 0.0599 |   0.7350 |      0.9438 |      0.6160 |
|    100 |     0.8874 ± 0.0612 |   0.7196 |      0.9438 |      0.5920 |

## Decision

**Selected fixed training epoch: `89`**

Epoch 89 produced the highest mean 5-fold cross-validation ROC-AUC.

---

# 10. Run 05: 5-Fold OOF Predictions

## Objective

To generate out-of-fold (OOF) predictions using the selected configuration from Runs 02–04.

### Fixed Configuration

| Parameter                        | Setting           |
| -------------------------------- | ----------------- |
| Learning rate                    | 0.001             |
| Dropout                          | 0.3               |
| Epochs                           | 89                |
| Cross-validation                 | Stratified 5-fold |
| Checkpointing                    | None              |
| Early stopping                   | None              |
| Validation-based epoch selection | None              |

Each fold was trained from scratch for exactly 89 epochs. Predictions were generated only for the held-out portion of each fold.

## Result

| Quantity            |     Result |
| ------------------- | ---------: |
| Development images  |        197 |
| OOF predictions     |        197 |
| Missing predictions |          0 |
| Number of folds     |          5 |
| Overall OOF ROC-AUC | **0.8801** |

Exact OOF ROC-AUC:

```text
0.8801111111111112
```

No classification threshold was selected in this run.

---

# 11. Run 06: OOF Threshold Selection

## Objective

To select a single classification threshold using only the OOF probabilities generated in Run 05.

The threshold was swept from `0.000` to `1.000` in increments of `0.001`.

### Selection Constraints

A threshold had to satisfy:

```text
Sensitivity > 0.90
Specificity > 0.60
```

A total of **138 thresholds** satisfied these constraints.

The selection priority was:

1. Highest sensitivity
2. Highest specificity
3. Highest threshold as a tiebreaker

## Selected Threshold

**`0.586`**

## Results

| Metric      | Result |
| ----------- | -----: |
| Sensitivity | 0.9444 |
| Specificity | 0.6160 |
| Accuracy    | 0.7360 |
| Precision   | 0.5862 |
| F1-score    | 0.7234 |

### Confusion Matrix

|                    | Predicted Healthy | Predicted Sick |
| ------------------ | ----------------: | -------------: |
| **Actual Healthy** |                77 |             48 |
| **Actual Sick**    |                 4 |             68 |

The threshold was selected exclusively from Development-set OOF predictions. The Test set was not used.

---

# 12. Run 07: Final Model Training

## Objective

To train the final EXP01 CNN using all 197 Development images and the configuration selected in Runs 02–06.

### Training Data

| Class     |  Images |
| --------- | ------: |
| Healthy   |     125 |
| Sick      |      72 |
| **Total** | **197** |

### Configuration

| Parameter          | Setting              |
| ------------------ | -------------------- |
| Input              | `128 × 128 × 1`      |
| Learning rate      | 0.001                |
| Dropout            | 0.3                  |
| Epochs             | 89                   |
| Batch size         | 16                   |
| Optimizer          | Adam                 |
| Loss               | Binary Cross-Entropy |
| Class weighting    | Balanced             |
| Validation data    | None                 |
| Checkpointing      | None                 |
| Early stopping     | None                 |
| Decision threshold | 0.586                |

The model was trained for exactly 89 epochs. The state of the model after epoch 89 was saved as the final EXP01 model.

Temperature normalization parameters and class weights were calculated using all 197 Development images.

The threshold `0.586` was not used during training. It is applied only during classification of model probabilities.

No Test images were accessed.

---

# 13. Run 08: Final Test Evaluation

## Objective

To evaluate the final Run 07 model on the completely untouched 37-image Test set.

### Procedure

1. Load the final Run 07 model.
2. Load the saved temperature normalization parameters.
3. Load the 37 Test images.
4. Generate sigmoid probabilities.
5. Calculate ROC-AUC using the raw probabilities.
6. Apply the predetermined threshold of `0.586`.
7. Calculate the final classification metrics.

The temperature range was not recalculated from the Test set.

## Final Test Results

| Metric            |     Result |
| ----------------- | ---------: |
| ROC-AUC           | **0.8686** |
| Accuracy          | **67.57%** |
| Balanced Accuracy | **73.24%** |
| Sensitivity       | **92.31%** |
| Specificity       | **54.17%** |
| Precision         | **52.17%** |
| F1-score          | **66.67%** |
| MCC               | **0.4575** |

### Confusion Matrix

|                    | Predicted Healthy | Predicted Sick |
| ------------------ | ----------------: | -------------: |
| **Actual Healthy** |                13 |             11 |
| **Actual Sick**    |                 1 |             12 |

The final model correctly classified:

* **12/13 Sick images**
* **13/24 Healthy images**

---

# 14. Final EXP01 Configuration

| Component            | Final Setting                |
| -------------------- | ---------------------------- |
| Input                | `128 × 128 × 1`              |
| Architecture         | Baseline CNN                 |
| Learning rate        | `0.001`                      |
| Dropout              | `0.3`                        |
| Training epochs      | `89`                         |
| Batch size           | `16`                         |
| Optimizer            | Adam                         |
| Loss                 | Binary Cross-Entropy         |
| Augmentation         | Rotation `0.05`, Zoom `0.05` |
| Class weighting      | Balanced                     |
| Decision threshold   | `0.586`                      |
| Development set      | 197 images                   |
| Independent Test set | 37 images                    |

---

# 15. Experiment Decision Pipeline

```text
ROI Pixel Value Inspection
        ↓
234/234 images passed
        ↓
Preprocessing Check
        ↓
234/234 images passed
        ↓
Run 01
Baseline CNN
        ↓
Run 02
Learning rate = 0.001
        ↓
Run 03
Dropout = 0.3
        ↓
Run 04
Fixed epoch = 89
        ↓
Run 05
OOF predictions
        ↓
Run 06
Threshold = 0.586
        ↓
Run 07
Final training on all 197 Development images
        ↓
Run 08
Final evaluation on untouched 37-image Test set
```

---

# 16. Final Test Performance

The final EXP01 model achieved:

| Metric            | Test Result |
| ----------------- | ----------: |
| ROC-AUC           |  **0.8686** |
| Accuracy          |  **67.57%** |
| Balanced Accuracy |  **73.24%** |
| Sensitivity       |  **92.31%** |
| Specificity       |  **54.17%** |
| Precision         |  **52.17%** |
| F1-score          |  **66.67%** |
| MCC               |  **0.4575** |

At the predetermined threshold of `0.586`:

|                    | Predicted Healthy | Predicted Sick |
| ------------------ | ----------------: | -------------: |
| **Actual Healthy** |       **TN = 13** |    **FP = 11** |
| **Actual Sick**    |        **FN = 1** |    **TP = 12** |

---

# 17. Key Takeaways

* EXP01 follows a sequential model-development pipeline consisting of dataset inspection, preprocessing verification, architecture establishment, learning-rate selection, dropout selection, epoch selection, OOF prediction generation, threshold selection, final training, and independent test evaluation.

* The ROI pixel-value inspection found **no abnormal pixel values** in any of the 234 DMR-IR ROI images using the threshold of 50.

* The standalone preprocessing check confirmed that all **234 images** could be loaded, processed, normalized, resized, and converted to the required `128 × 128 × 1` `float32` format without failures.

* The 234-image preprocessing check was an implementation verification step only. It was not used to determine normalization parameters for model development.

* Run 01 used the original Training and Validation sets separately. Temperature normalization parameters and class weights were calculated only from the 163 original Training images, while the 34 original Validation images were held out for validation.

* From Run 02 onward, the original Training and Validation sets were combined into a **197-image Development set**, consisting of 125 Healthy and 72 Sick images.

* Runs 02–05 used stratified 5-fold cross-validation on the 197-image Development set for model-development decisions.

* During cross-validation, temperature normalization parameters and class weights were calculated from the training portion of each fold and applied to the corresponding held-out fold.

* The original Validation images became part of the Development set from Run 02 onward and were therefore no longer an independent validation set.

* The final model configuration was determined entirely from Development data:

  * Learning rate = `0.001`
  * Dropout = `0.3`
  * Training epochs = `89`
  * Decision threshold = `0.586`

* Run 07 trained the final model using all 197 Development images. Temperature normalization parameters and class weights were calculated using these 197 images.

* Run 07 did not hold back a validation subset, use early stopping, or perform checkpoint-based model selection. The model state after exactly 89 epochs was used as the final model.

* The independent Test set remained untouched throughout Runs 01–07. It did not contribute to normalization, class-weight calculation, hyperparameter selection, epoch selection, threshold selection, or final model training.

* Run 08 was the first stage that accessed the 37 Test images for model evaluation.

* Run 08 used the normalization parameters saved during Run 07 rather than recalculating the temperature range from the Test set.

* The final Test ROC-AUC was **0.8686**.

* At the predetermined threshold of `0.586`, sensitivity was **92.31%** and specificity was **54.17%**.

* The final Test confusion matrix was **TN = 13, FP = 11, FN = 1, TP = 12**.

* Overall, EXP01 maintains a clear separation between **model development using the 197-image Development set** and **final independent evaluation using the 37-image Test set**.