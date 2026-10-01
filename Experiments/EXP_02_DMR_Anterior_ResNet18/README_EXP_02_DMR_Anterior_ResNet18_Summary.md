# EXP02_DMR_Anterior_ResNet18

## Experiment Overview

EXP02 evaluates a frozen-backbone, ImageNet-pretrained ResNet-18 for binary classification of anterior-view breast thermal images into **Healthy** and **Sick** classes.

Only the newly initialized classification head is trained. The entire ResNet-18 backbone remains frozen.

The experiment is structured as a sequential model-development pipeline:

```text
Run 01
Baseline Transfer Learning
        ↓
Run 02
5-Fold CV Training Dynamics
        ↓
Run 03
Fixed Epoch Selection
        ↓
Run 04
5-Fold OOF Predictions
        ↓
Run 05
OOF Threshold Selection
        ↓
Run 06
Final Model Training
        ↓
Run 07
Final Test Evaluation
```

The Test set remains completely untouched until the final evaluation in Run 07.

---

## 1. Objective

The primary objective of EXP02 is to evaluate a transfer-learning baseline in which an ImageNet-pretrained ResNet-18 with a frozen backbone distinguishes Healthy from Sick anterior-view breast thermography images.

Specifically, EXP02 investigates:

* A transfer-learning baseline on the predefined Train and Validation split.
* Training dynamics across stratified 5-fold cross-validation.
* Selection of a common fixed training epoch.
* Generation of out-of-fold predictions.
* Selection of a classification threshold using only development data.
* Final training using all available development images.
* One-time evaluation on the independent Test set.

Benign cases were excluded from this experiment.

The experiment is designed so that training duration and classification threshold are established before the Test set is accessed.

---

## 2. Dataset

Dataset: `DMR_IR_Anterior_View_ROI`

### 2.1 Dataset Organization

The original dataset is divided into:

```text
Original Train       = 163 images
Original Validation  = 34 images
Original Test        = 37 images
```

The original Train and Validation sets are combined to form the development dataset for cross-validation and final model training.

```text
Development = Original Train + Original Validation
            = 163 + 34
            = 197 images
```

The Test set remains completely separate.

### 2.2 Dataset Composition

| Split | Images | Healthy | Sick |
|---|---|---|---|
| Train | 163 | 103 | 60 |
| Validation | 34 | 22 | 12 |
| Development | 197 | 125 | 72 |
| Test | 37 | 24 | 13 |
| Total | 234 | 149 | 85 |

### 2.3 Independent Test Dataset

| Class | Label | Images |
|---|---|---|
| Healthy | 0 | 24 |
| Sick | 1 | 13 |
| Total | | 37 |

The Test set was not used during:

* Preprocessing verification
* Model training
* Cross-validation
* Model configuration selection
* Training epoch selection
* Threshold selection
* Temperature normalization range estimation

It was accessed only during Run 07.

---

## 3. Experiment Configuration

### 3.1 Model

EXP02 uses an ImageNet-pretrained ResNet-18.

```text
Architecture:
ResNet-18

Pretrained weights:
IMAGENET1K_V1

Input:
224 × 224 × 3

Backbone:
Frozen (conv1, bn1, layer1, layer2, layer3, layer4)

Trainable:
classifier only
```

The original ResNet-18 classification layer is replaced with:

```text
Linear(512, 64)
ReLU
Dropout(0.3)
Linear(64, 1)
```

The model contains:

```text
Total parameters       = 11,209,409
Trainable parameters   = 32,897
```

The frozen backbone is kept in evaluation mode during training to prevent the BatchNorm layers from updating their running statistics. The classification head remains in training mode.

### 3.2 Optimization

```text
Optimizer:
Adam

Learning rate:
1e-3

Batch size:
16

Dropout:
0.3

Loss:
Weighted BCEWithLogitsLoss

Class weighting:
Balanced

Seed:
10
```

### 3.3 Reproducibility

```text
Seed = 10
Seed reset at the beginning of every fold

Training device = CUDA
GPU = NVIDIA Quadro M1200
Final Test inference = CPU
```

---

## 4. Thermal Image Preprocessing

The same general preprocessing pipeline is maintained throughout the experiment.

### 4.1 Temperature Normalization

Valid temperature values are identified using `temperature > 0`.

Valid temperatures are normalized using a temperature range derived from training data only:

```text
normalized = (temperature - global_min) / (global_max - global_min)
```

Normalized values are clipped to the range 0 to 1. Invalid/background pixels (`<= 0`) remain zero.

For Run 01:

```text
Temperature range = calculated from the 163 training images
Minimum = 22.000000
Maximum = 36.779999
```

For cross-validation (Runs 02–04):

```text
Temperature range = calculated from fold-training data
```

The corresponding fold validation data is transformed using the same range.

For final training (Run 06):

```text
Temperature range = calculated from all 197 development images
Minimum = 22.000000
Maximum = 36.812618
```

The same development range is used for the Test set in Run 07.

No Test-set statistics are used to determine the normalization range.

#### Preprocessing Verification

All 234 images passed the preprocessing verification.

The overall verification range across all images was:

```text
Minimum = 22.000000
Maximum = 36.812618255615234
```

The overall range was used only for verification. Model development and final evaluation used the appropriate training/Development-derived temperature ranges.

### 4.2 Processing Pipeline

```text
Raw thermal image (.tif, 2D floating-point temperature array)
        ↓
Temperature normalization (valid pixels only)
        ↓
Clip to [0, 1]
        ↓
Centered zero-padding to square
        ↓
Convert to 8-bit
        ↓
Resize to 224 × 224 (bilinear interpolation)
        ↓
Convert to tensor (224 × 224 × 1)
        ↓
Repeat grayscale channel to 3 channels
        ↓
ImageNet normalization
        ↓
ResNet-18
```

ImageNet normalization uses:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

No separate TIFF handling pipeline was required.

### 4.3 Data Augmentation

Training images use:

```text
Random rotation ±18°
Random affine scale 0.95–1.05
```

No horizontal flipping is used.

Validation and Test images receive no random augmentation.

---

## 5. Run 01 — Baseline Transfer Learning

### Objective

Run 01 establishes a transfer-learning baseline using the predefined Train and Validation sets.

The Test set is not loaded or used.

### Dataset

```text
Training:
163 images
Healthy = 103
Sick = 60

Validation:
34 images
Healthy = 22
Sick = 12

Test:
Not used
```

### Training

```text
Maximum epochs = 100
Batch size = 16

Learning rate = 1e-3
Dropout = 0.3

Optimizer = Adam
Loss = Weighted BCEWithLogitsLoss
Seed = 10

Early stopping = Not used
Selection metric = Validation ROC-AUC
Threshold = 0.5
```

The temperature range was calculated only from the 163 training images and then applied to both the training and validation sets.

Balanced class weights were calculated from the training set only:

```text
Healthy (class 0) = 0.7913
Sick    (class 1) = 1.3583
```

### Model Selection

The checkpoint with the highest validation ROC-AUC was selected.

```text
Best epoch = 1
Best validation ROC-AUC = 0.9053
```

Although later epochs showed higher training performance, none exceeded the validation ROC-AUC obtained at epoch 1.

### Validation Results

The selected model was evaluated on the 34-image validation set using a fixed threshold of 0.5.

| Metric | Result |
|---|---|
| ROC-AUC | 0.9053 |
| PR-AUC | 0.7556 |
| Accuracy | 0.6471 |
| Balanced Accuracy | 0.5000 |
| Sensitivity | 0.0000 |
| Specificity | 1.0000 |
| Precision | 0.0000 |
| NPV | 0.6471 |
| F1-score | 0.0000 |
| MCC | 0.0000 |

Confusion matrix:

```text
                 Predicted
              Healthy   Sick

Actual Healthy    22       0
       Sick       12       0
```

Therefore:

```text
TN = 22
FP = 0
FN = 12
TP = 0
```

At the fixed threshold of 0.5, all 34 validation images were classified as Healthy. The ROC-AUC of 0.9053 reflects the ranking performance of the continuous model outputs across thresholds and is therefore not directly equivalent to the classification performance at the fixed 0.5 threshold.

### Role in EXP02

Run 01 establishes the frozen-backbone transfer-learning baseline.

The test set was preserved for later final evaluation.

---

## 6. Run 02 — 5-Fold CV Training Dynamics

### Objective

Run 02 evaluates the training dynamics of the frozen-backbone ResNet-18 across the complete 197-image development dataset using stratified 5-fold cross-validation.

Each fold is trained for 100 epochs to examine validation ROC-AUC across training time.

The Test set remains untouched.

### Methodology

For each fold:

1. Reset the seed to 10.
2. Calculate the fold-specific temperature range from fold-training data only.
3. Preprocess train/validation images.
4. Calculate class weights using fold-training data only.
5. Initialize ImageNet-pretrained ResNet-18.
6. Freeze the ResNet-18 backbone.
7. Train the classification head for 100 epochs.
8. Save the best validation-AUC checkpoint.
9. Identify the fold-specific best epoch.
10. Calculate validation metrics.

Five-fold aggregate training dynamics are then calculated by aggregating the validation ROC-AUC from all five folds epoch by epoch.

### Configuration

```text
Development images = 197
Healthy = 125
Sick    = 72

Folds = 5
Epochs per fold = 100

Batch size = 16

Learning rate = 1e-3
Dropout = 0.3

Optimizer = Adam
Threshold = 0.5

Selection metric = Mean epoch-wise validation ROC-AUC
```

### Fold-Level Results

| Fold | Train Samples | Validation Samples | Best Epoch | Epoch 1 AUC | Best Validation AUC | Epoch 100 AUC | ROC-AUC | PR-AUC | Accuracy | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 157 | 40 | 68 | 0.8800 | 0.9413 | 0.9280 | 0.9413 | 0.9527 | 0.9000 | 0.8000 | 0.9600 | 0.8571 |
| 2 | 157 | 40 | 89 | 0.8373 | 0.9253 | 0.9227 | 0.9253 | 0.8467 | 0.8250 | 0.8000 | 0.8400 | 0.7742 |
| 3 | 158 | 39 | 67 | 0.9314 | 0.9886 | 0.9800 | 0.9886 | 0.9805 | 0.9231 | 1.0000 | 0.8800 | 0.9032 |
| 4 | 158 | 39 | 32 | 0.8829 | 0.9114 | 0.8971 | 0.9114 | 0.8408 | 0.8462 | 0.7143 | 0.9200 | 0.7692 |
| 5 | 158 | 39 | 23 | 0.8114 | 0.8800 | 0.8086 | 0.8800 | 0.8518 | 0.8205 | 0.7143 | 0.8800 | 0.7407 |

### Common Training Epoch

The validation ROC-AUC from all five folds was aggregated separately for each epoch.

```text
Best epoch-wise mean validation ROC-AUC epoch = 32
Best mean validation ROC-AUC = 0.9178
```

The individual folds reached their own highest validation ROC-AUC at different epochs:

```text
Fold 1 = epoch 68
Fold 2 = epoch 89
Fold 3 = epoch 67
Fold 4 = epoch 32
Fold 5 = epoch 23
```

Epoch 32 therefore represents the common epoch with the highest mean validation ROC-AUC across the five folds. It is not the individual best epoch for every fold.

The five fold training histories were not retrained or modified during the aggregate analysis.

Run 02 does not itself perform fixed-epoch model selection.

### Role in EXP02

Run 02 establishes the training-duration candidates used in Run 03.

The candidate epochs are derived from:

```text
Fold-specific best epochs:
23, 32, 67, 68, 89

Aggregate epoch-wise best:
32

Full training duration:
100
```

Therefore:

```text
Candidate epochs = [23, 32, 67, 68, 89, 100]
```

---

## 7. Run 03 — Fixed Epoch Selection

### Objective

Run 03 formally selects a single common training epoch using stratified 5-fold cross-validation.

The same five stratified folds are evaluated for every candidate epoch. A new model is trained for every fold and candidate epoch.

The Test set remains untouched.

### Candidate Epochs

* 23
* 32
* 67
* 68
* 89
* 100

The candidates were derived from the Run 02 training dynamics. The full 100-epoch duration was retained as a reference candidate, and no arbitrary intermediate epochs were introduced.

### Methodology

For each candidate epoch:

1. Use the same five stratified folds.
2. Reset the seed to 10 for each fold.
3. Calculate fold-specific temperature ranges from training data only.
4. Calculate fold-specific class weights from training data only.
5. Initialize a new ImageNet-pretrained ResNet-18.
6. Freeze the ResNet-18 backbone.
7. Train the classification head for exactly the candidate number of epochs.
8. Use no early stopping and no best-epoch checkpoint.
9. Evaluate the held-out fold after the fixed epoch.
10. Calculate mean and SD across the five folds.

A total of:

```text
6 candidate epochs × 5 folds = 30 trainings
```

are performed.

### Candidate Comparison

| Epoch | Mean ROC-AUC | SD |
|---|---|---|
| 23 | 0.906667 | 0.049784 |
| 32 | 0.912229 | 0.045715 |
| 67 | 0.909524 | 0.054601 |
| 68 | 0.908419 | 0.055412 |
| 89 | 0.910210 | 0.056134 |
| 100 | 0.904990 | 0.060510 |

#### Additional Validation Metrics

| Epoch | Accuracy | Balanced Accuracy | Sensitivity | Specificity | Precision | F1 | MCC |
|---|---|---|---|---|---|---|---|
| 23 | 0.842564 | 0.812857 | 0.705714 | 0.920000 | 0.866667 | 0.755876 | 0.667297 |
| 32 | 0.852821 | 0.829810 | 0.747619 | 0.912000 | 0.860094 | 0.775036 | 0.692022 |
| 67 | 0.862949 | 0.847238 | 0.790476 | 0.904000 | 0.835233 | 0.803956 | 0.708051 |
| 68 | 0.857821 | 0.834762 | 0.749524 | 0.920000 | 0.856120 | 0.787845 | 0.696784 |
| 89 | 0.868077 | 0.848571 | 0.777143 | 0.920000 | 0.853252 | 0.806934 | 0.715505 |
| 100 | 0.847564 | 0.820952 | 0.721905 | 0.920000 | 0.879048 | 0.760977 | 0.684045 |

These additional metrics are reported for comparison but were not used to select the fixed epoch.

### Selected Epoch

The fixed training epoch was selected using the highest mean 5-fold validation ROC-AUC.

```text
Selected epoch = 32

Mean validation ROC-AUC = 0.912229
SD = 0.045715
```

The Test set was not used in this selection.

### Methodological Note

Run 02 was used to identify candidate fixed training epochs, and the same folds were reused in Run 03 to compare those candidates. Run 03 should therefore be interpreted as an internal Development-stage fixed training epoch selection procedure rather than an independent unbiased estimate of generalization performance.

### Role in EXP02

Run 03 establishes the fixed training duration used by all subsequent final-model development steps:

```text
Fixed training epoch = 32
```

---

## 8. Run 04 — 5-Fold Cross-Validation OOF Predictions

### Objective

Run 04 generates a complete set of out-of-fold predictions for the 197-image development dataset using the fixed 32-epoch configuration established in Run 03.

The purpose is to obtain development-set probabilities for threshold selection.

### Methodology

Five-fold stratified cross-validation is performed.

For each fold:

```text
Train on 4 folds
        ↓
Train exactly 32 epochs
        ↓
Predict the held-out fold
        ↓
Store raw Sick probabilities
```

Within each fold, the temperature range and class weights are calculated from fold-training data only, and the seed is reset to 10.

No threshold is applied during prediction generation.

No early stopping and no best-epoch checkpoint are used.

The final model state at epoch 32 is used for held-out predictions.

### OOF Integrity

```text
Total development images = 197
OOF predictions           = 197
Missing predictions       = 0
Missing fold assignments  = 0
```

Every development image therefore receives exactly one out-of-fold prediction.

### OOF Performance

```text
ROC-AUC = 0.892667
PR-AUC  = 0.832326
```

No classification threshold was applied when calculating these metrics.

### Fold Results

| Fold | n_train | n_validation | Train Healthy | Train Sick | Validation Healthy | Validation Sick | Healthy Class Weight | Sick Class Weight | Temperature Min | Temperature Max | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 157 | 40 | 100 | 57 | 25 | 15 | 0.785 | 1.377193 | 22.00000 | 36.81262 | 0.930667 | 0.949768 |
| 2 | 157 | 40 | 100 | 57 | 25 | 15 | 0.785 | 1.377193 | 22.00000 | 36.78000 | 0.893333 | 0.805030 |
| 3 | 158 | 39 | 100 | 58 | 25 | 14 | 0.790 | 1.362069 | 22.22035 | 36.81262 | 0.988571 | 0.981548 |
| 4 | 158 | 39 | 100 | 58 | 25 | 14 | 0.790 | 1.362069 | 22.00000 | 36.81262 | 0.897143 | 0.836683 |
| 5 | 158 | 39 | 100 | 58 | 25 | 14 | 0.790 | 1.362069 | 22.00000 | 36.81262 | 0.851429 | 0.822636 |

All folds used 32 epochs, learning rate 0.001, dropout 0.3 and 32,897 trainable parameters.

### Role in EXP02

Run 04 provides the complete OOF probability distribution required for threshold selection.

The Test set remains untouched.

---

## 9. Run 05 — OOF Threshold Selection

### Objective

Run 05 selects the final classification threshold using only the OOF predictions generated in Run 04.

No model retraining occurs in this run.

### Threshold Selection Criteria

Thresholds are evaluated using every unique OOF probability.

A threshold is considered valid only when:

```text
Sensitivity > 0.91
AND
Specificity > 0.60
```

Among valid thresholds, selection follows:

```text
1. Highest sensitivity
2. Highest specificity
3. Highest threshold
```

The implemented selection rule was:

```python
selected_result = max(
    valid_results,
    key=lambda x: (
        x["sensitivity"],
        x["specificity"],
        x["threshold"],
    ),
)
```

This is a constraint-based threshold selection procedure rather than selection based on maximum accuracy, F1-score, ROC-AUC or PR-AUC.

### Selected Threshold

```text
Selected threshold = 0.192174
```

Five thresholds satisfied the required sensitivity and specificity constraints.

### OOF Performance at Selected Threshold

| Metric | Value |
|---|---|
| Threshold | 0.192174 |
| Sensitivity | 0.9306 |
| Specificity | 0.6320 |
| Accuracy | 0.7411 |
| Balanced accuracy | 0.7813 |
| Precision | 0.5929 |
| F1-score | 0.7243 |
| TN | 79 |
| FP | 46 |
| FN | 5 |
| TP | 67 |

Confusion matrix:

```text
                 Predicted
              Healthy    Sick

Actual Healthy    79       46

Actual Sick        5       67
```

The selected threshold satisfies:

```text
Sensitivity = 0.9306 > 0.91
Specificity = 0.6320 > 0.60
```

The corresponding ranking metrics remain:

```text
ROC-AUC = 0.892667
PR-AUC  = 0.832326
```

### Role in EXP02

Run 05 fixes the classification threshold before final training and Test evaluation.

```text
Final threshold = 0.192174
```

The Test set is not used during threshold selection.

---

## 10. Run 06 — Final Model Training

### Objective

Run 06 trains the final EXP02 model using all available development data after the model configuration, training duration and classification threshold have been established.

### Training Dataset

```text
Development images = 197

Healthy = 125
Sick    = 72
```

The original Train (163) and Validation (34) sets are combined.

The Test set is not loaded or accessed.

### Final Preprocessing

Development-only temperature range:

```text
Minimum = 22.000000
Maximum = 36.812618
```

Final development class weights:

```text
Healthy = 0.788000
Sick    = 1.368056
```

### Model

```text
Architecture = ResNet-18
Weights = ImageNet1K_V1

Frozen:
Complete backbone

Trainable:
classifier (512 → 64 → 1)
```

Parameters:

```text
Total      = 11,209,409
Trainable  = 32,897
```

### Training

```text
Epochs = 32
Batch size = 16
Optimizer = Adam

Learning rate = 1e-3
Dropout = 0.3

Loss = Weighted BCEWithLogitsLoss
Class weighting = Balanced
Seed = 10

Device = CUDA
GPU = NVIDIA Quadro M1200
```

No validation data, early stopping, checkpoint selection or best-epoch calculation was used because all development images are used for final training.

### Training Loss

| Epoch | Training Loss |
|---|---|
| 1 | 0.851118 |
| 2 | 0.800015 |
| 3 | 0.723242 |
| 4 | 0.715102 |
| 5 | 0.665308 |
| 6 | 0.591522 |
| 7 | 0.615454 |
| 8 | 0.576032 |
| 9 | 0.594585 |
| 10 | 0.542768 |
| 11 | 0.535997 |
| 12 | 0.595188 |
| 13 | 0.528103 |
| 14 | 0.474266 |
| 15 | 0.516859 |
| 16 | 0.489175 |
| 17 | 0.461668 |
| 18 | 0.481583 |
| 19 | 0.499728 |
| 20 | 0.524360 |
| 21 | 0.503394 |
| 22 | 0.497393 |
| 23 | 0.462457 |
| 24 | 0.470047 |
| 25 | 0.468058 |
| 26 | 0.460774 |
| 27 | 0.423109 |
| 28 | 0.466241 |
| 29 | 0.414987 |
| 30 | 0.424674 |
| 31 | 0.443622 |
| 32 | 0.491160 |

The final training loss at epoch 32 was 0.491160.

### Final Model

The final model is the model state after exactly 32 epochs.

```text
Run_06_Final_Training/model/final_model_epoch32.pth
```

The threshold from Run 05 is carried forward:

```text
Threshold = 0.192174
```

The threshold was not used during training and did not affect the optimization process.

### Test Set Isolation

```text
Test set loaded during Run 06: NO
Test set evaluated during Run 06: NO
Test images used for training: NO
```

### Role in EXP02

Run 06 produces the final model used for independent Test evaluation.

---

## 11. Run 07 — Final Test Evaluation

### Objective

Run 07 performs the final one-time evaluation of the complete EXP02 pipeline on the independent Test set.

The final model from Run 06 and the fixed threshold from Run 05 are used without modification. Run 07 performs inference only.

### Test Dataset

```text
Healthy = 24
Sick    = 13
Total   = 37
```

A Test-set integrity check confirmed that exactly 37 images were loaded with the expected class distribution.

### Test Preprocessing

The Run 06 development-derived temperature range is loaded and used without recalculation:

```text
Minimum = 22.000000
Maximum = 36.812618
```

Processing:

```text
Raw thermal image
        ↓
Temperature normalization (development-set range)
        ↓
Pad to square
        ↓
Convert to 8-bit image
        ↓
Resize to 224 × 224
        ↓
Convert to tensor
        ↓
Repeat grayscale channel to 3 channels
        ↓
ImageNet normalization
        ↓
ResNet-18
```

No augmentation is applied.

### Classification Threshold

The threshold established in Run 05 is used without modification:

```text
Threshold = 0.192174
```

Classification:

```text
Probability >= 0.192174 → Sick
Probability <  0.192174 → Healthy
```

For each Test image, the model generates a sigmoid probability of the Sick class. ROC-AUC is calculated directly from these probabilities, before the threshold is applied.

No threshold optimization or model selection is performed using Test predictions.

### Evaluation Configuration

| Parameter | Value |
|---|---|
| Final model source | Run 06 |
| Model | final_model_epoch32.pth |
| Training epochs | 32 |
| Learning rate | 1e-3 |
| Dropout | 0.3 |
| Threshold source | Run 05 OOF Threshold Selection |
| Temperature range source | Run 06 Final Training |
| Device | CPU |
| Random seed | 10 |

### Final Test Performance

| Metric | Test Result |
|---|---|
| ROC-AUC | 0.9455 |
| Accuracy | 0.8378 |
| Balanced Accuracy | 0.8750 |
| Sensitivity | 1.0000 |
| Specificity | 0.7500 |
| Precision | 0.6842 |
| F1-score | 0.8125 |
| MCC | 0.7164 |
| Classification Threshold | 0.192174 |

### Confusion Matrix

```text
                 Predicted
              Healthy    Sick

Actual Healthy    18        6

Actual Sick        0       13
```

Therefore:

```text
TN = 18
FP = 6
FN = 0
TP = 13
```

The metrics are internally consistent:

```text
Sensitivity = 13 / 13 = 1.0000

Specificity = 18 / 24 = 0.7500

Accuracy = (18 + 13) / 37 = 31 / 37 = 0.8378

Balanced Accuracy = (1.0000 + 0.7500) / 2 = 0.8750
```

### ROC-AUC Interpretation

The final Test ROC-AUC is:

```text
0.9455
```

ROC-AUC evaluates the ranking of model predictions across thresholds.

Accuracy, sensitivity, specificity, precision, F1-score and MCC are threshold-dependent and were calculated using the pre-specified threshold:

```text
0.192174
```

Therefore, ROC-AUC should be interpreted separately from the threshold-dependent classification metrics.

---

## 12. EXP02 Experimental Progression

The complete EXP02 methodology can be summarized as:

```text
Frozen-Backbone Transfer Learning Baseline
        │
        │ Run 01
        ↓
Predefined Train/Validation baseline (best epoch 1)
        │
        │ Run 02
        ↓
5-Fold CV training dynamics
        │
        ↓
Identify representative candidate epochs
        │
        │ Run 03
        ↓
Formal fixed-epoch comparison
        │
        ↓
Select epoch 32
        │
        │ Run 04
        ↓
Generate 197 OOF predictions
        │
        │ Run 05
        ↓
Select threshold 0.192174
        │
        │ Run 06
        ↓
Train final model on all 197 development images
        │
        │ Run 07
        ↓
Evaluate once on independent 37-image Test set
```

This progression separates:

```text
Baseline transfer learning
        ↓
Training-dynamics analysis
        ↓
Training-duration selection
        ↓
Prediction generation
        ↓
Threshold selection
        ↓
Final training
        ↓
Final evaluation
```

The Test set is isolated from all development decisions.

---

## 13. Final EXP02 Results

### 13.1 Final Configuration

| Component | Final Setting |
|---|---|
| Experiment | EXP02_DMR_Anterior_ResNet18 |
| Dataset | DMR_IR_Anterior_View_ROI |
| Classes | Healthy = 0, Sick = 1 |
| Total images | 234 |
| Development set | 197 |
| Independent Test set | 37 |
| Model | ImageNet-pretrained ResNet-18 |
| Weights | IMAGENET1K_V1 |
| Backbone | Frozen |
| Classifier | Linear(512,64) → ReLU → Dropout(0.3) → Linear(64,1) |
| Total parameters | 11,209,409 |
| Trainable parameters | 32,897 |
| Input | 224 × 224 × 3 |
| Original thermal channels | 1 |
| Thermal channel handling | Repeated to 3 channels |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 16 |
| Loss | Weighted BCEWithLogitsLoss |
| Class weighting | Balanced |
| Random seed | 10 |
| Augmentation | Rotation ±18°, affine scale 0.95–1.05 |
| Horizontal flip | No |
| Selected fixed training epoch | 32 |
| Final training | All 197 Development images |
| Final model | final_model_epoch32.pth |
| Classification threshold | 0.192174 |
| Test inference | CPU |

### 13.2 Development-Stage Results

| Stage | Result |
|---|---|
| Run 01 best validation ROC-AUC (epoch 1) | 0.9053 |
| Run 02 best mean epoch-wise validation ROC-AUC (epoch 32) | 0.9178 |
| Run 03 selected epoch 32, mean CV ROC-AUC | 0.912229 ± 0.045715 |
| Run 04 overall OOF ROC-AUC | 0.892667 |
| Run 04 overall OOF PR-AUC | 0.832326 |
| Run 05 selected threshold | 0.192174 |
| Run 05 OOF sensitivity / specificity | 0.9306 / 0.6320 |

### 13.3 Final Test Performance

The final EXP02 model achieved the following performance on the held-out Test set:

```text
ROC-AUC           = 0.9455
Accuracy          = 83.78%
Balanced Accuracy = 87.50%
Sensitivity       = 100.00%
Specificity       = 75.00%
Precision         = 68.42%
F1-score          = 81.25%
MCC               = 71.64%
```

Confusion matrix:

```text
TN = 18
FP = 6
FN = 0
TP = 13
```

All 13 Sick Test images were correctly classified at the selected threshold, while 6 of the 24 Healthy Test images were classified as Sick.

The final Test ROC-AUC of 0.9455 reflects the ranking performance of the final model on the independent Test set.

The threshold-dependent metrics were calculated using the classification threshold of 0.192174 that had been fixed using development-set OOF predictions before Test evaluation.

---

## 14. Test-Set Isolation

A central methodological property of EXP02 is the strict separation between the Development set and the held-out Test set.

The Test set was not used for:

* Preprocessing verification
* Training
* Validation
* Cross-validation
* Training epoch selection
* Threshold selection
* Temperature normalization range estimation
* Model selection
* Early stopping

The Test set was accessed only in Run 07 after:

```text
Model configuration was fixed
        ↓
Training epoch was fixed
        ↓
OOF predictions were generated
        ↓
Classification threshold was fixed
        ↓
Final model was trained
        ↓
Final Test evaluation was performed
```

No model retraining, model selection or threshold adjustment was performed using the Test results.

---

## 15. Output Structure

The complete EXP02 directory contains the individual run outputs.

```text
EXP02_DMR_Anterior_ResNet18/

├── Run_01_Baseline_Transfer_Learning/
│
├── Run_02_5-fold_CV_Training_Dynamics/
│
├── Run_03_Fixed_Epoch_Selection/
│
├── Run_04_5-fold_CV_OOF_Predictions/
│
├── Run_05_OOF_Threshold_Selection/
│
├── Run_06_Final_Training/
│
└── Run_07_Final_Test_Evaluation/
```

### Run 01

```text
Run_01_Baseline_Transfer_Learning/

├── README.md
├── results/
│   ├── validation_metrics.json
│   ├── validation_roc_data.json
│   ├── training_history.json
│   ├── training_summary.json
│   └── model_summary.txt
├── checkpoints/
│   └── resnet18_frozen_best.pth
└── plots/
    ├── training_validation_loss.png
    ├── training_validation_accuracy.png
    ├── training_validation_auc.png
    └── validation_roc_curve.png
```

### Run 02

```text
Run_02_5-fold_CV_Training_Dynamics/

├── README.md
│
├── results/
│   ├── fold_temperature_ranges.json
│   ├── fold_summary.csv
│   ├── epoch_wise_cv_summary.csv
│   ├── cv_summary.json
│   ├── training_summary.json
│   └── model_summary.txt
│
├── plots/
│   ├── cv_validation_auc_mean_sd.png
│   └── cv_loss_mean_sd.png
│
├── fold_1/
│   ├── results/
│   │   ├── training_history.json
│   │   ├── validation_metrics.json
│   │   ├── validation_roc_data.json
│   │   ├── fold_summary.json
│   │   └── validation_predictions.csv
│   ├── checkpoints/
│   │   └── resnet18_frozen_best.pth
│   └── plots/
│       ├── training_validation_loss.png
│       ├── training_validation_accuracy.png
│       ├── training_validation_auc.png
│       └── validation_roc_curve.png
│
├── fold_2/
├── fold_3/
├── fold_4/
└── fold_5/

```

Each fold contains its validation metrics, training history, ROC data, checkpoint and associated plots.

### Run 03

```text
Run_03_Fixed_Epoch_Selection/

├── README.md
└── results/
    ├── candidate_epochs.json
    ├── candidate_epoch_comparison.json
    ├── candidate_epoch_comparison.csv
    ├── run_summary.json
    ├── epoch_23/
    ├── epoch_32/
    ├── epoch_67/
    ├── epoch_68/
    ├── epoch_89/
    └── epoch_100/
```

Each candidate epoch folder contains fold_1 to fold_5. Each fold contains the trained model, validation predictions and fold-level results, including the training configuration, temperature range and class weights.

### Run 04

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
    ├── fold_5_temperature_range.json
    ├── fold_1_training_history.json
    ├── fold_2_training_history.json
    ├── fold_3_training_history.json
    ├── fold_4_training_history.json
    ├── fold_5_training_history.json
    ├── fold_1_summary.json
    ├── fold_2_summary.json
    ├── fold_3_summary.json
    ├── fold_4_summary.json
    ├── fold_5_summary.json
    ├── oof_predictions.csv
    ├── fold_summary.csv
    └── run_summary.json
```

### Run 05

```text
Run_05_OOF_Threshold_Selection/

├── README.md
├── results/
│   ├── threshold_analysis.csv
│   ├── selected_threshold.json
│   └── overall_summary.json
└── plots/
    ├── sensitivity_specificity_vs_threshold.png
    └── valid_thresholds.png
```

### Run 06

```text
Run_06_Final_Training/

├── README.md
├── results/
│   ├── temperature_range.json
│   ├── training_history.json
│   ├── run_config.json
│   └── development_manifest.csv
├── model/
│   └── final_model_epoch32.pth
└── plots/
    └── training_loss.png
```

### Run 07

```text
Run_07_Final_Test_Evaluation/

├── README.md
├── results/
│   ├── test_metrics.json
│   ├── test_predictions.csv
│   └── run_config.json
└── plots/
    ├── test_roc_curve.png
    └── test_confusion_matrix.png
```

---

## 16. Final Experiment Summary

EXP02 evaluated a frozen-backbone, ImageNet-pretrained ResNet-18 for binary classification of anterior-view breast thermal images.

The experiment used a structured seven-run development pipeline.

Run 01 established the transfer-learning baseline using the predefined Train/Validation split. The best validation ROC-AUC of 0.9053 occurred at epoch 1, and at the fixed threshold of 0.5 all 34 validation images were classified as Healthy.

Run 02 expanded the analysis to stratified 5-fold cross-validation on all 197 development images and evaluated training dynamics over 100 epochs. The highest mean epoch-wise validation ROC-AUC (0.9178) occurred at epoch 32.

Run 03 formally compared six candidate training durations and selected a common fixed training epoch of 32 based on the highest mean 5-fold validation ROC-AUC:

```text
Mean ROC-AUC = 0.912229 ± 0.045715
```

Run 04 then generated one out-of-fold prediction for every development image using the fixed 32-epoch configuration:

```text
197 / 197 OOF predictions
OOF ROC-AUC = 0.892667
OOF PR-AUC  = 0.832326
```

Run 05 selected the final classification threshold from these OOF predictions:

```text
Threshold = 0.192174
```

Run 06 trained the final frozen-backbone ResNet-18 for exactly 32 epochs using all 197 development images.

Run 07 evaluated the resulting model once on the independent 37-image Test set.

The final Test ROC-AUC was:

```text
0.9455
```

At the pre-specified OOF-derived threshold of 0.192174:

```text
Accuracy          = 83.78%
Balanced Accuracy = 87.50%
Sensitivity       = 100.00%
Specificity       = 75.00%
Precision         = 68.42%
F1-score          = 81.25%
MCC               = 71.64%
```

The final confusion matrix was:

```text
TN = 18
FP = 6
FN = 0
TP = 13
```

The Test set was isolated from all model-development decisions and was used only for the final evaluation in Run 07.

Therefore, the complete EXP02 pipeline is:

```text
Frozen-Backbone Transfer Learning
        ↓
Cross-Validated Training Dynamics
        ↓
Fixed Epoch Selection
        ↓
OOF Prediction Generation
        ↓
OOF Threshold Selection
        ↓
Final Development Training
        ↓
Independent Test Evaluation
```

This completes EXP02.
