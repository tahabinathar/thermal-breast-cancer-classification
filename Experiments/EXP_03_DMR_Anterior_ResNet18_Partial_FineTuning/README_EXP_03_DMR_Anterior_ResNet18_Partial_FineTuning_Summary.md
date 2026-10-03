# EXP03_DMR_Anterior_ResNet18_Partial_FineTuning

## Experiment Overview

EXP03 evaluates a partially fine-tuned ImageNet-pretrained ResNet-18 for binary classification of anterior-view breast thermal images into **Healthy** and **Sick** classes.

The experiment extends the previous frozen-backbone approach by allowing the final ResNet-18 convolutional block (`layer4`) and the newly initialized classifier to be trained while keeping the earlier backbone layers frozen.

The experiment is structured as a sequential model-development pipeline:

```text
Run 01
Partial Fine-Tuning Baseline
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

The primary objective of EXP03 is to evaluate whether partial fine-tuning of an ImageNet-pretrained ResNet-18 can improve classification of anterior-view breast thermography images compared with a frozen-backbone configuration.

Specifically, EXP03 investigates:

* Training dynamics of partial fine-tuning.
* Cross-validated model behavior across the development dataset.
* Selection of a common fixed training epoch.
* Generation of out-of-fold predictions.
* Selection of a classification threshold using only development data.
* Final training using all available development images.
* One-time evaluation on the independent Test set.

The experiment is designed so that training duration and classification threshold are established before the Test set is accessed.

---

## 2. Dataset

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

### 2.2 Development Dataset

| Class | Label | Images |
|---|---|---|
| Healthy | 0 | 125 |
| Sick | 1 | 72 |
| Total | | 197 |

### 2.3 Original Train and Validation Composition

| Split | Healthy | Sick | Total |
|---|---|---|---|
| Original Train | 103 | 60 | 163 |
| Original Validation | 22 | 12 | 34 |
| Development | 125 | 72 | 197 |

### 2.4 Independent Test Dataset

| Class | Label | Images |
|---|---|---|
| Healthy | 0 | 24 |
| Sick | 1 | 13 |
| Total | | 37 |

The Test set was not used during:

* Model training
* Training epoch selection
* Threshold selection
* Model selection
* Hyperparameter tuning

It was accessed only during Run 07.

---

## 3. Experiment Configuration

### 3.1 Model

EXP03 uses an ImageNet-pretrained ResNet-18.

```text
Architecture:

ResNet-18

Pretrained weights:

ImageNet1K_V1

Input:

224 × 224 × 3

Frozen:

conv1

bn1

layer1

layer2

layer3

Trainable:

layer4

classifier
```

The original ResNet-18 classifier is replaced with:

```text
Linear(512, 64)

ReLU

Dropout(0.3)

Linear(64, 1)
```

The model contains:

```text
Total parameters       = 11,209,409

Trainable parameters   = 8,426,625

Frozen parameters      = 2,782,784
```

Approximately 75.17% of the model parameters are trainable.

### 3.2 Optimization

```text
Optimizer:

Adam

Layer4 learning rate:

1e-4

Classifier learning rate:

1e-3

Batch size:

16

Dropout:

0.3

Loss:

Weighted BCEWithLogitsLoss

Seed:

10
```

Balanced class weighting is calculated from the training portion only during cross-validation.

For final development training, the class weights are:

```text
Healthy weight = 0.788000

Sick weight    = 1.368056

pos_weight = 1.736111
```

### 3.3 Reproducibility

```text
Seed = 10

Python random seed

NumPy seed

PyTorch seed

CUDA seed when available

Deterministic algorithms enabled

NUM_WORKERS = 0
```

The seed is reset before each cross-validation fold/model training where applicable.

---

## 4. Thermal Image Preprocessing

The same general preprocessing pipeline is maintained throughout the experiment.

### 4.1 Temperature Normalization

Raw thermal values are normalized using a temperature range calculated from the training data only.

For cross-validation:

```text
Temperature range = calculated from fold-training data
```

The corresponding fold validation data is transformed using the same range.

For final training:

```text
Temperature range = calculated from all 197 development images
```

The final development temperature range used in Run 06 and Run 07 is:

```text
Minimum = 22.000000

Maximum = 36.812618
```

No Test-set statistics are used to determine the normalization range.

### 4.2 Processing Pipeline

```text
Raw thermal image

        ↓

Temperature normalization

        ↓

Valid-pixel masking

        ↓

Clip to [0, 1]

        ↓

Zero padding to square

        ↓

Convert to uint8

        ↓

Resize to 224 × 224

        ↓

Convert grayscale to 3 channels

        ↓

ImageNet normalization

        ↓

ResNet-18
```

The normalized thermal image is converted to 8-bit before resizing.

This preprocessing behavior is kept consistent throughout EXP03.

### 4.3 Data Augmentation

Training images use:

```text
RandomRotation(18°)

RandomAffine(scale = 0.95–1.05)
```

No horizontal flipping is used.

Validation and Test images receive no random augmentation.

---

## 5. Run 01 — Partial Fine-Tuning Baseline

### Objective

Run 01 establishes the initial partial fine-tuning baseline using the predefined Train and Validation sets.

Unlike the frozen-backbone configuration, the final ResNet-18 block (layer4) and classifier are trainable.

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

Layer4 LR = 1e-4

Classifier LR = 1e-3

Optimizer = Adam

Loss = Weighted BCEWithLogitsLoss

Seed = 10
```

The model is trained for the full 100 epochs to observe the complete training dynamics.

No early stopping is used.

The best validation-AUC checkpoint is saved for analysis.

### Training Dynamics

The best validation ROC-AUC occurred at:

```text
Best epoch = 96

Validation ROC-AUC = 0.9432
```

Training AUC approached 1.0 while validation performance fluctuated.

Examples:

| Epoch | Training AUC | Validation AUC |
|---|---|---|
| 24 | 1.0000 | 0.8788 |
| 49 | 1.0000 | 0.9129 |
| 74 | 1.0000 | 0.9242 |
| 86 | 0.9969 | 0.9091 |
| 96 | 0.9888 | 0.9432 |
| 100 | 1.0000 | 0.8977 |

This demonstrated strong training-data fitting with unstable validation behavior and motivated a more systematic cross-validation analysis of training duration.

### Validation Results at Epoch 96

Using threshold 0.5:

```text
ROC-AUC            = 0.9432

PR-AUC             = 0.9155

Accuracy           = 0.7353

Balanced Accuracy  = 0.7955

Sensitivity        = 1.0000

Specificity        = 0.5909

Precision          = 0.5714

NPV                = 1.0000

F1-score           = 0.7273

MCC                = 0.5811
```

Confusion matrix:

```text
TN = 13

FP = 9

FN = 0

TP = 12
```

### Role in EXP03

Run 01 establishes the partial fine-tuning baseline and demonstrates that training duration has a substantial effect on validation behavior.

Rather than selecting an epoch from a single predefined validation split, subsequent runs evaluate training dynamics using stratified 5-fold cross-validation.

---

## 6. Run 02 — 5-Fold CV Training Dynamics

### Objective

Run 02 evaluates partial fine-tuning across the complete 197-image development dataset using stratified 5-fold cross-validation.

Each fold is trained for 100 epochs to examine validation ROC-AUC across training time.

The Test set remains untouched.

### Methodology

For each fold:

1. Create a stratified training/validation split.
2. Calculate temperature normalization using fold-training data only.
3. Calculate class weights using fold-training data only.
4. Initialize ImageNet-pretrained ResNet-18.
5. Freeze conv1, bn1, layer1, layer2, and layer3.
6. Train layer4 and the classifier.
7. Train for 100 epochs.
8. Record training and validation loss and ROC-AUC at every epoch.
9. Save the best validation-AUC checkpoint.
10. Identify the fold-specific best epoch.

Five-fold aggregate training dynamics are then calculated using the validation results from all folds.

### Configuration

```text
Development images = 197

Folds = 5

Epochs per fold = 100

Batch size = 16

Layer4 LR = 1e-4

Classifier LR = 1e-3

Optimizer = Adam

Dropout = 0.3

Selection metric = Validation ROC-AUC

Diagnostic threshold = 0.5
```

### Fold-Specific Best Results

| Fold | Best Epoch | Best Validation ROC-AUC |
|---|---|---|
| 1 | 10 | 0.9413 |
| 2 | 12 | 0.9520 |
| 3 | 70 | 0.9914 |
| 4 | 26 | 0.9514 |
| 5 | 5 | 0.9600 |

Mean individual fold-best ROC-AUC:

```text
0.9592 ± 0.0185
```

Mean individual best epoch:

```text
24.60 ± 26.79
```

These are fold-specific optima and do not represent a single common training epoch.

### Epoch-Wise Aggregate Analysis

The highest mean epoch-wise validation ROC-AUC occurred at:

```text
Epoch = 26

Mean validation ROC-AUC = 0.9329

SD = 0.0200
```

At epoch 26, the fold validation AUCs were:

```text
Fold 1 = 0.9200

Fold 2 = 0.9360

Fold 3 = 0.9514

Fold 4 = 0.9514

Fold 5 = 0.9057
```

Run 02 therefore identifies several representative epochs for formal fixed-epoch comparison.

It does not itself perform fixed-epoch model selection.

### Role in EXP03

Run 02 establishes the training-duration candidates used in Run 03.

The candidate epochs are derived from:

```text
Fold-specific best epochs:

5, 10, 12, 26, 70

Aggregate epoch-wise best:

26

Full training duration:

100
```

Therefore:

```text
Candidate epochs = [5, 10, 12, 26, 70, 100]
```

---

## 7. Run 03 — Fixed Epoch Selection

### Objective

Run 03 formally selects a single common training epoch using 5-fold cross-validation.

The same five stratified folds are evaluated for every candidate epoch.

The Test set remains untouched.

### Candidate Epochs

* 5
* 10
* 12
* 26
* 70
* 100

These candidates were determined from the training dynamics observed in Run 02.

### Methodology

For each candidate epoch:

1. Use the same five stratified folds.
2. Reset the seed to 10 for each fold.
3. Calculate fold-specific temperature ranges from training data only.
4. Calculate fold-specific class weights from training data only.
5. Initialize a new ImageNet-pretrained ResNet-18.
6. Freeze the early backbone.
7. Train layer4 and the classifier.
8. Train exactly to the candidate epoch.
9. Evaluate the held-out fold.
10. Calculate mean and SD across the five folds.

A total of:

```text
6 candidate epochs × 5 folds = 30 trainings
```

are performed.

### Candidate Comparison

| Epoch | Mean ROC-AUC | SD | Mean Accuracy | Mean Sensitivity | Mean Specificity | Mean F1 |
|---|---|---|---|---|---|---|
| 5 | 0.920571 | 0.034775 | 0.847949 | 0.843810 | 0.848000 | 0.802373 |
| 10 | 0.932114 | 0.025803 | 0.847436 | 0.649524 | 0.960000 | 0.752019 |
| 12 | 0.923314 | 0.034431 | 0.862692 | 0.790476 | 0.904000 | 0.806799 |
| 26 | 0.930629 | 0.013967 | 0.862949 | 0.805714 | 0.896000 | 0.811431 |
| 70 | 0.915619 | 0.024693 | 0.802179 | 0.775238 | 0.816000 | 0.738574 |
| 100 | 0.926324 | 0.011275 | 0.862821 | 0.818095 | 0.888000 | 0.813286 |

### Selected Epoch

The fixed training epoch was selected using the highest mean 5-fold validation ROC-AUC.

```text
Selected epoch = 10

Mean validation ROC-AUC = 0.932114

SD = 0.025803
```

The lower epoch would be used as the tie-breaker if candidate epochs produced equal mean ROC-AUC.

The Test set was not used in this selection.

### Role in EXP03

Run 03 establishes the fixed training duration used by all subsequent final-model development steps:

```text
Fixed training epoch = 10
```

This prevents training duration from being adjusted using the final Test set.

---

## 8. Run 04 — 5-Fold Cross-Validation OOF Predictions

### Objective

Run 04 generates a complete set of out-of-fold predictions for the 197-image development dataset using the fixed 10-epoch configuration established in Run 03.

The purpose is to obtain unbiased development-set probabilities for threshold selection.

### Methodology

Five-fold stratified cross-validation is performed.

For each fold:

```text
Train on 4 folds

        ↓

Train exactly 10 epochs

        ↓

Predict the held-out fold

        ↓

Store raw Sick probabilities
```

No threshold is applied during prediction generation.

No early stopping is used.

The final model state at epoch 10 is used for held-out predictions.

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
ROC-AUC = 0.919444

PR-AUC  = 0.880197
```

These metrics describe the ranking performance of the OOF predictions before threshold selection.

### Fold Results

| Fold | Train Images | Validation Images | ROC-AUC | PR-AUC | Trainable Parameters |
|---|---|---|---|---|---|
| 1 | 157 | 40 | 0.922667 | 0.939398 | 8,426,625 |
| 2 | 157 | 40 | 0.922667 | 0.884191 | 8,426,625 |
| 3 | 158 | 39 | 0.957143 | 0.953640 | 8,426,625 |
| 4 | 158 | 39 | 0.902857 | 0.850208 | 8,426,625 |
| 5 | 158 | 39 | 0.925714 | 0.907854 | 8,426,625 |

### Role in EXP03

Run 04 provides the complete OOF probability distribution required for threshold selection.

The Test set remains untouched.

---

## 9. Run 05 — OOF Threshold Selection

### Objective

Run 05 selects the final classification threshold using only the OOF predictions generated in Run 04.

No model retraining occurs in this run.

### Threshold Selection Criteria

Thresholds are evaluated using the complete set of unique OOF probabilities.

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

This is a constraint-based threshold selection procedure rather than selection based on maximum accuracy, F1-score, ROC-AUC or PR-AUC.

### Selected Threshold

```text
Selected threshold = 0.027535
```

Eighteen thresholds satisfied the required sensitivity and specificity constraints.

### OOF Performance at Selected Threshold

```text
Sensitivity        = 0.9583

Specificity        = 0.6720

Accuracy           = 0.7766

Balanced Accuracy  = 0.8152

Precision          = 0.6273

F1-score           = 0.7582
```

Confusion matrix:

```text
TN = 84

FP = 41

FN = 3

TP = 69
```

The corresponding ranking metrics remain:

```text
ROC-AUC = 0.919444

PR-AUC  = 0.880197
```

### Role in EXP03

Run 05 fixes the classification threshold before final training and Test evaluation.

```text
Final threshold = 0.027535
```

The Test set is not used during threshold selection.

---

## 10. Run 06 — Final Model Training

### Objective

Run 06 trains the final EXP03 model using all available development data after the model configuration, training duration and classification threshold have been established.

### Training Dataset

```text
Development images = 197

Healthy = 125

Sick    = 72
```

The original Train and Validation sets are combined.

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

pos_weight = 1.736111
```

### Model

```text
Architecture = ResNet-18

Weights = ImageNet1K_V1

Frozen:

conv1

bn1

layer1

layer2

layer3

Trainable:

layer4

classifier
```

Parameters:

```text
Total      = 11,209,409

Trainable  = 8,426,625

Frozen     = 2,782,784
```

Classifier:

```text
Linear(512, 64)

ReLU

Dropout(0.3)

Linear(64, 1)
```

### Training

```text
Epochs = 10

Batch size = 16

Optimizer = Adam

Layer4 LR = 1e-4

Classifier LR = 1e-3

Loss = Weighted BCEWithLogitsLoss

Seed = 10
```

No validation-based model selection or early stopping is used because all development images are used for final training.

### Training Loss

| Epoch | Training Loss |
|---|---|
| 1 | 0.773133 |
| 2 | 0.568736 |
| 3 | 0.501632 |
| 4 | 0.399438 |
| 5 | 0.328628 |
| 6 | 0.247002 |
| 7 | 0.280955 |
| 8 | 0.296552 |
| 9 | 0.235190 |
| 10 | 0.172078 |

The final model checkpoint is:

```text
Results/Run_06_Final_Training/model/final_model_epoch10.pth
```

The threshold from Run 05 is carried forward:

```text
Threshold = 0.027535
```

The threshold is not involved in gradient-based model training.

### Role in EXP03

Run 06 produces the final model used for independent Test evaluation.

---

## 11. Run 07 — Final Test Evaluation

### Objective

Run 07 performs the final one-time evaluation of the complete EXP03 pipeline on the independent Test set.

The final model from Run 06 and the fixed threshold from Run 05 are used without modification.

### Test Dataset

```text
Healthy = 24

Sick    = 13

Total   = 37
```

The Test-set integrity check passed:

```text
Total images = 37

Healthy = 24

Sick = 13
```

### Test Preprocessing

The Run 06 development-derived temperature range is used:

```text
Minimum = 22.000000

Maximum = 36.812618
```

Processing:

```text
Raw thermal image

        ↓

Temperature normalization

        ↓

Valid-pixel masking

        ↓

Clip to [0, 1]

        ↓

Zero padding to square

        ↓

Convert to uint8

        ↓

Resize to 224 × 224

        ↓

Convert grayscale to 3 channels

        ↓

ImageNet normalization

        ↓

ResNet-18
```

No random augmentation is applied.

### Classification Threshold

The threshold established in Run 05 is used without modification:

```text
Threshold = 0.027535
```

Classification:

```text
Probability >= 0.027535 → Sick

Probability <  0.027535 → Healthy
```

No threshold tuning is performed using Test predictions.

### Final Test Performance

| Metric | Test Result |
|---|---|
| ROC-AUC | 0.9359 |
| Accuracy | 0.7297 |
| Balanced Accuracy | 0.7917 |
| Sensitivity | 1.0000 |
| Specificity | 0.5833 |
| Precision | 0.5652 |
| F1-score | 0.7222 |
| MCC | 0.5742 |
| Classification Threshold | 0.027535 |

### Confusion Matrix

```text
                Predicted

                Healthy   Sick

Actual Healthy     14       10

       Sick         0       13
```

Therefore:

```text
TN = 14

FP = 10

FN = 0

TP = 13
```

For the 24 Healthy images:

```text
Correctly classified as Healthy = 14

Incorrectly classified as Sick   = 10
```

For the 13 Sick images:

```text
Correctly classified as Sick      = 13

Incorrectly classified as Healthy = 0
```

The resulting metrics are internally consistent:

```text
Sensitivity = 13 / 13 = 1.0000

Specificity = 14 / 24 = 0.5833

Accuracy = (14 + 13) / 37 = 0.7297
```

### ROC-AUC Interpretation

The final Test ROC-AUC is:

```text
0.9359
```

ROC-AUC evaluates the ranking of model predictions across thresholds.

Accuracy, sensitivity, specificity, precision and F1-score are threshold-dependent and were calculated using the pre-specified threshold:

```text
0.027535
```

Therefore, ROC-AUC should be interpreted separately from the threshold-dependent classification metrics.

---

## 12. EXP03 Experimental Progression

The complete EXP03 methodology can be summarized as:

```text
Partial Fine-Tuning Baseline

        │

        │ Run 01

        ↓

Observe training/validation dynamics

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

Select epoch 10

        │

        │ Run 04

        ↓

Generate 197 OOF predictions

        │

        │ Run 05

        ↓

Select threshold 0.027535

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
Model configuration

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

## 13. Final EXP03 Results

The final EXP03 model is a partially fine-tuned ImageNet-pretrained ResNet-18 with:

```text
Trainable:

layer4 + classifier

Frozen:

conv1 + bn1 + layer1 + layer2 + layer3

Fixed training duration:

10 epochs

Final classification threshold:

0.027535
```

The model was trained using all 197 development images.

The independent Test set contained 37 images.

Final Test performance:

```text
ROC-AUC           = 0.9359

Accuracy          = 72.97%

Balanced Accuracy = 79.17%

Sensitivity       = 100.00%

Specificity       = 58.33%

Precision         = 56.52%

F1-score          = 72.22%

MCC               = 57.42%
```

Confusion matrix:

```text
TN = 14

FP = 10

FN = 0

TP = 13
```

The final Test ROC-AUC of 0.9359 reflects the ranking performance of the final model on the independent Test set.

The threshold-dependent metrics were calculated using the classification threshold of 0.027535 that had been fixed using development-set OOF predictions before Test evaluation.

---

## 14. Test-Set Isolation

A central methodological property of EXP03 is the isolation of the independent Test set.

The Test set was not used for:

* Training
* Validation
* Cross-validation
* Training epoch selection
* Threshold selection
* Model selection
* Hyperparameter tuning
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

The complete EXP03 directory contains the individual run outputs.

```text
EXP_03_DMR_Anterior_ResNet18_Partial_FineTuning/

├── Run_01_Partial_FineTuning_Baseline/

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
Run_01_Partial_FineTuning_Baseline/

├── README.md

├── results/

│   ├── validation_metrics.json

│   ├── validation_roc_data.json

│   ├── training_history.json

│   ├── training_summary.json

│   └── model_summary.txt

├── checkpoints/

│   └── resnet18_partial_finetuning_best.pth

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

├── results/

│   ├── fold_summary.csv

│   ├── fold_temperature_ranges.json

│   ├── aggregate_epoch_metrics.csv

│   ├── cv_summary.json

│   ├── training_summary.json

│   ├── model_summary.txt

│   └── run_summary.json

├── Fold_1/

├── Fold_2/

├── Fold_3/

├── Fold_4/

├── Fold_5/

└── plots/
```

### Run 03

```text
Run_03_Fixed_Epoch_Selection/

├── README.md

├── results/

│   ├── candidate_epochs.json

│   ├── candidate_epoch_comparison.json

│   ├── candidate_epoch_comparison.csv

│   └── run_summary.json

├── epoch_5/

├── epoch_10/

├── epoch_12/

├── epoch_26/

├── epoch_70/

└── epoch_100/
```

Each candidate epoch contains results from all five folds.

### Run 04

```text
Run_04_5-fold_CV_OOF_Predictions/

├── README.md

├── results/

│   ├── fold_1_predictions.csv

│   ├── fold_2_predictions.csv

│   ├── fold_3_predictions.csv

│   ├── fold_4_predictions.csv

│   ├── fold_5_predictions.csv

│   ├── fold_1_temperature_range.json

│   ├── fold_2_temperature_range.json

│   ├── fold_3_temperature_range.json

│   ├── fold_4_temperature_range.json

│   ├── fold_5_temperature_range.json

│   ├── fold_1_training_history.json

│   ├── fold_2_training_history.json

│   ├── fold_3_training_history.json

│   ├── fold_4_training_history.json

│   ├── fold_5_training_history.json

│   ├── fold_1_summary.json

│   ├── fold_2_summary.json

│   ├── fold_3_summary.json

│   ├── fold_4_summary.json

│   ├── fold_5_summary.json

│   ├── oof_predictions.csv

│   ├── fold_summary.csv

│   └── run_summary.json
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

├── model/

│   └── final_model_epoch10.pth

├── temperature_range.json

├── training_history.json

├── run_config.json

├── development_manifest.csv

└── plots/

    └── training_loss.png
```

### Run 07

```text
Run_07_Final_Test_Evaluation/

├── README.md

├── config/

│   └── run_config.json

├── metrics/

│   └── test_metrics.json

├── predictions/

│   └── test_predictions.csv

└── plots/

    ├── test_roc_curve.png

    └── test_confusion_matrix.png
```

---

## 16. Final Experiment Summary

EXP03 evaluated a partially fine-tuned ImageNet-pretrained ResNet-18 for binary classification of anterior-view breast thermal images.

The experiment used a structured seven-run development pipeline.

Run 01 established the partial fine-tuning baseline using the predefined Train/Validation split and demonstrated substantial variation in validation performance across training epochs.

Run 02 expanded the analysis to stratified 5-fold cross-validation on all 197 development images and evaluated training dynamics over 100 epochs.

Run 03 formally compared six candidate training durations and selected a common fixed training epoch of 10 based on the highest mean 5-fold validation ROC-AUC:

```text
Mean ROC-AUC = 0.932114 ± 0.025803
```

Run 04 then generated one out-of-fold prediction for every development image using the fixed 10-epoch configuration:

```text
197 / 197 OOF predictions

OOF ROC-AUC = 0.919444

OOF PR-AUC  = 0.880197
```

Run 05 selected the final classification threshold from these OOF predictions:

```text
Threshold = 0.027535
```

The threshold was selected using the constraints:

```text
Sensitivity > 0.91

Specificity > 0.60
```

Eighteen thresholds satisfied these constraints. The selected threshold achieved:

```text
Sensitivity        = 0.9583

Specificity        = 0.6720

Accuracy           = 0.7766

Balanced Accuracy  = 0.8152

Precision          = 0.6273

F1-score           = 0.7582
```

Run 06 trained the final partially fine-tuned ResNet-18 for exactly 10 epochs using all 197 development images.

Run 07 evaluated the resulting model once on the independent 37-image Test set.

The final Test ROC-AUC was:

```text
0.9359
```

At the pre-specified OOF-derived threshold of 0.027535:

```text
Accuracy          = 72.97%

Balanced Accuracy = 79.17%

Sensitivity       = 100.00%

Specificity       = 58.33%

Precision         = 56.52%

F1-score          = 72.22%

MCC               = 57.42%
```

The final confusion matrix was:

```text
TN = 14

FP = 10

FN = 0

TP = 13
```

The Test set was isolated from all model-development decisions and was used only for the final evaluation in Run 07.

Therefore, the complete EXP03 pipeline is:

```text
Partial Fine-Tuning

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

This completes EXP03.