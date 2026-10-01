# EXP03 Run 02: 5-Fold CV Training Dynamics

## 1. Objective

Evaluate the training dynamics of the EXP03 ResNet-18 partial fine-tuning model using stratified 5-fold cross-validation on the pooled development dataset.

The pretrained ResNet-18 backbone was partially fine-tuned by training `layer4` together with a newly initialized classification head, while `conv1`, `bn1`, `layer1`, `layer2` and `layer3` remained frozen.

Each fold was trained for 100 epochs. Validation ROC-AUC was recorded at every epoch and the best validation-AUC epoch was identified independently for each fold.

The test set was not loaded, evaluated or used during training or model selection.

---

## 2. Methodology

```text
RUN 02

│
├── Development Dataset
│   ├── Original Train + Original Validation = 197 images
│   ├── Healthy = 125
│   ├── Sick = 72
│   └── Test = NOT USED
│
├── 5-Fold Stratified CV
│   └── 5 folds
│
├── Within Each Fold
│   ├── Reset seed to 10
│   ├── Create fold-specific training and validation sets
│   ├── Calculate temperature range using fold training data only
│   ├── Preprocess train/validation images
│   ├── Calculate class weights from fold training data
│   ├── Initialize ImageNet-pretrained ResNet-18
│   ├── Freeze conv1, bn1, layer1, layer2 and layer3
│   ├── Fine-tune layer4
│   ├── Train new classification head
│   ├── Train for 100 epochs
│   ├── Record training and validation loss
│   ├── Record training and validation ROC-AUC
│   ├── Save best validation-AUC checkpoint
│   ├── Identify fold-specific best epoch
│   └── Calculate validation metrics
│
├── Aggregate Results
│   ├── Mean validation ROC-AUC for each epoch
│   ├── SD validation ROC-AUC for each epoch
│   ├── Fold-specific best validation ROC-AUC
│   ├── ROC-AUC
│   ├── PR-AUC
│   ├── Accuracy
│   ├── Balanced Accuracy
│   ├── Sensitivity
│   ├── Specificity
│   ├── Precision
│   ├── NPV
│   ├── F1-score
│   └── MCC
│
├── Identify Aggregate Training Epoch
│   └── Highest mean epoch-wise validation ROC-AUC
│
└── Training Dynamics
    ├── Load five fold training histories
    ├── Calculate mean validation ROC-AUC for each epoch
    ├── Calculate SD validation ROC-AUC for each epoch
    ├── Calculate mean/SD training and validation loss
    ├── Calculate mean/SD training and validation ROC-AUC
    └── Save epoch-wise results and plots
```

---

## 3. Configuration

| Parameter                | Value                              |
| ------------------------ | ---------------------------------- |
| Model                    | ResNet-18                          |
| Pretrained weights       | ImageNet-1K V1                     |
| Thermal image size       | 224 × 224 × 1                      |
| ResNet-18 input          | 224 × 224 × 3                      |
| Channel conversion       | Grayscale repeated to 3 channels   |
| Backbone strategy        | Partial fine-tuning                |
| Frozen layers            | conv1, bn1, layer1, layer2, layer3 |
| Trainable backbone layer | layer4                             |
| Classification head      | 512 → 64 → 1                       |
| Dropout                  | 0.3                                |
| Batch size               | 16                                 |
| Epochs                   | 100                                |
| Optimizer                | Adam                               |
| Layer4 learning rate     | 1e-4                               |
| Classifier learning rate | 1e-3                               |
| Augmentation             | Rotation 18°, scale 0.95–1.05      |
| Horizontal flip          | Not used                           |
| Folds                    | 5                                  |
| Threshold                | 0.5                                |
| Threshold status         | Diagnostic only                    |
| Seed                     | 10                                 |
| Seed reset               | At the beginning of every fold     |
| Device                   | CUDA                               |
| Selection metric         | Validation ROC-AUC                 |

---

## 4. Preprocessing

For each fold, the temperature range was calculated using **only the fold's training images**. The same fold-specific temperature range was then used to preprocess both the corresponding training and validation images.

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

The training transformation consisted of:

* Random rotation of ±18°
* Random affine scaling from 0.95 to 1.05
* Conversion to tensor
* Repetition of the single grayscale channel to three channels
* ImageNet normalization

The validation images were processed without augmentation.

Class weights were calculated separately for each fold using only the fold training data.

The resulting class weights were applied to the binary cross-entropy loss during training.

---

## 5. Results

### Fold-Level Results

| Fold | Train Samples | Validation Samples | Best Epoch | Epoch 1 AUC | Best Validation AUC | Epoch 100 AUC | ROC-AUC | PR-AUC | Accuracy | Sensitivity | Specificity |     F1 |
| ---: | ------------: | -----------------: | ---------: | ----------: | ------------------: | ------------: | ------: | -----: | -------: | ----------: | ----------: | -----: |
|    1 |           157 |                 40 |         10 |      0.8480 |              0.9413 |        0.9067 |  0.9413 | 0.9433 |   0.8500 |      0.8667 |      0.8400 | 0.8125 |
|    2 |           157 |                 40 |         12 |      0.8800 |              0.9520 |        0.9333 |  0.9520 | 0.9244 |   0.8500 |      0.8000 |      0.8800 | 0.8000 |
|    3 |           158 |                 39 |         70 |      0.9771 |              0.9914 |        0.9629 |  0.9914 | 0.9874 |   0.9231 |      0.7857 |      1.0000 | 0.8800 |
|    4 |           158 |                 39 |         26 |      0.8571 |              0.9514 |        0.8857 |  0.9514 | 0.9190 |   0.7949 |      1.0000 |      0.6800 | 0.7778 |
|    5 |           158 |                 39 |          5 |      0.8829 |              0.9600 |        0.9086 |  0.9600 | 0.9481 |   0.8718 |      0.8571 |      0.8800 | 0.8276 |

### Additional Cross-Validation Metrics

| Fold | Balanced Accuracy | Precision |    NPV |    MCC |
| ---: | ----------------: | --------: | -----: | -----: |
|    1 |            0.8533 |    0.7647 | 0.9130 | 0.6921 |
|    2 |            0.8400 |    0.8000 | 0.8800 | 0.6800 |
|    3 |            0.8929 |    1.0000 | 0.8929 | 0.8376 |
|    4 |            0.8400 |    0.6364 | 1.0000 | 0.6578 |
|    5 |            0.8686 |    0.8000 | 0.9167 | 0.7268 |

---

## 6. Aggregate Cross-Validation Results

The fold-specific best validation ROC-AUC values were:

* Fold 1: 0.9413 at epoch 10
* Fold 2: 0.9520 at epoch 12
* Fold 3: 0.9914 at epoch 70
* Fold 4: 0.9514 at epoch 26
* Fold 5: 0.9600 at epoch 5

The mean of the individual fold-best validation ROC-AUC values was:

**0.9592 ± 0.0185**

The mean individual best epoch was:

**24.60 ± 26.79 epochs**

These values describe the individual fold optima. They should not be interpreted as a common fixed training epoch.

---

## 7. Common Training Epoch

The validation ROC-AUC from all five folds was aggregated separately for every training epoch.

The highest mean epoch-wise validation ROC-AUC occurred at:

* **Best epoch-wise mean validation ROC-AUC epoch: 26**
* **Mean validation ROC-AUC: 0.9329**
* **Validation ROC-AUC SD: 0.0200**

At epoch 26, the five fold validation ROC-AUC values were:

| Fold | Validation ROC-AUC |
| ---: | -----------------: |
|    1 |             0.9200 |
|    2 |             0.9360 |
|    3 |             0.9514 |
|    4 |             0.9514 |
|    5 |             0.9057 |

Therefore, **epoch 26 represents the epoch with the highest mean validation ROC-AUC across the five folds in Run02**.

Importantly, this does **not** mean that epoch 26 was the best epoch for every individual fold.

The individual best epochs were:

* Fold 1: epoch 10
* Fold 2: epoch 12
* Fold 3: epoch 70
* Fold 4: epoch 26
* Fold 5: epoch 5

The large variation in fold-specific best epochs demonstrates that the individual fold optima are not consistent. The epoch-wise aggregate analysis therefore provides a common candidate epoch for the subsequent fixed-epoch experiment.

**Run02 itself does not perform fixed-epoch cross-validation.**

---

## 8. Epoch-wise Validation AUC

After all five folds were trained for 100 epochs, the validation ROC-AUC from each fold was aggregated epoch-wise.

For each epoch, the following were calculated:

* Mean validation ROC-AUC across five folds
* Standard deviation of validation ROC-AUC across five folds
* Validation ROC-AUC for each individual fold

The highest mean validation ROC-AUC occurred at:

* **Epoch:** 26
* **Mean validation ROC-AUC:** 0.9329
* **SD:** 0.0200

The epoch-wise results were saved as:

```text
Results/

├── aggregate_epoch_metrics.csv
├── fold_summary.csv
├── fold_temperature_ranges.json
├── cv_summary.json
├── training_summary.json
├── model_summary.txt
└── run_summary.json
```

The aggregate validation-AUC plot was saved as:

```text
plots/

└── cv_mean_validation_auc.png
```

Additional aggregate plots were generated for:

```text
plots/

├── cv_mean_validation_auc.png
├── cv_mean_loss.png
└── cv_mean_auc.png
```

These results describe the training dynamics of the partially fine-tuned ResNet-18 model across the five validation folds.

The epoch-wise analysis identifies **epoch 26 as the candidate common training epoch** for subsequent experimentation.

---

## 9. Model Architecture

The experiment used an ImageNet-pretrained ResNet-18.

The original classification layer was replaced with:

```text
512
 ↓
Linear
 ↓
64
 ↓
ReLU
 ↓
Dropout (0.3)
 ↓
Linear
 ↓
1
```

The fine-tuning strategy was:

```text
Frozen:
    conv1
    bn1
    layer1
    layer2
    layer3

Trainable:
    layer4
    fc
```

Separate learning rates were used for the trainable ResNet-18 layer4 and the classification head:

```text
Layer4 learning rate      = 1e-4
Classifier learning rate  = 1e-3
```

The optimizer was Adam.

---

## 10. Output Structure

The actual Run02 output structure generated by the code is:

```text
Run_02_5-fold_CV_Training_Dynamics/

├──README.md
|
├── results/
│   ├── fold_summary.csv
│   ├── fold_temperature_ranges.json
│   ├── aggregate_epoch_metrics.csv
│   ├── cv_summary.json
│   ├── training_summary.json
│   ├── model_summary.txt
│   └── run_summary.json
│
├── Fold_1/
│   ├── results/
│   │   ├── fold_split.csv
│   │   ├── temperature_range.json
│   │   ├── training_history.json
│   │   ├── validation_metrics.json
│   │   ├── validation_roc_data.json
│   │   ├── fold_summary.json
│   │   └── validation_predictions.csv
│   ├── checkpoints/
│   │   └── best_model.pth
│   └── plots/
│       ├── training_validation_loss.png
│       └── training_validation_auc.png
│
├── Fold_2/
│   ├── results/
│   ├── checkpoints/
│   └── plots/
│
├── Fold_3/
│   ├── results/
│   ├── checkpoints/
│   └── plots/
│
├── Fold_4/
│   ├── results/
│   ├── checkpoints/
│   └── plots/
│
├── Fold_5/
│   ├── results/
│   ├── checkpoints/
│   └── plots/
│
└── plots/
    ├── cv_mean_validation_auc.png
    ├── cv_mean_loss.png
    └── cv_mean_auc.png
```

Each fold contains its own:

* Fold split
* Training-only temperature range
* Training history
* Best-model checkpoint
* Validation metrics
* Validation ROC data
* Validation predictions
* Fold summary
* Training/validation plots

---

## 11. Final Summary

* **Development images:** 197
* **Healthy images:** 125
* **Sick images:** 72
* **Cross-validation:** 5-fold stratified
* **Total model trainings:** 5
* **Epochs per training:** 100
* **Optimizer:** Adam
* **Layer4 learning rate:** 1e-4
* **Classifier learning rate:** 1e-3
* **ResNet-18 backbone:** Partially fine-tuned
* **Frozen layers:** conv1, bn1, layer1, layer2, layer3
* **Trainable layers:** layer4, fc
* **Classification head:** 512 → 64 → 1
* **Dropout:** 0.3
* **Thermal image size:** 224 × 224 × 1
* **ResNet-18 input:** 224 × 224 × 3
* **Batch size:** 16
* **Seed:** 10
* **Seed reset:** At the beginning of every fold
* **Best aggregate epoch:** 26
* **Best mean epoch-wise validation ROC-AUC:** 0.9329
* **SD at aggregate best epoch:** 0.0200
* **Mean individual best-fold ROC-AUC:** 0.9592
* **SD individual best-fold ROC-AUC:** 0.0185
* **Mean individual best epoch:** 24.60
* **SD individual best epoch:** 26.79
* **Diagnostic threshold:** 0.5
* **Final decision threshold selected:** NO
* **Test set:** Not used

The experiment was completed successfully. The test set was not loaded, evaluated or used during training, preprocessing or model selection.

**EXP03 Run02 identifies epoch 26 as the common candidate epoch based on the highest mean validation ROC-AUC across the five folds. This result is used to inform the subsequent fixed-epoch experiment and does not itself constitute fixed-epoch cross-validation.**