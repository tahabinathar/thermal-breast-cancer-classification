# EXP02 Run 02: 5-Fold CV Training Dynamics

## 1. Objective

Evaluate the training dynamics of the EXP02 ResNet-18 transfer learning model using stratified 5-fold cross-validation on the pooled development dataset.

The entire ResNet-18 backbone was frozen and only the classification head was trained.

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
│   ├── StratifiedKFold (shuffle = True, random_state = 10)
│   └── 5 folds
│
├── Within Each Fold
│   ├── Reset seed to 10
│   ├── Fold-specific temperature range
│   ├── Preprocess train/validation images
│   ├── Calculate class weights from fold training data
│   ├── Initialize ImageNet-pretrained ResNet-18
│   ├── Freeze the ResNet-18 backbone
│   ├── Train classification head for 100 epochs
│   ├── Record training and validation history at every epoch
│   ├── Save best validation-AUC checkpoint
│   ├── Identify fold-specific best epoch
│   ├── Reload best checkpoint
│   └── Calculate validation metrics using the best checkpoint
│
├── Aggregate Results
│   ├── Mean ± SD of fold-level metrics
│   │   ├── Best validation ROC-AUC
│   │   ├── ROC-AUC
│   │   ├── PR-AUC
│   │   ├── Accuracy
│   │   ├── Balanced Accuracy
│   │   ├── Sensitivity
│   │   ├── Specificity
│   │   ├── Precision
│   │   ├── NPV
│   │   ├── F1-score
│   │   └── MCC
│   └── Fold-specific best epochs
│
├── Identify Common Training Epoch
│   └── Highest mean epoch-wise validation ROC-AUC
│
└── Training Dynamics
    ├── Collect the five fold training histories
    ├── Calculate mean ± SD of loss, accuracy, AUC, validation loss,
    │   validation accuracy and validation AUC for each epoch
    └── Save epoch-wise results and plots
```

---

## 3. Configuration

| Parameter           | Value                                         |
| ------------------- | --------------------------------------------- |
| Model               | ResNet-18                                     |
| Pretrained weights  | ImageNet-1K V1                                |
| Thermal image size  | 224 × 224 × 1                                 |
| ResNet-18 input     | 224 × 224 × 3                                 |
| Channel conversion  | Grayscale repeated to 3 channels              |
| Backbone            | Frozen                                        |
| Classification head | 512 → 64 → 1                                  |
| Dropout             | 0.3                                           |
| Batch size          | 16                                            |
| Epochs              | 100                                           |
| Optimizer           | Adam (classification head parameters only)    |
| Learning rate       | 1e-3                                          |
| Loss                | Weighted BCEWithLogitsLoss                    |
| Class weighting     | Fold-specific balanced weights                |
| Augmentation        | Rotation ±18°, scale 0.95–1.05                 |
| Horizontal flip     | Not used                                      |
| Folds               | 5                                             |
| CV splitter         | StratifiedKFold (shuffle = True, seed = 10)   |
| Threshold           | 0.5                                           |
| Seed                | 10                                            |
| Seed reset          | At beginning of every fold                    |
| Device              | CUDA                                          |
| Selection metric    | Mean epoch-wise validation ROC-AUC            |

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

Convert to PIL image

    ↓

Training augmentation (training images only)

    ↓

Convert to tensor

    ↓

Repeat grayscale channel to 3 channels

    ↓

ImageNet normalization
```

Data augmentation was applied only to the training images.

The validation images were processed without augmentation.

Class weights were calculated separately for each fold using only the fold training data and were applied as per-sample weights in the binary cross-entropy loss during training. The same weights were applied when calculating the validation loss.

The ResNet-18 backbone was kept frozen during training. The frozen backbone modules were kept in evaluation mode to prevent changes to BatchNorm running statistics, while the classification head remained in training mode.

---

## 5. Results

| Fold | Train Samples | Validation Samples | Best Epoch | Epoch 1 AUC | Best Validation AUC | Epoch 100 AUC | ROC-AUC | PR-AUC | Accuracy | Sensitivity | Specificity |     F1 |
| ---: | ------------: | -----------------: | ---------: | ----------: | ------------------: | ------------: | ------: | -----: | -------: | ----------: | ----------: | -----: |
|    1 |           157 |                 40 |         68 |      0.8800 |              0.9413 |        0.9280 |  0.9413 | 0.9527 |   0.9000 |      0.8000 |      0.9600 | 0.8571 |
|    2 |           157 |                 40 |         89 |      0.8373 |              0.9253 |        0.9227 |  0.9253 | 0.8467 |   0.8250 |      0.8000 |      0.8400 | 0.7742 |
|    3 |           158 |                 39 |         67 |      0.9314 |              0.9886 |        0.9800 |  0.9886 | 0.9805 |   0.9231 |      1.0000 |      0.8800 | 0.9032 |
|    4 |           158 |                 39 |         32 |      0.8829 |              0.9114 |        0.8971 |  0.9114 | 0.8408 |   0.8462 |      0.7143 |      0.9200 | 0.7692 |
|    5 |           158 |                 39 |         23 |      0.8114 |              0.8800 |        0.8086 |  0.8800 | 0.8518 |   0.8205 |      0.7143 |      0.8800 | 0.7407 |

The fold-level ROC-AUC, PR-AUC, accuracy, sensitivity, specificity and F1 values were calculated at a threshold of 0.5 using the **best validation-AUC checkpoint** of each fold. The fold-level ROC-AUC is therefore identical to the fold's Best Validation AUC.

### Common Training Epoch

* **Best epoch-wise mean validation ROC-AUC epoch:** 32
* **Best mean validation ROC-AUC:** 0.9178

The validation ROC-AUC from all five folds was aggregated separately for each epoch. The highest mean validation ROC-AUC occurred at **epoch 32**, with a mean value of **0.9178**.

The individual folds reached their own highest validation ROC-AUC at different epochs:

* Fold 1: epoch 68
* Fold 2: epoch 89
* Fold 3: epoch 67
* Fold 4: epoch 32
* Fold 5: epoch 23

Therefore, epoch 32 represents the **common epoch with the highest mean validation ROC-AUC across the five folds**. It is not the individual best epoch for every fold.

---

## 6. Epoch-wise Validation AUC

After the five folds were trained, the training and validation histories from each fold were aggregated epoch-wise.

For each epoch, the mean and standard deviation (population SD) of training loss, validation loss, training accuracy, validation accuracy, training AUC and validation AUC across the five folds were calculated.

The five fold training histories were not retrained or modified during this analysis.

The highest mean validation ROC-AUC occurred at:

* **Epoch:** 32
* **Mean validation ROC-AUC:** 0.9178

The epoch-wise results were saved under:

```text
Run_02_5-fold_CV_Training_Dynamics/
├── results/
│   ├── epoch_wise_cv_summary.csv
│   └── cv_summary.json
└── plots/
    ├── cv_validation_auc_mean_sd.png
    └── cv_loss_mean_sd.png
```

These results describe the training dynamics of the frozen-backbone ResNet-18 model across the five validation folds.

The epoch-wise analysis was used to identify a candidate common training epoch for subsequent experimentation.

---

## 7. Output Structure

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

Folders `fold_2` to `fold_5` have the same structure as `fold_1`.

The script also creates an empty run-level `checkpoints/` folder. All checkpoints are stored inside the fold folders.

---

## 8. Final Summary

* **Development images:** 197
* **Healthy images:** 125
* **Sick images:** 72
* **Cross-validation:** 5-fold stratified
* **Total model trainings:** 5
* **Epochs per training:** 100
* **Fixed learning rate:** 1e-3
* **ResNet-18 backbone:** Frozen
* **Classification head dropout:** 0.3
* **Thermal image size:** 224 × 224 × 1
* **ResNet-18 input:** 224 × 224 × 3
* **Seed:** 10
* **Seed reset:** At the beginning of every fold
* **Best epoch-wise mean validation ROC-AUC epoch:** 32
* **Best mean validation ROC-AUC:** 0.9178
* **Test set:** Not used

The experiment was completed successfully. The test set was not loaded, evaluated or used during training, preprocessing or model selection.
