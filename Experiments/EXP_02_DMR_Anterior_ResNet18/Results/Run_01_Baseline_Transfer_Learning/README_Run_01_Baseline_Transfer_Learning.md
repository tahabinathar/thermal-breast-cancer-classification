# EXP02_DMR_Anterior_ResNet18

## Run 01: Baseline Transfer Learning

## 1. Objective

Evaluate a transfer-learning baseline for distinguishing **Healthy** and **Sick** breast thermography images using an ImageNet-pretrained ResNet-18 model.

The predefined training and validation sets were used. The test set was not used.

---

## 2. Methodology

```text
RUN 01

│
├── Dataset
│   ├── Training = 163 images
│   │   ├── Healthy = 103
│   │   └── Sick = 60
│   ├── Validation = 34 images
│   │   ├── Healthy = 22
│   │   └── Sick = 12
│   └── Test = NOT USED
│
├── Preprocessing
│   ├── Calculate temperature range from training images only
│   ├── Normalize temperature values
│   ├── Pad to square
│   ├── Resize to 224×224
│   ├── Add single channel
│   ├── Replicate grayscale channel to 3 channels
│   └── Apply ImageNet normalization
│
├── Data Augmentation
│   ├── Random rotation = ±18°
│   ├── Random scale = 0.95–1.05
│   └── Horizontal flip = NOT USED
│
├── ResNet-18 Transfer Learning
│   ├── ImageNet pretrained ResNet-18
│   ├── Backbone = Frozen
│   ├── Input = 224×224×1
│   ├── Classification head = 2048 → 64 → 1
│   ├── ReLU activation
│   ├── Dropout = 0.3
│   └── Single binary logit output
│
├── Training
│   ├── Adam optimizer
│   ├── Learning rate = 0.001
│   ├── Batch size = 16
│   ├── Maximum epochs = 100
│   ├── Balanced class weights
│   ├── Weighted BCEWithLogitsLoss
│   └── Monitor validation ROC-AUC
│
├── Model Selection
│   ├── Save best validation-AUC checkpoint
│   └── Select highest validation ROC-AUC
│
└── Final Evaluation
    ├── Validation threshold = 0.5
    ├── Calculate classification metrics
    └── Test set = NOT USED
```

---

## 3. Configuration

| Parameter           | Value                          |
| ------------------- | ------------------------------ |
| Input size          | 224 × 224 × 1                  |
| ResNet input        | 224 × 224 × 3                  |
| Backbone            | ResNet-18                      |
| Pretrained weights  | ImageNet                       |
| Backbone training   | Frozen                         |
| Classification head | 2048 → 64 → 1                  |
| Dropout             | 0.3                            |
| Batch size          | 16                             |
| Epochs              | 100                            |
| Optimizer           | Adam                           |
| Learning rate       | 0.001                          |
| Loss function       | Weighted BCEWithLogitsLoss     |
| Augmentation        | Rotation ±18°, Scale 0.95–1.05 |
| Horizontal flip     | Not used                       |
| Class weighting     | Enabled                        |
| Threshold           | 0.5                            |
| Seed                | 10                             |
| Device              | CUDA                           |
| GPU                 | NVIDIA Quadro M1200            |
| Selection metric    | Validation ROC-AUC             |
| Early stopping      | Not used                       |

---

## 4. Preprocessing

The thermal TIFF images were processed using the predefined preprocessing pipeline.

For Run 01, the temperature range was calculated **only from the 163 training images** and then applied to both the training and validation sets.

The calculated training temperature range was:

```text
Minimum temperature = 22.000000
Maximum temperature = 36.779999
```

The processed images were resized to 224 × 224 with a single thermal channel. Since the ImageNet-pretrained ResNet-18 expects three input channels, the single grayscale channel was replicated to three channels before applying ImageNet normalization.

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

Replicate to 3 channels

    ↓

ImageNet normalization

    ↓

ResNet-18
```

The preprocessing output before channel replication was **224 × 224 × 1**.

---

## 5. Class Weighting

Balanced class weights were calculated from the training set only.

| Class   | Label | Weight |
| ------- | ----: | -----: |
| Healthy |     0 | 0.7913 |
| Sick    |     1 | 1.3583 |

The higher weight for the Sick class compensates for its lower representation in the training data.

---

## 6. Model Selection

The model was trained for a maximum of 100 epochs without early stopping.

The checkpoint with the highest validation ROC-AUC was selected.

| Parameter               |     Result |
| ----------------------- | ---------: |
| Best epoch              |      **1** |
| Best validation ROC-AUC | **0.9053** |

The validation ROC-AUC was highest at Epoch 1. Although later epochs showed higher training performance, none exceeded the validation ROC-AUC obtained at Epoch 1.

---

## 7. Validation Results

The selected model was evaluated on the 34-image validation set using a fixed threshold of 0.5.

| Metric            |     Result |
| ----------------- | ---------: |
| ROC-AUC           | **0.9053** |
| PR-AUC            | **0.7556** |
| Accuracy          | **0.6471** |
| Balanced Accuracy | **0.5000** |
| Sensitivity       | **0.0000** |
| Specificity       | **1.0000** |
| Precision         | **0.0000** |
| NPV               | **0.6471** |
| F1-score          | **0.0000** |
| MCC               | **0.0000** |

At the fixed threshold of 0.5, all 34 validation images were classified as Healthy. The ROC-AUC of 0.9053 reflects the ranking performance of the continuous model outputs across thresholds and is therefore not directly equivalent to the classification performance at the fixed 0.5 threshold.

---

## 8. Confusion Matrix

|                    | Predicted Healthy | Predicted Sick |
| ------------------ | ----------------: | -------------: |
| **Actual Healthy** |                22 |              0 |
| **Actual Sick**    |                12 |              0 |

Therefore:

* TN = 22
* FP = 0
* FN = 12
* TP = 0

All 22 Healthy validation images were correctly classified, while all 12 Sick validation images were classified as Healthy at the 0.5 threshold.

---

## 9. Test Set Status

The test set was **not loaded or used** during Run 01.

It was not used for preprocessing, training, model selection, or evaluation.

The independent test set was preserved for later final evaluation.

---

## 10. Outputs

```text
Run_01_Baseline_Transfer_Learning/

├──README.md
├── results/
│   ├── validation_metrics.json
│   ├── validation_roc_data.json
│   ├── training_history.json
│   ├── training_summary.json
│   └── model_summary.txt
│
├── checkpoints/
│   └── resnet18_frozen_best.pth
│
└── plots/
    ├── training_validation_loss.png
    ├── training_validation_accuracy.png
    ├── training_validation_auc.png
    └── validation_roc_curve.png
```

---

## 11. Final Summary

* Training images: **163**
* Validation images: **34**
* Test images: **37, not used**
* Model: **ImageNet-pretrained ResNet-18 with frozen backbone**
* Classification head: **2048 → 64 → 1**
* Maximum epochs: **100**
* Best epoch: **1**
* Best validation ROC-AUC: **0.9053**
* Validation PR-AUC: **0.7556**
* Validation accuracy: **64.71%**
* Balanced accuracy: **50.00%**
* Sensitivity: **0.00%**
* Specificity: **100.00%**
* Precision: **0.00%**
* F1-score: **0.00%**
* Classification threshold: **0.5**

Run 01 establishes the transfer-learning baseline for EXP02 using a frozen ImageNet-pretrained ResNet-18. The model achieved a validation ROC-AUC of **0.9053**, with the best validation performance occurring at Epoch 1. The test set remained untouched and is preserved for later final evaluation.