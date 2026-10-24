# EXP03_DMR_Anterior_ResNet18_Partial_FineTuning

## Run 01: Partial Fine-Tuning Baseline

## 1. Objective

Evaluate a partial fine-tuning baseline for distinguishing **Healthy** and **Sick** breast thermography images using an ImageNet-pretrained ResNet-18 model.

Unlike the frozen-backbone transfer-learning baseline, the final ResNet-18 feature block (`layer4`) and classification layer were allowed to update during training.

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
├── ResNet-18 Partial Fine-Tuning
│   ├── ImageNet pretrained ResNet-18
│   ├── Frozen = conv1, bn1, layer1, layer2, layer3
│   ├── Trainable = layer4 and fc
│   ├── Input = 224×224×1
│   ├── Classification head = 512 → 64 → 1
│   ├── ReLU activation
│   ├── Dropout = 0.3
│   └── Single binary logit output
│
├── Training
│   ├── Adam optimizer
│   ├── Classifier learning rate = 0.001
│   ├── Backbone learning rate = 0.0001
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

| Parameter                | Value                              |
| ------------------------ | ---------------------------------- |
| Input size               | 224 × 224 × 1                      |
| ResNet input             | 224 × 224 × 3                      |
| Backbone                 | ResNet-18                          |
| Pretrained weights       | ImageNet1K_V1                      |
| Backbone training        | Partial fine-tuning                |
| Frozen modules           | conv1, bn1, layer1, layer2, layer3 |
| Trainable modules        | layer4, fc                         |
| Classification head      | 512 → 64 → 1                       |
| Total parameters         | 11,209,409                         |
| Trainable parameters     | 8,426,625                          |
| Frozen parameters        | 2,782,784                          |
| Dropout                  | 0.3                                |
| Batch size               | 16                                 |
| Maximum epochs           | 100                                |
| Optimizer                | Adam                               |
| Classifier learning rate | 0.001                              |
| Backbone learning rate   | 0.0001                             |
| Loss function            | Weighted BCEWithLogitsLoss         |
| Augmentation             | Rotation ±18°, Scale 0.95–1.05     |
| Horizontal flip          | Not used                           |
| Class weighting          | Enabled                            |
| Threshold                | 0.5                                |
| Seed                     | 10                                 |
| Device                   | CUDA                               |
| GPU                      | NVIDIA Quadro M1200                |
| Selection metric         | Validation ROC-AUC                 |
| Early stopping           | Not used                           |

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

## 6. Partial Fine-Tuning Strategy

Run 01 used partial fine-tuning rather than a completely frozen ResNet-18 backbone.

The following modules remained frozen:

```text
conv1
bn1
layer1
layer2
layer3
```

The following modules were trainable:

```text
layer4
fc
```

This resulted in:

```text
Total parameters      = 11,209,409
Trainable parameters  =  8,426,625
Frozen parameters     =  2,782,784
```

Therefore, approximately 75.17% of the model parameters were trainable while the earlier feature-extraction layers remained frozen.

The classification head used the final ResNet-18 feature representation:

```text
ResNet-18 layer4 output
        ↓
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
1 binary logit
```

---

## 7. Training and Model Selection

The model was trained for a maximum of **100 epochs**.

Early stopping was **not used** in Run 01. Instead, the complete 100-epoch training history was retained and the checkpoint with the highest validation ROC-AUC was selected.

The highest validation ROC-AUC occurred at Epoch 96.

| Parameter               |     Result |
| ----------------------- | ---------: |
| Maximum epochs          |        100 |
| Best epoch              |     **96** |
| Best validation ROC-AUC | **0.9432** |

The training history demonstrates a substantial divergence between training and validation performance.

Training performance became extremely high during the later epochs, with training ROC-AUC reaching approximately 1.00 and training accuracy frequently approaching or reaching 100%. Validation ROC-AUC, however, remained substantially lower and fluctuated considerably across epochs.

For example:

```text
Epoch 024
Training AUC = 1.0000
Validation AUC = 0.8788

Epoch 049
Training AUC = 1.0000
Validation AUC = 0.9129

Epoch 074
Training AUC = 1.0000
Validation AUC = 0.9242

Epoch 086
Training AUC = 0.9969
Validation AUC = 0.9091

Epoch 096
Training AUC = 0.9888
Validation AUC = 0.9432
```

This indicates that the model had already achieved near-perfect fitting of the training data substantially before the end of training, while validation performance continued to fluctuate.

Importantly, **the best validation ROC-AUC was still obtained at Epoch 96**. Therefore, the Run 01 result is reported according to the predefined model-selection rule rather than selecting an epoch retrospectively based on the observed training overfitting.

---

## 8. Observed Training Behavior

The training history provides evidence of overfitting during the later stages of training.

Training loss decreased to very low values and training ROC-AUC approached 1.00, while validation performance did not improve consistently.

The later epochs illustrate this behavior particularly clearly:

```text
Epoch 073
Training Loss = 0.0087
Training AUC  = 1.0000
Validation AUC = 0.9129

Epoch 074
Training Loss = 0.0068
Training AUC  = 1.0000
Validation AUC = 0.9242

Epoch 075
Training Loss = 0.0066
Training AUC  = 1.0000
Validation AUC = 0.9205

Epoch 076
Training Loss = 0.0077
Training AUC  = 1.0000
Validation AUC = 0.9280
```

By the final epoch:

```text
Epoch 100
Training Loss = 0.0647
Training AUC  = 1.0000
Validation Loss = 0.6393
Validation AUC  = 0.8977
```

The model therefore showed a strong tendency to fit the training data while validation performance remained unstable.

This observation is important for subsequent experiments. Run 01 was allowed to complete 100 epochs specifically so that the complete training dynamics could be observed rather than imposing an early stopping rule after the fact.

---

## 9. Validation Results

The selected model was the checkpoint from **Epoch 96**, corresponding to the highest validation ROC-AUC of 0.9432.

The selected model was evaluated on the 34-image validation set using a fixed threshold of 0.5.

| Metric            |     Result |
| ----------------- | ---------: |
| ROC-AUC           | **0.9432** |
| PR-AUC            | **0.9155** |
| Accuracy          | **0.7353** |
| Balanced Accuracy | **0.7955** |
| Sensitivity       | **1.0000** |
| Specificity       | **0.5909** |
| Precision         | **0.5714** |
| NPV               | **1.0000** |
| F1-score          | **0.7273** |
| MCC               | **0.5811** |

The ROC-AUC of 0.9432 indicates strong ranking performance on the validation set. At the fixed diagnostic threshold of 0.5, the model achieved 100% sensitivity and 59.09% specificity.

Because the validation set contains only 34 images, these metrics should be interpreted as validation results for this predefined split rather than as estimates of general clinical performance.

---

## 10. Confusion Matrix

|                    | Predicted Healthy | Predicted Sick |
| ------------------ | ----------------: | -------------: |
| **Actual Healthy** |                13 |              9 |
| **Actual Sick**    |                 0 |             12 |

Therefore:

* TN = 13
* FP = 9
* FN = 0
* TP = 12

At the 0.5 threshold, all 12 Sick validation images were correctly identified, while 9 of the 22 Healthy validation images were classified as Sick.

---

## 11. Test Set Status

The test set was **not loaded or used** during Run 01.

It was not used for:

* preprocessing
* training
* model selection
* hyperparameter selection
* threshold selection
* evaluation

The independent test set remains preserved for later final evaluation.

---

## 12. Outputs

```text
Run_01_Partial_FineTuning_Baseline/

├── README.md
│
├── results/
│   ├── validation_metrics.json
│   ├── validation_roc_data.json
│   ├── training_history.json
│   ├── training_summary.json
│   └── model_summary.txt
│
├── checkpoints/
│   └── resnet18_partial_finetuning_best.pth
│
└── plots/
    ├── training_validation_loss.png
    ├── training_validation_accuracy.png
    ├── training_validation_auc.png
    └── validation_roc_curve.png
```

---

## 13. Final Summary

* Training images: **163**
* Validation images: **34**
* Test images: **37, not used**
* Model: **ImageNet-pretrained ResNet-18 with partial fine-tuning**
* Frozen modules: **conv1, bn1, layer1, layer2, layer3**
* Trainable modules: **layer4, fc**
* Classification head: **512 → 64 → 1**
* Total parameters: **11,209,409**
* Trainable parameters: **8,426,625**
* Maximum epochs: **100**
* Best epoch: **96**
* Best validation ROC-AUC: **0.9432**
* Validation PR-AUC: **0.9155**
* Validation accuracy: **73.53%**
* Balanced accuracy: **79.55%**
* Sensitivity: **100.00%**
* Specificity: **59.09%**
* Precision: **57.14%**
* F1-score: **72.73%**
* MCC: **0.5811**
* Classification threshold: **0.5**
* Early stopping: **Not used**

Run 01 establishes the partial fine-tuning baseline for EXP03 using an ImageNet-pretrained ResNet-18 with `layer4` and the classification head trainable. The model achieved a maximum validation ROC-AUC of **0.9432 at Epoch 96**.

The training history shows substantial overfitting, with training performance reaching nearly perfect levels while validation performance remained considerably lower and fluctuated across epochs. Nevertheless, the predefined selection rule selected Epoch 96 because it produced the highest validation ROC-AUC observed during the complete 100-epoch run.

The test set remained untouched and is preserved for later final evaluation.