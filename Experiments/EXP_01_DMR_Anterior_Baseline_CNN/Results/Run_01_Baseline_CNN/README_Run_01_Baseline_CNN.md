# EXP01_DMR_Anterior_Baseline

## Run 01: Baseline CNN

## 1. Objective

Establish a baseline classification model for distinguishing **Healthy** and **Sick** breast thermography images using a custom CNN.

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
│   ├── Resize to 128×128
│   └── Add single channel
│
├── Data Augmentation
│   ├── Random rotation = 0.05
│   ├── Random zoom = 0.05
│   └── Horizontal flip = NOT USED
│
├── Baseline CNN
│   ├── Input = 128×128×1
│   ├── Conv2D blocks: 32 → 64 → 128 filters
│   ├── Global Average Pooling
│   ├── Dense = 64
│   ├── Dropout = 0.5
│   └── Sigmoid output
│
├── Training
│   ├── Adam optimizer
│   ├── Learning rate = 0.001
│   ├── Batch size = 16
│   ├── Maximum epochs = 100
│   ├── Balanced class weights
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

| Parameter        | Value                    |
| ---------------- | ------------------------ |
| Input size       | 128 × 128 × 1            |
| Batch size       | 16                       |
| Epochs           | 100                      |
| Optimizer        | Adam                     |
| Learning rate    | 0.001                    |
| Dropout          | 0.5                      |
| Loss function    | Binary cross-entropy     |
| Augmentation     | Rotation 0.05, Zoom 0.05 |
| Class weighting  | Enabled                  |
| Threshold        | 0.5                      |
| Seed             | 10                       |
| Device           | CPU                      |
| Selection metric | Validation ROC-AUC       |
| Early stopping   | Not used                 |

---

## 4. Preprocessing

The thermal TIFF images were processed using the predefined preprocessing pipeline.

For Run 01, the temperature range was calculated **only from the 163 training images** and then applied to both the training and validation sets.

```text
Thermal image
    ↓
Temperature normalization
    ↓
Pad to square
    ↓
Resize to 128×128
    ↓
Add single channel
```

The resulting model input shape was **128 × 128 × 1**.

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
| Best epoch              |     **41** |
| Best validation ROC-AUC | **0.9318** |

---

## 7. Validation Results

The selected model was evaluated on the 34-image validation set using a fixed threshold of 0.5.

| Metric      |     Result |
| ----------- | ---------: |
| ROC-AUC     | **0.9318** |
| Accuracy    | **0.7941** |
| Sensitivity | **1.0000** |
| Specificity | **0.6818** |
| Precision   | **0.6316** |
| F1-score    | **0.7742** |

---

## 8. Confusion Matrix

|                    | Predicted Healthy | Predicted Sick |
| ------------------ | ----------------: | -------------: |
| **Actual Healthy** |                15 |              7 |
| **Actual Sick**    |                 0 |             12 |

Therefore:

* TN = 15
* FP = 7
* FN = 0
* TP = 12

All 12 Sick validation images were correctly identified, while 7 Healthy images were classified as Sick.

---

## 9. Test Set Status

The test set was **not loaded or used** during Run 01.

It was not used for preprocessing, training, model selection, or evaluation.

The independent test set was preserved for later final evaluation.

---

## 10. Outputs

```text
Run_01_Baseline_CNN/
├── README.md
├── results/
│   ├── validation_metrics.json
│   ├── validation_predictions.csv
│   ├── training_history.json
│   └── run_config.json
├── checkpoints/
│   └── best_model.keras
└── plots/
    ├── training_validation_auc.png
    └── validation_roc_curve.png
```

---

## 11. Final Summary

* Training images: **163**
* Validation images: **34**
* Test images: **37, not used**
* Model: **Custom baseline CNN**
* Maximum epochs: **100**
* Best epoch: **41**
* Best validation ROC-AUC: **0.9318**
* Validation accuracy: **79.41%**
* Sensitivity: **100.00%**
* Specificity: **68.18%**
* Precision: **63.16%**
* F1-score: **77.42%**
* Classification threshold: **0.5**

Run 01 establishes the baseline performance for EXP01. The selected baseline model and its validation performance provide the reference for subsequent EXP01 experiments, including the learning-rate sweep in Run 02.