# EXP03_DMR_Anterior_ResNet18_Partial_FineTuning

## Run 07: Final Test Evaluation

### 1. Objective

---

Run 07 performs the final one-time evaluation of the EXP03 model on the independent Test set.

The model used in this run was trained in **Run 06** using all 197 development images (original Train + Validation sets). The classification threshold was fixed in **Run 05** using out-of-fold predictions from the development data.

The Test set was not used during:

* Model training
* Training epoch selection
* Threshold selection
* Model selection
* Hyperparameter tuning

The purpose of Run 07 is to obtain the final unbiased Test-set performance of the complete EXP03 pipeline.

---

### 2. Methodology

---

Run 07 uses the final model produced by **Run 06: Final Training**.

The model architecture is based on ImageNet-pretrained ResNet-18.

#### Model Configuration

* Backbone: ResNet-18
* Pretrained weights: ImageNet1K_V1
* Input size: 224 × 224
* Input channels: 3
* Frozen layers:
  * `conv1`
  * `bn1`
  * `layer1`
  * `layer2`
  * `layer3`
* Trainable layers:
  * `layer4`
  * Fully connected classifier
* Classifier:

```text
Linear(512, 64)

ReLU

Dropout(0.3)

Linear(64, 1)
```

* Total parameters: 11,209,409
* Trainable parameters: 8,426,625

#### Training Configuration

The final model was trained in Run 06 for exactly 10 epochs using all 197 development images.

```text
Layer4 learning rate: 1e-4
Classifier learning rate: 1e-3
Optimizer: Adam
Dropout: 0.3
Batch size: 16
Seed: 10
```

The 10-epoch training duration was selected through the preceding 5-fold cross-validation procedure in Run 03. It is therefore treated as a fixed training epoch selected through cross-validation, not optimized using the Test set.

#### Classification Threshold

The final classification threshold was selected in Run 05 from development-set out-of-fold predictions.

```text
Selected threshold = 0.027535
```

The Test predictions were evaluated using this fixed threshold without any further adjustment.

---

### 3. Configuration

| Parameter | Value |
|---|---|
| Experiment | EXP03_DMR_Anterior_ResNet18_Partial_FineTuning |
| Run | Run_07_Final_Test_Evaluation |
| Seed | 10 |
| Model | ResNet-18 |
| Pretrained weights | ImageNet1K_V1 |
| Input size | 224 × 224 |
| Input channels | 3 |
| Batch size | 16 |
| Final training epochs | 10 |
| Layer4 learning rate | 0.0001 |
| Classifier learning rate | 0.001 |
| Dropout | 0.3 |
| Trainable layers | Layer4 + classifier |
| Frozen layers | Conv1 + BN1 + Layer1 + Layer2 + Layer3 |
| Device | CPU |
| Training data | Train + Validation |
| Development images | 197 |
| Test images | 37 |
| Threshold | 0.027535 |
| Threshold source | Run 05 OOF Predictions |
| Model source | Run 06 Final Training |
| Test used for training | No |
| Test used for threshold selection | No |
| Test used for model selection | No |
| Test used for tuning | No |

---

### 4. Test Dataset

The independent Test set contains 37 images.

| Class | Label | Number of Images |
|---|---|---|
| Healthy | 0 | 24 |
| Sick | 1 | 13 |
| Total | | 37 |

The Test set was loaded only during Run 07 for final evaluation.

An integrity check was performed before evaluation to confirm:

```text
Total images = 37
Healthy images = 24
Sick images = 13
```

The integrity check passed.

---

### 5. Preprocessing

The Test images were processed using the same preprocessing pipeline established during development and final training.

#### Temperature Normalization

The global temperature range saved during Run 06 was used.

```text
Global minimum = 22.000000
Global maximum = 36.812618
```

These values were calculated from the development data only.

No Test-set statistics were used to determine the normalization range.

#### Processing Steps

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

No random augmentation was applied during Test evaluation.

The normalized thermal image was converted to 8-bit before resizing, consistent with the pipeline used during model development and final training.

---

### 6. Final Model Evaluation

The final model checkpoint from Run 06 was loaded without retraining.

Model:

```text
Results/Run_06_Final_Training/model/final_model_epoch10.pth
```

The fixed threshold from Run 05 was loaded:

```text
Threshold = 0.027535
```

For each Test image, the model produced a probability of the Sick class.

Classification was performed as:

```text
Probability >= 0.027535  → Sick
Probability <  0.027535  → Healthy
```

No threshold tuning was performed using Test-set predictions.

The following metrics were calculated:

* ROC-AUC
* Accuracy
* Balanced Accuracy
* Sensitivity
* Specificity
* Precision
* F1-score
* Matthews Correlation Coefficient
* Confusion Matrix

---

### 7. Results

#### Final Test Performance

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

#### Confusion Matrix

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

#### Class-wise Interpretation

For the 24 Healthy Test images:

```text
Correctly classified as Healthy: 14
Incorrectly classified as Sick: 10
```

For the 13 Sick Test images:

```text
Correctly classified as Sick: 13
Incorrectly classified as Healthy: 0
```

The resulting sensitivity was:

```text
13 / 13 = 1.0000
```

The resulting specificity was:

```text
14 / 24 = 0.5833
```

The resulting accuracy was:

```text
(14 + 13) / 37 = 0.7297
```

#### ROC-AUC

The final Test ROC-AUC was:

```text
ROC-AUC = 0.9359
```

ROC-AUC evaluates the ranking of model predictions across thresholds, while accuracy, sensitivity, specificity, precision and F1-score are calculated using the fixed threshold of 0.027535.

Therefore, the ROC-AUC should be interpreted separately from the threshold-dependent classification metrics.

---

### 8. Final Run 07 Decision

Run 07 completed the final independent evaluation of EXP03.

The final model was evaluated on all 37 Test images using:

* The model trained in Run 06
* The development-derived temperature normalization range
* The fixed training duration selected through cross-validation
* The fixed classification threshold selected from Run 05 OOF predictions

No Test-set information was used to modify the model or threshold after evaluation.

The final Test results were:

```text
ROC-AUC           = 0.9359
Accuracy          = 0.7297
Balanced Accuracy = 0.7917
Sensitivity       = 1.0000
Specificity       = 0.5833
Precision         = 0.5652
F1-score          = 0.7222
MCC               = 0.5742
```

The final Test evaluation is therefore considered complete.

No retraining, model selection or threshold tuning should be performed using the Test results.

---

### 9. Output Structure

```text
EXP_03_DMR_Anterior_ResNet18_Partial_FineTuning/
│
├── README.md
│
└── Results/
    │
    └── Run_07_Final_Test_Evaluation/
        │
        ├── config/
        │   └── run_config.json
        │
        ├── metrics/
        │   └── test_metrics.json
        │
        ├── predictions/
        │   └── test_predictions.csv
        │
        └── plots/
            ├── test_roc_curve.png
            └── test_confusion_matrix.png
```

The Run 07 outputs contain the final Test-set predictions, evaluation metrics, configuration and evaluation plots.

---

### 10. Final Summary

Run 07 performed the final independent Test evaluation of the EXP03 partially fine-tuned ResNet-18 model.

The model was trained on all 197 development images in Run 06 for the fixed 10 epochs selected through cross-validation. The classification threshold of 0.027535 was determined from development-set out-of-fold predictions in Run 05.

The independent Test set contained 37 images, including 24 Healthy and 13 Sick cases.

The model achieved a Test ROC-AUC of 0.9359. Using the pre-specified threshold of 0.027535, the model achieved:

```text
Accuracy          = 72.97%
Balanced Accuracy = 79.17%
Sensitivity       = 100.00%
Specificity       = 58.33%
Precision         = 56.52%
F1-score          = 72.22%
MCC               = 57.42%
```

The Test set was used only once for final evaluation and was not used for training, model selection, threshold selection or tuning.

Run 07 therefore represents the final reported Test performance for EXP03.
