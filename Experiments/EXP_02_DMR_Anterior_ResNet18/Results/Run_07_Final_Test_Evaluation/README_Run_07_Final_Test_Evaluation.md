# EXP02_DMR_Anterior_ResNet18

## Run 07: Final Test Evaluation

### 1. Objective

The objective of Run 07 was to perform the final evaluation of the trained EXP02 ResNet18 model on the completely untouched independent Test dataset.

The final model trained in Run 06 was loaded without any further training or modification. The Test dataset was then preprocessed using the temperature normalization parameters calculated from the complete 197-image development dataset during Run 06.

The final classification threshold of **0.192174** selected from the development-set OOF predictions in Run 05 was applied to convert the model's predicted probabilities into binary Healthy and Sick predictions.

The final Test dataset consisted of:

* Total = 37 images
* Healthy = 24
* Sick = 13

The Test dataset was not used in any previous experiment, including model training, model configuration selection, OOF prediction generation, threshold selection, or temperature normalization range estimation.

No model configuration, threshold, or preprocessing parameter was modified based on the Test results.

---

### 2. Methodology

```text
RUN 07

│
├── Final Model
│   ├── Loaded from Run 06
│   ├── Trained on 197 development images
│   ├── ResNet18 ImageNet-pretrained backbone
│   ├── Backbone frozen during training
│   ├── Learning rate = 1e-3
│   ├── Dropout = 0.3
│   └── Final state after epoch 32
│
├── Test Dataset
│   ├── Healthy = 24 images
│   ├── Sick = 13 images
│   ├── Total = 37 images
│   └── Previously untouched
│
├── Temperature Normalization
│   ├── Range loaded from Run 06
│   ├── Calculated using development images only
│   ├── Global minimum = 22.0
│   └── Global maximum = 36.812618
│
├── Model Prediction
│   ├── Test images preprocessed
│   ├── Final ResNet18 loaded
│   └── Sigmoid probabilities generated
│
├── Classification
│   ├── Fixed threshold = 0.192174
│   ├── Probability < 0.192174 → Healthy
│   └── Probability ≥ 0.192174 → Sick
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

The final model from Run 06 was loaded directly without retraining.

The temperature normalization range stored during Run 06 was used to preprocess the Test images. The normalization range was not recalculated from the Test dataset.

The model generated a continuous sigmoid probability for each Test image.

ROC-AUC was calculated directly from these predicted probabilities.

The fixed threshold of **0.192174** selected in Run 05 from the development-set OOF predictions was then applied to generate binary predictions.

No threshold optimization was performed during Run 07.

No model selection was performed during Run 07.

---

### 3. Configuration

| Parameter                            | Value                                  |
| ------------------------------------ | -------------------------------------- |
| Experiment                           | EXP02_DMR_Anterior_ResNet18            |
| Input dataset                        | Test dataset                           |
| Test dataset                         | 37 images                              |
| Healthy                              | 24                                     |
| Sick                                 | 13                                     |
| Development dataset                  | 197 images                             |
| Final model source                   | Run 06                                 |
| Input size                           | 224 × 224 × 1 thermal input            |
| Model input                          | 224 × 224 × 3 after channel repetition |
| Training epochs                      | **32**                                 |
| Learning rate                        | 1e-3                                   |
| Dropout rate                         | 0.3                                    |
| Optimizer                            | Adam                                   |
| Batch size during training           | 16                                     |
| Class weighting                      | Balanced during training               |
| ResNet18 weights                     | ImageNet IMAGENET1K_V1                 |
| Backbone                             | Frozen during training                 |
| Trainable parameters during training | 32,897                                 |
| Total parameters                     | 11,209,409                             |
| Classification threshold             | **0.192174**                           |
| Threshold source                     | Run 05 OOF Threshold Selection         |
| Temperature range source             | Run 06 Final Training                  |
| Model training                       | Not performed                          |
| Model selection                      | None                                   |
| Threshold selection                  | None                                   |
| Test evaluation                      | Final                                  |
| Random seed                          | 10                                     |
| Device                               | CPU                                    |

The classification threshold of **0.192174** was carried forward from Run 05 and was fixed before the Test evaluation.

The Test results were not used to modify the threshold.

The ResNet18 backbone was frozen during final training in Run 06. Run 07 performed inference only and did not modify any model parameters.

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

A Test-set integrity check confirmed that exactly 37 images were loaded with the expected class distribution.

No Test images were included in the 197-image development dataset used during the previous EXP02 runs.

No Test labels were used for model training or configuration selection.

The Test dataset therefore remained completely isolated from the model-development process until this final evaluation.

---

### 5. Preprocessing

Each Test image was processed using the same fundamental preprocessing procedure used during final training.

The preprocessing sequence was:

```text
Raw thermal image

        ↓

Temperature normalization

using development-set range

        ↓

Pad to square

        ↓

Convert to 8-bit image

        ↓

Resize to 224 × 224

        ↓

Convert to tensor

        ↓

Repeat grayscale channel

to 3 channels

        ↓

ImageNet normalization

        ↓

Final tensor: 3 × 224 × 224
```

Temperature normalization used the global minimum and maximum calculated from the complete 197-image development dataset during Run 06.

The stored values were:

```text
Global minimum = 22.000000
Global maximum = 36.812618
```

The normalization range was loaded from the saved Run 06 configuration.

The normalization range was therefore not calculated from the Test images.

This ensured that the Test dataset did not contribute to the estimation of preprocessing parameters.

No augmentation was applied during Test evaluation.

The processed Test images were passed directly through the final trained ResNet18 model to obtain prediction probabilities.

---

### 6. Final Model Evaluation

The final ResNet18 model saved during Run 06 was loaded from:

```text
Run_06_Final_Training/

└── model/

    └── final_model_epoch32.pth
```

The model represents the state after exactly **32 epochs** of training on all 197 development images.

The model architecture consisted of an ImageNet-pretrained ResNet18 backbone with the original fully connected layer replaced by:

```text
Linear(512, 64)

        ↓

ReLU

        ↓

Dropout(0.3)

        ↓

Linear(64, 1)
```

The complete model contains **11,209,409 parameters**. During Run 06 final training, the ResNet18 backbone was frozen and **32,897 parameters** in the classification head were trainable.

The model architecture and learned weights were not modified during Run 07.

For each Test image, the model generated a sigmoid output representing the predicted probability of the Sick class.

Two forms of evaluation were performed.

#### Probability-based evaluation

ROC-AUC was calculated using the continuous predicted probabilities.

This metric was calculated before applying the classification threshold and therefore evaluates the model's ability to distinguish between Healthy and Sick cases across possible thresholds.

#### Threshold-based evaluation

The fixed threshold selected in Run 05 was applied:

```text
Probability < 0.192174  → Healthy

Probability ≥ 0.192174 → Sick
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

| Metric            |       Result |
| ----------------- | -----------: |
| Test images       |           37 |
| Healthy           |           24 |
| Sick              |           13 |
| ROC-AUC           |   **0.9455** |
| Accuracy          |   **0.8378** |
| Balanced Accuracy |   **0.8750** |
| Sensitivity       |   **1.0000** |
| Specificity       |   **0.7500** |
| Precision         |   **0.6842** |
| F1-score          |   **0.8125** |
| MCC               |   **0.7164** |
| Threshold         | **0.192174** |

The final confusion matrix was:

```text
                 Predicted

                 Healthy    Sick

Actual Healthy      18        6

Actual Sick          0       13
```

Therefore:

```text
TN = 18
FP = 6
FN = 0
TP = 13
```

The model correctly identified all 13 Sick images.

This resulted in a sensitivity of:

```text
Sensitivity = 13 / 13
            = 1.0000
```

The model correctly identified 18 of the 24 Healthy images.

This resulted in a specificity of:

```text
Specificity = 18 / 24
            = 0.7500
```

The overall accuracy was:

```text
Accuracy = (18 + 13) / 37

         = 31 / 37

         = 0.8378
```

The balanced accuracy was:

```text
Balanced Accuracy

= (Sensitivity + Specificity) / 2

= (1.0000 + 0.7500) / 2

= 0.8750
```

The final Test ROC-AUC was:

```text
ROC-AUC = 0.9455
```

The final Test evaluation was performed using the fixed threshold of **0.192174** selected previously from the development-set OOF predictions.

---

### 8. Final Run 07 Decision

Run 07 provides the final independent evaluation of the EXP02 ResNet18 model on the previously untouched Test dataset.

The final model was not retrained, modified, or selected during Test evaluation.

The classification threshold was not optimized using Test-set performance.

The final results are:

| Parameter                | Value        |
| ------------------------ | ------------ |
| Test dataset             | 37 images    |
| Healthy                  | 24           |
| Sick                     | 13           |
| Final model              | Run 06       |
| Training epoch           | **32**       |
| Learning rate            | 1e-3         |
| Dropout                  | 0.3          |
| Classification threshold | **0.192174** |
| **Test ROC-AUC**         | **0.9455**   |
| Test Accuracy            | **0.8378**   |
| Test Balanced Accuracy   | **0.8750**   |
| Test Sensitivity         | **1.0000**   |
| Test Specificity         | **0.7500**   |
| Test Precision           | **0.6842**   |
| Test F1-score            | **0.8125**   |
| Test MCC                 | **0.7164**   |

The final reported performance of the EXP02 ResNet18 model on the independent Test dataset is therefore:

**Test ROC-AUC = 0.9455**

Using the fixed threshold of **0.192174**, the model achieved a sensitivity of **100.00%** and a specificity of **75.00%**.

The Test results were not used to modify any model, threshold, or preprocessing configuration.

The Test dataset therefore served exclusively as the final held-out evaluation set.

---

### 9. Output Structure

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
* Confusion matrix components
* TN
* FP
* FN
* TP
* Classification threshold

#### `test_predictions.csv`

Contains the prediction results for each Test image, including:

* File path
* True label
* True class
* Predicted probability
* Predicted label
* Predicted class

The binary prediction was generated using the fixed threshold of **0.192174**.

#### `run_config.json`

Stores the Run 07 evaluation configuration, including:

* Experiment and run names
* Dataset information
* Test class distribution
* Final model source
* Input dimensions
* Training configuration
* Model architecture
* Parameter counts
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

The ROC curve visualizes classification performance across different probability thresholds.

The confusion matrix visualizes the final binary predictions obtained using the fixed threshold of **0.192174**.

---

### 10. Final Summary

* Experiment: **EXP02_DMR_Anterior_ResNet18**
* Run: **Run 07 Final Test Evaluation**
* Development images: **197**
* Test images: **37**
* Healthy Test images: **24**
* Sick Test images: **13**
* Input size: **224 × 224 × 1 thermal input**
* Model input after channel conversion: **224 × 224 × 3**
* Final model source: **Run 06**
* Training epochs: **32**
* Learning rate: **1e-3**
* Dropout: **0.3**
* Optimizer: **Adam**
* ResNet18 weights: **ImageNet IMAGENET1K_V1**
* Backbone: **Frozen during training**
* Total parameters: **11,209,409**
* Trainable parameters during training: **32,897**
* Classification threshold: **0.192174**
* Threshold source: **Run 05 OOF Threshold Selection**
* Temperature normalization source: **Run 06 Final Training**
* **Test ROC-AUC: 0.9455**
* Test Accuracy: **0.8378**
* Test Balanced Accuracy: **0.8750**
* Test Sensitivity: **1.0000**
* Test Specificity: **0.7500**
* Test Precision: **0.6842**
* Test F1-score: **0.8125**
* Test MCC: **0.7164**
* TN: **18**
* FP: **6**
* FN: **0**
* TP: **13**

Run 07 successfully completed the final independent evaluation of the EXP02 ResNet18 model using the previously untouched 37-image Test dataset.

The final model from Run 06 was evaluated without retraining or modification. The model represented the state after **32 epochs** of training on the complete 197-image development dataset.

Test images were normalized using the temperature range calculated from the 197-image development dataset, and the fixed classification threshold of **0.192174** selected in Run 05 was used for binary classification.

The final Test ROC-AUC was **0.9455**.

At the fixed threshold of **0.192174**, the model achieved **100.00% sensitivity**, **75.00% specificity**, **83.78% accuracy**, and an **F1-score of 0.8125**.

The Test dataset was used exclusively for this final evaluation and was not used for model selection, threshold selection, preprocessing-range estimation, or training.