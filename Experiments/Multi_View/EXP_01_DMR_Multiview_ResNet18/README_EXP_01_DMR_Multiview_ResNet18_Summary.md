# EXP_01_DMR_Multiview_ResNet18

## Multi-View Breast Thermography Classification Using ResNet-18

### 1. Overview

This experiment investigates multi-view breast thermography classification using a frozen ImageNet-pretrained ResNet-18 backbone and feature-level fusion.

Three thermal views are used for each patient:

- Anterior
- Left Oblique
- Right Oblique

A shared ResNet-18 extracts a 512-dimensional feature vector from each view. The three feature vectors are concatenated into a 1536-dimensional representation and passed through a trainable classification head to distinguish **Healthy** and **Sick** patients.

The experiment consists of seven sequential runs covering fusion-head comparison, cross-validation, fixed-epoch selection, out-of-fold prediction generation, classification-threshold selection, final model training, and independent test evaluation.

### 2. Dataset

The experiment uses multi-view thermal images from the DMR-IR dataset, organized at the patient level.

| Partition | Healthy | Sick | Total patients | Images |
|---|---:|---:|---:|---:|
| Train | 84 | 56 | 140 | 420 |
| Validation | 17 | 8 | 25 | 75 |
| **Development total** | **101** | **64** | **165** | **495** |
| Independent Test | 18 | 14 | 32 | 96 |
| **Overall** | **119** | **78** | **197** | **591** |

Each patient contributes three thermal images.

The Train and Validation partitions were combined for five-fold cross-validation and final training. The independent Test partition was reserved exclusively for Run_07.

Patient-level separation was maintained to prevent images from the same patient appearing in both training and evaluation partitions. A leakage check was also performed before final test evaluation.

### 3. Model Architecture

The model uses a shared frozen ResNet-18 backbone for feature extraction from all three thermal views.

**Feature-fusion pipeline:**

`Anterior (512) + Left Oblique (512) + Right Oblique (512) → Concatenation (1536) → Classification Head → Healthy/Sick Probability`

Three fusion-head configurations were initially investigated.

| Configuration | Classifier | Trainable parameters | Dropout | Weight decay |
|---|---|---:|---:|---:|
| A: Baseline | 1536 → 64 → 1 | 98,433 | 0.3 | 0 |
| B: Linear Fusion | 1536 → 1 | 1,537 | 0 | 0.001 |
| C: Small Regularized | 1536 → 16 → 1 | 24,609 | 0.5 | 0.001 |

All three configurations used the same frozen ImageNet-pretrained ResNet-18 backbone.

The **Baseline fusion head (A)** was ultimately selected for final training and independent evaluation.

### 4. Shared Experimental Configuration

| Parameter | Configuration |
|---|---|
| Backbone | ImageNet-pretrained ResNet-18 |
| Backbone training | Frozen |
| Input views | Anterior, Left Oblique, Right Oblique |
| Input size | 224 × 224 |
| Fusion method | Feature concatenation |
| Fused feature dimension | 1536 |
| Batch size | 16 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Class-weighted BCEWithLogitsLoss |
| Random seed | 10 |
| Cross-validation | 5-fold stratified patient-level |
| Development patients | 165 |
| Final selected head | A: Baseline |
| Final training duration | 82 epochs |
| Final classification threshold | 0.2041969746351242 |

Temperature normalization was calculated using training data only within each cross-validation fold. For final training, the normalization range was recalculated using the complete development dataset and subsequently reused without modification for independent test evaluation.

Preprocessing included background-mask preservation, global temperature normalization, square padding, resizing, grayscale-to-three-channel conversion, and ImageNet normalization. Training augmentation included random rotation and affine scaling.

### 5. Experimental Workflow

| Run | Name | Purpose | Outcome |
|---|---|---|---|
| Run_01 | Fusion_Head_Comparison | Compare three fusion heads on a single validation split | Initial comparison completed |
| Run_02 | Five_Fold_Cross_Validation | Evaluate all three heads across five patient-level folds | CV performance established |
| Run_03 | Fixed_Epoch_Selection | Select training duration using mean validation ROC-AUC | Baseline selected at epoch 82 |
| Run_04 | 5-fold_CV_OOF_Predictions | Generate held-out predictions for all development patients | 165 OOF predictions generated |
| Run_05 | OOF_Threshold_Selection | Select classification threshold using OOF predictions | Threshold = 0.204197 |
| Run_06 | Final_Training | Train selected model on all development patients | Final model trained for 82 epochs |
| Run_07 | Final_Test_Evaluation | Evaluate the final model on independent patients | Test ROC-AUC = 0.9286 |

### 6. Run_01: Fusion-Head Comparison

Run_01 compared the three fusion-head configurations using the original Train and Validation partitions.

The validation set contained 25 patients: 17 Healthy and 8 Sick.

| Metric | A: Baseline | B: Linear | C: Small |
|---|---:|---:|---:|
| Best ROC-AUC | 0.7132 | 0.7132 | 0.7132 |
| Average Precision | **0.6220** | 0.6115 | 0.5344 |
| Accuracy | 0.6000 | 0.6800 | 0.6800 |
| Balanced Accuracy | 0.5404 | **0.6324** | 0.5331 |
| Sensitivity | 0.3750 | **0.5000** | 0.1250 |
| Specificity | 0.7059 | 0.7647 | **0.9412** |
| F1-score | 0.3750 | **0.5000** | 0.2000 |

All three heads achieved the same best validation ROC-AUC of 0.7132.

Linear Fusion provided the strongest threshold-based balance on this split, while Baseline achieved the highest Average Precision.

Because the validation set contained only 25 patients, further evaluation using patient-level cross-validation was performed.

### 7. Run_02: Five-Fold Cross-Validation

Run_02 evaluated all three fusion-head configurations using five-fold stratified patient-level cross-validation on the complete development dataset.

Each fold contained 132 training patients and 33 validation patients.

Results represent the mean ± population standard deviation across five folds at the final training epoch.

| Metric | A: Baseline | B: Linear | C: Small |
|---|---:|---:|---:|
| ROC-AUC | 0.8721 ± 0.0442 | **0.8773 ± 0.0503** | 0.8666 ± 0.0391 |
| Average Precision | 0.8303 ± 0.0561 | **0.8388 ± 0.0682** | 0.8187 ± 0.0579 |
| Accuracy | 0.7697 ± 0.0166 | 0.7818 ± 0.0254 | **0.7879 ± 0.0371** |
| Balanced Accuracy | 0.7384 ± 0.0271 | 0.7592 ± 0.0348 | **0.7756 ± 0.0237** |
| Sensitivity | 0.5949 ± 0.1222 | 0.6564 ± 0.1388 | **0.7192 ± 0.0661** |
| Specificity | **0.8819 ± 0.0815** | 0.8619 ± 0.0948 | 0.8319 ± 0.0968 |
| F1-score | 0.6612 ± 0.0603 | 0.6947 ± 0.0563 | **0.7258 ± 0.0265** |
| MCC | 0.5177 ± 0.0326 | 0.5480 ± 0.0640 | **0.5634 ± 0.0825** |

Linear Fusion achieved the highest mean ROC-AUC and Average Precision.

Small Regularized achieved the highest mean sensitivity, balanced accuracy, F1-score, and MCC at the fixed classification threshold of 0.5.

Baseline achieved the highest mean specificity.

No configuration dominated all evaluation metrics.

### 8. Run_03: Fixed-Epoch Selection

Run_03 selected a fixed training duration for each fusion-head configuration using mean validation ROC-AUC across five folds.

| Configuration | Selected epoch | Mean ROC-AUC | ROC-AUC SD |
|---|---:|---:|---:|
| **A: Baseline** | **82** | **0.8859** | 0.0481 |
| B: Linear Fusion | 100 | 0.8812 | 0.0458 |
| C: Small Regularized | 98 | 0.8797 | 0.0426 |

Baseline achieved the highest mean validation ROC-AUC of 0.8859 at epoch 82.

Consequently, **Baseline was selected as the final fusion-head configuration**, with 82 epochs established as its fixed training duration.

The differences between configurations were small and were not interpreted as evidence of statistical superiority.

### 9. Run_04: Out-of-Fold Prediction Generation

Run_04 generated patient-level OOF predictions for the selected Baseline model using five-fold cross-validation and the fixed training duration of 82 epochs.

Each fold used a freshly initialized model trained only on its corresponding training patients.

Every development patient received exactly one prediction from a model that had not been trained on that patient.

**OOF coverage:**

| Parameter | Result |
|---|---:|
| Development patients | 165 |
| Generated OOF predictions | 165 |
| Missing predictions | 0 |
| Cross-validation folds | 5 |
| Fixed training epochs | 82 |

**Fold-level performance:**

| Fold | ROC-AUC | Average Precision |
|---|---:|---:|
| 1 | 0.8373 | 0.7449 |
| 2 | 0.8808 | 0.8353 |
| 3 | 0.9615 | 0.9488 |
| 4 | 0.8538 | 0.8479 |
| 5 | 0.8962 | 0.8481 |
| **Mean** | **0.8859** | **0.8450** |

**Pooled OOF performance:**

| Metric | Result |
|---|---:|
| ROC-AUC | **0.8639** |
| Average Precision | **0.8018** |
| Evaluated patients | 165 |

The pooled OOF ROC-AUC differs from the mean fold ROC-AUC because the pooled calculation evaluates all 165 probabilities together, including comparisons between predictions from different fold-specific models.

The OOF predictions were saved for subsequent threshold selection.

### 10. Run_05: OOF Threshold Selection

Run_05 selected a classification threshold using the 165 pooled OOF predictions generated in Run_04.

Every unique OOF probability was evaluated as a candidate threshold.

The predefined constraints were:

- Sensitivity > 0.91
- Specificity > 0.60

Among valid thresholds, the highest sensitivity was preferred, followed by highest specificity and then highest threshold.

Six candidate thresholds satisfied both constraints.

**Selected operating point:**

| Metric | Result |
|---|---:|
| **Classification threshold** | **0.204197** |
| Sensitivity | 0.9219 |
| Specificity | 0.6535 |
| Accuracy | 0.7576 |
| Balanced Accuracy | 0.7877 |
| Precision | 0.6277 |
| F1-score | 0.7468 |

**OOF confusion matrix:**

| Actual / Predicted | Healthy | Sick |
|---|---:|---:|
| Healthy | 66 | 35 |
| Sick | 5 | 59 |

The selected threshold correctly identified 59 of 64 Sick patients and 66 of 101 Healthy patients.

The threshold was selected exclusively using development-stage predictions. No independent Test data were used.

### 11. Run_06: Final Model Training

Run_06 trained a single final Baseline multi-view ResNet-18 model using all 165 development patients and 495 thermal images.

The selected architecture and fixed training duration were retained without additional model selection.

**Final training configuration:**

| Parameter | Value |
|---|---|
| Development patients | 165 |
| Healthy / Sick | 101 / 64 |
| Training images | 495 |
| Fusion head | A: Baseline |
| Total parameters | 11,274,945 |
| Trainable parameters | 98,433 |
| Frozen backbone | ResNet-18 |
| Fixed epochs | 82 |
| Learning rate | 0.001 |
| Batch size | 16 |
| Training device | NVIDIA Quadro M1200 |
| Selected threshold | 0.2041969746351242 |

The development-only global temperature normalization range was:

| Parameter | Value |
|---|---:|
| Global minimum | 19.7276096344 |
| Global maximum | 36.8400001526 |

Class weights were calculated using the complete development dataset:

- Healthy: 0.816832
- Sick: 1.289063

**Training loss:**

| Epoch | Loss |
|---|---:|
| 1 | 0.762557 |
| 20 | 0.366388 |
| 40 | 0.280184 |
| 60 | 0.281807 |
| 80 | 0.220212 |
| **82 (Final)** | **0.226719** |

Training loss generally decreased over the predefined 82 epochs.

The final model was saved as:

`Run_06_Final_Training/model/final_model_epoch82.pth`

The checkpoint, preprocessing parameters, training history, and development manifest were preserved for final evaluation.

No independent Test data were used during final training.

### 12. Run_07: Independent Test Evaluation

Run_07 evaluated the final model on the independent Test dataset using the fixed classification threshold selected in Run_05.

The Test dataset contained 32 patients: 18 Healthy and 14 Sick.

Each patient contributed three thermal views, giving 96 images in total.

The final checkpoint from Run_06 and the saved development-only temperature normalization parameters were used without modification.

**Final independent Test results:**

| Metric | Result |
|---|---:|
| **ROC-AUC** | **0.9286** |
| **Average Precision** | **0.9066** |
| Accuracy | 0.6875 |
| Balanced Accuracy | 0.7222 |
| Sensitivity | 1.0000 |
| Specificity | 0.4444 |
| Precision | 0.5833 |
| F1-score | 0.7368 |
| MCC | 0.5092 |
| Classification threshold | 0.204197 |

**Test confusion matrix:**

| Actual / Predicted | Healthy | Sick |
|---|---:|---:|
| Healthy | 8 | 10 |
| Sick | 0 | 14 |

The model correctly identified all 14 Sick patients, achieving **100% sensitivity** with zero false negatives.

However, 10 of the 18 Healthy patients were incorrectly classified as Sick, resulting in **44.44% specificity**.

The ROC-AUC of 0.9286 indicates strong discrimination based on predicted probabilities, while the lower accuracy and specificity reflect the sensitivity-oriented fixed classification threshold.

No retraining, threshold adjustment, or model selection was performed using the independent Test dataset.

### 13. Overall Performance Summary

The following table summarizes the main results from model selection through final evaluation.

| Stage | Dataset | ROC-AUC | Average Precision | Sensitivity | Specificity |
|---|---|---:|---:|---:|---:|
| Run_03: Selected Baseline epoch | 5-fold validation | 0.8859 ± 0.0481 | — | — | — |
| Run_04: Pooled OOF | 165 development patients | 0.8639 | 0.8018 | — | — |
| Run_05: Selected threshold | 165 OOF predictions | 0.8639 | 0.8018 | 0.9219 | 0.6535 |
| **Run_07: Final Test** | **32 independent patients** | **0.9286** | **0.9066** | **1.0000** | **0.4444** |

Run_05 and Run_07 use the same fixed threshold of 0.204197.

The Run_05 threshold-dependent metrics were calculated on pooled OOF predictions from five fold-specific models. Run_07 evaluated a single final model trained on all development patients.

The independent Test ROC-AUC was higher than the pooled OOF ROC-AUC, but this difference should not be interpreted as proof of improved generalization because the predictions came from different models and patient cohorts.

The independent Test specificity was lower than the OOF specificity, demonstrating that the development-selected operating point did not maintain its specificity requirement on the held-out Test cohort.

### 14. Key Findings

1. **Multi-view feature fusion achieved strong discrimination.** The selected Baseline model achieved an independent Test ROC-AUC of 0.9286 and Average Precision of 0.9066.

2. **Fusion-head performance depended on the evaluation metric.** Linear Fusion achieved the highest final-epoch mean ROC-AUC in Run_02, while Small Regularized achieved the highest mean sensitivity and balanced accuracy. Baseline achieved the highest mean ROC-AUC under the fixed-epoch selection procedure in Run_03.

3. **Baseline was selected at epoch 82.** The selected training duration was determined using development-only cross-validation and retained for OOF prediction generation and final model training.

4. **Complete patient-level OOF coverage was achieved.** All 165 development patients received one held-out prediction, yielding a pooled ROC-AUC of 0.8639 and Average Precision of 0.8018.

5. **The OOF-derived threshold prioritized sensitivity.** The selected threshold of 0.204197 achieved 92.19% sensitivity and 65.35% specificity on the development OOF predictions.

6. **The final model detected all Sick Test patients.** All 14 Sick patients were correctly classified, resulting in zero false negatives and 100% sensitivity.

7. **Independent Test specificity was limited.** Ten Healthy patients were incorrectly classified as Sick, producing a specificity of 44.44% and accuracy of 68.75%.

8. **The independent Test set was preserved for final evaluation.** No Test data were used during model selection, epoch selection, threshold optimization, or final training.

### 15. Limitations

The experiment has several limitations.

**Limited dataset size:** The final independent Test cohort contained only 32 patients, including 14 Sick patients. Performance estimates are therefore sensitive to individual classification outcomes.

**Sensitivity-specificity trade-off:** The selected threshold produced perfect sensitivity on the Test set but relatively low specificity, resulting in a high false-positive rate.

**Threshold transfer:** The classification threshold was selected using OOF predictions from five fold-specific models and subsequently applied to a single model trained on all development patients. Differences in probability distributions may affect threshold performance.

**Model-selection dependence:** The fixed training duration was selected using the same development folds subsequently used for OOF evaluation. Consequently, development-stage performance estimates should not be considered fully independent of model selection.

**Generalization scope:** The experiment evaluates performance on a held-out partition of the DMR-IR dataset. It does not establish cross-dataset, cross-device, or clinical generalization.

Further investigation using larger independent cohorts, external datasets, probability calibration, and threshold robustness analysis would be valuable.

### 16. Conclusion

This experiment developed and evaluated a multi-view breast thermography classification pipeline using a shared frozen ImageNet-pretrained ResNet-18 backbone and feature-level fusion.

Three fusion-head configurations were investigated through an initial validation comparison and five-fold patient-level cross-validation. The Baseline fusion head was selected at epoch 82 using mean validation ROC-AUC.

The selected model achieved a pooled OOF ROC-AUC of 0.8639 and Average Precision of 0.8018. Constraint-based OOF threshold selection identified a classification threshold of 0.204197, prioritizing sensitivity while maintaining predefined development-stage specificity requirements.

The final Baseline model was trained on all 165 development patients for 82 epochs and subsequently evaluated on 32 independent Test patients.

The final model achieved an **independent Test ROC-AUC of 0.9286, Average Precision of 0.9066, and sensitivity of 100%**. However, specificity was limited to **44.44%**, with 10 false-positive predictions among 18 Healthy patients.

Overall, the results demonstrate strong patient-level discrimination using multi-view thermal feature fusion, while highlighting the need to improve specificity and threshold robustness before considering clinical application.

**Experiment status: Completed — seven runs, including final independent Test evaluation.**