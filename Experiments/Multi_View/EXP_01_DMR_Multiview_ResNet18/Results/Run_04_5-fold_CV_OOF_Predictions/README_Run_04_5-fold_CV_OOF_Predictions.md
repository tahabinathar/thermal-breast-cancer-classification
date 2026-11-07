# EXP_01_DMR_Multiview_ResNet18
# Run_04_5-fold_CV_OOF_Predictions

## Objective

Generate patient-level out-of-fold (OOF) predictions for the selected Baseline fusion-head configuration using five-fold patient-level cross-validation on the development dataset.

Run_04 uses the fixed training duration of 82 epochs selected in Run_03. A fresh model is trained independently for each fold, and predictions are generated for the corresponding held-out validation patients.

The resulting OOF predictions are used to evaluate threshold-independent classification performance and will support subsequent threshold-selection analysis.

## Shared Configuration

- **Backbone:** Frozen ImageNet-pretrained ResNet-18
- **Input views:** Anterior, left-oblique and right-oblique
- **Feature extraction:** 512 features per view
- **Feature fusion:** Concatenation of three feature vectors (1536 features total)
- **Fusion head:** A: Baseline (1536 → 64 → 1)
- **Dropout:** 0.3
- **Weight decay:** 0
- **Trainable parameters:** 98,433
- **Input size:** 224 × 224 pixels
- **Batch size:** 16
- **Fixed training epochs:** 82
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Loss function:** Class-weighted binary cross-entropy with logits
- **Random seed:** 10, reset for each fold
- **Cross-validation:** 5-fold stratified patient-level cross-validation
- **Development dataset:** 165 patients (101 Healthy, 64 Sick)
- **Training per fold:** 132 patients
- **Validation per fold:** 33 patients
- **Threshold selection:** Not performed
- **Test set used:** No

The same five patient-level fold assignments from Run_02 were reused. Each fold was trained independently using its corresponding training patients, while the held-out validation patients were used exclusively for prediction generation.

Temperature normalization was calculated using training images within each fold. The independent test set was not loaded or evaluated.

## Fixed-Epoch Training

Run_03 selected the Baseline fusion-head configuration at epoch 82 based on the highest mean validation ROC-AUC across five folds.

Run_04 used this fixed training duration without further epoch selection.

For each fold:

1. A fresh ImageNet-pretrained ResNet-18 model was initialized.
2. The backbone parameters were frozen.
3. Features from the three input views were concatenated into a 1536-dimensional representation.
4. The Baseline classification head was trained for exactly 82 epochs using the corresponding training patients.
5. The trained model generated predicted probabilities for the held-out validation patients.
6. The resulting predictions were stored with patient identifiers, ground-truth labels and fold assignments.

Class weights were calculated separately for each training fold to account for class imbalance.

No early stopping, additional epoch selection or classification-threshold optimization was performed.

## Out-of-Fold Prediction Generation

Each of the five models generated predictions for 33 held-out validation patients.

The predictions were combined into a single OOF dataset containing 165 patient-level predictions.

| OOF coverage check | Result |
|---|---:|
| Development patients | 165 |
| Generated OOF predictions | 165 |
| Missing predictions | 0 |
| Missing fold assignments | 0 |
| Fixed training epochs | 82 |
| Number of folds | 5 |

Every development patient received one prediction from a model that was not trained on that patient's data.

The combined predictions were saved in `oof_predictions.csv` for subsequent development-stage analysis.

## Fold-Level Results

The ROC-AUC and Average Precision (AP) were calculated separately for each validation fold using predicted probabilities.

| Fold | Validation patients | ROC-AUC | Average Precision |
|---:|---:|---:|---:|
| 1 | 33 | 0.8373 | 0.7449 |
| 2 | 33 | 0.8808 | 0.8353 |
| 3 | 33 | 0.9615 | 0.9488 |
| 4 | 33 | 0.8538 | 0.8479 |
| 5 | 33 | 0.8962 | 0.8481 |
| **Mean** | **33** | **0.8859** | **0.8450** |

The mean fold ROC-AUC was 0.8859, matching the selected Baseline configuration's mean validation ROC-AUC reported in Run_03.

Fold 3 achieved the highest ROC-AUC (0.9615), while Fold 1 achieved the lowest ROC-AUC (0.8373).

## Overall OOF Performance

The 165 held-out predictions were pooled to calculate overall threshold-independent performance.

| Metric | Overall OOF result |
|---|---:|
| ROC-AUC | **0.8639** |
| Average Precision (AP) | **0.8018** |
| Evaluated patients | 165 |

The overall OOF ROC-AUC was 0.8639, while the overall Average Precision was 0.8018.

These metrics were calculated using the pooled patient-level probabilities without applying a classification threshold.

The overall OOF ROC-AUC differs from the mean fold ROC-AUC because the two calculations use different aggregation methods.

- **Mean fold ROC-AUC:** Arithmetic mean of the five individual validation ROC-AUC values.
- **Overall OOF ROC-AUC:** ROC-AUC calculated using all 165 pooled held-out predictions.

The pooled calculation also compares probabilities generated by different fold-specific models, which may have different probability distributions.

## Findings

- **Complete OOF coverage was achieved**, with all 165 development patients receiving exactly one held-out prediction.
- **The mean fold ROC-AUC was 0.8859**, consistent with the selected Baseline configuration from Run_03.
- **The overall OOF ROC-AUC was 0.8639**, indicating useful discrimination between Healthy and Sick patients within the development dataset.
- **The overall Average Precision was 0.8018**, summarizing precision-recall performance across classification thresholds.
- ROC-AUC varied across the five validation folds, ranging from 0.8373 to 0.9615.
- The experiment generated the patient-level predicted probabilities required for subsequent threshold-selection and classification analyses.
- No classification threshold was selected, and the independent test set remained untouched.

## Conclusion

Run_04 successfully generated 165 patient-level OOF predictions using the selected Baseline multi-view ResNet-18 configuration trained for 82 epochs across five patient-level folds.

The experiment achieved a mean fold ROC-AUC of **0.8859**, an overall OOF ROC-AUC of **0.8639**, and an overall OOF Average Precision of **0.8018**.

These results establish the threshold-independent development-stage performance of the selected Baseline configuration and provide the predicted probabilities required for subsequent classification-threshold selection.

The independent test set remains unevaluated.

## Next Step: OOF Threshold Selection

The next run will determine a classification threshold using the OOF predictions generated in Run_04.

The analysis will use `oof_predictions.csv`, containing patient identifiers, ground-truth labels and predicted probabilities.

A predefined threshold-selection criterion will be applied to the pooled OOF predictions. The selected threshold will then be used to calculate threshold-dependent classification metrics, including accuracy, sensitivity, specificity, precision, F1-score and confusion matrix.

No additional model training or independent test-set evaluation is required during threshold selection.

Following threshold selection, the final Baseline model will be trained using all 165 development patients for the selected 82 epochs. The trained model and selected classification threshold will subsequently be evaluated on the independent test set.

**Important:** The fixed training duration of 82 epochs was selected using validation performance across the same development folds. Therefore, the OOF results are suitable for development-stage analysis but should not be treated as a fully independent estimate of performance after model selection. The final independent test set remains necessary for unbiased held-out evaluation.