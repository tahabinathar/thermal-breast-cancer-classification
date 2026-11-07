# EXP_01_DMR_Multiview_ResNet18
# Run_05_OOF_Threshold_Selection

## Objective

Select a classification threshold for the Baseline multi-view ResNet-18 model using out-of-fold (OOF) predictions generated in Run_04.

Run_05 applies constraint-based threshold selection to the 165 development patients. The objective is to identify a threshold that achieves the highest possible sensitivity while satisfying predefined minimum sensitivity and specificity requirements.

The selected threshold is intended for subsequent classification and independent test-set evaluation.

## Shared Configuration

- **Backbone:** Frozen ImageNet-pretrained ResNet-18
- **Input views:** Anterior, left-oblique and right-oblique
- **Feature fusion:** Concatenation of three feature vectors (1536 features total)
- **Fusion head:** A: Baseline (1536 → 64 → 1)
- **Fixed training epochs:** 82
- **Random seed:** 10
- **Cross-validation:** 5-fold stratified patient-level cross-validation
- **Development dataset:** 165 patients (101 Healthy, 64 Sick)
- **Input:** Patient-level OOF predictions from Run_04
- **Threshold candidates:** All unique OOF predicted probabilities
- **Minimum sensitivity:** > 0.91
- **Minimum specificity:** > 0.60
- **Selection criterion:** Highest sensitivity
- **First tie-breaker:** Highest specificity
- **Final tie-breaker:** Highest threshold
- **Model training:** Not performed
- **Test set used:** No

The OOF predictions were generated using five independently trained models, each evaluated on its corresponding held-out validation patients.

Run_05 used these predictions without additional model training or changes to the previously selected fusion-head configuration and training duration.

## Threshold-Selection Method

Constraint-based threshold selection was performed using the pooled OOF predictions from all 165 development patients.

Every unique predicted probability was evaluated as a candidate classification threshold. Patients with predicted probabilities greater than or equal to the candidate threshold were classified as Sick, while the remaining patients were classified as Healthy.

For each candidate threshold, sensitivity, specificity, accuracy, balanced accuracy, precision, F1-score and confusion-matrix values were calculated.

A candidate threshold was considered valid only when both conditions were satisfied:

- **Sensitivity > 0.91**
- **Specificity > 0.60**

Among the valid thresholds, the threshold achieving the highest sensitivity was selected. If multiple thresholds achieved identical sensitivity, the threshold with the highest specificity was preferred. If both metrics were identical, the highest threshold was selected.

No Youden's J statistic, F1-score maximization or test-set performance was used for threshold selection.

## OOF Prediction Summary

The OOF predictions generated in Run_04 were loaded and validated before threshold analysis.

| Parameter | Result |
|---|---:|
| Development patients | 165 |
| Healthy patients | 101 |
| Sick patients | 64 |
| Cross-validation folds | 5 |
| Overall OOF ROC-AUC | 0.8639 |
| Overall OOF Average Precision | 0.8018 |
| Valid candidate thresholds | 6 |

All 165 OOF predictions were included in the analysis.

The overall OOF ROC-AUC and Average Precision were calculated using predicted probabilities and were independent of the selected classification threshold.

## Selected Classification Threshold

The constraint-based selection procedure identified six candidate thresholds satisfying both minimum performance requirements.

The selected threshold was:

| Parameter | Selected result |
|---|---:|
| **Classification threshold** | **0.204197** |
| Sensitivity | 0.9219 |
| Specificity | 0.6535 |
| Accuracy | 0.7576 |
| Balanced accuracy | 0.7877 |
| Precision | 0.6277 |
| F1-score | 0.7468 |

The selected threshold achieved a sensitivity of 92.19% and specificity of 65.35%, satisfying both predefined constraints.

The threshold was selected using the highest-sensitivity criterion among the valid candidates, with specificity and threshold value used as tie-breakers.

## Confusion Matrix at Selected Threshold

The classification results at the selected threshold were:

| Actual class | Predicted Healthy | Predicted Sick |
|---|---:|---:|
| Healthy | 66 | 35 |
| Sick | 5 | 59 |

The corresponding confusion-matrix values were:

- **True Negatives (TN):** 66
- **False Positives (FP):** 35
- **False Negatives (FN):** 5
- **True Positives (TP):** 59

Of the 64 Sick patients, 59 were correctly classified and five were incorrectly classified as Healthy.

Of the 101 Healthy patients, 66 were correctly classified and 35 were incorrectly classified as Sick.

The selected operating point prioritizes sensitivity, reducing missed Sick cases while accepting a higher number of false-positive classifications.

## Generated Outputs

Run_05 generated the following files:

| Output | Description |
|---|---|
| `threshold_analysis.csv` | Performance metrics and constraint-validation status for every candidate threshold |
| `selected_threshold.json` | Selected threshold, selection criteria, classification metrics and confusion-matrix values |
| `overall_summary.json` | OOF dataset summary, threshold-selection settings and selected result |
| `sensitivity_specificity_vs_threshold.png` | Sensitivity and specificity across candidate thresholds |
| `valid_thresholds.png` | Sensitivity and specificity for thresholds satisfying both constraints |

The results were saved under `Results/Run_05_OOF_Threshold_Selection/`, with separate `results` and `plots` directories.

## Findings

- **The selected classification threshold was 0.204197**, determined using the predefined constraint-based selection method.
- **Six candidate thresholds satisfied both sensitivity and specificity requirements.**
- **Sensitivity reached 0.9219**, exceeding the required minimum of 0.91.
- **Specificity reached 0.6535**, exceeding the required minimum of 0.60.
- The selected threshold correctly classified 59 of 64 Sick patients and 66 of 101 Healthy patients.
- The resulting accuracy was 0.7576, with a balanced accuracy of 0.7877 and an F1-score of 0.7468.
- The overall OOF ROC-AUC remained 0.8639, and Average Precision remained 0.8018, as these metrics do not depend on the selected classification threshold.
- No additional model training or independent test-set evaluation was performed.

## Conclusion

Run_05 successfully selected a classification threshold for the Baseline multi-view ResNet-18 model using pooled OOF predictions from 165 development patients.

The selected threshold of **0.204197** achieved a sensitivity of **0.9219** and specificity of **0.6535**, satisfying the predefined minimum requirements.

The selected threshold establishes the development-stage classification operating point for subsequent model evaluation.

The independent test set remains unevaluated.

## Next Step: Final Model Training

The next run will train a final Baseline multi-view ResNet-18 model using all 165 development patients.

The final training configuration will retain the previously selected architecture and fixed training duration:

- **Backbone:** Frozen ImageNet-pretrained ResNet-18
- **Fusion head:** A: Baseline (1536 → 64 → 1)
- **Training patients:** 165
- **Fixed training epochs:** 82
- **Selected classification threshold:** 0.204197

The final model will be trained using the complete development dataset, with temperature normalization parameters calculated exclusively from development data.

The trained model, preprocessing parameters and selected threshold will be saved for subsequent independent test-set evaluation.

The independent test set will not be used during final model training or for further threshold optimization.

**Important:** The selected threshold was optimized using the same OOF predictions on which its sensitivity, specificity and other classification metrics were calculated. Therefore, these threshold-dependent results represent development-stage performance and should not be interpreted as unbiased estimates of independent test performance. Additionally, the OOF predictions were generated by five separate fold-specific models, so the selected threshold may behave differently when applied to a single model trained on the full development dataset. The final independent test evaluation is necessary to assess generalization.