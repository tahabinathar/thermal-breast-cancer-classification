# EXP_01_DMR_Multiview_ResNet18

# Run_03_Fixed_Epoch_Selection

## Objective

Select a fixed training epoch for each fusion-head configuration using five-fold patient-level stratified cross-validation on the development dataset. The selected epochs are intended for subsequent out-of-fold (OOF) prediction generation and model evaluation.

Run_03 compares candidate training epochs for three fusion-head configurations using the same frozen multi-view ResNet-18 feature extraction pipeline. Each candidate epoch is evaluated across the same five patient-level folds, allowing training duration to be selected using mean validation ROC-AUC.

## Shared Configuration

- **Backbone:** Frozen ImageNet-pretrained ResNet-18
- **Input views:** Anterior, left-oblique and right-oblique
- **Feature extraction:** 512 features per view
- **Feature fusion:** Concatenation of three feature vectors (1536 features total)
- **Input size:** 224 × 224 pixels
- **Batch size:** 16
- **Maximum candidate epoch:** 100
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Random seed:** 10, reset for each fold
- **Cross-validation:** 5-fold stratified patient-level cross-validation
- **Development dataset:** 165 patients (101 Healthy, 64 Sick)
- **Training per fold:** 132 patients
- **Validation per fold:** 33 patients
- **Evaluation threshold:** 0.5
- **Model-selection metric:** Mean validation ROC-AUC across five folds
- **Test set used:** No

The development dataset was evaluated using five patient-level folds. The same fold assignments were used to compare the fusion-head configurations and their candidate epochs.

The independent test set was not used during candidate-epoch evaluation or selection.

## Fusion-Head Configurations

| Configuration | Classifier | Dropout | Weight decay |
|---|---|---:|---:|
| A: Baseline | 1536 → 64 → 1 | 0.3 | 0 |
| B: Linear Fusion | 1536 → 1 | 0 | 0.001 |
| C: Small Regularized | 1536 → 16 → 1 | 0.5 | 0.001 |

All configurations used the same frozen ResNet-18 backbone. The classification head was trained on the concatenated features from the three views.

## Candidate-Epoch Selection

Candidate training epochs were evaluated using the validation ROC-AUC from each of the five folds. For each fusion-head configuration, the mean validation ROC-AUC was calculated across the folds.

The selected epoch was the candidate with the highest mean validation ROC-AUC. If candidates had identical mean ROC-AUC values, the earlier epoch was preferred.

The selected epochs were:

| Configuration | Selected epoch | Mean ROC-AUC | ROC-AUC SD |
|---|---:|---:|---:|
| A: Baseline | 82 | 0.8859 | 0.0481 |
| B: Linear Fusion | 100 | 0.8812 | 0.0458 |
| C: Small Regularized | 98 | 0.8797 | 0.0426 |

Results are reported as mean ± population standard deviation across the five validation folds.

## Fold-Level ROC-AUC at Selected Epochs

| Fold | A: Baseline (Epoch 82) | B: Linear Fusion (Epoch 100) | C: Small Regularized (Epoch 98) |
|---:|---:|---:|---:|
| 1 | 0.8373 | 0.8214 | 0.8254 |
| 2 | 0.8808 | 0.8808 | 0.8846 |
| 3 | 0.9615 | 0.9423 | 0.9423 |
| 4 | 0.8538 | 0.8577 | 0.8615 |
| 5 | 0.8962 | 0.9038 | 0.8846 |
| **Mean** | **0.8859** | **0.8812** | **0.8797** |
| **SD** | **0.0481** | **0.0458** | **0.0426** |

## Findings

- **Baseline achieved the highest mean validation ROC-AUC (0.8859)** at epoch 82, making it the selected configuration under the predefined selection criterion.
- **Linear Fusion achieved a mean ROC-AUC of 0.8812** at epoch 100.
- **Small Regularized achieved a mean ROC-AUC of 0.8797** at epoch 98.
- The differences between the mean ROC-AUC values were small. These results do not establish statistically significant differences between configurations.
- The selected epochs define a fixed training duration for each configuration in subsequent development-stage evaluation. They do not represent independent test-set performance.

## Conclusion

Run_03 identified a candidate fixed training epoch for each fusion-head configuration using mean validation ROC-AUC across five patient-level folds. Baseline was selected at epoch 82, Linear Fusion at epoch 100 and Small Regularized at epoch 98.

These selections establish the training durations for the next stage. The independent test set remains unevaluated and must not be used for further model or threshold selection.

## Next Step: Out-of-Fold Prediction Generation

The next run will generate out-of-fold predictions using the selected fixed epoch for each fusion-head configuration.

For each fold, the classification head will be trained on the corresponding training patients for its selected number of epochs, then used to generate predictions for that fold's held-out validation patients. Each development patient should receive a prediction from a model that was not trained on that patient's data.

The OOF predictions will be retained alongside patient identifiers, ground-truth labels and predicted probabilities. These predictions will support subsequent threshold-selection and classification analyses using development data only.

The independent test set will remain untouched until the final held-out evaluation.

**Important:** The selected epochs were determined using validation performance across the same five folds. OOF predictions generated using these selections are useful for downstream development analyses, but the OOF results should not be treated as a fully independent estimate of performance after model selection. The final independent test set remains necessary for the final evaluation.