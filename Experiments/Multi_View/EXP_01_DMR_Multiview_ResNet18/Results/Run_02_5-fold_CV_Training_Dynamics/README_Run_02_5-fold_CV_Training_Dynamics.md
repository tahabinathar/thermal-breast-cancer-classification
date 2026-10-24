# EXP_01_DMR_Multiview_ResNet18

# Run_02_Five_Fold_Cross_Validation

## Objective

Compare three fusion-head configurations for classifying Healthy and Sick cases using anterior, left-oblique and right-oblique thermal images. A shared frozen ImageNet-pretrained ResNet-18 extracts features from each view. The three feature vectors are concatenated before classification.

Run_02 uses five-fold patient-level stratified cross-validation on the development dataset to evaluate the consistency of the fusion-head configurations across different patient splits.

## Shared Configuration

- **Backbone:** Frozen ImageNet-pretrained ResNet-18
- **Input views:** Anterior, left-oblique and right-oblique
- **Feature extraction:** 512 features per view
- **Feature fusion:** Concatenation of three feature vectors (1536 features total)
- **Input size:** 224 × 224 pixels
- **Batch size:** 16
- **Maximum epochs:** 100
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Random seed:** 10, reset for each fold
- **Cross-validation:** 5-fold stratified patient-level cross-validation
- **Development dataset:** 165 patients (101 Healthy, 64 Sick)
- **Training per fold:** 132 patients
- **Validation per fold:** 33 patients
- **Evaluation threshold:** 0.5
- **Evaluation checkpoint:** Final epoch

The training and validation partitions were pooled to form the development dataset for cross-validation. Patient-level stratification was used to assign patients to folds.

The independent test set was not loaded or evaluated in Run_02.

## Fusion-Head Comparison

| Configuration | Classifier | Dropout | Weight decay |
|---|---|---:|---:|
| A: Baseline | 1536 → 64 → 1 | 0.3 | 0 |
| B: Linear Fusion | 1536 → 1 | 0 | 0.001 |
| C: Small Regularized | 1536 → 16 → 1 | 0.5 | 0.001 |

All configurations used the same frozen ResNet-18 backbone. The classification head was trained on the concatenated multi-view features.

## Cross-Validation Results

Results are reported as **mean ± population standard deviation** across the five validation folds. Threshold-dependent metrics use a fixed classification threshold of 0.5.

| Metric | A: Baseline | B: Linear Fusion | C: Small Regularized |
|---|---:|---:|---:|
| Loss | 0.6427 ± 0.1101 | **0.4704 ± 0.0540** | 0.4813 ± 0.0423 |
| ROC-AUC | 0.8721 ± 0.0442 | **0.8773 ± 0.0503** | 0.8666 ± 0.0391 |
| Average precision | 0.8303 ± 0.0561 | **0.8388 ± 0.0682** | 0.8187 ± 0.0579 |
| Accuracy | 0.7697 ± 0.0166 | 0.7818 ± 0.0254 | **0.7879 ± 0.0371** |
| Balanced accuracy | 0.7384 ± 0.0271 | 0.7592 ± 0.0348 | **0.7756 ± 0.0237** |
| Precision | **0.7918 ± 0.1314** | 0.7799 ± 0.1361 | 0.7531 ± 0.1390 |
| Sensitivity (Recall) | 0.5949 ± 0.1222 | 0.6564 ± 0.1388 | **0.7192 ± 0.0661** |
| Specificity | **0.8819 ± 0.0815** | 0.8619 ± 0.0948 | 0.8319 ± 0.0968 |
| F1-score | 0.6612 ± 0.0603 | 0.6947 ± 0.0563 | **0.7258 ± 0.0265** |
| MCC | 0.5177 ± 0.0326 | 0.5480 ± 0.0640 | **0.5634 ± 0.0825** |

Bold values indicate the best mean result for each metric. Lower loss is better; higher values are better for the other metrics.

## Fold-Level Results

### A: Baseline

| Fold | ROC-AUC | Average Precision | Accuracy | Balanced Accuracy | Sensitivity | Specificity | F1-score | MCC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8373 | 0.7753 | 0.7576 | 0.7381 | 0.6667 | 0.8095 | 0.6667 | 0.4762 |
| 2 | 0.8846 | 0.8208 | 0.7879 | 0.7577 | 0.6154 | 0.9000 | 0.6957 | 0.5480 |
| 3 | 0.9423 | 0.9217 | 0.7576 | 0.6923 | 0.3846 | 1.0000 | 0.5556 | 0.5241 |
| 4 | 0.8346 | 0.8360 | 0.7879 | 0.7577 | 0.6154 | 0.9000 | 0.6957 | 0.5480 |
| 5 | 0.8615 | 0.7974 | 0.7576 | 0.7462 | 0.6923 | 0.8000 | 0.6923 | 0.4923 |

### B: Linear Fusion

| Fold | ROC-AUC | Average Precision | Accuracy | Balanced Accuracy | Sensitivity | Specificity | F1-score | MCC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8135 | 0.7448 | 0.7576 | 0.7381 | 0.6667 | 0.8095 | 0.6667 | 0.4762 |
| 2 | 0.8654 | 0.8117 | 0.8182 | 0.7962 | 0.6923 | 0.9000 | 0.7500 | 0.6139 |
| 3 | 0.9500 | 0.9309 | 0.7879 | 0.7308 | 0.4615 | 1.0000 | 0.6316 | 0.5847 |
| 4 | 0.8615 | 0.8454 | 0.7576 | 0.7327 | 0.6154 | 0.8500 | 0.6667 | 0.4824 |
| 5 | 0.8962 | 0.8612 | 0.7879 | 0.7981 | 0.8462 | 0.7500 | 0.7586 | 0.5829 |

### C: Small Regularized

| Fold | ROC-AUC | Average Precision | Accuracy | Balanced Accuracy | Sensitivity | Specificity | F1-score | MCC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8175 | 0.7463 | 0.7879 | 0.7798 | 0.7500 | 0.8095 | 0.7200 | 0.5508 |
| 2 | 0.8615 | 0.7990 | 0.7576 | 0.7462 | 0.6923 | 0.8000 | 0.6923 | 0.4923 |
| 3 | 0.9231 | 0.9059 | 0.8485 | 0.8077 | 0.6154 | 1.0000 | 0.7619 | 0.7016 |
| 4 | 0.8500 | 0.8296 | 0.7879 | 0.7846 | 0.7692 | 0.8000 | 0.7407 | 0.5628 |
| 5 | 0.8808 | 0.8129 | 0.7576 | 0.7596 | 0.7692 | 0.7500 | 0.7143 | 0.5095 |

## Findings

- **Linear Fusion achieved the highest mean ROC-AUC (0.8773) and average precision (0.8388).** It also produced the lowest mean loss (0.4704), indicating strong ranking performance across the validation folds.
- **Small Regularized achieved the highest mean accuracy (0.7879), balanced accuracy (0.7756), sensitivity (0.7192), F1-score (0.7258) and MCC (0.5634).** At the fixed threshold of 0.5, it provided the strongest overall balance across these threshold-dependent metrics.
- **Baseline achieved the highest mean specificity (0.8819) and precision (0.7918)** but had the lowest mean sensitivity (0.5949).
- **The configurations show a sensitivity-specificity trade-off.** Small Regularized detected more Sick cases on average but had lower specificity than the other configurations.
- **No configuration dominates every metric.** Linear Fusion performs best on threshold-independent ranking metrics, while Small Regularized performs best on several threshold-dependent metrics.

These findings are descriptive. Differences in mean metrics do not establish statistical superiority, and the relatively small cohort limits the precision of the estimates.

## Conclusion

Run_02 extends the initial single-split fusion-head comparison to five-fold patient-level cross-validation. Across the five folds, Linear Fusion achieved the highest mean ROC-AUC and average precision, while Small Regularized achieved the highest mean sensitivity, balanced accuracy, F1-score and MCC.

The results provide a more robust comparison across patient splits than the initial single validation split. However, no configuration dominates every metric, so the final choice should consider the sensitivity-specificity trade-off and the predefined model-selection protocol.

### Next Step: Fixed-Epoch Selection

The next run will evaluate candidate fixed training epochs using the Run_02 cross-validation results. The candidate epochs will be compared under a consistent selection procedure to identify a suitable training duration without selecting a separate best epoch for each fold.

The selected epoch will then be used as the fixed training duration for the subsequent out-of-fold prediction and threshold-selection stages, followed by final model training and independent test evaluation according to the planned workflow.

**Important:** All results in this README are development-set cross-validation results. The independent test set remains unevaluated in Run_02. Candidate-epoch selection and subsequent model-selection steps must use development data only. The independent test set should remain untouched until the final held-out evaluation.