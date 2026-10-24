# EXP_01_DMR_Multiview_ResNet18

# Run_01_Fusion_Head_Comparison

## Objective

Compare three fusion-head configurations for classifying healthy and sick cases using anterior, left-oblique and right-oblique thermal images. A shared frozen ImageNet-pretrained ResNet-18 extracts features from each view, which are concatenated before classification.

## Shared Configuration

- **Backbone:** Frozen ImageNet-pretrained ResNet-18
- **Input views:** Anterior, left-oblique and right-oblique
- **Feature fusion:** Concatenation of three 512-dimensional feature vectors (1536 features total)
- **Input size:** 224 × 224 pixels
- **Batch size:** 16
- **Maximum epochs:** 100
- **Learning rate:** 0.001
- **Random seed:** 10
- **Temperature normalization:** Global range of 21.51–36.84 °C, calculated from training images only
- **Patient-level split:** Complete three-view cases only
- **Training set:** 140 patients (84 Healthy, 56 Sick)
- **Validation set:** 25 patients (17 Healthy, 8 Sick)
- **Class weights:** Healthy = 0.8333, Sick = 1.25
- **Evaluation threshold:** 0.5

The test set was not loaded or evaluated in any of these runs.

## Fusion-Head Comparison

| Configuration        | Classifier    | Trainable parameters | Dropout | Weight decay | Best epoch | Best validation ROC-AUC |
| -------------------- | ------------- | -------------------: | ------: | -----------: | ---------: | ----------------------: |
| A: Baseline          | 1536 → 64 → 1 |               98,433 |     0.3 |            0 |         79 |                  0.7132 |
| B: Linear Fusion     | 1536 → 1      |                1,537 |       0 |        0.001 |         57 |                  0.7132 |
| C: Small Regularized | 1536 → 16 → 1 |               24,609 |     0.5 |        0.001 |         31 |                  0.7132 |

All configurations used the same frozen backbone, with 11,176,512 frozen parameters.

## Validation Results

Metrics below are calculated at the fixed classification threshold of 0.5.

| Metric            | A: Baseline |  B: Linear |   C: Small |
| ----------------- | ----------: | ---------: | ---------: |
| ROC-AUC           |      0.7132 |     0.7132 |     0.7132 |
| PR-AUC            |  **0.6220** |     0.6115 |     0.5344 |
| Accuracy          |      0.6000 |     0.6800 |     0.6800 |
| Balanced accuracy |      0.5404 | **0.6324** |     0.5331 |
| Sensitivity       |      0.3750 | **0.5000** |     0.1250 |
| Specificity       |      0.7059 |     0.7647 | **0.9412** |
| F1-score          |      0.3750 | **0.5000** |     0.2000 |
| MCC               |      0.0809 | **0.2647** |     0.1138 |

**Confusion matrices** (TN, FP, FN, TP):

- **A: Baseline:** 12, 5, 5, 3
- **B: Linear Fusion:** 13, 4, 4, 4
- **C: Small Regularized:** 16, 1, 7, 1

## Findings

- **All three heads achieved the same best validation ROC-AUC (0.7132).** No head demonstrated better overall ranking discrimination on this validation split.
- **Linear Fusion provided the strongest threshold-based balance.** It achieved the highest balanced accuracy, sensitivity, F1-score and MCC, with four false negatives and four false positives.
- **The Baseline head achieved the highest PR-AUC (0.6220)** but had lower accuracy and sensitivity than Linear Fusion at the selected threshold.
- **The Small Regularized head favored specificity over sensitivity.** It correctly classified most healthy cases but detected only one of the eight sick validation cases.

These findings are preliminary because they come from a single validation split containing only 25 patients. The identical best ROC-AUC values do not establish that the heads perform equivalently across different patient samples.

## Next Step: 5-Fold Cross-Validation

The next run will perform **5-fold cross-validation** to assess how consistently the fusion-head configurations perform across different patient-level splits. Results will be compared using fold-level and aggregate metrics, including ROC-AUC, PR-AUC, sensitivity, specificity and balanced accuracy.

The cross-validation results will help determine whether the apparent advantage of Linear Fusion is robust before proceeding to final model selection and held-out test evaluation.
