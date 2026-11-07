# EXP_01_DMR_Multiview_ResNet18
## Run_07_Final_Test_Evaluation

### 1. Objective

Evaluate the final multi-view ResNet-18 classification model on the independent Test dataset using the fixed classification threshold selected in Run_05.

The final model was trained in `Run_06_Final_Training` using all 165 development patients (Train + Validation) for 82 epochs.

This run assesses the model's generalization performance on previously unseen patients. No retraining, model selection, temperature-range recalculation or threshold optimization was performed.

### 2. Model Configuration

| Parameter | Configuration |
|---|---|
| Backbone | ImageNet-pretrained ResNet-18 |
| Backbone training | Frozen |
| Input views | Anterior, Left Oblique, Right Oblique |
| Input image size | 224 × 224 |
| Feature extraction | Shared ResNet-18, 512 features per view |
| Fusion method | Feature concatenation (1536 features) |
| Fusion head | A – Baseline |
| Classifier | Linear(1536,64) → ReLU → Dropout(0.3) → Linear(64,1) |
| Final training epochs | 82 |
| Random seed | 10 |
| Evaluation batch size | 16 |
| Evaluation device | CPU |
| Classification threshold | 0.2041969746351242 |
| Model source | Run_06_Final_Training |
| Threshold source | Run_05_OOF_Threshold_Selection |

### 3. Independent Test Dataset

The independent Test dataset contained 32 patients, each with three thermal views.

| Class | Patients | Thermal images |
|---|---:|---:|
| Healthy | 18 | 54 |
| Sick | 14 | 42 |
| **Total** | **32** | **96** |

The three thermal views were processed together to generate one predicted probability per patient.

A patient-level leakage check confirmed that no Test patient overlapped with the development dataset.

### 4. Preprocessing

The Test images were preprocessed using the same preprocessing procedure and normalization parameters as the final training run.

The global temperature normalization range was obtained exclusively from the development dataset.

| Parameter | Value |
|---|---:|
| Global minimum temperature | 19.727609634399414 |
| Global maximum temperature | 36.84000015258789 |
| Normalization source | Run_06_Final_Training |
| Image size | 224 × 224 |

Preprocessing included:

1. Global min-max temperature normalization.
2. Background masking and square padding.
3. Resizing to 224 × 224.
4. Conversion from grayscale to three channels.
5. ImageNet mean and standard deviation normalization.

No random data augmentation was applied during evaluation.

The temperature normalization range was not recalculated using Test images.

### 5. Evaluation Procedure

The final model checkpoint `final_model_epoch82.pth` was loaded from `Run_06_Final_Training`.

For each Test patient:

1. The anterior, left-oblique and right-oblique thermal images were loaded.
2. All three images were preprocessed using the saved development-only temperature normalization parameters.
3. Each image was passed through the shared frozen ResNet-18 backbone.
4. The three 512-dimensional feature vectors were concatenated into a 1536-dimensional representation.
5. The trained Baseline fusion head generated a patient-level logit.
6. The sigmoid function converted the logit into a predicted probability.
7. The fixed classification threshold of **0.2041969746351242** was applied to generate the final Healthy/Sick prediction.

The threshold was previously selected in Run_05 using out-of-fold predictions and predefined sensitivity and specificity constraints.

No model parameters or decision thresholds were modified during evaluation.

### 6. Final Test Results

| Metric | Result |
|---|---:|
| **ROC-AUC** | **0.9286** |
| Average Precision (PR-AUC summary) | 0.9066 |
| Accuracy | 0.6875 |
| Balanced Accuracy | 0.7222 |
| Sensitivity (Recall) | 1.0000 |
| Specificity | 0.4444 |
| Precision | 0.5833 |
| F1-score | 0.7368 |
| Matthews Correlation Coefficient | 0.5092 |
| Classification threshold | 0.204197 |

The model achieved a ROC-AUC of 0.9286, indicating strong discrimination between Healthy and Sick patients based on their predicted probabilities.

At the predefined threshold, the model achieved 100% sensitivity but 44.44% specificity.

### 7. Confusion Matrix

The final Test confusion matrix was:

| Actual / Predicted | Healthy | Sick |
|---|---:|---:|
| Healthy | 8 | 10 |
| Sick | 0 | 14 |

| Classification outcome | Count |
|---|---:|
| True Negatives (TN) | 8 |
| False Positives (FP) | 10 |
| False Negatives (FN) | 0 |
| True Positives (TP) | 14 |

All 14 Sick patients were correctly classified, resulting in zero false negatives.

However, 10 of the 18 Healthy patients were incorrectly classified as Sick.

This resulted in a relatively low specificity despite the high ROC-AUC.

### 8. Results Interpretation

**Discrimination performance**

The ROC-AUC of 0.9286 and Average Precision of 0.9066 indicate strong patient-level discrimination and ranking performance on the independent Test dataset.

These metrics evaluate the model's predicted probabilities across decision thresholds rather than its performance at one particular threshold.

**Sensitivity and specificity**

At the fixed threshold of 0.204197, the model correctly identified all Sick patients, achieving a sensitivity of 1.0000.

However, the specificity was 0.4444, indicating that only 8 of the 18 Healthy patients were correctly classified.

The low threshold favored sensitivity at the expense of specificity, resulting in 10 false-positive predictions.

**Threshold generalization**

The classification threshold was selected in Run_05 using out-of-fold development predictions under sensitivity and specificity constraints.

Although the threshold achieved the required constraints on the development data, its independent Test specificity was below the predefined 0.60 constraint.

This indicates that the selected operating point did not fully generalize to the independent Test cohort.

The threshold was not modified after observing these results.

**Overall classification performance**

The final accuracy of 0.6875 and balanced accuracy of 0.7222 reflect the trade-off between perfect sensitivity and relatively low specificity.

The F1-score of 0.7368 and MCC of 0.5092 provide additional measures of classification performance at the selected threshold.

The findings suggest that the model has strong discrimination capability but requires further investigation of probability calibration and threshold robustness in future development studies.

### 9. Generated Outputs

The evaluation generated the following files:

| File | Description |
|---|---|
| `results/test_metrics.json` | Final patient-level classification metrics and confusion matrix counts |
| `results/test_predictions.csv` | Patient IDs, three image paths, true labels, predicted probabilities and predicted classes |
| `results/run_config.json` | Evaluation configuration, model source, normalization parameters and threshold |
| `plots/test_roc_curve.png` | Independent Test ROC curve |
| `plots/test_pr_curve.png` | Independent Test precision-recall curve |
| `plots/test_confusion_matrix.png` | Independent Test confusion matrix |

All outputs were saved under `Results/Run_07_Final_Test_Evaluation/`.

### 10. Key Findings

- The final multi-view ResNet-18 model was successfully evaluated on **32 independent Test patients**, corresponding to 96 thermal images.
- The patient-level leakage check passed.
- The final model achieved **ROC-AUC = 0.9286** and **Average Precision = 0.9066**.
- All 14 Sick patients were correctly identified, resulting in **100% sensitivity** and zero false negatives.
- Ten Healthy patients were incorrectly classified as Sick, resulting in **44.44% specificity**.
- The final accuracy was **68.75%**, with balanced accuracy of **72.22%**.
- The predefined classification threshold of **0.2041969746351242** was applied without modification.
- No retraining, model selection or threshold optimization was performed using the Test dataset.

### 11. Conclusion

Run_07 completed the independent Test evaluation of the final Baseline multi-view ResNet-18 classification model.

The model demonstrated strong discrimination performance, achieving a **ROC-AUC of 0.9286** and **Average Precision of 0.9066** on 32 independent Test patients.

At the predefined classification threshold, the model correctly identified all 14 Sick patients, achieving 100% sensitivity. However, it misclassified 10 of the 18 Healthy patients, resulting in 44.44% specificity and 68.75% overall accuracy.

These results demonstrate a substantial sensitivity–specificity trade-off at the selected operating threshold. Although the model ranked patients effectively, its fixed-threshold classification performance was limited by a high false-positive rate.

The independent Test dataset was used exclusively for final evaluation, without retraining or threshold modification.

The results provide a final assessment of the multi-view model's performance and establish a reference for comparison with single-view approaches and alternative fusion architectures.

**Final status:** Independent Test evaluation completed. Final model and threshold preserved without modification.