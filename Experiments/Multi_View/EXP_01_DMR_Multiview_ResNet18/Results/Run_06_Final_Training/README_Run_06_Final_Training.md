# EXP_01_DMR_Multiview_ResNet18
# Run_06_Final_Training

## Objective

Train the final Baseline multi-view ResNet-18 model using all 165 development patients (Train + Validation), based on the fusion-head configuration and fixed training duration selected in previous runs.

Run_06 trains a single final model for 82 epochs using the complete development dataset. The classification threshold selected in Run_05 is retained for subsequent independent test-set evaluation.

No additional model selection, threshold optimization or independent test-set evaluation is performed.

## Shared Configuration

- **Backbone:** Frozen ImageNet-pretrained ResNet-18
- **Input views:** Anterior, left-oblique and right-oblique
- **Input image size:** 224 × 224
- **Feature extraction:** Shared ResNet-18 producing 512 features per view
- **Feature fusion:** Concatenation of three feature vectors (1536 features total)
- **Fusion head:** A: Baseline (1536 → 64 → ReLU → Dropout → 1)
- **Dropout:** 0.3
- **Trainable parameters:** 98,433
- **Total parameters:** 11,274,945
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Weight decay:** 0.0
- **Batch size:** 16
- **Fixed training epochs:** 82
- **Loss function:** Class-weighted BCEWithLogitsLoss
- **Random seed:** 10
- **Training device:** NVIDIA Quadro M1200 (CUDA)
- **Selected classification threshold:** 0.2041969746351242
- **Early stopping:** Disabled
- **Test set used:** No

The fixed training duration of 82 epochs was selected in Run_03, while the classification threshold was determined using OOF predictions in Run_05.

## Development Dataset

The final model was trained using the combined Train and Validation partitions.

| Partition | Healthy | Sick | Total |
|---|---:|---:|---:|
| Train | 84 | 56 | 140 |
| Validation | 17 | 8 | 25 |
| **Total** | **101** | **64** | **165** |

Each patient contributed three thermal images:

1. Anterior view
2. Left-oblique view
3. Right-oblique view

The complete development dataset therefore contained **165 patients and 495 thermal images**.

Patient-level data organization was preserved, and all three views belonging to a patient were processed together.

The independent Test partition was not loaded or used.

## Temperature Normalization and Preprocessing

The global temperature normalization range was calculated using all 495 development images.

| Parameter | Value |
|---|---:|
| Global minimum temperature | 19.7276096344 |
| Global maximum temperature | 36.8400001526 |
| Patients used | 165 |
| Images used | 495 |

The preprocessing procedure included:

- Global min-max normalization using the development-only temperature range.
- Preservation of the background mask.
- Padding images to a square aspect ratio.
- Resizing to 224 × 224.
- Converting grayscale images to three channels.
- Applying ImageNet mean and standard deviation normalization.
- Applying random rotation (±18°) and random affine scaling (0.95–1.05) during training.

The normalization parameters were saved in `temperature_range.json` and included in the final model checkpoint.

No independent test images were used to calculate the temperature range.

## Class Weighting

Class weights were calculated from the complete development dataset to account for the imbalance between Healthy and Sick patients.

| Class | Patients | Class weight |
|---|---:|---:|
| Healthy (0) | 101 | 0.816832 |
| Sick (1) | 64 | 1.289063 |

The weights were applied to the binary cross-entropy loss during final training.

The Sick class received a higher weight because it contained fewer patients.

## Final Model Architecture

The final model used a shared ImageNet-pretrained ResNet-18 backbone for all three thermal views.

Each view was processed through the same frozen backbone to extract a 512-dimensional feature vector.

The three feature vectors were concatenated into a 1536-dimensional representation and passed through the Baseline fusion head.

**Architecture:**

`Anterior (512) + Left Oblique (512) + Right Oblique (512) → Concatenation (1536) → Linear (64) → ReLU → Dropout (0.3) → Linear (1)`

| Component | Parameters |
|---|---:|
| Total model parameters | 11,274,945 |
| Trainable parameters | 98,433 |

The pretrained backbone remained frozen throughout training, while only the fusion head was optimized.

## Final Training

The final model was trained for exactly 82 epochs using all 165 development patients.

No validation-based checkpoint selection or early stopping was performed because the training duration had already been determined in Run_03.

The model state after epoch 82 was saved as the final model.

### Training Loss Summary

| Epoch | Training loss |
|---|---:|
| 1 | 0.762557 |
| 10 | 0.499835 |
| 20 | 0.366388 |
| 30 | 0.328165 |
| 40 | 0.280184 |
| 50 | 0.241605 |
| 60 | 0.281807 |
| 70 | 0.284999 |
| 80 | 0.220212 |
| **82 (Final)** | **0.226719** |

The training loss generally decreased over the 82 epochs, although fluctuations were observed during training.

The loss decreased from **0.762557 at epoch 1** to **0.226719 at epoch 82**, representing an approximate 70.3% reduction.

The final checkpoint corresponds to epoch 82, regardless of whether an earlier epoch achieved a lower training loss.

Training loss alone does not establish the model's generalization performance.

## Selected Classification Threshold

The classification threshold was obtained from Run_05_OOF_Threshold_Selection.

| Parameter | Value |
|---|---:|
| Selected threshold | 0.2041969746351242 |
| Threshold source | Run_05 |
| Selection method | Constraint-based OOF threshold selection |
| Minimum sensitivity | > 0.91 |
| Minimum specificity | > 0.60 |

The threshold was not recalculated or modified during final training.

It was stored in the final model checkpoint and run configuration for use during independent test-set evaluation.

## Generated Outputs

Run_06 generated the following files:

| Output | Description |
|---|---|
| `model/final_model_epoch82.pth` | Final trained model checkpoint after 82 epochs, including model parameters and configuration |
| `results/temperature_range.json` | Development-only global temperature normalization parameters |
| `results/training_history.json` | Training loss and learning rate for all 82 epochs |
| `results/development_manifest.csv` | Patient-level development dataset manifest with labels and three-view image paths |
| `results/run_config.json` | Final training configuration, architecture, class weights and selected threshold |
| `plots/training_loss.png` | Training loss across 82 epochs |

All outputs were saved under `Results/Run_06_Final_Training/`.

## Findings

- The final Baseline multi-view ResNet-18 model was successfully trained using all **165 development patients** and **495 thermal images**.
- The shared ResNet-18 backbone remained frozen, with **98,433 trainable parameters** in the fusion head.
- Development-only temperature normalization produced a global minimum of **19.7276** and maximum of **36.8400**.
- Class weighting was applied using the full development class distribution of 101 Healthy and 64 Sick patients.
- Training completed for the predefined **82 epochs**, without early stopping or additional model selection.
- Training loss decreased from **0.762557** to **0.226719**.
- The previously selected classification threshold of **0.2041969746351242** was retained.
- The final model checkpoint, training history, preprocessing parameters, dataset manifest and configuration were saved successfully.
- The independent test set remained untouched.

## Conclusion

Run_06 successfully completed final training of the Baseline multi-view ResNet-18 model using the complete development dataset.

The model was trained for the predefined 82 epochs using a shared frozen ImageNet-pretrained ResNet-18 backbone and a trainable feature-fusion classification head.

The final training loss was **0.226719**, and the model state after epoch 82 was saved as `final_model_epoch82.pth`.

The development-only normalization parameters and the classification threshold selected in Run_05 were preserved for subsequent evaluation.

No independent test-set performance metrics were calculated during this run.

## Next Step: Independent Test-Set Evaluation

The next run will evaluate the final trained model on the independent Test partition.

The evaluation will:

- Load the final checkpoint from Run_06.
- Apply the saved development-only temperature normalization parameters.
- Process the anterior, left-oblique and right-oblique views of each test patient.
- Generate patient-level predicted probabilities.
- Apply the fixed classification threshold of **0.2041969746351242**.
- Calculate ROC-AUC, Average Precision, accuracy, balanced accuracy, sensitivity, specificity, precision, F1-score, MCC and the confusion matrix.
- Save patient-level predictions, evaluation metrics and plots.

The test set will be used only for evaluation, without additional training, model selection or threshold adjustment.

**Important:** Run_06 reports final training behavior, not independent generalization performance. The OOF-derived threshold was selected using predictions from five fold-specific models and may behave differently with the final model trained on all development patients. The independent test-set evaluation is required to assess the final model's performance.