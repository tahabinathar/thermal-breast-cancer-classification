# EXP02 — DMR Anterior View ResNet18

## 1. Experiment Overview

**Experiment ID:** `EXP02_DMR_Anterior_ResNet18`

**Objective:**

Evaluate a transfer-learning-based ResNet-18 classifier for binary classification of anterior-view breast thermal images into:

* **Healthy = 0**
* **Sick = 1**

Benign cases were excluded from this experiment.

The experiment was designed to maintain strict separation between the Development set and the held-out Test set. The Test set was not used during preprocessing verification, cross-validation, model configuration selection, fixed training epoch selection, decision threshold selection, or final model training.

---

## 2. Dataset

Dataset: `DMR_IR_Anterior_View_ROI`

| Split       |  Images | Healthy |   Sick |
| ----------- | ------: | ------: | -----: |
| Train       |     163 |     103 |     60 |
| Validation  |      34 |      22 |     12 |
| Development |     197 |     125 |     72 |
| Test        |      37 |      24 |     13 |
| **Total**   | **234** | **149** | **85** |

The original Train and Validation sets were combined into the **Development set** for the later cross-validation and final training stages.

The **Test set remained completely untouched until Run 07**, where it was used only for final evaluation.

---

## 3. Preprocessing Pipeline

Each raw thermal image was processed using the following pipeline:

1. Load the `.tif` thermal image as a 2D floating-point temperature array.

2. Identify valid temperature values using `temperature > 0`.

3. Normalize valid temperatures using the appropriate training/Development-derived temperature range:

   `normalized = (temperature - global_min) / (global_max - global_min)`

4. Preserve invalid/background pixels (`<= 0`) as zero.

5. Apply centered zero-padding to obtain a square image.

6. Resize to `224 × 224` using bilinear interpolation.

7. Add the single thermal channel to obtain `224 × 224 × 1`.

8. Repeat the single channel three times to obtain `224 × 224 × 3`.

9. Apply ImageNet normalization for compatibility with the pretrained ResNet-18 backbone.

No separate TIFF handling pipeline was required.

### Preprocessing Verification

All 234 images passed the preprocessing verification.

The overall verification range across all images was:

* **Minimum:** `22.000000`
* **Maximum:** `36.812618255615234`

The overall range was used only for verification. Model development and final evaluation used the appropriate training/Development-derived temperature ranges.

---

## 4. Model Architecture

An ImageNet-pretrained ResNet-18 using `IMAGENET1K_V1` weights was used.

The original ResNet-18 classification layer was replaced with:

```text
Linear(512, 64)

→ ReLU

→ Dropout(0.3)

→ Linear(64, 1)
```

### Parameter Count

| Parameter Type       |      Count |
| -------------------- | ---------: |
| Total parameters     | 11,209,409 |
| Trainable parameters |     32,897 |

The ResNet-18 backbone remained frozen. Only the classification head was trained.

---

## 5. Common Training Configuration

| Parameter            | Setting                               |
| -------------------- | ------------------------------------- |
| Model                | ImageNet-pretrained ResNet-18         |
| Backbone             | Frozen                                |
| Input size           | 224 × 224                             |
| Input channels       | 3                                     |
| Thermal channels     | 1 repeated to 3                       |
| Optimizer            | Adam                                  |
| Learning rate        | 0.001                                 |
| Batch size           | 16                                    |
| Dropout              | 0.3                                   |
| Loss                 | Weighted BCEWithLogitsLoss            |
| Class weighting      | Balanced                              |
| Random seed          | 10                                    |
| Augmentation         | Rotation ±18°, affine scale 0.95–1.05 |
| Horizontal flip      | Not used                              |
| Training GPU         | NVIDIA Quadro M1200                   |
| Final Test inference | CPU                                   |

---

# 6. Run 01 — Baseline

Run 01 used the original Train and Validation split.

* Training images: 163
* Validation images: 34
* Test set: not used
* Maximum epochs: 100
* Model selection criterion: validation ROC-AUC

### Best Validation Result

The highest validation ROC-AUC occurred at epoch 1:

| Metric            |  Value |
| ----------------- | -----: |
| ROC-AUC           | 0.9053 |
| PR-AUC            | 0.7556 |
| Accuracy          | 0.6471 |
| Balanced Accuracy | 0.5000 |
| Sensitivity       | 0.0000 |
| Specificity       | 1.0000 |
| Precision         | 0.0000 |
| F1-score          | 0.0000 |
| MCC               | 0.0000 |

All 34 validation images were predicted as Healthy at the default threshold of 0.5.

Class weights:

* Healthy: `0.7913`
* Sick: `1.3583`

The Test set was not used.

---

# 7. Run 02 — Five-Fold Cross-Validation

Run 02 performed stratified 5-fold cross-validation using the complete Development set:

* Development images: 197
* Healthy: 125
* Sick: 72
* Test set: not used
* Maximum epochs: 100

Each fold used its own training-only temperature range and class weights.

The random seed was reset to 10 for each fold.

### Best Epoch per Fold

| Fold   | Best Epoch | Held-Out-Fold ROC-AUC |
| ------ | ---------: | --------------------: |
| Fold 1 |         68 |                0.9413 |
| Fold 2 |         89 |                0.9253 |
| Fold 3 |         67 |                0.9886 |
| Fold 4 |         32 |                0.9114 |
| Fold 5 |         23 |                0.8800 |

The highest mean epoch-wise **held-out-fold ROC-AUC** across the five folds occurred at **epoch 32**, with a mean ROC-AUC of approximately **0.9178**.

These results were used to identify the candidate fixed training epochs for Run 03.

Run 02 therefore served as an **epoch-candidate identification stage**, rather than the final fixed training epoch selection stage.

---

# 8. Run 03 — Fixed Training Epoch Selection

Run 03 compared the candidate fixed training epochs identified from Run 02:

```text
[23, 32, 67, 68, 89, 100]
```

The same five folds were reused for each candidate epoch. A fresh model was trained for every fold and candidate epoch.

### Mean Cross-Validation ROC-AUC

|  Epoch | Mean ROC-AUC | Standard Deviation |
| -----: | -----------: | -----------------: |
|     23 |     0.906667 |           0.049784 |
| **32** | **0.912229** |       **0.045715** |
|     67 |     0.909524 |           0.054601 |
|     68 |     0.908419 |           0.055412 |
|     89 |     0.910210 |           0.056134 |
|    100 |     0.904990 |           0.060510 |

### Selected Fixed Training Epoch

**Epoch 32** was selected because it produced the highest mean cross-validation ROC-AUC among the evaluated candidate epochs.

Secondary metrics were treated as descriptive and were not used as the primary fixed training epoch selection criterion.

The Test set remained untouched.

### Methodological Note

Run 02 was used to identify candidate fixed training epochs and the same folds were reused in Run 03 to compare those candidates. Therefore, Run 03 should be interpreted as an internal Development-stage fixed training epoch selection procedure rather than an independent unbiased estimate of generalization performance.

---

# 9. Run 04 — Five-Fold OOF Prediction Generation

Run 04 generated out-of-fold (OOF) predictions for all 197 Development images using the fixed **32-epoch** configuration selected in Run 03.

### Configuration

* Development images: 197
* Folds: 5
* Fixed training epochs: 32
* Backbone: frozen ResNet-18
* Trainable parameters: 32,897
* Learning rate: 0.001
* Dropout: 0.3
* Batch size: 16
* Seed: 10
* Test set: not used

Each Development image received exactly one OOF prediction from a model that was trained without that image.

### Fold Results

| Fold   | Held-Out Images |  ROC-AUC |   PR-AUC |
| ------ | --------------: | -------: | -------: |
| Fold 1 |              40 | 0.930667 | 0.949768 |
| Fold 2 |              40 | 0.893333 | 0.805030 |
| Fold 3 |              39 | 0.988571 | 0.981548 |
| Fold 4 |              39 | 0.897143 | 0.836683 |
| Fold 5 |              39 | 0.851429 | 0.822636 |

### Overall OOF Performance

| Metric  |    Value |
| ------- | -------: |
| ROC-AUC | 0.892667 |
| PR-AUC  | 0.832326 |

The OOF predictions were subsequently used for **decision threshold selection**.

The Test set remained untouched.

---

# 10. Run 05 — OOF Decision Threshold Selection

Run 05 selected the classification **decision threshold** using only the Run 04 Development-set OOF predictions.

The threshold constraints were:

* Sensitivity > 0.91
* Specificity > 0.60

Five thresholds satisfied both constraints.

The selection priority was:

1. Highest sensitivity
2. Highest specificity
3. Highest threshold

### Selected Decision Threshold

**Classification decision threshold: `0.19217434525489807`**

Displayed in experiment outputs as:

**`0.192174`**

### OOF Performance at Selected Decision Threshold

| Metric            |  Value |
| ----------------- | -----: |
| Sensitivity       | 0.9306 |
| Specificity       | 0.6320 |
| Accuracy          | 0.7411 |
| Balanced Accuracy | 0.7813 |
| Precision         | 0.5929 |
| F1-score          | 0.7243 |

Confusion matrix:

```text
[[79, 46],
 [ 5, 67]]
```

The decision threshold was selected exclusively from Development-set OOF predictions. The Test set was not used.

The selected threshold was fixed for final Test evaluation and was not subsequently optimized.

---

# 11. Run 06 — Final Model Training

Run 06 trained the final ResNet-18 model using the complete Development set and the model configuration selected during the preceding Development-stage runs.

### Configuration

| Parameter             | Value                    |
| --------------------- | ------------------------ |
| Training data         | Complete Development set |
| Development images    | 197                      |
| Healthy               | 125                      |
| Sick                  | 72                       |
| Fixed training epochs | **32**                   |
| Learning rate         | 0.001                    |
| Batch size            | 16                       |
| Dropout               | 0.3                      |
| Seed                  | 10                       |
| Device                | CUDA                     |
| GPU                   | Quadro M1200             |
| Decision threshold    | 0.192174                 |
| Test set              | Not used                 |

### Development Temperature Range

The temperature normalization range calculated using the complete Development set was:

* **Global minimum:** `22.0`
* **Global maximum:** `36.812618255615234`

### Development Class Weights

```text
Healthy: 0.788
Sick: 1.3680555555555556
```

### Model Parameters

```text
Total parameters: 11,209,409

Trainable parameters: 32,897
```

### Training Loss History

|  Epoch | Training Loss |
| -----: | ------------: |
|      1 |      0.851118 |
|      2 |      0.800015 |
|      3 |      0.723242 |
|      4 |      0.715102 |
|      5 |      0.665308 |
|      6 |      0.591522 |
|      7 |      0.615454 |
|      8 |      0.576032 |
|      9 |      0.594585 |
|     10 |      0.542768 |
|     11 |      0.535997 |
|     12 |      0.595188 |
|     13 |      0.528103 |
|     14 |      0.474266 |
|     15 |      0.516859 |
|     16 |      0.489175 |
|     17 |      0.461668 |
|     18 |      0.481583 |
|     19 |      0.499728 |
|     20 |      0.524360 |
|     21 |      0.503394 |
|     22 |      0.497393 |
|     23 |      0.462457 |
|     24 |      0.470047 |
|     25 |      0.468058 |
|     26 |      0.460774 |
|     27 |      0.423109 |
|     28 |      0.466241 |
|     29 |      0.414987 |
|     30 |      0.424674 |
|     31 |      0.443622 |
| **32** |  **0.491160** |

Epoch 17 is shown above only as an intermediate training-loss observation. It was **not** selected as the final model epoch.

### Final Model

The final model was the model state **after completion of epoch 32**.

Saved model:

```text
Run_06_Final_Training/model/final_model_epoch32.pth
```

Other saved outputs:

```text
Run_06_Final_Training/results/temperature_range.json

Run_06_Final_Training/results/training_history.json

Run_06_Final_Training/results/run_config.json

Run_06_Final_Training/results/development_manifest.csv
```

No early stopping, checkpoint selection, or model selection was performed during Run 06.

The model state after epoch 32 was directly used as the final model for Run 07.

The decision threshold of `0.192174` was not used during training. It was carried forward for classification during final Test evaluation.

The Test set was not loaded or evaluated during Run 06.

---

# 12. Run 07 — Final Test Evaluation

Run 07 evaluated the final model from Run 06 on the previously untouched Test set.

### Evaluation Configuration

| Parameter                         | Value                                   |
| --------------------------------- | --------------------------------------- |
| Model source                      | Run 06 Final Training                   |
| Model                             | `final_model_epoch32.pth`               |
| Fixed training epochs             | 32                                      |
| Decision threshold source         | Run 05 OOF Decision Threshold Selection |
| Classification decision threshold | 0.19217434525489807                     |
| Temperature minimum               | 22.0                                    |
| Temperature maximum               | 36.812618255615234                      |
| Device                            | CPU                                     |
| Test images                       | 37                                      |

### Test Set

| Class     | Images |
| --------- | -----: |
| Healthy   |     24 |
| Sick      |     13 |
| **Total** | **37** |

The Test-set integrity check passed.

The final model was loaded successfully, and no retraining, model configuration selection, fixed training epoch selection, or decision threshold optimization was performed during Run 07.

---

## Final Test Results

| Metric                |     Result |
| --------------------- | ---------: |
| **ROC-AUC**           | **0.9455** |
| **Accuracy**          | **0.8378** |
| **Balanced Accuracy** | **0.8750** |
| **Sensitivity**       | **1.0000** |
| **Specificity**       | **0.7500** |
| **Precision**         | **0.6842** |
| **F1-score**          | **0.8125** |
| **MCC**               | **0.7164** |
| Decision threshold    |   0.192174 |

### Confusion Matrix

```text
[[18,  6],
 [ 0, 13]]
```

Where:

```text
TN = 18

FP = 6

FN = 0

TP = 13
```

Thus, all 13 Sick Test images were correctly classified at the selected decision threshold, while 6 of the 24 Healthy Test images were classified as Sick.

Run 07 used the final model trained in Run 06 and the decision threshold determined from Development-set OOF predictions in Run 05.

The Test set was used **only for final evaluation**.

---

# 13. Complete Experiment Pipeline

The final EXP02 workflow was:

```text
Original Dataset

      │

      ├── Train: 163
      ├── Validation: 34
      └── Test: 37

              │
              │ Test kept untouched
              ▼

       Development Set

          197 images

              │

              ▼

       Run 01: Baseline

              │

              ▼

       Run 02: 5-Fold CV

              │

              ▼

     Candidate Fixed Training Epochs

       [23, 32, 67, 68, 89, 100]

              │

              ▼

       Run 03: Fixed Training Epoch Selection

              │

              ▼

       Fixed Training Epoch = 32

              │

              ▼

       Run 04: 5-Fold OOF Prediction Generation

              │
              │ using 32 epochs
              ▼

       Run 05: Decision Threshold Selection

              │

              ▼

       Decision Threshold = 0.192174

              │

              ▼

       Run 06: Final Model Training

       All 197 Development Images

              │
              │ 32 fixed training epochs
              ▼

       final_model_epoch32.pth

              │

              ▼

       Run 07: Final Test Evaluation

              │

              ▼

       Test = 37 images
```

---

# 14. Final EXP02 Configuration

The final EXP02 configuration was:

| Component                     | Final Setting                                       |
| ----------------------------- | --------------------------------------------------- |
| Experiment                    | EXP02_DMR_Anterior_ResNet18                         |
| Dataset                       | DMR_IR_Anterior_View_ROI                            |
| Classes                       | Healthy = 0, Sick = 1                               |
| Total images                  | 234                                                 |
| Development set               | 197                                                 |
| Independent Test set          | 37                                                  |
| Model                         | ImageNet-pretrained ResNet-18                       |
| Weights                       | IMAGENET1K_V1                                       |
| Backbone                      | Frozen                                              |
| Classifier                    | Linear(512,64) → ReLU → Dropout(0.3) → Linear(64,1) |
| Total parameters              | 11,209,409                                          |
| Trainable parameters          | 32,897                                              |
| Input                         | 224 × 224 × 3                                       |
| Original thermal channels     | 1                                                   |
| Thermal channel handling      | Repeated to 3 channels                              |
| Optimizer                     | Adam                                                |
| Learning rate                 | 0.001                                               |
| Batch size                    | 16                                                  |
| Loss                          | Weighted BCEWithLogitsLoss                          |
| Random seed                   | 10                                                  |
| Augmentation                  | Rotation ±18°, affine scale 0.95–1.05               |
| Horizontal flip               | No                                                  |
| Selected fixed training epoch | 32                                                  |
| Final training                | All 197 Development images                          |
| Final model                   | `final_model_epoch32.pth`                           |
| Decision threshold            | 0.19217434525489807                                 |
| Test inference                | CPU                                                 |

---

# 15. Final Test Performance

The final EXP02 model achieved the following performance on the held-out Test set:

| Metric            | Final Result |
| ----------------- | -----------: |
| ROC-AUC           |   **0.9455** |
| Accuracy          |   **0.8378** |
| Balanced Accuracy |   **0.8750** |
| Sensitivity       |   **1.0000** |
| Specificity       |   **0.7500** |
| Precision         |   **0.6842** |
| F1-score          |   **0.8125** |
| MCC               |   **0.7164** |

Confusion matrix:

```text
              Predicted

              Healthy  Sick

Actual Healthy    18      6

       Sick        0     13
```

---

# 16. Key Takeaways

1. The experiment used a strict **Development/Test separation**, with the 37-image Test set untouched until final evaluation.

2. Run 02 used stratified 5-fold cross-validation on the 197-image Development set to identify candidate fixed training epochs and examine epoch-wise held-out-fold performance.

3. The candidate fixed training epochs identified in Run 02 were `[23, 32, 67, 68, 89, 100]`.

4. Run 03 formally selected the **fixed training epoch of 32** based on the highest mean cross-validation ROC-AUC among the evaluated candidate epochs.

5. Run 03 reused the same Development-set folds from Run 02. Therefore, it represents an internal Development-stage fixed training epoch selection procedure rather than an independent estimate of generalization performance.

6. Run 04 generated five-fold OOF predictions using the fixed **32-epoch** configuration.

7. The final **decision threshold of 0.192174** was selected exclusively from Development-set OOF predictions in Run 05.

8. The selected decision threshold was fixed before Test evaluation and was not adjusted using Test-set performance.

9. Run 06 trained the final model on all **197 Development images for exactly 32 fixed training epochs**.

10. The final model is the model state **after completion of epoch 32**, saved as:

    ```text
    final_model_epoch32.pth
    ```

11. Run 06 did not use a held-out validation subset, early stopping, checkpoint selection, or Test-set model selection.

12. Run 07 evaluated this final model on the **37-image held-out Test set** without retraining, model configuration selection, fixed training epoch selection, or decision threshold optimization.

13. The final Test ROC-AUC was **0.9455**.

14. At the predetermined decision threshold, the final Test sensitivity was **1.0000** and specificity was **0.7500**.

15. The final confusion matrix contained **18 true negatives, 6 false positives, 0 false negatives, and 13 true positives**.

16. The final reported Test metrics therefore correspond specifically to the **epoch-32 final model and the pre-selected Development-set OOF decision threshold of 0.192174**.

17. Overall, EXP02 maintains a clear separation between **model configuration development using the 197-image Development set** and **final independent evaluation using the 37-image Test set**.