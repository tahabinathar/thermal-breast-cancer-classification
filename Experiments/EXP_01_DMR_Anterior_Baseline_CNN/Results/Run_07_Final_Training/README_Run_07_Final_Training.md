# EXP01_DMR_Anterior_Baseline

## Run 07: Final Model Training

### 1. Objective

The objective of Run 07 was to train the final EXP01 baseline CNN using the complete development dataset after all model configuration and training decisions had been finalized in Runs 02–06.

The final configuration was established as follows:

* Learning rate = 1e-3, selected in Run 02
* Dropout rate = 0.3, selected in Run 03
* Fixed training epoch = 89, selected in Run 04
* Classification threshold = 0.586, selected in Run 06

The complete development dataset consisted of the original Training and Validation splits combined:

* Total = 197 images
* Healthy = 125
* Sick = 72

The final model was trained from scratch on all 197 development images for exactly 89 epochs.

No validation split was used during final training because the original validation data had already been incorporated into the development dataset. No model checkpointing, early stopping, best-epoch selection, or validation-based model selection was performed.

The test set was not loaded or accessed in Run 07. It was reserved exclusively for the subsequent final evaluation.

---

### 2. Methodology

```text
RUN 07

│
├── Development Dataset
│   ├── Train = 163 images
│   ├── Validation = 34 images
│   ├── Total = 197 images
│   ├── Healthy = 125
│   ├── Sick = 72
│   └── Test = NOT USED
│
├── Temperature Normalization
│   ├── Range calculated from all 197 development images
│   ├── Global minimum
│   └── Global maximum
│
├── Class Weighting
│   ├── Calculated from all 197 development labels
│   └── Balanced class weights
│
├── Model Configuration
│   ├── Input = 128 × 128 × 1
│   ├── Learning rate = 1e-3
│   ├── Dropout = 0.3
│   ├── Batch size = 16
│   └── Epochs = 89
│
├── Training
│   ├── All 197 development images
│   ├── Data augmentation enabled
│   ├── Class weighting enabled
│   ├── No validation data
│   ├── No checkpointing
│   └── No early stopping
│
└── Final Model
    └── Model state after epoch 89
```

The Training and Validation splits were combined before final model training.

The temperature normalization range was calculated once using all 197 development images. This range was then used to preprocess every development image.

Balanced class weights were calculated from the labels of all 197 development images.

The model was trained for exactly 89 epochs using the finalized configuration. The model state after epoch 89 was saved as the final model.

No model selection was performed during Run 07.

The classification threshold of 0.586 selected in Run 06 was not applied during training. It will be used later when converting final test-set probabilities into binary predictions.

---

### 3. Configuration

| Parameter                | Value                       |
| ------------------------ | --------------------------- |
| Experiment               | EXP01_DMR_Anterior_Baseline |
| Input dataset            | Development dataset         |
| Training split           | Train + Validation          |
| Development dataset      | 197 images                  |
| Healthy                  | 125                         |
| Sick                     | 72                          |
| Test set                 | Not used                    |
| Input size               | 128 × 128 × 1               |
| Batch size               | 16                          |
| Training epochs          | 89                          |
| Optimizer                | Adam                        |
| Learning rate            | 1e-3                        |
| Dropout rate             | 0.3                         |
| Random rotation          | 0.05                        |
| Random zoom              | 0.05                        |
| Horizontal flip          | False                       |
| Class weighting          | Balanced                    |
| Early stopping           | False                       |
| Checkpointing            | False                       |
| Model selection          | None                        |
| Random seed              | 10                          |
| Device                   | CPU                         |
| Classification threshold | 0.586                       |
| Test evaluation          | Not performed               |

The classification threshold of 0.586 was carried forward from Run 06 but was not used during model training.

---

### 4. Development Dataset Preparation

The original Training and Validation datasets were combined to form the complete development dataset.

The resulting dataset contained:

* Training images = 163
* Validation images = 34
* Development images = 197
* Healthy = 125
* Sick = 72

The Test split was not loaded.

The combined development dataset was used for both temperature-range calculation and final model training.

The class distribution was therefore:

```text
Development Dataset

Healthy = 125
Sick    = 72
Total   = 197
```

Balanced class weights were calculated using the complete development label set rather than only the original Training split.

No information from the Test split was used during dataset preparation.

---

### 5. Preprocessing

Each development image was processed using the preprocessing pipeline defined in `preprocessing.py`.

The preprocessing sequence was:

```text
Raw thermal image
        ↓
Temperature normalization
        ↓
Pad to square
        ↓
Resize to 128 × 128
        ↓
Add single-channel dimension
        ↓
Final tensor: 128 × 128 × 1
```

Temperature normalization used a global minimum and maximum calculated from all 197 development images.

The normalization range was therefore based on the complete development dataset and was not calculated from the Test set.

The processed images were stored as `float32` arrays.

No test images were loaded during preprocessing.

The exact development temperature range was stored in `run_config.json` so that the same normalization parameters can be used later when preprocessing the untouched Test set.

---

### 6. Final Model Training

The final CNN architecture was identical to the EXP01 baseline architecture used in the previous runs, with the finalized dropout rate of 0.3.

```text
Input
128 × 128 × 1

        ↓

Random Rotation
factor = 0.05

        ↓

Random Zoom
factor = 0.05

        ↓

Conv2D
32 filters
2 × 2
ReLU
same padding

        ↓

Conv2D
32 filters
2 × 2
ReLU
same padding

        ↓

MaxPooling2D

        ↓

Conv2D
64 filters
2 × 2
ReLU
same padding

        ↓

MaxPooling2D

        ↓

Conv2D
128 filters
2 × 2
ReLU
same padding

        ↓

MaxPooling2D

        ↓

GlobalAveragePooling2D

        ↓

Dense
64 units
ReLU

        ↓

Dropout
0.3

        ↓

Dense
1 unit
Sigmoid
```

The finalized training configuration was:

```text
Optimizer       = Adam
Learning rate   = 1e-3
Batch size      = 16
Epochs          = 89
Dropout         = 0.3
Class weighting = Balanced
Seed            = 10
Device          = CPU
```

Training was performed using:

```python
model.fit(
    X_development,
    y_development,
    epochs=89,
    batch_size=16,
    class_weight=CLASS_WEIGHTS,
    shuffle=True,
)
```

No `validation_data` argument was supplied.

No `ModelCheckpoint` callback was used.

No early stopping was used.

No best epoch was calculated.

Therefore, the model state after epoch 89 was directly treated as the final trained model.

---

### 7. Results

Run 07 was a final model training run rather than a model-selection or evaluation run.

The final training dataset contained:

| Parameter                  | Result |
| -------------------------- | -----: |
| Training images            |    163 |
| Original validation images |     34 |
| Development images         |    197 |
| Healthy                    |    125 |
| Sick                       |     72 |
| Training epochs            |     89 |
| Learning rate              |   1e-3 |
| Dropout                    |    0.3 |
| Batch size                 |     16 |
| Test images used           |      0 |

The training history recorded the training loss, training accuracy, and training ROC-AUC for all 89 epochs.

No validation metrics were generated during Run 07 because no validation set was supplied to `model.fit()`.

No test metrics were generated.

The final model corresponds to the model state at the end of epoch 89.

---

### 8. Final Run 07 Decision

The final EXP01 baseline CNN was trained using all 197 development images with the configuration finalized through Runs 02–06.

| Parameter                | Value           |
| ------------------------ | --------------- |
| Development dataset      | 197 images      |
| Healthy                  | 125             |
| Sick                     | 72              |
| Learning rate            | 1e-3            |
| Dropout                  | 0.3             |
| Fixed training epoch     | **89**          |
| Batch size               | 16              |
| Optimizer                | Adam            |
| Data augmentation        | Rotation + Zoom |
| Class weighting          | Balanced        |
| Model selection          | None            |
| Checkpointing            | None            |
| Early stopping           | None            |
| Classification threshold | **0.586**       |
| Test set                 | **Not used**    |

The model state after exactly 89 epochs was saved as the final EXP01 model.

The threshold of 0.586 selected in Run 06 was carried forward for the later final test evaluation but was not used to influence the training process.

The final Test set remained completely untouched.

---

### 9. Output Structure

```text
Run_07_Final_Training/

├── results/
│
│   ├── training_history.json
│   └── run_config.json
│
├── model/
│
│   ├── final_model.keras
│   └── final_model.weights.h5
│
└── plots/
│
    ├── training_loss.png
    └── training_auc.png
```

#### `training_history.json`

Contains the training history recorded over all 89 epochs, including:

* Training loss
* Training accuracy
* Training ROC-AUC

No validation history is present because validation data was not used during final training.

#### `run_config.json`

Stores the complete Run 07 configuration, including:

* Experiment and run names
* Random seed
* Dataset information
* Development split information
* Input dimensions
* Batch size
* Number of epochs
* Optimizer
* Learning rate
* Dropout rate
* Data augmentation
* Class weights
* Temperature normalization method
* Development temperature range
* Model selection status
* Checkpointing status
* Test-set usage status
* Classification threshold carried forward from Run 06

#### `final_model.keras`

Contains the complete final trained Keras model, including architecture and learned weights.

The model represents the state after exactly 89 training epochs.

#### `final_model.weights.h5`

Contains the learned weights of the final model.

#### Plots

Two training-history plots were generated:

1. **Training loss vs epoch**
2. **Training ROC-AUC vs epoch**

No validation curves were generated because there was no validation dataset during Run 07.

---

### 10. Final Summary

* Experiment: **EXP01_DMR_Anterior_Baseline**
* Run: **Run 07 Final Training**
* Development images: **197**
* Healthy images: **125**
* Sick images: **72**
* Original Training images: **163**
* Original Validation images: **34**
* Input size: **128 × 128 × 1**
* Batch size: **16**
* Learning rate: **1e-3**
* Dropout: **0.3**
* Optimizer: **Adam**
* Training epochs: **89**
* Class weighting: **Balanced**
* Data augmentation: **Random Rotation + Random Zoom**
* Model selection: **None**
* Checkpointing: **None**
* Early stopping: **None**
* Final model state: **After epoch 89**
* Classification threshold from Run 06: **0.586**
* Test set: **Not used**

Run 07 successfully trained the final EXP01 baseline CNN using the complete 197-image development dataset with the configuration finalized through Runs 02–06.

The model after exactly **89 epochs** was saved as the final trained model. The development temperature normalization range was also preserved in the Run 07 configuration for consistent preprocessing of the Test set.

No Test-set images, predictions, or evaluation metrics were used in Run 07. The untouched 37-image Test set will be accessed only in the subsequent final evaluation run using the saved Run 07 model and the fixed classification threshold of **0.586** from Run 06.