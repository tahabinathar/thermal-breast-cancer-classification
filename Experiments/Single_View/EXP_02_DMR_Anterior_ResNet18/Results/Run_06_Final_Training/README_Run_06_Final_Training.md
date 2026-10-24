# EXP02_DMR_Anterior_ResNet18

## Run 06: Final Model Training

### 1. Objective

The objective of Run 06 was to train the final EXP02 ResNet18 model using the complete development dataset after the model configuration, fixed training epoch and classification threshold had been finalized in the preceding runs.

The final configuration was established as follows:

* Learning rate = 1e-3
* Dropout rate = 0.3
* Fixed training epoch = 32
* Classification threshold = 0.192174

The complete development dataset consisted of the original Training and Validation splits combined:

* Total = 197 images
* Healthy = 125
* Sick = 72

The final model was trained on all 197 development images for exactly 32 epochs.

The model architecture used ImageNet-pretrained ResNet18 with the complete backbone frozen and a trainable 512-64-1 classifier.

No validation split was used during final training because the original Validation data had already been incorporated into the development dataset. No checkpoint selection, early stopping, best-epoch selection or validation-based model selection was performed.

The Test set was not loaded or accessed in Run 06. It was reserved exclusively for the subsequent final evaluation.

---

### 2. Methodology

```text
RUN 06

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
│   ├── Global minimum = 22.000000
│   └── Global maximum = 36.812618
│
├── Class Weighting
│   ├── Calculated from all 197 development labels
│   ├── Healthy = 0.788000
│   └── Sick = 1.368056
│
├── Model Configuration
│   ├── Image input = 224 × 224
│   ├── 1-channel thermal image repeated to 3 channels
│   ├── ImageNet normalization
│   ├── ImageNet-pretrained ResNet18
│   ├── Frozen backbone
│   ├── Classifier = 512 → 64 → 1
│   ├── Dropout = 0.3
│   ├── Learning rate = 1e-3
│   ├── Batch size = 16
│   └── Epochs = 32
│
├── Training
│   ├── All 197 development images
│   ├── Random rotation = 18°
│   ├── Random affine scale = 0.95–1.05
│   ├── Class weighting enabled
│   ├── No validation data
│   ├── No checkpoint selection
│   └── No early stopping
│
└── Final Model
    └── Model state after epoch 32
```

The Training and Validation splits were combined before final model training.

The temperature normalization range was calculated once using all 197 development images. This range was then used to preprocess every development image.

The resulting global temperature range was:

```text
Global minimum = 22.000000
Global maximum = 36.812618
```

Balanced class weights were calculated from the labels of all 197 development images:

```text
Healthy = 0.788000
Sick    = 1.368056
```

The model was trained for exactly 32 epochs using the finalized configuration. The model state after epoch 32 was saved as the final model.

No model selection was performed during Run 06.

The classification threshold of 0.192174 was carried forward from the OOF threshold-selection stage. It was not applied during training. It will be used later when converting final Test-set probabilities into binary predictions.

---

### 3. Configuration

| Parameter                  | Value                       |
| -------------------------- | --------------------------- |
| Experiment                 | EXP02_DMR_Anterior_ResNet18 |
| Run                        | Run 06 Final Training       |
| Input dataset              | Development dataset         |
| Training split             | Train + Validation          |
| Development dataset        | 197 images                  |
| Healthy                    | 125                         |
| Sick                       | 72                          |
| Test set                   | Not used                    |
| Input image size           | 224 × 224                   |
| Input channels to ResNet18 | 3                           |
| Batch size                 | 16                          |
| Training epochs            | 32                          |
| Optimizer                  | Adam                        |
| Learning rate              | 1e-3                        |
| Dropout rate               | 0.3                         |
| ResNet18 weights           | ImageNet1K_V1               |
| Backbone                   | Frozen                      |
| Classifier                 | 512 → 64 → 1                |
| Trainable parameters       | 32,897                      |
| Total parameters           | 11,209,409                  |
| Random rotation            | 18°                         |
| Random affine scale        | 0.95–1.05                   |
| Horizontal flip            | False                       |
| Class weighting            | Balanced                    |
| Early stopping             | False                       |
| Checkpoint selection       | False                       |
| Model selection            | None                        |
| Random seed                | 10                          |
| Device                     | CUDA                        |
| GPU                        | NVIDIA Quadro M1200         |
| Classification threshold   | 0.192174                    |
| Test evaluation            | Not performed               |

The classification threshold of 0.192174 was carried forward from the OOF threshold-selection stage but was not used to influence the training process.

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

The class distribution was:

```text
Development Dataset

Healthy = 125
Sick    = 72
Total   = 197
```

Balanced class weights were calculated using the complete development label set:

```text
Healthy weight = 0.788000
Sick weight    = 1.368056
```

The corresponding positive-class weight used by `BCEWithLogitsLoss` was calculated from the ratio of the Sick and Healthy class weights.

No information from the Test split was used during dataset preparation.

---

### 5. Preprocessing

Each development thermal image was processed using the EXP02 thermal preprocessing pipeline implemented directly in the final-training code.

The preprocessing sequence was:

```text
Raw thermal image

        ↓

Load thermal temperature values

        ↓

Development-wide temperature normalization

        ↓

Pad to square

        ↓

Convert to 8-bit grayscale PIL image

        ↓

Resize to 224 × 224

        ↓

Random rotation

        ↓

Random affine scaling

        ↓

Convert to tensor

        ↓

Repeat 1 channel to 3 channels

        ↓

ImageNet normalization

        ↓

Final model input
```

Temperature normalization used the global minimum and maximum calculated from all 197 development images.

The recorded development temperature range was:

```text
Minimum temperature = 22.000000
Maximum temperature = 36.812618
```

For valid pixels, where the raw temperature value was greater than zero, normalization was performed using the development-wide minimum and maximum. The normalized values were clipped to the range 0 to 1.

Zero-valued pixels remained zero.

The normalized image was padded to a square before conversion to a PIL image.

The image was then resized to 224 × 224 pixels.

During training, the following augmentations were applied:

```text
Random Rotation = ±18°

Random Affine Scale = 0.95–1.05

Horizontal Flip = Disabled
```

The resulting single-channel image was repeated across three channels to match the three-channel input expected by the ImageNet-pretrained ResNet18.

ImageNet normalization was then applied using:

```text
Mean = [0.485, 0.456, 0.406]

Std  = [0.229, 0.224, 0.225]
```

No Test images were loaded or preprocessed during Run 06.

The development temperature range was saved separately so that the same normalization parameters can be used when preprocessing the untouched Test set during the subsequent final evaluation.

---

### 6. Final Model Training

The final EXP02 model used an ImageNet-pretrained ResNet18 architecture.

The complete ResNet18 backbone was frozen. Only the newly added classifier was trainable.

The model structure was:

```text
Input
224 × 224 × 3

        ↓

ImageNet-pretrained ResNet18

        ↓

Frozen convolutional backbone

        ↓

Adaptive Average Pooling

        ↓

512 features

        ↓

Linear
512 → 64

        ↓

ReLU

        ↓

Dropout
0.3

        ↓

Linear
64 → 1

        ↓

Output Logit
```

The model contained:

```text
Total parameters     = 11,209,409
Trainable parameters = 32,897
```

The finalized training configuration was:

```text
Optimizer       = Adam
Learning rate   = 1e-3
Batch size      = 16
Epochs          = 32
Dropout         = 0.3
Class weighting = Balanced
Seed            = 10
Device          = CUDA
```

A weighted `BCEWithLogitsLoss` was used for binary classification.

The final model was trained using all 197 development images.

The frozen ResNet18 backbone was kept in evaluation mode during training to prevent the BatchNorm layers in the frozen backbone from updating their running statistics.

Training was performed for exactly 32 epochs.

No validation data was supplied during training.

No early stopping was used.

No best epoch was calculated.

No checkpoint selection was performed.

Therefore, the model state after epoch 32 was directly treated as the final trained model.

The training loss recorded for each epoch was:

| Epoch | Training Loss |
| ----: | ------------: |
|     1 |      0.851118 |
|     2 |      0.800015 |
|     3 |      0.723242 |
|     4 |      0.715102 |
|     5 |      0.665308 |
|     6 |      0.591522 |
|     7 |      0.615454 |
|     8 |      0.576032 |
|     9 |      0.594585 |
|    10 |      0.542768 |
|    11 |      0.535997 |
|    12 |      0.595188 |
|    13 |      0.528103 |
|    14 |      0.474266 |
|    15 |      0.516859 |
|    16 |      0.489175 |
|    17 |      0.461668 |
|    18 |      0.481583 |
|    19 |      0.499728 |
|    20 |      0.524360 |
|    21 |      0.503394 |
|    22 |      0.497393 |
|    23 |      0.462457 |
|    24 |      0.470047 |
|    25 |      0.468058 |
|    26 |      0.460774 |
|    27 |      0.423109 |
|    28 |      0.466241 |
|    29 |      0.414987 |
|    30 |      0.424674 |
|    31 |      0.443622 |
|    32 |      0.491160 |

The final training loss at epoch 32 was **0.491160**.

---

### 7. Results

Run 06 was a final model training run rather than a model-selection or evaluation run.

The final training dataset contained:

| Parameter                  |     Result |
| -------------------------- | ---------: |
| Training images            |        163 |
| Original validation images |         34 |
| Development images         |        197 |
| Healthy                    |        125 |
| Sick                       |         72 |
| Training epochs            |         32 |
| Learning rate              |       1e-3 |
| Dropout                    |        0.3 |
| Batch size                 |         16 |
| Trainable parameters       |     32,897 |
| Total parameters           | 11,209,409 |
| Final training loss        |   0.491160 |
| Test images used           |          0 |

The training history recorded the training loss and learning rate for all 32 epochs.

No validation metrics were generated during Run 06 because no validation set was supplied during final training.

No Test-set predictions or evaluation metrics were generated.

The final model corresponds to the model state at the end of epoch 32.

The final model was trained using the NVIDIA Quadro M1200 GPU.

---

### 8. Final Run 06 Decision

The final EXP02 ResNet18 model was trained using all 197 development images with the configuration established through the preceding EXP02 runs.

| Parameter                | Value                        |
| ------------------------ | ---------------------------- |
| Development dataset      | 197 images                   |
| Healthy                  | 125                          |
| Sick                     | 72                           |
| Learning rate            | 1e-3                         |
| Dropout                  | 0.3                          |
| Fixed training epoch     | **32**                       |
| Batch size               | 16                           |
| Optimizer                | Adam                         |
| Architecture             | ImageNet-pretrained ResNet18 |
| Backbone                 | Frozen                       |
| Classifier               | 512 → 64 → 1                 |
| Data augmentation        | Rotation + affine scaling    |
| Class weighting          | Balanced                     |
| Model selection          | None                         |
| Checkpoint selection     | None                         |
| Early stopping           | None                         |
| Final training loss      | **0.491160**                 |
| Classification threshold | **0.192174**                 |
| Test set                 | **Not used**                 |

The model state after exactly 32 epochs was saved as the final EXP02 model.

The threshold of 0.192174 was carried forward from the OOF threshold-selection stage. It was not used during model training and did not affect the optimization process.

The final Test set remained completely untouched.

The saved Run 06 model will be used for the subsequent final evaluation on the independent 37-image Test set.

---

### 9. Output Structure

```text
Run_06_Final_Training/

├── README.md

├── results/

│   ├── temperature_range.json

│   ├── training_history.json

│   ├── run_config.json

│   └── development_manifest.csv

├── model/

│   └── final_model_epoch32.pth

└── plots/

    └── training_loss.png
```

#### `temperature_range.json`

Stores the development-wide temperature normalization parameters:

* Global minimum = 22.000000
* Global maximum = 36.812618
* Number of files = 197
* Range source = all development images

#### `training_history.json`

Contains the training history recorded over all 32 epochs, including:

* Training loss
* Learning rate

No validation history is present because validation data was not used during final training.

#### `run_config.json`

Stores the Run 06 configuration, including:

* Experiment and run names
* Random seed
* Device and GPU information
* Dataset information
* Development split information
* Input dimensions
* Batch size
* Number of epochs
* Optimizer
* Learning rate
* Dropout rate
* ResNet18 architecture
* ImageNet pretrained weights
* Frozen-backbone status
* Trainable parameter count
* Data augmentation
* Class weights
* Loss function
* Temperature normalization method
* Development temperature range
* Model selection status
* Checkpointing status
* Test-set usage status
* Classification threshold carried forward from the OOF threshold-selection stage

#### `development_manifest.csv`

Contains the complete list of the 197 development images used for final training, including:

* Filepath
* Numeric label
* Class name

This provides a record of exactly which images were used to train the final model.

#### `final_model_epoch32.pth`

Contains the final PyTorch model checkpoint, including:

* Model state dictionary
* Optimizer state dictionary
* Training epoch
* Learning rate
* Batch size
* Dropout
* Image size
* Classification threshold
* Random seed
* Model architecture
* Parameter counts
* Development dataset counts
* Temperature normalization range
* Class weights

The checkpoint represents the model state after exactly 32 training epochs.

#### `training_loss.png`

Shows the training loss across the 32 training epochs.

No validation curves were generated because no validation dataset was used during Run 06.

---

### 10. Final Summary

* Experiment: **EXP02_DMR_Anterior_ResNet18**
* Run: **Run 06 Final Training**
* Development images: **197**
* Healthy images: **125**
* Sick images: **72**
* Original Training images: **163**
* Original Validation images: **34**
* Input size: **224 × 224**
* Model input channels: **3**
* Architecture: **ImageNet-pretrained ResNet18**
* Backbone: **Frozen**
* Classifier: **512 → 64 → 1**
* Total parameters: **11,209,409**
* Trainable parameters: **32,897**
* Batch size: **16**
* Learning rate: **1e-3**
* Dropout: **0.3**
* Optimizer: **Adam**
* Training epochs: **32**
* Class weighting: **Balanced**
* Data augmentation: **Random Rotation + Random Affine Scaling**
* Temperature range: **22.000000 to 36.812618**
* Final training loss: **0.491160**
* Model selection: **None**
* Checkpoint selection: **None**
* Early stopping: **None**
* Final model state: **After epoch 32**
* Classification threshold: **0.192174**
* Test set: **Not used**
* Device: **CUDA**
* GPU: **NVIDIA Quadro M1200**

Run 06 successfully trained the final EXP02 ResNet18 model using the complete 197-image development dataset with the fixed **32-epoch** training duration established through the preceding EXP02 model-selection stages.

The final model state after exactly **32 epochs** was saved as `final_model_epoch32.pth`.

The development temperature normalization range and complete development file manifest were also preserved to support reproducible preprocessing and documentation during the subsequent Test-set evaluation.

No Test-set images, predictions or evaluation metrics were used in Run 06.

The untouched 37-image Test set will be accessed only in the subsequent final evaluation run using the saved Run 06 model and the fixed classification threshold of **0.192174**.