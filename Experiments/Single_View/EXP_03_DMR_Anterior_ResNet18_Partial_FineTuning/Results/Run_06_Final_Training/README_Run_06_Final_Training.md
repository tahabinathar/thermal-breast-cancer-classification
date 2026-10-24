# EXP03_DMR_Anterior_ResNet18_Partial_FineTuning

## Run 06: Final Model Training

---

### 1. Objective

Run 06 performs the final model training for EXP03 after the model configuration, fixed training epoch and classification threshold were established in the preceding runs.

The final ResNet-18 model is trained on all 197 development images, consisting of the original Train and Validation sets combined.

The final configuration is:

* ImageNet-pretrained ResNet-18
* Frozen `conv1`, `bn1`, `layer1`, `layer2` and `layer3`
* Trainable `layer4` and classifier
* Layer4 learning rate = `0.0001`
* Classifier learning rate = `0.001`
* Dropout = `0.3`
* Fixed training duration = `10 epochs`
* Batch size = `16`
* Seed = `10`
* Weighted BCEWithLogitsLoss
* Selected classification threshold = `0.027535`

The 10-epoch training duration was selected in Run 03 using 5-fold validation ROC-AUC. The classification threshold of `0.027535` was selected in Run 05 using out-of-fold predictions.

The test set was not loaded, accessed or evaluated during Run 06.

---

### 2. Methodology

The final training procedure follows the configuration established through the preceding EXP03 runs.

```text
Development Dataset
        │
        ├── Original Train: 163 images
        │       ├── Healthy = 103
        │       └── Sick    = 60
        │
        └── Original Validation: 34 images
                ├── Healthy = 22
                └── Sick    = 12
        │
        ▼
Combined Development Set
        │
        ├── Healthy = 125
        ├── Sick    = 72
        └── Total   = 197
        │
        ▼
Development-only Temperature Range
        │
        ├── Minimum = 22.000000
        └── Maximum = 36.812618
        │
        ▼
Preprocessing
        │
        ├── Temperature normalization
        ├── Padding / resizing to 224 × 224
        ├── Training augmentation
        ├── Grayscale → 3 channels
        └── ImageNet normalization
        │
        ▼
ImageNet-pretrained ResNet-18
        │
        ├── Frozen: conv1 + bn1
        ├── Frozen: layer1
        ├── Frozen: layer2
        ├── Frozen: layer3
        ├── Trainable: layer4
        └── Trainable: classifier
        │
        ▼
Weighted BCEWithLogitsLoss
        │
        ▼
10 Epochs of Final Training
        │
        ▼
Final Model
        │
        ▼
Classification threshold = 0.027535
        │
        ▼
Saved for final test evaluation
```

The test set remains completely isolated from the final training process.

---

### 3. Configuration

| Parameter | Value |
|---|---|
| Experiment | EXP03_DMR_Anterior_ResNet18_Partial_FineTuning |
| Run | Run 06 - Final Training |
| Model | ResNet-18 |
| Pretrained Weights | ImageNet1K_V1 |
| Image Size | 224 × 224 |
| Input Channels | 3 |
| Batch Size | 16 |
| Epochs | 10 |
| Seed | 10 |
| Device | CUDA |
| GPU | NVIDIA Quadro M1200 |
| Frozen Layers | conv1, bn1, layer1, layer2, layer3 |
| Trainable Layers | layer4 + classifier |
| Layer4 Learning Rate | 0.0001 |
| Classifier Learning Rate | 0.001 |
| Dropout | 0.3 |
| Loss Function | Weighted BCEWithLogitsLoss |
| Development Images | 197 |
| Healthy Images | 125 |
| Sick Images | 72 |
| Selected Threshold | 0.027535 |
| Test Set | Not Used |

---

### 4. Development Dataset Preparation

The final model was trained using the complete development dataset.

The original Train and Validation sets were combined:

| Original Split | Healthy | Sick | Total |
|---|---|---|---|
| Train | 103 | 60 | 163 |
| Validation | 22 | 12 | 34 |
| Development Total | 125 | 72 | 197 |

The original Test set contained 37 images:

```text
Healthy = 24
Sick = 13
```

The Test set was not loaded or accessed during Run 06.

The final model therefore uses all available development data while preserving the Test set for the subsequent final evaluation.

#### Development Temperature Range

The temperature normalization range was calculated exclusively from the 197 development images.

```text
Global minimum temperature = 22.000000
Global maximum temperature = 36.812618
```

This development-wide range is saved and will be used consistently when processing the held-out Test set during the final evaluation.

#### Class Distribution

```text
Healthy = 125
Sick    = 72
Total   = 197
```

#### Class Weights

Weighted binary cross-entropy was used to compensate for the class imbalance.

The calculated weights were:

```text
Healthy (class 0) = 0.788000
Sick    (class 1) = 1.368056
```

The resulting positive-class weighting corresponds to:

```text
pos_weight = 1.736111
```

---

### 5. Preprocessing

The final training preprocessing follows the EXP03 preprocessing configuration established during cross-validation.

#### Temperature Processing

The raw thermal values were normalized using the development-only temperature range:

```text
Minimum = 22.000000
Maximum = 36.812618
```

The range was calculated using all 197 development images.

#### Spatial Processing

Thermal images were padded and resized to:

```text
224 × 224 pixels
```

#### Training Augmentation

The training transform includes:

* Random rotation up to ±18°
* Random affine scaling from 0.95 to 1.05
* Conversion to tensor
* Repetition of the single thermal channel to 3 channels
* ImageNet normalization

No horizontal flip was used.

#### Input Format

The final ResNet-18 receives:

```text
224 × 224 × 3
```

The thermal image is originally single-channel and is repeated across three channels to match the ImageNet-pretrained ResNet-18 input format.

---

### 6. Final Model Training

The final model was constructed from ImageNet-pretrained ResNet-18 weights.

#### Model Structure

```text
Input
224 × 224 × 3
        │
        ▼
ImageNet-pretrained ResNet-18
        │
        ├── conv1                    [Frozen]
        ├── bn1                      [Frozen]
        ├── layer1                   [Frozen]
        ├── layer2                   [Frozen]
        ├── layer3                   [Frozen]
        │
        ├── layer4                   [Trainable]
        │
        ▼
Adaptive Average Pooling
        │
        ▼
512 features
        │
        ▼
Linear: 512 → 64
        │
        ▼
ReLU
        │
        ▼
Dropout: 0.3
        │
        ▼
Linear: 64 → 1
        │
        ▼
Output Logit
```

#### Parameter Count

```text
Total parameters      = 11,209,409
Trainable parameters  =  8,426,625
Frozen parameters     =  2,782,784
```

Only layer4 and the new classifier are updated during training.

#### Final Training Configuration

```text
Training images       = 197
Epochs                = 10
Batch size            = 16
Layer4 learning rate  = 0.0001
Classifier LR         = 0.001
Dropout               = 0.3
Seed                  = 10
Device                = CUDA
GPU                   = Quadro M1200
Loss                  = Weighted BCEWithLogitsLoss
```

#### Training Loss

| Epoch | Training Loss | Layer4 LR | Classifier LR |
|---|---|---|---|
| 1 | 0.773133 | 0.000100 | 0.001000 |
| 2 | 0.568736 | 0.000100 | 0.001000 |
| 3 | 0.501632 | 0.000100 | 0.001000 |
| 4 | 0.399438 | 0.000100 | 0.001000 |
| 5 | 0.328628 | 0.000100 | 0.001000 |
| 6 | 0.247002 | 0.000100 | 0.001000 |
| 7 | 0.280955 | 0.000100 | 0.001000 |
| 8 | 0.296552 | 0.000100 | 0.001000 |
| 9 | 0.235190 | 0.000100 | 0.001000 |
| 10 | 0.172078 | 0.000100 | 0.001000 |

The final model state after epoch 10 was saved as the final model.

No validation-based model selection or early stopping was performed during Run 06 because the complete development set was used for final training.

---

### 7. Results

#### Final Training Result

The model completed the predefined 10 training epochs.

```text
Initial training loss = 0.773133
Final training loss   = 0.172078
Final epoch           = 10
```

The training loss decreased from 0.773133 at epoch 1 to 0.172078 at epoch 10.

The final model contains:

```text
Trainable layers:
    Layer4
    Classifier

Frozen layers:
    conv1
    bn1
    layer1
    layer2
    layer3

Trainable parameters = 8,426,625
```

#### Final Temperature Range

```text
Global minimum = 22.000000
Global maximum = 36.812618
```

#### Selected Classification Threshold

The threshold was determined before final training from the OOF predictions generated in Run 04 and evaluated in Run 05.

```text
Selected threshold = 0.027535
```

The threshold is not used during gradient-based model training. It is carried forward for conversion of the final model output probabilities into binary predictions during the subsequent Test evaluation.

#### Test Set Isolation

```text
Test set loaded during Run 06: NO
Test set evaluated during Run 06: NO
Test images used for training: NO
```

The Test set remains completely untouched for the final evaluation stage.

---

### 8. Final Run 06 Decision

Run 06 successfully produced the final EXP03 model using the complete 197-image development dataset.

The final configuration was fixed before training:

```text
Architecture:
    ImageNet-pretrained ResNet-18
    Partial fine-tuning of Layer4 + classifier

Training:
    197 development images
    10 epochs
    Batch size = 16
    Layer4 LR = 0.0001
    Classifier LR = 0.001
    Weighted BCEWithLogitsLoss

Model:
    Total parameters = 11,209,409
    Trainable parameters = 8,426,625

Threshold:
    0.027535
```

The final model is the model state obtained after exactly 10 epochs.

No model selection was performed using the Test set.

The resulting checkpoint is ready for the subsequent final Test evaluation.

---

### 9. Output Structure

```text
Run_06_Final_Training/
│
├── README.md
│
├── model/
│   └── final_model_epoch10.pth
│
└── results/
    ├── temperature_range.json
    ├── training_history.json
    ├── run_config.json
    └── development_manifest.csv
```

#### Output Files

| File | Description |
|---|---|
| README.md | Documentation for Run 06 |
| final_model_epoch10.pth | Final ResNet-18 model checkpoint after 10 epochs |
| temperature_range.json | Development-only global temperature normalization range |
| training_history.json | Training loss and learning-rate history |
| run_config.json | Complete Run 06 configuration |
| development_manifest.csv | Manifest of the 197 development images used for final training |

---

### 10. Final Summary

Run 06 completed the final training stage of EXP03 using all 197 development images.

The final model is an ImageNet-pretrained ResNet-18 with partial fine-tuning. The convolutional stem and first three residual blocks were frozen, while layer4 and the classifier were trainable.

```text
Development dataset = 197 images
    Healthy = 125
    Sick    = 72

Input size = 224 × 224 × 3

Model:
    ResNet-18
    ImageNet pretrained
    Layer4 + classifier trainable

Parameters:
    Total      = 11,209,409
    Trainable  = 8,426,625

Training:
    Epochs              = 10
    Batch size          = 16
    Layer4 LR           = 0.0001
    Classifier LR       = 0.001
    Dropout             = 0.3
    Seed                = 10
    Device              = CUDA
    GPU                 = Quadro M1200

Temperature range:
    Minimum = 22.000000
    Maximum = 36.812618

Training loss:
    Epoch 1  = 0.773133
    Epoch 10 = 0.172078

Selected threshold:
    0.027535

Test set:
    NOT USED
```

The final checkpoint is:

```text
final_model_epoch10.pth
```

This model is the fixed EXP03 model to be used for the subsequent final Test evaluation. The Test set remains completely isolated from all training and model-selection procedures up to this point.
