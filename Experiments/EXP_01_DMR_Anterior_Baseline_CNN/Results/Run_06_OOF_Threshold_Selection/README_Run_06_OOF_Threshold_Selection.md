# EXP01_DMR_Anterior_Baseline

## Run 06: OOF Threshold Selection

### 1. Objective

The objective of Run 06 was to select the final classification threshold for the EXP01 baseline CNN using the complete out-of-fold (OOF) prediction set generated in Run 05.

Run 05 generated OOF probabilities for all 197 images in the development dataset using the fixed configuration selected in Runs 02–04:

* Learning rate = 1e-3
* Dropout rate = 0.3
* Fixed training epoch = 89

Run 06 evaluated classification thresholds from 0.000 to 1.000 at increments of 0.001.

A threshold was considered valid only if it satisfied both of the following conditions:

* Sensitivity > 0.90
* Specificity > 0.60

Among all valid thresholds, the selection criterion was highest sensitivity. If sensitivity was tied, the threshold with the highest specificity was selected. If both sensitivity and specificity were tied, the higher threshold was selected as the final deterministic tie-breaker.

The test set was not used in Run 06.

---

### 2. Methodology

```text
RUN 06

│
├── Input
│   └── Run 05 OOF predictions
│
├── Development Dataset
│   ├── Total = 197 images
│   ├── Healthy = 125
│   ├── Sick = 72
│   └── Test = NOT USED
│
├── OOF Predictions
│   ├── Fold 1
│   ├── Fold 2
│   ├── Fold 3
│   ├── Fold 4
│   └── Fold 5
│
├── Threshold Evaluation
│   ├── Minimum threshold = 0.000
│   ├── Maximum threshold = 1.000
│   └── Step = 0.001
│
├── Constraint Filtering
│   ├── Sensitivity > 0.90
│   └── Specificity > 0.60
│
├── Threshold Selection
│   ├── Highest sensitivity
│   ├── Tie-breaker = highest specificity
│   └── Final tie-breaker = highest threshold
│
└── Selected Threshold
    └── 0.586
```

The OOF probabilities from Run 05 were treated as continuous prediction scores. Each candidate threshold was applied to all 197 OOF probabilities to generate binary predictions.

For each threshold, the following metrics were calculated:

* Sensitivity
* Specificity
* Accuracy
* Precision
* F1-score
* Confusion matrix components: TN, FP, FN, TP

The threshold selection was performed entirely on the OOF predictions. No model retraining was performed in Run 06.

---

### 3. Configuration

| Parameter           | Value                  |
| ------------------- | ---------------------- |
| Input               | Run 05 OOF predictions |
| Development dataset | 197 images             |
| Healthy             | 125                    |
| Sick                | 72                     |
| OOF folds           | 5                      |
| Threshold minimum   | 0.000                  |
| Threshold maximum   | 1.000                  |
| Threshold step      | 0.001                  |
| Minimum sensitivity | > 0.90                 |
| Minimum specificity | > 0.60                 |
| Selection criterion | Highest sensitivity    |
| Tie-breaker         | Highest specificity    |
| Final tie-breaker   | Highest threshold      |
| Test set            | Not used               |

The overall OOF ROC-AUC of the prediction set was calculated before threshold selection.

---

### 4. Threshold Evaluation and Selection

The complete OOF prediction set contained 197 predictions:

* Healthy = 125
* Sick = 72
* Fold assignments = 1–5
* Missing OOF predictions = 0

The overall OOF ROC-AUC was:

**0.880111**

Thresholds were evaluated from 0.000 to 1.000 in increments of 0.001.

A threshold was retained as valid only when:

```text
Sensitivity > 0.90
AND
Specificity > 0.60
```

A total of **138 thresholds** satisfied both constraints.

The final threshold was selected using the following hierarchy:

```text
1. Highest sensitivity
2. Highest specificity if sensitivity was tied
3. Highest threshold if both were tied
```

The selected threshold was:

**0.586**

---

### 5. OOF Prediction Evaluation

The selected threshold produced the following results on the complete 197-image OOF prediction set:

| Metric      |  Value |
| ----------- | -----: |
| Threshold   |  0.586 |
| Sensitivity | 0.9444 |
| Specificity | 0.6160 |
| Accuracy    | 0.7360 |
| Precision   | 0.5862 |
| F1-score    | 0.7234 |
| TN          |     77 |
| FP          |     48 |
| FN          |      4 |
| TP          |     68 |

The confusion matrix was therefore:

```text
                 Predicted
              Healthy   Sick
Actual Healthy    77      48
Actual Sick       4       68
```

The selected threshold produced a sensitivity of 0.9444 and specificity of 0.6160, satisfying both predefined threshold constraints.

---

### 6. Selection Rule

The threshold selection was constraint-based rather than based on maximum accuracy, F1-score, or ROC-AUC.

The implemented selection rule was:

```python
selected_result = max(
    valid_results,
    key=lambda x: (
        x["sensitivity"],
        x["specificity"],
        x["threshold"],
    ),
)
```

Therefore, the threshold was selected according to the following priority:

```text
Highest sensitivity
        ↓
Highest specificity
        ↓
Highest threshold
```

The OOF ROC-AUC was used to describe the overall discrimination of the model predictions, but it was not used to select the classification threshold.

---

### 7. Results

#### OOF Prediction Summary

| Parameter            |   Result |
| -------------------- | -------: |
| OOF predictions used |      197 |
| Healthy              |      125 |
| Sick                 |       72 |
| Number of folds      |        5 |
| Overall OOF ROC-AUC  | 0.880111 |
| Valid thresholds     |      138 |
| Test set used        |       No |

#### Selected Threshold Results

| Metric             | Result |
| ------------------ | -----: |
| Selected threshold |  0.586 |
| Sensitivity        | 0.9444 |
| Specificity        | 0.6160 |
| Accuracy           | 0.7360 |
| Precision          | 0.5862 |
| F1-score           | 0.7234 |
| TN                 |     77 |
| FP                 |     48 |
| FN                 |      4 |
| TP                 |     68 |

The selected threshold satisfies:

```text
Sensitivity = 0.9444 > 0.90
Specificity = 0.6160 > 0.60
```

No test-set predictions were used for threshold selection.

---

### 8. Final Run 06 Decision

The final threshold selected from the complete OOF prediction set was **0.586**.

| Parameter              | Value       |
| ---------------------- | ----------- |
| OOF source             | Run 05      |
| Development dataset    | 197 images  |
| OOF predictions        | 197 / 197   |
| Overall OOF ROC-AUC    | 0.880111    |
| Threshold search range | 0.000–1.000 |
| Threshold step         | 0.001       |
| Minimum sensitivity    | > 0.90      |
| Minimum specificity    | > 0.60      |
| Valid thresholds       | 138         |
| Selected threshold     | **0.586**   |
| Sensitivity            | **0.9444**  |
| Specificity            | **0.6160**  |
| Accuracy               | 0.7360      |
| Precision              | 0.5862      |
| F1-score               | 0.7234      |
| Test set               | Not used    |

The threshold of **0.586** will be carried forward as the fixed classification threshold for the final model evaluation in Run 07.

---

### 9. Output Structure

```text
Run_06_OOF_Threshold_Selection/

├── results/
│
│   ├── threshold_analysis.csv
│   └── selected_threshold.json
│
└── plots/
    │
    ├── sensitivity_specificity_vs_threshold.png
    └── valid_thresholds.png
```

#### `threshold_analysis.csv`

Contains the evaluation results for all thresholds from 0.000 to 1.000 at 0.001 increments, including:

* Threshold
* Sensitivity
* Specificity
* Accuracy
* Precision
* F1-score
* TN
* FP
* FN
* TP
* Whether the threshold satisfies both constraints

#### `selected_threshold.json`

Stores the selected threshold, selection method, constraints, performance metrics, confusion matrix values, number of OOF predictions, overall OOF ROC-AUC, and confirmation that the test set was not used.

#### Plots

Two plots were generated:

1. **Sensitivity and specificity vs classification threshold**
2. **Thresholds satisfying sensitivity and specificity constraints**

---

### 10. Final Summary

* OOF source: Run 05
* Development images: 197
* Healthy images: 125
* Sick images: 72
* OOF folds: 5
* Overall OOF ROC-AUC: **0.880111**
* Threshold range: 0.000–1.000
* Threshold step: 0.001
* Minimum sensitivity: > 0.90
* Minimum specificity: > 0.60
* Valid thresholds: **138**
* Selected threshold: **0.586**
* Sensitivity: **0.9444**
* Specificity: **0.6160**
* Accuracy: **0.7360**
* Precision: **0.5862**
* F1-score: **0.7234**
* TN: 77
* FP: 48
* FN: 4
* TP: 68
* Test set: **Not used**

Run 06 successfully selected a fixed classification threshold of **0.586** from the complete 197-image OOF prediction set generated in Run 05. The selected threshold satisfies the predefined sensitivity and specificity constraints and will be used in Run 07 for final evaluation on the untouched test set.