# Decision Tree & Ensemble Learning for Imbalanced Data

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Decision%20Trees-orange.svg)]()

This repository contains implementations of **Decision Tree** and **Ensemble Learning** algorithms specifically designed for handling **imbalanced datasets**. The project was developed as part of a Machine Learning course assignment (Spring 2025) and includes custom implementations from scratch without relying on pre-built ML libraries for the core algorithms (e.g., no `sklearn` for tree logic).

## Project Objectives

1. Implement **Hellinger Distance Decision Tree (HDDT)** suitable for imbalanced data.
2. Develop **Bagging with Undersampling** ensemble method.
3. Implement **AdaBoost.M1 with SMOTE** for handling class imbalance.
4. Evaluate performance using appropriate metrics for imbalanced datasets (Precision, Recall, F-measure, AUC, G-mean).

## Project Structure
```
├── data/
│   ├── Covid19HDDT.csv     	  # Dataset for Part A (Decision Tree)
│   ├── Covid.csv         	  # Dataset for Part B Method I (Bagging)
│   └── abalone.data              # Dataset for Part B Method II (AdaBoost + SMOTE)
│  
├── result/
│   ├── bagging_results.csv       # Results from Bagging experiments
│   ├── method_II_results.csv     # Results from AdaBoost + SMOTE experiments
│   
├── code.ipynb
└── README.md                 	  # This file
```


## Implementation Details

### Part A: Hellinger Distance Decision Tree (HDDT)

**Dataset:** `Covid19HDDT.csv` (3-class imbalanced dataset)

**Key Features:**
- Custom HDDT implementation from scratch.
- **Hellinger Distance** used as the split criterion (robust to class imbalance).
- Support for both binary and multi-class classification.
- **OVO (One-vs-One)** and **OVA (One-vs-All)** strategies implemented for multi-class.
- Tree pruning with configurable max heights `{2, 3, 4, 5}`.

**Evaluation Metrics:**
- Precision, Recall, F-measure (for minority class).
- AUC (Area Under ROC Curve).
- G-mean (Geometric Mean of sensitivity and specificity).

### Part B: Ensemble Learning

#### Method I: Bagging with Undersampling

**Dataset:** `Covid.csv` (imbalanced binary classification)

**Implementation:**
- Bootstrap sampling from minority class.
- Random undersampling from majority class to balance each bootstrap sample.
- Base learners: **HDDT** and **Decision Stump**.
- Configurable number of estimators: `{11, 31, 51, 101}`.
- Evaluated over 10 independent runs.

#### Method II: AdaBoost with SMOTE

**Dataset:** `Abalone` (UCI Machine Learning Repository)

**Implementation:**
- **SMOTE (Synthetic Minority Over-sampling Technique)** implemented from scratch to generate synthetic samples.
- **AdaBoost.M1** with Decision Stump base learners.
- **5-fold cross-validation** repeated 5 times (25 total runs).
- **Preprocessing:** One-Hot Encoding for categorical features (Sex), Standard Scaling for numerical features.
- **Task:** Binary classification (≤7 rings vs >7 rings).

**Bonus Experiments:**
- Varying the number of synthetic samples (`N`) to analyze overfitting vs. performance trade-offs.
- Visualizing decision boundaries and SMOTE synthetic samples.

## Results Summary

### HDDT Performance (Part A)

| Method       | Accuracy | Precision | Recall | F-measure | AUC   | G-mean |
|--------------|----------|-----------|--------|-----------|-------|--------|
| **OVA**      | 0.988    | 0.971     | 0.945  | 0.957     | 0.969 | 0.968  |
| **OVO**      | 0.986    | 0.942     | 0.872  | 0.905     | 0.933 | 0.931  |
| **Two-class**| 0.986    | 0.942     | 0.872  | 0.905     | 0.933 | 0.931  |

*Observation: One-vs-All (OVA) approach showed slightly better performance than One-vs-One (OVO).*

### Bagging Performance (Part B - Method I)

| Base Learner | T   | Precision | Recall | F-measure | AUC   | G-mean |
|--------------|-----|-----------|--------|-----------|-------|--------|
| **HDDT**     | 11  | 0.994     | 0.733  | 0.844     | 0.837 | 0.827  |
| **HDDT**     | 101 | 0.995     | 0.740  | 0.849     | 0.850 | 0.835  |
| **Stump**    | 11  | 0.999     | 0.692  | 0.817     | 0.840 | 0.826  |
| **Stump**    | 101 | 1.000     | 0.692  | 0.818     | 0.842 | 0.830  |

*Observation: HDDT base learners generally achieved higher Recall and G-mean compared to Decision Stumps.*

### AdaBoost + SMOTE Performance (Part B - Method II)

| Method                 | Accuracy| AUROC  | AUPR   | G-mean |
|------------------------|---------|--------|--------|--------|
| **AdaBoost + SMOTE**   | 0.867   | 0.874  | 0.577  | 0.870  |
| **Baseline (AdaBoost)**| 0.905   | 0.939  | 0.820  | 0.837  |

*Observation: SMOTE improved G-mean (better minority class detection) but decreased overall accuracy compared to the baseline.*

## Installation & Usage

### Requirements

```bash
pip install numpy pandas scikit-learn matplotlib tabulate scipy


## Algorithm Highlights

### Hellinger Distance
$$
HD(p, q) = \sqrt{\sum(\sqrt{p_i} - \sqrt{q_i})^2}
$$
- More robust to class imbalance than Gini or Entropy.
- Doesn't require smoothing for zero probabilities.
- Naturally handles skewed class distributions.

### SMOTE Formula
$$
x_{new} = x_i + \lambda \times (x_{nn} - x_i), \quad \text{where } \lambda \in [0, 1]
$$
- Generates synthetic samples in feature space.
- Helps decision boundary learning.
- Reduces bias towards majority class.

### AdaBoost Weight Update
$$
w_i = w_i \times \exp(-\alpha \times y_i \times h(x_i))
$$
- Focuses on misclassified samples.
- Combines weak learners into strong classifier.
- Works well with SMOTE-balanced data.

## References
1. UCI Machine Learning Repository - Abalone Dataset
2. Hellinger Distance Decision Trees for Imbalanced Data
3. SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)
4. AdaBoost.M1: Freund & Schapire (1996)