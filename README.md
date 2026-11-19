# Credit Scoring Classification with AdaBoost

A machine learning project for predicting credit default risk (`loan_status`) using an **AdaBoostClassifier** pipeline with hyperparameter tuning (`GridSearchCV`) and classification threshold optimization.

## Project Overview

The goal of this project is to build a reliable binary classification model capable of accurately identifying high-risk borrowers while preventing data leakage through a unified `scikit-learn` preprocessing and modeling pipeline.

### Key Highlights
* **Leak-Free Pipeline:** Combined `ColumnTransformer` (`OneHotEncoder` for categorical features, `StandardScaler` for continuous numerical features, and `passthrough` for binary/ratio features) with `AdaBoostClassifier` inside a single `Pipeline`.
* **Hyperparameter Tuning:** Optimized `n_estimators`, `learning_rate`, and base `DecisionTreeClassifier` depth (`max_depth`) using 3-fold cross-validation (`GridSearchCV`) with `ROC-AUC` scoring.
* **Threshold Optimization:** Tuned the decision probability threshold to maximize the `F1-score` and balance `Precision` and `Recall` for the minority class.

---

## Model Performance

Evaluated on the held-out test set (`9,000` samples):

| Metric | Score |
| :--- | :--- |
| **Cross-Validation ROC-AUC** | `0.9757` |
| **Test ROC-AUC** | `0.9760` |
| **Test Accuracy** | `0.93` |
| **Class 1 Precision (Tuned Threshold)** | `0.84` |
| **Class 1 Recall (Tuned Threshold)** | `0.84` |
| **Class 1 F1-Score (Tuned Threshold)** | `0.84` |

### Confusion Matrix (Test Set, Tuned Threshold)
```text
[[6676,  314],
 [ 313, 1697]]
```

---

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/SamvelGalstyan15/CreditScoringAdaBoost
   cd CreditScoringAdaBoost

   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


---

## Usage (Inference with Pre-trained Pipeline)

The saved artifact (`best_model.joblib`) contains both the fitted `ColumnTransformer` and the tuned `AdaBoostClassifier`, allowing direct inference on raw input data.

* [Download Pre-trained Pipeline (best_model.joblib)](YOUR_GOOGLE_DRIVE_OR_RELEASE_LINK_HERE)

```python
import joblib
import pandas as pd

pipeline = joblib.load('best_model.joblib')

new_data = pd.read_csv('new_borrowers.csv')

probabilities = pipeline.predict_proba(new_data)[:, 1]

OPTIMAL_THRESHOLD = 0.45
predictions = (probabilities >= OPTIMAL_THRESHOLD).astype(int)
```

---


