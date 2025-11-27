# 📊 How to See Model Results - Complete Guide

## Overview

I've created a comprehensive **evaluation module** that allows you to see detailed results for XGBoost and all other models. This guide shows you how to use it.

---

## ✅ What Was Created

### 1. **Evaluation Module** - `src/evaluation.py`

A complete evaluation system with:
- **Metrics Calculation**: AUC-ROC, Accuracy, Precision, Recall, F1-Score, and 10+ more
- **Visualizations**:
  - ROC Curves
  - Precision-Recall Curves
  - Confusion Matrices (raw and normalized)
  - Threshold Analysis Plots
  - Model Comparison Charts
- **Reports**: CSV reports with all metrics
- **Threshold Optimization**: Find optimal thresholds for different metrics

### 2. **Demo Scripts**

**`scripts/demo_xgboost_evaluation.py`** - Full training and evaluation demo
**`scripts/quick_test.py`** - Quick test with synthetic data

---

## 🚀 How to Use the Evaluation Module

### Method 1: Quick Evaluation (Simplest)

```python
from src.evaluation import quick_evaluate
import numpy as np

# Assuming you have true labels and predictions
y_true = np.array([0, 1, 1, 0, 1, ...])
y_pred_proba = np.array([0.2, 0.8, 0.9, 0.1, 0.7, ...])

# This will calculate metrics AND generate all plots
metrics = quick_evaluate(
    y_true,
    y_pred_proba,
    model_name="XGBoost",
    threshold=0.5,
    show_plots=True
)

print(metrics)
```

### Method 2: Full Control (Recommended)

```python
from src.evaluation import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Step 1: Calculate metrics
metrics = evaluator.evaluate(
    y_true=y_test,
    y_pred_proba=predictions,
    threshold=0.5,
    model_name="XGBoost"
)

# Step 2: Generate all visualizations
figures = evaluator.evaluate_all_plots(
    y_true=y_test,
    y_pred_proba=predictions,
    model_name="XGBoost",
    save=True  # Saves to results/figures/
)

# Step 3: View metrics
print(f"AUC-ROC: {metrics['roc_auc']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
```

### Method 3: Compare Multiple Models

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate multiple models
results = {}

# XGBoost
xgb_metrics = evaluator.evaluate(y_test, xgb_predictions, model_name="XGBoost")
results['XGBoost'] = xgb_metrics

# LightGBM (when implemented)
lgb_metrics = evaluator.evaluate(y_test, lgb_predictions, model_name="LightGBM")
results['LightGBM'] = lgb_metrics

# Generate comparison plot
evaluator.compare_models(results, save=True)

# Generate CSV report
report_df = evaluator.generate_report(
    results,
    output_path='results/reports/model_comparison.csv'
)

print(report_df)
```

---

## 📝 Complete Example: Evaluate XGBoost on Your Data

Here's a complete script to evaluate XGBoost on your actual dataset:

```python
import sys
sys.path.insert(0, 'credit_risk_fyp')

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_DIR, DATASET_CONFIG
from src.data_loader import load_data
from src.preprocessor import DataPreprocessor
from src.feature_engineer import FeatureEngineer
from src.models.xgboost_model import XGBoostModel
from src.evaluation import ModelEvaluator

# 1. Load data
print("Loading data...")
train_path = RAW_DATA_DIR / DATASET_CONFIG['train_dataset']
df = load_data(train_path, nrows=10000)  # Use 10k samples for quick test

# 2. Prepare data
target_col = DATASET_CONFIG['target_column']
id_col = DATASET_CONFIG['id_column']

X = df.drop(columns=[target_col, id_col])
y = df[target_col]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Preprocess
print("Preprocessing...")
preprocessor = DataPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_test_processed = preprocessor.transform(X_test)

# 4. Feature engineering
print("Feature engineering...")
feature_engineer = FeatureEngineer()
X_train_final = feature_engineer.fit_transform(X_train_processed)
X_test_final = feature_engineer.transform(X_test_processed)

# 5. Train model
print("Training XGBoost...")
model = XGBoostModel()
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_final, y_train, test_size=0.2, random_state=42
)
model.train(X_tr, y_tr, X_val, y_val)

# 6. Predict
print("Making predictions...")
y_pred_proba = model.predict_proba(X_test_final)

# 7. EVALUATE AND SEE RESULTS
print("Evaluating model...")
evaluator = ModelEvaluator()

# Calculate metrics
metrics = evaluator.evaluate(
    y_test, y_pred_proba,
    threshold=0.5,
    model_name="XGBoost"
)

# Generate all visualizations
evaluator.evaluate_all_plots(
    y_test, y_pred_proba,
    model_name="XGBoost",
    save=True
)

# Print results
print("\n" + "="*60)
print("XGBOOST EVALUATION RESULTS")
print("="*60)
print(f"AUC-ROC:     {metrics['roc_auc']:.4f}")
print(f"AUC-PR:      {metrics['pr_auc']:.4f}")
print(f"Accuracy:    {metrics['accuracy']:.4f}")
print(f"Precision:   {metrics['precision']:.4f}")
print(f"Recall:      {metrics['recall']:.4f}")
print(f"F1-Score:    {metrics['f1_score']:.4f}")
print(f"Log Loss:    {metrics['log_loss']:.4f}")
print("="*60)

print(f"\nAll visualizations saved to: {evaluator.figures_dir}")
```

---

## 📊 What Results You'll Get

### 1. **Console Output** - Printed Metrics

```
XGBOOST EVALUATION RESULTS
============================================================
AUC-ROC:     0.8542
AUC-PR:      0.7123
Accuracy:    0.8234
Precision:   0.7891
Recall:      0.6543
F1-Score:    0.7156
Log Loss:    0.3421
============================================================
```

### 2. **Visualizations** (Saved to `credit_risk_fyp/results/figures/`)

- `roc_curve_xgboost.png` - ROC curve showing AUC
- `pr_curve_xgboost.png` - Precision-Recall curve
- `confusion_matrix_xgboost.png` - Raw confusion matrix
- `confusion_matrix_xgboost_normalized.png` - Normalized confusion matrix
- `threshold_analysis_xgboost.png` - How metrics change with threshold
- `feature_importance_gain_xgboost.png` - Top important features

### 3. **CSV Report** (Saved to `credit_risk_fyp/results/reports/`)

All metrics in a CSV file for easy comparison:

| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score | ... |
|-------|---------|----------|-----------|--------|----------|-----|
| XGBoost | 0.8542 | 0.8234 | 0.7891 | 0.6543 | 0.7156 | ... |

---

## 🎯 Available Metrics

The evaluation module calculates these metrics:

### Classification Metrics
- **accuracy** - Overall accuracy
- **precision** - Positive predictive value
- **recall** - Sensitivity / True positive rate
- **f1_score** - Harmonic mean of precision and recall
- **specificity** - True negative rate
- **false_positive_rate** - FPR
- **false_negative_rate** - FNR

### Confusion Matrix Values
- **true_positives** (TP)
- **true_negatives** (TN)
- **false_positives** (FP)
- **false_negatives** (FN)

### Advanced Metrics
- **roc_auc** - Area Under ROC Curve
- **pr_auc** - Area Under Precision-Recall Curve
- **log_loss** - Logarithmic loss
- **brier_score** - Brier score
- **matthews_corr_coef** - Matthews Correlation Coefficient
- **cohen_kappa** - Cohen's Kappa

---

## 🔧 Threshold Optimization

Find the best threshold for your use case:

```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator()

# Optimize for F1-Score
optimal_threshold, optimal_f1 = evaluator.optimize_threshold(
    y_true, y_pred_proba, metric='f1'
)
print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"F1-Score at optimal threshold: {optimal_f1:.4f}")

# Optimize for precision (if you want fewer false positives)
optimal_threshold_prec, optimal_prec = evaluator.optimize_threshold(
    y_true, y_pred_proba, metric='precision'
)

# Optimize for recall (if you want to catch more defaults)
optimal_threshold_rec, optimal_rec = evaluator.optimize_threshold(
    y_true, y_pred_proba, metric='recall'
)
```

---

## 🎨 Customizing Visualizations

```python
from src.evaluation import ModelEvaluator

# Custom figure size and DPI
evaluator = ModelEvaluator(
    figsize=(14, 10),
    dpi=150
)

# Generate individual plots
fig_roc = evaluator.plot_roc_curve(y_true, y_pred_proba, "XGBoost")
fig_pr = evaluator.plot_pr_curve(y_true, y_pred_proba, "XGBoost")
fig_cm = evaluator.plot_confusion_matrix(y_true, y_pred, "XGBoost")

# Don't save, just show
import matplotlib.pyplot as plt
plt.show()
```

---

## 📁 Where Are Results Saved?

```
credit_risk_fyp/
├── results/
│   ├── figures/                    ← All visualizations
│   │   ├── roc_curve_xgboost.png
│   │   ├── pr_curve_xgboost.png
│   │   ├── confusion_matrix_xgboost.png
│   │   └── ...
│   └── reports/                    ← CSV reports
│       └── evaluation_report.csv
```

---

## 🚀 Running the Demo Script

To see everything in action with your actual data:

```bash
# Navigate to project directory
cd "C:\Users\Faheem\Desktop\Umair FYP\FYP2025"

# Run the demo (uses 50k samples for speed)
python credit_risk_fyp/scripts/demo_xgboost_evaluation.py
```

**What the demo does:**
1. Loads your actual training data (50,000 samples)
2. Preprocesses and engineers features
3. Trains XGBoost model
4. Evaluates on validation and test sets
5. Generates ALL visualizations
6. Creates comprehensive CSV report
7. Shows feature importance
8. Optimizes thresholds

**Output:**
- Trained model saved to `models/xgboost_model.pkl`
- 12+ visualization files in `results/figures/`
- CSV report in `results/reports/`
- Console output with all metrics

---

## 💡 Tips for Your FYP

### 1. **Compare All Models**

After training all 5 models, compare them:

```python
results = {
    'XGBoost': xgb_metrics,
    'LightGBM': lgb_metrics,
    'CatBoost': catboost_metrics,
    'Random Forest': rf_metrics,
    'Neural Network': nn_metrics
}

evaluator.compare_models(results)
report = evaluator.generate_report(results, 'results/all_models_comparison.csv')
```

### 2. **For Your Report**

Use these visualizations:
- ROC curves (compare models)
- Confusion matrices (show actual vs predicted)
- Feature importance (explain what matters)
- Threshold analysis (justify your choice)

### 3. **Business Metrics**

Calculate cost of false positives vs false negatives:

```python
# Example: FP costs $100, FN costs $500
fp_cost = metrics['false_positives'] * 100
fn_cost = metrics['false_negatives'] * 500
total_cost = fp_cost + fn_cost

print(f"Estimated cost: ${total_cost:,.2f}")
```

---

## ❓ Troubleshooting

**Issue: "Module not found"**
```bash
# Make sure you're in the right directory
cd "C:\Users\Faheem\Desktop\Umair FYP\FYP2025"

# Install missing packages
pip install scikit-learn matplotlib seaborn
```

**Issue: "No data loaded"**
- Check data path in config.py
- Verify datasets are in credit_risk_fyp/data/raw/

**Issue: "Plots not showing"**
- Plots are automatically saved to results/figures/
- Open them with any image viewer
- Or add `plt.show()` to display them

---

## 📚 Next Steps

1. ✅ **Done**: Evaluation module created
2. **Next**: Train XGBoost on full data
3. **Then**: Implement other 4 models
4. **Finally**: Compare all models using the evaluation module

---

**The evaluation module is ready to use!** You can now see comprehensive results for any model you train.
