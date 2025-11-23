# Credit Risk Assessment FYP - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

This guide will help you train your first credit risk model and make predictions.

---

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (optional but recommended)
- 16GB+ RAM

---

## Installation

### 1. Clone and Setup

```bash
cd credit_risk_fyp
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Data

Download the Lending Club dataset from Kaggle:
- https://www.kaggle.com/datasets/wordsforthewise/lending-club

Place the CSV file in `data/raw/`:
```bash
mkdir -p data/raw
# Move your downloaded lending_club.csv to data/raw/
```

---

## Train Your First Model

### Option 1: Train All Models (Recommended)

```bash
python scripts/train_all_models.py \
    --data-path data/raw/lending_club.csv \
    --models all \
    --ensemble \
    --verbose
```

This will:
- ✓ Load and split data (70/15/15 train/val/test)
- ✓ Preprocess data (handle missing values, outliers, scaling)
- ✓ Engineer features (25+ derived features)
- ✓ Train 5 base models (XGBoost, LightGBM, CatBoost, Random Forest, Neural Network)
- ✓ Train 2 ensemble models (Stacking, Weighted Averaging)
- ✓ Save all models to `models/` directory

### Option 2: Train Specific Models

```bash
# Train only gradient boosting models
python scripts/train_all_models.py \
    --data-path data/raw/lending_club.csv \
    --models xgboost lightgbm catboost

# Train only ensemble models
python scripts/train_all_models.py \
    --data-path data/raw/lending_club.csv \
    --models all \
    --ensemble
```

---

## Make Predictions

### Using Python

```python
from src.inference import CreditRiskPredictor
import pandas as pd

# Load the best model (stacking ensemble)
predictor = CreditRiskPredictor(
    model_path='models/stacking_ensemble.pkl',
    preprocessor_path='models/preprocessor.pkl',
    feature_engineer_path='models/feature_engineer.pkl'
)

# Load new loan applications
new_loans = pd.read_csv('new_loan_applications.csv')

# Make predictions
predictions, probabilities = predictor.predict(
    new_loans,
    return_proba=True
)

# View results
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    risk = "HIGH RISK" if pred == 1 else "LOW RISK"
    print(f"Loan {i+1}: {risk} (Default Probability: {prob:.2%})")
```

### Generate Prediction Report

```python
# Generate comprehensive report
predictor.generate_report(
    new_loans,
    output_path='results/reports/predictions.csv'
)
```

The report includes:
- Original loan data
- Predictions (0=Non-default, 1=Default)
- Default probabilities
- Risk levels (Low/Medium/High)

---

## Understand Predictions with SHAP

```python
# Explain a single prediction
explanation = predictor.explain_prediction(
    new_loans.iloc[0],  # First loan
    num_features=10
)

print(f"Prediction: {explanation['prediction']}")
print(f"Probability: {explanation['probability']:.2%}")
print("\nTop Influencing Features:")
for feat in explanation['top_features']:
    print(f"  {feat['feature']}: SHAP={feat['shap_value']:.4f}")
```

---

## Evaluate Model Performance

### Using Python

```python
from src.evaluation import ModelEvaluator
from src.utils import load_pickle
import pandas as pd

# Load test data
test_data = pd.read_csv('data/splits/test.csv')
X_test = test_data.drop('loan_status', axis=1)
y_test = test_data['loan_status']

# Load model
model = load_pickle('models/stacking_ensemble.pkl')

# Make predictions
y_pred_proba = model.predict_proba(X_test)

# Evaluate
evaluator = ModelEvaluator()
metrics = evaluator.evaluate(y_test, y_pred_proba, model_name="Stacking Ensemble")

# Plot ROC curve
evaluator.plot_roc_curve(
    y_test,
    y_pred_proba,
    model_name="Stacking Ensemble",
    save_path='results/figures/roc_curve.png'
)

# Plot confusion matrix
y_pred = (y_pred_proba >= 0.5).astype(int)
evaluator.plot_confusion_matrix(
    y_test,
    y_pred,
    model_name="Stacking Ensemble",
    save_path='results/figures/confusion_matrix.png'
)
```

---

## Expected Results

After training on the Lending Club dataset, you should see:

### Individual Model Performance (AUC-ROC on validation set)
- XGBoost: ~0.70-0.73
- LightGBM: ~0.69-0.72
- CatBoost: ~0.70-0.73
- Random Forest: ~0.66-0.69
- Neural Network: ~0.68-0.71

### Ensemble Performance
- **Stacking Ensemble: ~0.73-0.76** (Best)
- Weighted Ensemble: ~0.72-0.75

*Note: Actual results depend on data quality, preprocessing, and hyperparameters.*

---

## Training Time Estimates

With GPU acceleration (NVIDIA RTX 3090):

| Model | Training Time | Memory Usage |
|-------|---------------|--------------|
| XGBoost | 2-5 min | 2-4 GB |
| LightGBM | 1-3 min | 2-3 GB |
| CatBoost | 3-6 min | 3-5 GB |
| Random Forest | 5-10 min | 4-6 GB |
| Neural Network | 10-20 min | 4-6 GB |
| Stacking Ensemble | 15-30 min | 6-8 GB |
| Weighted Ensemble | 5-10 min | 4-6 GB |

**Total Time (all models): 40-80 minutes**

Without GPU:
- Expect 2-3x longer for tree-based models
- Expect 5-10x longer for neural network

---

## Project Structure Quick Reference

```
credit_risk_fyp/
├── data/
│   ├── raw/              # Place your CSV files here
│   ├── processed/        # Automatically generated
│   └── splits/           # train.csv, val.csv, test.csv
├── models/               # Trained models saved here
│   ├── xgboost_model.pkl
│   ├── stacking_ensemble.pkl
│   ├── preprocessor.pkl
│   └── feature_engineer.pkl
├── results/
│   ├── figures/          # Plots and visualizations
│   └── reports/          # Prediction reports
├── src/
│   ├── models/           # Model implementations
│   ├── preprocessor.py   # Data preprocessing
│   ├── feature_engineer.py
│   ├── evaluation.py
│   └── inference.py
└── scripts/
    └── train_all_models.py  # Master training script
```

---

## Common Issues and Solutions

### Issue: "No GPU found"
**Solution:** Models will fallback to CPU. To use GPU:
1. Install CUDA toolkit (https://developer.nvidia.com/cuda-downloads)
2. Install cuDNN
3. Reinstall TensorFlow with GPU support: `pip install tensorflow-gpu==2.13.0`

### Issue: "Out of memory"
**Solution:**
1. Reduce batch size in config.py (NEURAL_NETWORK_PARAMS['batch_size'])
2. Use data chunking (enabled by default)
3. Reduce number of trees in tree-based models

### Issue: "Import errors"
**Solution:**
```bash
# Ensure you're in the virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "KeyError: 'loan_status'"
**Solution:** Ensure your dataset has the correct target column name or update DATASET_CONFIG['target_column'] in src/config.py

---

## Next Steps

1. **Hyperparameter Tuning:** Modify hyperparameters in `src/config.py`
2. **Feature Engineering:** Add custom features in `src/feature_engineer.py`
3. **Model Selection:** Compare models and choose the best for your use case
4. **Production Deployment:** Use `src/inference.py` for production predictions
5. **Documentation:** Read `docs/PROJECT_REPORT.md` for detailed methodology

---

## Need Help?

- **Documentation:** Check `README.md` and `IMPLEMENTATION_STATUS.md`
- **Code Examples:** See Jupyter notebooks in `notebooks/`
- **Configuration:** All hyperparameters in `src/config.py`
- **API Reference:** See docstrings in each module

---

## Performance Optimization Tips

### For Fastest Training:
1. **Enable GPU:** Ensure CUDA is installed
2. **Use Mixed Precision:** Enabled by default for Neural Network
3. **Reduce Data:** Use a sample for initial experiments:
   ```python
   df = df.sample(frac=0.1, random_state=42)  # Use 10% of data
   ```
4. **Parallel Processing:** Enabled by default for Random Forest

### For Best Accuracy:
1. **Hyperparameter Tuning:** Use grid/random search
2. **Feature Engineering:** Create domain-specific features
3. **Ensemble Methods:** Always use ensembles for production
4. **Threshold Optimization:** Find optimal decision threshold

---

## Example: Complete Workflow

```python
# 1. Train models
!python scripts/train_all_models.py \
    --data-path data/raw/lending_club.csv \
    --models all \
    --ensemble \
    --verbose

# 2. Load best model
from src.inference import CreditRiskPredictor

predictor = CreditRiskPredictor(
    model_path='models/stacking_ensemble.pkl',
    preprocessor_path='models/preprocessor.pkl',
    feature_engineer_path='models/feature_engineer.pkl'
)

# 3. Make predictions
import pandas as pd
new_loans = pd.read_csv('new_applications.csv')
predictions, probabilities = predictor.predict(new_loans, return_proba=True)

# 4. Generate report
predictor.generate_report(new_loans, 'results/reports/risk_assessment.csv')

# 5. Explain high-risk predictions
high_risk_indices = [i for i, p in enumerate(predictions) if p == 1]
for idx in high_risk_indices[:5]:  # First 5 high-risk loans
    explanation = predictor.explain_prediction(new_loans.iloc[idx])
    print(f"\nLoan {idx}: {explanation['probability']:.2%} default probability")
    print("Top risk factors:")
    for feat in explanation['top_features'][:3]:
        print(f"  - {feat['feature']}")
```

---

**You're ready to build your credit risk assessment system! 🚀**
