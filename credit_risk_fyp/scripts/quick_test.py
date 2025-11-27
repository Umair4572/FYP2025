"""
Quick Test Script: Minimal demo to test the evaluation module
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.models.xgboost_model import XGBoostModel
from src.evaluation import ModelEvaluator
from src.utils import setup_logging, set_seed

# Setup
setup_logging()
set_seed(42)

print("=" * 80)
print("QUICK TEST: XGBoost Training & Evaluation")
print("=" * 80)

# Generate synthetic data for quick testing
print("\n[1/4] Generating synthetic data...")
X, y = make_classification(
    n_samples=5000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.8, 0.2],  # Imbalanced like credit risk
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
print(f"Class distribution: {np.bincount(y_train)}")

# Train simple model
print("\n[2/4] Training XGBoost model...")
model = XGBoostModel(params={
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 50,
    'tree_method': 'hist',  # CPU version
    'random_state': 42
})

# Use 80% for training, 20% for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

model.train(X_tr, y_tr, X_val, y_val)
print("Training complete!")

# Make predictions
print("\n[3/4] Making predictions...")
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Evaluate
print("\n[4/4] Evaluating model...")
evaluator = ModelEvaluator()

# Get metrics
metrics = evaluator.evaluate(
    y_test, y_pred_proba,
    threshold=0.5,
    model_name="XGBoost_QuickTest"
)

# Generate plots
print("\nGenerating visualizations...")
figures = evaluator.evaluate_all_plots(
    y_test, y_pred_proba,
    model_name="XGBoost_QuickTest",
    threshold=0.5,
    save=True
)

# Display results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"\nAUC-ROC:     {metrics['roc_auc']:.4f}")
print(f"Accuracy:    {metrics['accuracy']:.4f}")
print(f"Precision:   {metrics['precision']:.4f}")
print(f"Recall:      {metrics['recall']:.4f}")
print(f"F1-Score:    {metrics['f1_score']:.4f}")

print("\n" + "=" * 80)
print("SUCCESS! Evaluation module is working correctly.")
print("=" * 80)
print(f"\nCheck the results directory for visualizations:")
print(f"  {evaluator.figures_dir}")
print("=" * 80)
