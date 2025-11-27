"""
Quick Test Script: Minimal demo to test the evaluation module with REAL data
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_DIR, DATASET_CONFIG
from src.data_loader import load_data
from src.preprocessor import DataPreprocessor
from src.models.xgboost_model import XGBoostModel
from src.evaluation import ModelEvaluator
from src.utils import setup_logging, set_seed

# Setup
setup_logging()
set_seed(42)

print("=" * 80)
print("QUICK TEST: XGBoost Training & Evaluation (REAL DATA)")
print("=" * 80)

# Load REAL training data
print("\n[1/5] Loading REAL training data...")
train_path = RAW_DATA_DIR / DATASET_CONFIG['train_dataset']
df = load_data(train_path, optimize=True, nrows=15000)  # Use 15k rows for quick test
print(f"Loaded {len(df):,} rows from: {DATASET_CONFIG['train_dataset']}")

# Separate features and target
target_col = DATASET_CONFIG['target_column']
id_col = DATASET_CONFIG['id_column']

X = df.drop(columns=[target_col, id_col])
y = df[target_col]

print(f"Dataset: {DATASET_CONFIG['train_dataset']}")
print(f"Target: {target_col}")
print(f"Class distribution: {y.value_counts().to_dict()}")

# Split into train and test (80/20 split, ensuring at least 10k for training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train):,} samples, Test: {len(X_test):,} samples")

# Preprocess data
print("\n[2/5] Preprocessing data...")
preprocessor = DataPreprocessor()

# Combine X_train and y_train for preprocessing
train_df_combined = X_train.copy()
train_df_combined[target_col] = y_train

X_train_processed, _ = preprocessor.fit_transform(train_df_combined)
X_test_processed, _ = preprocessor.transform(X_test)
print(f"Features after preprocessing: {X_train_processed.shape[1]}")

# Train simple model
print("\n[3/5] Training XGBoost model...")
print("Using quick parameters for fast testing...")
model = XGBoostModel(params={
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 50,
    'tree_method': 'hist',  # CPU version
    'random_state': 42
})

# Use 80% for training, 20% for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_processed, y_train, test_size=0.2, random_state=42, stratify=y_train
)

model.train(X_tr, y_tr, X_val, y_val)
print("Training complete!")

# Make predictions
print("\n[4/5] Making predictions on test set...")
y_test_proba = model.predict_proba(X_test_processed)
y_test_pred = model.predict(X_test_processed)
print(f"Predictions made on {len(y_test):,} test samples")

# Evaluate
print("\n[5/5] Evaluating model on test set...")
evaluator = ModelEvaluator()

# Get metrics
metrics = evaluator.evaluate(
    y_test, y_test_proba,
    threshold=0.5,
    model_name="XGBoost_QuickTest"
)

# Generate plots
print("\nGenerating visualizations...")
figures = evaluator.evaluate_all_plots(
    y_test, y_test_proba,
    model_name="XGBoost_QuickTest",
    threshold=0.5,
    save=True
)

# Display results
print("\n" + "=" * 80)
print("RESULTS (TEST SET)")
print("=" * 80)
print(f"\nDataset Used: {DATASET_CONFIG['train_dataset']}")
print(f"Total samples used: {len(df):,}")
print(f"Test samples: {len(y_test):,}")
print(f"Test positive class: {metrics['n_positive']:,}")
print(f"Test negative class: {metrics['n_negative']:,}")

print(f"\nAUC-ROC:     {metrics['roc_auc']:.4f}")
print(f"Accuracy:    {metrics['accuracy']:.4f}")
print(f"Precision:   {metrics['precision']:.4f}")
print(f"Recall:      {metrics['recall']:.4f}")
print(f"F1-Score:    {metrics['f1_score']:.4f}")

print("\n" + "=" * 80)
print("SUCCESS! XGBoost model trained and evaluated on REAL data.")
print("=" * 80)
print(f"\nCheck the results directory for visualizations:")
print(f"  {evaluator.figures_dir}")
print("=" * 80)
