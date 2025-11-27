"""
Demo Script: Train and Evaluate XGBoost Model
This script demonstrates how to train XGBoost and visualize results
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_DATA_DIR, DATASET_CONFIG, MODELS_DIR, XGBOOST_PARAMS
from src.data_loader import load_data
from src.preprocessor import DataPreprocessor
from src.feature_engineer import FeatureEngineer
from src.models.xgboost_model import XGBoostModel
from src.evaluation import ModelEvaluator
from src.utils import setup_logging, set_seed

# Setup
setup_logging()
set_seed(42)

print("=" * 80)
print("XGBOOST MODEL TRAINING AND EVALUATION DEMO")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[1/7] Loading training data...")
train_path = RAW_DATA_DIR / DATASET_CONFIG['train_dataset']
df = load_data(train_path, optimize=True, nrows=50000)  # Use subset for demo
print(f"Loaded {len(df):,} rows")

# ============================================================================
# STEP 2: SPLIT DATA
# ============================================================================
print("\n[2/7] Splitting data...")
target_col = DATASET_CONFIG['target_column']
id_col = DATASET_CONFIG['id_column']

# Separate features and target
X = df.drop(columns=[target_col, id_col])
y = df[target_col]

# Split: 70% train, 15% validation, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
)

print(f"Train set: {len(X_train):,} samples")
print(f"Validation set: {len(X_val):,} samples")
print(f"Test set: {len(X_test):,} samples")
print(f"Target distribution (train): {y_train.value_counts().to_dict()}")

# ============================================================================
# STEP 3: PREPROCESSING
# ============================================================================
print("\n[3/7] Preprocessing data...")
preprocessor = DataPreprocessor()

# Fit on training data
X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print(f"Features after preprocessing: {X_train_processed.shape[1]}")

# Save preprocessor
preprocessor_path = MODELS_DIR / 'preprocessor.pkl'
preprocessor.save(preprocessor_path)
print(f"Preprocessor saved to {preprocessor_path}")

# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================
print("\n[4/7] Engineering features...")
feature_engineer = FeatureEngineer()

# Fit on training data
X_train_final = feature_engineer.fit_transform(X_train_processed)
X_val_final = feature_engineer.transform(X_val_processed)
X_test_final = feature_engineer.transform(X_test_processed)

print(f"Features after engineering: {X_train_final.shape[1]}")

# Save feature engineer
fe_path = MODELS_DIR / 'feature_engineer.pkl'
feature_engineer.save(fe_path)
print(f"Feature engineer saved to {fe_path}")

# ============================================================================
# STEP 5: TRAIN XGBOOST MODEL
# ============================================================================
print("\n[5/7] Training XGBoost model...")
print("This may take a few minutes...")

# Update params for demo (reduce training time)
demo_params = XGBOOST_PARAMS.copy()
demo_params['n_estimators'] = 100  # Reduce from 1000
demo_params['early_stopping_rounds'] = 20  # Reduce from 50

# Initialize and train
xgb_model = XGBoostModel(params=demo_params)
xgb_model.train(
    X_train_final, y_train,
    X_val_final, y_val
)

# Save model
model_path = MODELS_DIR / 'xgboost_model.pkl'
xgb_model.save_model(model_path)
print(f"Model saved to {model_path}")

# ============================================================================
# STEP 6: MAKE PREDICTIONS
# ============================================================================
print("\n[6/7] Making predictions...")

# Predictions on validation set
y_val_proba = xgb_model.predict_proba(X_val_final)
y_val_pred = xgb_model.predict(X_val_final)

# Predictions on test set
y_test_proba = xgb_model.predict_proba(X_test_final)
y_test_pred = xgb_model.predict(X_test_final)

print("Predictions completed!")

# ============================================================================
# STEP 7: EVALUATE MODEL
# ============================================================================
print("\n[7/7] Evaluating model...")
print("\nGenerating comprehensive evaluation report and visualizations...")

evaluator = ModelEvaluator()

# Evaluate on validation set
print("\n" + "=" * 80)
print("VALIDATION SET RESULTS")
print("=" * 80)
val_metrics = evaluator.evaluate(
    y_val, y_val_proba,
    threshold=0.5,
    model_name="XGBoost_Validation"
)

# Generate all plots for validation
print("\nGenerating validation set plots...")
val_figures = evaluator.evaluate_all_plots(
    y_val, y_val_proba,
    model_name="XGBoost_Validation",
    threshold=0.5,
    save=True
)

# Evaluate on test set
print("\n" + "=" * 80)
print("TEST SET RESULTS")
print("=" * 80)
test_metrics = evaluator.evaluate(
    y_test, y_test_proba,
    threshold=0.5,
    model_name="XGBoost_Test"
)

# Generate all plots for test
print("\nGenerating test set plots...")
test_figures = evaluator.evaluate_all_plots(
    y_test, y_test_proba,
    model_name="XGBoost_Test",
    threshold=0.5,
    save=True
)

# ============================================================================
# DETAILED METRICS REPORT
# ============================================================================
print("\n" + "=" * 80)
print("DETAILED METRICS REPORT")
print("=" * 80)

print("\n📊 VALIDATION SET:")
print("-" * 80)
print(f"{'Metric':<30} {'Value':<15}")
print("-" * 80)
for metric, value in val_metrics.items():
    if isinstance(value, (int, float)) and metric != 'threshold':
        if isinstance(value, float):
            print(f"{metric:<30} {value:.4f}")
        else:
            print(f"{metric:<30} {value}")

print("\n📊 TEST SET:")
print("-" * 80)
print(f"{'Metric':<30} {'Value':<15}")
print("-" * 80)
for metric, value in test_metrics.items():
    if isinstance(value, (int, float)) and metric != 'threshold':
        if isinstance(value, float):
            print(f"{metric:<30} {value:.4f}")
        else:
            print(f"{metric:<30} {value}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("TOP 20 MOST IMPORTANT FEATURES")
print("=" * 80)

feature_importance = xgb_model.get_feature_importance(importance_type='gain')
xgb_model.plot_feature_importance(top_n=20, importance_type='gain', save=True)

top_features = feature_importance.head(20)
print(f"\n{'Rank':<6} {'Feature':<40} {'Importance':<15}")
print("-" * 80)
for idx, (feature, importance) in enumerate(top_features.items(), 1):
    print(f"{idx:<6} {feature:<40} {importance:.4f}")

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================
print("\n" + "=" * 80)
print("THRESHOLD OPTIMIZATION")
print("=" * 80)

# Find optimal thresholds for different metrics
metrics_to_optimize = ['f1', 'precision', 'recall']
print("\nOptimal thresholds on validation set:")
print("-" * 80)
for metric in metrics_to_optimize:
    optimal_threshold, optimal_score = evaluator.optimize_threshold(
        y_val, y_val_proba, metric=metric
    )
    print(f"{metric.upper():<15} Threshold: {optimal_threshold:.4f}, Score: {optimal_score:.4f}")

# ============================================================================
# SAVE COMPREHENSIVE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("SAVING REPORTS")
print("=" * 80)

# Create comparison report
results_dict = {
    'XGBoost_Validation': val_metrics,
    'XGBoost_Test': test_metrics
}

report_df = evaluator.generate_report(
    results_dict,
    output_path=evaluator.reports_dir / 'xgboost_evaluation_report.csv'
)

print("\n📄 Report Summary:")
print(report_df[['roc_auc', 'accuracy', 'precision', 'recall', 'f1_score']])

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("EVALUATION COMPLETE!")
print("=" * 80)
print(f"\n✓ Model saved to: {model_path}")
print(f"✓ Figures saved to: {evaluator.figures_dir}")
print(f"✓ Reports saved to: {evaluator.reports_dir}")

print("\n📁 Generated files:")
print("  - ROC curves (validation & test)")
print("  - Precision-Recall curves (validation & test)")
print("  - Confusion matrices (validation & test)")
print("  - Threshold analysis plots (validation & test)")
print("  - Feature importance plot")
print("  - CSV report with all metrics")

print("\n" + "=" * 80)
print("You can now:")
print("  1. Check the results/ directory for all visualizations")
print("  2. Review the CSV report in results/reports/")
print("  3. Use the saved model for predictions")
print("  4. Compare with other models when trained")
print("=" * 80)
