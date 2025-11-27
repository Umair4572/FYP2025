"""
Quick test to verify XGBoost and all packages are installed correctly
"""

print("=" * 60)
print("TESTING PACKAGE INSTALLATION")
print("=" * 60)

# Test imports
print("\n[1/5] Testing package imports...")
try:
    import xgboost as xgb
    import sklearn
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from tqdm import tqdm
    import joblib
    print("[OK] All packages imported successfully")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    exit(1)

# Print versions
print("\n[2/5] Package versions:")
print(f"  XGBoost:      {xgb.__version__}")
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  pandas:       {pd.__version__}")
print(f"  numpy:        {np.__version__}")

# Test XGBoost functionality
print("\n[3/5] Testing XGBoost functionality...")
try:
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    # Test prediction
    score = model.score(X_test, y_test)

    print(f"[OK] XGBoost model trained successfully")
    print(f"  Test accuracy: {score:.4f}")

except Exception as e:
    print(f"[ERROR] XGBoost test failed: {e}")
    exit(1)

# Test evaluation metrics
print("\n[4/5] Testing evaluation metrics...")
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score
    )

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("[OK] Evaluation metrics calculated successfully")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

except Exception as e:
    print(f"[ERROR] Evaluation test failed: {e}")
    exit(1)

# Test visualization
print("\n[5/5] Testing visualization...")
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title('Test Plot')
    plt.close(fig)

    print("[OK] Visualization libraries working")

except Exception as e:
    print(f"[ERROR] Visualization test failed: {e}")
    exit(1)

# Final summary
print("\n" + "=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
print("\n[SUCCESS] XGBoost and all required packages are installed correctly")
print("[SUCCESS] You're ready to start training models")
print("\nNext steps:")
print("  1. Run: python credit_risk_fyp/scripts/quick_test.py")
print("  2. Or: python credit_risk_fyp/scripts/demo_xgboost_evaluation.py")
print("=" * 60)
