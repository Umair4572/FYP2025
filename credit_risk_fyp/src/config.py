"""
Configuration file for Credit Risk Assessment FYP
Contains all hyperparameters, paths, and settings
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR,
                 MODELS_DIR, RESULTS_DIR, LOGS_DIR, FIGURES_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATASET_CONFIG = {
    'main_dataset': 'lending_club_2007_2020.csv',
    'test_dataset': 'lending_club_test.csv',
    'target_column': 'loan_status',
    'id_column': 'id',
    'random_seed': 42
}

# Train/Val/Test split ratios
SPLIT_RATIOS = {
    'train': 0.70,
    'validation': 0.15,
    'test': 0.15
}

# ============================================================================
# GPU AND PERFORMANCE CONFIGURATION
# ============================================================================
GPU_CONFIG = {
    'use_gpu': True,
    'gpu_id': 0,
    'memory_growth': True,  # Critical for TensorFlow
    'mixed_precision': True,  # Use float16 for speed
    'num_threads': -1,  # Use all available cores
}

# Data loading optimization
DATA_LOADING_CONFIG = {
    'chunk_size': 100000,
    'optimize_dtypes': True,
    'use_dask': False,  # Set True for very large datasets
    'parallel_processing': True
}

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================
PREPROCESSING_CONFIG = {
    'missing_threshold': 0.50,  # Drop columns with >50% missing
    'handle_outliers': True,
    'outlier_method': 'iqr',  # 'iqr' or 'zscore'
    'outlier_threshold': 3,
    'scale_features': True,
    'encoding_method': 'label',  # 'label', 'onehot', or 'target'
}

# Columns to drop (data leakage prevention)
LEAKAGE_COLUMNS = [
    'loan_status', 'funded_amnt_inv', 'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
    'recoveries', 'collection_recovery_fee', 'last_pymnt_d',
    'last_pymnt_amnt', 'next_pymnt_d', 'last_credit_pull_d',
    'collections_12_mths_ex_med', 'policy_code', 'out_prncp',
    'out_prncp_inv', 'hardship_flag', 'debt_settlement_flag'
]

# Target encoding (binary classification)
TARGET_ENCODING = {
    'default_class': ['Charged Off', 'Default', 'Late (31-120 days)',
                      'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off'],
    'non_default_class': ['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid'],
    'exclude_class': ['Current', 'In Grace Period']  # Exclude uncertain status
}

# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================
FEATURE_ENGINEERING_CONFIG = {
    'create_ratios': True,
    'create_interactions': True,
    'create_polynomials': False,
    'create_time_features': True,
    'create_aggregations': True,
    'binning_strategy': 'quantile',  # 'quantile' or 'uniform'
    'n_bins': 5
}

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# XGBoost Configuration
XGBOOST_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 5,  # Adjust based on class imbalance
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50,
    'verbose_eval': 50
}

# LightGBM Configuration
LIGHTGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'boosting_type': 'gbdt',
    'num_leaves': 64,
    'learning_rate': 0.05,
    'n_estimators': 1000,
    'subsample': 0.8,
    'subsample_freq': 5,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 5,
    'random_state': 42,
    'verbose': -1,
    'early_stopping_rounds': 50
}

# CatBoost Configuration
CATBOOST_PARAMS = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 8,
    'l2_leaf_reg': 3,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'task_type': 'GPU',
    'devices': '0',
    'random_seed': 42,
    'early_stopping_rounds': 50,
    'verbose': 50,
    'scale_pos_weight': 5
}

# Random Forest Configuration
RANDOM_FOREST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': 1
}

# Neural Network Configuration
NEURAL_NETWORK_PARAMS = {
    'architecture': [512, 256, 128, 64],  # Hidden layer sizes
    'activation': 'relu',
    'dropout_rate': [0.3, 0.3, 0.2, 0.2],
    'batch_normalization': True,
    'l2_regularization': 0.001,
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'batch_size': 2048,
    'epochs': 100,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.5,
    'min_lr': 1e-7
}

# ============================================================================
# ENSEMBLE CONFIGURATION
# ============================================================================
ENSEMBLE_CONFIG = {
    'stacking': {
        'use_cv': True,
        'cv_folds': 5,
        'meta_model': 'logistic_regression',
        'meta_model_params': {
            'max_iter': 1000,
            'random_state': 42,
            'n_jobs': -1
        }
    },
    'weighted_averaging': {
        'optimization_method': 'scipy',  # 'scipy' or 'grid_search'
        'metric': 'auc'
    }
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
EVALUATION_CONFIG = {
    'metrics': ['auc', 'accuracy', 'precision', 'recall', 'f1', 'confusion_matrix'],
    'threshold': 0.5,  # Classification threshold
    'threshold_optimization': True,  # Optimize threshold on validation set
    'plot_roc': True,
    'plot_pr_curve': True,
    'plot_confusion_matrix': True,
    'plot_feature_importance': True,
    'shap_analysis': True,
    'shap_sample_size': 1000  # Number of samples for SHAP
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_to_file': True,
    'log_file': LOGS_DIR / 'training.log'
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================
VISUALIZATION_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': 'husl',
    'save_format': 'png'
}
