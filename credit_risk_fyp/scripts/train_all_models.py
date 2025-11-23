#!/usr/bin/env python
"""
Master Training Script for Credit Risk Assessment FYP
Trains all base models and ensemble methods
"""

import argparse
import logging
import sys
from pathlib import Path
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    MODELS_DIR, SPLITS_DIR, DATASET_CONFIG, SPLIT_RATIOS
)
from src.utils import setup_logging, setup_gpu, set_random_seeds
from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.feature_engineer import FeatureEngineer
from src.evaluation import ModelEvaluator

# Import all models
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.catboost_model import CatBoostModel
from src.models.random_forest_model import RandomForestModel
from src.models.neural_network import NeuralNetworkModel
from src.models.stacking_ensemble import StackingEnsemble
from src.models.weighted_ensemble import WeightedEnsemble

# Setup logger
logger = logging.getLogger('credit_risk_fyp.train_all')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train all models for Credit Risk Assessment FYP'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to the dataset CSV file'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(MODELS_DIR),
        help='Directory to save trained models'
    )

    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        choices=['xgboost', 'lightgbm', 'catboost', 'rf', 'nn', 'all'],
        help='Models to train'
    )

    parser.add_argument(
        '--ensemble',
        action='store_true',
        help='Train ensemble models'
    )

    parser.add_argument(
        '--no-preprocessing',
        action='store_true',
        help='Skip preprocessing (use preprocessed data)'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    return parser.parse_args()


def load_and_split_data(data_path: Path, args):
    """Load and split data into train/val/test sets."""
    logger.info("="*80)
    logger.info("DATA LOADING AND SPLITTING")
    logger.info("="*80)

    # Load data
    loader = DataLoader(optimize_dtypes=True, verbose=True)
    df = loader.load_dataset(data_path)

    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Split data
    train_ratio = SPLIT_RATIOS['train']
    val_ratio = SPLIT_RATIOS['validation']
    test_ratio = SPLIT_RATIOS['test']

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        random_state=args.seed,
        stratify=df[DATASET_CONFIG['target_column']] if DATASET_CONFIG['target_column'] in df.columns else None
    )

    # Second split: separate train and validation
    val_size = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=args.seed,
        stratify=train_val_df[DATASET_CONFIG['target_column']] if DATASET_CONFIG['target_column'] in train_val_df.columns else None
    )

    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")

    # Save splits
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / 'train.csv', index=False)
    val_df.to_csv(SPLITS_DIR / 'val.csv', index=False)
    test_df.to_csv(SPLITS_DIR / 'test.csv', index=False)
    logger.info(f"\n✓ Saved data splits to {SPLITS_DIR}")

    return train_df, val_df, test_df


def preprocess_data(train_df, val_df, test_df, output_dir):
    """Preprocess data."""
    logger.info("\n" + "="*80)
    logger.info("DATA PREPROCESSING")
    logger.info("="*80)

    # Initialize and fit preprocessor on training data
    preprocessor = DataPreprocessor()
    preprocessor.fit(train_df)

    # Transform all datasets
    X_train, y_train = preprocessor.transform(train_df, is_training=True)
    X_val, y_val = preprocessor.transform(val_df, is_training=True)
    X_test, y_test = preprocessor.transform(test_df, is_training=True)

    logger.info(f"\nPreprocessed data shape:")
    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val:   {X_val.shape}")
    logger.info(f"  Test:  {X_test.shape}")

    # Save preprocessor
    preprocessor.save(output_dir / 'preprocessor.pkl')

    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessor


def engineer_features(X_train, X_val, X_test, output_dir):
    """Engineer features."""
    logger.info("\n" + "="*80)
    logger.info("FEATURE ENGINEERING")
    logger.info("="*80)

    # Initialize and fit feature engineer
    engineer = FeatureEngineer()
    engineer.fit(X_train)

    # Transform all datasets
    X_train_eng = engineer.transform(X_train)
    X_val_eng = engineer.transform(X_val)
    X_test_eng = engineer.transform(X_test)

    logger.info(f"\nEngineered feature shape:")
    logger.info(f"  Train: {X_train_eng.shape}")
    logger.info(f"  Val:   {X_val_eng.shape}")
    logger.info(f"  Test:  {X_test_eng.shape}")

    # Save feature engineer
    engineer.save(output_dir / 'feature_engineer.pkl')

    return X_train_eng, X_val_eng, X_test_eng, engineer


def train_base_models(X_train, y_train, X_val, y_val, models_to_train, output_dir, verbose=True):
    """Train base models."""
    logger.info("\n" + "="*80)
    logger.info("TRAINING BASE MODELS")
    logger.info("="*80)

    trained_models = {}
    results = {}

    # XGBoost
    if 'xgboost' in models_to_train or 'all' in models_to_train:
        logger.info("\n" + "-"*80)
        logger.info("Training XGBoost...")
        logger.info("-"*80)
        start_time = time.time()

        xgb_model = XGBoostModel()
        xgb_model.train(X_train, y_train, X_val, y_val, verbose=verbose)
        xgb_model.save_model(output_dir / 'xgboost_model.pkl')

        elapsed = time.time() - start_time
        logger.info(f"✓ XGBoost training completed in {elapsed:.1f}s")

        trained_models['xgboost'] = xgb_model

    # LightGBM
    if 'lightgbm' in models_to_train or 'all' in models_to_train:
        logger.info("\n" + "-"*80)
        logger.info("Training LightGBM...")
        logger.info("-"*80)
        start_time = time.time()

        lgb_model = LightGBMModel()
        lgb_model.train(X_train, y_train, X_val, y_val, verbose=verbose)
        lgb_model.save_model(output_dir / 'lightgbm_model.pkl')

        elapsed = time.time() - start_time
        logger.info(f"✓ LightGBM training completed in {elapsed:.1f}s")

        trained_models['lightgbm'] = lgb_model

    # CatBoost
    if 'catboost' in models_to_train or 'all' in models_to_train:
        logger.info("\n" + "-"*80)
        logger.info("Training CatBoost...")
        logger.info("-"*80)
        start_time = time.time()

        cat_model = CatBoostModel()
        cat_model.train(X_train, y_train, X_val, y_val, verbose=verbose)
        cat_model.save_model(output_dir / 'catboost_model.pkl')

        elapsed = time.time() - start_time
        logger.info(f"✓ CatBoost training completed in {elapsed:.1f}s")

        trained_models['catboost'] = cat_model

    # Random Forest
    if 'rf' in models_to_train or 'all' in models_to_train:
        logger.info("\n" + "-"*80)
        logger.info("Training Random Forest...")
        logger.info("-"*80)
        start_time = time.time()

        rf_model = RandomForestModel()
        rf_model.train(X_train, y_train, X_val, y_val, verbose=verbose)
        rf_model.save_model(output_dir / 'random_forest_model.pkl')

        elapsed = time.time() - start_time
        logger.info(f"✓ Random Forest training completed in {elapsed:.1f}s")

        trained_models['random_forest'] = rf_model

    # Neural Network
    if 'nn' in models_to_train or 'all' in models_to_train:
        logger.info("\n" + "-"*80)
        logger.info("Training Neural Network...")
        logger.info("-"*80)
        start_time = time.time()

        nn_model = NeuralNetworkModel(input_dim=X_train.shape[1])
        nn_model.train(X_train, y_train, X_val, y_val, verbose=1 if verbose else 0)
        nn_model.save_model(output_dir / 'neural_network_model.pkl')

        elapsed = time.time() - start_time
        logger.info(f"✓ Neural Network training completed in {elapsed:.1f}s")

        trained_models['neural_network'] = nn_model

    return trained_models


def train_ensembles(trained_models, X_train, y_train, X_val, y_val, output_dir, verbose=True):
    """Train ensemble models."""
    logger.info("\n" + "="*80)
    logger.info("TRAINING ENSEMBLE MODELS")
    logger.info("="*80)

    ensembles = {}

    # Stacking Ensemble
    logger.info("\n" + "-"*80)
    logger.info("Training Stacking Ensemble...")
    logger.info("-"*80)
    start_time = time.time()

    base_models = list(trained_models.values())
    stacking = StackingEnsemble(base_models=base_models, use_cv=True, cv_folds=5)
    stacking.fit(X_train, y_train, X_val, y_val, verbose=verbose)
    stacking.save_ensemble(output_dir / 'stacking_ensemble.pkl')

    elapsed = time.time() - start_time
    logger.info(f"✓ Stacking ensemble training completed in {elapsed:.1f}s")

    ensembles['stacking'] = stacking

    # Weighted Ensemble
    logger.info("\n" + "-"*80)
    logger.info("Training Weighted Ensemble...")
    logger.info("-"*80)
    start_time = time.time()

    weighted = WeightedEnsemble(models=base_models, optimization_metric='auc')
    weighted.fit(X_train, y_train, X_val, y_val, verbose=verbose)
    weighted.save_ensemble(output_dir / 'weighted_ensemble.pkl')

    elapsed = time.time() - start_time
    logger.info(f"✓ Weighted ensemble training completed in {elapsed:.1f}s")

    ensembles['weighted'] = weighted

    return ensembles


def main():
    """Main training pipeline."""
    args = parse_arguments()

    # Setup
    setup_logging()
    setup_gpu()
    set_random_seeds(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "="*80)
    logger.info("CREDIT RISK ASSESSMENT - TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Models to train: {args.models}")
    logger.info(f"Random seed: {args.seed}")

    # Load and split data
    train_df, val_df, test_df = load_and_split_data(Path(args.data_path), args)

    # Preprocess data
    if not args.no_preprocessing:
        X_train, y_train, X_val, y_val, X_test, y_test, preprocessor = preprocess_data(
            train_df, val_df, test_df, output_dir
        )

        # Engineer features
        X_train, X_val, X_test, engineer = engineer_features(
            X_train, X_val, X_test, output_dir
        )
    else:
        logger.info("Skipping preprocessing (using preprocessed data)")
        # Assume data is already preprocessed
        X_train, y_train = train_df.drop('target', axis=1), train_df['target']
        X_val, y_val = val_df.drop('target', axis=1), val_df['target']
        X_test, y_test = test_df.drop('target', axis=1), test_df['target']

    # Train base models
    trained_models = train_base_models(
        X_train, y_train, X_val, y_val,
        args.models, output_dir, args.verbose
    )

    # Train ensembles
    if args.ensemble and len(trained_models) > 1:
        ensembles = train_ensembles(
            trained_models, X_train, y_train, X_val, y_val,
            output_dir, args.verbose
        )
        trained_models.update(ensembles)

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Trained models: {list(trained_models.keys())}")
    logger.info(f"Models saved to: {output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
