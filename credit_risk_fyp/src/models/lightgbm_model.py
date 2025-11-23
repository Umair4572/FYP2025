"""
LightGBM Model for Credit Risk Assessment FYP
GPU-accelerated gradient boosting classifier
"""

import logging
from typing import Dict, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from ..config import LIGHTGBM_PARAMS, MODELS_DIR
from ..utils import save_pickle, load_pickle

# Setup logger
logger = logging.getLogger('credit_risk_fyp.models.lightgbm')


class LightGBMModel:
    """
    LightGBM classifier with GPU acceleration for credit risk prediction.

    Features:
    - GPU-optimized training
    - Early stopping
    - Native categorical feature support
    - Feature importance analysis
    - Model persistence
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize LightGBM model.

        Args:
            params: Optional model parameters (uses config defaults if None)
        """
        self.params = params or LIGHTGBM_PARAMS.copy()
        self.model = None
        self.best_iteration = None
        self.feature_names = None
        self.categorical_features = []
        self.training_history = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        categorical_features: Optional[list] = None,
        verbose: bool = True
    ) -> 'LightGBMModel':
        """
        Train LightGBM model with optional validation set.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            categorical_features: List of categorical feature names
            verbose: Print training progress

        Returns:
            Self for method chaining
        """
        logger.info("Training LightGBM model...")
        logger.info(f"Training set size: {len(X_train):,} samples, {X_train.shape[1]} features")

        # Store feature names
        self.feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
        self.categorical_features = categorical_features or []

        # Calculate class weights
        scale_pos_weight = self._calculate_scale_pos_weight(y_train)
        self.params['scale_pos_weight'] = scale_pos_weight

        logger.info(f"Class distribution - 0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,}")
        logger.info(f"Scale pos weight: {scale_pos_weight:.4f}")

        # Create Dataset for LightGBM
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.feature_names,
            categorical_feature=self.categorical_features,
            free_raw_data=False
        )

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(
                X_val,
                label=y_val,
                feature_name=self.feature_names,
                categorical_feature=self.categorical_features,
                reference=train_data,
                free_raw_data=False
            )
            valid_sets.append(valid_data)
            valid_names.append('valid')
            logger.info(f"Validation set size: {len(X_val):,} samples")

        # Extract training-specific parameters
        early_stopping_rounds = self.params.pop('early_stopping_rounds', 50)
        verbose_eval = 50 if verbose else False

        # Callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose),
            lgb.log_evaluation(period=verbose_eval if verbose else 0)
        ]

        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params.get('n_estimators', 1000),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        # Store best iteration
        self.best_iteration = self.model.best_iteration

        # Restore parameters
        self.params['early_stopping_rounds'] = early_stopping_rounds

        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            val_pred_proba = self.predict_proba(X_val)
            val_auc = roc_auc_score(y_val, val_pred_proba)
            logger.info(f"✓ Training complete. Best iteration: {self.best_iteration}")
            logger.info(f"✓ Validation AUC: {val_auc:.4f}")
        else:
            logger.info(f"✓ Training complete. Best iteration: {self.best_iteration}")

        return self

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions.

        Args:
            X: Features to predict
            threshold: Classification threshold

        Returns:
            Binary predictions (0 or 1)
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of default (class 1).

        Args:
            X: Features to predict

        Returns:
            Probability predictions
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.model.predict(X, num_iteration=self.best_iteration)

    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            importance_type: Type of importance ('gain' or 'split')

        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        importance = self.model.feature_importance(importance_type=importance_type)

        # Convert to DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names or [f'feature_{i}' for i in range(len(importance))],
            'importance': importance
        })

        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df

    def plot_feature_importance(
        self,
        top_n: int = 20,
        importance_type: str = 'gain',
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot feature importance.

        Args:
            top_n: Number of top features to show
            importance_type: Type of importance to plot
            figsize: Figure size
            save_path: Optional path to save figure
        """
        importance_df = self.get_feature_importance(importance_type)

        plt.figure(figsize=figsize)
        plt.barh(
            importance_df['feature'].head(top_n)[::-1],
            importance_df['importance'].head(top_n)[::-1]
        )
        plt.xlabel(f'Importance ({importance_type})')
        plt.title(f'LightGBM Top {top_n} Feature Importance')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")

        plt.show()

    def plot_training_history(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot training history using LightGBM's built-in plotting.

        Args:
            figsize: Figure size
            save_path: Optional path to save figure
        """
        if self.model is None:
            logger.warning("No model available for plotting")
            return

        plt.figure(figsize=figsize)
        lgb.plot_metric(self.model, metric='auc')
        plt.title('LightGBM Training History')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training history plot to {save_path}")

        plt.show()

    def _calculate_scale_pos_weight(self, y: pd.Series) -> float:
        """
        Calculate scale_pos_weight for imbalanced datasets.

        Args:
            y: Target labels

        Returns:
            Scale positive weight
        """
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()

        if pos_count == 0:
            return 1.0

        return neg_count / pos_count

    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save trained model to file.

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise RuntimeError("Cannot save untrained model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save full object
        save_pickle(self, filepath)

        logger.info(f"Saved LightGBM model to {filepath}")

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'LightGBMModel':
        """
        Load trained model from file.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded LightGBMModel instance
        """
        model = load_pickle(filepath)
        logger.info(f"Loaded LightGBM model from {filepath}")
        return model


# Convenience training function
def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict] = None,
    categorical_features: Optional[list] = None
) -> LightGBMModel:
    """
    Quick function to train LightGBM model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: Optional model parameters
        categorical_features: List of categorical feature names

    Returns:
        Trained LightGBMModel
    """
    model = LightGBMModel(params)
    model.train(X_train, y_train, X_val, y_val, categorical_features=categorical_features)
    return model
