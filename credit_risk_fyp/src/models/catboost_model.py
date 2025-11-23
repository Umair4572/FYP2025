"""
CatBoost Model for Credit Risk Assessment FYP
GPU-accelerated gradient boosting with native categorical feature support
"""

import logging
from typing import Dict, Optional, Tuple, Union, Any, List
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from ..config import CATBOOST_PARAMS, MODELS_DIR
from ..utils import save_pickle, load_pickle

# Setup logger
logger = logging.getLogger('credit_risk_fyp.models.catboost')


class CatBoostModel:
    """
    CatBoost classifier with GPU acceleration for credit risk prediction.

    Features:
    - GPU-optimized training
    - Native categorical feature handling
    - Early stopping
    - Built-in plotting capabilities
    - Feature importance analysis
    - Model persistence
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize CatBoost model.

        Args:
            params: Optional model parameters (uses config defaults if None)
        """
        self.params = params or CATBOOST_PARAMS.copy()
        self.model = None
        self.best_iteration = None
        self.feature_names = None
        self.categorical_features = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        categorical_features: Optional[List[str]] = None,
        verbose: bool = True
    ) -> 'CatBoostModel':
        """
        Train CatBoost model with optional validation set.

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
        logger.info("Training CatBoost model...")
        logger.info(f"Training set size: {len(X_train):,} samples, {X_train.shape[1]} features")

        # Store feature names
        self.feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None
        self.categorical_features = categorical_features or []

        # Calculate class weights
        scale_pos_weight = self._calculate_scale_pos_weight(y_train)

        logger.info(f"Class distribution - 0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,}")
        logger.info(f"Scale pos weight: {scale_pos_weight:.4f}")

        # Update params with class weight
        train_params = self.params.copy()
        train_params['scale_pos_weight'] = scale_pos_weight

        # Set verbosity
        if not verbose:
            train_params['verbose'] = 0

        # Create Pool objects for CatBoost
        train_pool = Pool(
            X_train,
            label=y_train,
            cat_features=self.categorical_features
        )

        eval_set = None
        if X_val is not None and y_val is not None:
            val_pool = Pool(
                X_val,
                label=y_val,
                cat_features=self.categorical_features
            )
            eval_set = val_pool
            logger.info(f"Validation set size: {len(X_val):,} samples")

        # Initialize and train model
        self.model = CatBoostClassifier(**train_params)

        self.model.fit(
            train_pool,
            eval_set=eval_set,
            use_best_model=True,
            plot=False  # Disable automatic plotting
        )

        # Store best iteration
        self.best_iteration = self.model.get_best_iteration()

        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            val_pred_proba = self.predict_proba(X_val)
            val_auc = roc_auc_score(y_val, val_pred_proba)
            logger.info(f"✓ Training complete. Best iteration: {self.best_iteration}")
            logger.info(f"✓ Validation AUC: {val_auc:.4f}")
        else:
            logger.info(f"✓ Training complete. Total iterations: {self.model.tree_count_}")

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

        # CatBoost returns probabilities for both classes, we want class 1
        proba = self.model.predict_proba(X)
        return proba[:, 1]

    def get_feature_importance(
        self,
        importance_type: str = 'FeatureImportance'
    ) -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            importance_type: Type of importance ('FeatureImportance', 'PredictionValuesChange', 'LossFunctionChange')

        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        importance = self.model.get_feature_importance(type=importance_type)

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
        importance_type: str = 'FeatureImportance',
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
        plt.title(f'CatBoost Top {top_n} Feature Importance')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")

        plt.show()

    def plot_learning_curves(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot learning curves (training and validation metrics).

        Args:
            figsize: Figure size
            save_path: Optional path to save figure
        """
        if self.model is None:
            logger.warning("No model available for plotting")
            return

        # Get evaluation results
        evals_result = self.model.get_evals_result()

        if not evals_result:
            logger.warning("No evaluation results available")
            return

        plt.figure(figsize=figsize)

        # Plot for each dataset
        for dataset_name, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                plt.plot(values, label=f'{dataset_name}_{metric_name}')

        plt.xlabel('Iteration')
        plt.ylabel('Metric Value')
        plt.title('CatBoost Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved learning curves to {save_path}")

        plt.show()

    def get_object_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get object (instance) importance - most influential training examples.

        Args:
            X: Features
            y: Labels
            top_n: Number of top instances to return

        Returns:
            DataFrame with object importances
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        pool = Pool(X, label=y, cat_features=self.categorical_features)
        importance = self.model.get_object_importance(pool, train_pool=pool, top_size=top_n)

        importance_df = pd.DataFrame({
            'index': range(len(importance)),
            'importance': importance
        })

        return importance_df

    def get_best_iteration(self) -> int:
        """
        Get the best iteration number.

        Returns:
            Best iteration number
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.best_iteration

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

        logger.info(f"Saved CatBoost model to {filepath}")

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'CatBoostModel':
        """
        Load trained model from file.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded CatBoostModel instance
        """
        model = load_pickle(filepath)
        logger.info(f"Loaded CatBoost model from {filepath}")
        return model


# Convenience training function
def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict] = None,
    categorical_features: Optional[List[str]] = None
) -> CatBoostModel:
    """
    Quick function to train CatBoost model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: Optional model parameters
        categorical_features: List of categorical feature names

    Returns:
        Trained CatBoostModel
    """
    model = CatBoostModel(params)
    model.train(X_train, y_train, X_val, y_val, categorical_features=categorical_features)
    return model
