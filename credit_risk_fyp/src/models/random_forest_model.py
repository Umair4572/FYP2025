"""
Random Forest Model for Credit Risk Assessment FYP
Multi-threaded ensemble of decision trees
"""

import logging
from typing import Dict, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

from ..config import RANDOM_FOREST_PARAMS, MODELS_DIR
from ..utils import save_pickle, load_pickle

# Setup logger
logger = logging.getLogger('credit_risk_fyp.models.random_forest')


class RandomForestModel:
    """
    Random Forest classifier with multi-threading for credit risk prediction.

    Features:
    - CPU parallelization with n_jobs=-1
    - Out-of-bag (OOB) scoring
    - Feature importance analysis
    - Tree visualization
    - Model persistence
    """

    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize Random Forest model.

        Args:
            params: Optional model parameters (uses config defaults if None)
        """
        self.params = params or RANDOM_FOREST_PARAMS.copy()
        self.model = None
        self.feature_names = None
        self.oob_score = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> 'RandomForestModel':
        """
        Train Random Forest model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for evaluation only)
            y_val: Validation labels (optional, for evaluation only)
            verbose: Print training progress

        Returns:
            Self for method chaining
        """
        logger.info("Training Random Forest model...")
        logger.info(f"Training set size: {len(X_train):,} samples, {X_train.shape[1]} features")

        # Store feature names
        self.feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None

        logger.info(f"Class distribution - 0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,}")

        # Add OOB score calculation
        train_params = self.params.copy()
        train_params['oob_score'] = True

        # Initialize and train model
        self.model = RandomForestClassifier(**train_params)

        logger.info(f"Training with {train_params['n_estimators']} trees...")
        self.model.fit(X_train, y_train)

        # Get OOB score
        self.oob_score = self.model.oob_score_
        logger.info(f"✓ Training complete")
        logger.info(f"✓ Out-of-bag Score: {self.oob_score:.4f}")

        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred_proba = self.predict_proba(X_val)
            val_auc = roc_auc_score(y_val, val_pred_proba)
            logger.info(f"✓ Validation AUC: {val_auc:.4f}")

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

        proba = self.model.predict_proba(X)
        return proba[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on Gini impurity.

        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        importance = self.model.feature_importances_

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
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot feature importance.

        Args:
            top_n: Number of top features to show
            figsize: Figure size
            save_path: Optional path to save figure
        """
        importance_df = self.get_feature_importance()

        plt.figure(figsize=figsize)
        plt.barh(
            importance_df['feature'].head(top_n)[::-1],
            importance_df['importance'].head(top_n)[::-1]
        )
        plt.xlabel('Importance (Gini Impurity)')
        plt.title(f'Random Forest Top {top_n} Feature Importance')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")

        plt.show()

    def get_oob_score(self) -> float:
        """
        Get out-of-bag score.

        Returns:
            OOB score
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.oob_score

    def plot_tree(
        self,
        tree_index: int = 0,
        max_depth: int = 3,
        figsize: Tuple[int, int] = (20, 10),
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot a single decision tree from the forest.

        Args:
            tree_index: Index of tree to plot
            max_depth: Maximum depth to display
            figsize: Figure size
            save_path: Optional path to save figure
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if tree_index >= len(self.model.estimators_):
            raise ValueError(f"Tree index {tree_index} out of range (0-{len(self.model.estimators_)-1})")

        plt.figure(figsize=figsize)
        plot_tree(
            self.model.estimators_[tree_index],
            feature_names=self.feature_names,
            class_names=['Non-default', 'Default'],
            filled=True,
            max_depth=max_depth,
            fontsize=10
        )
        plt.title(f'Decision Tree {tree_index} (max_depth={max_depth})')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved tree plot to {save_path}")

        plt.show()

    def get_tree_depths(self) -> pd.DataFrame:
        """
        Get depth statistics for all trees in the forest.

        Returns:
            DataFrame with tree depth statistics
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        depths = [tree.get_depth() for tree in self.model.estimators_]

        depth_stats = pd.DataFrame({
            'tree_index': range(len(depths)),
            'depth': depths
        })

        logger.info(f"Tree depth statistics:")
        logger.info(f"  Mean: {np.mean(depths):.2f}")
        logger.info(f"  Median: {np.median(depths):.2f}")
        logger.info(f"  Min: {np.min(depths)}")
        logger.info(f"  Max: {np.max(depths)}")

        return depth_stats

    def plot_tree_depths(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot distribution of tree depths.

        Args:
            figsize: Figure size
            save_path: Optional path to save figure
        """
        depth_stats = self.get_tree_depths()

        plt.figure(figsize=figsize)
        plt.hist(depth_stats['depth'], bins=30, edgecolor='black')
        plt.xlabel('Tree Depth')
        plt.ylabel('Frequency')
        plt.title('Distribution of Tree Depths in Random Forest')
        plt.axvline(depth_stats['depth'].mean(), color='r', linestyle='--', label=f'Mean: {depth_stats["depth"].mean():.2f}')
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved tree depth plot to {save_path}")

        plt.show()

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

        logger.info(f"Saved Random Forest model to {filepath}")

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'RandomForestModel':
        """
        Load trained model from file.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded RandomForestModel instance
        """
        model = load_pickle(filepath)
        logger.info(f"Loaded Random Forest model from {filepath}")
        return model


# Convenience training function
def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict] = None
) -> RandomForestModel:
    """
    Quick function to train Random Forest model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        params: Optional model parameters

    Returns:
        Trained RandomForestModel
    """
    model = RandomForestModel(params)
    model.train(X_train, y_train, X_val, y_val)
    return model
