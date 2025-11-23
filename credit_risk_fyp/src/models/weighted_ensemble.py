"""
Weighted Ensemble for Credit Risk Assessment FYP
Combines model predictions using optimized weights
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from ..config import ENSEMBLE_CONFIG, MODELS_DIR
from ..utils import save_pickle, load_pickle

# Setup logger
logger = logging.getLogger('credit_risk_fyp.models.weighted_ensemble')


class WeightedEnsemble:
    """
    Weighted averaging ensemble with automatic weight optimization.

    Features:
    - Scipy-based weight optimization
    - Constrained optimization (weights sum to 1)
    - Multiple optimization metrics
    """

    def __init__(
        self,
        models: List[Any],
        optimization_metric: str = 'auc'
    ):
        """
        Initialize Weighted Ensemble.

        Args:
            models: List of trained model instances
            optimization_metric: Metric to optimize ('auc', 'accuracy', 'f1')
        """
        self.models = models
        self.optimization_metric = optimization_metric
        self.weights = None
        self.model_names = []
        self.is_fitted = False

        # Get model names
        for i, model in enumerate(self.models):
            model_name = type(model).__name__
            self.model_names.append(f"{model_name}_{i}")

        logger.info(f"Initialized weighted ensemble with {len(self.models)} models")
        logger.info(f"Models: {self.model_names}")
        logger.info(f"Optimization metric: {optimization_metric}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        verbose: bool = True
    ) -> 'WeightedEnsemble':
        """
        Fit ensemble by optimizing weights on validation set.

        Args:
            X_train: Training features (not used, models already trained)
            y_train: Training labels (not used)
            X_val: Validation features
            y_val: Validation labels
            verbose: Print optimization progress

        Returns:
            Self for method chaining
        """
        logger.info("Optimizing weights for weighted ensemble...")
        logger.info(f"Validation set size: {len(X_val):,} samples")

        # Get predictions from all models on validation set
        predictions = self._collect_predictions(X_val)

        # Log individual model performance
        logger.info("\nIndividual model performance on validation set:")
        for i, model_name in enumerate(self.model_names):
            score = self._calculate_metric(y_val, predictions[:, i])
            logger.info(f"  {model_name}: {self.optimization_metric.upper()} = {score:.4f}")

        # Optimize weights
        self.weights = self.optimize_weights(predictions, y_val, verbose=verbose)

        # Evaluate ensemble
        ensemble_preds = np.average(predictions, axis=1, weights=self.weights)
        ensemble_score = self._calculate_metric(y_val, ensemble_preds)

        logger.info(f"\n✓ Weight optimization complete")
        logger.info(f"✓ Ensemble {self.optimization_metric.upper()}: {ensemble_score:.4f}")

        # Log optimized weights
        logger.info("\nOptimized weights:")
        for name, weight in zip(self.model_names, self.weights):
            logger.info(f"  {name}: {weight:.4f}")

        self.is_fitted = True

        return self

    def optimize_weights(
        self,
        predictions: np.ndarray,
        y_true: np.ndarray,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Optimize weights using scipy.optimize.

        Args:
            predictions: Predictions from all models (n_samples, n_models)
            y_true: True labels
            verbose: Print optimization progress

        Returns:
            Optimized weights
        """
        n_models = predictions.shape[1]

        # Objective function (negative metric to minimize)
        def objective(weights):
            weighted_pred = np.average(predictions, axis=1, weights=weights)
            score = self._calculate_metric(y_true, weighted_pred)
            return -score  # Negative because we minimize

        # Constraints: weights sum to 1
        constraints = {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1.0
        }

        # Bounds: each weight between 0 and 1
        bounds = [(0.0, 1.0) for _ in range(n_models)]

        # Initial weights (equal weighting)
        initial_weights = np.ones(n_models) / n_models

        # Optimize
        logger.info("Running scipy optimization...")
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': verbose}
        )

        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
            logger.warning("Using equal weights")
            return initial_weights

        optimized_weights = result.x

        # Ensure weights sum to 1 (numerical stability)
        optimized_weights = optimized_weights / np.sum(optimized_weights)

        return optimized_weights

    def _collect_predictions(self, X: pd.DataFrame) -> np.ndarray:
        """
        Collect predictions from all models.

        Args:
            X: Features

        Returns:
            Predictions array (n_samples, n_models)
        """
        n_samples = len(X)
        n_models = len(self.models)
        predictions = np.zeros((n_samples, n_models))

        for i, model in enumerate(self.models):
            try:
                preds = model.predict_proba(X)
                predictions[:, i] = preds
            except Exception as e:
                logger.error(f"Error getting predictions from {self.model_names[i]}: {e}")
                raise

        return predictions

    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate optimization metric.

        Args:
            y_true: True labels
            y_pred: Predicted probabilities

        Returns:
            Metric value
        """
        if self.optimization_metric == 'auc':
            return roc_auc_score(y_true, y_pred)
        elif self.optimization_metric == 'accuracy':
            y_pred_binary = (y_pred >= 0.5).astype(int)
            return (y_true == y_pred_binary).mean()
        elif self.optimization_metric == 'f1':
            from sklearn.metrics import f1_score
            y_pred_binary = (y_pred >= 0.5).astype(int)
            return f1_score(y_true, y_pred_binary)
        else:
            raise ValueError(f"Unknown metric: {self.optimization_metric}")

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Make binary predictions.

        Args:
            X: Features to predict
            threshold: Classification threshold

        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

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
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        # Get predictions from all models
        predictions = self._collect_predictions(X)

        # Weighted average
        weighted_predictions = np.average(predictions, axis=1, weights=self.weights)

        return weighted_predictions

    def get_weights(self) -> Dict[str, float]:
        """
        Get model weights as dictionary.

        Returns:
            Dictionary mapping model names to weights
        """
        if self.weights is None:
            return {}

        return dict(zip(self.model_names, self.weights))

    def get_model_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get predictions from all models.

        Args:
            X: Features to predict

        Returns:
            DataFrame with predictions from each model
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        predictions = self._collect_predictions(X)

        predictions_df = pd.DataFrame(
            predictions,
            columns=self.model_names
        )

        return predictions_df

    def plot_weights(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot model weights.

        Args:
            figsize: Figure size
            save_path: Optional path to save figure
        """
        if self.weights is None:
            logger.warning("No weights available")
            return

        plt.figure(figsize=figsize)
        plt.bar(self.model_names, self.weights)
        plt.xlabel('Model')
        plt.ylabel('Weight')
        plt.title('Optimized Model Weights in Weighted Ensemble')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved weights plot to {save_path}")

        plt.show()

    def plot_prediction_correlation(
        self,
        X: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot correlation between model predictions.

        Args:
            X: Features for predictions
            figsize: Figure size
            save_path: Optional path to save figure
        """
        import seaborn as sns

        predictions_df = self.get_model_predictions(X)
        correlation = predictions_df.corr()

        plt.figure(figsize=figsize)
        sns.heatmap(
            correlation,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1
        )
        plt.title('Model Prediction Correlation')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved correlation plot to {save_path}")

        plt.show()

    def save_ensemble(self, filepath: Union[str, Path]) -> None:
        """
        Save ensemble to file.

        Args:
            filepath: Path to save ensemble
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted ensemble")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_pickle(self, filepath)
        logger.info(f"Saved weighted ensemble to {filepath}")

    @classmethod
    def load_ensemble(cls, filepath: Union[str, Path]) -> 'WeightedEnsemble':
        """
        Load ensemble from file.

        Args:
            filepath: Path to saved ensemble

        Returns:
            Loaded WeightedEnsemble instance
        """
        ensemble = load_pickle(filepath)
        logger.info(f"Loaded weighted ensemble from {filepath}")
        return ensemble
