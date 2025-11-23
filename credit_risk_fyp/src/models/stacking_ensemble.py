"""
Stacking Ensemble for Credit Risk Assessment FYP
Combines multiple base models using a meta-learner
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from ..config import ENSEMBLE_CONFIG, MODELS_DIR
from ..utils import save_pickle, load_pickle

# Setup logger
logger = logging.getLogger('credit_risk_fyp.models.stacking_ensemble')


class StackingEnsemble:
    """
    Stacking ensemble that combines predictions from multiple base models.

    Architecture:
    - Level 0: Multiple base models (XGBoost, LightGBM, CatBoost, RF, NN)
    - Level 1: Meta-model trained on base model predictions

    Features:
    - Cross-validation for meta-feature generation
    - Prevents overfitting through out-of-fold predictions
    - Flexible meta-model selection
    """

    def __init__(
        self,
        base_models: List[Any],
        meta_model: Optional[Any] = None,
        use_cv: bool = True,
        cv_folds: int = 5
    ):
        """
        Initialize Stacking Ensemble.

        Args:
            base_models: List of base model instances
            meta_model: Meta-model instance (default: LogisticRegression)
            use_cv: Whether to use cross-validation for meta-features
            cv_folds: Number of CV folds
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.base_model_names = []
        self.is_fitted = False

        # Initialize meta-model if not provided
        if self.meta_model is None:
            meta_config = ENSEMBLE_CONFIG['stacking']['meta_model_params']
            self.meta_model = LogisticRegression(**meta_config)
            logger.info("Using Logistic Regression as meta-model")

        # Get base model names
        for i, model in enumerate(self.base_models):
            model_name = type(model).__name__
            self.base_model_names.append(f"{model_name}_{i}")

        logger.info(f"Initialized stacking ensemble with {len(self.base_models)} base models")
        logger.info(f"Base models: {self.base_model_names}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> 'StackingEnsemble':
        """
        Train stacking ensemble.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Print training progress

        Returns:
            Self for method chaining
        """
        logger.info("Training Stacking Ensemble...")
        logger.info(f"Training set size: {len(X_train):,} samples")

        if self.use_cv:
            logger.info(f"Using {self.cv_folds}-fold cross-validation for meta-features")

            # Generate meta-features using CV
            meta_X_train = self._generate_meta_features_cv(X_train, y_train, verbose)

            # Train base models on full training set
            self._train_base_models_full(X_train, y_train, X_val, y_val, verbose)

        else:
            logger.info("Using validation set for meta-features")

            if X_val is None or y_val is None:
                raise ValueError("Validation set required when use_cv=False")

            # Train base models on training set
            self._train_base_models(X_train, y_train, X_val, y_val, verbose)

            # Generate meta-features from validation set
            meta_X_train = self._generate_meta_features(X_val, is_training=False)
            y_train = y_val  # Use validation labels for meta-model

        # Train meta-model
        logger.info("Training meta-model...")
        self._train_meta_model(meta_X_train, y_train, X_val, y_val, verbose)

        self.is_fitted = True
        logger.info("✓ Stacking ensemble training complete")

        return self

    def _generate_meta_features_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Generate meta-features using cross-validation.

        Args:
            X: Features
            y: Labels
            verbose: Print progress

        Returns:
            Meta-features array (n_samples, n_base_models)
        """
        logger.info("Generating meta-features with cross-validation...")

        n_samples = len(X)
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))

        # Cross-validation setup
        kfold = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        for model_idx, model in enumerate(self.base_models):
            model_name = self.base_model_names[model_idx]
            logger.info(f"Processing {model_name} ({model_idx + 1}/{n_models})...")

            fold_predictions = np.zeros(n_samples)

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                if verbose:
                    logger.info(f"  Fold {fold_idx + 1}/{self.cv_folds}")

                # Split data
                X_fold_train = X.iloc[train_idx]
                y_fold_train = y.iloc[train_idx]
                X_fold_val = X.iloc[val_idx]

                # Create a copy of the model for this fold
                import copy
                fold_model = copy.deepcopy(model)

                # Train on fold
                try:
                    fold_model.train(X_fold_train, y_fold_train, verbose=False)
                except Exception as e:
                    logger.error(f"Error training {model_name} on fold {fold_idx}: {e}")
                    continue

                # Predict on held-out fold
                fold_preds = fold_model.predict_proba(X_fold_val)
                fold_predictions[val_idx] = fold_preds

            # Store out-of-fold predictions as meta-features
            meta_features[:, model_idx] = fold_predictions

            # Calculate OOF AUC
            oof_auc = roc_auc_score(y, fold_predictions)
            logger.info(f"  {model_name} Out-of-Fold AUC: {oof_auc:.4f}")

        return meta_features

    def _train_base_models_full(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        verbose: bool = True
    ) -> None:
        """
        Train base models on full training set.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            verbose: Print progress
        """
        logger.info("Training base models on full training set...")

        for model_idx, model in enumerate(self.base_models):
            model_name = self.base_model_names[model_idx]
            logger.info(f"Training {model_name} ({model_idx + 1}/{len(self.base_models)})...")

            try:
                model.train(X_train, y_train, X_val, y_val, verbose=verbose)

                # Evaluate
                if X_val is not None and y_val is not None:
                    val_preds = model.predict_proba(X_val)
                    val_auc = roc_auc_score(y_val, val_preds)
                    logger.info(f"  {model_name} Validation AUC: {val_auc:.4f}")

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                raise

    def _train_base_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        verbose: bool = True
    ) -> None:
        """
        Train base models on training set (without CV).

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            verbose: Print progress
        """
        logger.info("Training base models...")

        for model_idx, model in enumerate(self.base_models):
            model_name = self.base_model_names[model_idx]
            logger.info(f"Training {model_name} ({model_idx + 1}/{len(self.base_models)})...")

            try:
                model.train(X_train, y_train, X_val, y_val, verbose=verbose)

                # Evaluate
                val_preds = model.predict_proba(X_val)
                val_auc = roc_auc_score(y_val, val_preds)
                logger.info(f"  {model_name} Validation AUC: {val_auc:.4f}")

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                raise

    def _generate_meta_features(
        self,
        X: pd.DataFrame,
        is_training: bool = False
    ) -> np.ndarray:
        """
        Generate meta-features from base model predictions.

        Args:
            X: Features
            is_training: Whether generating for training (affects logging)

        Returns:
            Meta-features array (n_samples, n_base_models)
        """
        n_samples = len(X)
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))

        for model_idx, model in enumerate(self.base_models):
            predictions = model.predict_proba(X)
            meta_features[:, model_idx] = predictions

        return meta_features

    def _train_meta_model(
        self,
        meta_X: np.ndarray,
        y: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        verbose: bool = True
    ) -> None:
        """
        Train meta-model on meta-features.

        Args:
            meta_X: Meta-features
            y: Labels
            X_val: Validation features (for evaluation)
            y_val: Validation labels (for evaluation)
            verbose: Print progress
        """
        logger.info(f"Training meta-model with {meta_X.shape[1]} features...")

        # Train meta-model
        self.meta_model.fit(meta_X, y)

        # Evaluate meta-model
        meta_train_preds = self.meta_model.predict_proba(meta_X)[:, 1]
        meta_train_auc = roc_auc_score(y, meta_train_preds)
        logger.info(f"Meta-model training AUC: {meta_train_auc:.4f}")

        # Evaluate on validation set if available
        if X_val is not None and y_val is not None:
            val_meta_features = self._generate_meta_features(X_val)
            meta_val_preds = self.meta_model.predict_proba(val_meta_features)[:, 1]
            meta_val_auc = roc_auc_score(y_val, meta_val_preds)
            logger.info(f"Meta-model validation AUC: {meta_val_auc:.4f}")

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

        # Generate meta-features from base models
        meta_features = self._generate_meta_features(X)

        # Predict with meta-model
        predictions = self.meta_model.predict_proba(meta_features)[:, 1]

        return predictions

    def get_base_model_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get predictions from all base models.

        Args:
            X: Features to predict

        Returns:
            DataFrame with predictions from each base model
        """
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted. Call fit() first.")

        meta_features = self._generate_meta_features(X)

        predictions_df = pd.DataFrame(
            meta_features,
            columns=self.base_model_names
        )

        return predictions_df

    def get_meta_model_weights(self) -> Optional[np.ndarray]:
        """
        Get meta-model coefficients (if applicable).

        Returns:
            Meta-model coefficients or None
        """
        if hasattr(self.meta_model, 'coef_'):
            return self.meta_model.coef_[0]
        return None

    def plot_base_model_importance(
        self,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot meta-model coefficients (model importance).

        Args:
            figsize: Figure size
            save_path: Optional path to save figure
        """
        weights = self.get_meta_model_weights()

        if weights is None:
            logger.warning("Meta-model does not have coefficients")
            return

        plt.figure(figsize=figsize)
        plt.barh(self.base_model_names, weights)
        plt.xlabel('Meta-Model Coefficient')
        plt.title('Base Model Importance in Stacking Ensemble')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model importance plot to {save_path}")

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
        logger.info(f"Saved stacking ensemble to {filepath}")

    @classmethod
    def load_ensemble(cls, filepath: Union[str, Path]) -> 'StackingEnsemble':
        """
        Load ensemble from file.

        Args:
            filepath: Path to saved ensemble

        Returns:
            Loaded StackingEnsemble instance
        """
        ensemble = load_pickle(filepath)
        logger.info(f"Loaded stacking ensemble from {filepath}")
        return ensemble
