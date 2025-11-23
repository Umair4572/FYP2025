"""
Neural Network Model for Credit Risk Assessment FYP
Deep learning classifier with TensorFlow and GPU acceleration
"""

import logging
from typing import Dict, Optional, Tuple, Union, Any, List
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks, optimizers
    from tensorflow.keras.mixed_precision import Policy
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger = logging.getLogger('credit_risk_fyp.models.neural_network')
    logger.warning("TensorFlow not available. Neural Network model will not work.")

from sklearn.metrics import roc_auc_score

from ..config import NEURAL_NETWORK_PARAMS, MODELS_DIR
from ..utils import save_pickle, load_pickle, get_class_weights

# Setup logger
logger = logging.getLogger('credit_risk_fyp.models.neural_network')


class NeuralNetworkModel:
    """
    Deep Neural Network classifier with GPU acceleration for credit risk prediction.

    Features:
    - GPU-optimized training with mixed precision
    - Batch normalization and dropout
    - Early stopping and learning rate reduction
    - Training history visualization
    - Model persistence
    """

    def __init__(self, input_dim: Optional[int] = None, params: Optional[Dict] = None):
        """
        Initialize Neural Network model.

        Args:
            input_dim: Number of input features
            params: Optional model parameters (uses config defaults if None)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for Neural Network model")

        self.params = params or NEURAL_NETWORK_PARAMS.copy()
        self.input_dim = input_dim
        self.model = None
        self.history = None
        self.feature_names = None

        # Setup GPU and mixed precision
        self._setup_gpu()

    def _setup_gpu(self) -> None:
        """Setup GPU configuration for optimal performance."""
        # Enable memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logger.warning(f"GPU setup error: {e}")

        # Enable mixed precision for faster training
        try:
            policy = Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision (float16) enabled")
        except Exception as e:
            logger.warning(f"Mixed precision setup failed: {e}")

    def build_model(self, input_dim: Optional[int] = None) -> keras.Model:
        """
        Build the neural network architecture.

        Args:
            input_dim: Number of input features

        Returns:
            Compiled Keras model
        """
        if input_dim is None:
            input_dim = self.input_dim

        if input_dim is None:
            raise ValueError("input_dim must be specified")

        self.input_dim = input_dim

        # Get architecture parameters
        hidden_layers = self.params['architecture']
        dropout_rates = self.params['dropout_rate']
        activation = self.params['activation']
        l2_reg = self.params['l2_regularization']
        batch_norm = self.params['batch_normalization']

        # Ensure dropout_rates matches hidden_layers
        if len(dropout_rates) < len(hidden_layers):
            dropout_rates = dropout_rates + [dropout_rates[-1]] * (len(hidden_layers) - len(dropout_rates))

        logger.info(f"Building neural network with {len(hidden_layers)} hidden layers")
        logger.info(f"Architecture: {hidden_layers}")

        # Build model
        model = models.Sequential(name='CreditRiskNN')

        # Input layer
        model.add(layers.Input(shape=(input_dim,), name='input'))

        # Hidden layers
        for i, (units, dropout) in enumerate(zip(hidden_layers, dropout_rates)):
            # Dense layer with L2 regularization
            model.add(layers.Dense(
                units,
                activation=activation,
                kernel_regularizer=keras.regularizers.l2(l2_reg),
                name=f'dense_{i+1}'
            ))

            # Batch normalization
            if batch_norm:
                model.add(layers.BatchNormalization(name=f'bn_{i+1}'))

            # Dropout
            if dropout > 0:
                model.add(layers.Dropout(dropout, name=f'dropout_{i+1}'))

        # Output layer (binary classification with sigmoid)
        model.add(layers.Dense(1, activation='sigmoid', dtype='float32', name='output'))

        # Compile model
        optimizer_name = self.params['optimizer']
        lr = self.params['learning_rate']

        if optimizer_name.lower() == 'adam':
            optimizer = optimizers.Adam(learning_rate=lr)
        elif optimizer_name.lower() == 'sgd':
            optimizer = optimizers.SGD(learning_rate=lr)
        else:
            optimizer = optimizer_name

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

        logger.info(f"Model built with {model.count_params():,} parameters")

        return model

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: int = 1
    ) -> 'NeuralNetworkModel':
        """
        Train neural network model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)

        Returns:
            Self for method chaining
        """
        logger.info("Training Neural Network model...")
        logger.info(f"Training set size: {len(X_train):,} samples, {X_train.shape[1]} features")

        # Store feature names
        self.feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else None

        # Build model if not already built
        if self.model is None:
            self.model = self.build_model(input_dim=X_train.shape[1])

        # Calculate class weights
        class_weights = get_class_weights(y_train.values)

        logger.info(f"Class distribution - 0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,}")

        # Setup callbacks
        callback_list = self._setup_callbacks(X_val is not None and y_val is not None)

        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            logger.info(f"Validation set size: {len(X_val):,} samples")

        # Convert to numpy if needed
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, pd.Series) else y_train

        if validation_data:
            X_val_np = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_np = y_val.values if isinstance(y_val, pd.Series) else y_val
            validation_data = (X_val_np, y_val_np)

        # Train model
        logger.info("Starting training...")
        self.history = self.model.fit(
            X_train_np,
            y_train_np,
            validation_data=validation_data,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            class_weight=class_weights,
            callbacks=callback_list,
            verbose=verbose
        )

        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            val_pred_proba = self.predict_proba(X_val)
            val_auc = roc_auc_score(y_val, val_pred_proba)
            logger.info(f"✓ Training complete")
            logger.info(f"✓ Validation AUC: {val_auc:.4f}")
        else:
            logger.info(f"✓ Training complete")

        return self

    def _setup_callbacks(self, has_validation: bool = True) -> List[callbacks.Callback]:
        """
        Setup training callbacks.

        Args:
            has_validation: Whether validation data is available

        Returns:
            List of callbacks
        """
        callback_list = []

        # Early stopping
        if has_validation:
            early_stop = callbacks.EarlyStopping(
                monitor='val_auc',
                patience=self.params['early_stopping_patience'],
                mode='max',
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(early_stop)

        # Reduce learning rate on plateau
        if has_validation:
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                factor=self.params['reduce_lr_factor'],
                patience=self.params['reduce_lr_patience'],
                mode='max',
                min_lr=self.params['min_lr'],
                verbose=1
            )
            callback_list.append(reduce_lr)

        # Model checkpoint (save best model)
        checkpoint_path = MODELS_DIR / 'nn_checkpoint.h5'
        checkpoint = callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_auc' if has_validation else 'auc',
            mode='max',
            save_best_only=True,
            verbose=0
        )
        callback_list.append(checkpoint)

        return callback_list

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

        X_np = X.values if isinstance(X, pd.DataFrame) else X
        proba = self.model.predict(X_np, verbose=0)
        return proba.flatten()

    def plot_training_history(
        self,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[Path] = None
    ) -> None:
        """
        Plot training history.

        Args:
            metrics: List of metrics to plot (default: ['loss', 'auc', 'accuracy'])
            figsize: Figure size
            save_path: Optional path to save figure
        """
        if self.history is None:
            logger.warning("No training history available")
            return

        if metrics is None:
            metrics = ['loss', 'auc', 'accuracy']

        # Filter metrics that are actually in history
        available_metrics = [m for m in metrics if m in self.history.history]

        if not available_metrics:
            logger.warning("None of the specified metrics are available")
            return

        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]

            # Plot training metric
            ax.plot(self.history.history[metric], label=f'Train {metric}')

            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in self.history.history:
                ax.plot(self.history.history[val_metric], label=f'Val {metric}')

            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Over Epochs')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training history plot to {save_path}")

        plt.show()

    def get_model_summary(self) -> str:
        """
        Get model architecture summary.

        Returns:
            Model summary as string
        """
        if self.model is None:
            return "Model not built yet"

        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()

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

        # Also save Keras model separately
        keras_path = filepath.with_suffix('.h5')
        self.model.save(keras_path)

        logger.info(f"Saved Neural Network model to {filepath}")
        logger.info(f"Saved Keras model to {keras_path}")

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'NeuralNetworkModel':
        """
        Load trained model from file.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded NeuralNetworkModel instance
        """
        model = load_pickle(filepath)
        logger.info(f"Loaded Neural Network model from {filepath}")
        return model


# Convenience training function
def train_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[Dict] = None
) -> NeuralNetworkModel:
    """
    Quick function to train Neural Network model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        params: Optional model parameters

    Returns:
        Trained NeuralNetworkModel
    """
    model = NeuralNetworkModel(input_dim=X_train.shape[1], params=params)
    model.train(X_train, y_train, X_val, y_val)
    return model
