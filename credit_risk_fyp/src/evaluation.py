"""
Evaluation Module for Credit Risk Assessment FYP
Comprehensive model evaluation with metrics and visualizations
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve

from .config import EVALUATION_CONFIG, VISUALIZATION_CONFIG, FIGURES_DIR, REPORTS_DIR

# Setup logger
logger = logging.getLogger('credit_risk_fyp.evaluation')


class ModelEvaluator:
    """
    Comprehensive model evaluation suite.

    Features:
    - Multiple classification metrics
    - ROC and Precision-Recall curves
    - Confusion matrices
    - Calibration plots
    - Threshold optimization
    - Model comparison
    - Report generation
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ModelEvaluator.

        Args:
            config: Optional evaluation configuration
        """
        self.config = config or EVALUATION_CONFIG

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        model_name: str = "Model"
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            model_name: Name of model being evaluated

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {model_name}...")

        # Binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate all metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)

        # Log results
        logger.info(f"\n{model_name} Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        return metrics

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        # Additional metrics
        metrics['specificity'] = self._calculate_specificity(y_true, y_pred)
        metrics['false_positive_rate'] = 1 - metrics['specificity']
        metrics['false_negative_rate'] = 1 - metrics['recall']

        # Statistical metrics
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        # Probability-based metrics (if available)
        if y_pred_proba is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['auc_pr'] = average_precision_score(y_true, y_pred_proba)
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)

        return metrics

    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[Path] = None
    ) -> None:
        """Plot ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'{model_name} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved ROC curve to {save_path}")

        plt.show()

    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[Path] = None
    ) -> None:
        """Plot Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)

        plt.figure(figsize=figsize)
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'{model_name} (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved PR curve to {save_path}")

        plt.show()

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        normalize: bool = True,
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[Path] = None
    ) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title_suffix = '(Normalized)'
        else:
            fmt = 'd'
            title_suffix = ''

        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=['Non-default', 'Default'],
                   yticklabels=['Non-default', 'Default'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Confusion Matrix - {model_name} {title_suffix}')
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")

        plt.show()

    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "Model",
        n_bins: int = 10,
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[Path] = None
    ) -> None:
        """Plot calibration curve."""
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)

        plt.figure(figsize=figsize)
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=model_name)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Plot - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved calibration curve to {save_path}")

        plt.show()

    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, float]:
        """
        Find optimal threshold for given metric.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')

        Returns:
            Tuple of (optimal_threshold, metric_value)
        """
        thresholds = np.linspace(0, 1, 101)
        scores = []

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_true, y_pred)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            scores.append(score)

        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_score = scores[optimal_idx]

        logger.info(f"Optimal threshold for {metric}: {optimal_threshold:.3f} ({metric}={optimal_score:.4f})")

        return optimal_threshold, optimal_score

    def compare_models(
        self,
        results_dict: Dict[str, Dict[str, float]],
        metrics_to_plot: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            results_dict: Dictionary mapping model names to their metrics
            metrics_to_plot: List of metrics to plot
            figsize: Figure size
            save_path: Optional path to save figure

        Returns:
            DataFrame with comparison results
        """
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results_dict).T

        if metrics_to_plot is None:
            metrics_to_plot = ['auc_roc', 'accuracy', 'precision', 'recall', 'f1_score']

        # Filter to available metrics
        metrics_to_plot = [m for m in metrics_to_plot if m in comparison_df.columns]

        # Plot comparison
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=figsize)

        if len(metrics_to_plot) == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            comparison_df[metric].plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(metric.upper())
            ax.set_ylabel('Score')
            ax.set_xlabel('Model')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
            logger.info(f"Saved model comparison to {save_path}")

        plt.show()

        return comparison_df

    def generate_report(
        self,
        results: Dict[str, any],
        output_path: Union[str, Path]
    ) -> None:
        """
        Generate evaluation report.

        Args:
            results: Dictionary with evaluation results
            output_path: Path to save report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("# Model Evaluation Report\n\n")

            # Write metrics
            f.write("## Metrics\n\n")
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    f.write(f"- **{key}**: {value:.4f}\n")

            f.write("\n")

        logger.info(f"Saved evaluation report to {output_path}")
