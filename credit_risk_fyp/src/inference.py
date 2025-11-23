"""
Production Inference Pipeline for Credit Risk Assessment FYP
End-to-end prediction pipeline with explainability
"""

import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from .config import MODELS_DIR
from .utils import load_pickle
from .preprocessor import DataPreprocessor
from .feature_engineer import FeatureEngineer

# Setup logger
logger = logging.getLogger('credit_risk_fyp.inference')


class CreditRiskPredictor:
    """
    Production-ready credit risk prediction pipeline.

    Features:
    - End-to-end processing (raw data → prediction)
    - Automatic preprocessing and feature engineering
    - Batch prediction support
    - SHAP-based explainability
    - Prediction reports
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        preprocessor_path: Optional[Union[str, Path]] = None,
        feature_engineer_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize CreditRiskPredictor.

        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor (optional)
            feature_engineer_path: Path to saved feature engineer (optional)
        """
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path) if preprocessor_path else None
        self.feature_engineer_path = Path(feature_engineer_path) if feature_engineer_path else None

        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.explainer = None

        # Load all artifacts
        self.load_artifacts()

        logger.info("Credit Risk Predictor initialized")

    def load_artifacts(self) -> None:
        """Load model and preprocessing artifacts."""
        logger.info("Loading model and preprocessing artifacts...")

        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = load_pickle(self.model_path)
        logger.info(f"✓ Loaded model from {self.model_path}")

        # Load preprocessor
        if self.preprocessor_path and self.preprocessor_path.exists():
            self.preprocessor = DataPreprocessor.load(self.preprocessor_path)
            logger.info(f"✓ Loaded preprocessor from {self.preprocessor_path}")
        else:
            logger.warning("No preprocessor provided - input data must be preprocessed")

        # Load feature engineer
        if self.feature_engineer_path and self.feature_engineer_path.exists():
            self.feature_engineer = FeatureEngineer.load(self.feature_engineer_path)
            logger.info(f"✓ Loaded feature engineer from {self.feature_engineer_path}")
        else:
            logger.warning("No feature engineer provided - input data must have features engineered")

    def preprocess(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw data.

        Args:
            raw_data: Raw input data

        Returns:
            Preprocessed data
        """
        data = raw_data.copy()

        # Apply preprocessing if available
        if self.preprocessor:
            data, _ = self.preprocessor.transform(data, is_training=False)
            logger.info("✓ Applied preprocessing")

        # Apply feature engineering if available
        if self.feature_engineer:
            data = self.feature_engineer.transform(data)
            logger.info("✓ Applied feature engineering")

        return data

    def predict(
        self,
        raw_data: pd.DataFrame,
        return_proba: bool = False,
        threshold: float = 0.5
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions on raw data.

        Args:
            raw_data: Raw input data
            return_proba: Whether to return probabilities
            threshold: Classification threshold

        Returns:
            Predictions (and probabilities if return_proba=True)
        """
        logger.info(f"Making predictions for {len(raw_data)} samples...")

        # Preprocess data
        processed_data = self.preprocess(raw_data)

        # Make predictions
        probabilities = self.model.predict_proba(processed_data)

        if return_proba:
            predictions = (probabilities >= threshold).astype(int)
            return predictions, probabilities
        else:
            predictions = (probabilities >= threshold).astype(int)
            return predictions

    def predict_batch(
        self,
        raw_data_list: List[pd.DataFrame],
        return_proba: bool = False,
        threshold: float = 0.5
    ) -> List[np.ndarray]:
        """
        Make predictions on multiple batches.

        Args:
            raw_data_list: List of DataFrames to predict
            return_proba: Whether to return probabilities
            threshold: Classification threshold

        Returns:
            List of predictions
        """
        logger.info(f"Batch prediction for {len(raw_data_list)} batches...")

        results = []
        for i, raw_data in enumerate(raw_data_list):
            logger.info(f"Processing batch {i+1}/{len(raw_data_list)}...")
            result = self.predict(raw_data, return_proba=return_proba, threshold=threshold)
            results.append(result)

        return results

    def explain_prediction(
        self,
        raw_data: Union[pd.DataFrame, pd.Series],
        num_features: int = 10
    ) -> Dict[str, any]:
        """
        Explain prediction using SHAP values.

        Args:
            raw_data: Single instance or DataFrame
            num_features: Number of top features to show

        Returns:
            Dictionary with explanation
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
            return {"error": "SHAP not installed"}

        # Convert to DataFrame if Series
        if isinstance(raw_data, pd.Series):
            raw_data = raw_data.to_frame().T

        if len(raw_data) > 1:
            logger.warning("Multiple instances provided, using first instance")
            raw_data = raw_data.iloc[[0]]

        # Preprocess
        processed_data = self.preprocess(raw_data)

        # Get prediction
        probability = self.model.predict_proba(processed_data)[0]
        prediction = int(probability >= 0.5)

        # Initialize SHAP explainer if not already done
        if self.explainer is None:
            logger.info("Initializing SHAP explainer...")
            try:
                # Use TreeExplainer for tree-based models
                if hasattr(self.model, 'model'):
                    self.explainer = shap.TreeExplainer(self.model.model)
                else:
                    # Use KernelExplainer as fallback
                    background = processed_data.sample(min(100, len(processed_data)))
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        background
                    )
            except Exception as e:
                logger.error(f"Failed to initialize SHAP explainer: {e}")
                return {
                    "prediction": prediction,
                    "probability": float(probability),
                    "error": "Could not initialize SHAP explainer"
                }

        # Calculate SHAP values
        try:
            shap_values = self.explainer.shap_values(processed_data)

            # Get feature names
            feature_names = processed_data.columns.tolist()

            # Get SHAP values for this instance
            if isinstance(shap_values, list):
                # Binary classification - use class 1
                instance_shap = shap_values[1][0]
            else:
                instance_shap = shap_values[0]

            # Get top features
            top_indices = np.argsort(np.abs(instance_shap))[-num_features:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_values = [float(instance_shap[i]) for i in top_indices]
            top_feature_values = [float(processed_data.iloc[0, i]) for i in top_indices]

            explanation = {
                "prediction": prediction,
                "probability": float(probability),
                "top_features": [
                    {
                        "feature": feat,
                        "shap_value": val,
                        "feature_value": feat_val
                    }
                    for feat, val, feat_val in zip(top_features, top_values, top_feature_values)
                ]
            }

            return explanation

        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            return {
                "prediction": prediction,
                "probability": float(probability),
                "error": f"SHAP calculation failed: {str(e)}"
            }

    def generate_report(
        self,
        raw_data: pd.DataFrame,
        output_path: Union[str, Path]
    ) -> None:
        """
        Generate prediction report.

        Args:
            raw_data: Input data
            output_path: Path to save report
        """
        logger.info("Generating prediction report...")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Make predictions
        predictions, probabilities = self.predict(raw_data, return_proba=True)

        # Create report
        report_df = raw_data.copy()
        report_df['prediction'] = predictions
        report_df['default_probability'] = probabilities
        report_df['risk_level'] = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )

        # Save report
        report_df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved prediction report to {output_path}")

        # Print summary
        logger.info("\nPrediction Summary:")
        logger.info(f"  Total samples: {len(report_df)}")
        logger.info(f"  Predicted defaults: {predictions.sum()} ({predictions.mean()*100:.1f}%)")
        logger.info(f"  Average default probability: {probabilities.mean():.3f}")
        logger.info(f"\nRisk Distribution:")
        for level in ['Low', 'Medium', 'High']:
            count = (report_df['risk_level'] == level).sum()
            pct = (count / len(report_df)) * 100
            logger.info(f"  {level}: {count} ({pct:.1f}%)")


# Convenience function
def predict_credit_risk(
    model_path: Union[str, Path],
    data: pd.DataFrame,
    preprocessor_path: Optional[Union[str, Path]] = None,
    feature_engineer_path: Optional[Union[str, Path]] = None,
    return_proba: bool = False
) -> np.ndarray:
    """
    Quick function to make credit risk predictions.

    Args:
        model_path: Path to saved model
        data: Input data
        preprocessor_path: Path to saved preprocessor
        feature_engineer_path: Path to saved feature engineer
        return_proba: Whether to return probabilities

    Returns:
        Predictions or probabilities
    """
    predictor = CreditRiskPredictor(model_path, preprocessor_path, feature_engineer_path)
    return predictor.predict(data, return_proba=return_proba)
