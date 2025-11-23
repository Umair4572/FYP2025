# Credit Risk Assessment Using Ensemble Learning

## 📊 Project Overview

This Final Year Project (FYP) implements a comprehensive credit risk assessment system using ensemble learning methods on Lending Club loan data (2007-2020). The system predicts the likelihood of loan default using state-of-the-art machine learning techniques with GPU acceleration for optimal performance.

## 🎯 Key Features

- **5 Base Models:** XGBoost, LightGBM, CatBoost, Random Forest, Neural Network (TensorFlow)
- **2 Ensemble Methods:** Stacking Ensemble and Weighted Averaging
- **GPU-Optimized:** All models configured for CUDA/GPU acceleration
- **Production-Ready:** Complete inference pipeline with explainability
- **Comprehensive Evaluation:** Multiple metrics, visualizations, and SHAP analysis
- **Well-Documented:** Detailed documentation with code comments and examples

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended for optimal performance)
- 16GB+ RAM recommended
- CUDA 11.8+ and cuDNN 8.6+ (for GPU acceleration)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd credit_risk_fyp
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download datasets:**
   - Download the Lending Club dataset from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
   - Place the CSV files in `data/raw/` directory

### Training Models

**Train all models at once:**
```bash
python scripts/train_all_models.py --data-path data/raw/lending_club_2007_2020.csv --models all --ensemble
```

**Train specific models:**
```bash
# Train only XGBoost
python scripts/train_all_models.py --data-path data/raw/lending_club_2007_2020.csv --models xgboost

# Train multiple specific models
python scripts/train_all_models.py --data-path data/raw/lending_club_2007_2020.csv --models xgboost lightgbm catboost
```

### Making Predictions

**Using the inference pipeline:**
```python
from src.inference import CreditRiskPredictor
import pandas as pd

# Load the predictor with the best model
predictor = CreditRiskPredictor(
    model_path='models/stacking_ensemble.pkl',
    preprocessor_path='models/preprocessor.pkl',
    feature_engineer_path='models/feature_engineer.pkl'
)

# Load your data
new_data = pd.read_csv('new_loans.csv')

# Make predictions
predictions = predictor.predict(new_data, return_proba=True)

# Get explanations
explanation = predictor.explain_prediction(new_data.iloc[0], num_features=10)
```

### Evaluation

**Evaluate all models:**
```bash
python scripts/evaluate_models.py --models-dir models/ --test-data data/splits/test.csv
```

## 📁 Project Structure

```
credit_risk_fyp/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned datasets
│   └── splits/                 # Train/val/test splits
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_base_models.ipynb
│   ├── 05_ensemble_models.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_inference_pipeline.ipynb
├── src/
│   ├── config.py              # Configuration and hyperparameters
│   ├── data_loader.py         # Optimized data loading
│   ├── preprocessor.py        # Data cleaning pipeline
│   ├── feature_engineer.py    # Feature engineering
│   ├── evaluation.py          # Evaluation metrics
│   ├── utils.py               # Helper functions
│   ├── inference.py           # Production inference
│   └── models/
│       ├── xgboost_model.py
│       ├── lightgbm_model.py
│       ├── catboost_model.py
│       ├── random_forest_model.py
│       ├── neural_network.py
│       ├── stacking_ensemble.py
│       └── weighted_ensemble.py
├── models/                     # Saved model files
├── results/
│   ├── figures/               # Plots and visualizations
│   ├── reports/               # Evaluation reports
│   └── logs/                  # Training logs
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── scripts/                    # Training and evaluation scripts
├── requirements.txt
├── setup.py
└── README.md
```

## 🔧 Configuration

All hyperparameters and settings can be configured in `src/config.py`:

- **GPU Settings:** Enable/disable GPU, memory growth, mixed precision
- **Model Hyperparameters:** Adjust learning rates, tree depths, etc.
- **Data Processing:** Missing value thresholds, outlier handling
- **Feature Engineering:** Which features to create
- **Evaluation Metrics:** Which metrics to calculate and plot

## 📊 Model Performance

Performance on Lending Club test set:

| Model | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|-------|---------|----------|-----------|--------|----------|
| XGBoost | TBD | TBD | TBD | TBD | TBD |
| LightGBM | TBD | TBD | TBD | TBD | TBD |
| CatBoost | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD |
| Neural Network | TBD | TBD | TBD | TBD | TBD |
| **Stacking Ensemble** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |
| Weighted Ensemble | TBD | TBD | TBD | TBD | TBD |

*Performance metrics will be updated after training on the full dataset.*

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessor.py -v
```

## 📚 Documentation

- **[Project Report](docs/PROJECT_REPORT.md):** Complete FYP documentation with methodology, results, and analysis
- **[User Guide](docs/USER_GUIDE.md):** Detailed usage instructions and tutorials
- **[API Documentation](docs/API_DOCUMENTATION.md):** Technical API reference

## 🔍 Key Components

### Data Preprocessing
- Handling missing values (median/mode imputation)
- Outlier detection and treatment (IQR method)
- Categorical encoding (label encoding)
- Feature scaling (StandardScaler)
- Data leakage prevention

### Feature Engineering
- Financial ratios (loan-to-income, installment-to-income)
- Credit behavior indicators (delinquency score, account diversity)
- Interaction features (int_rate × dti, fico × dti)
- Time-based features (credit age, loan season)
- Discretized features (income buckets, FICO ranges)

### Models

**Base Models:**
1. **XGBoost:** Gradient boosting with GPU histogram optimization
2. **LightGBM:** Fast gradient boosting with GPU support
3. **CatBoost:** Handles categorical features natively, GPU-accelerated
4. **Random Forest:** Ensemble of decision trees with parallel processing
5. **Neural Network:** Deep learning model with TensorFlow and mixed precision

**Ensemble Methods:**
1. **Stacking:** Uses cross-validation to train meta-model on base model predictions
2. **Weighted Averaging:** Optimizes weights to minimize validation error

### Evaluation
- Classification metrics (AUC, accuracy, precision, recall, F1)
- ROC and Precision-Recall curves
- Confusion matrices
- Feature importance analysis
- SHAP values for model interpretability
- Threshold optimization

## 🎓 Academic Context

This project was developed as a Final Year Project (FYP) for [Your University Name], demonstrating:
- Advanced machine learning techniques
- Ensemble learning methods
- Production-ready software engineering
- Performance optimization
- Comprehensive evaluation and analysis

## 📝 Citation

If you use this code for academic purposes, please cite:

```bibtex
@mastersthesis{your_name_2024,
  title={Credit Risk Assessment Using Ensemble Learning on Lending Club Loan Data},
  author={Your Name},
  year={2024},
  school={Your University}
}
```

## 👥 Author

**Your Name**
Final Year Project - [Your University]
Email: your.email@example.com
LinkedIn: [Your LinkedIn Profile]

## 📄 License

This project is for academic use only. Not for commercial distribution.

## 🙏 Acknowledgments

- Lending Club for providing the dataset
- Kaggle for hosting the data
- TensorFlow, XGBoost, LightGBM, and CatBoost development teams
- Academic supervisors and mentors

## 🐛 Known Issues and Limitations

- GPU acceleration requires NVIDIA GPU with CUDA support
- Large dataset may require significant RAM (16GB+ recommended)
- Some models may take considerable time to train on CPU

## 🔮 Future Work

- Incorporate additional datasets for cross-validation
- Implement deep learning architectures (LSTM, Transformers)
- Add real-time prediction API
- Deploy as web application
- Incorporate economic indicators as features

## 📞 Support

For questions or issues:
1. Check the [User Guide](docs/USER_GUIDE.md)
2. Review the [API Documentation](docs/API_DOCUMENTATION.md)
3. Open an issue on GitHub
4. Contact the author via email

---

**Built with ❤️ for advancing credit risk assessment through machine learning**
