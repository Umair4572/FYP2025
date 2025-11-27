# Dataset Setup Summary

## ✓ Datasets Successfully Integrated

Your credit risk datasets have been successfully integrated into the FYP project!

### Dataset Locations

**Training Data:**
- Path: `credit_risk_fyp/data/raw/lending_club_train.csv`
- Size: 211.35 MB
- Columns: 103
- Rows: ~500,000+ (estimated)
- Target Distribution: ~81% no default (0), ~19% default (1)

**Test Data:**
- Path: `credit_risk_fyp/data/raw/lending_club_test.csv`
- Size: 63.58 MB
- Columns: 103
- Rows: 150,000
- Target: Not labeled (all NULL) - as expected for test predictions

### Configuration Updates

The following configuration has been updated in `credit_risk_fyp/src/config.py`:

1. **Dataset Names:**
   - `train_dataset`: 'lending_club_train.csv'
   - `test_dataset`: 'lending_club_test.csv'

2. **Target Column:**
   - Changed from `loan_status` to `default`
   - Binary encoded: 0 = No default (good), 1 = Default (bad)

3. **ID Column:**
   - `id` column for unique loan identification

### Key Dataset Features (103 columns)

**Target Variable:**
- `default` (0/1 binary)

**Loan Information:**
- loan_amnt, term, pymnt_plan, purpose, etc.

**Borrower Information:**
- annual_inc, emp_length, home_ownership, dti, etc.

**Credit History:**
- fico_range_low, fico_range_high, earliest_cr_line
- delinq_2yrs, inq_last_6mths, pub_rec
- open_acc, total_acc, revol_bal, revol_util

**Advanced Credit Features:**
- 70+ advanced credit bureau features
- Joint application features (for co-borrowers)
- Recent account activity metrics

### How to Load Data

You can now load your data using the project's DataLoader:

```python
from src.data_loader import load_data
from src.config import RAW_DATA_DIR, DATASET_CONFIG

# Load training data
train_df = load_data(
    RAW_DATA_DIR / DATASET_CONFIG['train_dataset'],
    optimize=True  # Optimizes memory usage
)

# Load test data
test_df = load_data(
    RAW_DATA_DIR / DATASET_CONFIG['test_dataset'],
    optimize=True
)
```

Or using pandas directly:

```python
import pandas as pd

train_df = pd.read_csv('credit_risk_fyp/data/raw/lending_club_train.csv')
test_df = pd.read_csv('credit_risk_fyp/data/raw/lending_club_test.csv')
```

### Next Steps

Now that your datasets are configured, you can:

1. **Explore the Data:**
   - Create a notebook for exploratory data analysis
   - Understand feature distributions
   - Analyze missing values and correlations

2. **Preprocess the Data:**
   - Handle missing values
   - Encode categorical features
   - Scale numerical features

3. **Train Models:**
   - Use the configured XGBoost, LightGBM, CatBoost models
   - Train ensemble models
   - Evaluate performance

4. **Make Predictions:**
   - Use trained models to predict on the test set
   - Generate submission files if needed

### Important Notes

- ⚠️ The test dataset has NO target labels (all NULL in 'default' column)
- This is expected - you'll predict labels for the test set after training
- Training data is imbalanced: ~81% non-default, ~19% default
- Consider using class weighting or SMOTE for handling imbalance
- The project is already configured with `scale_pos_weight=5` in model configs

---

**Status:** ✅ Ready for modeling!
