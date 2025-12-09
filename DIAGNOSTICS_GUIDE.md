# Comprehensive Diagnostics Guide

## Overview

A complete diagnostics section has been added to [notebooks/xgboost_improved.ipynb](notebooks/xgboost_improved.ipynb) (Section 16) to help you understand why precision dropped and what caused the changes in model performance.

## 12 Diagnostic Tests Implemented

### Test 1: Optimized Threshold Value
**Purpose**: Determine if the precision drop was caused by an overly aggressive threshold.

**What it checks**:
- The exact F1-optimized threshold value
- Whether threshold is below 0.40 (very aggressive)
- Comparison to baseline threshold (0.50)

**Key insight**: If threshold < 0.40, the precision drop is primarily due to threshold optimization, not SMOTE.

---

### Test 2: Confusion Matrix Comparison
**Purpose**: See exactly where predictions changed between baseline and improved models.

**What it checks**:
- True Positives, True Negatives, False Positives, False Negatives
- Absolute changes in each category
- Ratio of False Positive increase to True Positive increase

**Key insight**: Shows how many additional good customers were rejected (FP) vs. how many additional defaults were caught (TP).

---

### Test 3: SMOTE Resampling Statistics
**Purpose**: Verify SMOTE was correctly applied and achieved target ratio.

**What it checks**:
- Class distribution before and after SMOTE
- Whether 60:40 ratio was achieved
- Number of synthetic samples generated
- Validation/test sets remain unchanged

**Key insight**: Confirms SMOTE was applied only to training data and not to validation/test sets.

---

### Test 4: Prediction Probability Distribution
**Purpose**: Understand how the model's confidence levels are distributed.

**What it checks**:
- Statistical summary of predicted probabilities
- How many predictions fall in boundary region (0.3-0.5)
- Distribution by actual class
- Visual histogram plots

**Key insight**: If many predictions cluster near the threshold, small threshold changes will have large effects on metrics.

---

### Test 5: Precision at Different Recall Levels
**Purpose**: Determine if better precision is possible while maintaining current recall.

**What it checks**:
- Precision values at 50%, 60%, 65.7%, 70%, 80% recall
- Maximum achievable precision at current recall level
- Precision-recall trade-off curve

**Key insight**: Shows whether threshold adjustment can improve precision or if you're already at the optimal point.

---

### Test 6: Business Impact Analysis
**Purpose**: Translate metrics into business costs and determine if the trade-off is worthwhile.

**What it checks**:
- Absolute counts of FP and FN changes
- Cost of rejecting good customers (FP)
- Cost of approving bad loans (FN)
- Total cost change (savings or loss)

**Assumptions** (adjust in code):
- Average loan amount: $15,000
- Cost of default: 60% of loan amount
- Cost of rejection: $200 opportunity cost

**Key insight**: Determines if improved recall is worth the increased false positives from a business perspective.

---

### Test 7: SMOTE Synthetic Sample Quality
**Purpose**: Ensure synthetic samples are realistic and don't contain impossible values.

**What it checks**:
- Feature value ranges before and after SMOTE
- Impossible values (negative ages, FICO > 850, etc.)
- Statistical comparison of original vs. synthetic samples

**Key insight**: Poor quality synthetic samples can hurt model performance rather than help it.

---

### Test 8: Model Performance at Baseline Threshold (0.5)
**Purpose**: Isolate the impact of SMOTE from threshold optimization.

**What it checks**:
- SMOTE model performance at threshold=0.5
- Comparison to baseline model at threshold=0.5
- Impact of threshold optimization separately

**Key insight**: **This is the most important test for understanding what caused the precision drop**. It shows:
- Did SMOTE alone improve or hurt metrics?
- How much did threshold optimization contribute?

---

### Test 9: Data Leakage Check
**Purpose**: Verify no information leaked from validation/test into training.

**What it checks**:
- SMOTE applied only to training data
- Train-val-test split done before SMOTE
- Class distributions are consistent
- No overlap between dataset splits

**Key insight**: Critical for ensuring results are valid. Any leakage invalidates all metrics.

---

### Test 10: Feature Importance Comparison
**Purpose**: See if the 12 new risk indicator features are actually being used.

**What it checks**:
- Top 20 most important features
- How many risk indicators appear in top 20
- Feature importance distribution
- Visual plot with risk indicators highlighted in red

**Key insight**: If risk indicators don't appear in top features, they may not be adding value to the model.

---

### Test 11: Edge Case Analysis
**Purpose**: Understand model behavior on difficult-to-classify cases.

**What it checks**:
- Predictions in borderline region (0.4-0.6 probability)
- Accuracy on borderline cases
- Actual vs. predicted distribution
- Confusion matrix for edge cases only

**Key insight**: Shows if the model is making good decisions on ambiguous cases or just guessing.

---

### Test 12: Final Class Distribution Verification
**Purpose**: Final comprehensive check that all data splits are correct.

**What it checks**:
- Training set before and after SMOTE
- Validation set (should be unchanged)
- Test set (should be unchanged)
- All verification checks pass

**Key insight**: Final validation that your entire pipeline is correctly implemented. Should show "ALL CHECKS PASSED".

---

## How to Use the Diagnostics

### Step 1: Run the Notebook
```bash
# In Jupyter
# Kernel → Restart & Clear Output
# Cell → Run All
```

### Step 2: Review Section 16 (Diagnostics)

The diagnostics section will run automatically and produce:
- Text output for each of the 12 tests
- 4 diagnostic plots saved to `credit_risk_fyp/results/figures/`

### Step 3: Interpret the Key Results

#### Question 1: What caused the precision drop?

**Check Test 1** (Threshold Value):
- If `optimal_threshold < 0.40`: Precision drop is primarily from aggressive threshold
- If `optimal_threshold > 0.40`: SMOTE likely contributed to precision change

**Check Test 8** (Performance at 0.5):
```
BASELINE MODEL AT THRESHOLD=0.5:
  Precision: 0.3425

IMPROVED MODEL AT THRESHOLD=0.5:
  Precision: [value]
```

- If improved precision ≈ baseline precision: **SMOTE didn't help precision**
- If improved precision < baseline precision: **SMOTE hurt precision**
- If improved precision > baseline precision: **SMOTE helped, but threshold optimization lowered it**

#### Question 2: Is the trade-off worthwhile?

**Check Test 6** (Business Impact):
```
TOTAL COST:
  Net change: $[value]
```

- Negative value = cost savings = **good trade-off**
- Positive value = cost increase = **bad trade-off** (unless strategic reasons exist)

**Check Test 2** (Confusion Matrix):
```
False Positive / True Positive increase ratio: [value]
```

- Ratio < 2: For every default caught, reject <2 good customers = **acceptable**
- Ratio > 5: For every default caught, reject >5 good customers = **questionable**

#### Question 3: Is the pipeline correct?

**Check Test 12** (Final Verification):

Should show:
```
✓✓✓ ALL CHECKS PASSED ✓✓✓
```

If you see warnings, review them carefully before trusting any results.

---

## Common Diagnostic Scenarios

### Scenario 1: Low Threshold Caused Precision Drop

**Symptoms**:
- Test 1: Threshold < 0.40
- Test 8: SMOTE model at 0.5 has decent precision

**Solution**:
```python
# Try a higher threshold
new_threshold = 0.45  # Or 0.50

# Re-evaluate with new threshold
y_test_pred_new = (y_test_proba >= new_threshold).astype(int)
new_metrics = evaluator.calculate_metrics(y_test, y_test_pred_new, y_test_proba)
print(new_metrics)
```

---

### Scenario 2: SMOTE Hurt Precision

**Symptoms**:
- Test 8: SMOTE@0.5 precision < Baseline@0.5 precision
- Test 7: Synthetic samples look reasonable

**Solutions**:
1. **Try different SMOTE ratio**:
   ```python
   # In cell where SMOTE is applied
   resampler = DataResampler(
       strategy='smote',
       sampling_ratio=0.5,  # Changed from 0.6
       k_neighbors=5,
       random_state=42
   )
   ```

2. **Try different SMOTE variant**:
   ```python
   resampler = DataResampler(
       strategy='borderline',  # Or 'adasyn'
       sampling_ratio=0.6,
       k_neighbors=5,
       random_state=42
   )
   ```

3. **Use class weights instead**:
   ```python
   # In training cell
   params['scale_pos_weight'] = 4.0  # Adjust based on imbalance
   # Don't apply SMOTE
   xgb_model.train(X_train_final, y_train, X_val_final, y_val)
   ```

---

### Scenario 3: Risk Features Not Important

**Symptoms**:
- Test 10: No risk indicators in top 20 features
- Metrics didn't improve much

**Solutions**:
1. **Review feature engineering logic** in `src/feature_engineer.py:439`
2. **Check for feature scaling issues** - risk indicators might need different scaling
3. **Try feature selection** - remove low-importance features
4. **Consider domain expert review** - are the risk formulas correct?

---

### Scenario 4: Data Leakage Detected

**Symptoms**:
- Test 9 or Test 12: Warnings about data leakage
- Test 3: Validation/test sets show balanced classes

**Solutions**:
1. **Stop immediately** - current results are invalid
2. **Review pipeline** - ensure train/val/test split happens before SMOTE
3. **Re-run notebook from scratch** after fixing the pipeline

---

### Scenario 5: Business Cost Increased

**Symptoms**:
- Test 6: Positive net cost change
- Test 2: Many more false positives

**Solutions**:
1. **Adjust cost assumptions** in Test 6 cell if they're incorrect
2. **Increase threshold** to reduce false positives
3. **Consider strategic value** - is catching more defaults worth the cost?
4. **Try ensemble methods** (future work) for better precision-recall balance

---

## Diagnostic Plots Generated

All plots are saved to `credit_risk_fyp/results/figures/`:

1. **probability_distribution_diagnostic.png**
   - Shows distribution of predicted probabilities
   - Separate histograms by actual class
   - Threshold markers

2. **feature_importance_diagnostic.png**
   - Top 30 features by importance
   - Risk indicators highlighted in red
   - Shows which features the model relies on

3. **edge_case_analysis_diagnostic.png**
   - Borderline predictions (0.4-0.6 probability)
   - Confusion matrix for edge cases
   - Performance on difficult cases

4. **Standard evaluation plots** (from Section 13):
   - ROC curves
   - Precision-Recall curves
   - Confusion matrices
   - Calibration curves
   - Threshold analysis

---

## Recommended Workflow

1. **Run diagnostics** (Section 16)
2. **Review Test 1, 8, 12** first (threshold, SMOTE impact, data quality)
3. **Check Test 6** for business impact
4. **Based on findings**, implement one of the solutions above
5. **Re-run from the solution cell onwards**
6. **Re-check diagnostics** to verify improvement
7. **Iterate** until metrics meet targets

---

## Expected Diagnostic Output

When everything is working correctly, you should see:

- **Test 1**: Threshold in range 0.40-0.55
- **Test 3**: ~60:40 class ratio achieved in training
- **Test 6**: Negative net cost (savings)
- **Test 8**: SMOTE@0.5 shows improvement over baseline@0.5
- **Test 10**: At least 2-3 risk indicators in top 20
- **Test 12**: "ALL CHECKS PASSED"

---

## Troubleshooting

### Import Errors
```python
# If confusion_matrix import fails in Test 11
from sklearn.metrics import confusion_matrix
```

### Variable Not Found
- Make sure you ran all cells in order from the beginning
- Use "Kernel → Restart & Run All" to ensure clean execution

### Plots Not Displaying
```python
import matplotlib.pyplot as plt
plt.show()  # Add this after plot generation
```

---

## Next Steps After Diagnostics

Based on your diagnostic results:

### If precision needs improvement:
1. Increase threshold (0.45-0.50)
2. Try different SMOTE ratio (0.5)
3. Use class weights instead of SMOTE
4. Implement ensemble methods (Phase 4-5 from roadmap)

### If recall needs improvement:
1. Decrease threshold (0.35-0.40)
2. Increase SMOTE ratio (0.65-0.70)
3. Add more risk-focused features
4. Try cost-sensitive learning

### If both need improvement:
1. Try different SMOTE variants (BorderlineSMOTE, ADASYN)
2. Implement hyperparameter tuning (Phase 3 from roadmap)
3. Build ensemble models (Phase 4-5 from roadmap)
4. Collect more training data

---

## References

- Test 1-3: Data quality and pipeline validation
- Test 4-5: Model behavior analysis
- Test 6: Business value assessment
- Test 7-9: Technical correctness validation
- Test 10-11: Feature and prediction analysis
- Test 12: Final comprehensive verification

---

**Last Updated**: 2025-11-30

**Status**: ✅ All 12 diagnostic tests implemented and ready to run
