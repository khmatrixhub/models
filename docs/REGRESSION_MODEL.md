# Regression Model for Profit Prediction

## Overview

The `regression.py` script trains a **Random Forest Regressor** (or Gradient Boosting Regressor) to directly predict profit values instead of binary win/loss classification. This approach offers more flexibility and granular control over trade selection.

## Key Differences from Classification (`train.py`)

| Aspect | Classification (train.py) | Regression (regression.py) |
|--------|---------------------------|----------------------------|
| **Target** | Binary (1=win, 0=loss) | Continuous profit value |
| **Model** | RandomForestClassifier | RandomForestRegressor |
| **Output** | Win probability (0-1) | Expected profit (price units) |
| **Threshold** | Probability cutoff | Minimum profit cutoff |
| **Flexibility** | Less granular | More granular control |

## Advantages of Regression Approach

1. **Direct Profit Prediction**: Model learns to predict actual profit amounts, not just win/loss
2. **Better Trade Selection**: Can filter by expected profit magnitude, not just probability
3. **No Threshold Artifacts**: No need to reclassify labels with `min_profit_threshold`
4. **Magnitude Awareness**: Model learns that 1% profit is better than 0.1% profit
5. **Natural Ranking**: Trades automatically ranked by expected profit

## Model Types

Configure in `training` section of YAML:

```yaml
training:
  model_type: random_forest  # Default
  # OR
  model_type: gradient_boosting  # Alternative
```

### Random Forest Regressor (Default)
- Ensemble of decision trees
- Good for complex non-linear relationships
- Fast training and prediction
- Handles outliers well

### Gradient Boosting Regressor
- Sequential tree building
- Often higher accuracy
- Slower training
- More prone to overfitting

## Usage

### Basic Command

```bash
python regression.py --config configs/config_example.yaml \
                     --experiment profit_prediction \
                     --label-name conservative
```

### With Signal Reversal

```bash
python regression.py --config configs/config_example.yaml \
                     --experiment profit_prediction_reversed \
                     --label-name conservative
```

Add to config:
```yaml
training:
  reverse_signals: true  # Test inverse predictive power
```

## Configuration

### Complete Training Config

```yaml
training:
  n_estimators: 100              # Number of trees
  use_rfe: false                 # Recursive feature elimination
  max_features: 30               # Max features to select
  max_features_per_family: 3     # Max per feature family
  reverse_signals: false         # Reverse long/short
  model_type: random_forest      # random_forest or gradient_boosting
  
  bagging:
    enabled: false               # Use BaggingRegressor
    n_estimators: 200           # Number of bagging estimators
```

## Output Interpretation

### Predictions File

```python
# predicted_results.parquet
predicted_profit    actual_profit    trade_date    prediction_error
0.003245           0.004123         2025-01-02    0.000878
0.001234           0.000987         2025-01-02    -0.000247
-0.000543          -0.001234        2025-01-03    -0.000691
```

### Evaluation Metrics

**Regression Metrics:**
- **MSE (Mean Squared Error)**: Average squared prediction error
- **RMSE (Root MSE)**: Square root of MSE (same units as profit)
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **R² (R-squared)**: Proportion of variance explained (0-1, higher better)

**Trading Metrics (by threshold):**
- **Threshold**: Minimum predicted profit to take trade
- **Count**: Number of trades taken
- **Total Profit**: Sum of actual profits on selected trades
- **Mean Profit**: Average actual profit per trade
- **Win Rate**: Percentage of profitable trades

## Profit Thresholds

The model evaluates multiple thresholds for trade selection:

```python
thresholds = [0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
```

For each threshold, only trades with `predicted_profit >= threshold` are taken.

### Understanding Thresholds (USDJPY Example)

Remember: **Profits are in price units, NOT percentages**

For USDJPY at ~150:
- `0.0001` = 0.01 yen = ~0.00007% (0.7 pips)
- `0.001` = 0.1 yen = ~0.00067% (6.7 pips)
- `0.01` = 1 yen = ~0.0067% (67 pips)
- `0.1` = 10 yen = ~0.067% (667 pips)

### Recommended Thresholds

**Conservative (High Volume):**
```yaml
# Take all trades with predicted profit > 0
threshold: 0.0
```

**Moderate (Balanced):**
```yaml
# Take trades predicted to profit > 0.1 yen
threshold: 0.001  # ~7 pips for USDJPY
```

**Aggressive (High Quality):**
```yaml
# Take trades predicted to profit > 1 yen
threshold: 0.01  # ~67 pips for USDJPY
```

## Workflow Example

### 1. Train Regression Model

```bash
python regression.py \
  --config configs/config_example.yaml \
  --experiment baseline_regression \
  --label-name conservative \
  --retrain-frequency 5 \
  --min-train-days 30
```

### 2. Review Results

```bash
# Check evaluation results
cat results/config_example/models/baseline_regression/results/evaluation_results.csv

# View predictions
python -c "
import pandas as pd
df = pd.read_parquet('results/config_example/models/baseline_regression/results/predicted_results.parquet')
print(df.head(20))
print(f'\nCorrelation: {df[['predicted_profit', 'actual_profit']].corr()}')
"
```

### 3. Compare with Classification

```bash
# Train classification model
python train.py --config configs/config_example.yaml \
                --experiment baseline_classification \
                --label-name conservative

# Compare results
# Regression: Direct profit prediction, natural thresholding
# Classification: Win probability, requires threshold tuning
```

## Expected Performance

### Good Regression Model
- **R² > 0.3**: Model explains 30%+ of profit variance
- **MAE < 0.001**: Average error < 0.1 yen (for USDJPY)
- **Total Profit**: Higher at optimal threshold vs classification

### Poor Regression Model
- **R² < 0.1**: Model barely better than random
- **Predictions centered at 0**: Not learning profit magnitude
- **No threshold beats baseline**: Classification may be better

## Advantages Over `min_profit_threshold`

### Classification with `min_profit_threshold`
```yaml
training:
  min_profit_threshold: 0.01  # Reclassify labels
```
- ❌ Throws away information (0.005 profit treated as loss)
- ❌ Binary decision boundary
- ❌ Model doesn't learn profit magnitude

### Regression Approach
```yaml
# No label reclassification needed
```
- ✅ Model learns profit magnitude directly
- ✅ Natural ranking by expected profit
- ✅ Can apply threshold after training
- ✅ More information used during training

## Feature Importance

Regression models provide feature importance showing which features best predict profit magnitude:

```
Top Features for Profit Prediction:
1. conservative_profit_lag1       0.082341
2. returns_adj                    0.071234
3. conservative_profit_rolling... 0.065432
4. atr                           0.054321
5. ma_distance                   0.043210
```

Compare with classification to see:
- Classification: Features that separate win/loss
- Regression: Features that predict profit magnitude

## File Organization

```
results/<config_name>/
├── models/
│   └── <experiment>/
│       ├── results/
│       │   ├── evaluation_results.csv      # Threshold performance
│       │   ├── predicted_results.parquet   # All predictions
│       │   ├── feature_importance.csv      # Feature rankings
│       │   └── experiment_metadata.json    # Config snapshot
│       └── model_info_*.json               # Per-date model info
└── logs/
    └── regression_<experiment>_*.log       # Training logs
```

## Troubleshooting

### Low R² Score
**Problem**: Model doesn't predict profit well
**Solutions**:
- Increase `n_estimators` (more trees)
- Try `model_type: gradient_boosting`
- Check feature quality (are lagged profits included?)
- Verify profit units are correct (not all zeros)

### All Predictions Near Zero
**Problem**: Model predicts ~0 profit for everything
**Solutions**:
- Check target distribution (is actual profit too small?)
- Ensure lagged profit features are present
- Try signal reversal if baseline is negative
- Increase model complexity

### Worse Than Classification
**Problem**: Regression total profit < classification total profit
**Solutions**:
- Regression needs more samples to learn magnitude
- Try larger `min_train_days` (60-90 days)
- Adjust thresholds (regression natural threshold vs classification probability)
- Check if profit variance is too high for regression

## Best Practices

1. **Start Simple**: Use default Random Forest before trying Gradient Boosting
2. **Compare**: Run both regression and classification on same data
3. **Tune Thresholds**: Find optimal threshold for your risk tolerance
4. **Monitor R²**: If R² < 0.2, investigate feature quality
5. **Use Lagged Profits**: Ensure lagged profit features are included
6. **Validate**: Check prediction vs actual profit correlation

## Next Steps

After training regression model:

1. **Analyze predictions**:
   ```python
   df = pd.read_parquet('results/.../predicted_results.parquet')
   df.plot.scatter(x='predicted_profit', y='actual_profit')
   ```

2. **Find optimal threshold**:
   ```python
   results = pd.read_csv('results/.../evaluation_results.csv')
   best = results.loc[results['total_profit'].idxmax()]
   print(f"Use threshold: {best['threshold']}")
   ```

3. **Compare with classification**:
   - Total profit at best threshold
   - Trade count vs win rate
   - Consistency of performance

## Summary

**Use Regression When:**
- You want to predict profit magnitude, not just win/loss
- You want natural trade ranking by expected profit
- You want to avoid arbitrary threshold-based label reclassification
- Your profit distribution has meaningful variance

**Use Classification When:**
- Win/loss boundary is clear and meaningful
- Profit magnitude is less important than win rate
- You have limited training data
- You want interpretable probabilities

Both approaches are valid - try both and compare results!
