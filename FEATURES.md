# Feature Engineering Documentation

## Overview

The feature generation system combines two types of features:
1. **Technical Indicators** (11 features) - Calculated from OHLC data
2. **Bar Statistics** (16 features) - Pre-computed from tick-level data

Total: **27 features** per timestamp

---

## Technical Indicators (11 features)

These are calculated in `calculate_technical_indicators()` from OHLC bar data.

### Moving Averages (3)
| Feature | Description | Calculation |
|---------|-------------|-------------|
| `ma_5` | 5-period simple moving average | `close.rolling(5).mean()` |
| `ma_20` | 20-period simple moving average | `close.rolling(20).mean()` |
| `ma_50` | 50-period simple moving average | `close.rolling(50).mean()` |

**Use**: Trend identification, support/resistance levels

### Returns (2)
| Feature | Description | Calculation |
|---------|-------------|-------------|
| `returns` | Simple returns | `close.pct_change()` |
| `log_returns` | Log returns | `np.log(close / close.shift(1))` |

**Use**: Momentum, mean reversion signals

### Price Patterns (2)
| Feature | Description | Calculation |
|---------|-------------|-------------|
| `price_range` | High-low range | `(high - low) / close` |
| `body_size` | Candle body size | `abs(close - open) / close` |

**Use**: Volatility proxy, candlestick pattern strength

### Volatility (2)
| Feature | Description | Calculation |
|---------|-------------|-------------|
| `volatility_5` | 5-period returns std | `returns.rolling(5).std()` |
| `volatility_20` | 20-period returns std | `returns.rolling(20).std()` |

**Use**: Risk measurement, volatility regime detection

### Momentum (1)
| Feature | Description | Calculation |
|---------|-------------|-------------|
| `rsi` | 14-period RSI | Relative Strength Index |

**Use**: Overbought/oversold conditions

### Bollinger Bands (1)
| Feature | Description | Calculation |
|---------|-------------|-------------|
| `bb_position` | Position within bands | `(close - bb_lower) / (bb_upper - bb_lower)` |

**Use**: Mean reversion signals, volatility breakouts

---

## Bar Statistics (16 features)

These are **pre-computed** from tick-level data and included in the bar files. They capture microstructure properties not visible in OHLC data alone.

### Volume Metrics (3)
| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `afmlvol` | AFML volume measure | Advanced volume metric |
| `dvol` | Dollar volume | `price × volume` |
| `dvol_sign` | Signed dollar volume | Direction of volume flow |

**Use**: Liquidity, order flow analysis

### Volatility Distribution (7)
| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `vol_skew` | Volatility skewness | Asymmetry of return distribution |
| `vol_kurt` | Volatility kurtosis | Tail heaviness |
| `vol_of_vol` | Volatility of volatility | Second-order volatility |
| `vol_z` | Volatility z-score | Standardized volatility level |
| `vol_ratio` | Volatility ratio | Current vs historical volatility |
| `vol_slope` | Volatility slope | Trend in volatility |
| `vol_autocorr1` | Volatility autocorrelation | Persistence of volatility shocks |

**Use**: 
- **Skew**: Detect asymmetric risks
- **Kurtosis**: Identify fat-tail events
- **Vol-of-vol**: Volatility regime changes
- **Z-score**: Abnormal volatility detection
- **Ratio**: Volatility breakouts
- **Slope**: Volatility trending
- **Autocorr**: Volatility clustering (GARCH effects)

### Realized Measures (3)
| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `rv` | Realized volatility | High-frequency volatility estimate |
| `bpv` | Bipower variation | Jump-robust volatility |
| `rq` | Realized quarticity | Fourth moment of returns |

**Use**:
- **RV**: More accurate volatility than rolling std
- **BPV**: Detect continuous vs jump volatility
- **RQ**: Measure tail risk

### Jump & Tail Detection (2)
| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `jump_proxy` | Jump detection metric | `rv - bpv` (jump component) |
| `tail_exceed_rate` | Tail exceedance rate | Frequency of extreme moves |

**Use**: 
- **Jump proxy**: Identify discontinuous price moves
- **Tail exceed**: Risk of extreme events

### Timing (1)
| Feature | Description | Interpretation |
|---------|-------------|----------------|
| `seconds_diff` | Time between bars | Bar arrival frequency |

**Use**: Detect irregular trading activity, gaps

---

## Feature Selection

The training system uses **family-based feature selection**:

1. **Group by family**: `ma_*`, `vol_*`, `returns`, etc.
2. **Select top K per family**: Default 3 features per family
3. **Overall top M features**: Default 30 features total

This prevents over-representation of any single feature family.

### Example Selection
From 27 features, might select:
- Top 3 MA features: `ma_5`, `ma_20`, `ma_50`
- Top 3 volatility: `vol_z`, `vol_ratio`, `volatility_20`
- Top 3 volume: `afmlvol`, `dvol`, `dvol_sign`
- etc.

---

## Feature Scaling

Features are **not scaled** before training because:
1. RandomForest is scale-invariant
2. Raw values preserve interpretability
3. Feature importances remain meaningful

If switching to linear models, add standardization:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## Missing Value Handling

Strategy:
1. **Technical indicators**: May have NaN for initial periods (e.g., `ma_50` needs 50 bars)
2. **Bar statistics**: May have NaN if insufficient tick data
3. **Training**: Rows with any NaN features are dropped via `dropna()`

**Impact**: First ~50 bars of each day may be lost due to MA calculation warmup.

---

## Data Leakage Prevention

**Excluded from features**:
- `open`, `high`, `low`, `close` (raw prices)
- Future information (all features are point-in-time)

**Label generation**: Uses only future data (no look-ahead bias)

---

## Feature Importance Analysis

After training, check which features the model uses:

```python
import json
with open('experiments/my_exp/models/model_info_20250102.json') as f:
    info = json.load(f)
    print("Selected features:", info['selected_features'])
```

Top features typically include:
- Recent moving averages (`ma_5`, `ma_20`)
- Volatility metrics (`vol_z`, `vol_ratio`)
- Volume flow (`dvol_sign`, `afmlvol`)
- Returns (`returns`, `log_returns`)

---

## Extending Features

To add new features:

1. **Technical indicators**: Edit `calculate_technical_indicators()` in `mainoffset.py`
2. **Bar statistics**: Pre-compute in your bar generation pipeline
3. **Update feature list**: Add to `feature_columns` in `run_date_offset()`

Example:
```python
# In calculate_technical_indicators()
df['ema_12'] = df['close'].ewm(span=12).mean()  # Add EMA

# In run_date_offset()
technical_features = [
    'ma_5', 'ma_20', 'ma_50', 'ema_12',  # Added ema_12
    'returns', 'log_returns', ...
]
```

---

## Performance Considerations

**Feature count vs model performance**:
- More features ≠ better performance
- Risk of overfitting with too many features
- Family-based selection helps
- Monitor out-of-sample performance

**Computational cost**:
- 27 features × 1000 samples/day ≈ instant
- RandomForest training scales well
- Prediction is fast (tree traversal)

**Storage**:
- Parquet compression very efficient
- ~100KB per day for features
- ~10MB for full month

---

## References

**Volatility measures**:
- Realized volatility: Andersen & Bollerslev (1998)
- Bipower variation: Barndorff-Nielsen & Shephard (2004)
- Jump detection: Lee & Mykland (2008)

**AFML**:
- Advances in Financial Machine Learning, Marcos López de Prado
