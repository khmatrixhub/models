# Bollinger Band Mean Reversion Signal Strategy

## Overview

Added support for Bollinger Band-based mean reversion signals as an alternative to the default MA crossover strategy.

## Strategy Description

**Mean Reversion Theory**: When price extends too far from its mean, it tends to revert back.

### Bollinger Band Signals

- **Long Signal (Buy)**: Price touches or crosses below the lower Bollinger Band → Oversold condition
- **Short Signal (Sell)**: Price touches or crosses above the upper Bollinger Band → Overbought condition

### Optional RSI Filter

For more conservative signals, add RSI confirmation:
- **Long**: Price below lower BB AND RSI < 30 (oversold)
- **Short**: Price above upper BB AND RSI > 70 (overbought)

## Configuration

### Basic BB Mean Reversion (No RSI Filter)

```yaml
signal_strategy: "bb_mean_reversion"

bollinger_bands:
  window: 20     # Period for middle line (SMA)
  std: 2.0       # Standard deviations for bands
  rsi_threshold: null  # No RSI filter
```

### BB with RSI Confirmation

```yaml
signal_strategy: "bb_mean_reversion"

bollinger_bands:
  window: 20
  std: 2.0
  rsi_threshold: [30, 70]  # Long when RSI<30, Short when RSI>70
```

## Usage

### 1. Generate Data with BB Signals

```bash
python generate.py --config configs/config_bb_mean_reversion.yaml
```

### 2. Train Models

```bash
# Classification
python train.py --config configs/config_bb_mean_reversion.yaml \
                --experiment bb_mean_reversion \
                --label-name conservative

# Regression
python regression.py --config configs/config_bb_mean_reversion.yaml \
                     --experiment bb_mean_reversion \
                     --label-name conservative
```

## Strategy Comparison

| Strategy | Type | Entry Condition | Expected Behavior |
|----------|------|-----------------|-------------------|
| `ma_cross` | Trend Following | MA crossover | Enter on trend start, ride the trend |
| `bb_mean_reversion` | Mean Reversion | Price at BB extremes | Enter on overextension, profit on reversal |

## Parameters

### Bollinger Bands

- **window** (default: 20): Period for middle line (SMA)
  - Smaller = More sensitive, more signals
  - Larger = Smoother, fewer signals
  
- **std** (default: 2.0): Standard deviations for bands
  - Smaller (1.5-2.0) = Narrower bands, more signals
  - Larger (2.5-3.0) = Wider bands, fewer but stronger signals

- **rsi_threshold** (default: null): Optional RSI filter
  - `null`: Use BB only
  - `[30, 70]`: Standard RSI oversold/overbought
  - `[20, 80]`: More conservative (fewer signals)
  - `[40, 60]`: More aggressive (more signals)

## Label Configurations for Mean Reversion

Mean reversion strategies typically work best with:
- **Shorter max_hold** (12-24 hours): Expect quick reversal
- **Tighter targets** (0.1-0.2%): Take profits quickly on reversal
- **Tighter stops** (0.1%): Cut losses if reversal doesn't happen

Example:
```yaml
labels:
  mean_reversion_quick:
    pt: 0.0015   # 0.15% target
    sl: 0.001    # 0.1% stop
    max_hold: 12 # 12 hours (expect quick move)
```

## Example Configs

Three pre-configured examples provided:

1. **config_bb_mean_reversion.yaml** (USDJPY)
   - BB signals without RSI filter
   - Multiple label strategies
   
2. **config_bb_rsi_filter.yaml** (EURUSD)
   - BB signals WITH RSI confirmation
   - More conservative signal generation
   
3. **config_example.yaml** (original)
   - MA crossover strategy (default)
   - Trend following approach

## Implementation Details

### Signal Generation

The `generate_bb_mean_reversion_signals()` function:

1. Calculates Bollinger Bands (middle, upper, lower)
2. Identifies price touches/crosses of bands
3. Optionally checks RSI confirmation
4. Filters to first touch only (prevents multiple signals in same zone)

### Features

All technical features (MA, RSI, ATR, etc.) are still calculated and available for ML models, regardless of which signal strategy is used.

### Labels

Label generation (profit/loss calculation) works identically for both strategies - it's based on the PT/SL/max_hold parameters, not the signal generation method.

## Expected Differences

### Signal Count

- **MA Cross**: Typically 80-100 signals per day (crossovers)
- **BB Mean Reversion**: Varies widely (20-150 per day)
  - Volatile markets → More BB touches → More signals
  - Ranging markets → Frequent touches → Many signals
  - Trending markets → Fewer touches → Fewer signals

### Win Rate

- **MA Cross**: ~52-54% (trend following has ~50% baseline)
- **BB Mean Reversion**: Expected 55-60% (mean reversion has statistical edge)
  - Without RSI: More signals, lower win rate
  - With RSI: Fewer signals, higher win rate

## Future Enhancements

Potential additions:
- Combined signals (MA + BB)
- Volume confirmation
- Multiple timeframe analysis
- Adaptive BB parameters
- Squeeze detection (low volatility → high volatility transition)
