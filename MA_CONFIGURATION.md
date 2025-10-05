# Moving Average Configuration Guide

## Overview

The MA crossover signal generation now supports configurable window sizes through the YAML config file. This allows you to experiment with different MA strategies without modifying code.

## Configuration

Add the `ma_windows` section to your config file:

```yaml
# Moving Average parameters for signal generation
ma_windows:
  fast: 5   # Fast MA window (default: 5)
  slow: 20  # Slow MA window (default: 20)
```

## Signal Generation Strategy

The system generates MA crossover signals:
- **Bullish Signal (+1)**: Fast MA crosses above Slow MA
- **Bearish Signal (-1)**: Fast MA crosses below Slow MA
- **No Signal (0)**: No crossover occurred

## Common MA Configurations

| Strategy | Fast | Slow | Signals/Day | Description |
|----------|------|------|-------------|-------------|
| **Very Fast** | 3 | 10 | ~150 | Very responsive, many signals |
| **Fast** (default) | 5 | 20 | ~80 | Good balance of frequency and quality |
| **Medium** | 10 | 50 | ~30 | More selective, fewer false signals |
| **Slow** | 20 | 100 | ~15 | Very selective, strong trends only |
| **Golden Cross** | 50 | 200 | ~5 | Classic long-term strategy |

## Signal Frequency Impact

Based on 1 day of USDJPY data (1410 bars):

```
MA(3,10):   153 signals (11% of bars) - Very active
MA(5,20):    81 signals (6% of bars)  - Default
MA(10,50):   33 signals (2% of bars)  - Selective
MA(20,100):  15 signals (1% of bars)  - Conservative
```

## Feature Alignment

The MA windows used for signal generation are **automatically applied** to feature calculation:
- `ma_5` feature uses `fast_window` (e.g., 5)
- `ma_20` feature uses `slow_window` (e.g., 20)
- `ma_50` remains fixed at 50 bars (long-term trend)

This ensures consistency between signals and features used for ML training.

## Choosing MA Windows

### Faster MAs (e.g., 3/10, 5/20)
**Pros:**
- More trading opportunities
- Captures short-term moves
- Good for range-bound markets

**Cons:**
- More false signals (whipsaws)
- Higher transaction costs
- Model sees more noisy data

### Slower MAs (e.g., 20/100, 50/200)
**Pros:**
- Higher quality signals
- Better for trending markets
- Fewer false breakouts

**Cons:**
- Fewer trading opportunities
- May miss short-term moves
- Requires longer data history

## Example Configurations

### High-Frequency Day Trading
```yaml
ma_windows:
  fast: 3
  slow: 10
```
Results: ~150 signals/day, need strong ML model to filter

### Standard Intraday Trading
```yaml
ma_windows:
  fast: 5
  slow: 20
```
Results: ~80 signals/day, balanced approach (default)

### Swing Trading
```yaml
ma_windows:
  fast: 10
  slow: 50
```
Results: ~30 signals/day, quality over quantity

### Position Trading
```yaml
ma_windows:
  fast: 20
  slow: 100
```
Results: ~15 signals/day, only strong trends

## Testing Different Configurations

To test different MA windows:

1. Edit `config_example.yaml`:
```yaml
ma_windows:
  fast: 10
  slow: 50
```

2. Regenerate data:
```bash
python mainoffset.py --config config_example.yaml \
  --start-date 01022025 --end-date 01312025 --n_jobs 4
```

3. Train and compare:
```bash
python train_forex.py --config config_example.yaml \
  --label-name ma_crossover --experiment ma_10_50
```

4. Compare results across different MA windows to find optimal settings

## Notes

- Smaller windows = More signals but more noise
- Larger windows = Fewer signals but higher quality
- ML model can help filter noisy signals from fast MAs
- Consider your time horizon and risk tolerance
- Test on historical data before live trading

## Related Documentation

- [MA Exit Strategy](MA_EXIT_STRATEGY.md) - Using MA signals for exits
- [Label Configuration Guide](LABEL_CONFIG_GUIDE.md) - Label strategies
- [Features Guide](FEATURES.md) - All available features
