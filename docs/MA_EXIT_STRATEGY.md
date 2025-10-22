# MA Crossover Exit Strategy

## Overview

The MA (Moving Average) crossover exit strategy is a more realistic labeling approach for MA-based trading systems. Instead of exiting only on profit target (PT), stop loss (SL), or max hold, it **exits and reverses position when an opposite MA signal occurs**.

## Comparison: Standard vs MA Exit

### Standard Strategy (`use_ma_exit: false`)
Enters on every bar and exits on PT/SL/max_hold:
- **Entry**: Every bar (not signal-dependent)
- **Exit**: First of PT, SL, or max_hold
- **Use case**: General ML labels, any entry strategy

### MA Exit Strategy (`use_ma_exit: true`)
Enters on MA crossover and exits on opposite signal:
- **Entry**: Only on MA crossover signals (1 or -1)
- **Exit**: Opposite MA signal, or PT/SL/max_hold (backup)
- **Use case**: MA crossover trading systems, signal-following strategies

## Configuration

### Example Config

```yaml
labels:
  # Standard strategy
  conservative:
    pt: 0.001
    sl: 0.001
    max_hold: 24
    use_ma_exit: false  # Default
  
  # MA exit strategy
  ma_crossover:
    pt: 0.002
    sl: 0.001
    max_hold: 72
    use_ma_exit: true  # Use MA signals as exits
```

## How It Works

### Entry Logic
- **Bullish Crossover** (signal = 1): Enter long position
- **Bearish Crossover** (signal = -1): Enter short position
- **No Signal** (signal = 0): No label generated (skip this bar)

**Important**: Every signal bar receives a label. There are no NaN values for signal bars, even if they occur near the end of the data (the function uses whatever future data is available up to `max_hold_hours`).

### Exit Logic (Priority Order)

### Profit Calculation

For **long positions** (signal = 1):
```
profit = (exit_price - entry_price) / entry_price
```

For **short positions** (signal = -1):
```
profit = (entry_price - exit_price) / entry_price
```

## Label Structure

Labels file includes **4 columns** (vs 3 for standard):

| Column | Type | Description |
|--------|------|-------------|
| `label` | int | 1 (profitable) or -1 (loss) |
| `profit` | float | Actual profit/loss as decimal |
| `exit_bars` | float | Number of bars until exit |
| `exit_reason` | str | `'signal'`, `'pt'`, `'sl'`, or `'max_hold'` |

## Example Output

```python
                     label  profit  exit_bars  exit_reason
2025-01-01 08:00:00    1.0  0.0045       18.0       signal  # Long, exited on bearish signal
2025-01-01 09:00:00   -1.0 -0.0012       24.0     max_hold  # Short, held to max
2025-01-02 14:00:00    1.0  0.0020       12.0           pt  # Long, hit profit target
2025-01-02 15:00:00   -1.0 -0.0010        8.0           sl  # Short, hit stop loss
```

## Usage

### Generate Data

```bash
# Generate labels with MA exit strategy
python mainoffset.py \
  --config config_example.yaml \
  --start-date 01022025 \
  --end-date 01312025
```

This creates: `{date}_{spot}_{pt}_{sl}_{max_hold}_y.parquet`

### Train Model

```bash
# Train using MA crossover labels
python train_forex.py \
  --config config_example.yaml \
  --experiment ma_strategy_test \
  --label-name ma_crossover
```

The model learns to predict: **"Will this MA crossover signal be profitable?"**

## Key Differences in Behavior

| Aspect | Standard | MA Exit |
|--------|----------|---------|
| **Labels per day** | ~1000 (all bars) | ~5-20 (only signals) |
| **Position holding** | Independent trades | Follows MA signals |
| **Exit timing** | PT/SL/time | Primarily opposite signal |
| **Realism** | Generic | MA strategy-specific |
| **Use with signals** | Any | MA crossover only |

## Statistics Example

From a typical day:

**Standard strategy**:
- 1,440 labels (1 per minute)
- 720 profitable (50%)
- Avg profit: 0.0001 (0.01%)

**MA exit strategy**:
- 12 labels (12 crossovers)
- 7 profitable (58%)
- Avg profit: 0.0015 (0.15%)
- Exit reasons:
  - 6 signal exits
  - 3 profit targets
  - 2 stop losses
  - 1 max hold

## When to Use MA Exit

### Use MA Exit When:
✅ Trading MA crossover systems  
✅ Want realistic MA strategy backtests  
✅ Exit discipline follows signals  
✅ Modeling signal-following behavior  

### Use Standard When:
✅ General ML predictions  
✅ Entry strategy is not MA-based  
✅ Want more training samples  
✅ Testing different exit rules  

## Analysis Tips

### Exit Reason Distribution
```python
labels = pd.read_parquet('output/20250102_EURUSD_0.002_0.001_72_y.parquet')
print(labels['exit_reason'].value_counts())

# Expected output:
# signal      45%  (most exits from opposite signals)
# pt          25%  (some hit profit target)
# sl          20%  (some hit stop loss)
# max_hold    10%  (few held to max)
```

### Compare Strategies
```python
# Load both label types
standard = pd.read_parquet('output/20250102_EURUSD_0.001_0.001_24_y.parquet')
ma_exit = pd.read_parquet('output/20250102_EURUSD_0.002_0.001_72_y.parquet')

print(f"Standard: {len(standard)} labels, {(standard['profit'] > 0).mean():.1%} win rate")
print(f"MA exit: {len(ma_exit)} labels, {(ma_exit['profit'] > 0).mean():.1%} win rate")
```

## Performance Considerations

- **Fewer labels**: MA exit generates 50-100x fewer labels per day
- **Longer holds**: Trades typically last until next signal (hours to days)
- **More realistic**: Matches actual MA trading behavior
- **Less data**: May need longer time periods for training

## Future Enhancements

Potential additions:
- Support multiple MA periods (5/20, 10/50, etc.)
- Add position sizing to profit calculation
- Include partial exits (scale out)
- Support other signal types (RSI, MACD, etc.)
