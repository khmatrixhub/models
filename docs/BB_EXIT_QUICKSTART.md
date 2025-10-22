# BB Exit Strategies - Quick Start Guide

## What's New

Added 4 BB-specific exit strategies for mean reversion trading:
1. **Middle Band** - Exit at MA line (fastest, highest win rate)
2. **Opposite Band** - Exit at opposite band (slowest, highest profit)
3. **Partial Reversion** - Exit at MA ± multiplier*std (configurable)
4. **Signal Reversal** - Exit on opposite signal (regime-based)

## Quick Usage

### 1. Create Config File

```yaml
# config_bb_example.yaml
spot: "USDJPY"
signal_strategy: "bb_mean_reversion"
signal_params:
  bb_window: 20
  bb_std: 2.0

labels:
  bb_middle_exit:
    pt: 0.002              # Backup exits
    sl: 0.001
    max_hold: 24
    use_bb_exit: true      # Enable BB exit
    bb_exit_strategy: "middle"  # Choose: middle, opposite_band, partial, signal
    bb_window: 20          # MUST match signal_params
    bb_std: 2.0
```

### 2. Generate Labels

```bash
python generate.py --config configs/config_bb_example.yaml --start-date 01022025 --end-date 01312025
```

### 3. Train Models

```bash
python train.py --config configs/config_bb_example.yaml --experiment bb_middle
```

## Exit Strategy Comparison

| Strategy | Config Value | Exit Trigger | Win Rate | Avg Profit | Best For |
|----------|--------------|--------------|----------|------------|----------|
| Middle | `"middle"` | Price crosses MA | High | Low | Consistent wins |
| Opposite | `"opposite_band"` | Price touches opposite band | Low | High | Patient trading |
| Partial | `"partial"` | MA ± 0.5*std | Medium | Medium | Balanced |
| Signal | `"signal"` | Opposite signal* | ~Same as Opposite | ~Same | Nearly same as opposite** |

\* For BB mean reversion, opposite signal = price touches opposite band  
\*\* `signal` and `opposite_band` are nearly equivalent - both exit when price reaches opposite band. Only subtle difference is signal deduplication on consecutive touches.

## Config Parameters

### Required for All BB Exits
```yaml
use_bb_exit: true
bb_exit_strategy: "..."    # middle | opposite_band | partial | signal
bb_window: 20              # MUST match signal generation
bb_std: 2.0                # MUST match signal generation
```

### Additional for Partial Strategy
```yaml
exit_std_multiplier: 0.5   # 0.0=at MA, 0.5=halfway, 1.0=at entry band
```

### Backup Exits (Required)
```yaml
pt: 0.002                  # Profit target (backup)
sl: 0.001                  # Stop loss (safety)
max_hold: 24               # Max hold hours (backup)
```

## Example Configs

See these files:
- `configs/config_bb_exits.yaml` - All 4 strategies compared
- `configs/config_bb_mean_reversion.yaml` - Simple BB example
- `configs/config_bb_rsi_filter.yaml` - BB with RSI filter

## Exit Priority

1. **BB Exit** (primary) - Strategy-specific exit
2. **Profit Target** (backup) - If PT hit first
3. **Stop Loss** (safety) - If SL hit first
4. **Max Hold** (backup) - If timeout

## Testing Different Strategies

```bash
# Test all 4 strategies at once
python generate.py --config configs/config_bb_exits.yaml --start-date 01022025 --end-date 01312025

# Compare results
python working_files/analysis/compare_bb_exits.py
```

## Implementation Details

### Function: `generate_labels_bb_exit()`

Located in `generate.py` after `generate_labels_ma_exit()` (line 664).

**Parameters**:
- `df` - DataFrame with OHLC data
- `signals` - BB signals (1=long, -1=short, 0=none)
- `pt_pct` - Profit target (backup)
- `sl_pct` - Stop loss
- `max_hold_hours` - Max hold (backup)
- `bb_exit_config` - Dict with:
  - `strategy`: 'middle' | 'opposite_band' | 'partial' | 'signal'
  - `bb_window`: BB calculation window (must match signal)
  - `bb_std`: BB std multiplier (must match signal)
  - `exit_std_multiplier`: For 'partial' strategy only

**Returns**: DataFrame with columns:
- `label` - Binary (1=profitable, -1=loss)
- `profit` - Actual profit in price units
- `exit_bars` - Bars until exit
- `exit_reason` - 'bb_middle' | 'bb_opposite' | 'bb_partial' | 'signal' | 'pt' | 'sl' | 'max_hold'
- Entry/exit prices and timestamps
- Signal direction and MA values

### Exit Detection Logic

For each future bar after entry:

```python
# Middle band exit
if signal_direction == 1:  # Long
    if price >= bb_middle:
        exit_reason = 'bb_middle'

# Opposite band exit  
if signal_direction == 1:  # Long from lower band
    if price >= bb_upper:
        exit_reason = 'bb_opposite'

# Partial exit
if signal_direction == 1:  # Long
    exit_level = bb_middle + (exit_std_multiplier * std_dev)
    if price >= exit_level:
        exit_reason = 'bb_partial'

# Signal exit
if opposing_signal appears:
    exit_reason = 'signal'

# Backup: PT/SL
if returns >= pt_pct:
    exit_reason = 'pt'
elif returns <= -sl_pct:
    exit_reason = 'sl'
```

## Train.py Compatibility

✅ **NO CHANGES NEEDED** - Works with existing `train.py`:
- BB exit labels are just another label strategy
- Same feature set (75 features)
- Same signal filtering logic
- Same training workflow

## Best Practices

1. **Match BB Parameters**: Ensure `bb_window` and `bb_std` match signal generation
2. **Start with Middle**: Test middle band first (safest)
3. **Adjust Backup Exits**: Set PT/SL/max_hold appropriate for each strategy
4. **Compare Strategies**: Run all 4 to find best fit for your market
5. **Check Exit Distribution**: Most exits should be BB-based, not backups

## Troubleshooting

**All exits are 'max_hold' or 'pt'**
- BB parameters don't match signal generation
- Verify `bb_window` and `bb_std` match `signal_params`

**Very low win rate with opposite band**
- Expected! Opposite band requires full reversion
- Try 'partial' strategy with exit_std_multiplier = 0.5

**Too many 'bb_middle' exits with low profit**
- Middle band too aggressive
- Try 'partial' with exit_std_multiplier = 0.5 or 0.75

## Documentation

- `BB_EXIT_STRATEGIES.md` - Comprehensive guide with all details
- `BB_MEAN_REVERSION.md` - Signal generation strategy
- `TRAIN_COMPATIBILITY.md` - Integration with train.py
- `FOLDER_STRUCTURE.md` - Output organization

## Files Modified

- `generate.py` - Added `generate_labels_bb_exit()` function (line 664)
- `generate.py` - Updated label generation loop to support `use_bb_exit` (line 1026)
- `configs/config_bb_exits.yaml` - Example config with all 4 strategies
- `BB_EXIT_STRATEGIES.md` - Comprehensive documentation
- `working_files/tests/test_bb_exits.py` - Unit tests

## Next Steps

1. Generate labels with BB exits
2. Train models on different exit strategies
3. Compare performance metrics
4. Choose best strategy for production
5. Optimize `exit_std_multiplier` for 'partial' strategy
