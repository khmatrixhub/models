# Bollinger Band Exit Strategies

## Overview

The BB exit strategies implement realistic exits for Bollinger Band mean reversion trading. Instead of generic profit target/stop loss, these exits are aligned with the BB mean reversion logic itself.

## Mean Reversion Logic

**Entry**: When price touches BB outer bands (oversold/overbought)
- Long: Price touches lower band (oversold)
- Short: Price touches upper band (overbought)

**Exit Philosophy**: Exit when mean reversion completes
- Price returns toward the mean (middle band)
- Or price fully reverts to opposite extreme (opposite band)
- Or opposite condition occurs (signal reversal)

## Exit Strategies

### 1. Middle Band Exit (`bb_exit_strategy: "middle"`)

**Logic**: Exit when price crosses back to the moving average (middle line)

**Characteristics**:
- **Quickest exits**: First to trigger as price reverts
- **Highest win rate**: Exits early, takes small profits consistently
- **Lowest avg profit**: Doesn't wait for full reversion
- **Best for**: High-frequency, low-risk trading

**Example**:
```yaml
bb_middle:
  use_bb_exit: true
  bb_exit_strategy: "middle"
  bb_window: 20
  bb_std: 2.0
  pt: 0.002      # Backup exit (rarely hit)
  sl: 0.001
  max_hold: 24
```

**Exit Condition**:
- Long: Exit when `close >= BB_middle`
- Short: Exit when `close <= BB_middle`

---

### 2. Opposite Band Exit (`bb_exit_strategy: "opposite_band"`)

**Logic**: Exit when price touches the opposite BB outer band (full reversion)

**Characteristics**:
- **Slowest exits**: Waits for complete reversion
- **Lower win rate**: Many trades hit PT/SL/max_hold instead
- **Highest avg profit**: Captures full mean reversion move
- **Best for**: Patient trading, strong mean reversion markets

**Example**:
```yaml
bb_opposite:
  use_bb_exit: true
  bb_exit_strategy: "opposite_band"
  bb_window: 20
  bb_std: 2.0
  pt: 0.005      # Higher PT (opposite band may be far)
  sl: 0.001
  max_hold: 48   # Longer hold needed
```

**Exit Condition**:
- Long (entered at lower band): Exit when `close >= BB_upper`
- Short (entered at upper band): Exit when `close <= BB_lower`

---

### 3. Partial Reversion Exit (`bb_exit_strategy: "partial"`)

**Logic**: Exit at an intermediate level between entry and middle band

**Characteristics**:
- **Balanced approach**: Between middle and opposite band
- **Configurable risk/reward**: Adjust `exit_std_multiplier`
- **Moderate win rate and profit**: Goldilocks zone
- **Best for**: Fine-tuning risk/reward profile

**Example**:
```yaml
bb_partial_50:
  use_bb_exit: true
  bb_exit_strategy: "partial"
  bb_window: 20
  bb_std: 2.0
  exit_std_multiplier: 0.5  # Exit at MA ± 0.5*std (halfway)
  pt: 0.003
  sl: 0.001
  max_hold: 36
```

**Exit Levels**:
```
exit_std_multiplier = 0.5:
  Long exit:  MA + 0.5*std (halfway between MA and upper band)
  Short exit: MA - 0.5*std (halfway between MA and lower band)

exit_std_multiplier = 0.25:
  Closer to middle (quicker exits, higher win rate)

exit_std_multiplier = 0.75:
  Closer to opposite band (slower exits, higher profit)
```

**Exit Condition**:
- Long: Exit when `close >= (BB_middle + exit_std_multiplier * std_dev)`
- Short: Exit when `close <= (BB_middle - exit_std_multiplier * std_dev)`

---

### 4. Signal Reversal Exit (`bb_exit_strategy: "signal"`)

**Logic**: Exit when an opposite BB signal occurs (market regime change)

**⚠️ Important**: For BB mean reversion, this is **nearly equivalent** to `opposite_band` because:
- BB signals are generated when price touches upper/lower bands
- Opposite signal = price touched opposite band
- **Difference**: Signal deduplication prevents multiple signals on consecutive bars at same band
- In practice: `opposite_band` exits on ANY touch, `signal` exits only on FIRST touch (new signal)

**Characteristics**:
- **Regime-based**: Exits when market switches from oversold to overbought (or vice versa)
- **Variable holding period**: Depends on market volatility
- **Nearly identical to opposite_band**: Both exit when price reaches opposite band
- **Subtle difference**: Signal strategy won't exit if price stays at band without generating new signal
- **Best for**: When you want to ignore "riding the band" scenarios (price stays at extreme)

**Example**:
```yaml
bb_signal:
  use_bb_exit: true
  bb_exit_strategy: "signal"
  bb_window: 20
  bb_std: 2.0
  pt: 0.003
  sl: 0.001
  max_hold: 48
```

**Exit Condition**:
- Long (signal=1): Exit when new short signal (signal=-1) appears
- Short (signal=-1): Exit when new long signal (signal=1) appears

**Opposite Band vs Signal Strategy**:

For BB mean reversion, these are conceptually similar but with a key difference:

| Aspect | opposite_band | signal |
|--------|---------------|--------|
| Exit Trigger | Price touches opposite band | Opposite signal generated |
| Multiple Touches | Exits on ANY touch | Exits only on FIRST touch (new signal) |
| Riding the Band | Exits immediately | May continue if price stays at band |
| Implementation | Price-based check | Signal-based check |

**Example Scenario**:
```
Day 1, 10am: Long entry at lower band ($100.00)
Day 1, 2pm:  Price reaches upper band ($100.50)  ← Both strategies exit here
Day 1, 3pm:  Price still at upper band ($100.51)  ← opposite_band would exit, signal already exited

Special case:
Day 1, 10am: Long entry at lower band ($100.00)
Day 1, 2pm:  Price reaches upper band ($100.50), short signal generated
Day 1, 3pm:  Price dips to $100.45 (still above MA), rises to $100.52
Day 1, 4pm:  Price at $100.52 (upper band) ← opposite_band exits, signal doesn't (no new signal)
```

**Recommendation**: Use `opposite_band` for cleaner logic - it exits whenever price reaches the target, regardless of signal generation quirks.

---

## Configuration Parameters

### Required for All BB Exits

```yaml
labels:
  strategy_name:
    use_bb_exit: true          # Enable BB exit logic
    bb_exit_strategy: "..."    # Choose: middle, opposite_band, partial, signal
    bb_window: 20              # MUST match signal generation
    bb_std: 2.0                # MUST match signal generation
```

### Additional for Partial Strategy

```yaml
    exit_std_multiplier: 0.5   # Only for bb_exit_strategy: "partial"
                               # Range: 0.0 (at MA) to 1.0 (at entry band)
```

### Backup Exits (Required for Safety)

```yaml
    pt: 0.002                  # Profit target (backup)
    sl: 0.001                  # Stop loss (always needed)
    max_hold: 24               # Max hold hours (backup)
```

**Important**: PT/SL/max_hold are still checked on EVERY bar. They act as safety nets when BB exits don't trigger.

---

## Exit Priority Order

The function checks exit conditions in this order:

1. **BB Exit** (primary strategy)
   - Checks BB-specific condition each bar
   - `bb_middle`: Price crosses middle line
   - `bb_opposite`: Price touches opposite band
   - `bb_partial`: Price reaches intermediate level
   - `signal`: Opposite signal appears

2. **Profit Target** (backup)
   - If return >= `pt_pct` before BB exit

3. **Stop Loss** (safety)
   - If return <= `-sl_pct` before BB exit

4. **Max Hold** (backup)
   - If no exit after `max_hold_hours`

---

## Comparison Table

| Strategy | Exit Timing | Win Rate | Avg Profit | Risk Level | Use Case |
|----------|-------------|----------|------------|------------|----------|
| **Middle** | Fast (1-12h) | High (60-70%) | Low | Low | High-frequency, consistent wins |
| **Opposite** | Slow (12-48h) | Low (40-50%) | High | High | Patient, strong reversion |
| **Partial (0.5)** | Medium (6-24h) | Medium (50-60%) | Medium | Medium | Balanced risk/reward |
| **Signal** | ~Same as Opposite | ~Same as Opposite | ~Same as Opposite | High | Similar to opposite_band* |

\* **Note**: `signal` is nearly identical to `opposite_band` for BB mean reversion. Both exit when price reaches opposite band. Only difference: `signal` requires a new signal to be generated (first touch), while `opposite_band` exits on any touch. In practice, results should be very similar.

---

## Example: Testing All 4 Strategies

See `configs/config_bb_exits.yaml`:

```yaml
spot: "USDJPY"
signal_strategy: "bb_mean_reversion"

labels:
  bb_middle:
    use_bb_exit: true
    bb_exit_strategy: "middle"
    ...
  
  bb_opposite:
    use_bb_exit: true
    bb_exit_strategy: "opposite_band"
    ...
  
  bb_partial_50:
    use_bb_exit: true
    bb_exit_strategy: "partial"
    exit_std_multiplier: 0.5
    ...
  
  bb_signal:
    use_bb_exit: true
    bb_exit_strategy: "signal"
    ...
```

**Run**:
```bash
python generate.py --config configs/config_bb_exits.yaml --start-date 01022025 --end-date 01312025
```

**Compare Results**:
```python
import pandas as pd

# Load all 4 label strategies
middle = pd.read_parquet('results/config_bb_exits/data/20250102_USDJPY_0.002_0.001_24_y.parquet')
opposite = pd.read_parquet('results/config_bb_exits/data/20250102_USDJPY_0.005_0.001_48_y.parquet')
partial = pd.read_parquet('results/config_bb_exits/data/20250102_USDJPY_0.003_0.001_36_y.parquet')
signal_exit = pd.read_parquet('results/config_bb_exits/data/20250102_USDJPY_0.003_0.001_48_y.parquet')

# Compare metrics
for name, df in [('Middle', middle), ('Opposite', opposite), ('Partial', partial), ('Signal', signal_exit)]:
    print(f"\n{name} Band Exit:")
    print(f"  Win Rate: {(df['label'] == 1).mean():.1%}")
    print(f"  Avg Profit: {df['profit'].mean():.6f}")
    print(f"  Avg Exit Bars: {df['exit_bars'].mean():.1f}")
    print(f"  Exit Reasons:")
    print(df['exit_reason'].value_counts())
```

---

## Implementation Details

### Bollinger Band Calculation

The BB values are recalculated inside `generate_labels_bb_exit()`:

```python
bb_middle = df['close'].rolling(window=bb_window).mean()
bb_std_dev = df['close'].rolling(window=bb_window).std()
bb_upper = bb_middle + (bb_std * bb_std_dev)
bb_lower = bb_middle - (bb_std * bb_std_dev)
```

**Critical**: `bb_window` and `bb_std` MUST match the signal generation parameters to ensure consistency.

### Exit Detection Logic

For each future bar after entry, the function checks:

```python
# Middle band exit
if signal_direction == 1:  # Long
    if future_price >= future_bb_middle[j]:
        exit_reason = 'bb_middle'
        break

# Opposite band exit
if signal_direction == 1:  # Long from lower band
    if future_price >= future_bb_upper[j]:
        exit_reason = 'bb_opposite'
        break

# Partial exit (configurable)
if signal_direction == 1:  # Long
    exit_level = bb_middle + (exit_std_multiplier * bb_std_dev)
    if future_price >= exit_level:
        exit_reason = 'bb_partial'
        break

# Signal exit
if opposing_signal appears:
    exit_reason = 'signal'
    break

# Backup: PT/SL still checked
if returns[j] >= pt_pct:
    exit_reason = 'pt'
    break
elif returns[j] <= -sl_pct:
    exit_reason = 'sl'
    break
```

---

## Best Practices

1. **Match BB Parameters**: Always set `bb_window` and `bb_std` in labels to match signal generation
   ```yaml
   signal_params:
     bb_window: 20
     bb_std: 2.0
   
   labels:
     my_strategy:
       bb_window: 20    # MUST match signal_params
       bb_std: 2.0      # MUST match signal_params
   ```

2. **Start with Middle Band**: Test `bb_exit_strategy: "middle"` first (safest, fastest)

3. **Adjust Backup Exits**: Set PT/SL/max_hold appropriate for each strategy
   - Middle: Tight PT (0.001-0.002), short hold (12-24h)
   - Opposite: Wide PT (0.003-0.005), long hold (36-48h)
   - Partial: Medium values

4. **Test Multiple Multipliers**: For partial strategy, try 0.25, 0.5, 0.75
   ```yaml
   bb_partial_25:
     exit_std_multiplier: 0.25  # Conservative
   
   bb_partial_50:
     exit_std_multiplier: 0.5   # Balanced
   
   bb_partial_75:
     exit_std_multiplier: 0.75  # Aggressive
   ```

5. **Compare Exit Distributions**: Check `exit_reason` value counts
   ```python
   df['exit_reason'].value_counts()
   # Ideally most exits should be BB-based, not backup PT/SL/max_hold
   ```

---

## Troubleshooting

**Problem**: All exits are 'max_hold' or 'pt'
- **Cause**: BB parameters don't match signal generation
- **Fix**: Verify `bb_window` and `bb_std` are identical in signal_params and labels

**Problem**: Very low win rate with opposite band
- **Cause**: Market not reverting fully, PT/SL/max_hold hit first
- **Fix**: This is expected behavior. Opposite band requires strong mean reversion. Consider:
  - Increase `pt` and `max_hold`
  - Or use 'partial' strategy instead

**Problem**: Too many 'bb_middle' exits with low profit
- **Cause**: Middle band exit too aggressive
- **Fix**: Switch to 'partial' with exit_std_multiplier = 0.5 or 0.75

---

## Next Steps

1. **Generate Labels**: Run `generate.py` with BB exit config
2. **Train Models**: Use `train.py` with any experiment name
3. **Compare Strategies**: Analyze which exit strategy works best for your market/timeframe
4. **Optimize Parameters**: Test different `exit_std_multiplier` values
5. **Production**: Deploy best-performing strategy

---

**Related Documentation**:
- `BB_MEAN_REVERSION.md` - Signal generation strategy
- `TRAIN_COMPATIBILITY.md` - Model training integration
- `FOLDER_STRUCTURE.md` - Output organization
