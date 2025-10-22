# Signal Reversal Feature Documentation

## Overview

The signal reversal feature allows you to flip all trading signals (long ↔ short) to test if your signals have predictive power in the opposite direction. This is particularly useful when your baseline signals show negative performance.

## Why Use Signal Reversal?

### Scenario 1: Negative Baseline Performance
If your original signals show significant negative returns:
```
threshold  count  accuracy  precision   recall       f1  total_profit  mean_profit
    0.00  16244  0.272593   0.272593 1.000000 0.428406      -38.1440    -0.002348
```

This negative performance (-0.002348 mean profit) suggests the signals have predictive power, **but in the wrong direction**.

### What This Means:
- **Normal expectation**: Random signals → weakly positive, lose to spread
- **Your case**: Significantly negative → consistently wrong direction
- **Solution**: Reverse the signals to make them profitable

## How It Works

When `reverse_signals: true` is enabled:

1. **Signals are flipped**:
   - Original `1` (long) → `-1` (short)
   - Original `-1` (short) → `1` (long)

2. **Profits are negated**:
   - Original profit: +2% for long → Reversed: -2% (would be loss for short)
   - This maintains correct profit calculation for reversed direction

3. **Labels are flipped**:
   - Original label: `1` (profit) → `-1` (loss)
   - Original label: `-1` (loss) → `1` (profit)

4. **Signal-adjusted features are reversed**:
   - Features multiplied by signal direction are negated:
     - `returns_adj`
     - `log_returns_adj`
     - `ma5_distance_adj`
     - `ma20_distance_adj`

## Configuration

### In YAML Config File

```yaml
# Training configuration
training:
  n_estimators: 100
  use_rfe: false
  max_features: 30
  max_features_per_family: 3
  reverse_signals: false  # Set to true to flip all signals
  bagging:
    enabled: true
    n_estimators: 200
```

### Running Experiments

**Baseline (original signals)**:
```bash
python train.py --config configs/config_example.yaml \
                --experiment baseline \
                --label-name conservative
```

**Reversed signals**:
```bash
# Edit config: set reverse_signals: true
python train.py --config configs/config_example.yaml \
                --experiment reversed \
                --label-name conservative
```

**Or create separate config**:
```bash
# configs/config_example_reversed.yaml
cp configs/config_example.yaml configs/config_example_reversed.yaml
# Edit: set reverse_signals: true

python train.py --config configs/config_example_reversed.yaml \
                --experiment baseline \
                --label-name conservative
```

## Expected Results

### Original Signals (Negative)
```
Total profit (all trades): -38.1440
Mean profit per trade: -0.002348
Median profit per trade: -0.02100
Win rate: 0.273
```

### Reversed Signals (Should be Positive)
```
Total profit (all trades): +38.1440  (flipped sign)
Mean profit per trade: +0.002348    (flipped sign)
Median profit per trade: +0.02100   (flipped sign)
Win rate: 0.727                      (1 - 0.273)
```

## Interpretation

### If reversal works (profits flip positive):
✅ **Your signals have genuine predictive power**
- The MA crossover strategy works, just in reverse
- Trading opposite to signals would be profitable
- Consider investigating why signals are inverted:
  - MA window periods might be backwards (fast/slow swapped)?
  - Price data might be inverted?
  - Signal logic bug in generate.py?

### If reversal doesn't help (still negative or random):
⚠️ **Signals may be truly random**
- No predictive power in either direction
- Market conditions changed
- MA crossover doesn't work for this pair/timeframe
- Need different signal generation approach

## Metadata Tracking

The reversal status is saved in experiment metadata:

```json
{
  "experiment_name": "reversed",
  "training_config": {
    "reverse_signals": true,
    "n_estimators": 100,
    "bagging": {"enabled": true}
  },
  "label_config": {
    "name": "conservative",
    "profit_target": 0.001,
    "stop_loss": 0.001
  }
}
```

## Debugging Output

When reversal is enabled, you'll see:

```
============================================================
SIGNAL REVERSAL ENABLED
All signals will be reversed: Long->Short, Short->Long
Profits will be flipped accordingly
============================================================

DEBUG: Reversed signals (original range: [-1, 1] -> reversed: [-1, 1])
DEBUG: Reversed profits (original mean: -0.002348 -> reversed mean: 0.002348)
DEBUG: Reversed labels (1<->-1)
DEBUG: Reversed feature returns_adj
DEBUG: Reversed feature log_returns_adj
DEBUG: Reversed feature ma5_distance_adj
DEBUG: Reversed feature ma20_distance_adj
```

## Common Questions

**Q: Why not just fix the signal generation instead of reversing?**
A: Reversal is a diagnostic tool to confirm predictive power exists. Once confirmed, you can investigate the root cause.

**Q: Do lagged profit features need special handling?**
A: No - they're calculated from the (now reversed) profits, so they automatically reflect the reversed strategy.

**Q: Can I compare reversed experiments directly to originals?**
A: Yes - the feature importances and model structure remain the same, only the direction flips.

**Q: What about transaction costs?**
A: Reversal doesn't change trade count or timing, so relative performance (profitable vs unprofitable) is valid.

## Files Modified

- `train.py`: Added signal/profit/label/feature reversal logic
- `configs/config_example.yaml`: Added `reverse_signals` option
- `experiment_metadata.json`: Includes `training_config` with reversal status

## Testing

Run the example script to see how reversal works:
```bash
python working_files/analysis/test_signal_reversal.py
```
