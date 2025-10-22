# Label Generation System

## Overview

The label generation system creates forward-looking trading labels by simulating a profit target (PT) and stop loss (SL) strategy for each bar in the data.

## Label Structure

Each label file contains a DataFrame with three columns:

| Column | Type | Description |
|--------|------|-------------|
| `label` | int | Binary classification: `1` (profitable trade), `-1` (loss) |
| `profit` | float | Actual profit/loss as decimal (e.g., 0.001 = 0.1%) |
| `exit_bars` | float | Number of bars until exit |

## Label Generation Logic

For each bar at time `t`, the system:

1. **Entry Price**: Uses `close` price at time `t`
2. **Look Forward**: Examines next `max_hold_hours` bars
3. **Check Conditions**:
   - **Profit Target Hit**: Price moves up by `pt_pct` or more
   - **Stop Loss Hit**: Price moves down by `sl_pct` or more
   - **Neither Hit**: Hold until `max_hold_hours` expires

4. **Determine Exit**:
   - If **both PT and SL hit**: Exit at whichever occurred first
   - If **only PT hit**: Exit at PT with `+pt_pct` profit
   - If **only SL hit**: Exit at SL with `-sl_pct` loss
   - If **neither hit**: Exit at `max_hold_hours` with actual return

## Example Configuration

```yaml
labels:
  conservative:
    pt: 0.001      # 0.1% profit target
    sl: 0.001      # 0.1% stop loss
    max_hold: 24   # 24 bars maximum hold
```

## Example Label Output

```
                     label    profit  exit_bars
2025-01-01 00:00:00    1.0  0.001000       12.0  # Hit PT after 12 bars
2025-01-01 01:00:00   -1.0 -0.001000        8.0  # Hit SL after 8 bars
2025-01-01 02:00:00    1.0  0.000456       24.0  # Held max, closed positive
2025-01-01 03:00:00   -1.0 -0.000234       24.0  # Held max, closed negative
```

## Usage in Training

The training script (`train_forex.py`) loads these labels and:

1. **Reads the label file**: `{date}_{spot}_{pt}_{sl}_{max_hold}_y.parquet`
2. **Converts to binary**: `binary_labels = (labels_df['label'] > 0).astype(int)`
3. **Extracts profits**: `profits = labels_df['profit']`
4. **Trains classifier**: Predicts whether trade will be profitable (`label > 0`)
5. **Evaluates with actual profits**: Uses `profit` column for P&L calculations

## Label Statistics

After generation, the system logs:
- **Win rate**: Percentage of profitable labels
- **Average profit**: Mean of all profits (positive and negative)
- **Profit distribution**: Statistics on profit column

Example log output:
```
Saved labels for 'conservative': 20250102_EURUSD_0.001_0.001_24_y.parquet
  Stats: 543/1000 profitable (54.3%), avg profit: 0.000123
```

## Files Generated

For each date and label configuration:

```
output/
├── 20250102_EURUSD_features.parquet          # Features (27 total: 11 technical + 16 bar statistics)
├── 20250102_EURUSD_signals.parquet           # MA crossover signals
└── 20250102_EURUSD_0.001_0.001_24_y.parquet # Labels (label, profit, exit_bars)
```

### Feature Breakdown

**Technical Indicators (11)**:
- Moving averages: `ma_5`, `ma_20`, `ma_50`
- Returns: `returns`, `log_returns`
- Price patterns: `price_range`, `body_size`
- Volatility: `volatility_5`, `volatility_20`
- Momentum: `rsi`
- Bollinger Bands: `bb_position`

**Bar Statistics (16)** - Pre-computed from tick-level data:
- Volume: `afmlvol` (AFML volume), `dvol` (dollar volume), `dvol_sign`
- Volatility metrics: `vol_skew`, `vol_kurt`, `vol_of_vol`, `vol_z`, `vol_ratio`, `vol_slope`, `vol_autocorr1`
- Realized measures: `rv` (realized volatility), `bpv` (bipower variation), `rq` (realized quarticity)
- Jump/tail detection: `jump_proxy`, `tail_exceed_rate`
- Timing: `seconds_diff` (time between bars)

## Validation

The label generation includes several validations:

1. **Index Alignment**: Labels share same index as features
2. **No Look-Ahead Bias**: Only uses data strictly after entry time
3. **Realistic Execution**: Simulates actual PT/SL hit order
4. **Missing Data Handling**: Labels as NaN for bars without enough future data

## Regenerating Labels

To regenerate with new label configurations:

```bash
# Update config_example.yaml with new label strategies
# Then regenerate:
conda run -n models python mainoffset.py \
  --config config_example.yaml \
  --start-date 01022025 \
  --end-date 01312025 \
  --n_jobs 4
```

This will create label files for all configured strategies.

## Backwards Compatibility

**Old format (deprecated)**: Single column Series with 1/-1 values  
**New format**: DataFrame with `label`, `profit`, `exit_bars` columns

The training script now **requires** the new format with the `profit` column.

## Key Improvements

✅ **Real Profit Tracking**: No more random dummy data  
✅ **Exit Timing**: Records when trades exit  
✅ **Better Evaluation**: Can calculate actual P&L, not just win rate  
✅ **Transparency**: Clear what profit is expected from each prediction  
✅ **Multiple Strategies**: Compare different PT/SL configurations accurately
