# Label Configuration Guide

## Overview

This guide explains how to configure and use named label strategies for forex trading model training. The system supports two exit strategies:

1. **Standard PT/SL Strategy**: Exit on profit target, stop loss, or max hold
2. **MA Crossover Exit Strategy**: Exit on opposite MA signal (more realistic for MA systems)

See [MA_EXIT_STRATEGY.md](MA_EXIT_STRATEGY.md) for details on the MA exit strategy.

## Configuration Format

Labels are now defined as **named strategies** in the YAML config file:

```yaml
labels:
  conservative:
    pt: 0.001      # Profit target (0.1%)
    sl: 0.001      # Stop loss (0.1%)
    max_hold: 24   # Maximum hold time in hours
    # use_ma_exit: false (default - standard PT/SL strategy)
  
  aggressive:
    pt: 0.002      # Profit target (0.2%)
    sl: 0.001      # Stop loss (0.1%)
    max_hold: 48   # Maximum hold time in hours
  
  swing:
    pt: 0.005      # Profit target (0.5%)
    sl: 0.002      # Stop loss (0.2%)
    max_hold: 120  # Maximum hold time in hours
  
  ma_crossover:
    pt: 0.002      # Backup profit target
    sl: 0.001      # Backup stop loss
    max_hold: 72   # Backup max hold
    use_ma_exit: true  # Use MA signal-based exit strategy
```

### Standard vs MA Exit Strategy

**Standard Strategy** (`use_ma_exit: false` or omitted):
- Labels generated for ALL bars
- Exit on PT, SL, or max_hold
- Typical: 500-2000 labels per day
- Best for: Modeling general market behavior

**MA Exit Strategy** (`use_ma_exit: true`):
- Labels generated only on MA signal bars (crossovers)
- Exit on opposite signal, PT, SL, or max_hold
- Typical: 5-20 labels per day
- Best for: Modeling MA trading systems realistically
- See [MA_EXIT_STRATEGY.md](MA_EXIT_STRATEGY.md) for details

## Workflow

### 1. Generate Data with Multiple Label Configurations

First, run `mainoffset.py` to generate features and labels for all configurations:

```bash
conda run -n models python mainoffset.py \
  --config config_example.yaml \
  --start-date 01022025 \
  --end-date 01312025 \
  --n_jobs 4
```

This will create label files for each configuration:
- `20250102_EURUSD_0.001_0.001_24_y.parquet` (conservative)
- `20250102_EURUSD_0.002_0.001_48_y.parquet` (aggressive)
- `20250102_EURUSD_0.005_0.002_120_y.parquet` (swing)

### 2. Train Models for Specific Label Strategy

Use `train_forex.py` with the `--label-name` parameter to select which strategy to train:

```bash
# Train with conservative strategy
conda run -n models python train_forex.py \
  --config config_example.yaml \
  --experiment conservative_test \
  --label-name conservative

# Train with aggressive strategy
conda run -n models python train_forex.py \
  --config config_example.yaml \
  --experiment aggressive_test \
  --label-name aggressive

# Train with swing strategy
conda run -n models python train_forex.py \
  --config config_example.yaml \
  --experiment swing_test \
  --label-name swing
```

### 3. Default Behavior

If you don't specify `--label-name`, the script will use the **first label configuration** in the config file:

```bash
# Uses 'conservative' (first in config)
conda run -n models python train_forex.py \
  --config config_example.yaml \
  --experiment default_test
```

## File Structure

After running the full pipeline:

```
output/
├── 20250102_EURUSD_features.parquet
├── 20250102_EURUSD_signals.parquet
├── 20250102_EURUSD_0.001_0.001_24_y.parquet     # conservative labels
├── 20250102_EURUSD_0.002_0.001_48_y.parquet     # aggressive labels
└── 20250102_EURUSD_0.005_0.002_120_y.parquet    # swing labels

experiments/
├── conservative_test/
│   ├── results/
│   │   ├── experiment_metadata.json   # Includes label_config info
│   │   ├── results.parquet
│   │   └── ...
│   ├── models/
│   │   └── model_info_20250102.json   # Includes label_config info
│   └── logs/
├── aggressive_test/
│   └── ...
└── swing_test/
    └── ...
```

## Experiment Metadata

Each experiment saves metadata showing which label configuration was used:

```json
{
  "experiment_name": "conservative_test",
  "label_config": {
    "name": "conservative",
    "profit_target": 0.001,
    "stop_loss": 0.001,
    "max_hold": 24,
    "all_configs": {
      "conservative": {"pt": 0.001, "sl": 0.001, "max_hold": 24},
      "aggressive": {"pt": 0.002, "sl": 0.001, "max_hold": 48},
      "swing": {"pt": 0.005, "sl": 0.002, "max_hold": 120}
    }
  }
}
```

## Comparing Strategies

You can run multiple experiments and compare results:

```bash
# Run all strategies
for strategy in conservative aggressive swing; do
  conda run -n models python train_forex.py \
    --config config_example.yaml \
    --experiment ${strategy}_jan2025 \
    --label-name ${strategy}
done

# Compare results
conda run -n models python -c "
import pandas as pd
strategies = ['conservative', 'aggressive', 'swing']
for s in strategies:
    df = pd.read_parquet(f'experiments/{s}_jan2025/results/results.parquet')
    best = df.loc[df['total_profit'].idxmax()]
    print(f'{s}: Best profit={best[\"total_profit\"]:.6f} at threshold={best[\"threshold\"]:.2f}')
"
```

## Adding New Label Configurations

Simply add a new named entry to your config:

```yaml
labels:
  conservative:
    pt: 0.001
    sl: 0.001
    max_hold: 24
  
  scalping:      # New strategy
    pt: 0.0005   # Tight 0.05% target
    sl: 0.0003   # Tight 0.03% stop
    max_hold: 4  # Very short 4 hour hold
  
  # ... other strategies
```

Then regenerate data and train:

```bash
# Regenerate with new label config
python mainoffset.py --config config_example.yaml ...

# Train with new strategy
python train_forex.py --config config_example.yaml --label-name scalping --experiment scalping_test
```

## Backwards Compatibility

The system still supports the old list format for compatibility:

```yaml
# Old format (still works but generates warnings)
labels:
  - [0.001, 0.001, 24]
  - [0.002, 0.001, 48]
```

These will be automatically converted to `config_0`, `config_1`, etc.

## Tips

1. **Name your experiments after the strategy**: Use `--experiment conservative_MMYYYY` for clarity
2. **Generate all labels once**: Run `mainoffset.py` once to create all label files, then train multiple strategies
3. **Compare apples to apples**: Use the same date range for all strategy experiments
4. **Check metadata**: Look at `experiment_metadata.json` to verify which label config was used
5. **Avoid regenerating unnecessarily**: Once label files exist, you can train different strategies without regenerating
