# Training Script Integration with New Folder Structure

## Overview

`train_forex.py` now works seamlessly with the new `results/<config_name>/` folder structure created by `mainoffset.py`.

## Updated Workflow

### 1. Data Processing (mainoffset.py)
```bash
python mainoffset.py --config config_example.yaml --start-date 01022025 --end-date 01312025
```

Creates:
```
results/config_example/
├── data/                    # ← train_forex reads from here
│   ├── 20250102_USDJPY_features.parquet
│   ├── 20250102_USDJPY_0.001_0.001_24_y.parquet
│   └── ...
├── signals/
│   └── ...
└── ...
```

### 2. Model Training (train_forex.py)
```bash
python train_forex.py --config config_example.yaml --experiment my_model
```

Reads from and writes to:
```
results/config_example/
├── data/                    # ← Reads features/labels
├── logs/                    # ← Adds training logs
│   └── train_forex_my_model_0001.log
├── models/                  # ← Saves trained models
│   └── my_model/
│       ├── results/
│       │   ├── config.yaml
│       │   ├── experiment_metadata.json
│       │   ├── feature_importance.csv
│       │   └── predictions.csv
│       └── model_20250115.pkl
└── predictions/             # ← Saves predictions (future)
    └── my_model/
        └── ...
```

## Key Changes in train_forex.py

### Before (Old Structure)
```python
data_dir = base_dir / "output"                           # ❌ Old location
results_dir = base_dir / "experiments" / experiment / "results"
models_dir = base_dir / "experiments" / experiment / "models"
```

### After (New Structure)
```python
config_name = config_path.stem                          # e.g., "config_example"
base_results_dir = Path("results") / config_name        # e.g., "results/config_example"

data_dir = base_results_dir / "data"                    # ✅ Where mainoffset.py saves data
models_dir = base_results_dir / "models" / experiment   # ✅ Organized by experiment
results_dir = models_dir / "results"                    # ✅ Results within experiment
```

## Complete Folder Structure

```
results/
  └── config_example/                     # From config_example.yaml
        ├── logs/                         # Logs from both scripts
        │   ├── feature_001.log           # From mainoffset.py
        │   └── train_forex_my_model_0001.log  # From train_forex.py
        │
        ├── signals/                      # From mainoffset.py
        │   └── 20250102_USDJPY_signals.parquet
        │
        ├── features/                     # From mainoffset.py (intermediate)
        │   └── ...
        │
        ├── data/                         # From mainoffset.py
        │   ├── 20250102_USDJPY_features.parquet     # ← train_forex reads
        │   └── 20250102_USDJPY_0.001_0.001_24_y.parquet
        │
        ├── models/                       # From train_forex.py
        │   └── my_model/                 # Experiment name
        │       ├── results/
        │       │   ├── config.yaml
        │       │   ├── experiment_metadata.json
        │       │   ├── feature_importance.csv
        │       │   └── predictions.csv
        │       └── model_20250115.pkl
        │
        └── predictions/                  # From train_forex.py (future)
            └── my_model/
                └── ...
```

## Running Multiple Experiments

### Same config, different experiments
```bash
# Train with default settings
python train_forex.py --config config_example.yaml --experiment baseline

# Train with feature selection
python train_forex.py --config config_example.yaml --experiment with_rfe

# Train conservative strategy only
python train_forex.py --config config_example.yaml --experiment conservative --label-name conservative
```

Creates:
```
results/config_example/
  └── models/
      ├── baseline/
      ├── with_rfe/
      └── conservative/
```

### Different configs (USDJPY vs EURUSD)
```bash
# Process and train USDJPY
python mainoffset.py --config config_usdjpy.yaml --start-date 01022025 --end-date 01312025
python train_forex.py --config config_usdjpy.yaml --experiment baseline

# Process and train EURUSD
python mainoffset.py --config config_eurusd.yaml --start-date 01022025 --end-date 01312025
python train_forex.py --config config_eurusd.yaml --experiment baseline
```

Creates:
```
results/
  ├── config_usdjpy/
  │   ├── data/
  │   └── models/
  │       └── baseline/
  └── config_eurusd/
      ├── data/
      └── models/
          └── baseline/
```

## Benefits

1. **Automatic Integration**: train_forex.py automatically finds data in the right place
2. **Isolated Experiments**: Each config gets its own models folder
3. **Multiple Experiments**: Run different training strategies within same config
4. **Clean Logs**: Training logs go to the same logs/ folder as data processing
5. **Easy Comparison**: Compare results across configs and experiments easily

## Migration Notes

If you have old training results in `experiments/`, they won't be affected. The new structure only applies to new training runs using the updated `train_forex.py`.

To use the new structure:
1. Re-run `mainoffset.py` with your config to generate data in new location
2. Run `train_forex.py` with same config - it will automatically find the data
3. Your trained models and results will be organized under `results/<config_name>/models/`
