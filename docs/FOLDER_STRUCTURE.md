# Results Folder Structure

## Overview

All outputs from `mainoffset.py` are now organized by config file name under the `results/` directory. Each experiment has a clean, organized structure with dedicated subfolders for different types of outputs.

## Directory Structure

```
results/
├── config_example/              # From config_example.yaml
│   ├── logs/                    # Processing logs
│   │   └── feature_001.log
│   │
│   ├── signals/                 # Trading signals (parquet)
│   │   ├── 20250102_USDJPY_signals.parquet
│   │   ├── 20250103_USDJPY_signals.parquet
│   │   └── ...
│   │
│   ├── features/                # Intermediate feature files (compressed CSV)
│   │   ├── tick_bars_20250102_USDJPY.csv.bz2
│   │   ├── features_20250102_USDJPY.csv.bz2
│   │   └── ...
│   │
│   ├── data/                    # Processed ML-ready data (parquet + CSV)
│   │   ├── 20250102_USDJPY_features.parquet
│   │   ├── 20250102_USDJPY_0.001_0.001_24_y.parquet
│   │   ├── 20250102_USDJPY_0.001_0.001_24_y.csv
│   │   └── ...
│   │
│   ├── models/                  # Trained models (for future use)
│   │   ├── model_conservative.pkl
│   │   ├── model_aggressive.pkl
│   │   └── ...
│   │
│   └── predictions/             # Model predictions (for future use)
│       ├── predictions_20250102.parquet
│       └── ...
│
├── config_eurusd/               # From config_eurusd.yaml
│   ├── logs/
│   ├── signals/
│   ├── features/
│   ├── data/
│   ├── models/
│   └── predictions/
│
└── my_test/                     # From my_test.yaml
    ├── logs/
    ├── signals/
    ├── features/
    ├── data/
    ├── models/
    └── predictions/
```

## Subdirectories Explained

### `logs/`
**Purpose:** Processing and debugging logs
- Numbered log files: `feature_001.log`, `feature_002.log`, etc.
- New log file created for each run
- Contains processing info, errors, statistics, and timing information

### `signals/`
**Purpose:** Trading signals at crossover points
- `YYYYMMDD_PAIR_signals.parquet` - Signal direction (1=long, -1=short)
- One file per day
- Used to understand when trades were triggered

### `features/`
**Purpose:** Intermediate compressed feature files
- `tick_bars_YYYYMMDD_PAIR.csv.bz2` - OHLC data at signal points
- `features_YYYYMMDD_PAIR.csv.bz2` - Technical indicators before filtering
- Compressed with bz2 for space efficiency
- Useful for debugging feature generation

### `data/`
**Purpose:** Final ML-ready datasets
- `YYYYMMDD_PAIR_features.parquet` - All features ready for training
- `YYYYMMDD_PAIR_PT_SL_HOLD_y.parquet` - Labels for each strategy
- `YYYYMMDD_PAIR_PT_SL_HOLD_y.csv` - Labels in CSV format
- This is what you'll load for model training

### `models/`
**Purpose:** Trained ML models (future use)
- Will store pickled scikit-learn models
- Model metadata and hyperparameters
- Feature importance rankings
- Currently empty, ready for training scripts

### `predictions/`
**Purpose:** Model predictions and backtesting results (future use)
- Will store prediction outputs
- Backtest performance metrics
- Trade-by-trade predictions
- Currently empty, ready for inference scripts

## File Naming Convention

All data files include:
- **Date**: `YYYYMMDD` format (e.g., `20250102` for Jan 2, 2025)
- **Pair**: Currency pair (e.g., `USDJPY`, `EURUSD`)
- **Strategy params**: `PT_SL_HOLD` for label files (e.g., `0.001_0.001_24`)

## Running Multiple Experiments

Keep experiments separate by creating different config files:

```bash
# USDJPY with conservative settings
python mainoffset.py --config config_conservative.yaml --start-date 01022025 --end-date 01312025

# EURUSD with aggressive settings  
python mainoffset.py --config config_eurusd_aggressive.yaml --start-date 01022025 --end-date 01312025

# Test run with small date range
python mainoffset.py --config test_debug.yaml --start-date 01022025 --end-date 01022025
```

Each creates its own isolated folder structure under `results/`:
- `results/config_conservative/`
- `results/config_eurusd_aggressive/`
- `results/test_debug/`

## Typical Workflow

1. **Data Processing** (mainoffset.py)
   - Reads from `data/output/YYYYMMDD_PAIR_bars.csv`
   - Generates signals, features, labels
   - Saves to `results/<config>/signals/`, `features/`, `data/`

2. **Model Training** (your training script)
   - Loads features from `results/<config>/data/`
   - Trains models
   - Saves to `results/<config>/models/`

3. **Prediction/Backtesting** (your inference script)
   - Loads models from `results/<config>/models/`
   - Generates predictions
   - Saves to `results/<config>/predictions/`

## Benefits

1. **Clean organization**: No files in root, everything categorized
2. **Isolated experiments**: Each config has its own complete folder structure
3. **Easy comparison**: Compare results from different configs side-by-side
4. **Future-ready**: Pre-created folders for models and predictions
5. **Clear workflow**: Logical progression from signals → features → data → models → predictions
6. **No overwrites**: Different configs never interfere with each other

## Migration from Old Structure

**Old structure:**
```
output/
  └── (all files mixed together)
features/
  └── (feature files)
logs/
  └── (logs)
```

**New structure:**
```
results/
  └── <config_name>/
      ├── logs/
      ├── signals/
      ├── features/
      ├── data/
      ├── models/
      └── predictions/
```

All folders are automatically created when you run `mainoffset.py` with a config file.
