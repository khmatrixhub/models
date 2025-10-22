# Copilot Instructions for Financial Data Processing Pipeline

## Project Overview
This is a financial market data processing pipeline for forex trading (EURUSD, USDJPY). The system generates trading signals, features, and labels for machine learning models using multiple trading strategies with time-lagged profit features.

## File Organization

### Production Code (Root Directory)
- `generate.py` - Data processing and feature generation
- `train.py` - Model training with walk-forward validation
- `config_*.yaml` - Configuration files

### Generated/Temporary Files
**IMPORTANT**: All temporary test scripts, analysis files, and exploration code should go in `working_files/`:
- `working_files/tests/` - Test scripts and validation code
- `working_files/analysis/` - Data analysis and exploration
- `working_files/temp/` - Temporary scratch files

This folder is in `.gitignore` to keep the repository clean. When creating test or analysis scripts during development, always save them here.

## Architecture & Data Flow

### Core Scripts
1. **`generate.py`** (formerly `mainoffset.py`) - Data processing and feature generation
   - Processes daily OHLC bars with time-based offsets
   - Generates technical features, signals, and labels
   - Outputs organized by config name into `results/<config_name>/`

2. **`train.py`** (formerly `train_forex.py`) - Model training
   - Walk-forward cross-validation
   - Multiple experiments per config
   - Reads from `results/<config_name>/data/`
   - Saves to `results/<config_name>/models/<experiment>/`

### Folder Structure (Config-Based Organization)
```
results/<config_name>/          # One folder per config file
  ├── data/                     # Features & labels (Parquet)
  ├── signals/                  # Trading signals (Parquet)
  ├── features/                 # Intermediate features (CSV.bz2)
  ├── logs/                     # Processing & training logs
  ├── models/                   # Trained models by experiment
  │   ├── baseline/
  │   ├── with_rfe/
  │   └── conservative/
  └── predictions/              # Future: prediction outputs
```

### Data Flow
```
data/output/YYYYMMDD_PAIR_bars.csv 
  → generate.py → results/<config_name>/{signals/, data/, features/}
  → train.py → results/<config_name>/models/<experiment>/
```

## Development Workflows

### Environment Setup
```bash
conda env create -f environment.yml
conda activate models
```

### Complete Workflow
```bash
# 1. Generate features and labels
python generate.py --config config_example.yaml --start-date 01022025 --end-date 01312025

# 2. Train models
python train.py --config config_example.yaml --experiment baseline

# 3. Run different experiments
python train.py --config config_example.yaml --experiment with_rfe --use-rfe
python train.py --config config_example.yaml --experiment conservative --label-name conservative
```

## Feature Engineering (75 Total Features)

### Technical Features (11 features)
- Moving averages: `ma_5`, `ma_20`, `ma_distance`
- Returns: `returns`, `log_returns`
- Volatility: `atr`, `bb_position`
- Momentum: `rsi`
- Volume: `volume`, `volume_ma_ratio`

### Signal-Adjusted Directional Features (4 features)
**IMPORTANT**: These are multiplied by signal direction for directional interpretation
- `returns_adj = returns * signal_direction`
- `log_returns_adj = log_returns * signal_direction`
- `ma5_distance_adj = (close - ma_5) / ma_5 * signal_direction`
- `ma20_distance_adj = (close - ma_20) / ma_20 * signal_direction`

**Interpretation**: Positive values = favorable for trade, negative = unfavorable (regardless of long/short)

### Bar Statistics (16 features)
- AFML volume and related metrics
- Volatility: `vol_skew`, `vol_kurt`, `vol_range`
- Jump detection and timing metrics

### Lagged Profit Features (44 features = 11 per strategy × 4 strategies)

**Strategy Prefixes**:
- `conservative_` - 0.001 PT/SL, 24h hold
- `aggressive_` - 0.002 PT/SL, 48h hold
- `swing_` - 0.005 PT/SL, 120h hold
- `macrossover_` - MA-based exits

**11 Features per Strategy**:
1. `{strategy}_profit_lag1/2/3/5` - Recent profit outcomes
2. `{strategy}_profit_rolling_mean_5/10` - Average recent performance
3. `{strategy}_profit_rolling_std_5` - Volatility of results
4. `{strategy}_winning_streak` - Current consecutive wins
5. `{strategy}_losing_streak` - Current consecutive losses
6. `{strategy}_recent_win_rate` - Win rate over last 5 trades
7. `{strategy}_profit_volatility` - Coefficient of variation

**CRITICAL**: 
- Profit values are already position-relative (positive = win, negative = loss)
- DO NOT multiply profit features by signal direction
- All lagged features use `.shift(1)` to prevent data leakage

## Configuration Structure

### YAML Config Format
```yaml
spot: "USDJPY"
offsets: [0, 1, 2, 3]
start_date: "01022025"
end_date: "01312025"

# Signal strategy selection
signal_strategy: "ma_crossover"  # or "bb_mean_reversion"
signal_params:
  bb_window: 20     # For BB strategy
  bb_std: 2.0
  use_rsi_filter: false

# Exit strategies (3 types available)
labels:
  # 1. Standard PT/SL exits
  conservative:
    pt: 0.001
    sl: 0.001
    max_hold: 24
    use_ma_exit: false
    use_bb_exit: false
  
  # 2. MA signal-based exits
  ma_crossover:
    pt: null
    sl: null
    max_hold: null
    use_ma_exit: true
  
  # 3. BB mean reversion exits (4 strategies)
  bb_middle:
    pt: 0.002              # Backup exit
    sl: 0.001
    max_hold: 24
    use_bb_exit: true
    bb_exit_strategy: "middle"  # middle | opposite_band | partial | signal
    bb_window: 20          # MUST match signal_params
    bb_std: 2.0
    exit_std_multiplier: 0.5  # For 'partial' strategy only
```

### Exit Strategy Types

**1. Standard PT/SL** (`use_ma_exit=false, use_bb_exit=false`)
- Traditional profit target and stop loss
- Fixed percentage exits
- Max hold period as backup

**2. MA Signal-Based** (`use_ma_exit=true`)
- Exit on opposite MA crossover signal
- No PT (set to null)
- Best for trending strategies

**3. BB Mean Reversion** (`use_bb_exit=true`)
- 4 strategies: `middle`, `opposite_band`, `partial`, `signal`
- **Middle**: Exit at MA line (fastest, highest win rate)
- **Opposite Band**: Exit at opposite band (slowest, highest profit)
- **Partial**: Exit at MA ± exit_std_multiplier*std (configurable)
- **Signal**: Exit on opposite BB signal (regime-based)
- PT/SL/max_hold still act as backup exits

## Key Patterns & Conventions

### Date Handling
- **Input format**: `%m%d%Y` (MMDDYYYY) via CLI
- **File format**: `%Y%m%d` (YYYYMMDD) for filenames
- **Processing**: Weekdays only (Monday-Friday)
- **Data timing**: 5PM EST (day N-1) → 4:59PM EST (day N), files named by end date

### File Naming Conventions
- **Input**: `data/output/YYYYMMDD_PAIR_bars.csv`
- **Signals**: `results/<config>/signals/YYYYMMDD_PAIR_signals.parquet`
- **Features**: `results/<config>/data/YYYYMMDD_PAIR_features.parquet`
- **Labels**: `results/<config>/data/YYYYMMDD_PAIR_PT_SL_HOLD_y.parquet`

### Parallel Processing
- Uses `joblib.Parallel` for date-based parallelization
- Each date processed independently via `process_date()`
- Default `n_jobs=-1` (all CPUs)

### Error Handling
```python
try:
    # Process date
except Exception as e:
    logging.exception(f"skipping {date} due to {e}")
```

## Important Implementation Details

### Feature Generation in generate.py

**add_lagged_profit_features()** (lines 80-169):
```python
def add_lagged_profit_features(features_df, labels_df, strategy_name='', lookback_days=5):
    """
    Creates 11 lagged profit features per strategy.
    Uses .shift(1) to prevent data leakage.
    Returns: (features_df, added_features_list)
    """
```

**add_signal_adjusted_features()** (lines 218-261):
```python
def add_signal_adjusted_features(features_df, signals):
    """
    Creates 4 directional features multiplied by signal.
    Only for price-based features, NOT profit features.
    """
```

**Label Generation Functions**:
- `generate_labels()` (lines 371-527): Standard PT/SL/max_hold exits
- `generate_labels_ma_exit()` (lines 530-662): MA signal-based exits
- `generate_labels_bb_exit()` (lines 664-847): BB mean reversion exits
  - Supports 4 exit strategies: 'middle', 'opposite_band', 'partial', 'signal'
  - Calculates BB dynamically for exit detection
  - PT/SL/max_hold act as backup exits
  - Returns labels with exit_reason column

**Multi-Strategy Loop** (lines 1015-1060):
```python
# Add lagged features for ALL strategies
for label_name, label_info in labels.items():
    label_df = label_info['labels']
    prefix = label_name.replace('_', '')  # ma_crossover → macrossover
    
    features_df, added_features = add_lagged_profit_features(
        features_df, label_df, strategy_name=prefix)
    all_lagged_features.extend(added_features)
```

### Path Resolution in train.py

**Config-Based Paths** (lines 330-377):
```python
config_name = config_path.stem  # config_example.yaml → config_example
base_results_dir = Path("results") / config_name

data_dir = base_results_dir / "data"              # Read features/labels
signals_dir = base_results_dir / "signals"        # Read signals
models_dir = base_results_dir / "models" / experiment
predictions_dir = base_results_dir / "predictions" / experiment
```

## Data Schema

### Input Market Data (OHLC Bars)
- `Datetime`, `open`, `high`, `low`, `close`
- `afmlvol` - AFML volume metric
- Volatility metrics: `vol_skew`, `vol_kurt`, `vol_range`
- Timing: `seconds_diff`, jump detection

### Generated Features (75 columns)
- 11 technical features
- 4 signal-adjusted features
- 16 bar statistics
- 44 lagged profit features (11 per strategy × 4 strategies)

### Labels (per strategy)
- Binary classification: 1 (profitable), -1 (unprofitable)
- Separate label file per strategy configuration

## Critical Don'ts

❌ **DO NOT**:
- Multiply profit features by signal (they're already position-relative)
- Use future data in feature generation (always use `.shift(1)`)
- Abbreviate strategy names in feature prefixes (use full names)
- Save files to root directories (use config-based subfolders)
- Mix data from different configs (each config gets isolated folder)

✅ **DO**:
- Use full strategy names: `conservative_`, `aggressive_`, `swing_`, `macrossover_`
- Keep profit features as-is (positive = win works for both long/short)
- Organize all outputs under `results/<config_name>/`
- Log all directory paths for debugging
- Process dates independently for parallelization

## Documentation Files
- `FOLDER_STRUCTURE.md` - Complete folder organization reference
- `TRAINING_INTEGRATION.md` - How train.py integrates with folder structure
- `BB_EXIT_STRATEGIES.md` - Comprehensive BB exit strategies guide
- `BB_EXIT_QUICKSTART.md` - Quick start guide for BB exits
- `BB_MEAN_REVERSION.md` - BB signal generation strategy
- `TRAIN_COMPATIBILITY.md` - Model training integration
- `environment.yml` - Conda environment specification

## Data Schema
Market data includes: `Datetime`, `open`, `high`, `low`, `close`, `afmlvol`, volatility metrics (`vol_skew`, `vol_kurt`, etc.), and timing (`seconds_diff`).