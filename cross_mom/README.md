# Cross-Sectional Momentum Model

This folder contains the production code for cross-sectional momentum strategies in FX markets.

## Contents

### Core Implementation
- `fx_backtest_base.py`: Shared backtesting framework with unified PnL calculation
- `fx_ltr.py`: Learn-to-Rank (LTR) strategy using shared framework
- `fx_ltr_strategy.py`: LTR strategy interface implementation
- `fx_simple_momentum.py`: Simple momentum strategy using shared framework
- `fx_cross_momentum.py`: Legacy simple cross-sectional momentum strategy
- `fx_ltr_momentum.py`: Legacy standalone LTR implementation (baseline)

### Tests
- `tests/`: Unit tests for core functionality
- `run_tests.py`: Test runner for all unit tests

## Testing

### Run all tests
```bash
python cross_mom/run_tests.py
```

### Run specific test
```bash
python cross_mom/tests/test_calculate_pnl.py
```

See `tests/README.md` for detailed test documentation.

## Description

This is the canonical workspace for research and production development of cross-sectional momentum models. All future work should be done here instead of the scratch or working_files folders.

### Shared Framework Architecture
The codebase uses a shared backtesting framework (`fx_backtest_base.py`) that provides:
- Unified position management
- Consistent PnL calculation using `pnl_usd = direction * base_notional * price_change_pct`
- Real bid/ask spread handling
- Gross vs Net PnL tracking

### Key Implementations

**fx_ltr.py** (Recommended): 
- Uses shared framework
- Learn-to-Rank model with LightGBM
- Real bid/ask data for accurate transaction costs
- Retrain on Saturdays for production deployment

**fx_simple_momentum.py**:
- Uses shared framework
- Simple interpretable momentum ranking
- Good baseline for comparison

**fx_ltr_momentum.py** (Legacy baseline):
- Standalone implementation
- Kept for validation and comparison
- Uses estimated PAIR_SPREADS (less accurate)

## Usage

Run each script with `--help` for command-line options. Data should be organized as described in the project root documentation.

### Example
```bash
# Run LTR strategy on Feb-June 2025
python cross_mom/fx_ltr.py \
  --start-date 02032025 \
  --end-date 06112025 \
  --rebalance-freq 60 \
  --top-n 2 \
  --training-days 30 \
  --retrain-frequency 7
```

---

*This folder is the production workspace for cross-sectional momentum research and development.*
