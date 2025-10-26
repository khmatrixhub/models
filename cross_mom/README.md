# Cross-Sectional Momentum Model

This folder contains the production code for cross-sectional momentum strategies in FX markets.

## Contents
- `fx_cross_momentum.py`: Simple cross-sectional momentum strategy (rank and trade top/bottom pairs).
- `fx_ltr_momentum.py`: Learn-to-Rank (LTR) model for cross-sectional FX momentum (synthetic spread).
- `fx_ltr_momentum_bidask.py`: LTR model using real bid/ask data for realistic transaction cost modeling.

## Description
This is the canonical workspace for research and production development of cross-sectional momentum models. All future work should be done here instead of the scratch or working_files folders.

- **fx_cross_momentum.py**: Implements a simple, interpretable momentum ranking and trading strategy.
- **fx_ltr_momentum.py**: Implements a machine learning (LightGBM) ranking model to predict and rank pairs for trading.
- **fx_ltr_momentum_bidask.py**: Same as above, but uses real market bid/ask data for more accurate backtesting.

## Usage
Run each script with `--help` for command-line options. Data should be organized as described in the project root documentation.

---

*This folder is the new production workspace for cross-sectional momentum research and development.*
