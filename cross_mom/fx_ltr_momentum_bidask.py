"""
Learn-to-Rank (LTR) Cross-sectional FX Momentum Strategy - BIDASK VERSION.

This version uses REAL bid/ask data from data/bidask/output/ instead of synthetic spreads.

Strategy:
- Train LightGBM ranking model every Saturday using past N days of data
- At each rebalance, use model to rank pairs by predicted next-hour PnL
- Go long top N ranked pairs, short bottom N ranked pairs (USD neutral)
- Features: momentum, volatility, z-score, time-of-day, pair characteristics
- USES REAL BID/ASK PRICES from market data

Key Differences from Simple Momentum:
- Models learn which feature combinations predict profitable trades
- Can capture non-linear patterns (momentum + low volatility, time effects, etc.)
- Walk-forward validation with weekly retraining
- Real market bid/ask spreads (not synthetic)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple
import lightgbm as lgb
import pickle

# ...existing code...
