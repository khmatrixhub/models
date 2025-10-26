"""
Shared feature engineering for FX LTR strategies.

This module provides modular feature blocks that can be used by both
LightGBM and XGBoost implementations. Features are designed for 60-min
rebalancing with intraday momentum focus.

Usage:
    from fx_ltr_features import calculate_features, build_universe_snapshot
    
    # Single pair features
    features = calculate_features(df, pair)
    
    # Cross-sectional features across all pairs
    snapshot = build_universe_snapshot(all_data, idx)
    snapshot_with_cs = apply_cross_sectional_blocks(snapshot)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable


# =============================================================================
# COLUMN NAME CONFIGURATION
# =============================================================================

BID_COL = 'bid_usd'      # last best bid
ASK_COL = 'ask_usd'      # last best offer
OPEN_COL = 'open_usd'
HIGH_COL = 'high_usd'
LOW_COL = 'low_usd'
CLOSE_COL = 'close_usd'
TIME_COL = 'Datetime'     # tz-aware or naive ok (we only use .hour/.minute/.dayofweek)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _safe_len(df, w):
    """Check if dataframe has at least w rows."""
    return len(df) >= w


def _last_dt(df):
    """Get last datetime as pandas Timestamp."""
    dt = df[TIME_COL].iloc[-1]
    return pd.to_datetime(dt) if isinstance(dt, str) else dt


def _mid_series(df):
    """Calculate mid price series from bid/ask or use close."""
    if {BID_COL, ASK_COL}.issubset(df.columns):
        return (df[BID_COL] + df[ASK_COL]) * 0.5
    return df[CLOSE_COL]


def _ret(s):
    """Calculate log returns."""
    return np.log(s).diff()


# =============================================================================
# FEATURE BLOCKS (Modular components)
# =============================================================================

def block_spread_micro(df: pd.DataFrame, pair: str) -> Dict[str, float]:
    """
    Spread and microstructure features.
    
    Features:
    - spread_bps: Bid-ask spread in basis points
    - close_minus_mid_bps: How far close is from mid (execution quality proxy)
    """
    f = {'spread_bps': np.nan, 'close_minus_mid_bps': np.nan}
    
    mid = _mid_series(df)
    if {BID_COL, ASK_COL}.issubset(df.columns):
        spr = df[ASK_COL].iloc[-1] - df[BID_COL].iloc[-1]
        f['spread_bps'] = float(1e4 * spr / mid.iloc[-1])  # ≈ bps on price
        f['close_minus_mid_bps'] = float(1e4 * (df[CLOSE_COL].iloc[-1] - mid.iloc[-1]) / mid.iloc[-1])
    
    return f


def block_intraday_mom(df: pd.DataFrame, pair: str) -> Dict[str, float]:
    """
    Intraday momentum features aligned to 60-min rebalance.
    All USD-normalized via mid price.
    
    Features:
    - ret_{w}m: Log return over w minutes (5, 15, 30, 60)
    - zret_{w}m: Z-scored return over w minutes
    - slope_10: Linear slope of last 10 bars (trend strength)
    """
    f = {}
    mid = _mid_series(df)
    
    for w in (5, 15, 30, 60):  # 5-60 min lookbacks
        if not _safe_len(df, w + 1):
            f[f'ret_{w}m'] = np.nan
            f[f'zret_{w}m'] = np.nan
        else:
            # Log return
            r = float(np.log(mid.iloc[-1]) - np.log(mid.iloc[-1 - w]))
            f[f'ret_{w}m'] = r
            
            # Z-scored return (relative to recent window)
            win = np.log(mid.iloc[-w-1:-1])
            mu, sd = win.mean(), win.std()
            f[f'zret_{w}m'] = float((np.log(mid.iloc[-1]) - mu) / (sd if sd > 0 else 1.0))
    
    # Short-term trend slope
    if _safe_len(df, 10):
        y = np.log(mid.iloc[-10:]).values
        x = np.arange(len(y))
        b = np.polyfit(x, y, 1)[0]
        f['slope_10'] = float(b)
    else:
        f['slope_10'] = np.nan
    
    return f


def block_breakout_rolling(df: pd.DataFrame, pair: str) -> Dict[str, float]:
    """
    Breakout and price location features.
    
    Features:
    - donch_{w}m: Donchian channel position (0=low, 1=high) over w minutes
    - clv_1: Close location value for last bar ((C-L)-(H-C))/(H-L)
    """
    f = {}
    
    # Donchian breakout % over 30 and 60 min
    for w in (30, 60):
        if not _safe_len(df, w):
            f[f'donch_{w}m'] = np.nan
        else:
            hi = df[HIGH_COL].iloc[-w:].max()
            lo = df[LOW_COL].iloc[-w:].min()
            cl = df[CLOSE_COL].iloc[-1]
            f[f'donch_{w}m'] = float((cl - lo) / (hi - lo + 1e-12))
    
    # Close location value for last bar (HLC)
    H = df[HIGH_COL].iloc[-1]
    L = df[LOW_COL].iloc[-1]
    C = df[CLOSE_COL].iloc[-1]
    f['clv_1'] = float(((C - L) - (H - C)) / (H - L + 1e-12))
    
    return f


def block_garman_klass_vol(df: pd.DataFrame, pair: str) -> Dict[str, float]:
    """
    Intraday realized volatility proxies.
    
    Features:
    - gk_vol_60m: Garman-Klass volatility over last 60 bars
    - rv_60m: Simple realized volatility (std of returns) over 60 bars
    """
    f = {'gk_vol_60m': np.nan, 'rv_60m': np.nan}
    
    if _safe_len(df, 60):
        # Garman-Klass per bar: 0.5*ln(H/L)^2 - (2*ln(2)-1)*ln(C/O)^2
        gk = 0.5 * np.log(df[HIGH_COL] / df[LOW_COL])**2 - \
             (2 * np.log(2) - 1) * np.log(df[CLOSE_COL] / df[OPEN_COL])**2
        f['gk_vol_60m'] = float(np.sqrt(gk.iloc[-60:].sum()))
        
        # Simple realized vol
        mid = _mid_series(df)
        rets = _ret(mid).dropna()
        if len(rets) >= 60:
            f['rv_60m'] = float(rets.iloc[-60:].std())
    
    return f


def block_rebalance_clock(df: pd.DataFrame, pair: str) -> Dict[str, float]:
    """
    Time features aligned to hourly rebalances.
    
    Features:
    - bar_in_hour: Minutes into current hour (0-59)
    - mins_to_rebalance: Minutes until next hour (59-0)
    - hour_sin/hour_cos: Cyclical hour encoding
    - dow: Day of week (0=Monday, 6=Sunday)
    """
    dt = _last_dt(df)
    minute = int(dt.minute)
    
    f = {
        'bar_in_hour': float(minute),                      # 0..59
        'mins_to_rebalance': float((60 - minute) % 60),   # 59..0
        'hour_sin': float(np.sin(2 * np.pi * dt.hour / 24)),
        'hour_cos': float(np.cos(2 * np.pi * dt.hour / 24)),
        'dow': float(dt.dayofweek),
    }
    
    return f


def block_pair_characteristics(df: pd.DataFrame, pair: str) -> Dict[str, float]:
    """
    Static pair characteristics.
    
    Features:
    - is_usd_inverse: 1 if pair starts with USD (USDJPY, USDCAD, etc.)
    - is_major: 1 if G7 major pair
    - is_em: 1 if emerging market pair
    """
    f = {
        'is_usd_inverse': 1.0 if pair.startswith('USD') else 0.0,
        'is_major': 1.0 if pair in ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF'] else 0.0,
        'is_em': 1.0 if pair in ['USDMXN', 'USDZAR', 'USDTRY'] else 0.0,
    }
    
    return f


def block_legacy_features(df: pd.DataFrame, pair: str) -> Dict[str, float]:
    """
    Legacy features from original implementation for backwards compatibility.
    These are longer-term features (30, 60, 120 bars).
    
    Features:
    - returns_{w}: Simple returns over w bars
    - log_returns_{w}: Log returns over w bars
    - volatility_{w}: Rolling volatility over w bars
    - high_low_range_{w}: Average H-L range over w bars
    - zscore_{w}: Z-score of price over w bars
    """
    f = {}
    
    # Use close_usd for consistency with original implementation
    close_usd = df[CLOSE_COL]
    
    # Momentum features
    for w in [30, 60, 120]:
        if len(df) < w:
            f[f'returns_{w}'] = np.nan
            f[f'log_returns_{w}'] = np.nan
        else:
            f[f'returns_{w}'] = (close_usd.iloc[-1] / close_usd.iloc[-w]) - 1
            f[f'log_returns_{w}'] = np.log(close_usd.iloc[-1] / close_usd.iloc[-w])
    
    # Volatility features
    for w in [30, 60]:
        if len(df) < w:
            f[f'volatility_{w}'] = np.nan
            f[f'high_low_range_{w}'] = np.nan
        else:
            returns = close_usd.iloc[-w:].pct_change().dropna()
            f[f'volatility_{w}'] = returns.std()
            
            if {HIGH_COL, LOW_COL}.issubset(df.columns):
                high_low = (df[HIGH_COL].iloc[-w:] - df[LOW_COL].iloc[-w:]) / close_usd.iloc[-w:]
                f[f'high_low_range_{w}'] = high_low.mean()
            else:
                f[f'high_low_range_{w}'] = 0
    
    # Z-score features
    for w in [30, 60, 120]:
        if len(df) < w:
            f[f'zscore_{w}'] = np.nan
        else:
            prices = close_usd.iloc[-w:]
            mean = prices.mean()
            std = prices.std()
            if std == 0 or pd.isna(std):
                f[f'zscore_{w}'] = 0
            else:
                f[f'zscore_{w}'] = (close_usd.iloc[-1] - mean) / std
    
    return f


# =============================================================================
# FEATURE BLOCK REGISTRY
# =============================================================================

# Default feature blocks (used by calculate_features)
DEFAULT_FEATURE_BLOCKS: List[Callable] = [
    block_spread_micro,
    block_intraday_mom,
    block_breakout_rolling,
    block_garman_klass_vol,
    block_rebalance_clock,
    block_pair_characteristics,
    block_legacy_features,  # Include legacy for backwards compatibility
]


# =============================================================================
# MAIN FEATURE CALCULATION
# =============================================================================

def calculate_features(df: pd.DataFrame, pair: str, 
                      feature_blocks: List[Callable] = None) -> Dict[str, float]:
    """
    Calculate all features for a single pair at current point in time.
    
    Uses modular feature blocks that can be customized.
    All features use USD-normalized prices for consistent cross-sectional comparison.
    
    Args:
        df: DataFrame with OHLC data and USD-normalized columns
        pair: Currency pair name (e.g., 'EURUSD', 'USDJPY')
        feature_blocks: Optional list of feature block functions to use.
                       If None, uses DEFAULT_FEATURE_BLOCKS.
    
    Returns:
        Dictionary of feature name -> value
    """
    if feature_blocks is None:
        feature_blocks = DEFAULT_FEATURE_BLOCKS
    
    # Need minimum history for most features
    if len(df) < 10:
        return {}
    
    # Collect features from all blocks
    features = {}
    for block_func in feature_blocks:
        try:
            block_features = block_func(df, pair)
            features.update(block_features)
        except Exception as e:
            # Log warning but continue with other blocks
            print(f"Warning: Feature block {block_func.__name__} failed for {pair}: {e}")
            continue
    
    return features


# =============================================================================
# CROSS-SECTIONAL FEATURES
# =============================================================================

def build_universe_snapshot(all_data: Dict[str, pd.DataFrame], idx: int,
                            feature_blocks: List[Callable] = None) -> pd.DataFrame:
    """
    Build snapshot of features for all pairs at given timestamp.
    
    This creates a DataFrame with one row per pair, containing all per-pair features.
    Used as input for cross-sectional transformations.
    
    Args:
        all_data: Dictionary of pair -> DataFrame
        idx: Bar index (current timestamp)
        feature_blocks: Optional list of feature block functions
    
    Returns:
        DataFrame with one row per pair, columns are features
    """
    snapshots = []
    
    for pair, df in all_data.items():
        if idx >= len(df):
            continue
        
        # Calculate features for this pair
        features = calculate_features(df.iloc[:idx+1], pair, feature_blocks)
        
        if features:
            features['pair'] = pair
            snapshots.append(features)
    
    if not snapshots:
        return pd.DataFrame()
    
    return pd.DataFrame(snapshots).set_index('pair')


def apply_cross_sectional_blocks(df_cs: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cross-sectional transformations to universe snapshot.
    
    Creates relative features across all pairs at the same timestamp:
    - Z-scores (cross-sectional standardization)
    - Percentile ranks (0-1)
    - Risk-adjusted momentum
    - Neutralized features (remove USD bias)
    
    Args:
        df_cs: DataFrame from build_universe_snapshot (one row per pair)
    
    Returns:
        Same DataFrame with additional cross-sectional features
    """
    if len(df_cs) < 2:
        return df_cs  # Need at least 2 pairs for cross-sectional
    
    # Select momentum columns (if they exist)
    mom_cols = [c for c in ['ret_5m', 'ret_15m', 'ret_30m', 'ret_60m', 'zret_60m', 'slope_10',
                             'returns_30', 'returns_60', 'returns_120'] 
                if c in df_cs.columns]
    
    # Select volatility/risk columns (if they exist)
    vol_cols = [c for c in ['gk_vol_60m', 'rv_60m', 'spread_bps', 
                             'volatility_30', 'volatility_60']
                if c in df_cs.columns]
    
    # Z-score and ranks for momentum and volatility
    for c in mom_cols + vol_cols:
        mu = df_cs[c].mean()
        sd = df_cs[c].std()
        
        # Cross-sectional z-score
        df_cs[f'cs_z_{c}'] = (df_cs[c] - mu) / (sd if sd > 0 else 1.0)
        
        # Percentile rank (0-1)
        df_cs[f'cs_rank_{c}'] = df_cs[c].rank(pct=True, method='first')
    
    # Risk-adjusted momentum + rank
    if {'ret_30m', 'rv_60m'}.issubset(df_cs.columns):
        x = df_cs['ret_30m']
        v = df_cs['rv_60m'].replace(0, np.nan)
        df_cs['mom30_over_rv60'] = x / v
        df_cs['cs_rank_mom30_over_rv60'] = df_cs['mom30_over_rv60'].rank(pct=True, method='first')
    elif {'returns_30', 'volatility_60'}.issubset(df_cs.columns):
        # Legacy version
        x = df_cs['returns_30']
        v = df_cs['volatility_60'].replace(0, np.nan)
        df_cs['mom30_over_vol60'] = x / v
        df_cs['cs_rank_mom30_over_vol60'] = df_cs['mom30_over_vol60'].rank(pct=True, method='first')
    
    # Optional neutralization against USD orientation
    xcols = [c for c in ['is_usd_inverse', 'is_major'] if c in df_cs.columns]
    if xcols:
        # Create design matrix with intercept
        X = pd.concat([
            pd.Series(1.0, index=df_cs.index, name='_c'),
            df_cs[xcols]
        ], axis=1).values
        
        # Neutralize key momentum features
        for y in [c for c in ['ret_60m', 'mom30_over_rv60', 'returns_60', 'mom30_over_vol60'] 
                  if c in df_cs.columns]:
            yv = df_cs[y].values
            
            try:
                # Regress out the USD/major effects
                beta = np.linalg.lstsq(X, yv, rcond=None)[0]
                resid = yv - X @ beta
            except Exception:
                # Fallback to simple demeaning
                resid = yv - np.nanmean(yv)
            
            df_cs[f'neu_{y}'] = resid
            df_cs[f'cs_rank_neu_{y}'] = pd.Series(resid, index=df_cs.index).rank(pct=True, method='first')
    
    return df_cs


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_feature_names(feature_blocks: List[Callable] = None) -> List[str]:
    """
    Get list of all feature names that will be generated.
    
    Useful for model initialization and debugging.
    
    Args:
        feature_blocks: Optional list of feature block functions
    
    Returns:
        List of feature names
    """
    if feature_blocks is None:
        feature_blocks = DEFAULT_FEATURE_BLOCKS
    
    # Create dummy data to extract feature names
    dummy_df = pd.DataFrame({
        TIME_COL: pd.date_range('2025-01-01', periods=150, freq='1min'),
        OPEN_COL: 1.0,
        HIGH_COL: 1.01,
        LOW_COL: 0.99,
        CLOSE_COL: 1.0,
        BID_COL: 0.999,
        ASK_COL: 1.001,
    })
    
    features = calculate_features(dummy_df, 'EURUSD', feature_blocks)
    return list(features.keys())


def get_cross_sectional_feature_names(base_features: List[str]) -> List[str]:
    """
    Get list of cross-sectional feature names that will be added.
    
    Args:
        base_features: List of base feature names from calculate_features
    
    Returns:
        List of additional cross-sectional feature names
    """
    cs_features = []
    
    # Momentum and volatility base columns
    mom_cols = [c for c in base_features if any(x in c for x in ['ret_', 'returns_', 'slope_', 'zret_'])]
    vol_cols = [c for c in base_features if any(x in c for x in ['vol_', 'volatility_', 'spread_'])]
    
    # Z-scores and ranks
    for c in mom_cols + vol_cols:
        cs_features.append(f'cs_z_{c}')
        cs_features.append(f'cs_rank_{c}')
    
    # Risk-adjusted
    if any('ret_30m' in c or 'returns_30' in c for c in base_features):
        cs_features.extend(['mom30_over_rv60', 'cs_rank_mom30_over_rv60'])
    
    # Neutralized
    if 'is_usd_inverse' in base_features:
        for c in [x for x in base_features if 'ret_60m' in x or 'returns_60' in x or 'mom30' in x]:
            cs_features.append(f'neu_{c}')
            cs_features.append(f'cs_rank_neu_{c}')
    
    return cs_features


if __name__ == '__main__':
    # Demo usage
    print("Available feature blocks:")
    for i, block in enumerate(DEFAULT_FEATURE_BLOCKS, 1):
        print(f"{i}. {block.__name__}")
    
    print(f"\nTotal features generated: {len(get_feature_names())}")
    print("\nFeature names:")
    for name in sorted(get_feature_names()):
        print(f"  - {name}")
