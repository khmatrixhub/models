"""
Cross-sectional feature blocks for FX LTR strategy.

These features compute relative statistics across pairs at each timestamp,
capturing cross-sectional momentum, risk-adjusted returns, liquidity penalties,
and neutralized signals to avoid structural biases.

Usage:
    # At each rebalance, build snapshot of all pairs
    snapshot_df = pd.DataFrame([features_dict1, features_dict2, ...])
    
    # Apply cross-sectional transformations
    snapshot_df = apply_cross_sectional_blocks(
        snapshot_df, 
        prev_ranks=prev_hour_ranks
    )
    
    # Use cs_* features for ranking or as model inputs
"""

import pandas as pd
import numpy as np
from typing import Optional


# =============================================================================
# Cross-sectional utilities
# =============================================================================

def _cs_z(df: pd.DataFrame, col: str, out: str) -> pd.DataFrame:
    """
    Compute cross-sectional z-score for a column.
    
    Args:
        df: DataFrame with all pairs at current timestamp
        col: Column to z-score
        out: Output column name
        
    Returns:
        DataFrame with added z-score column
    """
    mu = df[col].mean()
    sd = df[col].std()
    df[out] = (df[col] - mu) / (sd if sd > 0 else 1.0)
    return df


def _winsor(s: pd.Series, q: float = 0.01) -> pd.Series:
    """
    Winsorize series at specified quantiles to reduce outlier impact.
    
    Args:
        s: Series to winsorize
        q: Quantile threshold (default 1%)
        
    Returns:
        Winsorized series
    """
    lo = s.quantile(q)
    hi = s.quantile(1 - q)
    return s.clip(lo, hi)


# =============================================================================
# Cross-sectional feature blocks
# =============================================================================

def cs_block_basic_stats(df_cs: pd.DataFrame) -> pd.DataFrame:
    """
    Z-scores & percentile ranks for momentum/volatility/liquidity.
    
    Creates cs_rank_* and cs_z_* features for:
    - Momentum: ret_5m, ret_15m, ret_30m, ret_60m, zret_60m, slope_10
    - Volatility: rv_60m, gk_vol_60m
    - Liquidity: spread_bps
    
    Args:
        df_cs: DataFrame with all pairs at current timestamp
        
    Returns:
        DataFrame with added cross-sectional rank and z-score features
    """
    # Momentum features
    mom = [c for c in ['ret_5m', 'ret_15m', 'ret_30m', 'ret_60m', 'zret_60m', 'slope_10'] 
           if c in df_cs.columns]
    
    # Volatility features
    vol = [c for c in ['rv_60m', 'gk_vol_60m'] 
           if c in df_cs.columns]
    
    # Liquidity features
    liq = [c for c in ['spread_bps'] 
           if c in df_cs.columns]
    
    # Compute ranks and z-scores for all features
    for c in mom + vol + liq:
        # Percentile rank (0 to 1)
        df_cs[f'cs_rank_{c}'] = df_cs[c].rank(pct=True, method='first')
        
        # Z-score (standardized)
        df_cs = _cs_z(df_cs, c, f'cs_z_{c}')
    
    return df_cs


def cs_block_risk_adjusted_mom(df_cs: pd.DataFrame) -> pd.DataFrame:
    """
    Risk-adjusted momentum (Sharpe-like ratio) + ranks.
    
    Computes momentum / volatility ratio to reward high returns per unit risk.
    Creates: mom30_over_rv60, cs_rank_mom30_over_rv60, cs_z_mom30_over_rv60
    
    Args:
        df_cs: DataFrame with all pairs at current timestamp
        
    Returns:
        DataFrame with added risk-adjusted momentum features
    """
    if {'ret_30m', 'rv_60m'}.issubset(df_cs.columns):
        x = df_cs['ret_30m']
        v = df_cs['rv_60m'].replace(0, np.nan)  # Avoid division by zero
        
        # Risk-adjusted momentum
        df_cs['mom30_over_rv60'] = x / v
        
        # Rank and z-score
        df_cs['cs_rank_mom30_over_rv60'] = df_cs['mom30_over_rv60'].rank(pct=True, method='first')
        df_cs = _cs_z(df_cs, 'mom30_over_rv60', 'cs_z_mom30_over_rv60')
    
    return df_cs


def cs_block_liquidity_penalty(df_cs: pd.DataFrame) -> pd.DataFrame:
    """
    Penalize wide spreads (lower spread = higher score).
    
    Creates liquidity score where 1.0 = tightest spread, 0.0 = widest spread.
    This can be used to penalize pairs with high transaction costs.
    
    Args:
        df_cs: DataFrame with all pairs at current timestamp
        
    Returns:
        DataFrame with added liq_score column
    """
    if 'spread_bps' in df_cs.columns:
        # Rank ascending (tight spread = high rank)
        r = df_cs['spread_bps'].rank(pct=True, ascending=True, method='first')
        df_cs['liq_score'] = 1.0 - r  # Invert so tight = 1, wide = 0
    else:
        df_cs['liq_score'] = 0.5  # Neutral if no spread data
    
    return df_cs


def cs_block_winsorize(df_cs: pd.DataFrame) -> pd.DataFrame:
    """
    Winsorize key noisy columns before using in composite features.
    
    Clips extreme values at 1% and 99% quantiles to reduce outlier impact.
    Creates win_* versions of key features.
    
    Args:
        df_cs: DataFrame with all pairs at current timestamp
        
    Returns:
        DataFrame with added winsorized feature columns
    """
    cols = [c for c in ['ret_60m', 'ret_30m', 'rv_60m', 'gk_vol_60m', 
                        'spread_bps', 'mom30_over_rv60'] 
            if c in df_cs.columns]
    
    for c in cols:
        df_cs[f'win_{c}'] = _winsor(df_cs[c], q=0.01)
    
    return df_cs


def cs_block_session_dummies(df_cs: pd.DataFrame) -> pd.DataFrame:
    """
    FX session indicators from hour_of_day: Asia/EU/US.
    
    Creates binary features indicating which trading session is active:
    - sess_asia: 00:00-08:00 EST (Asia/Pacific)
    - sess_eu: 08:00-14:00 EST (European)
    - sess_us: 14:00-22:00 EST (US)
    
    Args:
        df_cs: DataFrame with all pairs at current timestamp
        
    Returns:
        DataFrame with added session indicator columns
    """
    if 'hour_of_day' in df_cs.columns:
        h = df_cs['hour_of_day']
        df_cs['sess_asia'] = ((h >= 0) & (h < 8)).astype(float)
        df_cs['sess_eu'] = ((h >= 8) & (h < 14)).astype(float)
        df_cs['sess_us'] = ((h >= 14) & (h < 22)).astype(float)
    else:
        df_cs[['sess_asia', 'sess_eu', 'sess_us']] = 0.0
    
    return df_cs


def cs_block_neutralize(df_cs: pd.DataFrame) -> pd.DataFrame:
    """
    Neutralize momentum vs structural flags to avoid USD/major pair bias.
    
    Uses OLS regression to remove systematic relationships between returns
    and structural pair characteristics (is_usd_inverse, is_major).
    This helps isolate idiosyncratic pair moves from structural effects.
    
    Creates: neu_ret_60m, neu_mom30_over_rv60 + their ranks
    
    Args:
        df_cs: DataFrame with all pairs at current timestamp
        
    Returns:
        DataFrame with added neutralized momentum features
    """
    # Features to use as controls
    xcols = [c for c in ['is_usd_inverse', 'is_major'] 
             if c in df_cs.columns]
    
    # Features to neutralize
    ycols = [c for c in ['ret_60m', 'mom30_over_rv60'] 
             if c in df_cs.columns]
    
    if not xcols or not ycols:
        return df_cs
    
    # Build design matrix (intercept + controls)
    X = pd.concat([
        pd.Series(1.0, index=df_cs.index, name='_const'),
        df_cs[xcols]
    ], axis=1).values
    
    # Neutralize each y-variable
    for y in ycols:
        yv = df_cs[y].values
        
        try:
            # OLS: y = X * beta + residual
            beta = np.linalg.lstsq(X, yv, rcond=None)[0]
            resid = yv - (X @ beta)
        except Exception:
            # Fallback: just demean
            resid = yv - np.nanmean(yv)
        
        # Store neutralized version
        df_cs[f'neu_{y}'] = resid
        
        # Rank neutralized values
        df_cs[f'cs_rank_neu_{y}'] = pd.Series(resid, index=df_cs.index).rank(
            pct=True, method='first'
        )
    
    return df_cs


def cs_block_composite_score(df_cs: pd.DataFrame) -> pd.DataFrame:
    """
    Simple composite score combining multiple signals.
    
    Weights:
    - Risk-adjusted momentum (mom30/rv60): 0.5
    - Raw 60m return: 0.3
    - Trend slope: 0.2
    - Liquidity bonus: 0.25
    
    Creates: cs_score_v1, cs_rank_score_v1
    
    Args:
        df_cs: DataFrame with all pairs at current timestamp
        
    Returns:
        DataFrame with added composite score and rank
    """
    parts = []
    
    # Risk-adjusted momentum (highest weight)
    if 'cs_z_mom30_over_rv60' in df_cs.columns:
        parts.append(0.5 * df_cs['cs_z_mom30_over_rv60'])
    
    # Raw momentum
    if 'cs_z_ret_60m' in df_cs.columns:
        parts.append(0.3 * df_cs['cs_z_ret_60m'])
    
    # Trend strength
    if 'cs_z_slope_10' in df_cs.columns:
        parts.append(0.2 * df_cs['cs_z_slope_10'])
    
    # Combine parts
    score = sum(parts) if parts else 0.0
    
    # Add liquidity bonus (tight spreads get boost)
    if 'liq_score' in df_cs.columns:
        score = score + 0.25 * df_cs['liq_score']
    
    df_cs['cs_score_v1'] = score
    df_cs['cs_rank_score_v1'] = df_cs['cs_score_v1'].rank(pct=True, method='first')
    
    return df_cs


def cs_block_turnover_guard(df_cs: pd.DataFrame, 
                            prev_ranks: Optional[pd.Series] = None, 
                            alpha: float = 0.7) -> pd.DataFrame:
    """
    Smooth ranks vs previous hour to reduce portfolio turnover.
    
    Uses exponential moving average of ranks to avoid excessive rebalancing.
    rank_t = alpha * rank_{t-1} + (1 - alpha) * rank_t
    
    Args:
        df_cs: DataFrame with all pairs at current timestamp
        prev_ranks: Series indexed by pair with previous cs_rank_score_v1 values
        alpha: Smoothing factor (0.7 = 70% weight on previous rank)
        
    Returns:
        DataFrame with added cs_rank_score_smooth column
    """
    if prev_ranks is None or 'pair' not in df_cs.columns or 'cs_rank_score_v1' not in df_cs.columns:
        return df_cs
    
    # Get previous ranks (default to 0.5 if new pair)
    pr = prev_ranks.reindex(df_cs['pair']).fillna(0.5).values
    
    # Get current ranks
    cr = df_cs['cs_rank_score_v1'].values
    
    # Smooth
    df_cs['cs_rank_score_smooth'] = alpha * pr + (1 - alpha) * cr
    
    return df_cs


# =============================================================================
# Main entry point
# =============================================================================

def apply_cross_sectional_blocks(df_cs: pd.DataFrame,
                                 prev_ranks: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Apply all cross-sectional feature transformations.
    
    This is the main entry point that chains all CS feature blocks together.
    
    Pipeline:
    1. Basic stats (z-scores and ranks for momentum/vol/liquidity)
    2. Risk-adjusted momentum (Sharpe-like ratios)
    3. Liquidity penalty (penalize wide spreads)
    4. Winsorize (clip outliers)
    5. Session dummies (Asia/EU/US indicators)
    6. Neutralize (remove USD/major bias)
    7. Composite score (combine signals)
    8. Turnover guard (smooth ranks to reduce churn)
    
    Args:
        df_cs: DataFrame with all pairs at current timestamp (one row per pair)
        prev_ranks: Optional Series with previous cs_rank_score_v1 values
        
    Returns:
        DataFrame with ~40 additional cross-sectional features
    """
    df_cs = cs_block_basic_stats(df_cs)
    df_cs = cs_block_risk_adjusted_mom(df_cs)
    df_cs = cs_block_liquidity_penalty(df_cs)
    df_cs = cs_block_winsorize(df_cs)
    df_cs = cs_block_session_dummies(df_cs)
    df_cs = cs_block_neutralize(df_cs)          # Creates neu_* + ranks
    df_cs = cs_block_composite_score(df_cs)     # cs_score_v1 + rank
    df_cs = cs_block_turnover_guard(df_cs, prev_ranks=prev_ranks, alpha=0.7)
    
    return df_cs


def get_cross_sectional_feature_names() -> list:
    """
    Get list of all cross-sectional features created by the blocks.
    
    Returns:
        List of feature names (approximately 40-50 features)
    """
    base_features = [
        'ret_5m', 'ret_15m', 'ret_30m', 'ret_60m', 'zret_60m', 'slope_10',
        'rv_60m', 'gk_vol_60m', 'spread_bps'
    ]
    
    cs_features = []
    
    # Basic stats: cs_rank_* and cs_z_* for each base feature
    for feat in base_features:
        cs_features.append(f'cs_rank_{feat}')
        cs_features.append(f'cs_z_{feat}')
    
    # Risk-adjusted momentum
    cs_features.extend([
        'mom30_over_rv60',
        'cs_rank_mom30_over_rv60',
        'cs_z_mom30_over_rv60'
    ])
    
    # Liquidity
    cs_features.append('liq_score')
    
    # Winsorized versions
    winsor_cols = ['ret_60m', 'ret_30m', 'rv_60m', 'gk_vol_60m', 
                   'spread_bps', 'mom30_over_rv60']
    cs_features.extend([f'win_{c}' for c in winsor_cols])
    
    # Session dummies
    cs_features.extend(['sess_asia', 'sess_eu', 'sess_us'])
    
    # Neutralized features
    cs_features.extend([
        'neu_ret_60m',
        'cs_rank_neu_ret_60m',
        'neu_mom30_over_rv60',
        'cs_rank_neu_mom30_over_rv60'
    ])
    
    # Composite scores
    cs_features.extend([
        'cs_score_v1',
        'cs_rank_score_v1',
        'cs_rank_score_smooth'
    ])
    
    return cs_features
