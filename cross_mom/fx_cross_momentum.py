"""
Cross-sectional momentum strategy for intraday FX trading - REAL BID/ASK DATA VERSION

Strategy:
- Every X minutes, rank currency pairs by momentum relative to USD
- Go long top N pairs, short bottom N pairs (USD neutral)
- Intraday trading session: 5PM NY to 5PM NY next day

This version uses REAL bid/ask data from data/bidask/output/ directory.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple, Callable
import glob


# =============================================================================
# TRANSACTION COSTS - REAL BID/ASK DATA
# =============================================================================

# This version uses REAL bid/ask data from the CSV files.
# The data files contain: Datetime, open, high, low, close, vol, seconds_diff, bid, offer
# 
# BID/ASK MODEL:
# - 'close' is the mid price
# - 'bid' is the actual bid price from the market
# - 'offer' (renamed to 'ask') is the actual ask/offer price from the market
# - When BUYING (entering long or exiting short): Pay the ASK (higher price)
# - When SELLING (exiting long or entering short): Receive the BID (lower price)
# - Transaction cost is implicit in the bid/ask prices (not deducted separately)
# - Gross PnL = what we would have made at mid prices
# - Net PnL = what we actually made using bid/ask prices
# - Transaction Cost = Gross PnL - Net PnL
#


# =============================================================================
# MOMENTUM RANKING FUNCTIONS
# =============================================================================

def calculate_returns(df: pd.DataFrame, window: int) -> float:
    """Simple return over window periods (USD-normalized)."""
    if len(df) < window:
        return np.nan
    return (df['close_usd'].iloc[-1] / df['close_usd'].iloc[-window]) - 1


def calculate_log_returns(df: pd.DataFrame, window: int) -> float:
    """Log return over window periods (USD-normalized)."""
    if len(df) < window:
        return np.nan
    return np.log(df['close_usd'].iloc[-1] / df['close_usd'].iloc[-window])


def calculate_zscore(df: pd.DataFrame, window: int) -> float:
    """Z-score of current price vs rolling mean/std (USD-normalized)."""
    if len(df) < window:
        return np.nan
    prices = df['close_usd'].iloc[-window:]
    mean = prices.mean()
    std = prices.std()
    if std == 0:
        return 0
    return (df['close_usd'].iloc[-1] - mean) / std


def calculate_rsi(df: pd.DataFrame, window: int) -> float:
    """RSI momentum indicator (0-100) - USD-normalized."""
    if len(df) < window + 1:
        return np.nan
    
    prices = df['close_usd'].iloc[-(window+1):]
    deltas = prices.diff()
    gain = deltas.where(deltas > 0, 0).rolling(window=window).mean()
    loss = -deltas.where(deltas < 0, 0).rolling(window=window).mean()
    
    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
    rsi = 100 - (100 / (1 + rs))
    return rsi - 50  # Center at 0


def calculate_price_change(df: pd.DataFrame, window: int) -> float:
    """Absolute price change over window (USD-normalized)."""
    if len(df) < window:
        return np.nan
    return df['close_usd'].iloc[-1] - df['close_usd'].iloc[-window]


def calculate_momentum(df: pd.DataFrame, window: int) -> float:
    """Rate of change momentum (USD-normalized)."""
    if len(df) < window:
        return np.nan
    return ((df['close_usd'].iloc[-1] - df['close_usd'].iloc[-window]) / 
            df['close_usd'].iloc[-window]) * 100


# Dictionary of available ranking functions
RANKING_FUNCTIONS = {
    'returns': calculate_returns,
    'log_returns': calculate_log_returns,
    'zscore': calculate_zscore,
    'rsi': calculate_rsi,
    'price_change': calculate_price_change,
    'momentum': calculate_momentum,
}


# =============================================================================
# DATA LOADING AND ALIGNMENT
# =============================================================================

def get_available_pairs(data_dir: Path) -> List[str]:
    """Get list of all available currency pairs from data directory."""
    # Get unique pairs from filenames
    files = list(data_dir.glob('*_bars.csv'))
    pairs = sorted(set([f.stem.split('_')[1] for f in files]))
    return pairs


def load_pair_data(data_dir: Path, date_str: str, pair: str) -> pd.DataFrame:
    """Load 1-minute bar data for a specific pair and date."""
    filepath = data_dir / f"{date_str}_{pair}_bars.csv"
    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath, parse_dates=['Datetime'])
    df = df.sort_values('Datetime').reset_index(drop=True)
    return df


def is_usd_inverse(pair: str) -> bool:
    """Check if pair is inverse (USD is base: USDJPY, USDCAD, etc.)."""
    return pair.startswith('USD')


def get_usd_relative_price(pair: str, price: float) -> float:
    """
    Transform all pairs so USD is the QUOTE currency.
    
    Goal: Make USD the quote currency for all pairs so features are consistent.
    - GBP/USD: Already has USD as quote → KEEP AS IS
    - USD/JPY: Has USD as base → INVERT to get JPY/USD
    
    This way all momentum features measure: Base Currency / USD
    - Higher value = Base currency stronger vs USD
    - Lower value = Base currency weaker vs USD
    """
    if is_usd_inverse(pair):
        # USD/JPY → Invert to JPY/USD (make USD the quote)
        return 1.0 / price if price != 0 else 0
    else:
        # GBP/USD → Keep as is (USD already the quote)
        return price


def normalize_pair_data(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Normalize pair data to USD strength for momentum ranking.
    
    We rank USD STRENGTH across all pairs:
    - Higher momentum = USD is stronger vs that currency
    - Lower momentum = USD is weaker vs that currency
    """
    df = df.copy()
    
    # Rename 'offer' column to 'ask' for consistency
    if 'offer' in df.columns:
        df['ask'] = df['offer']
    
    # Create bid/ask columns for OHLC prices
    # Note: The data file has 'bid' and 'offer' for close price
    # We need to create bid/ask for open, high, low as well
    # For simplicity, use the same spread as close
    if 'bid' in df.columns and 'ask' in df.columns and 'close' in df.columns:
        # Calculate spread from close bid/ask
        spread = df['ask'] - df['bid']
        half_spread_pct = (spread / df['close']) / 2
        
        # Apply to OHLC
        for col in ['open', 'high', 'low']:
            if col in df.columns:
                df[f'{col}_bid'] = df[col] * (1 - half_spread_pct)
                df[f'{col}_ask'] = df[col] * (1 + half_spread_pct)
        
        # Use existing bid/ask for close
        df['close_bid'] = df['bid']
        df['close_ask'] = df['ask']
    
    # Convert prices to USD strength metric
    for col in ['open', 'high', 'low', 'close']:
        df[f'{col}_usd'] = df[col].apply(lambda x: get_usd_relative_price(pair, x))
    
    # Calculate returns based on transformed prices
    # Since all pairs now have USD as quote (Base/USD format):
    # - Returns are consistent across all pairs
    # - No need for negation since transformation handles directionality
    if is_usd_inverse(pair):
        # Was USD/JPY, now JPY/USD after transformation
        df['returns'] = -df['close'].pct_change()  # Negate because we inverted the price
    else:
        # Was GBP/USD, still GBP/USD (no transformation)
        df['returns'] = df['close'].pct_change()
    
    return df


def align_pair_data(pair_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Align all pair dataframes to common timestamps.
    Fill missing timestamps with forward-fill.
    """
    if not pair_data:
        return {}
    
    # Get union of all timestamps
    all_timestamps = set()
    for df in pair_data.values():
        all_timestamps.update(df['Datetime'].values)
    
    common_timestamps = pd.DataFrame({'Datetime': sorted(all_timestamps)})
    
    # Reindex each pair to common timestamps
    aligned_data = {}
    for pair, df in pair_data.items():
        merged = common_timestamps.merge(df, on='Datetime', how='left')
        # Forward fill prices (use last known price if missing)
        merged[['open', 'high', 'low', 'close']] = merged[['open', 'high', 'low', 'close']].ffill()
        aligned_data[pair] = merged
    
    return aligned_data


def filter_ny_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter data to NY trading session: 5PM NY (previous day) to 5PM NY (current day).
    Data is in UTC, so 5PM NY = 10PM UTC (standard time) or 9PM UTC (daylight time).
    
    Simplification: Use 10PM UTC cutoff (standard time).
    """
    df = df.copy()
    df['hour_utc'] = df['Datetime'].dt.hour
    
    # Get date range
    start_date = df['Datetime'].min().date()
    end_date = df['Datetime'].max().date()
    
    # 5PM NY previous day in UTC (approx 10PM UTC)
    # Keep data from 10PM previous day to 10PM current day
    session_start = pd.Timestamp(start_date) + pd.Timedelta(hours=22)  # 10PM start
    session_end = session_start + pd.Timedelta(hours=24)  # 24 hours later
    
    # Filter to session window
    df_session = df[(df['Datetime'] >= session_start) & (df['Datetime'] < session_end)]
    
    return df_session


# =============================================================================
# CROSS-SECTIONAL RANKING
# =============================================================================

def rank_pairs(pair_data: Dict[str, pd.DataFrame], 
               ranking_func: Callable,
               window: int,
               timestamp_idx: int) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Rank all pairs at a specific timestamp using the ranking function.
    
    Returns:
        Tuple of (ranked scores Series, dict of raw momentum values)
    """
    scores = {}
    
    for pair, df in pair_data.items():
        # Get data up to current timestamp
        df_history = df.iloc[:timestamp_idx + 1]
        
        if len(df_history) < window:
            scores[pair] = np.nan
            continue
        
        # Calculate momentum score
        score = ranking_func(df_history, window)
        scores[pair] = score
    
    # Convert to series and rank
    scores_series = pd.Series(scores)
    scores_series = scores_series.dropna()
    
    return scores_series, scores


def select_positions(ranked_scores: pd.Series, top_n: int, 
                    pair_data: Dict[str, pd.DataFrame], 
                    timestamp_idx: int,
                    all_momentum: Dict[str, float],
                    usd_notional: float = 1_000_000) -> Dict[str, Dict]:
    """
    Select top N long and bottom N short positions with USD-neutral sizing.
    
    For USD neutrality, all positions must have equal USD notional value.
    - EUR/USD at 1.05: To get $1M exposure, buy 952,381 EUR
    - USD/JPY at 150: To get $1M exposure, buy $1M (or 150M JPY)
    - GBP/USD at 1.27: To get $1M exposure, buy 787,402 GBP
    
    Returns dict of pair -> {
        'direction': +1 (long) or -1 (short) or 0 (neutral),
        'usd_notional': USD value of position,
        'pair_notional': Position size in pair terms,
        'entry_price': Entry price,
        'entry_momentum': Momentum value at entry time
    }
    """
    if len(ranked_scores) < 2 * top_n:
        # Not enough pairs
        return {}
    
    # Sort by score
    sorted_scores = ranked_scores.sort_values(ascending=False)
    
    # CRITICAL: After transformation, all pairs are in Base/USD format
    # 
    # High momentum (top pairs) = Base currency STRONG vs USD → We want to SHORT USD (LONG base)
    # Low momentum (bottom pairs) = Base currency WEAK vs USD → We want to LONG USD (SHORT base)
    #
    # Since USD is always the quote currency after transformation:
    # - Top pairs: Base currency strengthening → LONG the pair (buy base, sell USD)
    # - Bottom pairs: Base currency weakening → SHORT the pair (sell base, buy USD)
    #
    # But remember: Original USDJPY was inverted to JPY/USD!
    # - If original pair is USD/JPY (inverse), we need to flip the direction back
    # - If original pair is GBP/USD (regular), direction stays the same
    
    positions = {}
    
    top_pairs = sorted_scores.head(top_n).index.tolist()
    bottom_pairs = sorted_scores.tail(top_n).index.tolist()
    
    for pair in ranked_scores.index:
        # Get entry price based on direction (use bid/ask spread)
        # LONG (buy): Pay the ask (higher price)
        # SHORT (sell): Receive the bid (lower price)
        row = pair_data[pair].iloc[timestamp_idx]
        
        if pair in top_pairs:
            # Top pairs = base currency STRONG vs USD → We want USD LONG (short base)
            if is_usd_inverse(pair):
                # Original: USD/JPY, Transformed: JPY/USD  
                # JPY strong vs USD → We SHORT JPY = LONG USD
                # In original terms: LONG USD/JPY
                direction = 1.0
                entry_price = row['close_ask']  # Pay ask when buying
            else:
                # Original: GBP/USD, Transformed: GBP/USD (same)
                # GBP strong vs USD → We SHORT GBP = LONG USD
                # In original terms: SHORT GBP/USD
                direction = -1.0
                entry_price = row['close_bid']  # Receive bid when selling
        elif pair in bottom_pairs:
            # Bottom pairs = base currency WEAK vs USD → We want USD SHORT (long base)
            if is_usd_inverse(pair):
                # Original: USD/JPY, Transformed: JPY/USD
                # JPY weak vs USD → We LONG JPY = SHORT USD
                # In original terms: SHORT USD/JPY
                direction = -1.0
                entry_price = row['close_bid']  # Receive bid when selling
            else:
                # Original: GBP/USD, Transformed: GBP/USD (same)
                # GBP weak vs USD → We LONG GBP = SHORT USD
                # In original terms: LONG GBP/USD
                direction = 1.0
                entry_price = row['close_ask']  # Pay ask when buying
        else:
            direction = 0.0  # Neutral
            entry_price = row['close']  # Use mid for neutral positions
            positions[pair] = {
                'direction': direction,
                'usd_notional': 0.0,
                'pair_notional': 0.0,
                'entry_price': entry_price,
                'entry_momentum': all_momentum.get(pair, np.nan)
            }
            continue
        
        # Calculate position size for USD neutrality
        if is_usd_inverse(pair):
            # USD/JPY = 150 means 1 USD = 150 JPY
            # To get $1M exposure, we need $1M USD (which equals 150M JPY)
            # Position notional is in USD
            pair_notional = usd_notional
        else:
            # EUR/USD = 1.05 means 1 EUR = 1.05 USD
            # To get $1M USD exposure, we need 1M / 1.05 = 952,381 EUR
            # Position notional is in EUR (or GBP, AUD, etc.)
            pair_notional = usd_notional / entry_price
        
        positions[pair] = {
            'direction': direction,
            'usd_notional': usd_notional,
            'pair_notional': pair_notional,
            'entry_price': entry_price,
            'entry_momentum': all_momentum.get(pair, np.nan)
        }
    
    return positions


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def calculate_pnl(pair_data: Dict[str, pd.DataFrame],
                 positions: Dict[str, Dict],
                 entry_idx: int,
                 exit_idx: int) -> Dict[str, Dict]:
    """
    Calculate PnL for each pair between entry and exit timestamps.
    Uses proper position sizing for USD neutrality.
    
    Returns dict of pair -> {
        'pnl_usd': PnL in USD,
        'pnl_pct': PnL as percentage,
        'exit_price': Exit price,
        'price_change': Exit - Entry price,
        'price_change_pct': Price change %
    }
    """
    pnls = {}
    
    for pair, pos_info in positions.items():
        if pos_info['direction'] == 0:
            pnls[pair] = {
                'pnl_usd': 0.0,
                'pnl_pct': 0.0,
                'exit_price': 0.0,
                'price_change': 0.0,
                'price_change_pct': 0.0
            }
            continue
        
        df = pair_data[pair]
        entry_price = pos_info['entry_price']
        
        # Get exit price based on direction (use bid/ask spread)
        # LONG exit (sell): Receive the bid (lower price)
        # SHORT exit (buy): Pay the ask (higher price)
        row = df.iloc[exit_idx]
        if pos_info['direction'] > 0:
            # LONG position: exit by selling, receive bid
            exit_price = row['close_bid']
        else:
            # SHORT position: exit by buying, pay ask
            exit_price = row['close_ask']
        
        direction = pos_info['direction']
        pair_notional = pos_info['pair_notional']
        
        # Calculate price change
        price_change = exit_price - entry_price
        price_change_pct = (exit_price - entry_price) / entry_price
        
        # Calculate PnL in USD
        if is_usd_inverse(pair):
            # For USD/JPY, USD/CHF, etc.: USD is base currency
            # pair_notional is stored in USD ($1M), but represents USD amount
            # PnL = direction * USD_notional * price_change_pct
            pnl_usd = direction * pair_notional * price_change_pct
        else:
            # For EUR/USD, GBP/USD, etc.: USD is quote currency
            # pair_notional is in foreign currency (EUR, GBP, etc.)
            # PnL = direction * foreign_amount * price_change
            pnl_usd = direction * pair_notional * price_change
        
        # PnL as percentage of notional
        pnl_pct = pnl_usd / pos_info['usd_notional']
        
        pnls[pair] = {
            'pnl_usd': pnl_usd,
            'pnl_pct': pnl_pct,
            'exit_price': exit_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct
        }
    
    return pnls


def run_backtest(data_dir: Path,
                 start_date: str,
                 end_date: str,
                 rebalance_minutes: int,
                 lookback_window: int,
                 top_n: int,
                 ranking_method: str,
                 usd_notional_per_position: float = 1_000_000,
                 trading_hours: Tuple[int, int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run cross-sectional momentum backtest.
    
    Parameters:
    - data_dir: Path to data directory
    - start_date: Start date (YYYYMMDD format)
    - end_date: End date (YYYYMMDD format)
    - rebalance_minutes: Rebalance interval in minutes
    - lookback_window: Momentum lookback window in bars
    - top_n: Number of pairs to long/short
    - ranking_method: Name of ranking function to use
    - usd_notional_per_position: USD notional size per position (default $1M)
    - trading_hours: Optional tuple of (start_hour, end_hour) in EST (e.g., (9, 14) for 9am-2pm)
    
    Returns:
    - Tuple of (summary_df, trades_df)
      - summary_df: Rebalance-level summary (timestamp, equity, pnl)
      - trades_df: Trade-level detail (all positions with entry/exit/pnl)
    """
    # Get ranking function
    if ranking_method not in RANKING_FUNCTIONS:
        raise ValueError(f"Unknown ranking method: {ranking_method}")
    ranking_func = RANKING_FUNCTIONS[ranking_method]
    
    # Get available pairs
    pairs = get_available_pairs(data_dir)
    print(f"Found {len(pairs)} currency pairs: {pairs}")
    
    # Generate date range (trading days only - Mon-Fri)
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    
    date_range = pd.date_range(start_dt, end_dt, freq='D')
    date_range = [d for d in date_range if d.weekday() < 5]  # Mon-Fri only
    
    # Results storage
    results = []
    trades = []
    signals = []  # NEW: Track all signals generated
    equity = 1.0  # Start with $1
    
    print(f"\nRunning backtest from {start_date} to {end_date}")
    print(f"Rebalance: every {rebalance_minutes} min, Lookback: {lookback_window} bars")
    print(f"Top N: {top_n}, Ranking: {ranking_method}")
    print(f"USD Notional per position: ${usd_notional_per_position:,.0f}")
    if trading_hours:
        print(f"Trading Hours: {trading_hours[0]:02d}:00 - {trading_hours[1]:02d}:00 EST")
    print("=" * 80)
    
    for date in date_range:
        date_str = date.strftime('%Y%m%d')
        print(f"\nProcessing {date_str}...")
        
        # Load data for all pairs
        pair_data = {}
        for pair in pairs:
            df = load_pair_data(data_dir, date_str, pair)
            if df is not None and len(df) > 0:
                # Normalize to USD-relative
                df = normalize_pair_data(df, pair)
                # Filter to NY session
                df = filter_ny_session(df)
                if len(df) > 0:
                    pair_data[pair] = df
        
        if len(pair_data) < 2:
            print(f"  Insufficient data for {date_str}, skipping")
            continue
        
        # Align timestamps
        pair_data = align_pair_data(pair_data)
        
        # Get common timestamps
        timestamps = pair_data[list(pair_data.keys())[0]]['Datetime'].values
        num_bars = len(timestamps)
        
        if num_bars < lookback_window:
            print(f"  Not enough bars ({num_bars}), skipping")
            continue
        
        print(f"  Loaded {len(pair_data)} pairs, {num_bars} bars")
        
        # Backtest loop for this day
        current_positions = {}
        rebalance_counter = 0
        
        for idx in range(lookback_window, num_bars, rebalance_minutes):
            timestamp = timestamps[idx]
            
            # Apply trading hours filter if specified (convert UTC timestamp to EST)
            if trading_hours:
                timestamp_dt = pd.Timestamp(timestamp)
                # Convert UTC to EST (UTC - 5 hours, no DST for simplicity)
                timestamp_est = timestamp_dt - pd.Timedelta(hours=5)
                hour_est = timestamp_est.hour
                
                # Skip if outside trading hours
                if hour_est < trading_hours[0] or hour_est >= trading_hours[1]:
                    continue
            
            # Rank pairs and get momentum values
            scores, all_momentum = rank_pairs(pair_data, ranking_func, lookback_window, idx)
            
            if len(scores) < 2 * top_n:
                continue
            
            # Select new positions with proper sizing (pass momentum for storing with position)
            new_positions = select_positions(scores, top_n, pair_data, idx, all_momentum, usd_notional_per_position)
            
            # NEW: Log all signals (for every pair at this rebalance)
            sorted_scores = scores.sort_values(ascending=False)
            top_pairs = sorted_scores.head(top_n).index.tolist()
            bottom_pairs = sorted_scores.tail(top_n).index.tolist()
            
            for pair in scores.index:
                signal_type = 'NEUTRAL'
                usd_position = 'NEUTRAL'
                execution_side = 'NONE'
                
                if pair in top_pairs:
                    signal_type = 'TOP'
                    if is_usd_inverse(pair):
                        usd_position = 'LONG_USD'
                        execution_side = 'BUY'
                    else:
                        usd_position = 'LONG_USD'
                        execution_side = 'SELL'
                elif pair in bottom_pairs:
                    signal_type = 'BOTTOM'
                    if is_usd_inverse(pair):
                        usd_position = 'SHORT_USD'
                        execution_side = 'SELL'
                    else:
                        usd_position = 'SHORT_USD'
                        execution_side = 'BUY'
                
                signals.append({
                    'timestamp': timestamp,
                    'date': date_str,
                    'pair': pair,
                    'momentum': all_momentum.get(pair, np.nan),
                    'rank': list(sorted_scores.index).index(pair) + 1,
                    'signal_type': signal_type,  # TOP, BOTTOM, or NEUTRAL
                    'usd_position': usd_position,  # LONG_USD, SHORT_USD, or NEUTRAL
                    'execution_side': execution_side,  # BUY, SELL, or NONE
                    'mid_price': pair_data[pair].iloc[idx]['close'],
                    'bid_price': pair_data[pair].iloc[idx]['close_bid'],
                    'ask_price': pair_data[pair].iloc[idx]['close_ask'],
                })
            
            # Determine which positions changed (for turnover optimization)
            positions_to_close = {}  # Positions that need to be closed
            positions_to_keep = {}   # Positions that continue unchanged
            
            if current_positions:
                for pair in current_positions:
                    old_dir = current_positions[pair]['direction']
                    new_dir = new_positions.get(pair, {'direction': 0.0})['direction']
                    
                    if old_dir != 0 and old_dir == new_dir:
                        # Position continues - HOLD it, don't close and reopen
                        # Keep original entry price and timestamp
                        positions_to_keep[pair] = current_positions[pair].copy()
                        # Don't update entry info - we're still in the same trade
                    elif old_dir != 0:
                        # Position changed or closed - need to close it
                        positions_to_close[pair] = current_positions[pair]
            
            # Calculate PnL from positions being closed
            if positions_to_close:
                # Calculate PnL from entry to exit (using stored entry_idx from position)
                pnl_details = {}
                for pair, pos_info in positions_to_close.items():
                    df = pair_data[pair]
                    entry_price = pos_info['entry_price']
                    entry_idx = pos_info['entry_idx']  # Use stored entry index
                    exit_idx = idx
                    
                    # Get exit price based on direction (use bid/ask spread)
                    row = df.iloc[exit_idx]
                    if pos_info['direction'] > 0:
                        exit_price = row['close_bid']
                    else:
                        exit_price = row['close_ask']
                    
                    direction = pos_info['direction']
                    pair_notional = pos_info['pair_notional']
                    
                    # Calculate price change
                    price_change = exit_price - entry_price
                    price_change_pct = (exit_price - entry_price) / entry_price
                    
                    # Calculate PnL in USD
                    if is_usd_inverse(pair):
                        pnl_usd = direction * pair_notional * price_change_pct
                    else:
                        pnl_usd = direction * pair_notional * price_change
                    
                    pnl_pct = pnl_usd / pos_info['usd_notional']
                    
                    pnl_details[pair] = {
                        'pnl_usd': pnl_usd,
                        'pnl_pct': pnl_pct,
                        'exit_price': exit_price,
                        'price_change': price_change,
                        'price_change_pct': price_change_pct
                    }
                
                # Sum up total PnL in USD (already includes bid/ask costs)
                total_pnl_usd = sum([p['pnl_usd'] for p in pnl_details.values()])
                
                # Calculate implicit transaction costs from bid/ask spread
                # Cost = sum of (entry_spread + exit_spread) for each position
                # Since we use real bid/ask prices, the cost is already in the PnL
                # We'll calculate it from the actual entry/exit prices for reporting
                total_transaction_cost = 0.0
                for pair in positions_to_close:
                    pos_info = positions_to_close[pair]
                    pnl_info = pnl_details[pair]
                    
                    # Get mid prices at entry and exit
                    df = pair_data[pair]
                    entry_mid = df.iloc[pos_info['entry_idx']]['close']
                    exit_mid = df.iloc[idx]['close']
                    
                    # Calculate what PnL would have been at mid prices
                    direction = pos_info['direction']
                    pair_notional = pos_info['pair_notional']
                    
                    mid_price_change = exit_mid - entry_mid
                    mid_price_change_pct = (exit_mid - entry_mid) / entry_mid
                    
                    if is_usd_inverse(pair):
                        pnl_at_mid = direction * pair_notional * mid_price_change_pct
                    else:
                        pnl_at_mid = direction * pair_notional * mid_price_change
                    
                    # Transaction cost = PnL at mid - PnL with bid/ask
                    trade_cost = pnl_at_mid - pnl_info['pnl_usd']
                    total_transaction_cost += trade_cost
                
                # Net PnL is just total_pnl_usd (costs already embedded in bid/ask prices)
                net_pnl_usd = total_pnl_usd
                
                # Total notional (sum of all position notionals)
                total_notional = sum([p['usd_notional'] for p in current_positions.values()])
                
                # PnL as fraction of total notional (after costs)
                total_pnl_pct = net_pnl_usd / total_notional if total_notional > 0 else 0
                
                # Update equity with net PnL
                equity *= (1 + total_pnl_pct)
                
                # Log trade details for CLOSED positions only
                for pair in positions_to_close:
                    # Skip if pair not in pnl_details (shouldn't happen, but defensive)
                    if pair not in pnl_details:
                        continue
                    
                    pos_info = positions_to_close[pair]
                    pnl_info = pnl_details[pair]
                    
                    # Calculate hold time in minutes
                    entry_timestamp = timestamps[pos_info['entry_idx']]
                    exit_timestamp = timestamp
                    hold_time_minutes = (pd.Timestamp(exit_timestamp) - pd.Timestamp(entry_timestamp)).total_seconds() / 60
                    
                    # Determine execution side (BUY/SELL pair) and currency position
                    is_inverse = is_usd_inverse(pair)
                    direction_value = pos_info['direction']
                    
                    if direction_value > 0:
                        # direction = +1 = LONG the pair
                        execution_side = 'BUY'
                        if is_inverse:
                            # BUY USD/JPY = Long USD, Short JPY
                            currency_position = 'LONG_USD'
                        else:
                            # BUY EUR/USD = Long EUR, Short USD
                            currency_position = 'SHORT_USD'
                    else:
                        # direction = -1 = SHORT the pair
                        execution_side = 'SELL'
                        if is_inverse:
                            # SELL USD/JPY = Short USD, Long JPY
                            currency_position = 'SHORT_USD'
                        else:
                            # SELL EUR/USD = Short EUR, Long USD
                            currency_position = 'LONG_USD'
                    
                    # Calculate implicit transaction cost from actual bid/ask spread
                    # Get mid prices at entry and exit
                    entry_mid = pair_data[pair].iloc[pos_info['entry_idx']]['close']
                    exit_mid = pair_data[pair].iloc[idx]['close']
                    
                    # Calculate what PnL would have been at mid prices
                    mid_price_change = exit_mid - entry_mid
                    mid_price_change_pct = (exit_mid - entry_mid) / entry_mid
                    
                    if is_inverse:
                        pnl_at_mid = direction_value * pos_info['pair_notional'] * mid_price_change_pct
                    else:
                        pnl_at_mid = direction_value * pos_info['pair_notional'] * mid_price_change
                    
                    # Transaction cost = PnL at mid - PnL with bid/ask
                    trade_net_pnl_usd = pnl_info['pnl_usd']  # This already includes bid/ask impact
                    transaction_cost = pnl_at_mid - trade_net_pnl_usd  # Cost is the difference
                    trade_gross_pnl_usd = pnl_at_mid  # What we would have made at mid prices
                    trade_net_pnl_pct = trade_net_pnl_usd / pos_info['usd_notional']
                    
                    trades.append({
                        'timestamp': timestamp,
                        'date': date_str,
                        'pair': pair,
                        'execution_side': execution_side,  # BUY or SELL the pair
                        'currency_position': currency_position,  # LONG_USD or SHORT_USD
                        'entry_price': pos_info['entry_price'],
                        'exit_price': pnl_info['exit_price'],
                        'entry_timestamp': entry_timestamp,
                        'exit_timestamp': exit_timestamp,
                        'hold_time_minutes': hold_time_minutes,
                        'price_change': pnl_info['price_change'],
                        'price_change_pct': pnl_info['price_change_pct'],
                        'usd_notional': pos_info['usd_notional'],
                        'pair_notional': pos_info['pair_notional'],
                        'pnl_gross': trade_gross_pnl_usd,  # What we would have made at mid prices
                        'transaction_cost': transaction_cost,  # Implicit cost from bid/ask spread
                        'pnl_net': trade_net_pnl_usd,  # What we actually made with bid/ask
                        'pnl_pct': trade_net_pnl_pct,  # Net PnL as percentage for THIS trade
                        'entry_momentum': pos_info.get('entry_momentum', np.nan),  # Momentum when opened
                        'exit_momentum': all_momentum.get(pair, np.nan),  # Momentum when closed
                    })
                
                # Record rebalance summary
                # Note: total_pnl_usd already includes bid/ask costs (it's net PnL)
                # Calculate gross PnL by adding back transaction costs
                total_pnl_gross = total_pnl_usd + total_transaction_cost
                
                result = {
                    'timestamp': timestamp,
                    'date': date_str,
                    'equity': equity,
                    'pnl_gross': total_pnl_gross,  # What we would have made at mid prices
                    'transaction_cost': total_transaction_cost,  # Implicit cost from bid/ask
                    'pnl_net': net_pnl_usd,  # What we actually made (same as total_pnl_usd)
                    'pnl_pct': total_pnl_pct,  # Net PnL as percentage
                    'total_notional': total_notional,
                    'num_positions': len([p for p in current_positions.values() if p['direction'] != 0]),
                }
                
                # Add momentum for all pairs
                for pair in sorted(all_momentum.keys()):
                    result[f'momentum_{pair}'] = all_momentum[pair]
                
                # Add position info (NEW positions being entered, not old ones being closed)
                # Separate by currency position (LONG_USD vs SHORT_USD), not execution side
                long_usd_pairs = []
                short_usd_pairs = []
                
                for pair, pos_info in new_positions.items():
                    if pos_info['direction'] == 0:
                        continue
                    
                    is_inverse = is_usd_inverse(pair)
                    direction = pos_info['direction']
                    
                    # Determine USD position
                    if direction > 0:  # LONG the pair (BUY)
                        if is_inverse:
                            # BUY USD/JPY = Long USD, Short JPY
                            long_usd_pairs.append(pair)
                        else:
                            # BUY EUR/USD = Long EUR, Short USD
                            short_usd_pairs.append(pair)
                    else:  # SHORT the pair (SELL)
                        if is_inverse:
                            # SELL USD/JPY = Short USD, Long JPY
                            short_usd_pairs.append(pair)
                        else:
                            # SELL EUR/USD = Short EUR, Long USD
                            long_usd_pairs.append(pair)
                
                result['long_usd_pairs'] = long_usd_pairs
                result['short_usd_pairs'] = short_usd_pairs
                
                results.append(result)
            
            # Update positions - merge kept positions with new positions
            # For kept positions: preserve original entry info
            # For new positions: add entry_idx
            updated_positions = {}
            
            # First, add all kept positions (these maintain their original entry info)
            for pair, pos_info in positions_to_keep.items():
                updated_positions[pair] = pos_info
            
            # Then, add new positions and positions that flipped
            for pair, pos_info in new_positions.items():
                if pair not in positions_to_keep and pos_info['direction'] != 0:
                    # New position - add entry_idx
                    pos_info['entry_idx'] = idx
                    updated_positions[pair] = pos_info
                elif pos_info['direction'] == 0:
                    # Neutral position
                    updated_positions[pair] = pos_info
            
            current_positions = updated_positions
            rebalance_counter += 1
        
        print(f"  Completed {rebalance_counter} rebalances, Equity: {equity:.4f}")
    
    # Convert to DataFrames
    results_df = pd.DataFrame(results)
    trades_df = pd.DataFrame(trades)
    signals_df = pd.DataFrame(signals)  # NEW: Signals dataframe
    
    return results_df, trades_df, signals_df  # Return 3 dataframes now


# =============================================================================
# PERFORMANCE STATISTICS
# =============================================================================

def calculate_statistics(results_df: pd.DataFrame) -> Dict:
    """Calculate performance statistics from backtest results."""
    if len(results_df) == 0:
        return {}
    
    # Total return
    final_equity = results_df['equity'].iloc[-1]
    total_return = (final_equity - 1.0) * 100
    
    # PnL stats (use pnl_pct for percentage returns)
    pnls = results_df['pnl_pct'].values
    num_trades = len(pnls)
    winning_trades = len([p for p in pnls if p > 0])
    losing_trades = len([p for p in pnls if p < 0])
    
    win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0
    
    avg_win = np.mean([p for p in pnls if p > 0]) * 100 if winning_trades > 0 else 0
    avg_loss = np.mean([p for p in pnls if p < 0]) * 100 if losing_trades > 0 else 0
    
    # Sharpe ratio (annualized)
    # Assuming ~1440 minutes per day / rebalance_minutes rebalances per day
    returns = results_df['pnl_pct'].values
    if len(returns) > 0 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
    else:
        sharpe = 0
    
    # Max drawdown
    equity_curve = results_df['equity'].values
    cummax = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - cummax) / cummax
    max_drawdown = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0
    
    # Date range
    num_days = results_df['date'].nunique()
    
    # PnL breakdown
    total_pnl_gross = results_df['pnl_gross'].sum() if 'pnl_gross' in results_df.columns else 0
    total_transaction_cost = results_df['transaction_cost'].sum() if 'transaction_cost' in results_df.columns else 0
    total_pnl_net = results_df['pnl_net'].sum() if 'pnl_net' in results_df.columns else 0
    avg_notional = results_df['total_notional'].mean() if 'total_notional' in results_df.columns else 0
    
    stats = {
        'Total Return (%)': total_return,
        'Final Equity': final_equity,
        'PnL Gross (USD)': total_pnl_gross,
        'Transaction Costs (USD)': total_transaction_cost,
        'PnL Net (USD)': total_pnl_net,
        'Avg Notional (USD)': avg_notional,
        'Num Rebalances': num_trades,
        'Win Rate (%)': win_rate,
        'Avg Win (%)': avg_win,
        'Avg Loss (%)': avg_loss,
        'Sharpe Ratio': sharpe,
        'Max Drawdown (%)': max_drawdown,
        'Num Days': num_days,
    }
    
    return stats


def print_statistics(stats: Dict):
    """Pretty print statistics."""
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:>15.4f}")
        else:
            print(f"{key:.<40} {value:>15}")
    
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='FX Cross-Sectional Momentum Backtest')
    parser.add_argument('--start-date', default='20250102', help='Start date MMDDYYYY')
    parser.add_argument('--end-date', default='20250930', help='End date MMDDYYYY')
    parser.add_argument('--rebalance-freq', type=int, default=60, help='Rebalance frequency in minutes')
    parser.add_argument('--lookback', type=int, default=30, help='Momentum lookback window in bars')
    parser.add_argument('--top-n', type=int, default=2, help='Number of pairs to long/short')
    parser.add_argument('--start-hour', type=int, default=None, help='Trading start hour in EST (e.g., 9)')
    parser.add_argument('--end-hour', type=int, default=None, help='Trading end hour in EST (e.g., 14)')
    
    args = parser.parse_args()
    
    # Convert date format from MMDDYYYY to YYYYMMDD
    start_date = args.start_date
    end_date = args.end_date
    
    # If dates are in MMDDYYYY format, convert to YYYYMMDD
    if len(start_date) == 8 and start_date[0:2] in ['01','02','03','04','05','06','07','08','09','10','11','12']:
        # Looks like MMDDYYYY
        start_date = start_date[4:8] + start_date[0:2] + start_date[2:4]
    if len(end_date) == 8 and end_date[0:2] in ['01','02','03','04','05','06','07','08','09','10','11','12']:
        # Looks like MMDDYYYY
        end_date = end_date[4:8] + end_date[0:2] + end_date[2:4]
    
    # Configuration - USE REAL BID/ASK DATA
    DATA_DIR = Path("data/bidask/output")
    START_DATE = start_date
    END_DATE = end_date
    
    # Strategy parameters
    REBALANCE_MINUTES = args.rebalance_freq
    LOOKBACK_WINDOW = args.lookback
    TOP_N = args.top_n
    RANKING_METHOD = 'returns'  # Use simple returns
    USD_NOTIONAL = 1_000_000  # $1M per position
    
    # Trading hours filter
    TRADING_HOURS = None
    if args.start_hour is not None and args.end_hour is not None:
        TRADING_HOURS = (args.start_hour, args.end_hour)
    
    # Run backtest
    results_df, trades_df, signals_df = run_backtest(  # Now returns 3 dataframes
        data_dir=DATA_DIR,
        start_date=START_DATE,
        end_date=END_DATE,
        rebalance_minutes=REBALANCE_MINUTES,
        lookback_window=LOOKBACK_WINDOW,
        top_n=TOP_N,
        ranking_method=RANKING_METHOD,
        usd_notional_per_position=USD_NOTIONAL,
        trading_hours=TRADING_HOURS
    )
    
    # Calculate and print statistics
    if len(results_df) > 0:
        stats = calculate_statistics(results_df)
        print_statistics(stats)
        
        # Save results
        output_dir = Path("working_files/fx_momentum_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary results
        hours_suffix = f"_h{TRADING_HOURS[0]}-{TRADING_HOURS[1]}" if TRADING_HOURS else ""
        results_file = output_dir / f"backtest_{START_DATE}_{END_DATE}_rebal{REBALANCE_MINUTES}_top{TOP_N}{hours_suffix}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\nSummary results saved to: {results_file}")
        
        # Save detailed trades (executions)
        if len(trades_df) > 0:
            trades_file = output_dir / f"executions_{START_DATE}_{END_DATE}_rebal{REBALANCE_MINUTES}_top{TOP_N}{hours_suffix}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"Executions saved to: {trades_file}")
            
            # Print sample trades
            print("\nSample executions (first 10):")
            print(trades_df.head(10).to_string())
        
        # Save signals
        if len(signals_df) > 0:
            signals_file = output_dir / f"signals_{START_DATE}_{END_DATE}_rebal{REBALANCE_MINUTES}_top{TOP_N}{hours_suffix}.csv"
            signals_df.to_csv(signals_file, index=False)
            print(f"Signals saved to: {signals_file}")
            print(f"\nTotal signals generated: {len(signals_df):,}")
            print(f"  TOP signals: {(signals_df['signal_type'] == 'TOP').sum():,}")
            print(f"  BOTTOM signals: {(signals_df['signal_type'] == 'BOTTOM').sum():,}")
            print(f"  NEUTRAL signals: {(signals_df['signal_type'] == 'NEUTRAL').sum():,}")
        
        # Generate per-pair statistics
        if len(trades_df) > 0:
            print("\n" + "=" * 80)
            print("PER-PAIR STATISTICS")
            print("=" * 80)
            
            per_pair_stats = []
            for pair in sorted(trades_df['pair'].unique()):
                pair_trades = trades_df[trades_df['pair'] == pair]
                
                num_trades = len(pair_trades)
                wins = (pair_trades['pnl_net'] > 0).sum()
                losses = (pair_trades['pnl_net'] < 0).sum()
                win_rate = (wins / num_trades * 100) if num_trades > 0 else 0
                
                total_gross = pair_trades['pnl_gross'].sum()
                total_cost = pair_trades['transaction_cost'].sum()
                total_net = pair_trades['pnl_net'].sum()
                
                avg_win = pair_trades[pair_trades['pnl_net'] > 0]['pnl_net'].mean() if wins > 0 else 0
                avg_loss = pair_trades[pair_trades['pnl_net'] < 0]['pnl_net'].mean() if losses > 0 else 0
                
                avg_hold = pair_trades['hold_time_minutes'].mean()
                
                per_pair_stats.append({
                    'pair': pair,
                    'num_trades': num_trades,
                    'win_rate': win_rate,
                    'gross_pnl': total_gross,
                    'costs': total_cost,
                    'net_pnl': total_net,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'avg_hold_min': avg_hold
                })
            
            per_pair_df = pd.DataFrame(per_pair_stats)
            per_pair_df = per_pair_df.sort_values('net_pnl', ascending=False)
            
            # Save per-pair stats
            per_pair_file = output_dir / f"per_pair_stats_{START_DATE}_{END_DATE}_rebal{REBALANCE_MINUTES}_top{TOP_N}{hours_suffix}.csv"
            per_pair_df.to_csv(per_pair_file, index=False)
            print(f"\nPer-pair statistics saved to: {per_pair_file}")
            print("\nPer-Pair Summary:")
            print(per_pair_df.to_string(index=False))
    else:
        print("\nNo results generated!")
