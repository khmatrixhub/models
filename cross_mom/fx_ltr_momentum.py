"""
Learn-to-Rank (LTR) Cross-sectional FX Momentum Strategy.

Strategy:
- Train LightGBM ranking model every Saturday using past N days of data
- At each rebalance, use model to rank pairs by predicted next-hour PnL
- Go long top N ranked pairs, short bottom N ranked pairs (USD neutral)
- Features: momentum, volatility, z-score, time-of-day, pair characteristics

Key Differences from Simple Momentum:
- Models learn which feature combinations predict profitable trades
- Can capture non-linear patterns (momentum + low volatility, time effects, etc.)
- Walk-forward validation with weekly retraining
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple
import lightgbm as lgb
import pickle

# =============================================================================
# TRANSACTION COSTS - BID/ASK SPREAD MODEL
# =============================================================================

# Realistic bid-ask spreads by pair (in basis points)
# These are institutional spreads - will be replaced with actual bid/ask data later
# 
# BID/ASK MODEL:
# - Bid/ask prices are calculated from mid price: bid = mid * (1 - spread/2), ask = mid * (1 + spread/2)
# - When BUYING (entering long or exiting short): Pay the ASK (higher price)
# - When SELLING (exiting long or entering short): Receive the BID (lower price)
# - Transaction cost is implicit in the bid/ask prices (not deducted separately)
# - Gross PnL = what we would have made at mid prices
# - Net PnL = what we actually made using bid/ask prices
# - Transaction Cost = Gross PnL - Net PnL
#
PAIR_SPREADS = {
    'EURUSD': 0.5,   # Tightest (most liquid pair globally)
    'USDJPY': 0.5,   # Very liquid
    'GBPUSD': 1.0,   # Liquid major
    'AUDUSD': 1.0,   # Liquid major  
    'USDCAD': 1.5,   # Minor pair
    'USDCHF': 1.5,   # Minor pair
    'USDNOK': 1.71,  # ACTUAL from user's data
    'USDSEK': 3.0,   # Scandinavian
    'USDMXN': 10.0,  # EM pair (wider spread)
    'USDZAR': 15.0,  # EM pair (widest spread)
}


def add_bid_ask_to_data(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Add bid/ask prices to dataframe based on spread.
    Spread is split equally: bid = mid - spread/2, ask = mid + spread/2
    
    Args:
        df: DataFrame with 'close' column (mid price)
        pair: Currency pair for spread lookup
    
    Returns:
        DataFrame with 'bid' and 'ask' columns added
    """
    df = df.copy()
    spread_bp = PAIR_SPREADS.get(pair, 2.0)  # Default 2bp
    spread_decimal = spread_bp / 10000
    half_spread = spread_decimal / 2
    
    # For price columns, calculate bid/ask around mid
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[f'{col}_bid'] = df[col] * (1 - half_spread)
            df[f'{col}_ask'] = df[col] * (1 + half_spread)
    
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def calculate_momentum_features(df: pd.DataFrame, windows: List[int] = [30, 60, 120]) -> Dict[str, float]:
    """Calculate momentum features across multiple windows."""
    features = {}
    
    for w in windows:
        if len(df) < w:
            features[f'returns_{w}'] = np.nan
            features[f'log_returns_{w}'] = np.nan
        else:
            features[f'returns_{w}'] = (df['close_usd'].iloc[-1] / df['close_usd'].iloc[-w]) - 1
            features[f'log_returns_{w}'] = np.log(df['close_usd'].iloc[-1] / df['close_usd'].iloc[-w])
    
    return features


def calculate_volatility_features(df: pd.DataFrame, windows: List[int] = [30, 60]) -> Dict[str, float]:
    """Calculate volatility features."""
    features = {}
    
    for w in windows:
        if len(df) < w:
            features[f'volatility_{w}'] = np.nan
            features[f'high_low_range_{w}'] = np.nan
        else:
            returns = df['close_usd'].iloc[-w:].pct_change().dropna()
            features[f'volatility_{w}'] = returns.std()
            
            high_low = (df['high_usd'].iloc[-w:] - df['low_usd'].iloc[-w:]) / df['close_usd'].iloc[-w:]
            features[f'high_low_range_{w}'] = high_low.mean()
    
    return features


def calculate_zscore_features(df: pd.DataFrame, windows: List[int] = [60, 120]) -> Dict[str, float]:
    """Calculate z-score features (current price vs historical mean)."""
    features = {}
    
    for w in windows:
        if len(df) < w:
            features[f'zscore_{w}'] = np.nan
        else:
            prices = df['close_usd'].iloc[-w:]
            mean = prices.mean()
            std = prices.std()
            if std == 0:
                features[f'zscore_{w}'] = 0
            else:
                features[f'zscore_{w}'] = (df['close_usd'].iloc[-1] - mean) / std
    
    return features


def calculate_time_features(timestamp: pd.Timestamp) -> Dict[str, float]:
    """Calculate time-of-day features (normalized)."""
    # Convert to EST for trading hours
    est = pytz.timezone('US/Eastern')
    ts_est = timestamp.tz_localize('UTC').tz_convert(est)
    
    return {
        'hour_of_day': ts_est.hour,
        'minute_of_hour': ts_est.minute,
        'day_of_week': ts_est.dayofweek,
        'hour_sin': np.sin(2 * np.pi * ts_est.hour / 24),
        'hour_cos': np.cos(2 * np.pi * ts_est.hour / 24),
    }


def calculate_pair_features(pair: str) -> Dict[str, float]:
    """Calculate pair-specific features (one-hot encoding)."""
    # Pair type indicators
    return {
        'is_usd_inverse': 1.0 if pair.startswith('USD') else 0.0,
        'is_major': 1.0 if pair in ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF'] else 0.0,
        'is_em': 1.0 if pair in ['USDMXN', 'USDZAR', 'USDTRY'] else 0.0,
    }


def extract_features_at_timestamp(pair: str, 
                                   df: pd.DataFrame, 
                                   timestamp_idx: int,
                                   timestamp: pd.Timestamp) -> Dict[str, float]:
    """Extract all features for a pair at a specific timestamp."""
    df_history = df.iloc[:timestamp_idx + 1]
    
    features = {}
    features.update(calculate_momentum_features(df_history))
    features.update(calculate_volatility_features(df_history))
    features.update(calculate_zscore_features(df_history))
    features.update(calculate_time_features(timestamp))
    features.update(calculate_pair_features(pair))
    
    return features


# =============================================================================
# DATA UTILITIES (from original)
# =============================================================================

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
    """Normalize pair data to USD-relative prices."""
    df = df.copy()
    
    # Add bid/ask prices first (based on close as mid price)
    df = add_bid_ask_to_data(df, pair)
    
    for col in ['open', 'high', 'low', 'close']:
        df[f'{col}_usd'] = df[col].apply(lambda x: get_usd_relative_price(pair, x))
    
    # Calculate returns based on transformed prices
    # Since all pairs now have USD as quote (Base/USD format):
    # - Returns are consistent across all pairs
    if is_usd_inverse(pair):
        # Was USD/JPY, now JPY/USD after transformation
        df['returns'] = -df['close'].pct_change()  # Negate because we inverted the price
    else:
        # Was GBP/USD, still GBP/USD (no transformation)
        df['returns'] = df['close'].pct_change()
    
    return df


def get_available_pairs(data_dir: Path) -> List[str]:
    """Get list of all available currency pairs from data directory."""
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


def align_pair_data(pair_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Align all pair dataframes to common timestamps."""
    if not pair_data:
        return {}
    
    all_timestamps = set()
    for df in pair_data.values():
        all_timestamps.update(df['Datetime'].values)
    
    common_timestamps = pd.DataFrame({'Datetime': sorted(all_timestamps)})
    
    aligned_data = {}
    for pair, df in pair_data.items():
        merged = common_timestamps.merge(df, on='Datetime', how='left')
        merged[['open', 'high', 'low', 'close']] = merged[['open', 'high', 'low', 'close']].ffill()
        aligned_data[pair] = merged
    
    return aligned_data


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================

def generate_training_data(data_dir: Path,
                          pairs: List[str],
                          start_date: str,
                          end_date: str,
                          rebalance_minutes: int = 60,
                          prediction_horizon: int = 60) -> pd.DataFrame:
    """
    Generate training dataset for Learn-to-Rank model.
    
    Each row = one pair at one timestamp
    Features = momentum, volatility, zscore, time features
    Label = actual PnL over next prediction_horizon minutes
    Group = timestamp (for ranking within each rebalance period)
    
    Returns DataFrame with columns:
    - pair
    - timestamp
    - group_id (for LTR grouping)
    - features (momentum_30, volatility_60, etc.)
    - label (next_hour_pnl)
    """
    print(f"Generating training data from {start_date} to {end_date}...")
    
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    
    date_range = pd.date_range(start_dt, end_dt, freq='D')
    date_range = [d for d in date_range if d.weekday() < 5]  # Mon-Fri only
    
    training_rows = []
    
    for date in date_range:
        date_str = date.strftime('%Y%m%d')
        
        # Load data for all pairs
        pair_data = {}
        for pair in pairs:
            df = load_pair_data(data_dir, date_str, pair)
            if df is not None and len(df) > 0:
                df = normalize_pair_data(df, pair)
                pair_data[pair] = df
        
        if len(pair_data) < 2:
            continue
        
        pair_data = align_pair_data(pair_data)
        
        # Get common timestamps
        first_pair = list(pair_data.keys())[0]
        timestamps = pair_data[first_pair]['Datetime'].values
        num_bars = len(timestamps)
        
        # Generate training samples at each rebalance point
        for idx in range(120, num_bars - prediction_horizon, rebalance_minutes):
            timestamp = timestamps[idx]
            group_id = f"{date_str}_{timestamp}"
            
            # For each pair, extract features and calculate label
            for pair in pair_data.keys():
                df = pair_data[pair]
                
                # Extract features at current timestamp
                features = extract_features_at_timestamp(pair, df, idx, pd.Timestamp(timestamp))
                
                # Calculate label (next-hour PnL)
                current_price = df.iloc[idx]['close']
                future_price = df.iloc[idx + prediction_horizon]['close']
                
                # PnL calculation (simple returns)
                if is_usd_inverse(pair):
                    # For USD/JPY etc, we short when ranked high (strong JPY)
                    next_hour_pnl = -(future_price - current_price) / current_price
                else:
                    # For EUR/USD etc, we long when ranked high (strong EUR)
                    next_hour_pnl = (future_price - current_price) / current_price
                
                # Create training row
                row = {
                    'pair': pair,
                    'timestamp': timestamp,
                    'group_id': group_id,
                    'label': next_hour_pnl,
                }
                row.update(features)
                
                training_rows.append(row)
    
    df_train = pd.DataFrame(training_rows)
    
    if len(df_train) > 0:
        print(f"Generated {len(df_train)} training samples across {df_train['group_id'].nunique()} rebalance periods")
    else:
        print("WARNING: No training samples generated!")
    
    return df_train


# =============================================================================
# LTR MODEL TRAINING
# =============================================================================

def train_ltr_model(df_train: pd.DataFrame, 
                    model_save_path: Path = None) -> lgb.Booster:
    """
    Train LightGBM ranking model.
    
    Uses regression objective to predict PnL, then uses predictions for ranking.
    (LambdaRank requires integer relevance labels, but we have continuous PnL)
    """
    print("\nTraining LightGBM ranking model...")
    
    # Prepare features and labels
    feature_cols = [c for c in df_train.columns if c not in ['pair', 'timestamp', 'group_id', 'label']]
    X = df_train[feature_cols].values
    y = df_train['label'].values
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(
        X, 
        label=y, 
        feature_name=feature_cols,
        free_raw_data=False
    )
    
    # Configure for optimal performance
    # GPU: Requires building from source with CUDA (see GPU_SETUP_GUIDE.md)
    # CPU: Uses all threads - fast enough for dataset size (<5K samples)
    import os
    import multiprocessing
    
    # Determine number of threads (0 = use all)
    n_threads = multiprocessing.cpu_count()
    
    params = {
        'objective': 'regression',
        'metric': 'l1',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'device': 'gpu',           # Try GPU first
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
    }
    
    print(f"Attempting GPU training (fallback to CPU with {n_threads} threads)...")
    
    # Train model (with automatic GPU→CPU fallback)
    try:
        import time
        start_time = time.time()
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data],
            valid_names=['train'],
        )
        
        train_time = time.time() - start_time
        print(f"✓ Model trained using {params['device'].upper()} in {train_time:.2f}s")
        
    except Exception as e:
        if params.get('device') == 'gpu':
            print(f"GPU not available ({str(e).split(':')[0]})")
            print(f"→ Using CPU with {n_threads} threads...")
            
            # Fallback to optimized CPU configuration
            params['device'] = 'cpu'
            params['num_threads'] = n_threads
            params.pop('gpu_platform_id', None)
            params.pop('gpu_device_id', None)
            
            start_time = time.time()
            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[train_data],
                valid_names=['train'],
            )
            train_time = time.time() - start_time
            print(f"✓ Model trained using CPU in {train_time:.2f}s")
        else:
            raise
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance.head(10).to_string(index=False))
    
    # Save model
    if model_save_path:
        model.save_model(str(model_save_path))
        print(f"\nModel saved to: {model_save_path}")
    
    return model


# =============================================================================
# POSITION SELECTION WITH LTR
# =============================================================================

def select_positions_ltr(model: lgb.Booster,
                         pair_data: Dict[str, pd.DataFrame],
                         timestamp_idx: int,
                         timestamp: pd.Timestamp,
                         top_n: int,
                         feature_cols: List[str],
                         usd_notional: float = 1_000_000) -> Dict[str, Dict]:
    """
    Select positions using LTR model predictions.
    
    Returns dict of pair -> position info (same format as simple momentum version)
    """
    # Extract features for all pairs
    features_list = []
    pairs_list = []
    
    for pair in pair_data.keys():
        df = pair_data[pair]
        if timestamp_idx >= len(df):
            continue
            
        features = extract_features_at_timestamp(pair, df, timestamp_idx, timestamp)
        
        # Convert to feature vector
        feature_vector = [features.get(col, np.nan) for col in feature_cols]
        
        # Skip if missing critical features
        if any(np.isnan(feature_vector)):
            continue
        
        features_list.append(feature_vector)
        pairs_list.append(pair)
    
    if len(pairs_list) < 2 * top_n:
        return {}
    
    # Predict scores using LTR model
    X = np.array(features_list)
    predicted_scores = model.predict(X)
    
    # Create ranking
    pair_scores = pd.Series(predicted_scores, index=pairs_list)
    pair_scores = pair_scores.sort_values(ascending=False)
    
    # Select top N and bottom N
    top_pairs = pair_scores.head(top_n).index.tolist()
    bottom_pairs = pair_scores.tail(top_n).index.tolist()
    
    positions = {}
    
    for pair in pairs_list:
        row = pair_data[pair].iloc[timestamp_idx]
        predicted_score = pair_scores[pair]
        
        if pair in top_pairs:
            # CRITICAL: Top ranked = base currency STRONG vs USD → SHORT base (LONG USD)
            # Since USD is always quote after transformation, high momentum = base strengthening
            if is_usd_inverse(pair):
                # Original: USD/JPY, Transformed: JPY/USD
                # SHORT JPY (sell JPY, buy USD) = LONG original USD/JPY
                direction = +1
                execution_side = 'BUY'
                currency_position = 'LONG_USD'
                entry_price = row['close_ask']  # Pay ask when buying
            else:
                # Original: GBP/USD, Transformed: GBP/USD (same)
                # SHORT GBP (sell GBP, buy USD) = SHORT original GBP/USD
                direction = -1
                execution_side = 'SELL'
                currency_position = 'LONG_USD'
                entry_price = row['close_bid']  # Receive bid when selling
            
            # Calculate position size: always $1M USD exposure
            if is_usd_inverse(pair):
                # USD/JPY @ 157: USD is BASE currency
                # $1M USD (base) = 157M JPY (quote)
                base_notional = usd_notional  # $1M USD
                quote_notional = usd_notional * entry_price  # 157M JPY
            else:
                # EUR/USD @ 1.05: USD is QUOTE currency
                # 952K EUR (base) = $1M USD (quote)
                base_notional = usd_notional / entry_price  # 952K EUR
                quote_notional = usd_notional  # $1M USD
            
            positions[pair] = {
                'direction': direction,
                'usd_notional': usd_notional,
                'base_notional': base_notional,
                'quote_notional': quote_notional,
                'entry_price': entry_price,
                'predicted_score': predicted_score,
                'execution_side': execution_side,
                'currency_position': currency_position,
            }
        
        elif pair in bottom_pairs:
            # CRITICAL: Bottom ranked = base currency WEAK vs USD → LONG base (SHORT USD)
            if is_usd_inverse(pair):
                # Original: USD/JPY, Transformed: JPY/USD
                # LONG JPY (buy JPY, sell USD) = SHORT original USD/JPY
                direction = -1
                execution_side = 'SELL'
                currency_position = 'SHORT_USD'
                entry_price = row['close_bid']  # Receive bid when selling
            else:
                # Original: GBP/USD, Transformed: GBP/USD (same)
                # LONG GBP (buy GBP, sell USD) = LONG original GBP/USD
                direction = +1
                execution_side = 'BUY'
                currency_position = 'SHORT_USD'
                entry_price = row['close_ask']  # Pay ask when buying
            
            # Calculate position size: always $1M USD exposure
            if is_usd_inverse(pair):
                # USD/JPY @ 157: USD is BASE currency
                # $1M USD (base) = 157M JPY (quote)
                base_notional = usd_notional  # $1M USD
                quote_notional = usd_notional * entry_price  # 157M JPY
            else:
                # EUR/USD @ 1.05: USD is QUOTE currency
                # 952K EUR (base) = $1M USD (quote)
                base_notional = usd_notional / entry_price  # 952K EUR
                quote_notional = usd_notional  # $1M USD
            
            positions[pair] = {
                'direction': direction,
                'usd_notional': usd_notional,
                'base_notional': base_notional,
                'quote_notional': quote_notional,
                'entry_price': entry_price,
                'predicted_score': predicted_score,
                'execution_side': execution_side,
                'currency_position': currency_position,
            }
    
    return positions


# =============================================================================
# PNL CALCULATION (from original)
# =============================================================================

def calculate_pnl(pair_data: Dict[str, pd.DataFrame],
                 positions: Dict[str, Dict],
                 entry_idx: int,
                 exit_idx: int) -> Dict[str, Dict]:
    """Calculate PnL for positions from entry to exit."""
    pnls = {}
    
    for pair, pos_info in positions.items():
        entry_price = pos_info['entry_price']
        
        # Get exit price based on direction (use bid/ask spread)
        # LONG exit (sell): Receive the bid (lower price)
        # SHORT exit (buy): Pay the ask (higher price)
        row = pair_data[pair].iloc[exit_idx]
        if pos_info['direction'] > 0:
            # LONG position: exit by selling, receive bid
            exit_price = row['close_bid']
        else:
            # SHORT position: exit by buying, pay ask
            exit_price = row['close_ask']
        
        direction = pos_info['direction']
        base_notional = pos_info['base_notional']
        
        price_change = exit_price - entry_price
        price_change_pct = price_change / entry_price
        
        # Calculate PnL in USD
        # For ALL pairs: PnL = direction * base_notional * price_change_pct
        # USD/JPY: base is USD ($1M), price_change_pct is % change in JPY/USD rate
        # EUR/USD: base is EUR (952K), price_change_pct is % change in USD/EUR rate
        # Both give PnL in USD
        pnl_usd = direction * base_notional * price_change_pct
        
        pnl_pct = pnl_usd / pos_info['usd_notional']
        
        pnls[pair] = {
            'pnl_usd': pnl_usd,
            'pnl_pct': pnl_pct,
            'exit_price': exit_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct
        }
    
    return pnls


# =============================================================================
# BACKTEST WITH LTR
# =============================================================================

def run_ltr_backtest(data_dir: Path,
                    start_date: str,
                    end_date: str,
                    rebalance_minutes: int,
                    top_n: int,
                    training_window_days: int = 30,
                    retrain_day: int = 5,  # Saturday = 5
                    usd_notional_per_position: float = 1_000_000,
                    model_dir: Path = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run backtest using LTR model for pair ranking.
    
    Model is retrained every Saturday using past training_window_days.
    """
    pairs = get_available_pairs(data_dir)
    print(f"Found {len(pairs)} currency pairs: {pairs}")
    
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    
    date_range = pd.date_range(start_dt, end_dt, freq='D')
    date_range = [d for d in date_range if d.weekday() < 5]  # Mon-Fri only
    
    if model_dir is None:
        model_dir = Path('working_files/ltr_models')
        model_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning LTR backtest from {start_date} to {end_date}")
    print(f"Rebalance: every {rebalance_minutes} min, Top N: {top_n}")
    print(f"Model retraining: Every Saturday using {training_window_days} days")
    print(f"USD Notional per position: ${usd_notional_per_position:,.0f}")
    print("=" * 80)
    
    # Initialize
    current_model = None
    feature_cols = None
    results = []
    trades = []
    equity = 1.0
    current_positions = {}
    
    for date in date_range:
        date_str = date.strftime('%Y%m%d')
        
        # Check if we need to retrain (every Saturday, or first day if no model yet)
        is_saturday = date.weekday() == retrain_day
        is_first_day = current_model is None
        
        if is_saturday or is_first_day:
            day_name = "Saturday" if is_saturday else date.strftime("%A")
            print(f"\n{'='*80}")
            print(f"RETRAINING MODEL on {date_str} ({day_name})")
            print(f"{'='*80}")
            
            # Training data from past N days
            train_end_date = (date - timedelta(days=1)).strftime('%Y%m%d')
            train_start_date = (date - timedelta(days=training_window_days)).strftime('%Y%m%d')
            
            df_train = generate_training_data(
                data_dir, pairs, train_start_date, train_end_date, 
                rebalance_minutes=rebalance_minutes
            )
            
            # Need reasonable training data (lower threshold for small datasets)
            min_samples = 100  # Was 1000, too high for small training windows
            
            if len(df_train) >= min_samples:  # Sufficient data to train
                print(f"Training with {len(df_train)} samples...")
                model_path = model_dir / f"ltr_model_{date_str}.txt"
                current_model = train_ltr_model(df_train, model_path)
                feature_cols = [c for c in df_train.columns if c not in ['pair', 'timestamp', 'group_id', 'label']]
            else:
                print(f"Insufficient training data ({len(df_train)} < {min_samples} samples), skipping retrain")
                continue
        
        if current_model is None:
            continue
        
        print(f"\nProcessing {date_str}...")
        
        # Load data for all pairs
        pair_data = {}
        for pair in pairs:
            df = load_pair_data(data_dir, date_str, pair)
            if df is not None and len(df) > 0:
                df = normalize_pair_data(df, pair)
                pair_data[pair] = df
        
        if len(pair_data) < 2:
            print(f"  Insufficient data for {date_str}, skipping")
            continue
        
        pair_data = align_pair_data(pair_data)
        
        # Get common timestamps
        first_pair = list(pair_data.keys())[0]
        timestamps = pair_data[first_pair]['Datetime'].values
        num_bars = len(timestamps)
        
        print(f"  Loaded {len(pair_data)} pairs, {num_bars} bars")
        
        # Run rebalances
        num_rebalances = 0
        
        for idx in range(120, num_bars, rebalance_minutes):
            timestamp = pd.Timestamp(timestamps[idx])
            
            # Close existing positions
            if current_positions:
                pnl_details = calculate_pnl(pair_data, current_positions, idx - rebalance_minutes, idx)
                
                # Sum up total PnL in USD (already includes bid/ask costs)
                total_pnl_usd = sum([p['pnl_usd'] for p in pnl_details.values()])
                
                # Calculate implicit transaction costs from bid/ask spread
                total_transaction_cost = 0.0
                for pair in current_positions:
                    spread_bp = PAIR_SPREADS.get(pair, 2.0)
                    spread_decimal = spread_bp / 10000
                    # Full spread cost (entry + exit both cross the spread)
                    total_transaction_cost += current_positions[pair]['usd_notional'] * spread_decimal
                
                # Net PnL is just total_pnl_usd (costs already embedded in bid/ask prices)
                net_pnl_usd = total_pnl_usd
                
                total_notional = sum([p['usd_notional'] for p in current_positions.values()])
                total_pnl_pct = net_pnl_usd / total_notional if total_notional > 0 else 0
                
                equity *= (1 + total_pnl_pct)
                
                # Log trades
                for pair in current_positions:
                    if pair not in pnl_details:
                        continue
                    
                    pos = current_positions[pair]
                    pnl = pnl_details[pair]
                    
                    # Calculate implicit transaction cost from bid/ask spread
                    spread_bp = PAIR_SPREADS.get(pair, 2.0)
                    spread_decimal = spread_bp / 10000
                    transaction_cost = pos['usd_notional'] * spread_decimal
                    
                    # Net PnL is what we actually made with bid/ask (already in pnl)
                    # Gross PnL is what we would have made at mid prices
                    trade_net_pnl_usd = pnl['pnl_usd']
                    trade_gross_pnl_usd = trade_net_pnl_usd + transaction_cost
                    
                    trades.append({
                        'timestamp': timestamp,
                        'date': date_str,
                        'pair': pair,
                        'execution_side': pos['execution_side'],
                        'currency_position': pos['currency_position'],
                        'entry_price': pos['entry_price'],
                        'exit_price': pnl['exit_price'],
                        'price_change': pnl['price_change'],
                        'price_change_pct': pnl['price_change_pct'],
                        'usd_notional': pos['usd_notional'],
                        'base_notional': pos['base_notional'],
                        'quote_notional': pos['quote_notional'],
                        'pnl_gross': trade_gross_pnl_usd,  # At mid prices
                        'transaction_cost': transaction_cost,  # Implicit cost
                        'pnl_net': trade_net_pnl_usd,  # At bid/ask (actual)
                        'pnl_pct': pnl['pnl_pct'],
                        'predicted_score': pos['predicted_score'],
                    })
            
            # Select new positions using LTR model
            new_positions = select_positions_ltr(
                current_model, pair_data, idx, timestamp, top_n, 
                feature_cols, usd_notional_per_position
            )
            
            current_positions = new_positions
            num_rebalances += 1
        
        print(f"  Completed {num_rebalances} rebalances, Equity: {equity:.4f}")
    
    # Create summary DataFrames
    df_trades = pd.DataFrame(trades)
    
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    
    if len(df_trades) > 0:
        # Calculate statistics
        total_pnl_gross = df_trades['pnl_gross'].sum() if 'pnl_gross' in df_trades.columns else 0
        total_transaction_cost = df_trades['transaction_cost'].sum() if 'transaction_cost' in df_trades.columns else 0
        total_pnl_net = df_trades['pnl_net'].sum() if 'pnl_net' in df_trades.columns else df_trades['pnl_usd'].sum()
        num_trades = len(df_trades)
        win_rate = (df_trades['pnl_net'] > 0).sum() / num_trades * 100 if 'pnl_net' in df_trades.columns else (df_trades['pnl_usd'] > 0).sum() / num_trades * 100
        avg_win = df_trades[df_trades['pnl_net'] > 0]['pnl_pct'].mean() * 100 if 'pnl_net' in df_trades.columns and len(df_trades[df_trades['pnl_net'] > 0]) > 0 else 0
        avg_loss = df_trades[df_trades['pnl_net'] < 0]['pnl_pct'].mean() * 100 if 'pnl_net' in df_trades.columns and len(df_trades[df_trades['pnl_net'] < 0]) > 0 else 0
        
        # Count unique dates and rebalances
        df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
        unique_dates = df_trades['date'].nunique()
        unique_date_hours = df_trades.groupby(['date', df_trades['timestamp'].dt.hour]).ngroups
        num_signals = num_trades  # Each trade comes from a signal
        
        print(f"Total Return (%)........................ {(equity - 1) * 100:15.4f}")
        print(f"Final Equity............................ {equity:15.4f}")
        print(f"Unique Trading Days..................... {unique_dates:15.0f}")
        print(f"Rebalance Periods....................... {unique_date_hours:15.0f}")
        print(f"Total Signals........................... {num_signals:15.0f}")
        print(f"Total Trades............................ {num_trades:15.0f}")
        print(f"PnL Gross (USD)......................... {total_pnl_gross:15.4f}")
        print(f"Transaction Costs (USD)................. {total_transaction_cost:15.4f}")
        print(f"PnL Net (USD)........................... {total_pnl_net:15.4f}")
        print(f"Win Rate (%)............................ {win_rate:15.4f}")
        print(f"Avg Win (%)............................ {avg_win:15.4f}")
        print(f"Avg Loss (%)............................ {avg_loss:15.4f}")
    
    print("=" * 80)
    
    return None, df_trades


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='FX Learn-to-Rank Momentum Strategy')
    parser.add_argument('--start-date', type=str, required=True, help='Start date MMDDYYYY')
    parser.add_argument('--end-date', type=str, required=True, help='End date MMDDYYYY')
    parser.add_argument('--rebalance-freq', type=int, default=60, help='Rebalance frequency in minutes')
    parser.add_argument('--top-n', type=int, default=2, help='Number of pairs to long/short')
    parser.add_argument('--training-days', type=int, default=30, help='Training window in days')
    parser.add_argument('--data-dir', type=str, default='data/fx', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='working_files/ltr_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Convert date format
    start_date = datetime.strptime(args.start_date, '%m%d%Y').strftime('%Y%m%d')
    end_date = datetime.strptime(args.end_date, '%m%d%Y').strftime('%Y%m%d')
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run backtest
    summary_df, trades_df = run_ltr_backtest(
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        rebalance_minutes=args.rebalance_freq,
        top_n=args.top_n,
        training_window_days=args.training_days,
    )
    
    # Save results
    if trades_df is not None and len(trades_df) > 0:
        trades_path = output_dir / f"ltr_trades_{start_date}_{end_date}_rebal{args.rebalance_freq}_top{args.top_n}.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"\nTrade details saved to: {trades_path}")
        
        # Show sample trades
        print("\nSample trades (first 10):")
        print(trades_df.head(10).to_string())
