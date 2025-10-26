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
# DATA UTILITIES
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
    """
    Normalize pair data to USD-relative prices.
    BIDASK VERSION: Uses real bid/offer columns from data files.
    """
    df = df.copy()
    
    # Rename 'offer' to 'ask' for consistency with code
    if 'offer' in df.columns:
        df['ask'] = df['offer']
    
    # Transform all price columns to USD-relative
    for col in ['open', 'high', 'low', 'close', 'bid', 'ask']:
        if col in df.columns:
            df[f'{col}_usd'] = df[col].apply(lambda x: get_usd_relative_price(pair, x))
    
    # Calculate returns based on transformed prices
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
        # Forward fill prices AND bid/ask
        fill_cols = ['open', 'high', 'low', 'close']
        if 'bid' in merged.columns:
            fill_cols.append('bid')
        if 'ask' in merged.columns:
            fill_cols.append('ask')
        if 'offer' in merged.columns:
            fill_cols.append('offer')
        merged[fill_cols] = merged[fill_cols].ffill()
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
    BIDASK VERSION: Uses real bid/ask prices for entry.
    
    Returns dict of pair -> position info
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
        
        # Use REAL bid/ask from data
        bid_price = row['bid']
        ask_price = row['ask']
        
        if pair in top_pairs:
            # Top ranked = base currency STRONG vs USD → SHORT base (LONG USD)
            if is_usd_inverse(pair):
                # Original: USD/JPY → LONG USD/JPY
                direction = +1
                execution_side = 'BUY'
                currency_position = 'LONG_USD'
                entry_price = ask_price  # Pay ask when buying
            else:
                # Original: GBP/USD → SHORT GBP/USD
                direction = -1
                execution_side = 'SELL'
                currency_position = 'LONG_USD'
                entry_price = bid_price  # Receive bid when selling
            
            if is_usd_inverse(pair):
                pair_notional = usd_notional
            else:
                pair_notional = usd_notional / entry_price
            
            positions[pair] = {
                'direction': direction,
                'usd_notional': usd_notional,
                'pair_notional': pair_notional,
                'entry_price': entry_price,
                'predicted_score': predicted_score,
                'execution_side': execution_side,
                'currency_position': currency_position,
            }
        
        elif pair in bottom_pairs:
            # Bottom ranked = base currency WEAK vs USD → LONG base (SHORT USD)
            if is_usd_inverse(pair):
                # Original: USD/JPY → SHORT USD/JPY
                direction = -1
                execution_side = 'SELL'
                currency_position = 'SHORT_USD'
                entry_price = bid_price  # Receive bid when selling
            else:
                # Original: GBP/USD → LONG GBP/USD
                direction = +1
                execution_side = 'BUY'
                currency_position = 'SHORT_USD'
                entry_price = ask_price  # Pay ask when buying
            
            if is_usd_inverse(pair):
                pair_notional = usd_notional
            else:
                pair_notional = usd_notional / entry_price
            
            positions[pair] = {
                'direction': direction,
                'usd_notional': usd_notional,
                'pair_notional': pair_notional,
                'entry_price': entry_price,
                'predicted_score': predicted_score,
                'execution_side': execution_side,
                'currency_position': currency_position,
            }
    
    return positions


# =============================================================================
# PNL CALCULATION
# =============================================================================

def calculate_pnl(pair_data: Dict[str, pd.DataFrame],
                 positions: Dict[str, Dict],
                 entry_idx: int,
                 exit_idx: int) -> Dict[str, Dict]:
    """
    Calculate PnL for positions from entry to exit.
    BIDASK VERSION: Uses real bid/ask for exits.
    """
    pnls = {}
    
    for pair, pos_info in positions.items():
        # Skip if pair data not available (data gap)
        if pair not in pair_data:
            continue
        
        entry_price = pos_info['entry_price']
        
        # Get exit price using REAL bid/ask
        row = pair_data[pair].iloc[exit_idx]
        if pos_info['direction'] > 0:
            # LONG position: exit by selling, receive bid
            exit_price = row['bid']
        else:
            # SHORT position: exit by buying, pay ask
            exit_price = row['ask']
        
        direction = pos_info['direction']
        pair_notional = pos_info['pair_notional']
        
        price_change = exit_price - entry_price
        price_change_pct = price_change / entry_price
        
        # Calculate PnL in USD
        if is_usd_inverse(pair):
            pnl_usd = direction * pair_notional * price_change_pct
        else:
            pnl_usd = direction * pair_notional * price_change
        
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
                    model_dir: Path = None,
                    trading_hours: Tuple[int, int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run backtest using LTR model for pair ranking.
    BIDASK VERSION: Uses real bid/ask data from data/bidask/output/
    
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
    if trading_hours:
        print(f"Trading Hours: {trading_hours[0]:02d}:00 - {trading_hours[1]:02d}:00 EST")
    print(f"Data Source: {data_dir} (REAL BID/ASK)")
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
            
            min_samples = 100
            
            if len(df_train) >= min_samples:
                print(f"Training with {len(df_train)} samples...")
                model_path = model_dir / f"ltr_model_bidask_{date_str}.txt"
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
            
            # Apply trading hours filter if specified
            if trading_hours:
                est = pytz.timezone('US/Eastern')
                timestamp_est = timestamp.tz_localize('UTC').tz_convert(est)
                hour_est = timestamp_est.hour
                
                if hour_est < trading_hours[0] or hour_est >= trading_hours[1]:
                    continue
            
            # Close existing positions
            if current_positions:
                pnl_details = calculate_pnl(pair_data, current_positions, idx - rebalance_minutes, idx)
                
                total_pnl_usd = sum([p['pnl_usd'] for p in pnl_details.values()])
                total_notional = sum([p['usd_notional'] for p in current_positions.values()])
                total_pnl_pct = total_pnl_usd / total_notional if total_notional > 0 else 0
                
                equity *= (1 + total_pnl_pct)
                
                # Log trades
                for pair in current_positions:
                    if pair not in pnl_details:
                        continue
                    
                    pos = current_positions[pair]
                    pnl = pnl_details[pair]
                    
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
                        'pair_notional': pos['pair_notional'],
                        'pnl_usd': pnl['pnl_usd'],
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
        total_pnl_net = df_trades['pnl_usd'].sum()
        num_trades = len(df_trades)
        win_rate = (df_trades['pnl_usd'] > 0).sum() / num_trades * 100
        avg_win = df_trades[df_trades['pnl_usd'] > 0]['pnl_pct'].mean() * 100 if len(df_trades[df_trades['pnl_usd'] > 0]) > 0 else 0
        avg_loss = df_trades[df_trades['pnl_usd'] < 0]['pnl_pct'].mean() * 100 if len(df_trades[df_trades['pnl_usd'] < 0]) > 0 else 0
        volume_m = df_trades['usd_notional'].sum() / 1_000_000
        
        print(f"Total Return (%)........................ {(equity - 1) * 100:15.4f}")
        print(f"Final Equity............................ {equity:15.4f}")
        print(f"PnL Net (USD)........................... {total_pnl_net:15.4f}")
        print(f"Volume Traded ($M)...................... {volume_m:15.1f}")
        print(f"Net per $1M Traded...................... {total_pnl_net/volume_m:15.2f}")
        print(f"Num Trades.............................. {num_trades:15.0f}")
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
    
    parser = argparse.ArgumentParser(description='FX Learn-to-Rank Momentum Strategy (BIDASK)')
    parser.add_argument('--start-date', type=str, required=True, help='Start date MMDDYYYY')
    parser.add_argument('--end-date', type=str, required=True, help='End date MMDDYYYY')
    parser.add_argument('--rebalance-freq', type=int, default=60, help='Rebalance frequency in minutes')
    parser.add_argument('--top-n', type=int, default=2, help='Number of pairs to long/short')
    parser.add_argument('--training-days', type=int, default=30, help='Training window in days')
    parser.add_argument('--start-hour', type=int, default=None, help='Trading start hour in EST')
    parser.add_argument('--end-hour', type=int, default=None, help='Trading end hour in EST')
    parser.add_argument('--output-dir', type=str, default='working_files/ltr_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Convert date format
    start_date = datetime.strptime(args.start_date, '%m%d%Y').strftime('%Y%m%d')
    end_date = datetime.strptime(args.end_date, '%m%d%Y').strftime('%Y%m%d')
    
    # BIDASK DATA DIRECTORY
    data_dir = Path('data/bidask/output')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Trading hours filter
    trading_hours = None
    if args.start_hour is not None and args.end_hour is not None:
        trading_hours = (args.start_hour, args.end_hour)
    
    # Run backtest
    summary_df, trades_df = run_ltr_backtest(
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        rebalance_minutes=args.rebalance_freq,
        top_n=args.top_n,
        training_window_days=args.training_days,
        trading_hours=trading_hours,
    )
    
    # Save results
    if trades_df is not None and len(trades_df) > 0:
        hours_suffix = f"_h{trading_hours[0]}-{trading_hours[1]}" if trading_hours else ""
        trades_path = output_dir / f"ltr_trades_bidask_{start_date}_{end_date}_rebal{args.rebalance_freq}_top{args.top_n}{hours_suffix}.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"\nTrade details saved to: {trades_path}")
        
        # Show sample trades
        print("\nSample trades (first 10):")
        print(trades_df.head(10).to_string())
