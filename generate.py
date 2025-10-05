
import click
from datetime import datetime,timedelta
import pandas as pd
import os
import numpy as np
import traceback
import logging
import yaml
import os
from pathlib import Path
from joblib import Parallel, delayed


def find_next_log_filename(log_dir, base_filename):
    """Find the next available log filename with incrementing number"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    counter = 1
    while True:
        log_filename = log_dir / f"{base_filename}_{counter:04d}.log"
        if not log_filename.exists():
            return str(log_filename)
        counter += 1


def setup_logging(log_filename):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )


def validate_labels(settings):
    """Validate label configuration in settings"""
    required_fields = ['spot']
    for field in required_fields:
        if field not in settings:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Set default label configurations if not provided
    if 'labels' not in settings:
        settings['labels'] = {
            'conservative': {'pt': 0.001, 'sl': 0.001, 'max_hold': 24},
            'aggressive': {'pt': 0.002, 'sl': 0.001, 'max_hold': 48},
        }
    
    # Validate label format (support both dict and list for backwards compatibility)
    if isinstance(settings['labels'], dict):
        # New format: named configurations
        for name, label_config in settings['labels'].items():
            if not isinstance(label_config, dict):
                raise ValueError(f"Label '{name}' should be a dict with pt, sl, max_hold keys")
            required_keys = ['pt', 'sl', 'max_hold']
            for key in required_keys:
                if key not in label_config:
                    raise ValueError(f"Label '{name}' missing required key: {key}")
    elif isinstance(settings['labels'], list):
        # Old format: list of [pt, sl, max_hold] - convert to dict
        logging.warning("Using deprecated list format for labels, consider using named dict format")
        converted_labels = {}
        for i, label_config in enumerate(settings['labels']):
            if not isinstance(label_config, list) or len(label_config) != 3:
                raise ValueError(f"Label config {i} should be [pt, sl, max_hold], got {label_config}")
            pt, sl, max_hold = label_config
            converted_labels[f'config_{i}'] = {'pt': pt, 'sl': sl, 'max_hold': max_hold}
        settings['labels'] = converted_labels
    else:
        raise ValueError("Labels should be a dict of named configurations or list of [pt, sl, max_hold] configurations")
    
    logging.info(f"Validated config for {len(settings.get('labels', {}))} label configurations: {list(settings['labels'].keys())}")


def add_lagged_profit_features(features_df, labels_df, strategy_name='', lookback_days=5):
    """
    Add lagged profit features to help predict future performance.
    
    Creates 11 features per strategy:
    1. profit_lag1, profit_lag2, profit_lag3, profit_lag5
    2. profit_rolling_mean_5, profit_rolling_mean_10
    3. profit_rolling_std_5
    4. winning_streak, losing_streak
    5. recent_win_rate (10-period rolling)
    6. profit_volatility
    
    All features use .shift(1) to prevent data leakage.
    
    Args:
        features_df: DataFrame with index matching signals
        labels_df: DataFrame containing 'profit' column
        strategy_name: Prefix for feature names (e.g., 'cons', 'ma')
        lookback_days: Not used anymore (kept for compatibility)
    
    Returns:
        Tuple of (features_df, list of added feature names)
    """
    features_df = features_df.copy()
    
    # Create name prefix
    prefix = f'{strategy_name}_' if strategy_name else ''
    added_features = []
    
    # Get profit series aligned with features
    profit = labels_df['profit']
    
    # 1. Simple lags
    for lag in [1, 2, 3, 5]:
        col_name = f'{prefix}profit_lag{lag}'
        features_df[col_name] = profit.shift(lag)
        added_features.append(col_name)
    
    # 2. Rolling statistics
    col_name = f'{prefix}profit_rolling_mean_5'
    features_df[col_name] = profit.shift(1).rolling(window=5, min_periods=1).mean()
    added_features.append(col_name)
    
    col_name = f'{prefix}profit_rolling_std_5'
    features_df[col_name] = profit.shift(1).rolling(window=5, min_periods=1).std()
    added_features.append(col_name)
    
    col_name = f'{prefix}profit_rolling_mean_10'
    features_df[col_name] = profit.shift(1).rolling(window=10, min_periods=1).mean()
    added_features.append(col_name)
    
    # 3. Winning/losing streak
    wins = (profit > 0).astype(int)
    
    def calculate_streak(series):
        """Calculate current winning/losing streak"""
        result = pd.Series(index=series.index, dtype=float)
        current_streak = 0
        
        for idx in series.index:
            w = series.loc[idx]
            if w == 1:
                current_streak = max(1, current_streak + 1) if current_streak > 0 else 1
            else:
                current_streak = min(-1, current_streak - 1) if current_streak < 0 else -1
            result.loc[idx] = current_streak
        
        return result
    
    streaks = calculate_streak(wins)
    
    col_name = f'{prefix}winning_streak'
    features_df[col_name] = (streaks > 0).astype(int) * streaks  # Positive int for wins
    features_df[col_name] = features_df[col_name].shift(1).fillna(0)
    added_features.append(col_name)
    
    col_name = f'{prefix}losing_streak'
    features_df[col_name] = (streaks < 0).astype(int) * abs(streaks)  # Positive int for losses
    features_df[col_name] = features_df[col_name].shift(1).fillna(0)
    added_features.append(col_name)
    
    # 4. Regime detection: Are we in a "good" or "bad" period for this strategy?
    recent_win_rate = profit.shift(1).rolling(window=10, min_periods=3).apply(lambda x: (x > 0).sum() / len(x))
    
    col_name = f'{prefix}recent_win_rate'
    features_df[col_name] = recent_win_rate
    added_features.append(col_name)
    
    # 5. Volatility of recent profits (high volatility = uncertain regime)
    col_name = f'{prefix}profit_volatility'
    features_df[col_name] = profit.shift(1).rolling(window=5, min_periods=2).std()
    added_features.append(col_name)
    
    return features_df, added_features


def calculate_technical_indicators(df, fast_window=5, slow_window=20):
    """
    Calculate technical indicators for features
    
    Parameters:
    - df: DataFrame with OHLC data
    - fast_window: Span for fast EMA (default: 5)
    - slow_window: Span for slow EMA (default: 20)
    """
    df = df.copy()
    
    # Exponential Moving Averages - use configurable windows
    df['ma_5'] = df['close'].ewm(span=fast_window, adjust=False).mean()
    df['ma_20'] = df['close'].ewm(span=slow_window, adjust=False).mean()
    df['ma_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    
    # Volatility features
    df['volatility_5'] = df['returns'].rolling(window=5).std()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    
    # RSI-like momentum
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Band position
    df['bb_upper'] = df['ma_20'] + (df['volatility_20'] * 2)
    df['bb_lower'] = df['ma_20'] - (df['volatility_20'] * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    return df


def add_signal_adjusted_features(features_df, signals):
    """
    Add signal-adjusted directional features.
    
    Some features are directional - their interpretation depends on whether 
    we're going long (+1) or short (-1). For example:
    - Long position: positive returns are good
    - Short position: negative returns are good (positive returns bad)
    
    We multiply these features by the signal direction so that:
    - Positive adjusted feature = move in our favor
    - Negative adjusted feature = move against us
    
    Args:
        features_df: DataFrame with features
        signals: Series with signal direction (1=long, -1=short)
    
    Returns:
        features_df with additional signal-adjusted features
    """
    features_df = features_df.copy()
    
    # Directional features that should be adjusted by signal
    directional_features = [
        'returns',          # Price returns
        'log_returns',      # Log returns
    ]
    
    # Add signal-adjusted versions
    for feature in directional_features:
        if feature in features_df.columns:
            adjusted_name = f'{feature}_adj'
            features_df[adjusted_name] = features_df[feature] * signals
    
    # Add momentum indicator (distance from MAs) - also directional
    if 'ma_5' in features_df.columns and 'close' in features_df.columns:
        # When long, being above MA is good. When short, being below MA is good.
        ma_distance = (features_df['close'] - features_df['ma_5']) / features_df['ma_5']
        features_df['ma5_distance_adj'] = ma_distance * signals
    
    if 'ma_20' in features_df.columns and 'close' in features_df.columns:
        ma_distance = (features_df['close'] - features_df['ma_20']) / features_df['ma_20']
        features_df['ma20_distance_adj'] = ma_distance * signals
    
    return features_df


def generate_ma_cross_signals(df, fast_window=5, slow_window=20):
    """
    Generate exponential moving average crossover signals
    
    Parameters:
    - df: DataFrame with OHLC data
    - fast_window: Span for fast EMA (default: 5)
    - slow_window: Span for slow EMA (default: 20)
    
    Returns:
    - signals: Series with 1 (bullish), -1 (bearish), 0 (no signal)
    """
    signals = pd.Series(index=df.index, data=0, name='signal')
    
    # Calculate EMAs with configurable windows
    ma_fast = df['close'].ewm(span=fast_window, adjust=False).mean()
    ma_slow = df['close'].ewm(span=slow_window, adjust=False).mean()
    
    # Generate signals
    # 1 for bullish crossover (fast MA crosses above slow MA)
    # -1 for bearish crossover (fast MA crosses below slow MA)
    # 0 for no signal
    
    ma_diff = ma_fast - ma_slow
    ma_diff_prev = ma_diff.shift(1)
    
    # Bullish crossover: current diff > 0 and previous diff <= 0
    bullish_cross = (ma_diff > 0) & (ma_diff_prev <= 0)
    
    # Bearish crossover: current diff < 0 and previous diff >= 0
    bearish_cross = (ma_diff < 0) & (ma_diff_prev >= 0)
    
    signals[bullish_cross] = 1
    signals[bearish_cross] = -1
    
    return signals


def generate_labels(df, signals, pt_pct, sl_pct, max_hold_hours):
    """
    Generate forward-looking labels for ML training
    pt_pct: profit target as percentage (e.g., 0.001 = 0.1%)
    sl_pct: stop loss as percentage 
    max_hold_hours: maximum holding period in hours
    signals: Series with signal direction (1=long, -1=short, 0=no signal)
    
    Returns DataFrame with columns:
    - label: binary 1 (profitable) or -1 (loss)
    - profit: actual profit/loss in PRICE UNITS (e.g., 0.367 for USDJPY = 36.7 pips)
    - exit_bars: number of bars until exit
    - entry_timestamp: timestamp of entry
    - exit_timestamp: timestamp of exit
    - entry_price: price at entry
    - exit_price: price at exit
    - signal_direction: direction of entry signal (1=long, -1=short)
    - ma_fast: fast MA value at entry
    - ma_slow: slow MA value at entry
    """
    labels = pd.DataFrame(index=df.index, columns=[
        'label', 'profit', 'exit_bars', 
        'entry_timestamp', 'exit_timestamp', 'entry_price', 'exit_price',
        'signal_direction', 'ma_fast', 'ma_slow'
    ])
    labels['label'] = np.nan
    labels['profit'] = np.nan
    labels['exit_bars'] = np.nan
    labels['entry_timestamp'] = pd.NaT
    labels['exit_timestamp'] = pd.NaT
    labels['entry_price'] = np.nan
    labels['exit_price'] = np.nan
    labels['signal_direction'] = np.nan
    labels['ma_fast'] = np.nan
    labels['ma_slow'] = np.nan
    
    # Process all bars - even near the end we can create labels with available data
    for i in range(len(df) - 1):  # Need at least 1 future bar
        entry_price = df['close'].iloc[i]
        entry_timestamp = df.index[i]
        
        # Get signal direction (1=long, -1=short)
        signal_direction = signals.iloc[i] if i < len(signals) else 0
        if signal_direction == 0:
            continue  # Skip if no signal
        
        # Get MA values at entry (if available in df)
        ma_fast_val = df['ma_5'].iloc[i] if 'ma_5' in df.columns else np.nan
        ma_slow_val = df['ma_20'].iloc[i] if 'ma_20' in df.columns else np.nan
        
        # Look forward up to max_hold_hours (or whatever data is available)
        max_future_bars = min(max_hold_hours, len(df) - i - 1)
        future_prices = df['close'].iloc[i+1:i+1+max_future_bars]
        
        if len(future_prices) == 0:
            continue
            
        # Calculate returns from entry - ADJUSTED FOR SIGNAL DIRECTION
        # For long positions (signal_direction=1): profit when price goes up
        # For short positions (signal_direction=-1): profit when price goes down
        price_change_pct = (future_prices - entry_price) / entry_price
        returns = price_change_pct * signal_direction  # Flip returns for short positions
        
        # Check for profit target hit (positive return in direction of trade)
        pt_hit = returns >= pt_pct
        # Check for stop loss hit (negative return in direction of trade)
        sl_hit = returns <= -sl_pct
        
        exit_idx = None
        exit_price = None
        exit_return = None
        
        if pt_hit.any():
            # Profit target hit at some point
            pt_idx = pt_hit.idxmax()
            pt_bar = (pt_idx - entry_timestamp).total_seconds() / 3600 if hasattr(entry_timestamp, 'total_seconds') else pt_hit.values.argmax() + 1
            
            if sl_hit.any():
                sl_idx = sl_hit.idxmax()
                sl_bar = (sl_idx - entry_timestamp).total_seconds() / 3600 if hasattr(entry_timestamp, 'total_seconds') else sl_hit.values.argmax() + 1
                
                if pt_idx <= sl_idx:
                    # Profit target hit first
                    exit_idx = pt_idx
                    exit_price = df['close'].loc[pt_idx]
                    labels.iloc[i, 0] = 1  # label
                    labels.iloc[i, 1] = (exit_price - entry_price) * signal_direction  # profit in price units
                    labels.iloc[i, 2] = pt_bar  # exit_bars
                else:
                    # Stop loss hit first
                    exit_idx = sl_idx
                    exit_price = df['close'].loc[sl_idx]
                    labels.iloc[i, 0] = -1  # label
                    labels.iloc[i, 1] = (exit_price - entry_price) * signal_direction  # profit in price units (negative)
                    labels.iloc[i, 2] = sl_bar  # exit_bars
            else:
                # Only profit target hit
                exit_idx = pt_idx
                exit_price = df['close'].loc[pt_idx]
                labels.iloc[i, 0] = 1  # label
                labels.iloc[i, 1] = (exit_price - entry_price) * signal_direction  # profit in price units
                labels.iloc[i, 2] = pt_bar  # exit_bars
        elif sl_hit.any():
            # Only stop loss hit
            sl_idx = sl_hit.idxmax()
            exit_idx = sl_idx
            exit_price = df['close'].loc[sl_idx]
            sl_bar = (sl_idx - entry_timestamp).total_seconds() / 3600 if hasattr(entry_timestamp, 'total_seconds') else sl_hit.values.argmax() + 1
            labels.iloc[i, 0] = -1  # label
            labels.iloc[i, 1] = (exit_price - entry_price) * signal_direction  # profit in price units (negative)
            labels.iloc[i, 2] = sl_bar  # exit_bars
        else:
            # Neither hit within max_hold_hours - use final return
            final_return = returns.iloc[-1]
            exit_idx = future_prices.index[-1]
            exit_price = df['close'].loc[exit_idx]
            labels.iloc[i, 0] = 1 if final_return > 0 else -1  # label
            labels.iloc[i, 1] = (exit_price - entry_price) * signal_direction  # profit in price units
            labels.iloc[i, 2] = len(future_prices)  # exit_bars (max hold)
        
        # Set entry/exit timestamps and prices
        labels.iloc[i, 3] = entry_timestamp  # entry_timestamp
        labels.iloc[i, 4] = exit_idx  # exit_timestamp
        labels.iloc[i, 5] = entry_price  # entry_price
        labels.iloc[i, 6] = exit_price  # exit_price
        labels.iloc[i, 7] = signal_direction  # signal_direction
        labels.iloc[i, 8] = ma_fast_val  # ma_fast
        labels.iloc[i, 9] = ma_slow_val  # ma_slow
    
    return labels


def generate_labels_ma_exit(df, signals, pt_pct, sl_pct, max_hold_hours):
    """
    Generate labels using MA crossover signals as exits (more realistic for MA strategies)
    
    Strategy:
    - Enter long on bullish crossover (signal = 1)
    - Enter short on bearish crossover (signal = -1)
    - Exit/reverse position on opposite signal
    - Also exit on PT/SL or max_hold if no signal
    
    Parameters:
    - df: DataFrame with OHLC data
    - signals: Series with MA crossover signals (1, -1, 0)
    - pt_pct: profit target percentage
    - sl_pct: stop loss percentage
    - max_hold_hours: maximum hold period
    
    Returns DataFrame with columns:
    - label: binary 1 (profitable) or -1 (loss)
    - profit: actual profit/loss in PRICE UNITS (e.g., 0.367 for USDJPY = 36.7 pips)
    - exit_bars: number of bars until exit
    - exit_reason: 'signal', 'pt', 'sl', or 'max_hold'
    - entry_timestamp: timestamp of entry
    - exit_timestamp: timestamp of exit
    - entry_price: price at entry
    - exit_price: price at exit
    - signal_direction: signal at entry (1=long, -1=short)
    - ma_fast: fast MA value at entry
    - ma_slow: slow MA value at entry
    """
    labels = pd.DataFrame(index=df.index, columns=[
        'label', 'profit', 'exit_bars', 'exit_reason',
        'entry_timestamp', 'exit_timestamp', 'entry_price', 'exit_price',
        'signal_direction', 'ma_fast', 'ma_slow'
    ])
    labels['label'] = np.nan
    labels['profit'] = np.nan
    labels['exit_bars'] = np.nan
    labels['exit_reason'] = ''
    labels['entry_timestamp'] = pd.NaT
    labels['exit_timestamp'] = pd.NaT
    labels['entry_price'] = np.nan
    labels['exit_price'] = np.nan
    labels['signal_direction'] = np.nan
    labels['ma_fast'] = np.nan
    labels['ma_slow'] = np.nan
    
    # Process ALL bars (signals can appear anywhere, even near the end)
    for i in range(len(df)):
        # Check if there's a signal at this bar
        current_signal = signals.iloc[i]
        
        if current_signal == 0:
            # No entry signal
            continue
        
        entry_price = df['close'].iloc[i]
        entry_timestamp = df.index[i]
        position_direction = current_signal  # 1 for long, -1 for short
        
        # Get MA values at entry (if available in df)
        ma_fast_val = df['ma_5'].iloc[i] if 'ma_5' in df.columns else np.nan
        ma_slow_val = df['ma_20'].iloc[i] if 'ma_20' in df.columns else np.nan
        
        # Look forward to find exit
        exit_idx = None
        exit_timestamp = None
        exit_price = None
        exit_reason = None
        exit_profit = None
        
        for j in range(1, min(max_hold_hours + 1, len(df) - i)):
            future_idx = i + j
            future_price = df['close'].iloc[future_idx]
            future_timestamp = df.index[future_idx]
            future_signal = signals.iloc[future_idx]
            
            # Calculate profit based on position direction
            price_change = (future_price - entry_price) / entry_price
            profit_pct = price_change * position_direction  # Long: +return, Short: -return
            
            # Check for opposite signal (exit/reverse)
            if future_signal != 0 and future_signal != position_direction:
                exit_idx = future_idx
                exit_timestamp = future_timestamp
                exit_price = future_price
                exit_reason = 'signal'
                exit_profit = (exit_price - entry_price) * position_direction  # Price units
                break
            
            # Check for profit target
            if profit_pct >= pt_pct:
                exit_idx = future_idx
                exit_timestamp = future_timestamp
                exit_price = future_price
                exit_reason = 'pt'
                exit_profit = (exit_price - entry_price) * position_direction  # Price units
                break
            
            # Check for stop loss
            if profit_pct <= -sl_pct:
                exit_idx = future_idx
                exit_timestamp = future_timestamp
                exit_price = future_price
                exit_reason = 'sl'
                exit_profit = (exit_price - entry_price) * position_direction  # Price units (negative)
                break
        
        # If no exit found, use max hold
        if exit_idx is None:
            future_idx = min(i + max_hold_hours, len(df) - 1)
            future_price = df['close'].iloc[future_idx]
            future_timestamp = df.index[future_idx]
            exit_idx = future_idx
            exit_timestamp = future_timestamp
            exit_price = future_price
            exit_reason = 'max_hold'
            exit_profit = (exit_price - entry_price) * position_direction  # Price units
        
        # Record the label
        labels.iloc[i, 0] = 1 if exit_profit > 0 else -1  # label
        labels.iloc[i, 1] = exit_profit  # profit
        labels.iloc[i, 2] = exit_idx - i  # exit_bars
        labels.iloc[i, 3] = exit_reason  # exit_reason
        labels.iloc[i, 4] = entry_timestamp  # entry_timestamp
        labels.iloc[i, 5] = exit_timestamp  # exit_timestamp
        labels.iloc[i, 6] = entry_price  # entry_price
        labels.iloc[i, 7] = exit_price  # exit_price
        labels.iloc[i, 8] = position_direction  # signal_direction
        labels.iloc[i, 9] = ma_fast_val  # ma_fast
        labels.iloc[i, 10] = ma_slow_val  # ma_slow
    
    return labels


def run_date_offset(date_obj, spot, config, offset=0):
    """
    Core processing function for a specific date
    Returns signals, features, labels, and tick data
    """
    # Load the data file
    date_str = date_obj.strftime("%Y%m%d")
    data_file = f"data/output/{date_str}_{spot}_bars.csv"
    
    if not os.path.exists(data_file):
        logging.warning(f"Data file not found: {data_file}")
        # Return empty dataframes
        empty_index = pd.DatetimeIndex([], name='Datetime')
        return (
            pd.Series([], name='signal', dtype=float),
            pd.DataFrame(index=empty_index),
            {},
            pd.DataFrame(index=empty_index)
        )
    
    # Load and prepare data
    df = pd.read_csv(data_file, parse_dates=['Datetime'], index_col='Datetime')
    
    # No offset processing needed anymore
    
    # Get MA window parameters from config (with defaults)
    ma_windows = config.get('ma_windows', {})
    fast_window = ma_windows.get('fast', 5)
    slow_window = ma_windows.get('slow', 20)
    
    logging.info(f"Using MA windows: fast={fast_window}, slow={slow_window}")
    
    # Generate signals using moving average crossover
    signals = generate_ma_cross_signals(df, fast_window=fast_window, slow_window=slow_window)
    
    # Calculate technical indicators for features (using same MA windows)
    features_df = calculate_technical_indicators(df, fast_window=fast_window, slow_window=slow_window)
    
    # IMPORTANT: Filter to only signal bars (MA crossovers)
    # We only want features/labels for bars where we'd actually trade
    signal_mask = signals != 0
    logging.info(f"Filtering to {signal_mask.sum()} signal bars (out of {len(df)} total bars)")
    
    # Filter signals, features, and df to only signal bars
    signals = signals[signal_mask]
    features_df = features_df[signal_mask]
    df = df[signal_mask].copy()
    
    # Add signal-adjusted directional features
    features_df = add_signal_adjusted_features(features_df, signals)
    
    # Define feature columns to use
    # 1. Technical indicators we calculate
    technical_features = [
        'ma_5', 'ma_20', 'ma_50', 'returns', 'log_returns', 
        'price_range', 'body_size', 'volatility_5', 'volatility_20',
        'rsi', 'bb_position',
        # Signal-adjusted directional features
        'returns_adj', 'log_returns_adj', 'ma5_distance_adj', 'ma20_distance_adj'
    ]
    
    # 2. Pre-computed bar statistics (from tick-level data)
    bar_statistics = [
        'afmlvol',           # AFML volume
        'vol_skew',          # Volatility skewness
        'vol_kurt',          # Volatility kurtosis
        'vol_of_vol',        # Volatility of volatility
        'vol_z',             # Volatility z-score
        'vol_ratio',         # Volatility ratio
        'vol_slope',         # Volatility slope
        'vol_autocorr1',     # Volatility autocorrelation lag 1
        'rv',                # Realized volatility
        'bpv',               # Bipower variation
        'rq',                # Realized quarticity
        'jump_proxy',        # Jump detection proxy
        'tail_exceed_rate',  # Tail exceedance rate
        'dvol',              # Dollar volume
        'dvol_sign',         # Dollar volume sign
        'seconds_diff'       # Time between bars
    ]
    
    # Combine all feature columns
    feature_columns = technical_features + bar_statistics
    
    # Keep only feature columns that exist in the data
    available_features = [col for col in feature_columns if col in features_df.columns]
    
    # Log which features are being used
    missing_bar_stats = [col for col in bar_statistics if col not in features_df.columns]
    if missing_bar_stats:
        logging.warning(f"Missing bar statistics: {missing_bar_stats}")
    
    logging.info(f"Using {len(available_features)} features: {len([f for f in available_features if f in technical_features])} technical + {len([f for f in available_features if f in bar_statistics])} bar statistics")
    
    features = features_df[available_features].dropna()
    
    # Add MA columns to df for label generation (so we can record MA values at entry)
    df['ma_5'] = features_df['ma_5']
    df['ma_20'] = features_df['ma_20']
    
    # Generate labels for different configurations
    label_configs = config.get('labels', {
        'conservative': {'pt': 0.001, 'sl': 0.001, 'max_hold': 24},
        'aggressive': {'pt': 0.002, 'sl': 0.001, 'max_hold': 48},
    })
    
    labels = {}
    primary_label_df = None  # Will use MA-based exit labels for lagged features
    ma_label_name = None
    
    for label_name, label_config in label_configs.items():
        pt = label_config['pt']
        sl = label_config['sl']
        max_hold = label_config['max_hold']
        
        # Check if using MA exit strategy (default is False for backwards compatibility)
        use_ma_exit = label_config.get('use_ma_exit', False)
        
        if use_ma_exit:
            # Use MA crossover signals as exits
            label_df = generate_labels_ma_exit(df, signals, pt, sl, max_hold)
            logging.info(f"Generated labels for '{label_name}' using MA exit strategy")
            # Prefer MA-based exit labels for lagged features (matches trading strategy)
            if primary_label_df is None:
                primary_label_df = label_df
                ma_label_name = label_name
        else:
            # Use standard PT/SL/max_hold strategy - ALSO pass signals for direction
            label_df = generate_labels(df, signals, pt, sl, max_hold)
            logging.info(f"Generated labels for '{label_name}' using standard PT/SL strategy")
        
        labels[label_name] = {
            'pt': pt,
            'sl': sl,
            'max_hold': max_hold,
            'use_ma_exit': use_ma_exit,
            'labels': label_df
        }
    
    # Fallback: use first label config if no MA-based config exists
    if primary_label_df is None and len(labels) > 0:
        first_label_name = list(labels.keys())[0]
        primary_label_df = labels[first_label_name]['labels']
        logging.info(f"Using '{first_label_name}' (first config) for lagged profit features")
    elif ma_label_name:
        logging.info(f"Using '{ma_label_name}' (MA-based exits) for lagged profit features")
    
    # Add lagged profit features for ALL label strategies
    # This allows models to learn from multiple exit strategy perspectives
    all_lagged_features = []
    
    for label_name, label_info in labels.items():
        label_df = label_info['labels']
        
        if 'profit' in label_df.columns:
            try:
                # Create feature prefix from label name
                # Remove underscores and use full name
                # e.g., 'ma_crossover' -> 'macrossover', 'conservative' -> 'conservative'
                prefix = label_name.replace('_', '')
                
                features_df, added_features = add_lagged_profit_features(features_df, label_df, strategy_name=prefix)
                all_lagged_features.extend(added_features)
                logging.info(f"Added {len(added_features)} lagged features for '{label_name}' with prefix '{prefix}'")
                
            except Exception as e:
                logging.warning(f"Could not add lagged profit features for '{label_name}': {e}")
    
    # Re-extract features with all lagged profit features included
    if all_lagged_features:
        logging.info(f"Total lagged features added: {len(all_lagged_features)} from {len(labels)} strategies")
        all_features = available_features + all_lagged_features
        features = features_df[all_features].dropna()
    
    # Prepare tick data (using original OHLC data)
    tick_data = df[['open', 'high', 'low', 'close']].copy()
    
    logging.info(f"Processed {date_str}: {len(signals)} signals, "
                f"{len(features)} features, {len(labels)} label configs")
    
    return signals, features, labels, tick_data

def run_date(date_obj, spot, config, features_dir="features"):
    """
    Process a single date and save outputs.
    
    Args:
        date_obj: datetime object for the date to process
        spot: Currency pair (e.g., 'EURUSD')
        config: Configuration dictionary
        features_dir: Directory to save feature files (default: 'features')
    """
    d = date_obj.strftime("%Y%m%d")
    
    # Process single date without offsets
    signals, features, labels, ticks = run_date_offset(date_obj, spot, config, 0)
    
    # Save outputs with currency pair in filename to the specified features directory
    ticks.to_csv(f"{features_dir}/tick_bars_{d}_{spot}.csv.bz2")
    features.to_csv(f"{features_dir}/features_{d}_{spot}.csv.bz2")
    
    return signals, features, labels 
   

def process_date(date, spot, settings, signals_dir, data_dir, features_dir):
    """
    Process a single date and save all outputs to organized folders.
    
    Args:
        date: datetime object for the date to process
        spot: Currency pair
        settings: Configuration dictionary
        signals_dir: Directory for signal files
        data_dir: Directory for processed data (features + labels)
        features_dir: Directory for intermediate feature files
    """
    logging.info(f"running date {date}")
    d = date.strftime("%Y%m%d")
    try:
        signals, features, y_s = run_date(date, spot, settings, features_dir)
        
        # Save signals to signals/ subfolder
        signals.to_frame().to_parquet(f"{signals_dir}/{d}_{spot}_signals.parquet")
        
        # Save features to data/ subfolder
        features.to_parquet(f"{data_dir}/{d}_{spot}_features.parquet")
        
        # Handle new dict format for labels - now returns DataFrame with label, profit, exit_bars
        for label_name, label_data in y_s.items():
            pt = label_data['pt']
            sl = label_data['sl']
            max_hold = label_data['max_hold']
            label_df = label_data['labels']  # Now a DataFrame with multiple columns
            
            # Save labels to data/ subfolder (both Parquet and CSV)
            base_filename = f"{d}_{spot}_{pt}_{sl}_{max_hold}_y"
            label_df.to_parquet(f"{data_dir}/{base_filename}.parquet")
            label_df.to_csv(f"{data_dir}/{base_filename}.csv")
            
            # Log statistics
            if 'profit' in label_df.columns:
                profitable = (label_df['profit'] > 0).sum()
                total = label_df['profit'].notna().sum()
                avg_profit = label_df['profit'].mean()
                logging.info(f"Saved labels for '{label_name}': {base_filename} (.parquet + .csv)")
                logging.info(f"  Stats: {profitable}/{total} profitable ({profitable/total*100:.1f}%), avg profit: {avg_profit:.6f}")
    except Exception as e:
        logging.exception(f"skipping {date} due to {e}")


def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

@click.command()
@click.option('--config',type=click.Path(exists=True), help='Path to the configuration YAML file.')
@click.option('--n_jobs',default=-1,help='Number of jobs to run in parallel')
@click.option('--start-date', type=click.DateTime(formats=["%m%d%Y"]), help='Start date in YYYY-MM-DD format')
@click.option('--end-date', type=click.DateTime(formats=["%m%d%Y"]), help='End date in YYYY-MM-DD format')
def main(config,n_jobs,start_date,end_date):

    # Load settings from YAML file
    settings = load_config(config)
    spot = settings.get('spot')
    
    # Use config filename (without extension) as the folder name
    config_path = Path(config)
    config_name = config_path.stem  # Gets filename without extension
    
    # Create organized folder structure: results/<config_name>/
    base_results_dir = Path("results") / config_name
    
    # Subdirectories for different types of outputs
    signals_dir = base_results_dir / "signals"      # Trading signals
    features_dir = base_results_dir / "features"    # Feature CSVs (intermediate)
    data_dir = base_results_dir / "data"            # Processed parquet files (features + labels)
    logs_dir = base_results_dir / "logs"            # Processing logs
    models_dir = base_results_dir / "models"        # Trained models (for future use)
    predictions_dir = base_results_dir / "predictions"  # Model predictions (for future use)
    
    # Create all directories
    signals_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    base_filename = 'feature'
    next_log_filename = find_next_log_filename(logs_dir, base_filename)
    setup_logging(next_log_filename)
   
    logging.info(f"Config: {config_name}")
    logging.info(f"Results directory: {base_results_dir}")
    logging.info(f"  - Signals: {signals_dir}")
    logging.info(f"  - Features: {features_dir}")
    logging.info(f"  - Data: {data_dir}")
    logging.info(f"  - Logs: {logs_dir}")
    logging.info(f"  - Models: {models_dir}")
    logging.info(f"  - Predictions: {predictions_dir}")
    
    validate_labels(settings) 
    print(start_date, type(start_date))
    start_date = start_date or datetime.strptime(settings.get('start_date'), "%m%d%Y")
    end_date = end_date or datetime.strptime(settings.get('end_date'), "%m%d%Y")
    

    logging.info(f"start {start_date} end {end_date} for pair {spot}")
    
    weekdays = []
    while start_date <= end_date:
        if start_date.weekday() < 5:  # 0-4 denotes Monday to Friday
            weekdays.append(start_date)
        start_date += timedelta(days=1)

    Parallel(n_jobs=n_jobs)(delayed(process_date)(date, spot, settings, signals_dir, data_dir, features_dir) for date in weekdays)
  

if __name__ == '__main__':
    main()
