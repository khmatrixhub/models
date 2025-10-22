
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
import zoneinfo  # For timezone handling (Python 3.9+)


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
            # pt, sl, max_hold are optional - can be null or omitted
            # They act as backup exits when BB/MA exits are enabled
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
    
    # Set default for lagged profit features (default: enabled)
    if 'use_lagged_profit_features' not in settings:
        settings['use_lagged_profit_features'] = True
    
    # Validate time filter if provided
    if 'time_filter' in settings:
        time_filter = settings['time_filter']
        if not isinstance(time_filter, dict):
            raise ValueError("time_filter should be a dict with 'start_hour_est' and 'end_hour_est' keys")
        if 'start_hour_est' not in time_filter or 'end_hour_est' not in time_filter:
            raise ValueError("time_filter must have both 'start_hour_est' and 'end_hour_est' keys")
        
        start_hour = time_filter['start_hour_est']
        end_hour = time_filter['end_hour_est']
        
        if not (0 <= start_hour < 24) or not (0 <= end_hour <= 24):
            raise ValueError(f"time_filter hours must be between 0 and 24, got start={start_hour}, end={end_hour}")
        
        logging.info(f"Time filter configured: {start_hour}:00-{end_hour}:00 EST (entry signals only)")
    
    logging.info(f"Validated config for {len(settings.get('labels', {}))} label configurations: {list(settings['labels'].keys())}")
    logging.info(f"Lagged profit features: {'enabled' if settings['use_lagged_profit_features'] else 'DISABLED'}")


def add_lagged_profit_features(features_df, labels_df, strategy_name='', lookback_days=5):
    """
    Add lagged profit features to help predict future performance.
    
    VALIDATION: Checks that exit_timestamp <= next signal timestamp to prevent data leakage.
    If validation fails, returns empty feature list and skip status.
    
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
        labels_df: DataFrame containing 'profit' column and 'exit_timestamp'
        strategy_name: Prefix for feature names (e.g., 'conservative', 'aggressive')
        lookback_days: Not used anymore (kept for compatibility)
    
    Returns:
        Tuple of (features_df, list of added feature names, validation_status dict)
    """
    features_df = features_df.copy()
    prefix = f'{strategy_name}_' if strategy_name else ''
    added_features = []
    
    # Validate that all exits happen before the next signal
    validation_status = {
        'valid': False,
        'total_signals': len(labels_df),
        'violations': 0,
        'max_overlap': None,
        'strategy_name': strategy_name
    }
    
    if 'exit_timestamp' in labels_df.columns:
        # Check each signal's exit against the next signal's entry
        signal_times = labels_df.index.to_series()
        exit_times = labels_df['exit_timestamp']
        
        violations = 0
        max_overlap = pd.Timedelta(0)
        
        for i in range(len(labels_df) - 1):
            current_exit = exit_times.iloc[i]
            next_signal = signal_times.iloc[i + 1]
            
            if pd.notna(current_exit) and current_exit > next_signal:
                violations += 1
                overlap = current_exit - next_signal
                if overlap > max_overlap:
                    max_overlap = overlap
        
        validation_status['violations'] = violations
        validation_status['max_overlap'] = str(max_overlap) if max_overlap > pd.Timedelta(0) else None
        validation_status['valid'] = (violations == 0)
        
        if violations > 0:
            logging.warning(f"Strategy '{strategy_name}': {violations}/{len(labels_df)-1} exits occur AFTER next signal (max overlap: {max_overlap}). Skipping lagged features.")
            return features_df, [], validation_status
        else:
            logging.info(f"Strategy '{strategy_name}': Validation passed - all exits occur before next signal")
    else:
        logging.warning(f"Strategy '{strategy_name}': No exit_timestamp column found, skipping validation")
        validation_status['valid'] = True  # Assume valid if no timestamp to check
    
    # Get profit series - simple shift is safe because trades close before next signal
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
    
    return features_df, added_features, validation_status


def calculate_kama(close, period=10, fast_ema=2, slow_ema=30):
    """
    Calculate Kaufman's Adaptive Moving Average
    
    KAMA adapts its smoothing constant based on market efficiency:
    - In strong trends: behaves like fast EMA
    - In choppy markets: behaves like slow EMA
    
    Parameters:
    - close: Price series
    - period: Lookback period for efficiency ratio (default: 10)
    - fast_ema: Fast EMA period for trending markets (default: 2)
    - slow_ema: Slow EMA period for ranging markets (default: 30)
    
    Returns:
    - kama: Adaptive moving average series
    """
    # Efficiency Ratio: measures trend strength
    # ER = |Price Change| / Sum(|Volatility|)
    change = abs(close - close.shift(period))
    volatility = (close.diff().abs()).rolling(window=period).sum()
    er = change / volatility
    
    # Smoothing Constant: blend between fast and slow alphas based on ER
    fast_alpha = 2 / (fast_ema + 1)
    slow_alpha = 2 / (slow_ema + 1)
    sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
    
    # Calculate KAMA recursively
    kama = pd.Series(index=close.index, dtype=float)
    kama.iloc[period] = close.iloc[period]  # Initialize with first valid price
    
    for i in range(period + 1, len(close)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
    
    return kama


def calculate_technical_indicators(df, fast_window=5, slow_window=20, use_kama=False, kama_params=None):
    """
    Calculate technical indicators for features
    
    Parameters:
    - df: DataFrame with OHLC data
    - fast_window: Span for fast EMA (default: 5)
    - slow_window: Span for slow EMA (default: 20)
    - use_kama: If True, calculate KAMA for slow MA (default: False)
    - kama_params: Dict with KAMA parameters: {'period', 'fast_ema', 'slow_ema'}
    """
    df = df.copy()
    
    # Exponential Moving Averages - use configurable windows
    df['ma_5'] = df['close'].ewm(span=fast_window, adjust=False).mean()
    
    # For slow MA, use KAMA if requested, otherwise use EMA
    if use_kama and kama_params is not None:
        df['ma_20'] = calculate_kama(
            df['close'],
            period=kama_params.get('period', 10),
            fast_ema=kama_params.get('fast_ema', 2),
            slow_ema=kama_params.get('slow_ema', 30)
        )
    else:
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


def generate_ema_kama_cross_signals(df, fast_window=5, kama_params=None):
    """
    Generate EMA/KAMA crossover signals
    
    Uses fast EMA for responsiveness and KAMA for adaptive slow MA.
    KAMA automatically adjusts to market conditions:
    - Fast in trends (behaves like fast EMA)
    - Slow in ranges (behaves like slow EMA)
    
    Parameters:
    - df: DataFrame with OHLC data
    - fast_window: Span for fast EMA (default: 5)
    - kama_params: Dict with KAMA parameters:
        - 'period': Lookback for efficiency ratio (default: 10)
        - 'fast_ema': Fast EMA period for trending (default: 2)
        - 'slow_ema': Slow EMA period for ranging (default: 30)
    
    Returns:
    - signals: Series with 1 (bullish), -1 (bearish), 0 (no signal)
    """
    if kama_params is None:
        kama_params = {'period': 10, 'fast_ema': 2, 'slow_ema': 30}
    
    signals = pd.Series(index=df.index, data=0, name='signal')
    
    # Calculate fast EMA
    ma_fast = df['close'].ewm(span=fast_window, adjust=False).mean()
    
    # Calculate KAMA for slow MA
    ma_slow = calculate_kama(
        df['close'],
        period=kama_params.get('period', 10),
        fast_ema=kama_params.get('fast_ema', 2),
        slow_ema=kama_params.get('slow_ema', 30)
    )
    
    # Generate signals
    # 1 for bullish crossover (fast EMA crosses above KAMA)
    # -1 for bearish crossover (fast EMA crosses below KAMA)
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


def generate_bb_mean_reversion_signals(df, bb_window=20, bb_std=2.0, rsi_threshold=None):
    """
    Generate Bollinger Band mean reversion signals
    
    Strategy:
    - Long signal when price touches/crosses below lower band (oversold)
    - Short signal when price touches/crosses above upper band (overbought)
    - Optional: Use RSI confirmation (RSI < 30 for long, RSI > 70 for short)
    
    Parameters:
    - df: DataFrame with OHLC data
    - bb_window: Window for Bollinger Bands middle line (SMA) (default: 20)
    - bb_std: Number of standard deviations for bands (default: 2.0)
    - rsi_threshold: If provided, use RSI confirmation (default: None)
                     Tuple (oversold, overbought) e.g., (30, 70)
    
    Returns:
    - signals: Series with 1 (long/buy), -1 (short/sell), 0 (no signal)
    """
    signals = pd.Series(index=df.index, data=0, name='signal')
    
    # Calculate Bollinger Bands
    bb_middle = df['close'].rolling(window=bb_window).mean()
    bb_std_dev = df['close'].rolling(window=bb_window).std()
    bb_upper = bb_middle + (bb_std * bb_std_dev)
    bb_lower = bb_middle - (bb_std * bb_std_dev)
    
    # Calculate RSI if threshold provided
    if rsi_threshold is not None:
        rsi_oversold, rsi_overbought = rsi_threshold
        
        # Calculate RSI (using existing column if available, otherwise calculate)
        if 'rsi' in df.columns:
            rsi = df['rsi']
        else:
            # Simple RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        # Long signal: price at/below lower band AND RSI oversold
        long_signal = (df['close'] <= bb_lower) & (rsi <= rsi_oversold)
        
        # Short signal: price at/above upper band AND RSI overbought
        short_signal = (df['close'] >= bb_upper) & (rsi >= rsi_overbought)
    else:
        # No RSI confirmation - just use BB touches
        # Long signal: price at or below lower band (oversold)
        long_signal = df['close'] <= bb_lower
        
        # Short signal: price at or above upper band (overbought)
        short_signal = df['close'] >= bb_upper
    
    # Track last signal direction to ensure proper alternation
    # Only fire a signal when:
    # 1. This is the first signal, OR
    # 2. Price has returned to neutral zone AND crossed into opposite band
    # This ensures: long → short → long (perfect alternation)
    
    last_signal_direction = 0  # Track last fired signal: 0=none, 1=long, -1=short
    
    for i in range(len(df)):
        # Long signal conditions:
        # - Price touches lower band AND
        # - Last signal was either nothing (0) or short (-1)
        if long_signal.iloc[i] and last_signal_direction != 1:
            signals.iloc[i] = 1
            last_signal_direction = 1
        
        # Short signal conditions:
        # - Price touches upper band AND
        # - Last signal was either nothing (0) or long (1)
        elif short_signal.iloc[i] and last_signal_direction != -1:
            signals.iloc[i] = -1
            last_signal_direction = -1
    
    return signals


def generate_labels(df, signals, pt_pct, sl_pct, max_hold_hours):
    """
    Generate forward-looking labels for ML training
    
    CRITICAL: Exits on opposing signal OR max_hold, whichever comes first.
    This prevents data leakage from using profits of unclosed trades.
    
    Exit conditions (in order of priority):
    1. Opposing signal appears (signal reverses)
    2. Profit target hit
    3. Stop loss hit
    4. Max hold time reached
    
    Parameters:
    - pt_pct: profit target as percentage (e.g., 0.001 = 0.1%)
    - sl_pct: stop loss as percentage 
    - max_hold_hours: maximum holding period in hours
    - signals: Series with signal direction (1=long, -1=short, 0=no signal)
    
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
    for i in range(len(df)):  # Need at least 1 future bar
        entry_price = df['close'].iloc[i]
        entry_timestamp = df.index[i]
        
        # Get signal direction (1=long, -1=short)
        signal_direction = signals.iloc[i] if i < len(signals) else 0
        if signal_direction == 0:
            continue  # Skip if no signal
        
        # Get MA values at entry (if available in df)
        ma_fast_val = df['ma_5'].iloc[i] if 'ma_5' in df.columns else np.nan
        ma_slow_val = df['ma_20'].iloc[i] if 'ma_20' in df.columns else np.nan
        
        # CRITICAL: Look forward until OPPOSING SIGNAL or max_hold, whichever comes first
        # Exit AT the opposing signal bar (realistic - we exit when new signal appears)
        max_future_bars = min(max_hold_hours, len(df) - i - 1)
        
        # Find next opposing signal (if any) within max_hold
        opposing_signal_bar = None
        for j in range(1, max_future_bars + 1):
            future_signal = signals.iloc[i + j] if (i + j) < len(signals) else 0
            # Check if signal opposes our position (opposite direction or any new signal)
            if future_signal != 0 and future_signal != signal_direction:
                opposing_signal_bar = j
                break
        
        # Limit look-ahead to opposing signal if found
        # We exit AT the signal bar (when we see the new signal, we close and reverse)
        if opposing_signal_bar is not None:
            max_future_bars = opposing_signal_bar
        
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
        
        # Determine max range for search (handle None max_hold_hours)
        max_range = len(df) - i if max_hold_hours is None else min(max_hold_hours + 1, len(df) - i)
        
        for j in range(1, max_range):
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
            
            # Check for profit target (if PT is set)
            if pt_pct is not None and profit_pct >= pt_pct:
                exit_idx = future_idx
                exit_timestamp = future_timestamp
                exit_price = future_price
                exit_reason = 'pt'
                exit_profit = (exit_price - entry_price) * position_direction  # Price units
                break
            
            # Check for stop loss (if SL is set)
            if sl_pct is not None and profit_pct <= -sl_pct:
                exit_idx = future_idx
                exit_timestamp = future_timestamp
                exit_price = future_price
                exit_reason = 'sl'
                exit_profit = (exit_price - entry_price) * position_direction  # Price units (negative)
                break
        
        # If no exit found, use max hold (or last bar if max_hold is None)
        if exit_idx is None:
            if max_hold_hours is None:
                future_idx = len(df) - 1  # Use last bar
            else:
                future_idx = min(i + max_hold_hours, len(df) - 1)
            future_price = df['close'].iloc[future_idx]
            future_timestamp = df.index[future_idx]
            exit_idx = future_idx
            exit_timestamp = future_timestamp
            exit_price = future_price
            exit_reason = 'max_hold' if max_hold_hours is not None else 'no_signal'
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


def generate_labels_bb_exit(df, signals, pt_pct, sl_pct, max_hold_hours, bb_exit_config=None, df_full=None):
    """
    Generate labels with Bollinger Band-specific exit strategies for mean reversion
    
    Strategy:
    - Enter on BB touch (oversold/overbought)
    - Exit on mean reversion back to middle or opposite band
    
    Parameters:
    - df: DataFrame filtered to signal bars only (for indexing)
    - df_full: Full DataFrame with all bars (for searching exit conditions)
    - signals: Signal series (filtered to signal bars)
    - bb_exit_config: Dict with 'strategy', 'bb_window', 'bb_std', 'exit_std_multiplier'
    
    Exit conditions (in priority order):
    1. BB Exit Strategy (configurable):
       - 'middle': Exit when price crosses back to middle line (BB MA)
       - 'opposite_band': Exit when price touches opposite band
       - 'partial': Exit at MA + exit_std_multiplier * std_dev
       - 'signal': Exit on opposite signal (same as generate_labels_ma_exit)
    2. Profit target (PT) hit
    3. Stop loss (SL) hit  
    4. Max hold period reached
    
    Parameters:
    - df: DataFrame with OHLC data and BB indicators
    - signals: Series with BB signals (1=long from lower band, -1=short from upper band)
    - pt_pct: profit target percentage (backup exit)
    - sl_pct: stop loss percentage
    - max_hold_hours: maximum hold period (backup exit)
    - bb_exit_config: Dict with BB exit parameters:
        {
            'strategy': 'middle' | 'opposite_band' | 'partial' | 'signal',
            'bb_window': 20,  # BB calculation window (must match signal generation)
            'bb_std': 2.0,    # BB std multiplier (must match signal generation)
            'exit_std_multiplier': 0.5  # For 'partial' strategy: exit at MA ± 0.5*std
        }
    
    Returns DataFrame with columns:
    - label: binary 1 (profitable) or -1 (loss)
    - profit: actual profit/loss in PRICE UNITS
    - exit_bars: number of bars until exit
    - exit_reason: 'bb_middle', 'bb_opposite', 'bb_partial', 'signal', 'pt', 'sl', 'max_hold'
    - entry_timestamp, exit_timestamp, entry_price, exit_price
    - signal_direction, ma_fast, ma_slow
    """
    # Use full DataFrame if provided, otherwise use filtered df
    # df_full contains ALL bars for searching exit conditions
    # df contains only signal bars for indexing labels
    df_search = df_full if df_full is not None else df
    
    # Parse BB exit config
    if bb_exit_config is None:
        bb_exit_config = {'strategy': 'middle', 'bb_window': 20, 'bb_std': 2.0}
    
    exit_strategy = bb_exit_config.get('strategy', 'middle')
    bb_window = bb_exit_config.get('bb_window', 20)
    bb_std = bb_exit_config.get('bb_std', 2.0)
    exit_std_multiplier = bb_exit_config.get('exit_std_multiplier', 0.5)
    
    # Calculate Bollinger Bands for exit detection on FULL DataFrame
    bb_middle = df_search['close'].rolling(window=bb_window).mean()
    bb_std_dev = df_search['close'].rolling(window=bb_window).std()
    bb_upper = bb_middle + (bb_std * bb_std_dev)
    bb_lower = bb_middle - (bb_std * bb_std_dev)
    
    # For 'partial' strategy, calculate intermediate exit levels
    if exit_strategy == 'partial':
        bb_upper_exit = bb_middle + (exit_std_multiplier * bb_std_dev)
        bb_lower_exit = bb_middle - (exit_std_multiplier * bb_std_dev)
    
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
    
    # Process all signal bars
    for i in range(len(df)):
        entry_timestamp = df.index[i]
        entry_price = df['close'].iloc[i]
        
        # Find position in full DataFrame for this signal
        try:
            entry_pos_full = df_search.index.get_loc(entry_timestamp)
        except KeyError:
            logging.warning(f"Signal timestamp {entry_timestamp} not found in full DataFrame")
            continue
        
        # Get signal direction
        signal_direction = signals.iloc[i] if i < len(signals) else 0
        if signal_direction == 0:
            continue
        
        # Get MA values at entry
        ma_fast_val = df['ma_5'].iloc[i] if 'ma_5' in df.columns else np.nan
        ma_slow_val = df['ma_20'].iloc[i] if 'ma_20' in df.columns else np.nan
        
        # Determine max future bars to check in FULL DataFrame
        # If max_hold_hours is None, check until end of day
        if max_hold_hours is None:
            max_future_bars = len(df_search) - entry_pos_full - 1
        else:
            max_future_bars = min(max_hold_hours, len(df_search) - entry_pos_full - 1)
        
        # For 'signal' strategy: look for opposite signal in future signal bars
        # Signals exist at specific timestamps, not necessarily contiguous bars
        opposing_signal_timestamp = None
        if exit_strategy == 'signal':
            # Find future signal bars after current entry
            future_signal_timestamps = signals.index[signals.index > entry_timestamp]
            for future_ts in future_signal_timestamps:
                future_signal_val = signals.loc[future_ts]
                if future_signal_val != 0 and future_signal_val != signal_direction:
                    opposing_signal_timestamp = future_ts
                    break
            
            # If opposite signal found, limit search to before that timestamp
            if opposing_signal_timestamp is not None:
                opposing_pos_full = df_search.index.get_loc(opposing_signal_timestamp)
                max_future_bars = opposing_pos_full - entry_pos_full
        
        if max_future_bars <= 0:
            # Edge case: last bar or no future bars to search
            labels.iloc[i, 0] = 0
            labels.iloc[i, 1] = 0.0
            labels.iloc[i, 2] = 0
            labels.iloc[i, 3] = 'edge_case'
            labels.iloc[i, 4] = entry_timestamp
            labels.iloc[i, 5] = entry_timestamp
            labels.iloc[i, 6] = entry_price
            labels.iloc[i, 7] = entry_price
            labels.iloc[i, 8] = signal_direction
            labels.iloc[i, 9] = ma_fast_val
            labels.iloc[i, 10] = ma_slow_val
            continue
        
        # Get future prices and BB values from FULL DataFrame
        future_prices = df_search['close'].iloc[entry_pos_full+1:entry_pos_full+1+max_future_bars]
        future_highs = df_search['high'].iloc[entry_pos_full+1:entry_pos_full+1+max_future_bars]
        future_lows = df_search['low'].iloc[entry_pos_full+1:entry_pos_full+1+max_future_bars]
        
        # Get BB values from PREVIOUS bar (entry_pos_full is current bar, so start from there for "previous")
        # This simulates live trading: current bar's high/low vs BB calculated from previous close
        future_bb_middle = bb_middle.iloc[entry_pos_full:entry_pos_full+max_future_bars]
        future_bb_upper = bb_upper.iloc[entry_pos_full:entry_pos_full+max_future_bars]
        future_bb_lower = bb_lower.iloc[entry_pos_full:entry_pos_full+max_future_bars]
        
        if exit_strategy == 'partial':
            future_bb_upper_exit = bb_upper_exit.iloc[entry_pos_full:entry_pos_full+max_future_bars]
            future_bb_lower_exit = bb_lower_exit.iloc[entry_pos_full:entry_pos_full+max_future_bars]
        
        # Calculate returns (adjusted for signal direction)
        price_change_pct = (future_prices - entry_price) / entry_price
        returns = price_change_pct * signal_direction
        
        # Check exit conditions
        exit_idx = None
        exit_reason = None
        exit_price = None
        
        # Check BB exits and opposing signal
        for j, future_idx in enumerate(future_prices.index):
            # Get current bar's high/low and previous bar's BB levels
            current_high = future_highs.iloc[j]
            current_low = future_lows.iloc[j]
            current_close = future_prices.iloc[j]
            prev_bb_middle = future_bb_middle.iloc[j]
            prev_bb_upper = future_bb_upper.iloc[j]
            prev_bb_lower = future_bb_lower.iloc[j]
            
            # Check BB exit conditions FIRST (using intrabar high/low vs previous BB)
            if exit_strategy == 'middle':
                # Exit when current bar's high/low touches previous bar's middle line
                if signal_direction == 1:  # Long position (entered at lower band)
                    if current_high >= prev_bb_middle:
                        exit_idx = future_idx
                        exit_price = prev_bb_middle  # Exit AT the BB level
                        exit_reason = 'bb_middle'
                        break
                else:  # Short position (entered at upper band)
                    if current_low <= prev_bb_middle:
                        exit_idx = future_idx
                        exit_price = prev_bb_middle  # Exit AT the BB level
                        exit_reason = 'bb_middle'
                        break
            
            elif exit_strategy == 'opposite_band':
                # Exit when current bar's high/low touches previous bar's opposite band
                if signal_direction == 1:  # Long position (entered at lower band)
                    if current_high >= prev_bb_upper:
                        exit_idx = future_idx
                        exit_price = prev_bb_upper  # Exit AT the BB level
                        exit_reason = 'bb_opposite'
                        break
                else:  # Short position (entered at upper band)
                    if current_low <= prev_bb_lower:
                        exit_idx = future_idx
                        exit_price = prev_bb_lower  # Exit AT the BB level
                        exit_reason = 'bb_opposite'
                        break
            
            elif exit_strategy == 'partial':
                # Exit when current bar's high/low touches previous bar's partial exit level
                prev_bb_upper_exit = future_bb_upper_exit.iloc[j]
                prev_bb_lower_exit = future_bb_lower_exit.iloc[j]
                
                if signal_direction == 1:  # Long
                    if current_high >= prev_bb_upper_exit:
                        exit_idx = future_idx
                        exit_price = prev_bb_upper_exit  # Exit AT the BB level
                        exit_reason = 'bb_partial'
                        break
                else:  # Short
                    if current_low <= prev_bb_lower_exit:
                        exit_idx = future_idx
                        exit_price = prev_bb_lower_exit  # Exit AT the BB level
                        exit_reason = 'bb_partial'
                        break
            
            elif exit_strategy == 'signal':
                # For 'signal' strategy, exit when we reach the opposing signal bar
                if opposing_signal_timestamp is not None and future_idx == opposing_signal_timestamp:
                    exit_idx = future_idx
                    exit_price = current_close  # Use close for signal exits
                    exit_reason = 'signal'
                    break
            
            # Check PT/SL (backup exits) - use close prices
            if exit_reason is None:
                if returns.iloc[j] >= pt_pct:
                    exit_idx = future_idx
                    exit_price = current_close
                    exit_reason = 'pt'
                    break
                elif returns.iloc[j] <= -sl_pct:
                    exit_idx = future_idx
                    exit_price = current_close
                    exit_reason = 'sl'
                    break
        
        # If no exit found, use max hold
        if exit_idx is None:
            exit_idx = future_prices.index[-1]
            exit_price = df_search['close'].loc[exit_idx]
            exit_reason = 'max_hold'
        
        # Calculate profit and label (exit_price already set during loop for BB exits)
        exit_profit = (exit_price - entry_price) * signal_direction
        
        # Calculate exit_bars as the actual number of bars in FULL DataFrame
        exit_bar_position = df_search.index.get_loc(exit_idx)
        exit_bars = exit_bar_position - entry_pos_full
        
        # Set label based on profit
        label_val = 1 if exit_profit > 0 else -1
        
        # Store results
        labels.iloc[i, 0] = label_val
        labels.iloc[i, 1] = exit_profit
        labels.iloc[i, 2] = exit_bars
        labels.iloc[i, 3] = exit_reason
        labels.iloc[i, 4] = entry_timestamp
        labels.iloc[i, 5] = exit_idx
        labels.iloc[i, 6] = entry_price
        labels.iloc[i, 7] = exit_price
        labels.iloc[i, 8] = signal_direction
        labels.iloc[i, 9] = ma_fast_val
        labels.iloc[i, 10] = ma_slow_val
    
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
    
    # Get signal strategy from config (default: 'ma_cross')
    signal_strategy = config.get('signal_strategy', 'ma_cross')
    
    # Generate signals based on strategy
    if signal_strategy == 'ma_cross':
        # Moving Average Crossover (trend following) - EMA/EMA
        ma_windows = config.get('ma_windows', {})
        fast_window = ma_windows.get('fast', 5)
        slow_window = ma_windows.get('slow', 20)
        logging.info(f"Using MA crossover strategy with windows: fast={fast_window}, slow={slow_window}")
        signals = generate_ma_cross_signals(df, fast_window=fast_window, slow_window=slow_window)
        
    elif signal_strategy == 'ema_kama_cross':
        # EMA/KAMA Crossover (adaptive trend following)
        ma_windows = config.get('ma_windows', {})
        fast_window = ma_windows.get('fast', 5)
        
        # KAMA parameters with slow_ prefix
        kama_params = {
            'period': config.get('slow_period', 10),
            'fast_ema': config.get('slow_fast_ema', 2),
            'slow_ema': config.get('slow_slow_ema', 30)
        }
        
        logging.info(f"Using EMA/KAMA crossover strategy: fast_ema={fast_window}, "
                    f"kama(period={kama_params['period']}, fast_ema={kama_params['fast_ema']}, "
                    f"slow_ema={kama_params['slow_ema']})")
        signals = generate_ema_kama_cross_signals(df, fast_window=fast_window, kama_params=kama_params)
        
    elif signal_strategy == 'bb_mean_reversion':
        # Bollinger Band Mean Reversion
        signal_params = config.get('signal_params', {})
        bb_window = signal_params.get('bb_window', 20)
        bb_std = signal_params.get('bb_std', 2.0)
        # Check for use_rsi_filter boolean or legacy rsi_threshold tuple
        use_rsi_filter = signal_params.get('use_rsi_filter', False)
        rsi_threshold = signal_params.get('rsi_threshold', None) if not use_rsi_filter else (30, 70)
        logging.info(f"Using Bollinger Band mean reversion strategy: window={bb_window}, std={bb_std}, rsi_filter={use_rsi_filter}, rsi_threshold={rsi_threshold}")
        signals = generate_bb_mean_reversion_signals(df, bb_window=bb_window, bb_std=bb_std, rsi_threshold=rsi_threshold)
        
    else:
        raise ValueError(f"Unknown signal strategy: {signal_strategy}. Supported: 'ma_cross', 'ema_kama_cross', 'bb_mean_reversion'")
    
    # Apply time-of-day filter if configured
    # Filter signals to only allow entries during specified time window
    # (positions can be held outside the window, but no new entries)
    time_filter = config.get('time_filter', None)
    if time_filter is not None:
        start_hour_est = time_filter.get('start_hour_est', 0)
        end_hour_est = time_filter.get('end_hour_est', 23)
        
        # Convert EST hours to GMT using proper timezone conversion
        # Create a sample datetime for the date being processed to get correct offset
        # (handles daylight saving time automatically)
        sample_datetime = df.index[0] if len(df) > 0 else datetime.now()
        
        # Create timezone objects
        est_tz = zoneinfo.ZoneInfo('America/New_York')
        gmt_tz = zoneinfo.ZoneInfo('UTC')
        
        # Convert start hour: create naive datetime with EST hour, localize to EST, convert to GMT
        start_est_dt = sample_datetime.replace(hour=start_hour_est, minute=0, second=0, microsecond=0, tzinfo=None)
        start_est_aware = start_est_dt.replace(tzinfo=est_tz)
        start_gmt_dt = start_est_aware.astimezone(gmt_tz)
        start_hour_gmt = start_gmt_dt.hour
        
        # Convert end hour: create naive datetime with EST hour, localize to EST, convert to GMT
        end_est_dt = sample_datetime.replace(hour=end_hour_est % 24, minute=0, second=0, microsecond=0, tzinfo=None)
        end_est_aware = end_est_dt.replace(tzinfo=est_tz)
        end_gmt_dt = end_est_aware.astimezone(gmt_tz)
        end_hour_gmt = end_gmt_dt.hour
        
        # Get hour from datetime index (signals index is datetime)
        signal_hours = signals.index.hour
        
        # Handle time ranges that cross midnight GMT
        if start_hour_gmt <= end_hour_gmt:
            # Normal range (e.g., 7:00 to 17:00 GMT)
            time_mask = (signal_hours >= start_hour_gmt) & (signal_hours < end_hour_gmt)
        else:
            # Range crosses midnight (e.g., 22:00 to 05:00 GMT)
            time_mask = (signal_hours >= start_hour_gmt) | (signal_hours < end_hour_gmt)
        
        # Filter signals to only trading hours
        signals_before = signals[signals != 0].count()
        signals = signals.copy()
        signals[~time_mask] = 0  # Zero out signals outside trading hours
        signals_after = signals[signals != 0].count()
        
        # Determine if we're in EST or EDT based on the offset
        tz_name = start_est_aware.tzname()  # Will be 'EST' or 'EDT'
        
        logging.info(f"Time filter applied: {start_hour_est}:00-{end_hour_est}:00 {tz_name} ({start_hour_gmt}:00-{end_hour_gmt}:00 UTC)")
        logging.info(f"Signals filtered: {signals_before} -> {signals_after} (removed {signals_before - signals_after})")
    
    # Get MA windows for feature calculation (use config or defaults)
    ma_windows = config.get('ma_windows', {})
    fast_window = ma_windows.get('fast', 5)
    slow_window = ma_windows.get('slow', 20)
    
    # Check if we should use KAMA for slow MA in features
    use_kama = (signal_strategy == 'ema_kama_cross')
    kama_params = None
    if use_kama:
        kama_params = {
            'period': config.get('slow_period', 10),
            'fast_ema': config.get('slow_fast_ema', 2),
            'slow_ema': config.get('slow_slow_ema', 30)
        }
    
    # Calculate technical indicators for features
    features_df = calculate_technical_indicators(
        df, 
        fast_window=fast_window, 
        slow_window=slow_window,
        use_kama=use_kama,
        kama_params=kama_params
    )
    
    # IMPORTANT: Save full unfiltered DataFrame for label generation
    # BB exit strategies need all bars to search for exit conditions
    df_full = df.copy()
    
    # Filter to only signal bars (MA crossovers)
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
        pt = label_config.get('pt', None)
        sl = label_config.get('sl', None)
        max_hold = label_config.get('max_hold', None)
        
        # Check exit strategy type
        use_ma_exit = label_config.get('use_ma_exit', False)
        use_bb_exit = label_config.get('use_bb_exit', False)
        
        if use_bb_exit:
            # Use Bollinger Band-based exits for mean reversion
            # Read BB params from signal_params to ensure consistency
            signal_params = config.get('signal_params', {})
            bb_exit_config = {
                'strategy': label_config.get('bb_exit_strategy', 'middle'),  # 'middle', 'opposite_band', 'partial', 'signal'
                'bb_window': signal_params.get('bb_window', 20),  # Read from signal config
                'bb_std': signal_params.get('bb_std', 2.0),  # Read from signal config
                'exit_std_multiplier': label_config.get('exit_std_multiplier', 0.5)  # For 'partial' strategy
            }
            # For BB exits, PT/SL/max_hold are optional (can be None to disable backup exits)
            pt_use = pt if pt is not None else float('inf')  # Effectively disabled
            sl_use = sl if sl is not None else float('inf')  # Effectively disabled
            max_hold_use = max_hold if max_hold is not None else 1000000  # Effectively disabled
            label_df = generate_labels_bb_exit(df, signals, pt_use, sl_use, max_hold_use, bb_exit_config, df_full=df_full)
            logging.info(f"Generated labels for '{label_name}' using BB exit strategy: {bb_exit_config['strategy']}")
            # Prefer BB-based exit labels for lagged features (matches trading strategy)
            if primary_label_df is None:
                primary_label_df = label_df
                ma_label_name = label_name
        elif use_ma_exit:
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
            'use_bb_exit': use_bb_exit,
            'labels': label_df
        }
    
    # Fallback: use first label config if no MA-based config exists
    if primary_label_df is None and len(labels) > 0:
        first_label_name = list(labels.keys())[0]
        primary_label_df = labels[first_label_name]['labels']
        logging.info(f"Using '{first_label_name}' (first config) for lagged profit features")
    elif ma_label_name:
        logging.info(f"Using '{ma_label_name}' (MA-based exits) for lagged profit features")
    
    # Add lagged profit features for ALL label strategies (if enabled)
    # This allows models to learn from multiple exit strategy perspectives
    all_lagged_features = []
    validation_summary = []
    
    use_lagged_features = config.get('use_lagged_profit_features', True)
    
    if use_lagged_features:
        logging.info("Adding lagged profit features for all strategies...")
        for label_name, label_info in labels.items():
            label_df = label_info['labels']
            
            if 'profit' in label_df.columns:
                try:
                    # Create feature prefix from label name
                    # Remove underscores and use full name
                    # e.g., 'ma_crossover' -> 'macrossover', 'conservative' -> 'conservative'
                    prefix = label_name.replace('_', '')
                    
                    features_df, added_features, validation_status = add_lagged_profit_features(features_df, label_df, strategy_name=prefix)
                    validation_summary.append(validation_status)
                    
                    if validation_status['valid']:
                        all_lagged_features.extend(added_features)
                        logging.info(f"Added {len(added_features)} lagged features for '{label_name}' with prefix '{prefix}'")
                    else:
                        logging.warning(f"Skipped lagged features for '{label_name}' due to validation failure: {validation_status['violations']} violations")
                    
                except Exception as e:
                    logging.warning(f"Could not add lagged profit features for '{label_name}': {e}")
                    validation_summary.append({
                    'valid': False,
                    'strategy_name': label_name.replace('_', ''),
                    'error': str(e)
                })
        
        # Log validation summary
        if validation_summary:
            valid_strategies = [v for v in validation_summary if v['valid']]
            invalid_strategies = [v for v in validation_summary if not v['valid']]
            
            logging.info(f"Lagged feature validation summary: {len(valid_strategies)} valid, {len(invalid_strategies)} invalid")
            
            if invalid_strategies:
                logging.warning("Strategies excluded from lagged features due to data leakage:")
                for vstatus in invalid_strategies:
                    if 'violations' in vstatus:
                        logging.warning(f"  - {vstatus['strategy_name']}: {vstatus['violations']} violations, max overlap: {vstatus.get('max_overlap', 'N/A')}")
                    else:
                        logging.warning(f"  - {vstatus['strategy_name']}: {vstatus.get('error', 'Unknown error')}")
    else:
        logging.info("Lagged profit features DISABLED by config - using only technical indicators and bar statistics")

    
    # Re-extract features with all lagged profit features included (if any were added)
    if all_lagged_features:
        logging.info(f"Total lagged features added: {len(all_lagged_features)} from {len(labels)} strategies")
        all_features = available_features + all_lagged_features
        features = features_df[all_features].dropna()
    else:
        logging.info(f"Using {len(available_features)} base features (no lagged features)")

    
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
            # For BB exits with no backup PT/SL, use the label_name instead of PT_SL_HOLD
            if pt is None and sl is None and max_hold is None:
                base_filename = f"{d}_{spot}_{label_name}_y"
            else:
                base_filename = f"{d}_{spot}_{pt}_{sl}_{max_hold}_y"
            label_df.to_parquet(f"{data_dir}/{base_filename}.parquet")
            label_df.to_csv(f"{data_dir}/{base_filename}.csv")
            
            # Log statistics
            if 'profit' in label_df.columns:
                profitable = (label_df['profit'] > 0).sum()
                total = label_df['profit'].notna().sum()
                avg_profit = label_df['profit'].mean()
                logging.info(f"Saved labels for '{label_name}': {base_filename} (.parquet + .csv)")
                if total > 0:
                    logging.info(f"  Stats: {profitable}/{total} profitable ({profitable/total*100:.1f}%), avg profit: {avg_profit:.6f}")
                else:
                    logging.info(f"  Stats: No signals/trades generated")
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
