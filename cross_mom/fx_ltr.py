"""
Learn-to-Rank (LTR) cross-sectional momentum strategy using shared base.

Uses LightGBM to learn optimal pair rankings from features.
All backtesting logic is shared with other strategies via fx_backtest_base.

Usage:
    python cross_mom/fx_ltr.py --start-date 01022025 --end-date 01102025 \
        --rebalance-freq 60 --top-n 2 --training-days 30 --retrain-frequency 7
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import lightgbm as lgb
from typing import Dict, Optional
from datetime import datetime, timedelta
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fx_backtest_base import FXBacktester, calculate_statistics, print_statistics
from fx_ltr_features import calculate_features  # NEW: Use modular features
from fx_ltr_cs_features import apply_cross_sectional_blocks  # NEW: Cross-sectional features


# =============================================================================
# G7 PAIRS AND VOLATILITY FILTERING
# =============================================================================

# Only trade G7 pairs (exclude EM pairs with wide spreads)
G7_PAIRS = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY']

# Volatility thresholds - only trade when prev bar volatility > threshold
VOL_THRESHOLDS_BPS = {
    'EURUSD': 3.0,
    'USDJPY': 3.5,
    'AUDUSD': 3.5,
    'GBPUSD': 3.0,
    'USDCAD': 3.0,
    'USDCHF': 2.5,
}


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# Features are now imported from fx_ltr_features module with additional filtering features
# This gives us 37 base features across 7 modular blocks PLUS filtering features:
# - block_spread_micro (2 features): spread_bps, close_minus_mid_bps
# - block_intraday_mom (9 features): ret_5m/15m/30m/60m, zret_*, slope_10
# - block_breakout_rolling (3 features): donch_30m/60m, clv_1
# - block_garman_klass_vol (2 features): gk_vol_60m, rv_60m
# - block_rebalance_clock (5 features): bar_in_hour, mins_to_rebalance, hour_sin/cos, dow
# - block_pair_characteristics (3 features): is_usd_inverse, is_major, is_em
# - block_legacy_features (13 features): returns_*, volatility_*, zscore_* (backwards compatible)
# 
# FILTERING FEATURES (added to modular features):
# - prev_bar_volatility: Previous bar's HL range in bps (for filtering)
# - vol_spread_ratio: Volatility / spread ratio
# - is_tradeable: Binary flag if vol > threshold


def add_filtering_features(features: Dict[str, float], df: pd.DataFrame, pair: str) -> Dict[str, float]:
    """
    Add volatility filtering features to the base modular features.
    
    Args:
        features: Base features from fx_ltr_features module
        df: Historical data DataFrame
        pair: Currency pair name
        
    Returns:
        Features dict with added filtering features
    """
    # Previous bar volatility (for filtering decision)
    if len(df) >= 2:
        prev_hl = df['high'].iloc[-2] - df['low'].iloc[-2]
        prev_close = df['close'].iloc[-2]
        features['prev_bar_volatility'] = (prev_hl / prev_close) * 10000 if prev_close > 0 else 0
    else:
        features['prev_bar_volatility'] = 0
    
    # Vol/Spread ratio (already have spread_bps from modular features)
    if features.get('spread_bps', 0) > 0:
        features['vol_spread_ratio'] = features['prev_bar_volatility'] / features['spread_bps']
    else:
        features['vol_spread_ratio'] = 0
    
    # Is tradeable flag (meets volatility threshold)
    vol_threshold = VOL_THRESHOLDS_BPS.get(pair, 2.5)
    features['is_tradeable'] = 1.0 if features['prev_bar_volatility'] > vol_threshold else 0.0
    
    return features


def calculate_features_with_filter(df: pd.DataFrame, pair: str) -> Dict[str, float]:
    """
    Wrapper that combines modular features with filtering features.
    """
    # Get base modular features
    features = calculate_features(df, pair)
    
    if not features:
        return features
    
    # Add filtering features
    features = add_filtering_features(features, df, pair)
    
    return features


def generate_training_data_multiday(data_dir: Path, pairs: list, start_date: str, end_date: str,
                                     rebalance_minutes: int = 60, prediction_horizon: int = 60) -> pd.DataFrame:
    """
    Generate training data from multiple days of historical data.
    Matches baseline fx_ltr_momentum_bidask.py exactly.
    
    Args:
        data_dir: Path to data directory (data/bidask/output)
        pairs: List of currency pairs
        start_date: Start date YYYYMMDD
        end_date: End date YYYYMMDD
        rebalance_minutes: Rebalance frequency
        prediction_horizon: Minutes ahead to predict
        
    Returns:
        DataFrame with features and labels for each pair at each rebalance
    """
    from fx_backtest_base import load_pair_data
    
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
            df = load_pair_data(date_str, pair, str(data_dir))
            if df is not None and len(df) > 0:
                pair_data[pair] = df
        
        if len(pair_data) < 2:
            continue
        
        # Align timestamps - match baseline exactly
        all_timestamps = set()
        for df in pair_data.values():
            all_timestamps.update(df['Datetime'].values)
        
        common_timestamps = pd.DataFrame({'Datetime': sorted(all_timestamps)})
        
        aligned_data = {}
        for pair, df in pair_data.items():
            merged = common_timestamps.merge(df, on='Datetime', how='left')
            # Forward fill ONLY price columns (match baseline behavior exactly)
            fill_cols = ['open', 'high', 'low', 'close']
            if 'bid' in merged.columns:
                fill_cols.append('bid')
            if 'ask' in merged.columns:
                fill_cols.append('ask')
            if 'offer' in merged.columns:
                fill_cols.append('offer')
            # Also need to fill USD-normalized columns
            for col in ['open_usd', 'high_usd', 'low_usd', 'close_usd', 'bid_usd', 'ask_usd']:
                if col in merged.columns:
                    fill_cols.append(col)
            merged[fill_cols] = merged[fill_cols].ffill()
            aligned_data[pair] = merged
        
        pair_data = aligned_data
        
        # Get common timestamps
        first_pair = list(pair_data.keys())[0]
        timestamps = pair_data[first_pair]['Datetime'].values
        num_bars = len(timestamps)
        
        # Generate training samples at each rebalance point
        for idx in range(120, num_bars - prediction_horizon, rebalance_minutes):
            timestamp = timestamps[idx]
            group_id = f"{date_str}_{timestamp}"
            
            # STEP 1: Build snapshot for all pairs at this timestamp
            snapshot_rows = []
            labels = {}
            
            for pair in pair_data.keys():
                df = pair_data[pair]
                
                # Extract per-pair features at current timestamp
                df_history = df.iloc[:idx + 1]
                features = calculate_features_with_filter(df_history, pair)
                
                if not features:
                    continue
                
                # CRITICAL: Skip if below volatility threshold (filtering)
                if features.get('is_tradeable', 0) == 0:
                    continue  # Skip low-volatility scenarios
                
                # Calculate label (next-hour PnL)
                current_price = df.iloc[idx]['close']
                future_price = df.iloc[idx + prediction_horizon]['close']
                
                # PnL calculation with USD direction correction
                if pair.startswith('USD'):  # USD inverse pairs
                    # For USD/JPY etc, we short when ranked high (strong JPY)
                    next_hour_pnl = -(future_price - current_price) / current_price
                else:  # Regular pairs
                    # For EUR/USD etc, we long when ranked high (strong EUR)
                    next_hour_pnl = (future_price - current_price) / current_price
                
                # Store features and label
                features_copy = features.copy()
                features_copy['pair'] = pair
                snapshot_rows.append(features_copy)
                labels[pair] = next_hour_pnl
            
            # Skip if not enough pairs
            if len(snapshot_rows) < 2:
                continue
            
            # STEP 2: Apply cross-sectional transformations to snapshot
            snapshot_df = pd.DataFrame(snapshot_rows)
            snapshot_df = apply_cross_sectional_blocks(snapshot_df, prev_ranks=None)
            
            # STEP 3: Create training rows with CS features
            for _, row in snapshot_df.iterrows():
                pair = row['pair']
                
                # Create training row with all features (per-pair + CS)
                training_row = {
                    'pair': pair,
                    'timestamp': timestamp,
                    'group_id': group_id,
                    'label': labels[pair],
                }
                
                # Add all features (per-pair + cross-sectional)
                training_row.update(row.drop('pair').to_dict())
                
                training_rows.append(training_row)
    
    df_train = pd.DataFrame(training_rows)
    
    if len(df_train) > 0:
        print(f"Generated {len(df_train)} training samples across {df_train['group_id'].nunique()} rebalance periods")
    else:
        print("WARNING: No training samples generated!")
    
    return df_train


def generate_training_data(all_data: Dict[str, pd.DataFrame], rebalance_indices: list,
                          top_n: int = 2) -> pd.DataFrame:
    """
    Generate training samples from historical data.
    
    For each rebalance, create samples with features + forward returns as target.
    This creates a learning-to-rank problem where we learn to predict which pairs
    will have the best forward returns.
    
    Args:
        all_data: Dictionary of pair -> DataFrame
        rebalance_indices: List of bar indices where rebalances occur
        top_n: Number of top/bottom pairs to select
        
    Returns:
        DataFrame with features and targets for training
    """
    samples = []
    
    for idx in rebalance_indices:
        if idx < 120:  # Need history for features
            continue
        
        # For each pair, calculate features and forward return
        for pair, df in all_data.items():
            if idx >= len(df) - 1:  # Need next bar for target
                continue
            
            # Calculate features at this point
            features = calculate_features_with_filter(df.iloc[:idx+1], pair)
            if not features:
                continue
            
            # CRITICAL: Skip if below volatility threshold
            if features.get('is_tradeable', 0) == 0:
                continue
            
            # Calculate forward return (target to rank by)
            # CRITICAL: For USD-inverse pairs (USDJPY, USDCAD, etc), we need to NEGATE
            # the return because when USD strengthens vs JPY, close_usd goes UP but we
            # want to SHORT that pair (not LONG it).
            current_price = df['close_usd'].iloc[idx]
            next_price = df['close_usd'].iloc[idx+1]
            raw_return = (next_price / current_price) - 1
            
            # Negate for USD-inverse pairs so model learns correct direction
            if pair.startswith('USD'):  # USDJPY, USDCAD, etc.
                forward_return = -raw_return
            else:  # EURUSD, GBPUSD, etc.
                forward_return = raw_return
            
            sample = features.copy()
            sample['pair'] = pair
            sample['timestamp'] = df['Datetime'].iloc[idx] if 'Datetime' in df.columns else idx
            sample['target'] = forward_return
            samples.append(sample)
    
    return pd.DataFrame(samples)


# =============================================================================
# LTR MODEL
# =============================================================================

class LTRModel:
    """LightGBM ranking model for pair selection."""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
    
    def train(self, train_df: pd.DataFrame):
        """
        Train ranking model using LightGBM regression to predict PnL.
        
        We use regression (not lambdarank) because:
        - Our targets are continuous PnL values (not integer relevance scores)
        - We want to predict which pairs will have best forward returns
        - Rankings are derived from the predicted values
        
        Args:
            train_df: DataFrame with features, pairs, timestamps, and targets
        """
        if len(train_df) == 0:
            raise ValueError("No training data provided")
        
        # Prepare features and target (column is named 'label' in training data)
        feature_cols = [c for c in train_df.columns if c not in ['pair', 'timestamp', 'group_id', 'label']]
        X = train_df[feature_cols].values
        y = train_df['label'].values
        
        # Create LightGBM dataset (no grouping needed for regression)
        train_data = lgb.Dataset(X, label=y, feature_name=feature_cols, free_raw_data=False)
        
        # Train LightGBM regressor
        print("Training LightGBM regression model...")
        print(f"Attempting GPU training (fallback to CPU with 32 threads)...")
        
        try:
            # Try GPU first
            params = {
                'objective': 'regression',
                'metric': 'l1',
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'verbosity': -1,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'num_threads': 32,
            }
            
            import time
            start_time = time.time()
            self.model = lgb.train(params, train_data, num_boost_round=100)
            elapsed = time.time() - start_time
            print(f"✓ Model trained using GPU in {elapsed:.2f}s")
            
        except Exception as e:
            # Fallback to CPU
            print(f"GPU not available ({e})")
            print(f"→ Using CPU with 32 threads...")
            
            params = {
                'objective': 'regression',
                'metric': 'l1',
                'verbosity': -1,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'num_threads': 32,
            }
            
            import time
            start_time = time.time()
            self.model = lgb.train(params, train_data, num_boost_round=100)
            elapsed = time.time() - start_time
            print(f"✓ Model trained using CPU in {elapsed:.2f}s")
        
        self.feature_columns = feature_cols
        
        # Show feature importance
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        print()
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores for pairs."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X = features_df[self.feature_columns].values
        return self.model.predict(X)
    
    def get_feature_importance(self) -> list:
        """Get feature importance as sorted list of (feature, importance) tuples."""
        if self.model is None:
            return []
        
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = [(self.feature_columns[i], importance[i]) 
                             for i in range(len(self.feature_columns))]
        return sorted(feature_importance, key=lambda x: x[1], reverse=True)
    
    def save(self, filepath: str):
        """Save model to disk."""
        if self.model is not None:
            self.model.save_model(filepath)
            print(f"Model saved to: {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk."""
        self.model = lgb.Booster(model_file=filepath)
        print(f"Model loaded from: {filepath}")


# =============================================================================
# LTR RANKER (Manages model training and prediction)
# =============================================================================

class LTRRanker:
    """
    Manages LTR model training and prediction.
    Handles periodic retraining on rolling windows of data.
    """
    
    def __init__(self, retrain_frequency: int = 7, training_days: int = 30):
        """
        Args:
            retrain_frequency: Retrain model every N days
            training_days: Use last N days for training (currently not fully implemented)
        """
        self.retrain_frequency = retrain_frequency
        self.training_days = training_days
        self.model = LTRModel()
        self.last_train_date = None
        self.model_dir = Path("working_files/ltr_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.training_data_accumulated = []  # Store training samples
    
    def should_retrain(self, current_date) -> bool:
        """Check if model should be retrained on Saturdays."""
        # Convert to datetime if string
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
            
        if self.last_train_date is None:
            return True
        
        # Retrain on Saturdays only (weekday() returns 5 for Saturday)
        if current_date.weekday() == 5:  # Saturday
            days_since_train = (current_date - self.last_train_date).days
            return days_since_train >= self.retrain_frequency
        
        return False
    
    def retrain(self, data_dir: Path, pairs: list, current_date, 
                rebalance_freq: int = 60, prediction_horizon: int = 60):
        """
        Retrain model on multi-day historical data.
        Uses past training_days of data to train model, matching baseline behavior.
        
        Args:
            data_dir: Path to data directory
            pairs: List of currency pairs
            current_date: Current date (datetime or string)
            rebalance_freq: Rebalance frequency in minutes
            prediction_horizon: Minutes ahead to predict (for training labels)
        """
        # Convert to datetime if string
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        print("=" * 80)
        print(f"RETRAINING MODEL on {current_date.strftime('%Y%m%d')} ({current_date.strftime('%A')})")
        print("=" * 80)
        
        # Calculate training window (past N days before current date)
        end_date = current_date - timedelta(days=1)  # Don't include current date in training
        start_date = end_date - timedelta(days=self.training_days)
        
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # Generate training data from historical dates
        train_df = generate_training_data_multiday(
            data_dir, pairs, start_str, end_str, 
            rebalance_minutes=rebalance_freq,
            prediction_horizon=prediction_horizon
        )
        
        if len(train_df) < 100:
            print(f"Insufficient training data ({len(train_df)} < 100 samples), skipping retrain")
            return False
        
        print(f"Training with {len(train_df)} samples...")
        print()
        
        self.model.train(train_df)
        self.last_train_date = current_date
        
        # Print feature importance
        print("\nTop 20 Feature Importances:")
        print("-" * 60)
        feature_importance = self.model.get_feature_importance()
        for i, (feat, imp) in enumerate(feature_importance[:20], 1):
            print(f"{i:2}. {feat:30} {imp:>10.1f}")
        print()
        
        # Save model
        model_file = self.model_dir / f"ltr_model_{current_date.strftime('%Y%m%d')}.txt"
        self.model.save(str(model_file))
        
        return True
    
    def rank_pairs(self, all_data: Dict[str, pd.DataFrame], idx: int) -> Dict[str, float]:
        """
        Rank pairs using LTR model with cross-sectional features.
        
        Args:
            all_data: Dictionary of pair -> DataFrame
            idx: Current bar index
            
        Returns:
            Dictionary of pair -> ranking score (higher is better)
        """
        if self.model.model is None:
            # No model trained yet - use random ranking as fallback
            print("WARNING: No trained model - using random rankings")
            return {pair: np.random.random() for pair in all_data.keys()}
        
        # STEP 1: Calculate per-pair features for all pairs
        features_list = []
        pairs_list = []
        
        for pair, df in all_data.items():
            features = calculate_features_with_filter(df.iloc[:idx+1], pair)
            if features:
                # CRITICAL: Only include pairs that pass volatility filter
                if features.get('is_tradeable', 0) == 1:
                    features['pair'] = pair
                    features_list.append(features)
                    pairs_list.append(pair)
        
        if not features_list:
            return {pair: 0.0 for pair in all_data.keys()}
        
        # STEP 2: Build snapshot and apply cross-sectional features
        snapshot_df = pd.DataFrame(features_list)
        
        # TODO: Pass prev_ranks for turnover smoothing (store from previous rebalance)
        snapshot_df = apply_cross_sectional_blocks(snapshot_df, prev_ranks=None)
        
        # STEP 3: Predict scores using model
        scores = self.model.predict(snapshot_df)
        
        # Create ranking dictionary
        rankings = {pair: float(score) for pair, score in zip(pairs_list, scores)}
        
        # For missing pairs, assign lowest score
        for pair in all_data.keys():
            if pair not in rankings:
                rankings[pair] = -999.0
        
        return rankings


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FX LTR Momentum Strategy')
    parser.add_argument('--start-date', type=str, required=True, help='Start date MMDDYYYY')
    parser.add_argument('--end-date', type=str, required=True, help='End date MMDDYYYY')
    parser.add_argument('--rebalance-freq', type=int, default=60, help='Rebalance frequency in minutes')
    parser.add_argument('--top-n', type=int, default=2, help='Number of pairs to go long/short')
    parser.add_argument('--training-days', type=int, default=30, help='Days of history for training')
    parser.add_argument('--retrain-frequency', type=int, default=7, help='Retrain model every N days')
    parser.add_argument('--start-hour', type=int, default=0, help='Start hour for trading (EST)')
    parser.add_argument('--end-hour', type=int, default=23, help='End hour for trading (EST)')
    
    args = parser.parse_args()
    
    # Parse dates from MMDDYYYY to YYYYMMDD
    start_dt = pd.to_datetime(args.start_date, format='%m%d%Y')
    end_dt = pd.to_datetime(args.end_date, format='%m%d%Y')
    start_date = start_dt.strftime('%Y%m%d')
    end_date = end_dt.strftime('%Y%m%d')
    
    # Configuration
    DATA_DIR = Path("data/bidask/output")
    
    # Currency pairs - G7 ONLY (filtered for tight spreads)
    pairs = G7_PAIRS
    
    print(f"Found {len(pairs)} G7 currency pairs: {pairs}\n")
    
    print("Running backtest from {} to {}".format(start_date, end_date))
    print(f"Rebalance: every {args.rebalance_freq} min, Top N: {args.top_n}")
    print(f"USD Notional per position: $1,000,000")
    print(f"Trading Hours: {args.start_hour:02d}:00 - {args.end_hour:02d}:00 EST")
    print(f"Model Retraining: Every {args.retrain_frequency} days")
    print(f"NEW Features: ~80 features (37 per-pair + 40 cross-sectional + 3 filtering)")
    print(f"Volatility Filtering: Enabled (G7 pairs only)")
    print("=" * 80)
    print()
    
    # Create LTR ranker
    ltr_ranker = LTRRanker(
        retrain_frequency=args.retrain_frequency,
        training_days=args.training_days
    )
    
    # Create backtester
    backtester = FXBacktester(
        pairs=pairs,
        start_date=start_date,
        end_date=end_date,
        rebalance_freq=args.rebalance_freq,
        top_n=args.top_n,
        usd_notional=1_000_000,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        data_dir=str(DATA_DIR),
        lookback_bars=120,  # Use 120-bar lookback for LTR features
    )
    
    # Create ranking function that uses LTR
    def ranking_func(all_data, idx):
        """Ranking function that uses LTR model."""
        # Get current date for potential retraining
        current_date = list(all_data.values())[0]['Datetime'].iloc[idx]
        
        # Check if we need to retrain
        if ltr_ranker.should_retrain(current_date):
            # Retrain on multi-day historical data
            ltr_ranker.retrain(
                data_dir=DATA_DIR,
                pairs=pairs,
                current_date=current_date,
                rebalance_freq=args.rebalance_freq,
                prediction_horizon=args.rebalance_freq  # Predict one rebalance period ahead
            )
        
        # Rank pairs using current model
        return ltr_ranker.rank_pairs(all_data, idx)
    
    # Run backtest
    results_df, trades_df = backtester.run(ranking_func)
    
    # Calculate and print statistics
    stats = calculate_statistics(results_df, trades_df)
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print_statistics(stats)
    
    # Save results
    output_dir = Path("working_files/fx_momentum_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    hours_suffix = f"_h{args.start_hour}-{args.end_hour}" if args.start_hour is not None else ""
    base_filename = f"ltr_{start_date}_{end_date}_rebal{args.rebalance_freq}_top{args.top_n}_train{args.training_days}_retrain{args.retrain_frequency}{hours_suffix}"
    
    results_file = output_dir / f"results_{base_filename}.csv"
    trades_file = output_dir / f"trades_{base_filename}.csv"
    
    results_df.to_csv(results_file, index=False)
    trades_df.to_csv(trades_file, index=False)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Trades saved to: {trades_file}")
    
    # Show sample trades
    if len(trades_df) > 0:
        print(f"\nSample trades (first 10):")
        print(trades_df[['timestamp', 'date', 'pair', 'execution_side', 'entry_price', 
                         'exit_price', 'pnl_usd', 'pnl_pct', 'rank_score']].head(10).to_string(index=False))
