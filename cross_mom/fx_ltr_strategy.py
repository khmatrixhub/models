"""
Learn-to-Rank (LTR) cross-sectional momentum strategy.

Uses LightGBM to learn optimal pair rankings from features.
All backtesting logic is shared with other strategies via fx_backtest_base.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import lightgbm as lgb
from typing import Dict, Optional
from datetime import datetime, timedelta
from fx_backtest_base import FXBacktester, calculate_statistics, print_statistics


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def calculate_features(df: pd.DataFrame, pair: str) -> Dict[str, float]:
    """Calculate features for a single pair at current point in time."""
    features = {}
    
    # Use USD-normalized prices for features
    prices = df['close_usd'].values
    
    if len(prices) < 120:
        return features
    
    # Returns over various windows
    features['returns_30'] = (prices[-1] / prices[-30] - 1) if len(prices) >= 30 else 0
    features['returns_60'] = (prices[-1] / prices[-60] - 1) if len(prices) >= 60 else 0
    features['returns_120'] = (prices[-1] / prices[-120] - 1) if len(prices) >= 120 else 0
    
    # Volatility (std of returns)
    if len(prices) >= 30:
        returns_30 = pd.Series(prices[-30:]).pct_change().dropna()
        features['volatility_30'] = returns_30.std()
    else:
        features['volatility_30'] = 0
    
    if len(prices) >= 60:
        returns_60 = pd.Series(prices[-60:]).pct_change().dropna()
        features['volatility_60'] = returns_60.std()
    else:
        features['volatility_60'] = 0
    
    # Z-scores
    if len(prices) >= 30:
        mean_30 = prices[-30:].mean()
        std_30 = prices[-30:].std()
        features['zscore_30'] = (prices[-1] - mean_30) / std_30 if std_30 > 0 else 0
    else:
        features['zscore_30'] = 0
    
    if len(prices) >= 60:
        mean_60 = prices[-60:].mean()
        std_60 = prices[-60:].std()
        features['zscore_60'] = (prices[-1] - mean_60) / std_60 if std_60 > 0 else 0
    else:
        features['zscore_60'] = 0
    
    if len(prices) >= 120:
        mean_120 = prices[-120:].mean()
        std_120 = prices[-120:].std()
        features['zscore_120'] = (prices[-1] - mean_120) / std_120 if std_120 > 0 else 0
    else:
        features['zscore_120'] = 0
    
    # Pair characteristics
    features['is_em'] = 1 if pair in ['USDMXN', 'USDZAR', 'USDTRY'] else 0
    
    # Time features (if datetime available)
    if 'datetime' in df.columns and len(df) > 0:
        dt = df['datetime'].iloc[-1]
        features['hour_of_day'] = dt.hour
        features['day_of_week'] = dt.dayofweek
        features['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
    else:
        features['hour_of_day'] = 0
        features['day_of_week'] = 0
        features['hour_sin'] = 0
        features['hour_cos'] = 0
    
    return features


def generate_training_data(all_data: Dict[str, pd.DataFrame], rebalance_indices: list,
                          top_n: int = 2) -> pd.DataFrame:
    """
    Generate training samples from historical data.
    
    For each rebalance, create samples with features + forward returns as target.
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
            features = calculate_features(df.iloc[:idx+1], pair)
            if not features:
                continue
            
            # Calculate forward return (target)
            current_price = df['close_usd'].iloc[idx]
            next_price = df['close_usd'].iloc[idx+1]
            forward_return = (next_price / current_price) - 1
            
            sample = features.copy()
            sample['pair'] = pair
            sample['timestamp'] = df['datetime'].iloc[idx] if 'datetime' in df.columns else idx
            sample['target'] = forward_return
            samples.append(sample)
    
    return pd.DataFrame(samples)


# =============================================================================
# LTR MODEL TRAINING
# =============================================================================

class LTRModel:
    """LightGBM ranking model for pair selection."""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
    
    def train(self, train_df: pd.DataFrame):
        """Train ranking model."""
        if len(train_df) == 0:
            raise ValueError("No training data provided")
        
        # Prepare features and target
        feature_cols = [c for c in train_df.columns if c not in ['pair', 'timestamp', 'target']]
        X = train_df[feature_cols].values
        y = train_df['target'].values
        
        # Create group sizes for ranking (one group per timestamp)
        if 'timestamp' in train_df.columns:
            groups = train_df.groupby('timestamp').size().values
        else:
            groups = [len(train_df)]
        
        # Train LightGBM ranker
        print("Training LightGBM ranking model...")
        print(f"Attempting GPU training (fallback to CPU with 32 threads)...")
        
        try:
            # Try GPU first
            train_data = lgb.Dataset(X, label=y, group=groups)
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'verbosity': -1,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'num_threads': 32,
            }
            
            import time
            start_time = time.time()
            self.model = lgb.train(params, train_data, num_boost_round=100, verbose_eval=False)
            elapsed = time.time() - start_time
            print(f"✓ Model trained using GPU in {elapsed:.2f}s")
            
        except Exception as e:
            # Fallback to CPU
            print(f"GPU not available ({e})")
            print(f"→ Using CPU with 32 threads...")
            
            train_data = lgb.Dataset(X, label=y, group=groups)
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'verbosity': -1,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'num_threads': 32,
            }
            
            import time
            start_time = time.time()
            self.model = lgb.train(params, train_data, num_boost_round=100, verbose_eval=False)
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
        """Predict scores for pairs."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X = features_df[self.feature_columns].values
        return self.model.predict(X)
    
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
# LTR RANKING FUNCTION
# =============================================================================

class LTRRanker:
    """Manages LTR model training and prediction."""
    
    def __init__(self, retrain_frequency: int = 7, training_days: int = 30):
        """
        Args:
            retrain_frequency: Retrain model every N days
            training_days: Use last N days for training
        """
        self.retrain_frequency = retrain_frequency
        self.training_days = training_days
        self.model = LTRModel()
        self.last_train_date = None
        self.model_dir = Path("working_files/ltr_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def should_retrain(self, current_date: datetime) -> bool:
        """Check if model should be retrained."""
        if self.last_train_date is None:
            return True
        
        days_since_train = (current_date - self.last_train_date).days
        return days_since_train >= self.retrain_frequency
    
    def retrain(self, all_data: Dict[str, pd.DataFrame], current_date: datetime):
        """Retrain model on recent data."""
        print("=" * 80)
        print(f"RETRAINING MODEL on {current_date.strftime('%Y%m%d')} ({current_date.strftime('%A')})")
        print("=" * 80)
        
        # Find all rebalance points in recent history
        sample_df = list(all_data.values())[0]
        total_bars = len(sample_df)
        rebalance_indices = list(range(0, total_bars, 60))  # Assuming 60min rebalance
        
        print(f"Generating training data from {total_bars} bars...")
        train_df = generate_training_data(all_data, rebalance_indices)
        
        if len(train_df) < 100:
            print(f"WARNING: Only {len(train_df)} training samples - skipping retrain")
            return False
        
        print(f"Generated {len(train_df)} training samples across {len(train_df['timestamp'].unique())} rebalance periods")
        print(f"Training with {len(train_df)} samples...")
        print()
        
        self.model.train(train_df)
        self.last_train_date = current_date
        
        # Save model
        model_file = self.model_dir / f"ltr_model_{current_date.strftime('%Y%m%d')}.txt"
        self.model.save(str(model_file))
        
        return True
    
    def rank_pairs(self, all_data: Dict[str, pd.DataFrame], idx: int) -> Dict[str, float]:
        """Rank pairs using LTR model."""
        if self.model.model is None:
            # No model trained yet - use random ranking
            return {pair: np.random.random() for pair in all_data.keys()}
        
        # Calculate features for all pairs
        features_list = []
        pairs_list = []
        
        for pair, df in all_data.items():
            features = calculate_features(df.iloc[:idx+1], pair)
            if features:
                features_list.append(features)
                pairs_list.append(pair)
        
        if not features_list:
            return {pair: 0.0 for pair in all_data.keys()}
        
        # Predict scores
        features_df = pd.DataFrame(features_list)
        scores = self.model.predict(features_df)
        
        # Create ranking dictionary
        rankings = {pair: float(score) for pair, score in zip(pairs_list, scores)}
        
        # For missing pairs, assign lowest score
        for pair in all_data.keys():
            if pair not in rankings:
                rankings[pair] = -999.0
        
        return rankings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FX LTR Momentum Strategy')
    parser.add_argument('--start-date', type=str, required=True, help='Start date MMDDYYYY')
    parser.add_argument('--end-date', type=str, required=True, help='End date MMDDYYYY')
    parser.add_argument('--rebalance-freq', type=int, default=60, help='Rebalance frequency in minutes')
    parser.add_argument('--top-n', type=int, default=2, help='Number of pairs to go long/short')
    parser.add_argument('--training-days', type=int, default=30, help='Days of history for training')
    parser.add_argument('--retrain-frequency', type=int, default=7, help='Retrain model every N days')
    parser.add_argument('--start-hour', type=int, default=None, help='Start hour for trading (EST)')
    parser.add_argument('--end-hour', type=int, default=None, help='End hour for trading (EST)')
    parser.add_argument('--data-dir', type=str, default='data/bidask/output', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='working_files/ltr_results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Parse dates from MMDDYYYY to YYYYMMDD
    start_dt = pd.to_datetime(args.start_date, format='%m%d%Y')
    end_dt = pd.to_datetime(args.end_date, format='%m%d%Y')
    start_date = start_dt.strftime('%Y%m%d')
    end_date = end_dt.strftime('%Y%m%d')
    
    # Currency pairs
    pairs = ['AUDUSD', 'EURUSD', 'GBPUSD', 'USDCAD', 'USDCHF', 'USDJPY', 
             'USDMXN', 'USDNOK', 'USDSEK', 'USDZAR']
    
    print(f"Found {len(pairs)} currency pairs: {pairs}")
    
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
        data_dir=args.data_dir,
        lookback_bars=120,  # Use 120-bar lookback for LTR features
    )
    
    # Create ranking function that uses LTR
    def ranking_func(all_data, idx):
        # Check if we need to retrain
        current_date = list(all_data.values())[0]['datetime'].iloc[idx]
        
        if ltr_ranker.should_retrain(current_date):
            # Load training data from recent history
            # (In real implementation, we'd load historical dates)
            # For now, use current data as proxy
            ltr_ranker.retrain(all_data, current_date)
        
        return ltr_ranker.rank_pairs(all_data, idx)
    
    # Run backtest
    results_df, trades_df = backtester.run(ranking_func)
    
    # Calculate and print statistics
    stats = calculate_statistics(results_df, trades_df)
    print_statistics(stats)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    hours_suffix = f"_h{args.start_hour}-{args.end_hour}" if args.start_hour is not None else ""
    base_filename = f"ltr_{start_date}_{end_date}_rebal{args.rebalance_freq}_top{args.top_n}_train{args.training_days}{hours_suffix}"
    
    results_file = output_dir / f"results_{base_filename}.csv"
    trades_file = output_dir / f"trades_{base_filename}.csv"
    
    results_df.to_csv(results_file, index=False)
    trades_df.to_csv(trades_file, index=False)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Trades saved to: {trades_file}")
    
    # Show sample trades
    if len(trades_df) > 0:
        print(f"\nSample trades (first 10):")
        print(trades_df.head(10).to_string())
