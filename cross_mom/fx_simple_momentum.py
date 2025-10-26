"""
Simple cross-sectional momentum strategy.

Ranks pairs by simple returns over a lookback window.
All backtesting logic is shared with other strategies via fx_backtest_base.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict
from fx_backtest_base import FXBacktester, calculate_statistics, print_statistics


def calculate_momentum(df: pd.DataFrame, window: int) -> float:
    """Calculate simple momentum (returns) over window periods."""
    if len(df) < window:
        return np.nan
    return (df['close_usd'].iloc[-1] / df['close_usd'].iloc[-window]) - 1


def rank_by_momentum(all_data: Dict[str, pd.DataFrame], idx: int, 
                     momentum_window: int = 30) -> Dict[str, float]:
    """
    Rank all pairs by momentum.
    
    Args:
        all_data: Dictionary of {pair: dataframe}
        idx: Current bar index
        momentum_window: Lookback window for momentum calculation
    
    Returns:
        Dictionary of {pair: momentum_score}
    """
    rankings = {}
    
    for pair, df in all_data.items():
        if idx < momentum_window:
            rankings[pair] = 0.0
            continue
        
        # Calculate momentum using USD-normalized prices
        momentum = calculate_momentum(df.iloc[:idx+1], momentum_window)
        rankings[pair] = momentum if not np.isnan(momentum) else 0.0
    
    return rankings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FX Simple Momentum Strategy')
    parser.add_argument('--start-date', type=str, required=True, help='Start date MMDDYYYY')
    parser.add_argument('--end-date', type=str, required=True, help='End date MMDDYYYY')
    parser.add_argument('--rebalance-freq', type=int, default=60, help='Rebalance frequency in minutes')
    parser.add_argument('--top-n', type=int, default=2, help='Number of pairs to go long/short')
    parser.add_argument('--momentum-window', type=int, default=30, help='Momentum lookback window (bars)')
    parser.add_argument('--start-hour', type=int, default=3, help='Start hour for trading (EST), default 3am')
    parser.add_argument('--end-hour', type=int, default=14, help='End hour for trading (EST), default 2pm')
    parser.add_argument('--data-dir', type=str, default='data/bidask/output', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='working_files/fx_momentum_results', 
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
        lookback_bars=args.momentum_window,  # Use momentum window as lookback
    )
    
    # Create ranking function with momentum window
    def ranking_func(all_data, idx):
        return rank_by_momentum(all_data, idx, args.momentum_window)
    
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
    base_filename = f"simple_{start_date}_{end_date}_rebal{args.rebalance_freq}_top{args.top_n}_mom{args.momentum_window}{hours_suffix}"
    
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
