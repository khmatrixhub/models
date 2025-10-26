"""
Shared backtesting infrastructure for FX cross-sectional momentum strategies.

This module provides all common functionality:
- Data loading and USD normalization
- Position management
- Trade execution with bid/ask spreads
- PnL calculation
- Performance statistics and reporting

Strategy-specific code (e.g., ranking functions) should be implemented separately.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Tuple, Callable, Optional
import glob


# =============================================================================
# TRANSACTION COSTS - BID/ASK SPREAD MODEL
# =============================================================================

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
    Add bid/ask prices to dataframe.
    If bid/ask columns already exist (from real market data), use them.
    Otherwise, calculate from mid prices using spread model.
    """
    df = df.copy()
    
    # Rename 'offer' to 'ask' for consistency if it exists
    if 'offer' in df.columns and 'ask' not in df.columns:
        df['ask'] = df['offer']
    
    # Check if we already have bid/ask data (either as close_bid/close_ask or bid/ask)
    has_close_bid_ask = all(col in df.columns for col in ['close_bid', 'close_ask'])
    has_bid_ask = all(col in df.columns for col in ['bid', 'ask'])
    
    if has_close_bid_ask:
        # Real bid/ask data exists - just ensure all price columns have bid/ask
        for col in ['open', 'high', 'low']:
            if f'{col}_bid' not in df.columns and col in df.columns:
                # If mid price exists but not bid/ask, derive from close bid/ask spread
                spread = (df['close_ask'] - df['close_bid']) / df['close']
                df[f'{col}_bid'] = df[col] * (1 - spread/2)
                df[f'{col}_ask'] = df[col] * (1 + spread/2)
    elif has_bid_ask:
        # Data has 'bid' and 'ask' columns - use them for close and derive for OHLC
        df['close_bid'] = df['bid']
        df['close_ask'] = df['ask']
        
        # Calculate spread and apply to OHLC
        if 'close' in df.columns:
            spread = df['ask'] - df['bid']
            half_spread_pct = (spread / df['close']) / 2
            
            for col in ['open', 'high', 'low']:
                if col in df.columns:
                    df[f'{col}_bid'] = df[col] * (1 - half_spread_pct)
                    df[f'{col}_ask'] = df[col] * (1 + half_spread_pct)
    else:
        # No real bid/ask - calculate from spread model
        spread_bp = PAIR_SPREADS.get(pair, 2.0)
        spread_decimal = spread_bp / 10000
        half_spread = spread_decimal / 2
        
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[f'{col}_bid'] = df[col] * (1 - half_spread)
                df[f'{col}_ask'] = df[col] * (1 + half_spread)
    
    return df


# =============================================================================
# USD NORMALIZATION
# =============================================================================

def is_usd_inverse(pair: str) -> bool:
    """Check if pair is inverse quoted (USD in base position)."""
    return pair.startswith('USD')


def transform_to_usd_terms(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """
    Transform ALL pairs to USD-in-base terms for consistent momentum calculation.
    - Regular pairs (EUR/USD): already USD-denominated, no transformation
    - Inverse pairs (USD/JPY): invert to get JPY/USD equivalent
    """
    df = df.copy()
    
    if is_usd_inverse(pair):
        # Inverse pair: USD/JPY → invert to JPY/USD equivalent
        for col in ['open', 'high', 'low', 'close', 'open_bid', 'open_ask', 
                    'high_bid', 'high_ask', 'low_bid', 'low_ask', 'close_bid', 'close_ask']:
            if col in df.columns:
                df[f'{col}_usd'] = 1.0 / df[col]
        
        # For inverse pairs: higher USD/JPY = stronger USD = WEAKER "JPY/USD equivalent"
        # So we need to NEGATE returns to get correct USD perspective
        # If USD/JPY goes up 1%, that means JPY weakened 1% vs USD
        # In our USD-centric view, this should be a NEGATIVE return for JPY
        df['returns_usd'] = -df['close'].pct_change()
    else:
        # Regular pair: EUR/USD is already in USD terms
        for col in ['open', 'high', 'low', 'close', 'open_bid', 'open_ask',
                    'high_bid', 'high_ask', 'low_bid', 'low_ask', 'close_bid', 'close_ask']:
            if col in df.columns:
                df[f'{col}_usd'] = df[col]
        
        df['returns_usd'] = df['close'].pct_change()
    
    return df


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pair_data(date_str: str, pair: str, data_dir: str = "data/bidask/output") -> Optional[pd.DataFrame]:
    """Load and prepare data for a single pair on a given date."""
    file_path = Path(data_dir) / f"{date_str}_{pair}_bars.csv"
    
    if not file_path.exists():
        return None
    
    df = pd.read_csv(file_path)
    
    # Parse datetime
    if 'Datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['Datetime'])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        raise ValueError(f"No datetime column found in {file_path}")
    
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # If we have bid/ask data but no close, calculate close as midpoint
    if 'close_bid' in df.columns and 'close_ask' in df.columns and 'close' not in df.columns:
        df['close'] = (df['close_bid'] + df['close_ask']) / 2
    
    # Same for other OHLC columns
    for col in ['open', 'high', 'low']:
        if f'{col}_bid' in df.columns and f'{col}_ask' in df.columns and col not in df.columns:
            df[col] = (df[f'{col}_bid'] + df[f'{col}_ask']) / 2
    
    # Add bid/ask spreads (will use existing data if available)
    df = add_bid_ask_to_data(df, pair)
    
    # Forward fill 0 and NaN values in price columns to prevent divide-by-zero errors
    price_cols = ['close', 'close_bid', 'close_ask', 'open', 'open_bid', 'open_ask',
                  'high', 'high_bid', 'high_ask', 'low', 'low_bid', 'low_ask']
    
    for col in price_cols:
        if col in df.columns:
            # Replace 0 with NaN, then forward fill
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].ffill()
            # If still NaN at start, backfill
            df[col] = df[col].bfill()
    
    # Transform to USD terms for momentum calculation
    df = transform_to_usd_terms(df, pair)
    
    return df


def load_all_pairs(date_str: str, pairs: List[str], data_dir: str = "data/bidask/output") -> Dict[str, pd.DataFrame]:
    """Load data for all pairs on a given date."""
    data = {}
    for pair in pairs:
        df = load_pair_data(date_str, pair, data_dir)
        if df is not None:
            data[pair] = df
    return data


# =============================================================================
# POSITION MANAGEMENT
# =============================================================================

def calculate_pnl(position_info: Dict, current_price_bid: float, current_price_ask: float, 
                  current_price_close: float, pair: str) -> Dict:
    """
    Calculate PnL for a position using bid/ask prices.
    Also calculates gross PnL using mid/close prices.
    
    Position entry uses bid (sell) or ask (buy) prices.
    Position exit uses opposite: ask (sell to close long) or bid (buy to close short).
    
    PnL Formula (works for ALL pairs):
        PnL_USD = direction * base_notional * price_change_pct
    
    Where:
        - base_notional is always in the BASE currency of the pair
        - USD/JPY: base_notional = $1M USD, price_change_pct = % change in JPY/USD rate
        - EUR/USD: base_notional = ~952K EUR, price_change_pct = % change in USD/EUR rate
        - Both formulas give PnL in USD
    """
    entry_price = position_info['entry_price']
    entry_close = position_info.get('entry_close', entry_price)  # Fallback for old positions
    direction = position_info['direction']
    base_notional = position_info['base_notional']
    
    # Determine exit price based on position direction
    if direction > 0:  # LONG position
        # To close: SELL at BID
        exit_price = current_price_bid
    else:  # SHORT position
        # To close: BUY at ASK
        exit_price = current_price_ask
    
    # Exit close is just the mid price
    exit_close = current_price_close
    
    # Calculate price change (net, with spreads)
    price_change = exit_price - entry_price
    price_change_pct = price_change / entry_price
    
    # Calculate price change (gross, mid-to-mid)
    price_change_gross = exit_close - entry_close
    price_change_pct_gross = price_change_gross / entry_close
    
    # Calculate PnL in USD using unified formula
    # For ALL pairs: PnL = direction * base_notional * price_change_pct
    pnl_usd = direction * base_notional * price_change_pct
    pnl_usd_gross = direction * base_notional * price_change_pct_gross
    
    return {
        'pnl_usd': pnl_usd,
        'pnl_usd_gross': pnl_usd_gross,
        'exit_price': exit_price,
        'exit_close': exit_close,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'price_change_gross': price_change_gross,
        'price_change_pct_gross': price_change_pct_gross,
    }


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

class FXBacktester:
    """
    Backtesting engine for FX cross-sectional momentum strategies.
    
    Handles all common functionality:
    - Data loading
    - Position management
    - Trade execution
    - PnL calculation
    - Performance reporting
    
    Strategy-specific ranking is provided via callback function.
    """
    
    def __init__(
        self,
        pairs: List[str],
        start_date: str,
        end_date: str,
        rebalance_freq: int = 60,
        top_n: int = 2,
        usd_notional: float = 1_000_000,
        start_hour: Optional[int] = None,
        end_hour: Optional[int] = None,
        data_dir: str = "data/bidask/output",
        lookback_bars: int = 30,
    ):
        """
        Initialize backtester.
        
        Args:
            pairs: List of currency pairs to trade
            start_date: Start date (YYYYMMDD format)
            end_date: End date (YYYYMMDD format)
            rebalance_freq: Rebalance frequency in minutes
            top_n: Number of pairs to go long and short
            usd_notional: USD notional per position
            start_hour: Optional start hour for trading (EST)
            end_hour: Optional end hour for trading (EST)
            data_dir: Directory containing market data
            lookback_bars: Number of historical bars needed for strategy calculations
        """
        self.pairs = sorted(pairs)
        self.start_date = datetime.strptime(start_date, '%Y%m%d')
        self.end_date = datetime.strptime(end_date, '%Y%m%d')
        self.rebalance_freq = rebalance_freq
        self.top_n = top_n
        self.usd_notional = usd_notional
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.data_dir = data_dir
        self.lookback_bars = lookback_bars
        
        # State
        self.current_positions = {}
        self.equity = 1.0
        self.results = []
        self.trades = []
        self.rebalance_counter = 0
    
    def run(self, rank_function: Callable[[Dict[str, pd.DataFrame], int], Dict[str, float]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run backtest with provided ranking function.
        
        Args:
            rank_function: Function that takes (data_dict, idx) and returns {pair: rank_score}
        
        Returns:
            (results_df, trades_df)
        """
        # Generate list of trading dates
        date_range = pd.date_range(self.start_date, self.end_date, freq='D')
        trading_dates = [d for d in date_range if d.weekday() < 5]  # Mon-Fri only
        
        print(f"\nRunning backtest from {self.start_date.strftime('%Y%m%d')} to {self.end_date.strftime('%Y%m%d')}")
        print(f"Rebalance: every {self.rebalance_freq} min, Top N: {self.top_n}")
        print(f"USD Notional per position: ${self.usd_notional:,.0f}")
        if self.start_hour is not None and self.end_hour is not None:
            print(f"Trading Hours: {self.start_hour:02d}:00 - {self.end_hour:02d}:00 EST")
        print("=" * 80)
        print()
        
        # Process each date
        for date in trading_dates:
            self._process_date(date, rank_function)
        
        # Convert to DataFrames
        results_df = pd.DataFrame(self.results)
        trades_df = pd.DataFrame(self.trades)
        
        return results_df, trades_df
    
    def _process_date(self, date: datetime, rank_function: Callable):
        """Process a single trading date."""
        date_str = date.strftime('%Y%m%d')
        
        # Load data for all pairs
        all_data = load_all_pairs(date_str, self.pairs, self.data_dir)
        
        if not all_data:
            return
        
        # Get common bars across all pairs
        common_length = min(len(df) for df in all_data.values())
        if common_length == 0:
            return
        
        # Trim all data to common length
        for pair in all_data:
            all_data[pair] = all_data[pair].iloc[:common_length].copy()
        
        # Check if we have enough data for lookback
        if common_length < self.lookback_bars:
            return
        
        # Process each rebalance point - start at lookback_bars to have enough history
        rebalance_indices = range(self.lookback_bars, common_length, self.rebalance_freq)
        
        for idx in rebalance_indices:
            
            # Check if within trading hours
            timestamp = all_data[self.pairs[0]]['datetime'].iloc[idx]
            if self.start_hour is not None and self.end_hour is not None:
                # Assume timestamp is in UTC, convert to EST
                if timestamp.tz is None:
                    timestamp_utc = timestamp.tz_localize('UTC')
                else:
                    timestamp_utc = timestamp
                est = pytz.timezone('US/Eastern')
                timestamp_est = timestamp_utc.astimezone(est)
                hour = timestamp_est.hour
                if not (self.start_hour <= hour < self.end_hour):
                    continue
            
            # Get rankings from strategy-specific function
            rankings = rank_function(all_data, idx)
            
            # Execute rebalance
            self._rebalance(all_data, idx, rankings, timestamp, date_str)
        
        if len(rebalance_indices) > 1:  # Only print if we had rebalances
            print(f"Processing {date_str}:")
            print(f"  Loaded {len(all_data)} pairs, {common_length} bars")
            print(f"  Completed {self.rebalance_counter} total rebalances, Equity: {self.equity:.4f}")

    
    def _rebalance(self, all_data: Dict[str, pd.DataFrame], idx: int, 
                   rankings: Dict[str, float], timestamp: pd.Timestamp, date_str: str):
        """Execute a rebalance based on rankings."""
        # Sort pairs by rank
        sorted_pairs = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        
        # Select top N longs and bottom N shorts
        long_pairs = [pair for pair, _ in sorted_pairs[:self.top_n]]
        short_pairs = [pair for pair, _ in sorted_pairs[-self.top_n:]]
        
        # Determine new positions with USD-centric logic (matching fx_cross_momentum.py)
        # Top pairs = base currency STRONG vs USD → SHORT base (LONG USD)
        # Bottom pairs = base currency WEAK vs USD → LONG base (SHORT USD)
        new_positions = {}
        for pair in self.pairs:
            if pair in long_pairs:
                # Top pairs: base strong → we want LONG USD
                if is_usd_inverse(pair):
                    # USD/JPY: JPY strong → LONG USD/JPY (direction +1)
                    new_positions[pair] = {'direction': 1}
                else:
                    # EUR/USD: EUR strong → SHORT EUR/USD (direction -1)
                    new_positions[pair] = {'direction': -1}
            elif pair in short_pairs:
                # Bottom pairs: base weak → we want SHORT USD
                if is_usd_inverse(pair):
                    # USD/JPY: JPY weak → SHORT USD/JPY (direction -1)
                    new_positions[pair] = {'direction': -1}
                else:
                    # EUR/USD: EUR weak → LONG EUR/USD (direction +1)
                    new_positions[pair] = {'direction': 1}
            else:
                new_positions[pair] = {'direction': 0}  # FLAT
        
        # Calculate PnL for positions being closed
        total_pnl_usd = 0
        total_transaction_cost = 0
        total_notional = 0
        positions_to_keep = {}
        
        for pair, pos_info in self.current_positions.items():
            if pos_info['direction'] == 0:
                continue
            
            new_direction = new_positions[pair]['direction']
            
            # If direction changes, close position
            if new_direction != pos_info['direction']:
                current_price_bid = all_data[pair]['close_bid'].iloc[idx]
                current_price_ask = all_data[pair]['close_ask'].iloc[idx]
                current_price_close = all_data[pair]['close'].iloc[idx]
                
                pnl_info = calculate_pnl(pos_info, current_price_bid, current_price_ask, 
                                        current_price_close, pair)
                total_pnl_usd += pnl_info['pnl_usd']
                
                # Record trade
                self._record_trade(pair, pos_info, pnl_info, timestamp, date_str, idx)
            else:
                # Keep position
                positions_to_keep[pair] = pos_info
        
        # Open new positions
        for pair, new_pos in new_positions.items():
            if new_pos['direction'] == 0:
                continue
            
            # Check if this is a new position (not kept from before)
            if pair not in positions_to_keep:
                self._open_position(pair, new_pos, all_data[pair], idx, rankings[pair])
            
            # Add to total notional
            if pair in self.current_positions and self.current_positions[pair]['direction'] != 0:
                total_notional += self.current_positions[pair]['usd_notional']
        
        # Update equity
        net_pnl_usd = total_pnl_usd
        total_pnl_pct = net_pnl_usd / (len(self.pairs) * self.usd_notional) if len(self.pairs) > 0 else 0
        self.equity *= (1 + total_pnl_pct)
        
        # Record rebalance result
        result = {
            'timestamp': timestamp,
            'date': date_str,
            'equity': self.equity,
            'pnl_net': net_pnl_usd,
            'pnl_pct': total_pnl_pct,
            'total_notional': total_notional,
            'num_positions': len([p for p in self.current_positions.values() if p['direction'] != 0]),
        }
        
        # Add rankings
        for pair in sorted(self.pairs):
            result[f'rank_{pair}'] = rankings.get(pair, np.nan)
        
        self.results.append(result)
        self.rebalance_counter += 1
        
        # CRITICAL: Update current_positions to keep positions for next rebalance
        # Merge kept positions with newly opened positions
        updated_positions = {}
        for pair in self.pairs:
            if pair in positions_to_keep:
                # Keep the existing position
                updated_positions[pair] = positions_to_keep[pair]
            elif pair in new_positions and new_positions[pair]['direction'] != 0:
                # Use the newly opened position (which is now in self.current_positions)
                if pair in self.current_positions:
                    updated_positions[pair] = self.current_positions[pair]
            # If neither, the pair stays flat (not in updated_positions)
        
        self.current_positions = updated_positions
    
    def _open_position(self, pair: str, position: Dict, data: pd.DataFrame, idx: int, rank_score: float):
        """Open a new position."""
        direction = position['direction']
        
        # Get mid/close price for gross PnL calculation
        entry_close = data['close'].iloc[idx]
        
        # Determine entry price based on direction
        if direction > 0:  # LONG
            entry_price = data['close_ask'].iloc[idx]  # BUY at ASK
        else:  # SHORT
            entry_price = data['close_bid'].iloc[idx]  # SELL at BID
        
        # Calculate position size
        is_inverse = is_usd_inverse(pair)
        if is_inverse:
            # USD/JPY @ 157: USD is BASE currency
            # $1M USD (base) = 157M JPY (quote)
            base_notional = self.usd_notional  # $1M USD
            quote_notional = self.usd_notional * entry_price  # 157M JPY
        else:
            # EUR/USD @ 1.05: USD is QUOTE currency
            # 952K EUR (base) = $1M USD (quote)
            base_notional = self.usd_notional / entry_price  # 952K EUR
            quote_notional = self.usd_notional  # $1M USD
        
        # For backwards compatibility: pair_notional = base_notional (amount in base currency)
        pair_notional = base_notional
        
        self.current_positions[pair] = {
            'direction': direction,
            'entry_price': entry_price,
            'entry_close': entry_close,  # Store mid price for gross PnL
            'entry_idx': idx,
            'usd_notional': self.usd_notional,
            'pair_notional': pair_notional,  # Kept for backwards compatibility
            'base_notional': base_notional,
            'quote_notional': quote_notional,
            'rank_score': rank_score,
        }
    
    def _record_trade(self, pair: str, pos_info: Dict, pnl_info: Dict, 
                     timestamp: pd.Timestamp, date_str: str, exit_idx: int):
        """Record a completed trade."""
        direction = pos_info['direction']
        is_inverse = is_usd_inverse(pair)
        
        # Determine execution side (ENTRY side, not exit) and currency position
        if direction > 0:  # Was LONG (entered by BUYING)
            execution_side = 'BUY'  # Entered by buying
            if is_inverse:
                currency_position = 'LONG_USD'  # Long USD via USD/JPY
            else:
                currency_position = 'SHORT_USD'  # Long EUR (short USD) via EUR/USD
        else:  # Was SHORT (entered by SELLING)
            execution_side = 'SELL'  # Entered by selling
            if is_inverse:
                currency_position = 'SHORT_USD'  # Short USD via USD/JPY
            else:
                currency_position = 'LONG_USD'  # Short EUR (long USD) via EUR/USD
        
        trade = {
            'timestamp': timestamp,
            'date': date_str,
            'pair': pair,
            'execution_side': execution_side,
            'currency_position': currency_position,
            'entry_price': pos_info['entry_price'],
            'entry_close': pos_info.get('entry_close', pos_info['entry_price']),
            'exit_price': pnl_info['exit_price'],
            'exit_close': pnl_info['exit_close'],
            'price_change': pnl_info['price_change'],
            'price_change_pct': pnl_info['price_change_pct'],
            'price_change_gross': pnl_info['price_change_gross'],
            'price_change_pct_gross': pnl_info['price_change_pct_gross'],
            'usd_notional': pos_info['usd_notional'],
            'pair_notional': pos_info['pair_notional'],
            'base_notional': pos_info['base_notional'],
            'quote_notional': pos_info['quote_notional'],
            'pnl_usd': pnl_info['pnl_usd'],
            'pnl_usd_gross': pnl_info['pnl_usd_gross'],
            'pnl_pct': pnl_info['pnl_usd'] / pos_info['usd_notional'],
            'pnl_pct_gross': pnl_info['pnl_usd_gross'] / pos_info['usd_notional'],
            'rank_score': pos_info.get('rank_score', np.nan),
        }
        
        self.trades.append(trade)


# =============================================================================
# PERFORMANCE STATISTICS
# =============================================================================

def calculate_statistics(results_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
    """Calculate performance statistics."""
    if len(results_df) == 0:
        return {}
    
    # Total return
    final_equity = results_df['equity'].iloc[-1]
    total_return = (final_equity - 1.0) * 100
    
    # Trade statistics
    num_trades = len(trades_df)
    num_rebalances = len(results_df)
    
    if num_trades > 0:
        winning_trades = (trades_df['pnl_usd'] > 0).sum()
        losing_trades = (trades_df['pnl_usd'] < 0).sum()
        win_rate = (winning_trades / num_trades * 100)
        
        avg_win = trades_df[trades_df['pnl_usd'] > 0]['pnl_pct'].mean() * 100 if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl_usd'] < 0]['pnl_pct'].mean() * 100 if losing_trades > 0 else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
    
    # PnL stats
    total_pnl_net = results_df['pnl_net'].sum()
    
    # Sharpe ratio
    returns = results_df['pnl_pct'].values
    if len(returns) > 0 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    # Max drawdown
    equity_curve = results_df['equity'].values
    cummax = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - cummax) / cummax
    max_drawdown = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0
    
    # Date range
    num_days = results_df['date'].nunique()
    
    stats = {
        'Total Return (%)': total_return,
        'Final Equity': final_equity,
        'PnL Net (USD)': total_pnl_net,
        'Num Rebalances': num_rebalances,
        'Total Trades': num_trades,
        'Trades per Rebalance': num_trades / num_rebalances if num_rebalances > 0 else 0,
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
