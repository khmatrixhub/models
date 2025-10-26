#!/usr/bin/env python3
"""
Pytest unit tests for calculate_pnl() function in fx_backtest_base.py

Tests the unified PnL formula:
    pnl_usd = direction * base_notional * price_change_pct

Covers both USD-base pairs (USD/JPY) and USD-quote pairs (EUR/USD),
and both LONG and SHORT positions.
"""

import sys
import os
import logging

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from cross_mom.fx_backtest_base import calculate_pnl

logger = logging.getLogger(__name__)


def test_usdjpy_long_profit():
    """Test LONG USD/JPY with profit (USD strengthens)."""
    position = {
        'entry_price': 155.00,
        'entry_close': 155.00,
        'direction': 1,
        'base_notional': 1_000_000,
        'quote_notional': 155_000_000,
        'usd_notional': 1_000_000,
    }
    
    result = calculate_pnl(position, 156.00, 156.01, 156.005, 'USDJPY')
    expected_pnl = 1_000_000 * (156.00 - 155.00) / 155.00
    
    logger.info(f"USD/JPY LONG profit: PnL=${result['pnl_usd']:,.2f}")
    assert abs(result['pnl_usd'] - expected_pnl) < 0.01


def test_usdjpy_short_profit():
    """Test SHORT USD/JPY with profit (USD weakens)."""
    position = {
        'entry_price': 155.00,
        'entry_close': 155.00,
        'direction': -1,
        'base_notional': 1_000_000,
        'quote_notional': 155_000_000,
        'usd_notional': 1_000_000,
    }
    
    result = calculate_pnl(position, 154.00, 154.01, 154.005, 'USDJPY')
    expected_pnl = -1 * 1_000_000 * (154.01 - 155.00) / 155.00
    
    logger.info(f"USD/JPY SHORT profit: PnL=${result['pnl_usd']:,.2f}")
    assert abs(result['pnl_usd'] - expected_pnl) < 0.01


def test_eurusd_long_profit():
    """Test LONG EUR/USD with profit (EUR strengthens)."""
    usd_notional = 1_000_000
    entry_price = 1.0500
    base_notional = usd_notional / entry_price
    
    position = {
        'entry_price': entry_price,
        'entry_close': entry_price,
        'direction': 1,
        'base_notional': base_notional,
        'quote_notional': usd_notional,
        'usd_notional': usd_notional,
    }
    
    result = calculate_pnl(position, 1.0600, 1.0601, 1.06005, 'EURUSD')
    expected_pnl = base_notional * (1.0600 - 1.0500) / 1.0500
    
    logger.info(f"EUR/USD LONG profit: PnL=${result['pnl_usd']:,.2f}")
    assert abs(result['pnl_usd'] - expected_pnl) < 0.01


def test_eurusd_short_profit():
    """Test SHORT EUR/USD with profit (EUR weakens)."""
    usd_notional = 1_000_000
    entry_price = 1.0500
    base_notional = usd_notional / entry_price
    
    position = {
        'entry_price': entry_price,
        'entry_close': entry_price,
        'direction': -1,
        'base_notional': base_notional,
        'quote_notional': usd_notional,
        'usd_notional': usd_notional,
    }
    
    result = calculate_pnl(position, 1.0400, 1.0401, 1.04005, 'EURUSD')
    expected_pnl = -1 * base_notional * (1.0401 - 1.0500) / 1.0500
    
    logger.info(f"EUR/USD SHORT profit: PnL=${result['pnl_usd']:,.2f}")
    assert abs(result['pnl_usd'] - expected_pnl) < 0.01


def test_usdjpy_long_loss():
    """Test LONG USD/JPY with loss (USD weakens)."""
    position = {
        'entry_price': 155.00,
        'entry_close': 155.00,
        'direction': 1,
        'base_notional': 1_000_000,
        'quote_notional': 155_000_000,
        'usd_notional': 1_000_000,
    }
    
    result = calculate_pnl(position, 154.00, 154.01, 154.005, 'USDJPY')
    expected_pnl = 1_000_000 * (154.00 - 155.00) / 155.00
    
    logger.info(f"USD/JPY LONG loss: PnL=${result['pnl_usd']:,.2f}")
    assert abs(result['pnl_usd'] - expected_pnl) < 0.01


def test_eurusd_long_loss():
    """Test LONG EUR/USD with loss (EUR weakens)."""
    usd_notional = 1_000_000
    entry_price = 1.0500
    base_notional = usd_notional / entry_price
    
    position = {
        'entry_price': entry_price,
        'entry_close': entry_price,
        'direction': 1,
        'base_notional': base_notional,
        'quote_notional': usd_notional,
        'usd_notional': usd_notional,
    }
    
    result = calculate_pnl(position, 1.0400, 1.0401, 1.04005, 'EURUSD')
    expected_pnl = base_notional * (1.0400 - 1.0500) / 1.0500
    
    logger.info(f"EUR/USD LONG loss: PnL=${result['pnl_usd']:,.2f}")
    assert abs(result['pnl_usd'] - expected_pnl) < 0.01


def test_gross_vs_net_pnl():
    """Test that gross PnL > net PnL (spread cost verification)."""
    position = {
        'entry_price': 155.01,
        'entry_close': 155.00,
        'direction': 1,
        'base_notional': 1_000_000,
        'quote_notional': 155_010_000,
        'usd_notional': 1_000_000,
    }
    
    result = calculate_pnl(position, 154.99, 155.00, 154.995, 'USDJPY')
    
    net_pnl = result['pnl_usd']
    gross_pnl = result['pnl_usd_gross']
    spread_cost = gross_pnl - net_pnl
    
    logger.info(f"Gross vs Net: Spread=${spread_cost:,.2f}")
    
    assert gross_pnl > net_pnl, f"Gross (${gross_pnl:.2f}) should be > Net (${net_pnl:.2f})"
    assert spread_cost > 0, f"Spread cost should be positive, got ${spread_cost:.2f}"
