# EMA/KAMA Crossover Strategy

## Overview

The EMA/KAMA crossover strategy combines a fast Exponential Moving Average (EMA) with Kaufman's Adaptive Moving Average (KAMA) to generate trading signals. This hybrid approach provides responsive signals in trending markets while reducing whipsaws in choppy conditions.

## Strategy Components

### Fast MA: Exponential Moving Average (EMA)
- **Purpose**: Responsive to recent price changes
- **Default**: 5-period EMA
- **Behavior**: Fixed smoothing, always responsive

### Slow MA: Kaufman's Adaptive Moving Average (KAMA)
- **Purpose**: Adaptive smoothing based on market efficiency
- **Parameters**:
  - `slow_period`: Lookback period for efficiency ratio (default: 10)
  - `slow_fast_ema`: Fast EMA period for trending markets (default: 2)
  - `slow_slow_ema`: Slow EMA period for ranging markets (default: 30)

## How KAMA Works

KAMA automatically adjusts its smoothing constant based on market conditions:

### Efficiency Ratio (ER)
```
ER = |Price Change| / Sum(|Volatility|)
```

- **High ER** (strong trend): KAMA behaves like fast EMA (2-period) → responsive
- **Low ER** (choppy market): KAMA behaves like slow EMA (30-period) → smooth
- **Medium ER**: KAMA adaptively blends between fast and slow

### Smoothing Constant (SC)
```
fast_alpha = 2 / (fast_ema + 1)
slow_alpha = 2 / (slow_ema + 1)
SC = (ER * (fast_alpha - slow_alpha) + slow_alpha)²
```

### KAMA Calculation
```
KAMA[t] = KAMA[t-1] + SC[t] * (Price[t] - KAMA[t-1])
```

## Signal Generation

### Long Signal (Bullish Crossover)
- Fast EMA crosses **above** KAMA
- Indicates upward momentum

### Short Signal (Bearish Crossover)
- Fast EMA crosses **below** KAMA
- Indicates downward momentum

### No Signal
- No crossover occurred
- Continue holding existing position or stay flat

## Configuration Parameters

### Required Parameters

```yaml
signal_strategy: "ema_kama_cross"

ma_windows:
  fast: 5  # Fast EMA period

# KAMA parameters (slow MA)
slow_period: 10        # Efficiency ratio lookback
slow_fast_ema: 2       # Fast smoothing for trends
slow_slow_ema: 30      # Slow smoothing for ranges
```

### Parameter Tuning Guide

#### Fast EMA Period (`ma_windows.fast`)
- **Lower (3-5)**: More signals, more responsive, more noise
- **Higher (8-15)**: Fewer signals, less responsive, smoother
- **Recommended**: 5 for balanced responsiveness

#### KAMA Period (`slow_period`)
- **Lower (5-10)**: More sensitive to recent efficiency changes
- **Higher (15-30)**: More stable efficiency measurement
- **Recommended**: 10 for hourly data

#### KAMA Fast EMA (`slow_fast_ema`)
- **Lower (2)**: Very responsive in strong trends
- **Higher (5-10)**: Less aggressive in trends
- **Recommended**: 2 for maximum trend capture

#### KAMA Slow EMA (`slow_slow_ema`)
- **Lower (20)**: Less smoothing in ranges
- **Higher (40-60)**: Maximum smoothing in choppy markets
- **Recommended**: 30 for balanced noise reduction

## Advantages

### 1. Adaptive Behavior
- KAMA automatically adjusts to market conditions
- No manual regime detection needed
- Single configuration works across different market states

### 2. Reduced Whipsaws
- KAMA becomes very smooth in choppy markets
- Fewer false signals compared to dual-EMA systems
- Lower transaction costs

### 3. Quick Trend Capture
- KAMA becomes responsive in strong trends
- Fast EMA provides additional responsiveness
- Earlier entry compared to dual-EMA with conservative slow period

### 4. Clear Signals
- Simple crossover logic
- Binary decision: long or short
- No ambiguous zones

## Comparison with Standard MA Crossover

| Aspect | EMA/EMA Cross | EMA/KAMA Cross |
|--------|---------------|----------------|
| Responsiveness | Fixed | Adaptive |
| Whipsaws in ranges | High | Low |
| Trend capture | Good | Better |
| Parameter tuning | 2 params | 4 params |
| Complexity | Simple | Moderate |

## Usage Example

### Basic Configuration

```yaml
spot: "USDJPY"
signal_strategy: "ema_kama_cross"

ma_windows:
  fast: 5

slow_period: 10
slow_fast_ema: 2
slow_slow_ema: 30

labels:
  conservative:
    pt: 0.001
    sl: 0.001
    max_hold: 24
```

### With MA Signal Exits

```yaml
labels:
  ma_crossover:
    pt: null  # No PT - rely on signal reversal
    sl: 0.002  # Backup stop loss
    max_hold: null  # Wait for opposite signal
    use_ma_exit: true
```

### Generation Command

```bash
python generate.py \
  --config configs/config_ema_kama.yaml \
  --start-date 01022025 \
  --end-date 01312025
```

### Training Command

```bash
python train.py \
  --config configs/config_ema_kama.yaml \
  --label-name ma_crossover
```

## Feature Generation

When using `ema_kama_cross` strategy, the following features are generated:

### MA Features
- `ma_5`: Fast 5-period EMA
- `ma_20`: KAMA (slow adaptive MA)
- `ma_50`: Standard 50-period EMA (reference)

### Signal-Adjusted Features
- `ma5_distance_adj`: (close - ma_5) / ma_5 * signal_direction
- `ma20_distance_adj`: (close - ma_20) / ma_20 * signal_direction

Note: `ma_20` contains KAMA values when `ema_kama_cross` is used, providing adaptive distance features.

## Implementation Details

### Function: `calculate_kama()`
```python
def calculate_kama(close, period=10, fast_ema=2, slow_ema=30):
    """
    Calculate Kaufman's Adaptive Moving Average
    
    Returns: pandas Series with KAMA values
    """
```

### Function: `generate_ema_kama_cross_signals()`
```python
def generate_ema_kama_cross_signals(df, fast_window=5, kama_params=None):
    """
    Generate EMA/KAMA crossover signals
    
    Returns: pandas Series with 1 (long), -1 (short), 0 (no signal)
    """
```

### Function: `calculate_technical_indicators()`
```python
def calculate_technical_indicators(
    df, 
    fast_window=5, 
    slow_window=20,
    use_kama=False,
    kama_params=None
):
    """
    Calculate technical indicators
    
    When use_kama=True, ma_20 becomes KAMA instead of EMA
    """
```

## Testing

Run the test suite to verify implementation:

```bash
python working_files/tests/test_ema_kama.py
```

Tests cover:
1. KAMA calculation correctness
2. Signal generation and alternation
3. Technical indicator integration
4. Config parameter extraction

## Performance Expectations

### Compared to Standard MA Cross

**Expected Improvements:**
- 20-30% reduction in whipsaw trades (choppy markets)
- 5-15% faster trend entries (strong trends)
- Similar or better Sharpe ratio

**Trade-offs:**
- Slightly more complex to understand
- 4 parameters vs 2 (more tuning options)
- Marginally slower computation

### Best Market Conditions

**Optimal:**
- Trending markets with intermittent consolidations
- Markets with clear trend/range cycles
- Medium to high volatility

**Challenging:**
- Extremely choppy, mean-reverting markets
- Very low volatility (all MAs converge)
- Markets with abrupt regime changes

## Troubleshooting

### No Signals Generated

**Check:**
- KAMA parameters reasonable? (period > 0, fast_ema < slow_ema)
- Sufficient data? (need at least `slow_period` + warmup)
- Price data has variance? (flat prices = no signals)

### Too Many Signals

**Solutions:**
- Increase `fast_window` (5 → 8)
- Increase `slow_period` (10 → 15)
- Reduce `slow_fast_ema` responsiveness (2 → 3)

### Too Few Signals

**Solutions:**
- Decrease `fast_window` (5 → 3)
- Decrease `slow_period` (10 → 7)
- Increase `slow_fast_ema` responsiveness (2 → 1)

### Signals Not Alternating

**This should never happen** - if it does:
- Check implementation in `generate_ema_kama_cross_signals()`
- Verify crossover logic uses `ma_diff` and `ma_diff_prev`
- Run test suite: `python working_files/tests/test_ema_kama.py`

## References

### KAMA Original Paper
- Kaufman, Perry J. (1995). "Smarter Trading"
- KAMA designed to adapt smoothing to market efficiency

### Implementation Notes
- KAMA calculation is recursive (requires loop)
- First `period` bars have NaN values (warmup)
- Efficiency Ratio ranges from 0 (no trend) to 1 (pure trend)
- Smoothing Constant is squared for smoother response

## Related Strategies

- **Standard MA Cross**: `signal_strategy: "ma_cross"` - EMA/EMA crossover
- **BB Mean Reversion**: `signal_strategy: "bb_mean_reversion"` - Bollinger Band strategy

## File Locations

- **Config**: `configs/config_ema_kama.yaml`
- **Implementation**: `generate.py` (lines 238-460)
- **Tests**: `working_files/tests/test_ema_kama.py`
- **Documentation**: `docs/EMA_KAMA_STRATEGY.md` (this file)
