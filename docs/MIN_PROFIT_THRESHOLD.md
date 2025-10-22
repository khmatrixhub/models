# Minimum Profit Threshold Feature

## Problem Statement

Your baseline results show good accuracy (72.7%) but very small average profit per trade (0.002348 or 0.23%). This is too small to be profitable after accounting for:
- Bid-ask spread
- Transaction costs
- Slippage

```
threshold  count  accuracy  mean_profit
    0.00  16244  0.727407     0.002348   ← 0.23% too small!
```

**Goal**: Train the model to identify trades with LARGER profit potential (e.g., > 0.5% or > 1.0%).

## Solution: Minimum Profit Threshold

Instead of training the model to distinguish "profit vs loss", train it to distinguish "BIG profit vs everything else".

### How It Works

```yaml
training:
  min_profit_threshold: 0.005  # 0.5% minimum profit
```

**Label Reclassification**:
- Original: `profit > 0` → Win (label=1)
- With threshold: `profit > 0.005` → Win (label=1)
- Small profits (0% to 0.5%) → Loss (label=0)

**Effect**: Model learns to identify trades with potential for larger gains, not just any positive return.

## Configuration Options

### Conservative (Find 0.5%+ winners)
```yaml
training:
  min_profit_threshold: 0.005  # 0.5%
```
**Use case**: 2-3x your current average, more achievable

### Moderate (Find 1%+ winners)
```yaml
training:
  min_profit_threshold: 0.01  # 1.0%
```
**Use case**: 4x your current average, solid returns

### Aggressive (Find 2%+ winners)
```yaml
training:
  min_profit_threshold: 0.02  # 2.0%
```
**Use case**: Very selective, only big moves

### Disabled (Default behavior)
```yaml
training:
  min_profit_threshold: 0.0  # Any profit counts as win
```

## Expected Trade-offs

| Threshold | Win Rate | Avg Profit | Trade Count | Selectivity |
|-----------|----------|------------|-------------|-------------|
| 0.000     | 72.7%    | 0.23%      | High        | Low         |
| 0.005     | ~40-50%  | 0.8-1.2%   | Medium      | Medium      |
| 0.010     | ~20-30%  | 1.5-2.5%   | Low         | High        |
| 0.020     | ~5-10%   | 3.0-5.0%   | Very Low    | Very High   |

**Key Insight**: You're training the model to be more selective, trading frequency for quality.

## Usage Examples

### Example 1: Find 0.5% Winners
```bash
# Edit config
training:
  reverse_signals: true
  min_profit_threshold: 0.005

# Run training
python train.py --config configs/config_example.yaml \
                --experiment big_winners_0.5pct \
                --label-name conservative
```

**Expected Results**:
- Win rate: ~45% (down from 72%)
- Mean profit: ~1.0% (up from 0.23%)
- Trade count: ~7,000 (down from 16,000)
- **Total profit**: Potentially HIGHER due to better quality

### Example 2: Find 1% Winners
```bash
# Edit config
training:
  reverse_signals: true
  min_profit_threshold: 0.01

# Run training
python train.py --config configs/config_example.yaml \
                --experiment big_winners_1.0pct \
                --label-name conservative
```

**Expected Results**:
- Win rate: ~25% (down from 72%)
- Mean profit: ~2.0% (up from 0.23%)
- Trade count: ~4,000 (down from 16,000)
- **ROI**: Potentially MUCH higher per trade

## Interpreting Results

### Success Criteria

✅ **Good outcome**:
```
threshold  count  accuracy  mean_profit  total_profit
    0.50   5234  0.45       0.010        52.34
```
- Lower trade count but higher quality
- Mean profit significantly above threshold
- Total profit competitive with or better than baseline

✅ **Excellent outcome**:
```
threshold  count  accuracy  mean_profit  total_profit
    0.50   3821  0.38       0.015        57.32
```
- Very selective (23% of original trades)
- Much higher profit per trade
- Better total profit despite fewer trades

### Warning Signs

⚠️ **Model can't find bigger winners**:
```
threshold  count  accuracy  mean_profit  total_profit
    0.50    156  0.25       0.012        1.87
```
- Very few predictions above threshold
- Model lacks signal to distinguish big winners
- May need better features or different approach

⚠️ **Mean profit barely above threshold**:
```
threshold  count  accuracy  mean_profit  total_profit
    0.50   5000  0.40       0.0055       27.50
```
- Model is just barely meeting threshold
- Not finding genuinely large winners
- Try higher threshold or review features

## Feature Engineering Implications

When training for bigger winners, certain features become more important:

### More Important:
- **Volatility**: Bigger moves need bigger price swings
  - `atr`, `vol_range`, `bb_position`
- **Momentum**: Strong trends produce larger profits
  - `rsi` extremes, `ma_distance`
- **Recent performance**: Hot streaks may continue
  - `winning_streak`, `recent_win_rate`

### Less Important:
- Small technical signals
- Marginal crossovers
- Noise-sensitive features

**Recommendation**: After training with min_profit_threshold, check feature importance to see what predicts big winners.

## Combining with Signal Reversal

You can use BOTH features together:

```yaml
training:
  reverse_signals: true          # Flip signals if currently negative
  min_profit_threshold: 0.01     # Find 1%+ winners in reversed direction
```

This is powerful when:
1. Original signals are negative (need reversal)
2. Even reversed signals produce small profits (need quality filter)

## Workflow Recommendation

### Step 1: Baseline (Already Done)
```yaml
reverse_signals: false
min_profit_threshold: 0.0
```
Result: -0.23% mean profit (needs reversal)

### Step 2: Reverse Signals
```yaml
reverse_signals: true
min_profit_threshold: 0.0
```
Expected: +0.23% mean profit (confirmed edge, but too small)

### Step 3: Find Bigger Winners
```yaml
reverse_signals: true
min_profit_threshold: 0.005  # Start conservative
```
Expected: Higher quality trades, better total profit

### Step 4: Iterate
Try different thresholds:
- 0.005 (0.5%)
- 0.0075 (0.75%)
- 0.01 (1.0%)
- 0.015 (1.5%)

Compare total_profit across experiments to find optimal balance.

## Metadata Tracking

The threshold is saved in experiment metadata:

```json
{
  "experiment_name": "big_winners_1pct",
  "training_config": {
    "reverse_signals": true,
    "min_profit_threshold": 0.01,
    "n_estimators": 100
  }
}
```

## Logging Output

When enabled, you'll see:

```
============================================================
MINIMUM PROFIT THRESHOLD ENABLED
Only trades with profit > 0.0100 (1.00%) will be labeled as 'wins'
This trains the model to identify BIGGER winners
Trades with smaller profits will be treated as losses
============================================================

DEBUG: Applied min_profit_threshold=0.0100
DEBUG: Original wins: 11811/16244, Filtered wins: 4127/16244 (25.4%)
DEBUG: Big winner profits - Mean: 0.0245, Median: 0.0210
```

## Practical Considerations

### Transaction Costs

Typical forex costs per round-trip:
- Spread: 0.5-2 pips (0.005%-0.02% for most pairs)
- Commission: 0.001%-0.003%
- **Total**: ~0.01%-0.03% per trade

**Break-even**: Need mean profit > 0.01% just to cover costs

**Profitable**: Need mean profit > 0.02%-0.03% for consistent gains

**Recommended**: Set `min_profit_threshold` to at least 0.005 (0.5%), ideally 0.01 (1.0%) or higher.

### Model Capacity

If model struggles to find bigger winners:
1. **Check feature coverage**: Do features capture volatility/momentum?
2. **Try different label**: Maybe use `swing` label config (0.5% PT)
3. **Add regime features**: Market volatility, trend strength
4. **Consider multi-class**: Predict profit bucket, not just binary

## Summary

| Setting | Purpose | Trade-off |
|---------|---------|-----------|
| `0.0` | Default - any profit | Max trades, min quality |
| `0.005` | 0.5% minimum | Balanced |
| `0.01` | 1.0% minimum | Quality over quantity |
| `0.02+` | 2.0%+ minimum | Very selective |

**Key Principle**: Trading is about QUALITY, not QUANTITY. Better to make 10 trades at 2% profit than 100 trades at 0.2% profit.

Your 0.23% mean profit suggests you need `min_profit_threshold` of at least 0.005, likely 0.01 for practical profitability.
