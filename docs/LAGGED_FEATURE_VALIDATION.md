# Lagged Profit Feature Validation

## Summary

Added validation to the lagged profit feature generation to ensure no data leakage occurs. The system now checks that **every exit timestamp occurs before or at the next signal timestamp** for each strategy on each day.

## Validation Logic

During `add_lagged_profit_features()`:

1. **Check each signal's exit** against the next signal's entry time
2. **Count violations** where `exit_timestamp > next_signal_timestamp`  
3. **Calculate max overlap** to show worst-case leakage
4. **Skip lagged features** for any strategy that has violations
5. **Report summary** at INFO level for valid strategies, WARNING level for invalid

## Validation Results (2025-01-06)

All BB exit strategies **PASSED** validation with 0 violations:

- ✅ **bb_middle**: All exits occur before next signal
- ✅ **bb_opposite**: All exits occur before next signal  
- ✅ **bb_partial_50**: All exits occur before next signal
- ✅ **bb_signal**: All exits occur before next signal

**Validation Summary**: 4 valid, 0 invalid

## Log Output Format

### When Valid (all strategies passed):
```
2025-10-06 11:06:31,662 - INFO - Strategy 'bbmiddle': Validation passed - all exits occur before next signal
2025-10-06 11:06:31,665 - INFO - Added 11 lagged features for 'bb_middle' with prefix 'bbmiddle'
...
2025-10-06 11:06:31,676 - INFO - Lagged feature validation summary: 4 valid, 0 invalid
2025-10-06 11:06:31,676 - INFO - Total lagged features added: 44 from 4 strategies
```

### When Invalid (hypothetical example):
```
2025-10-06 XX:XX:XX,XXX - WARNING - Strategy 'bad_strategy': 15/80 exits occur AFTER next signal (max overlap: 0 days 02:30:00). Skipping lagged features.
2025-10-06 XX:XX:XX,XXX - WARNING - Strategies excluded from lagged features due to data leakage:
2025-10-06 XX:XX:XX,XXX - WARNING -   - bad_strategy: 15 violations, max overlap: 0 days 02:30:00
```

## Why This Matters

**Geometric Constraint**: For BB mean reversion strategies, if a long signal fires at `price < MA - 2σ`, the price MUST cross `MA` before reaching `MA + 2σ` (the opposing signal threshold). This is mathematically guaranteed since `MA < MA + 2σ`.

**Implication**: Any exit strategy based on MA crossings (middle, partial, or signal) will naturally exit before an opposing signal can fire. The validation confirms this mathematical property holds in practice.

**Cross-Day Behavior**: Signal alternation state resets each day, but lagged features use `.shift(1)` which pulls from the previous day's signals. The validation ensures that even when consecutive same-direction signals occur (within a day), the exit times don't violate the no-future-data constraint.

## Code Changes

### Modified Function Signature
```python
def add_lagged_profit_features(features_df, labels_df, strategy_name='', lookback_days=5):
    """
    Returns:
        Tuple of (features_df, list of added feature names, validation_status dict)
    """
```

### Validation Status Dictionary
```python
validation_status = {
    'valid': True/False,           # Overall pass/fail
    'total_signals': int,          # Number of signals checked
    'violations': int,             # Count of exit > next_signal
    'max_overlap': str or None,    # Worst-case time overlap
    'strategy_name': str           # Strategy identifier
}
```

### Calling Code Updates
- Caller now receives validation status
- Only adds features if `validation_status['valid'] == True`
- Logs summary of valid vs invalid strategies
- Reports details for any excluded strategies

## Testing

### Unit Test Results
```python
# Good data: All exits before next signal
validation_status = {
    'valid': True, 
    'total_signals': 10, 
    'violations': 0, 
    'max_overlap': None
}
Features added: 11

# Bad data: Some exits after next signal  
validation_status = {
    'valid': False,
    'total_signals': 10,
    'violations': 4,
    'max_overlap': '0 days 01:00:00'
}
Features added: 0
```

### Real Data Test (2025-01-06)
- **37 signals** across 4 strategies
- **0 violations** for all strategies
- **44 lagged features** successfully added (11 per strategy × 4)

## Conclusion

The validation system confirms that:

1. All BB exit strategies respect the no-future-data constraint
2. The geometric properties of BB mean reversion hold in practice
3. Lagged profit features can be safely used without data leakage concerns
4. Any future strategy additions will be automatically validated

This validates your insight that "it isn't possible to have an opposite signal before you hit the MA line" - the code confirms this mathematically guaranteed property.
