# BB Exit Configuration Update

## Changes Made

### 1. Removed Backup PT/SL/max_hold for BB Exits

**Previous Behavior:**
- BB exit strategies required `pt`, `sl`, and `max_hold` values as backup exits
- These backup exits could trigger before the intended BB exit logic
- Config had to specify values like `pt: 0.002`, `sl: 0.001`, `max_hold: 24`

**New Behavior:**
- BB exit strategies can set `pt: null`, `sl: null`, `max_hold: null` to disable backup exits
- The strategy relies purely on BB exit logic (middle band, opposite band, partial reversion, or signal reversal)
- No premature exits from PT/SL - only the BB strategy determines exit timing

**Code Changes:**
```python
# In generate.py line ~1095
if use_bb_exit:
    # For BB exits, PT/SL/max_hold are optional (can be None to disable backup exits)
    pt_use = pt if pt is not None else float('inf')  # Effectively disabled
    sl_use = sl if sl is not None else float('inf')  # Effectively disabled
    max_hold_use = max_hold if max_hold is not None else 1000000  # Effectively disabled
```

### 2. Read BB Parameters from Signal Configuration

**Previous Behavior:**
- Each label config had to duplicate `bb_window` and `bb_std` settings
- Risk of mismatch between signal generation and exit detection
- Example:
  ```yaml
  signal_params:
    bb_window: 20
    bb_std: 2.0
  
  labels:
    bb_middle:
      bb_window: 20  # Duplicated - must match manually
      bb_std: 2.0    # Duplicated - must match manually
  ```

**New Behavior:**
- BB exit configs automatically read `bb_window` and `bb_std` from `signal_params`
- Single source of truth for BB parameters
- No manual synchronization needed

**Code Changes:**
```python
# In generate.py line ~1095
if use_bb_exit:
    # Read BB params from signal_params to ensure consistency
    signal_params = config.get('signal_params', {})
    bb_exit_config = {
        'strategy': label_config.get('bb_exit_strategy', 'middle'),
        'bb_window': signal_params.get('bb_window', 20),  # Read from signal config
        'bb_std': signal_params.get('bb_std', 2.0),  # Read from signal config
        'exit_std_multiplier': label_config.get('exit_std_multiplier', 0.5)
    }
```

### 3. Improved Filename Generation

**Previous Behavior:**
- Filenames used `{date}_{pair}_{PT}_{SL}_{HOLD}_y.parquet` format
- With null values: `20250106_USDJPY_None_None_None_y.parquet` ❌

**New Behavior:**
- BB exits without backup use strategy name in filename
- Format: `{date}_{pair}_{label_name}_y.parquet`
- Example: `20250106_USDJPY_bb_middle_y.parquet` ✅

**Code Changes:**
```python
# In generate.py line ~1257
if pt is None and sl is None and max_hold is None:
    base_filename = f"{d}_{spot}_{label_name}_y"
else:
    base_filename = f"{d}_{spot}_{pt}_{sl}_{max_hold}_y"
```

## Updated Config Format

### Old Config (verbose, risk of mismatch)
```yaml
signal_strategy: "bb_mean_reversion"
signal_params:
  bb_window: 20
  bb_std: 2.0

labels:
  bb_middle:
    pt: 0.002              # Backup exit
    sl: 0.001              # Stop loss
    max_hold: 24           # Backup max hold
    use_bb_exit: true
    bb_exit_strategy: "middle"
    bb_window: 20          # Duplicated ❌
    bb_std: 2.0            # Duplicated ❌
```

### New Config (clean, single source of truth)
```yaml
signal_strategy: "bb_mean_reversion"
signal_params:
  bb_window: 20
  bb_std: 2.0

labels:
  bb_middle:
    pt: null               # No backup PT - rely on BB exit only ✅
    sl: null               # No backup SL - rely on BB exit only ✅
    max_hold: null         # No backup max hold - rely on BB exit only ✅
    use_bb_exit: true
    bb_exit_strategy: "middle"
    # BB params read from signal_params automatically ✅
```

## Benefits

### 1. Cleaner Strategy Testing
- Test pure BB mean reversion without interference from arbitrary PT/SL levels
- Exit timing driven entirely by market structure (BB levels) not preset thresholds

### 2. Configuration Simplicity
- Fewer parameters to maintain
- No risk of signal/exit parameter mismatch
- Easier to modify BB settings (change once in signal_params)

### 3. Better Filenames
- Clear identification of which strategy generated the labels
- Filename reflects the actual exit logic used

### 4. Validation Confidence
- All 4 BB exit strategies pass validation (0 violations)
- Exits occur before next signal fires (no data leakage)
- Geometric properties of BB mean reversion confirmed in practice

## Migration Guide

If you have existing configs with BB exits:

1. **Set PT/SL/max_hold to null:**
   ```yaml
   pt: null
   sl: null
   max_hold: null
   ```

2. **Remove BB parameter duplicates:**
   - Delete `bb_window` and `bb_std` from label configs
   - Keep only in `signal_params`

3. **Update train.py label file lookups:**
   - Old: `20250106_USDJPY_0.002_0.001_24_y.parquet`
   - New: `20250106_USDJPY_bb_middle_y.parquet`

4. **Regenerate all label files** to get consistent naming

## Testing

Tested on 2025-01-06 with config_bb_exits.yaml:
- ✅ All 4 strategies generate labels successfully
- ✅ Validation passes (0 violations)
- ✅ Filenames use strategy names correctly
- ✅ BB params read from signal_params (no duplication)
- ✅ No backup exits triggered (pure BB logic)

Example output:
```
Strategy 'bbmiddle': Validation passed - all exits occur before next signal
Added 11 lagged features for 'bb_middle' with prefix 'bbmiddle'
Saved labels for 'bb_middle': 20250106_USDJPY_bb_middle_y (.parquet + .csv)
  Stats: 25/37 profitable (67.6%), avg profit: -0.022459
```
