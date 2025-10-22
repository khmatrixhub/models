# BB Exit Training Compatibility Fix

## Problem
After updating `generate.py` to use strategy-based filenames for BB exits (when PT/SL/max_hold are all null), `train.py` was still looking for files with the old `None_None_None` naming pattern.

**Error**:
```
2025-10-06 11:30:46,356 - INFO - Looking for label file: 20250930_USDJPY_None_None_None_y.parquet
2025-10-06 11:30:46,356 - WARNING - Missing label file for 20250930: 20250930_USDJPY_None_None_None_y.parquet
```

## Root Cause
Two issues in `train.py`:

1. **Parameter extraction** (line 404-406): Used direct dictionary access `selected_label['pt']` which would fail if key missing
2. **Filename construction** (line 499): Always used `{pt}_{sl}_{max_hold}` format, creating `None_None_None` when values were null

## Solution

### Change 1: Optional Parameter Extraction (lines 404-406)
```python
# OLD (required parameters):
pt = selected_label['pt']
sl = selected_label['sl']
max_hold = selected_label['max_hold']

# NEW (optional parameters):
pt = selected_label.get('pt', None)
sl = selected_label.get('sl', None)
max_hold = selected_label.get('max_hold', None)
```

### Change 2: Conditional Filename Construction (lines 499-503)
```python
# OLD (always uses PT_SL_HOLD format):
labels_file = data_dir / f"{date_str}_{spot}_{pt}_{sl}_{max_hold}_y.parquet"

# NEW (matches generate.py logic):
if pt is None and sl is None and max_hold is None:
    labels_file = data_dir / f"{date_str}_{spot}_{label_name}_y.parquet"
else:
    labels_file = data_dir / f"{date_str}_{spot}_{pt}_{sl}_{max_hold}_y.parquet"
```

## Result
Training now correctly finds label files for BB exit strategies:

**Success**:
```
2025-10-06 11:33:46,313 - INFO - Looking for label file: 20250102_USDJPY_bb_middle_y.parquet
2025-10-06 11:33:46,467 - INFO - Looking for label file: 20250106_USDJPY_bb_middle_y.parquet
✓ Files found and loaded successfully
```

## Compatibility
- ✅ **BB exits with null PT/SL**: Uses strategy name (e.g., `bb_middle_y.parquet`)
- ✅ **Standard PT/SL exits**: Uses numeric format (e.g., `0.002_0.001_24_y.parquet`)
- ✅ **Omitted parameters**: `.get()` with None default allows config flexibility
- ✅ **Backward compatible**: Old configs with numeric PT/SL/max_hold still work

## Files Modified
- `/home/khegeman/source/models/train.py` (lines 404-406, 499-503)

## Testing
Verified with:
```bash
conda run -n models python train.py --config configs/config_bb_exits.yaml \
  --label-name bb_middle --experiment bb_middle \
  --start-train-date 01022025 --end-date 01102025
```

Result: ✅ All label files found, training proceeding normally

## Related Documentation
- `BB_EXIT_CONFIG_UPDATE.md` - Original config changes for BB exits
- `FOLDER_STRUCTURE.md` - File naming conventions
- `TRAINING_INTEGRATION.md` - How train.py integrates with generate.py outputs
