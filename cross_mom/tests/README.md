# Cross Momentum Tests

Unit tests for the FX momentum trading framework using pytest.

## Running Tests

### Run all tests
```bash
# Using pytest directly
pytest cross_mom/tests/ -v

# Using the test runner script
python cross_mom/run_tests.py
```

### Run specific test file
```bash
pytest cross_mom/tests/test_calculate_pnl.py -v
```

### Run specific test
```bash
pytest cross_mom/tests/test_calculate_pnl.py::test_usdjpy_long_profit -v
```

## Test Coverage

### test_calculate_pnl.py
Comprehensive pytest unit tests for PnL calculation function.

**Coverage:**
- USD/JPY (USD-base) LONG with profit
- USD/JPY SHORT with profit
- EUR/USD (USD-quote) LONG with profit
- EUR/USD SHORT with profit
- USD/JPY LONG with loss
- EUR/USD LONG with loss
- Gross vs Net PnL (spread cost verification)

**Formula tested:**
```python
pnl_usd = direction * base_notional * price_change_pct
```

This unified formula works for both USD-base (USD/JPY) and USD-quote (EUR/USD) pairs.

## Test Philosophy

- **Pytest framework**: Industry-standard testing with proper assertions and fixtures
- **Logging over printing**: Tests use `logger.info()` for pytest-compatible output
- **Permanent tests**: These tests are part of the production codebase
- **Unit-level**: Focus on individual functions with clear inputs/outputs
- **Comprehensive**: Cover both profit and loss scenarios
- **Documentation**: Tests serve as documentation of expected behavior

## Configuration

Tests are configured via `pytest.ini` at the project root:
- Test discovery: `test_*.py` files in `cross_mom/tests/`
- Verbose output by default
- Short tracebacks for cleaner error messages

## Adding New Tests

When adding new tests:
1. Place in appropriate test file (or create new `test_*.py` file)
2. Follow naming convention: `test_<functionality>.py`
3. Include docstrings explaining what is being tested
4. Verify edge cases and error conditions
5. Update this README with coverage information
