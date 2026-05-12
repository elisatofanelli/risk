# Test Plan

## Purpose

Verify that the risk project computes prices, calibrations, VaR estimates, and backtests correctly using modular, reusable code, in accordance with standard model risk management practices.

## Scope

This test plan covers:

- Data quality and CSV input validation
- Black-Scholes pricing and delta behaviors
- Portfolio valuation and aggregation
- Historical calibration
- Historical, Monte Carlo, and parametric VaR
- Rolling VaR backtesting and clustering analysis
- Report generation

## Unit Tests

- Black-Scholes call and put prices are strictly positive. [`tests/test_pricing.py`]
- Option prices maintain strict monotonicity with respect to spot price and implied volatility. [`tests/test_pricing.py`]
- Portfolio valuation computes correct aggregated values and returns a numeric float result. [`tests/test_portfolio.py`]
- Dynamic volatility helper correctly increases implied volatility when spot prices experience a negative shock (leverage effect). [`tests/test_risk_models.py`]

## Special Cases and Behavioral Tests

- **Put-Call Parity:** Verifies internal consistency by confirming the parity holds across a range of strikes. [`tests/test_pricing.py`]
- **Delta Boundaries:** Option delta is bounded correctly ([0, 1] for calls, [-1, 0] for puts). [`tests/test_pricing.py`]
- **Time to Maturity Zero:** Confirms the pricing engine correctly defaults to intrinsic value (avoiding division by zero) when options expire. [`tests/test_pricing.py`]
- **Zero Volatility / Zero Strike:** Ensures the system mathematically rejects impossible physical states (volatility or strike = 0) to prevent silent computational failures. [`tests/test_pricing.py`]

## Data Testing and Input Validation

- **Missing Fields:** Input validation actively rejects missing required columns or missing option parameters. [`tests/test_portfolio.py`]
- **Invalid Types:** Validation rejects invalid instrument types (accepting only 'stock' and 'option', with 'call'/'put' sub-types). [`tests/test_portfolio.py`]
- **Data Reasonableness:** Non-numeric fields and non-positive historical prices are explicitly rejected to prevent corrupted log-return calculations. [`tests/test_risk_models.py`]

## Integration Tests

- Load an arbitrary sample portfolio and sample historical prices together. [`tests/test_risk_models.py`]
- Calibrate statistical parameters (mean, covariance) from history and successfully run all three risk models. [`tests/test_risk_models.py`]
- Write the final computed risk metrics smoothly to the `outputs/` directory. [`tests/test_risk_models.py`]

## Model Validation Tests

- Covariance matrix dimensions precisely match the number of assets in the underlying price history. [`tests/test_risk_models.py`]
- Historical scenario repricing accurately uses current spot prices combined with historical log-return shocks. [`tests/test_risk_models.py`]
- Monte Carlo simulation paths are fully reproducible across multiple runs by utilizing a fixed random seed. [`tests/test_risk_models.py`]
- VaR and Expected Shortfall outputs strictly produce non-negative loss values. [`tests/test_risk_models.py`]

## Robustness Tests

- **Extreme Scenarios:** Evaluate model behavior when subjected to exceptionally large historical outliers (e.g., extreme single-day market crashes) to ensure the system does not fail on NaN standard deviations. [`tests/test_risk_models.py`]
- **Covariance Stability:** Confirm the calibration engine gracefully handles short historical windows without crashing when generating matrices. [`tests/test_risk_models.py`]

## Backtesting Tests

- Rolling backtest effectively generates the required schema and columns for daily evaluation. [`tests/test_backtesting.py`]
- Exception flags strictly return boolean structures. [`tests/test_backtesting.py`]
- Actual exception rate is bounded reasonably between 0 and 1. [`tests/test_backtesting.py`]
- No lookahead bias is introduced; the engine is strictly prohibited from accessing data beyond the rolling calibration window. [`tests/test_backtesting.py`]
- The summary function effectively captures unconditional coverage metrics and executes a clustering analysis to detect consecutive daily exceptions. [`tests/test_backtesting.py`]
- Summary output is automatically created and persisted to disk. [`tests/test_backtesting.py`]

## Known Limitations

- Parametric VaR relies strictly on a first-order delta-normal approximation, ignoring gamma, vega, and higher-order convex option risks.
- The simple backtest pass/fail rule serves as a quick diagnostic tolerance check, not a fully comprehensive regulatory validation.
- Kupiec unconditional coverage is evaluated, but more complex independence testing (like Christoffersen's) is not implemented for the scope of this engine.