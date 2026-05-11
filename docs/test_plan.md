# Test Plan

## Purpose

Verify that the risk project computes prices, calibrations, VaR estimates, and backtests correctly using modular, reusable code.

## Scope

This test plan covers:

- CSV input validation
- Black-Scholes pricing and delta
- Portfolio valuation
- Historical calibration
- Historical, Monte Carlo, and parametric VaR
- Rolling VaR backtesting
- Report generation

## Unit Tests

- Black-Scholes call and put prices are positive.
- Put-call parity holds approximately.
- Input validation rejects missing option fields.
- Input validation rejects invalid instrument types and non-positive prices.
- Portfolio valuation returns a numeric result.

## Integration Tests

- Load sample portfolio and sample historical prices together.
- Calibrate returns from history and run all three risk models.
- Write the risk report CSV to `outputs/risk_report.csv`.

## Model Validation Tests

- Covariance matrix dimensions match the number of assets.
- Historical scenario repricing uses current spot prices and historical log returns.
- Monte Carlo results are reproducible with a fixed random seed.
- VaR outputs are non-negative.

## Backtesting Tests

- Rolling backtest returns the required columns.
- Exception flags are boolean.
- Actual exception rate is between 0 and 1.
- Summary output is created and persisted.
- No lookahead bias is introduced beyond the calibration window.
- Backtest exception count is reported for each method.

## Input Validation Tests

- Missing required columns raise `ValueError`.
- Invalid `type` values raise `ValueError`.
- Missing option fields raise `ValueError`.
- Non-numeric fields are rejected.
- Non-positive historical prices are rejected.

## Known Limitations

- Parametric VaR uses a delta-normal approximation and ignores gamma, vega, and higher-order option risks.
- The simple backtest pass/fail rule is a diagnostic only, not a full regulatory test.
- Kupiec unconditional coverage is included as a basic statistical check, but Christoffersen independence testing is not yet implemented.
- Scenario-based methods use one-day log-return shocks and reprice the full portfolio at each scenario.

