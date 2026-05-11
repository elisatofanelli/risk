# Software Design

## Architecture Overview

The project follows a layered design:

1. Input and validation layer
2. Market data and calibration layer
3. Pricing and portfolio valuation layer
4. Risk model layer
5. Backtesting and reporting layer
6. Command-line entry point

The goal is to keep responsibilities separated so each module is testable and reusable.

## Module-by-Module Description

### `src/input_loader.py`

- Loads standardized portfolio CSV files
- Validates columns, types, numeric fields, and option-specific requirements
- Converts maturity to datetime

### `src/market_data.py`

- Loads historical price data
- Validates date formatting and positive prices
- Returns a date-indexed `DataFrame`

### `src/calibration.py`

- Computes log returns
- Estimates mean, covariance, volatility, and correlation statistics
- Produces daily and annualized calibration outputs

### `src/pricing.py`

- Implements Black-Scholes price and delta for European calls and puts
- Handles expiry and edge-case validation

### `src/portfolio.py`

- Values single instruments and full portfolios
- Computes delta exposures by underlying
- Keeps stocks and options on the same valuation path

### `src/risk_models.py`

- Implements historical VaR/ES
- Implements Monte Carlo VaR/ES
- Implements parametric delta-normal VaR

### `src/backtesting.py`

- Runs rolling one-day-ahead backtests
- Summarizes exception behavior
- Includes Kupiec unconditional coverage as a basic statistical diagnostic

### `src/reporting.py`

- Writes risk report CSVs
- Writes backtest result and summary CSVs

### `src/main.py`

- Orchestrates the full workflow
- Loads inputs
- Calibrates market data
- Runs risk models
- Runs backtests
- Saves outputs

## Data Flow Diagram in Text Form

```text
portfolio CSV + historical price CSV
        |
        v
input_loader.py + market_data.py
        |
        v
calibration.py ----> calibration_result
        |
        +--> risk_models.py ----> VaR / ES results
        |
        +--> backtesting.py ----> rolling backtest rows + summary
        |
        v
reporting.py ----> CSV outputs in outputs/
```

## Main Functions and Responsibilities

- `load_portfolio(path)` validates and loads the portfolio
- `load_price_history(path)` validates and loads historical prices
- `compute_log_returns(price_df)` computes return series
- `calibrate_from_history(price_df)` estimates market statistics
- `black_scholes_price(...)` prices European options
- `black_scholes_delta(...)` computes option delta
- `value_portfolio(...)` computes current portfolio value
- `historical_var_es(...)` computes scenario-based risk
- `monte_carlo_var_es(...)` computes simulation-based risk
- `parametric_var(...)` computes delta-normal VaR
- `backtest_var(...)` produces rolling backtest data
- `summarize_backtest(...)` summarizes exceptions and coverage
- `create_risk_report(...)` writes model outputs to CSV
- `save_backtest_results(...)` writes backtest paths to CSV
- `save_backtest_summary(...)` writes summary tables to CSV

## Input and Output Schemas

### Portfolio Input Schema

`instrument_id,type,underlying,position,strike,maturity,option_type,implied_vol,risk_free_rate`

### Historical Price Input Schema

`date,AAPL,MSFT,SPY`

### Risk Report Output Schema

- method
- VaR
- ES
- confidence_level
- current_value
- number_of_scenarios

### Backtest Output Schema

- date
- method
- confidence_level
- portfolio_value_t
- portfolio_value_t_plus_1
- actual_pnl
- actual_loss
- VaR
- exception

## Error Handling

The code uses explicit `ValueError` exceptions for invalid inputs:

- missing CSV columns
- invalid instrument types
- missing option parameters
- invalid date formats
- non-numeric fields
- non-positive prices
- unsupported model names
- insufficient data for calibration windows

This makes failure modes easy to identify in grading and testing.

## Reproducibility Choices

- Monte Carlo simulation uses a fixed random seed
- Historical data is loaded from a saved CSV file
- Output file names are deterministic
- The project avoids absolute paths and resolves files relative to the project root

## Extensibility

The design is intentionally open for future work:

- add more instruments in `portfolio.py`
- add Greeks beyond delta in `pricing.py`
- add volatility surface support
- add additional risk measures in `risk_models.py`
- add formal statistical tests in `backtesting.py`
- add richer output formatting or plots in `reporting.py`

