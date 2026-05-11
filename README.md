# Risk Project

## Project Overview

This project implements a modular Python risk calculation system for a portfolio of stocks and European options. It is designed for a course project where software design, documentation quality, model clarity, and testing rigor matter as much as numerical output.

The codebase supports:

- portfolio input validation
- Black-Scholes pricing and delta for European options
- historical calibration from price data
- historical VaR and ES
- parametric delta-normal VaR
- Monte Carlo VaR and ES
- rolling VaR backtesting
- CSV report generation

## Requirements Covered

- Standardized portfolio CSV input format
- Support for both stocks and European options
- Historical calibration from market prices
- User-provided option parameters
- Historical VaR and ES
- Parametric VaR
- Monte Carlo VaR and ES
- VaR backtesting
- Clean modular code structure
- Basic unit and integration tests

## Folder Structure

```text
risk_project/
├── data/
├── docs/
├── outputs/
├── src/
│   ├── input_loader.py
│   ├── market_data.py
│   ├── calibration.py
│   ├── pricing.py
│   ├── portfolio.py
│   ├── risk_models.py
│   ├── backtesting.py
│   ├── reporting.py
│   └── main.py
├── tests/
├── README.md
└── requirements.txt
```

## Input Files

### Portfolio file

`data/sample_portfolio.csv` uses the required schema:

```csv
instrument_id,type,underlying,position,strike,maturity,option_type,implied_vol,risk_free_rate
```

Rules:

- `type` must be `stock` or `option`
- `option_type` must be `call` or `put` for options
- `position` can be positive or negative
- `maturity` must use `YYYY-MM-DD`
- `implied_vol` and `risk_free_rate` are decimal inputs such as `0.25` and `0.04`

### Historical prices file

`data/sample_historical_prices.csv` contains synthetic but realistic daily prices with:

- `date`
- `AAPL`
- `MSFT`
- `SPY`

The file contains more than 300 business days of history so the calibration and backtesting steps can run immediately.

## Output Files

The project writes reports into `outputs/`:

- `outputs/risk_report.csv`
- `outputs/backtest_historical.csv`
- `outputs/backtest_parametric.csv`
- `outputs/backtest_monte_carlo.csv`
- `outputs/backtest_summary.csv`

## Install Dependencies

Install the Python dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Run the Project

Run the main workflow from the project root:

```bash
python src/main.py
```

This:

- loads the sample portfolio
- loads historical prices
- calibrates market data
- runs the three VaR models
- performs backtesting
- saves the output CSV files

You can also run it as a module:

```bash
python -m src.main
```

## Run Tests

Execute the automated test suite:

```bash
pytest -q
```

## Summary of Risk Models

### Historical VaR and ES

- Uses historical log returns from the price history
- Revalues the full portfolio under each historical scenario
- Computes VaR as the loss quantile and ES as the tail mean beyond VaR

### Parametric VaR

- Uses a delta-normal approximation
- Aggregates portfolio delta exposures by underlying
- Uses the calibrated covariance matrix
- Fast and useful for a first-order approximation, but it does not fully capture nonlinear option behavior

### Monte Carlo VaR and ES

- Simulates correlated daily log returns from the calibrated mean vector and covariance matrix
- Revalues the full portfolio under each simulated scenario
- Uses a fixed random seed for reproducibility

## Summary of Backtesting

Backtesting uses rolling one-day-ahead VaR:

- calibrate on data available up to day `t`
- compute the VaR forecast for day `t+1`
- revalue the portfolio at `t` and `t+1`
- compute actual PnL and actual loss
- count an exception when actual loss exceeds the forecast VaR

The project reports:

- the backtest path by day
- a summary of exception counts and exception rates
- a simple pass/fail diagnostic
- a Kupiec unconditional coverage p-value when available

## Known Limitations

- Parametric VaR uses delta-only approximation
- Gamma, vega, volatility smile, and nonlinear tail effects are not fully captured in parametric VaR
- No transaction costs, liquidity constraints, early exercise, or dividend modeling
- Historical methods assume past return dynamics are informative for the future
- Monte Carlo model assumes multivariate normal log returns

## Future Improvements

- Add gamma and vega sensitivity reporting
- Add volatility surface support
- Add dividend-aware option pricing
- Add more formal statistical backtests
- Add richer output formatting and charts
- Add support for more asset classes and larger portfolios
- Add a packaged CLI entry point

