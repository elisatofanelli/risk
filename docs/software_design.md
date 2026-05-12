# Software Design

## Architecture Overview

The project follows a layered design:

1. Input and validation layer
2. Market data and calibration layer
3. Pricing and portfolio valuation layer
4. Risk model layer
5. Backtesting and reporting layer
6. Command-line entry point

The goal is to keep responsibilities separated so each module is highly cohesive, independently testable, and reusable.

## Module-by-Module Description

### `src/input_loader.py`
- **Purpose:** Loads and validates standardized portfolio CSV files, ensuring all required fields and correct data types are present before valuation begins.
- **Assumptions:** Assumes the input file is a valid CSV. Assumes positions and rates are numeric, options have explicitly defined 'call' or 'put' types, and maturity dates are parsable.
- **Interfaces:** Exports `load_portfolio(path: str) -> pd.DataFrame`.
- **Data Structures:** Reads from a physical CSV format. Returns a pandas `DataFrame` with coerced data types (e.g., datetime for maturities, floats for numeric fields). 
  * *Input Schema:* `instrument_id, type, underlying, position, strike, maturity, option_type, implied_vol, risk_free_rate`.

### `src/market_data.py`
- **Purpose:** Loads historical price data from a CSV, validates temporal formatting, and ensures all price values are strictly positive to prevent logarithmic mathematical errors.
- **Assumptions:** Assumes the CSV contains a 'date' column and one column per underlying ticker. Assumes all prices are strictly positive (no zero or negative prices) and chronological.
- **Interfaces:** Exports `load_price_history(path: str) -> pd.DataFrame`.
- **Data Structures:** Reads from a CSV format (`date, AAPL, MSFT, SPY`). Returns a date-indexed pandas `DataFrame` populated with float values for asset prices.

### `src/calibration.py`
- **Purpose:** Computes log returns from historical price series and estimates key statistical parameters (mean, covariance, volatility, correlation) needed for parametric and Monte Carlo risk models. Produces both daily and annualized metrics.
- **Assumptions:** Assumes the input price data is clean, chronological, strictly positive, and contains no missing values. Assumes stationarity of returns over the calibration window. The annualization factor defaults to 252 trading days.
- **Interfaces:** Exports `compute_log_returns(price_df)` and `calibrate_from_history(price_df, annualization_factor)`.
- **Data Structures:** Takes a date-indexed pandas `DataFrame` as input. Returns a Python `dict` containing pandas `Series` (for vectors like means and volatilities) and pandas `DataFrames` (for 2D matrices like covariance and correlation).

### `src/pricing.py`
- **Purpose:** Implements Black-Scholes analytical formulas to calculate the price and delta of European options (calls and puts), incorporating strict mathematical edge-case validation.
- **Assumptions:** Assumes European exercise only, constant risk-free rate, constant volatility, lognormal asset dynamics, and no dividends. Assumes inputs (spot, strike, vol) are strictly positive.
- **Interfaces:** Exports `black_scholes_price(...)` and `black_scholes_delta(...)`.
- **Data Structures:** Takes standard Python `float` and `str` types as inputs. Returns a single `float` representing the option's theoretical price or mathematical delta.

### `src/portfolio.py`
- **Purpose:** Values single instruments and aggregates them to value full portfolios. Computes delta exposures grouped by underlying asset to support parametric approximations.
- **Assumptions:** Assumes the provided spot prices map correctly to the 'underlying' symbols in the portfolio input. Assumes 'stock' and 'option' are the only valid and supported instrument types.
- **Interfaces:** Exports `value_portfolio(...)`, `value_instrument(...)`, and `portfolio_delta_exposures(...)`.
- **Data Structures:** Takes a pandas `DataFrame` (portfolio) and a Python `dict` (spot prices). Returns a `float` for valuation totals or a `dict[str, float]` mapping underlying tickers to aggregate delta exposures.

### `src/risk_models.py`
- **Purpose:** Implements the core risk calculation engines for Historical VaR/ES, Parametric (Delta-Normal) VaR, and Monte Carlo VaR/ES. Also handles dynamic volatility adjustments during scenario generation.
- **Assumptions:** Historical assumes past returns represent future risks. Parametric assumes delta-linear exposure and normal distribution of returns. Monte Carlo assumes multivariate normal distribution of log returns. The dynamic volatility adjustment assumes a static leverage effect elasticity of -0.5.
- **Interfaces:** Exports `historical_var_es(...)`, `monte_carlo_var_es(...)`, and `parametric_var(...)`. Utilizes internal helper `_shock_portfolio_volatility(portfolio_df, log_returns)` to dynamically adjust option parameters.
- **Data Structures:** Takes a portfolio `DataFrame` alongside either historical prices or a calibration `dict`. Returns a standard `dict` containing computed risk metrics: `{VaR, ES, confidence_level, current_value}`.

### `src/backtesting.py`
- **Purpose:** Runs rolling one-day-ahead backtests to compare predicted VaR against actual realized losses. Evaluates model performance via simple tolerance checks, exception clustering, and unconditional coverage tests.
- **Assumptions:** Assumes a rolling calibration window is valid. Assumes the portfolio positions remain entirely static over the 1-day holding period (no intraday trading).
- **Interfaces:** Exports `backtest_var(...)` and `summarize_backtest(...)`.
- **Data Structures:** `backtest_var` returns a pandas `DataFrame` where each row represents a backtest date with columns: `date, method, actual_loss, VaR, exception`. `summarize_backtest` returns a `dict` of aggregated test statistics.

### `src/reporting.py`
- **Purpose:** Formats the computed risk and backtest results and writes them to standard CSV files for grading, auditing, and visualization.
- **Assumptions:** Assumes the host filesystem allows write access to the designated `outputs/` directory and creates parent directories if they do not exist.
- **Interfaces:** Exports `create_risk_report(...)`, `save_backtest_results(...)`, and `save_backtest_summary(...)`.
- **Data Structures:** Takes native Python dictionaries or lists of dictionaries, converts them to pandas `DataFrames`, and serializes them to physical `.csv` files on disk.

### `src/main.py`
- **Purpose:** Acts as the primary orchestrator. Coordinates data loading, parameter calibration, model execution, backtesting, and reporting in a single sequential workflow.
- **Assumptions:** Assumes all required source files (portfolio CSV, price history CSV) exist at the specified relative paths inside the project structure.
- **Interfaces:** Standard Python `main()` entry point callable via the command line interface.
- **Data Structures:** Manages the flow of `DataFrames` and `dicts` sequentially between the independent sub-modules.

## Data Flow Diagram

```text
portfolio CSV + historical price CSV
        |
        v
input_loader.py + market_data.py
        |
        v
calibration.py ----> calibration_result dict
        |
        +--> risk_models.py ----> VaR / ES results dict
        |
        +--> backtesting.py ----> rolling backtest DataFrame + summary
        |
        v
reporting.py ----> CSV outputs in outputs/

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

