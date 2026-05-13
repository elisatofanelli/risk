# Risk Project

## Project Overview

This project implements a modular Python risk calculation system for a portfolio of stocks and European options. It is designed for a course project where software design, documentation quality, model clarity, and testing rigor matter as much as numerical output.

The codebase supports:

- portfolio input validation
- Black-Scholes pricing, delta, and gamma for European options
- historical calibration from price data (equally-weighted and EWMA)
- historical VaR and ES (equally-weighted and EWMA)
- parametric delta-normal VaR (equally-weighted and EWMA)
- Monte Carlo VaR and ES (equally-weighted and EWMA)
- user-supplied mean and covariance parameter models (Monte Carlo and Parametric)
- rolling VaR backtesting across all six model variants
- visual plot tests for pricing behaviour and EWMA weights
- CSV report generation

## Requirements Covered

- Standardized portfolio CSV input format
- Support for both stocks and European options
- Historical calibration from market prices
- User-provided distribution parameters (mean and covariance CSV)
- Historical VaR and ES — equally-weighted and EWMA
- Parametric VaR — equally-weighted and EWMA
- Monte Carlo VaR and ES — equally-weighted and EWMA
- VaR backtesting across all six model variants
- Clean modular code structure
- Comprehensive unit, integration, robustness, stability, and plot tests

## Folder Structure

```text
risk_project/
├── data/
│   ├── sample_portfolio.csv
│   ├── sample_historical_prices.csv
│   ├── user_params_mean.csv          # optional: user-supplied mean vector
│   └── user_params_cov.csv           # optional: user-supplied covariance matrix
├── docs/
│   ├── model_documentation.md
│   └── software_design.md
├── notebooks/
│   └── plot_tests.ipynb              # visual plot tests
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
│   ├── conftest.py
│   ├── test_pricing.py
│   ├── test_portfolio.py
│   ├── test_risk_models.py
│   └── test_backtesting.py
├── README.md
└── requirements.txt
```

## Input Files

### Portfolio file

`data/sample_portfolio.csv` uses the required schema:

```csv
instrument_id,type,underlying,position,strike,maturity,option_type,implied_vol,risk_free_rate
```

The sample portfolio contains four positions across two underlyings (AAPL and MSFT): two stock positions (one long, one short) and two option positions (one long call, one short put).

Rules:

- `type` must be `stock` or `option`
- `option_type` must be `call` or `put` for options
- `position` can be positive (long) or negative (short)
- `maturity` must use `YYYY-MM-DD`
- `implied_vol` and `risk_free_rate` are decimal inputs such as `0.25` and `0.04`

### Historical prices file

`data/sample_historical_prices.csv` contains real-world adjusted closing prices for three underlyings:

- `date`
- `AAPL`
- `MSFT`
- `SPY`

The file covers approximately 1,250 trading days (January 2021 to December 2025), providing enough history for the 250-day rolling calibration window and a 1,000-observation out-of-sample backtest.

### User-supplied parameters (optional)

If `data/user_params_mean.csv` and `data/user_params_cov.csv` are present, `main.py` will additionally run the Monte Carlo and Parametric models with user-provided mean and covariance estimates, bypassing historical calibration entirely. Results are written to a dedicated `outputs/risk_report_user_params.csv` and also included in the main `outputs/risk_report.csv`. See `src/calibration.py` (`load_user_params`) for the expected CSV schema.

## Documentation

Full model and design documentation is in `docs/`:

- **`docs/model_documentation.md`** — theoretical framework, mathematical specifications, and model risk management rationale for all six VaR methodologies, the Black-Scholes pricing engine, the leverage-effect volatility adjustment, backtesting, and a discussion of modeling alternatives considered.
- **`docs/software_design.md`** — module-by-module architecture description with full mathematical specifications for each module, data flow diagram, error handling strategy, and extensibility notes.

## Output Files

The project writes reports into `outputs/`:

- `outputs/risk_report.csv` — VaR and ES for all six models (plus user-param models if applicable)
- `outputs/risk_report_user_params.csv` — VaR and ES for user-supplied parameter models (written only when user param files are present)
- `outputs/backtest_historical.csv`
- `outputs/backtest_parametric.csv`
- `outputs/backtest_monte_carlo.csv`
- `outputs/backtest_ewma_historical.csv`
- `outputs/backtest_ewma_parametric.csv`
- `outputs/backtest_ewma_monte_carlo.csv`
- `outputs/backtest_summary.csv` — exception rates, Kupiec p-values, and clustering stats for all six models

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

- loads the sample portfolio and historical prices
- calibrates equally-weighted and EWMA market parameters
- runs all six VaR and ES models
- optionally runs user-supplied parameter models if `data/user_params_mean.csv` and `data/user_params_cov.csv` are present
- performs rolling backtesting for all six variants
- saves all output CSV files to `outputs/`

You can also run it as a module:

```bash
python -m src.main
```

## Run Tests

Execute the automated test suite from the project root:

```bash
pytest -v
```

The suite currently contains **63 passing tests** covering pricing, portfolio valuation, risk models (including parameter and output stability tests), and backtesting. See `docs/test_plan.md` and `docs/test_results.md` for the full test inventory and results.

To run the visual plot tests, execute the notebook:

```bash
jupyter nbconvert --to notebook --execute notebooks/plot_tests.ipynb
```

This runs all 10 plot tests with embedded assertions and saves figures to `notebooks/plot_test_*.png`.

## Summary of Risk Models

The engine implements six VaR and ES methodologies across two families.

### Equally-Weighted Family

All three models calibrate on a 250-day rolling window with uniform weights (probability 1/T per observation).

**Historical VaR and ES**
- Replays historical log-return scenarios against the current portfolio using full Black-Scholes repricing.
- Applies a dynamic leverage-effect volatility adjustment (elasticity −0.5): when spot prices fall, implied volatility rises proportionally, increasing long option values and buffering portfolio losses.
- VaR is the empirical loss quantile; ES is the mean of tail losses beyond VaR.

**Parametric VaR**
- Delta-normal approximation: aggregates portfolio delta exposures by underlying and applies the calibrated equally-weighted covariance matrix analytically.
- VaR = z_α × σ_p, where σ_p² = δᵀΣδ and z_α is the standard normal quantile.
- Fast and useful as a first-order estimate; does not capture gamma, vega, or non-linear option behaviour.

**Monte Carlo VaR and ES**
- Simulates 10,000 correlated daily log-return paths from the calibrated mean vector and covariance matrix (multivariate normal, Cholesky decomposition).
- Reprices the full portfolio under each simulated scenario with the same leverage-effect volatility adjustment as the historical model.
- Uses a fixed random seed for reproducibility.

### EWMA-Weighted Family

All three EWMA models use the same 250-day window but weight observations geometrically: `λ^(T-1-t)` (normalised), with λ = 0.94 by default (RiskMetrics daily standard, half-life ≈ 12 days). More recent observations therefore dominate the loss distribution and calibration, making estimates more responsive to current volatility regimes.

**EWMA Historical VaR and ES**
- Identical scenario construction to the equally-weighted historical model.
- VaR is the weighted quantile; ES is the weighted average of tail losses.

**EWMA Parametric VaR**
- Same delta-normal formula as the equally-weighted parametric model.
- The equally-weighted covariance matrix is replaced by an EWMA estimate, making the risk estimate react faster to volatility clustering.

**EWMA Monte Carlo VaR and ES**
- Same simulation structure as the equally-weighted Monte Carlo model.
- Mean vector and covariance matrix are EWMA estimates, so simulated paths reflect recent volatility conditions more strongly.

### Model Ordering

Both families exhibit the same qualitative VaR ordering: Parametric > Monte Carlo > Historical. Parametric is highest because the delta-normal approximation ignores the protective gamma of long options. Monte Carlo captures gamma through full repricing. Historical is the lowest because it additionally captures the empirical leverage-effect volatility buffering. EWMA estimates are lower than their equally-weighted counterparts when recent volatility is below the historical average, and higher when recent volatility has spiked.

## Summary of Risk Output

The system produced the following risk metrics for the mixed portfolio (Current Value: **$4,505.33**) as of the May 2026 valuation date:

| Model | VaR | ES |
|---|---|---|
| Historical | 672.51 | 989.99 |
| Monte Carlo | 700.31 | 889.14 |
| Parametric | 717.05 | 900.22 |
| EWMA Historical | 485.15 | 536.94 |
| EWMA Parametric | 572.21 | 718.76 |
| EWMA Monte Carlo | 562.33 | 703.70 |
| MC (User Params) | 776.22 | 983.82 |
| Parametric (User Params) | 793.94 | 996.95 |

## Summary of Backtesting

Backtesting uses rolling one-day-ahead VaR across all six models:

- calibrate on all data available up to day `t`
- compute the VaR forecast for day `t+1`
- revalue the portfolio at `t` and `t+1`
- compute actual P&L and actual loss
- count an exception when actual loss exceeds the forecast VaR

The project reports:

- the full backtest path by day (one CSV per model)
- a consolidated summary (`backtest_summary.csv`) with exception counts, exception rates, a simple pass/fail diagnostic, Kupiec unconditional coverage p-values, and consecutive exception pair counts for clustering analysis

| Model | Exceptions | Rate | Kupiec p-value | Consecutive Pairs | Verdict |
|---|---|---|---|---|---|
| Historical | 71 | 7.08% | 0.0044 | 14 | **Fails** |
| Parametric | 62 | 6.18% | 0.0972 | 14 | Passes |
| Monte Carlo | 62 | 6.18% | 0.0972 | 13 | Passes |
| EWMA Historical | 60 | 5.98% | 0.1657 | 5 | Passes |
| EWMA Parametric | 51 | 5.08% | 0.9023 | 6 | Passes |
| EWMA Monte Carlo | 53 | 5.28% | 0.6823 | 6 | Passes |

Over the 1,003-observation backtest window, the EWMA variants pass the Kupiec test convincingly (p-values 0.17–0.90) while the equally-weighted Historical model fails (p = 0.0044), with consecutive exception pairs dropping from 13–14 to 5–6 when switching to EWMA weighting. See `docs/test_results.md` for the full backtesting analysis.

## Known Limitations

- Parametric VaR (both equally-weighted and EWMA) uses a delta-only approximation and ignores gamma, vega, and non-linear tail effects
- Monte Carlo and EWMA Monte Carlo assume multivariate normal log returns, which understates fat tails and skewness
- Historical methods assume past return dynamics are informative for the future
- The EWMA decay factor λ = 0.94 is a fixed default; a production system would calibrate it per asset class
- No transaction costs, liquidity constraints, early exercise, or dividend modeling
- Kupiec unconditional coverage is evaluated; Christoffersen independence testing is not implemented

## Future Improvements

- Add gamma and vega sensitivity (Greeks) reporting
- Add volatility surface and smile support
- Add GARCH(1,1) as an alternative to EWMA covariance estimation
- Add Student's T simulation for fatter-tailed Monte Carlo
- Add Christoffersen independence test to the backtesting module
- Add dividend-aware option pricing
- Add a packaged CLI entry point
- Add richer output formatting and interactive charts
