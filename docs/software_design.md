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

---

### `src/market_data.py`
- **Purpose:** Loads historical price data from a CSV, validates temporal formatting, and ensures all price values are strictly positive to prevent logarithmic mathematical errors.
- **Assumptions:** Assumes the CSV contains a 'date' column and one column per underlying ticker. Assumes all prices are strictly positive (no zero or negative prices) and chronological.
- **Interfaces:** Exports `load_price_history(path: str) -> pd.DataFrame`.
- **Data Structures:** Reads from a CSV format (`date, AAPL, MSFT, SPY`). Returns a date-indexed pandas `DataFrame` populated with float values for asset prices.

---

### `src/calibration.py`
- **Purpose:** Computes log returns from historical price series and estimates key statistical parameters (mean, covariance, volatility, correlation) needed for parametric and Monte Carlo risk models. Produces both daily and annualized metrics. Also implements EWMA-based calibration as an alternative to equally-weighted historical estimation.
- **Assumptions:** Assumes the input price data is clean, chronological, strictly positive, and contains no missing values. Assumes stationarity of returns over the calibration window. The annualization factor defaults to 252 trading days. EWMA calibration additionally requires the decay parameter λ ∈ (0, 1).
- **Interfaces:** Exports `compute_log_returns(price_df)`, `calibrate_from_history(price_df, annualization_factor)`, `ewma_calibrate(price_df, lam, annualization_factor)`, and `load_user_params(mean_path, cov_path)`.
- **Data Structures:** Takes a date-indexed pandas `DataFrame` as input. Both `calibrate_from_history` and `ewma_calibrate` return a Python `dict` with the same schema — pandas `Series` for vectors (means, volatilities) and pandas `DataFrames` for matrices (covariance, correlation) — making them drop-in substitutes for all downstream consumers. `ewma_calibrate` additionally includes `ewma_lambda` (the λ used) and `ewma_weights` (a `Series` of per-observation normalised weights indexed by date).

#### Mathematical Specification

**Log Returns**

Daily log returns for asset $i$ at time $t$ are computed as:

$$r_{i,t} = \ln\!\left(\frac{P_{i,t}}{P_{i,t-1}}\right)$$

where $P_{i,t}$ is the closing price of asset $i$ on day $t$.

**Equally-Weighted Calibration**

Given $T$ observations of log returns, the equally-weighted sample mean and covariance are:

$$\hat{\mu}_i = \frac{1}{T}\sum_{t=1}^{T} r_{i,t}$$

$$\hat{\Sigma}_{ij} = \frac{1}{T-1}\sum_{t=1}^{T}(r_{i,t} - \hat{\mu}_i)(r_{j,t} - \hat{\mu}_j)$$

Daily volatility for asset $i$ is $\hat{\sigma}_i = \sqrt{\hat{\Sigma}_{ii}}$. Annualised metrics multiply the daily mean by 252 and daily variance by 252 (so annualised volatility is $\hat{\sigma}_i^{\text{ann}} = \hat{\sigma}_i\sqrt{252}$).

**EWMA Calibration**

The EWMA estimator assigns geometrically decaying weights to observations. For a decay factor $\lambda \in (0,1)$, the unnormalised weight for observation $t$ (where $t=0$ is the oldest and $t=T-1$ is the most recent) is:

$$w_t^{\text{raw}} = \lambda^{T-1-t}$$

Normalised weights that sum to 1 are:

$$w_t = \frac{\lambda^{T-1-t}}{\sum_{s=0}^{T-1}\lambda^{T-1-s}}$$

The EWMA mean vector and covariance matrix are then:

$$\hat{\mu}_i^{\text{EWMA}} = \sum_{t=0}^{T-1} w_t \, r_{i,t}$$

$$\hat{\Sigma}_{ij}^{\text{EWMA}} = \sum_{t=0}^{T-1} w_t \,(r_{i,t} - \hat{\mu}_i^{\text{EWMA}})(r_{j,t} - \hat{\mu}_j^{\text{EWMA}})$$

No Bessel correction is applied, consistent with standard EWMA estimator practice. With $\lambda = 0.94$ (the RiskMetrics daily standard), the effective half-life is approximately $\ln(2)/\ln(1/\lambda) \approx 12$ trading days.

---

### `src/pricing.py`
- **Purpose:** Implements Black-Scholes analytical formulas to calculate the price and delta of European options (calls and puts), incorporating strict mathematical edge-case validation.
- **Assumptions:** Assumes European exercise only, constant risk-free rate, constant volatility, lognormal asset dynamics, and no dividends. Assumes inputs (spot, strike, vol) are strictly positive.
- **Interfaces:** Exports `black_scholes_price(...)`, `black_scholes_delta(...)`, and `black_scholes_gamma(...)`.
- **Data Structures:** Takes standard Python `float` and `str` types as inputs. Returns a single `float` representing the option's theoretical price, delta, or gamma.

#### Mathematical Specification

**Black-Scholes Price**

Let $S$ be the current spot price, $K$ the strike, $T$ the time to maturity (in years), $\sigma$ the implied volatility, and $r$ the continuously compounded risk-free rate. Define:

$$d_1 = \frac{\ln(S/K) + (r + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

The call and put prices are:

$$C = S\,\Phi(d_1) - K e^{-rT}\Phi(d_2)$$
$$P = K e^{-rT}\Phi(-d_2) - S\,\Phi(-d_1)$$

where $\Phi(\cdot)$ is the standard normal CDF. When $T \to 0$ the formula degenerates to the intrinsic value ($\max(S-K,0)$ for calls) to avoid division by zero.

**Delta**

$$\Delta_{\text{call}} = \Phi(d_1), \qquad \Delta_{\text{put}} = \Phi(d_1) - 1$$

**Gamma**

$$\Gamma = \frac{\phi(d_1)}{S\,\sigma\sqrt{T}}$$

where $\phi(\cdot)$ is the standard normal PDF. Gamma is identical for calls and puts and is always non-negative, reflecting the convexity of option payoffs.

**Put-Call Parity**

The model satisfies $C - P = S - Ke^{-rT}$ exactly (arbitrage-free pricing), verified to machine precision (residual < 1e-8) across all strikes in the plot test suite.

---

### `src/portfolio.py`
- **Purpose:** Values single instruments and aggregates them to value full portfolios. Computes delta exposures grouped by underlying asset to support parametric approximations.
- **Assumptions:** Assumes the provided spot prices map correctly to the 'underlying' symbols in the portfolio input. Assumes 'stock' and 'option' are the only valid and supported instrument types.
- **Interfaces:** Exports `value_portfolio(...)`, `value_instrument(...)`, and `portfolio_delta_exposures(...)`.
- **Data Structures:** Takes a pandas `DataFrame` (portfolio) and a Python `dict` (spot prices). Returns a `float` for valuation totals or a `dict[str, float]` mapping underlying tickers to aggregate delta exposures.

#### Mathematical Specification

**Instrument Valuation**

For a stock position with $n$ shares and current spot price $S$:

$$V_{\text{stock}} = n \cdot S$$

For an option position with $n$ contracts priced at $C$ (or $P$ for a put):

$$V_{\text{option}} = n \cdot \text{BS\_price}(S, K, T, \sigma, r, \text{type})$$

**Portfolio Value**

$$V_{\text{portfolio}} = \sum_{i=1}^{N} V_i$$

**Delta Exposure by Underlying**

The aggregate delta exposure for underlying $u$ is the dollar-delta — the sum of position-weighted Black-Scholes deltas across all instruments referencing $u$:

$$\Delta_u = \sum_{i:\, \text{underlying}_i = u} n_i \cdot \Delta_i(S_u)$$

where $n_i$ is the signed position size and $\Delta_i$ is the instrument's per-unit delta. For stocks, $\Delta_i = 1$. These dollar-delta exposures are the inputs to the Parametric and EWMA Parametric VaR computations.

---

### `src/risk_models.py`
- **Purpose:** Implements six risk calculation engines across two families: equally-weighted (Historical, Parametric, Monte Carlo) and EWMA-weighted (EWMA Historical, EWMA Parametric, EWMA Monte Carlo) VaR and ES. Also handles dynamic volatility adjustments during scenario generation.
- **Assumptions:** Historical and EWMA Historical assume past returns represent future risks. Parametric and EWMA Parametric assume delta-linear exposure and normal distribution of returns. Monte Carlo and EWMA Monte Carlo assume multivariate normal distribution of log returns. All scenario-based methods apply the same dynamic volatility adjustment (leverage effect elasticity of -0.5). EWMA methods additionally assume that a geometrically decaying weighting scheme (λ = 0.94 by default) better reflects current market conditions than an equally-weighted window.
- **Interfaces:** Exports `historical_var_es(...)`, `monte_carlo_var_es(...)`, `parametric_var(...)`, `ewma_historical_var_es(...)`, `ewma_parametric_var(...)`, and `ewma_monte_carlo_var_es(...)`. Utilizes internal helpers `_shock_portfolio_volatility(...)` and `_weighted_loss_summary(...)`.
- **Data Structures:** All six functions return a standard `dict` containing `{VaR, ES, confidence_level, current_value, number_of_scenarios}`. EWMA variants additionally include `ewma_lambda` for reporting traceability.

#### Mathematical Specification

**Dynamic Volatility Shock (Leverage Effect)**

Based on Black (1976) and Christie (1982), when a scenario applies log-return shock $r$ to an underlying, the implied volatility of options on that underlying is updated as:

$$\sigma_{\text{shocked}} = \sigma_0 \cdot e^{-0.5 \cdot r}$$

A negative return ($r < 0$) increases $\sigma$, raising option values and buffering portfolio losses during market crashes. This captures the well-documented inverse relationship between equity prices and volatility.

**Historical VaR and ES**

For $T$ equally-weighted historical scenarios, each portfolio is repriced under the realised log-return vector $\mathbf{r}_t$ to obtain a loss $L_t = V_0 - V_t$. Sorting losses in ascending order:

$$\text{VaR}_\alpha = L_{(\lceil \alpha T \rceil)}$$

$$\text{ES}_\alpha = \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}} L_t, \quad \mathcal{T} = \{t : L_t \geq \text{VaR}_\alpha\}$$

where $\alpha$ is the confidence level (0.95) and $L_{(k)}$ denotes the $k$-th order statistic.

**Parametric (Delta-Normal) VaR**

The portfolio dollar-delta vector $\boldsymbol{\delta} \in \mathbb{R}^N$ (one entry per underlying) is assembled from `portfolio_delta_exposures`. The one-day portfolio variance and mean are:

$$\sigma_p^2 = \boldsymbol{\delta}^\top \hat{\Sigma} \, \boldsymbol{\delta}, \qquad \mu_p = \boldsymbol{\delta}^\top \hat{\boldsymbol{\mu}}$$

where $\hat{\Sigma}$ is the $N \times N$ daily covariance matrix and $\hat{\boldsymbol{\mu}}$ is the daily mean return vector. VaR at confidence level $\alpha$ under the normal assumption is:

$$\text{VaR}_\alpha = z_\alpha \, \sigma_p - \mu_p$$

where $z_\alpha = \Phi^{-1}(\alpha)$ is the $\alpha$-quantile of the standard normal distribution. 
*Note: Parametric Expected Shortfall is intentionally omitted. As discussed in Lecture 5, assuming a Normal distribution severely underestimates extreme tail risks (fat tails), making a purely parametric ES a misleading metric compared to Historical or Monte Carlo ES.*

**Monte Carlo VaR and ES**

$M = 10{,}000$ correlated log-return vectors are drawn from a multivariate normal distribution:

$$\mathbf{r}^{(m)} \sim \mathcal{N}\!\left(\hat{\boldsymbol{\mu}},\, \hat{\Sigma}\right)$$

using a Cholesky decomposition $\hat{\Sigma} = \mathbf{L}\mathbf{L}^\top$, so that $\mathbf{r}^{(m)} = \hat{\boldsymbol{\mu}} + \mathbf{L}\mathbf{z}^{(m)}$ with $\mathbf{z}^{(m)} \overset{iid}{\sim} \mathcal{N}(\mathbf{0}, \mathbf{I})$. Each simulated return vector is applied to the portfolio with the dynamic volatility shock, yielding loss $L^{(m)}$. VaR and ES are then the empirical quantile and conditional tail mean of the $\{L^{(m)}\}$ distribution, exactly as in the historical method.

**EWMA Historical VaR and ES**

The same $T$ historical scenarios are used as in the equally-weighted historical method. The weight assigned to scenario at time $t$ is:

$$w_t = \frac{\lambda^{T-1-t}}{\sum_{s=0}^{T-1}\lambda^{T-1-s}}$$

The weighted VaR is the smallest loss $L^*$ such that the cumulative EWMA weight of scenarios with loss ≤ $L^*$ meets or exceeds $\alpha$:

$$\text{VaR}_\alpha^{\text{EWMA}} = \inf\!\left\{L : \sum_{t:\, L_t \leq L} w_t \geq \alpha \right\}$$

The EWMA ES is the weighted average of tail losses:

$$\text{ES}_\alpha^{\text{EWMA}} = \frac{\sum_{t:\, L_t \geq \text{VaR}_\alpha^{\text{EWMA}}} w_t L_t}{\sum_{t:\, L_t \geq \text{VaR}_\alpha^{\text{EWMA}}} w_t}$$

This is implemented in the internal helper `_weighted_loss_summary`.

**EWMA Parametric VaR**

Identical analytical formula to the equally-weighted Parametric model; the only substitution is replacing $\hat{\Sigma}$ with the EWMA covariance matrix $\hat{\Sigma}^{\text{EWMA}}$ from `ewma_calibrate`:

$$\text{VaR}_\alpha^{\text{EWMA-P}} = z_\alpha\sqrt{\boldsymbol{\delta}^\top \hat{\Sigma}^{\text{EWMA}} \boldsymbol{\delta}}$$

**EWMA Monte Carlo VaR**

Identical simulation architecture to the equally-weighted Monte Carlo; the mean vector and covariance matrix fed to the Cholesky decomposition are EWMA estimates $(\hat{\boldsymbol{\mu}}^{\text{EWMA}}, \hat{\Sigma}^{\text{EWMA}})$.

---

### `src/backtesting.py`
- **Purpose:** Runs rolling one-day-ahead backtests to compare predicted VaR against actual realized losses. Evaluates model performance via simple tolerance checks, exception clustering, and unconditional coverage tests. Supports all six model variants.
- **Assumptions:** Assumes a rolling calibration window is valid. Assumes the portfolio positions remain entirely static over the 1-day holding period (no intraday trading). For Monte Carlo and EWMA Monte Carlo backtests, a fixed random seed ensures reproducibility while limiting observation of simulation variance across time.
- **Interfaces:** Exports `backtest_var(...)` and `summarize_backtest(...)`. The `method` argument to `backtest_var` now accepts six values: `"historical"`, `"parametric"`, `"monte_carlo"`, `"ewma_historical"`, `"ewma_parametric"`, and `"ewma_monte_carlo"`.
- **Data Structures:** `backtest_var` returns a pandas `DataFrame` where each row represents a backtest date with columns: `date, method, actual_loss, VaR, exception`. `summarize_backtest` returns a `dict` of aggregated test statistics.

#### Mathematical Specification

**Rolling Backtest Procedure**

For each date $t$ in the out-of-sample period, the engine:
1. Calibrates on the $W = 250$ observations $\{t-W, \ldots, t-1\}$ (no lookahead).
2. Computes $\widehat{\text{VaR}}_t$ using the chosen model.
3. Observes the actual one-day P&L: $\text{PnL}_t = V_t - V_{t-1}$.
4. Records an exception when the actual loss exceeds the VaR forecast:

$$\mathbb{1}_t = \mathbf{1}\!\left[-\text{PnL}_t > \widehat{\text{VaR}}_t\right]$$

**Kupiec Unconditional Coverage Test**

Let $n$ be the number of exceptions in $T$ observations and $p = 1 - \alpha$ the expected exception probability (0.05 at 95% confidence). The Kupiec log-likelihood ratio statistic is:

$$\text{LR}_{\text{POF}} = -2\ln\!\left[\frac{p^n(1-p)^{T-n}}{\hat{p}^n(1-\hat{p})^{T-n}}\right]$$

where $\hat{p} = n/T$ is the observed exception rate. Under the null hypothesis of correct coverage, $\text{LR}_{\text{POF}} \overset{d}{\to} \chi^2(1)$, and the p-value is computed accordingly.

**Consecutive Exception Pairs (Clustering)**

The number of consecutive exception pairs counts how many times an exception on day $t$ is immediately followed by an exception on day $t+1$. High counts signal exception clustering, which is evidence against the independence of exceptions — a necessary condition for a well-specified VaR model under Christoffersen (1998) conditional coverage.

---

### `src/reporting.py`
- **Purpose:** Formats the computed risk and backtest results and writes them to standard CSV files for grading, auditing, and visualization.
- **Assumptions:** Assumes the host filesystem allows write access to the designated `outputs/` directory and creates parent directories if they do not exist.
- **Interfaces:** Exports `create_risk_report(...)`, `save_backtest_results(...)`, and `save_backtest_summary(...)`.
- **Data Structures:** Takes native Python dictionaries or lists of dictionaries, converts them to pandas `DataFrames`, and serializes them to physical `.csv` files on disk.

---

### `src/main.py`
- **Purpose:** Acts as the primary orchestrator. Coordinates data loading, parameter calibration, execution of all six risk models, optional user-supplied parameter models, backtesting, and reporting in a single sequential workflow.
- **Assumptions:** Assumes all required source files (portfolio CSV, price history CSV) exist at the specified relative paths inside the project structure. User-supplied parameter files (`user_params_mean.csv`, `user_params_cov.csv`) are optional; their presence is detected at runtime.
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
calibration.py ──────────────────────────────┐
  calibrate_from_history() → calibration dict │
  ewma_calibrate()         → ewma calib dict  │
        │                                     │
        ├──► risk_models.py                   │
        │      historical_var_es()            │
        │      parametric_var()               │
        │      monte_carlo_var_es()           │
        │      ewma_historical_var_es()       │
        │      ewma_parametric_var()          │
        │      ewma_monte_carlo_var_es()      │
        │         └──► VaR / ES results dict  │
        │                                     │
        └──► backtesting.py ←────────────────┘
               (all 6 methods)
               └──► rolling backtest DataFrame + summary
        |
        v
reporting.py ──► CSV outputs in outputs/
                  risk_report.csv
                  risk_report_user_params.csv     (optional)
                  backtest_historical.csv
                  backtest_parametric.csv
                  backtest_monte_carlo.csv
                  backtest_ewma_historical.csv
                  backtest_ewma_parametric.csv
                  backtest_ewma_monte_carlo.csv
                  backtest_summary.csv