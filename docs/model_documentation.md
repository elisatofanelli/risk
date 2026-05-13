# Model Documentation: Risk Engine and Valuation System

## A. Purpose and Scope
This document outlines the theoretical framework, mathematical models and risk management assumptions underlying the Python-based risk calculation engine. The system is designed to value mixed portfolios of equities and European options and compute Value at Risk (VaR) and Expected Shortfall (ES) using eight distinct methodologies: Historical Simulation, Parametric, and Monte Carlo Simulation, each available in an equally-weighted and an EWMA-weighted (Exponentially Weighted Moving Average) variant, alongside a dedicated framework for custom parameter injection (stress testing). The model operates at a standard 95% confidence level.

## B. Governance and Model Risk Management
In alignment with standard Model Risk Management (MRM) practices and regulatory guidelines (e.g., OCC 2011-12, Basel FRTB), this system incorporates robust software design, continuous validation testing and explicit handling of numerical limitations. Code is strictly segregated between data loading, calibration, pricing and reporting to ensure independent testability and prevent lookahead bias during backtesting. The architecture further supports cross-validation: equally-weighted, EWMA, and user-supplied estimates can be compared directly to identify model sensitivity to weighting schemes, recent volatility regimes, and hypothetical macro shocks.

## C. Methodologies Overview
The engine calculates risk metrics utilizing multiple methodologies to provide a comprehensive view of portfolio risk and to allow for cross-validation:

**Equally-weighted family (calibrated on a 250-day rolling window with uniform weights):**
1. **Historical VaR/ES:** Reprices the current portfolio using historical log-return scenarios, assigning equal probability 1/T to each.
2. **Parametric VaR:** Uses an analytical delta-normal variance-covariance matrix approach.
3. **Monte Carlo VaR/ES:** Simulates 10,000 future market states using a calibrated covariance matrix and reprices the portfolio in each state.

**EWMA-weighted family (same window, geometrically decaying weights, λ = 0.94 by default):**
4. **EWMA Historical VaR/ES:** Identical scenario construction to Historical, but each scenario is weighted by λ^(T-1-t) rather than 1/T.
5. **EWMA Parametric VaR:** Analytical delta-normal approach using an EWMA covariance matrix instead of the equally-weighted sample covariance.
6. **EWMA Monte Carlo VaR/ES:** Monte Carlo simulation parameterised by EWMA mean and covariance estimates.

**User-Supplied Parameter Models (Stress Testing / Forward-Looking Views):**
7. **Custom Parametric VaR:** Delta-normal approach utilizing a manually injected mean vector and covariance matrix.
8. **Custom Monte Carlo VaR/ES:** Simulates 10,000 paths drawn from a user-defined multivariate normal distribution.

The calibration layer is highly modular: `ewma_calibrate()`, `calibrate_from_history()`, and `load_user_params()` all return the identical dictionary schema, allowing downstream risk functions to seamlessly accept varying statistical assumptions.

## D. Data Quality and Sources

**Data Sources and Assessment**
The risk engine ingests data from three distinct formats, each serving a specific role in the valuation and risk generation process:

1. **Historical Market Data (`sample_historical_prices.csv`):** Consists of real-world adjusted closing prices for AAPL, MSFT, and SPY, sourced via Yahoo Finance.
   - **Scope:** The data covers approximately 1,250 trading days (5 years), specifically from January 2021 to December 2025, providing a sufficient sample for both the 250-day rolling calibration and the 1,000-day out-of-sample backtest.
   - **Proxies:** No data proxies were required or utilized for this analysis, as complete historical time series were directly available for all underlying instruments.

2. **Portfolio Holdings (`sample_portfolio.csv`):** A strictly formatted snapshot of the current positions. It defines the composition of the portfolio across asset classes (equities and options).
   - **Scope:** Includes directional exposure (long/short positions), option contract specifications (strike, maturity, call/put) and current market conditions tied to the specific contract (implied volatility, risk-free rate).

3. **User-Supplied Parameters (`user_params_mean.csv`, `user_params_cov.csv`):** Forward-looking, user-defined statistical matrices used exclusively for stress testing.
   - **Scope:** Defines the assumed daily mean return for each underlying asset and the complete N × N covariance matrix representing hypothetical future market regimes.

**Validation Logic**
The system implements automated validation logic (`input_loader.py`) to verify structural integrity upon ingestion:
- **Portfolio Integrity:** The loader strictly enforces schema rules. If an instrument is classified as an option, missing required fields (e.g., strike, maturity, implied volatility) trigger explicit `ValueErrors`. Unsupported instrument types are immediately rejected.
- **Price Data Integrity:** The system actively checks for and rejects non-positive historical prices to prevent fatal mathematical errors or infinite values during log-return calculations. Valuation dates are strictly enforced to prevent lookahead bias. 
- **Parameter Calibration:** The EWMA calibration function validates that the decay parameter satisfies λ ∈ (0, 1). For user-supplied parameters, the engine automatically aligns and reindexes the ingested covariance matrices to match the exact underlying assets present in the current portfolio, ensuring matrix multiplication operations do not fail due to dimension mismatch.

## E. Pricing Engine: Black-Scholes Formula
European options are priced using the standard Black-Scholes-Merton model.
- **Mathematical Limits:** The code mathematically handles the T → 0 expiration boundary by cleanly returning the option's intrinsic value, avoiding division-by-zero errors. Implied volatility or strike inputs of 0 are structurally rejected.
- **Full Repricing vs. Delta-Gamma:** The Historical, EWMA Historical, Monte Carlo and EWMA Monte Carlo engines utilize full Black-Scholes repricing for every scenario rather than relying on Taylor series approximations (Delta/Gamma). Component accuracy tests prove that full repricing correctly captures the positive gamma of long options during severe market shocks, whereas linear approximations systematically overstate losses.

## F. Historical VaR Specifications
- **Confidence Level:** Set at 95%, the industry standard threshold for internal daily risk reporting.
- **Calibration Window:** 250 trading days. This represents one full trading year and strictly aligns with standard Basel regulatory frameworks for VaR calibration.
- **Dynamic Volatility Shock (Leverage Effect):** The historical repricing engine utilizes an inverse elasticity of -0.5. This models the well-documented asymmetric volatility phenomenon (leverage effect) in equity markets, originally observed by Black (1976) and Christie (1982), where negative returns yield proportional increases in implied volatility. This effect correctly buffers the value of long options during market crashes.
- **Weighting:** Each historical scenario receives equal probability 1/T.

## G. EWMA Historical VaR Specifications
- **Confidence Level:** 95% (consistent with all other models).
- **Calibration Window:** 250 trading days — same as the equally-weighted historical model.
- **Scenario Construction:** Identical to the equally-weighted historical method. The same historical log-return scenarios are replayed against the current portfolio with the same dynamic volatility shocks (leverage effect elasticity -0.5).
- **Weighting Scheme:** Instead of uniform weights, each observation at time t receives weight proportional to λ^(T-1-t), where T is the total number of observations and t = 0 is the oldest. Weights are normalised to sum to 1. The most recent day therefore carries approximately (1-λ) times more weight than the prior day.
- **VaR Computation:** The weighted quantile: the smallest loss L such that the cumulative EWMA weight of scenarios with loss ≤ L meets or exceeds the confidence level.
- **ES Computation:** Weighted average of losses that meet or exceed the VaR threshold, using the normalised EWMA tail weights.
- **Decay Factor λ:** Defaults to 0.94 (the RiskMetrics daily standard), giving an effective half-life of approximately 12 trading days. The parameter is configurable; values closer to 1 extend memory while values closer to 0 concentrate weight on very recent observations.
- **Key Advantage over Equally-Weighted Historical:** Responds more rapidly to volatility regime changes. During a sudden market spike, the EWMA variant will elevate VaR within days, whereas the equally-weighted version moves only as stale observations roll out of the 250-day window.

## H. Parametric VaR Specifications
- **Model:** First-order Delta-Normal.
- **Covariance Matrix:** Calibrated using the same 250-day historical window with equal weights. The matrix maps an N × N structure accurately reflecting the number of underlying assets.
- **Expected Shortfall Omission:** Expected Shortfall is intentionally omitted from all Parametric models. Because the Parametric method assumes a strictly Normal distribution, computing the tail average systematically and severely underestimates the true risk of extreme market events (fat tails). ES is instead reliably computed via the Historical and Monte Carlo methodologies.
- **Limitation:** Assumes a linear relationship between asset price changes and option values. While computationally fast, this model ignores convexity (Gamma) and volatility sensitivity (Vega), making it less accurate for highly non-linear portfolios during large shocks.

## I. EWMA Parametric VaR Specifications
- **Model:** First-order Delta-Normal: identical analytical structure to the equally-weighted Parametric model.
- **Covariance Matrix:** Estimated via `ewma_calibrate` using the same 250-day window but with geometrically decaying weights (λ = 0.94). The EWMA covariance is computed as the weighted outer-product sum of demeaned log-returns. No Bessel correction is applied, consistent with standard EWMA estimator practice.
- **VaR Formula:** Uses the same analytical delta-normal expressions as the equally-weighted parametric model; only the input covariance matrix differs. (Expected Shortfall is omitted).
- **Key Advantage:** The EWMA covariance reacts faster to volatility clustering, making the parametric VaR more responsive to recent market conditions without increasing computational cost relative to the equally-weighted version.

## J. Monte Carlo VaR Specifications
- **Simulation Count:** Runs 10,000 paths by default. Numerical convergence testing confirms that VaR estimates stabilize significantly as the simulation count increases from 200 to 10,000.
- **Process:** Generates multivariate normal shocks using a Cholesky decomposition of the calibrated equally-weighted historical covariance matrix.
- **Volatility Dynamics:** Incorporates the same dynamic volatility shock (inverse elasticity of -0.5, Black 1976, Christie 1982) as the Historical model to accurately reprice options in simulated stress events.

## K. EWMA Monte Carlo VaR Specifications
- **Simulation Count:** 10,000 paths by default: identical to the equally-weighted Monte Carlo model.
- **Process:** Identical simulation structure (multivariate normal, Cholesky decomposition, full Black-Scholes repricing, leverage-effect volatility shocks). The sole difference from the equally-weighted Monte Carlo is that the mean vector and covariance matrix fed to the simulation are EWMA estimates from `ewma_calibrate` rather than equally-weighted sample moments.
- **Decay Factor λ:** Defaults to 0.94, consistent with the other EWMA models.
- **Key Advantage:** Simulated paths reflect recent volatility regimes more strongly, making the model more responsive to volatility clustering without requiring a different simulation architecture.

## L. Stress Testing and Expert Parameter Injection
To supplement historical backward-looking calibration, the system supports robust forward-looking stress testing. The engine can dynamically detect and load user-supplied distribution parameters (`user_params_mean.csv` and `user_params_cov.csv`) and compute risk via the Monte Carlo and Parametric models.
- **Purpose:** Historical data inherently lags sudden macroeconomic events (e.g., surprise interest rate hikes). Parameter injection allows risk managers to model hypothetical extreme events by forcing the mathematical engine to evaluate a customized covariance matrix.
- **System Integration:** These customized models are run point-in-time and output to a dedicated report (`risk_report_user_params.csv`). They are intentionally excluded from the historical rolling backtest engine, as applying a static, hypothetical future shock retroactively across 1,000 historical days would invalidate the statistical integrity of the Kupiec test.

## M. Comparison of Modeling Alternatives
As part of the design process, alternative models were considered and evaluated against the chosen implementations:

- **Volatility Modeling (EWMA vs. GARCH):** EWMA weighting was chosen over a static equally-weighted window because it reacts faster to volatility regime shifts while remaining computationally trivial to implement alongside the existing calibration infrastructure. A GARCH(1,1) model would offer a theoretically richer treatment of volatility clustering, with separate persistence and shock parameters, but at the cost of non-linear estimation and additional model risk. The six-model framework presented here exposes the EWMA sensitivity by allowing direct comparison with the equally-weighted counterparts; GARCH remains a natural extension.

- **EWMA Decay Factor (λ = 0.94 vs. alternatives):** The RiskMetrics daily standard of λ = 0.94 (half-life ≈ 12 days) was adopted as the default. A higher λ (e.g., 0.97, half-life ≈ 33 days) would give a smoother, less reactive estimate; a lower λ (e.g., 0.90, half-life ≈ 7 days) would react faster but increase noise. The λ parameter is exposed in all EWMA function signatures to allow sensitivity analysis.

- **Return Distribution (Normal vs. Student's T):** The Monte Carlo and Parametric models (both equally-weighted and EWMA) assume multivariate normal returns. While empirically convenient, replacing this with a Student's T-distribution would better capture the fat tails observed in our data assessment without relying solely on the Historical VaR models.

## N. Outcome Analysis and Validation
The model's outputs are continuously verified through rigorous outcome analysis:

- **P&L Attribution:** Testing proves that the aggregated portfolio P&L perfectly matches the sum of the individual instrument P&Ls, verifying the integrity of the portfolio valuation engine.


- **Cross-Model Comparison:** The parallel equally-weighted and EWMA model families allow direct comparison of VaR estimates and backtest exception rates. A well-behaved EWMA model is expected to show fewer clustered exceptions than its equally-weighted counterpart during periods of sudden volatility, because its covariance (and scenario weights) update faster. Persistent divergence between the two families signals a volatility regime shift and warrants model review.