# Test Plan

## Purpose

Verify that the risk project computes prices, calibrations, VaR estimates, and backtests correctly using modular, reusable code, in accordance with standard model risk management practices.

## Scope

This test plan covers:

- Data quality and CSV input validation
- Black-Scholes pricing and delta behaviors
- Portfolio valuation and aggregation
- Historical calibration (equally-weighted and EWMA)
- Historical, Monte Carlo, and Parametric VaR — equally-weighted and EWMA variants
- Rolling VaR backtesting across all six models and clustering analysis
- Visual / plotting tests for pricing behaviour and EWMA weights
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
- Calibrate statistical parameters (mean, covariance) from history and successfully run all three equally-weighted risk models (Historical, Monte Carlo, Parametric). [`tests/test_risk_models.py`]
- Write the final computed risk metrics smoothly to the `outputs/` directory. [`tests/test_risk_models.py`]

## Model Validation Tests

- **Covariance Dimensions:** Covariance matrix dimensions precisely match the number of assets in the underlying price history. [`tests/test_risk_models.py`]
- **Loss Positivity:** VaR outputs strictly produce non-negative loss values across the three equally-weighted model variants. [`tests/test_risk_models.py`]
- **P&L Attribution:** Confirms that the aggregate portfolio P&L equals the exact sum of individual stock and option P&L components under a price shock, ensuring no leakages in aggregation. [`tests/test_portfolio.py`]
- **Vega Effect (Volatility Buffering):** Confirms that the dynamic leverage effect (rising volatility during market drops) correctly increases the value of long option positions, acting as a buffer that reduces overall portfolio VaR compared to an unadjusted model. [`tests/test_risk_models.py`]

## Robustness Tests

- **Extreme Scenarios:** Evaluate model behavior when subjected to exceptionally large historical outliers (e.g., extreme single-day market crashes) to ensure the system does not fail on NaN standard deviations. [`tests/test_risk_models.py`]
- **Covariance Stability:** Confirm the calibration engine gracefully handles short historical windows without crashing when generating matrices. [`tests/test_risk_models.py`]
- **Monte Carlo Convergence:** Verify that Monte Carlo VaR estimates stabilise as `n_sims` increases from 100 to 10,000, confirming the mathematical consistency of the simulation engine. [`tests/test_risk_models.py`]
- **Calibration Window Sensitivity:** Evaluates how the VaR estimate responds to different historical calibration window sizes (e.g., 250 days vs 60 days), ensuring the model dynamically adapts to recent market volatility rather than remaining static. [`tests/test_risk_models.py`]

## Stability Tests

- **Covariance Stability — Input Perturbation (EW):** A uniform 0.1% multiplicative price shock must produce a maximum absolute change of less than 1e-6 in any entry of the equally-weighted covariance matrix. [`tests/test_risk_models.py`]
- **Covariance Stability — Input Perturbation (EWMA):** The same 0.1% shock must produce a maximum absolute change of less than 1e-6 in any entry of the EWMA covariance matrix. [`tests/test_risk_models.py`]
- **VaR Output Stability — Input Perturbation (all 6 models):** A 0.1% uniform shift in current spot prices must cause a relative VaR change of less than 2% across all six model variants (Historical, Parametric, Monte Carlo, EWMA Historical, EWMA Parametric, EWMA Monte Carlo). [`tests/test_risk_models.py`]
- **Covariance Stability Over Rolling Windows (EW & EWMA):** Covariance matrices calibrated on three consecutive non-overlapping 60-day windows must not exceed 5× the median magnitude for the equally-weighted calibrator, or 10× for the EWMA calibrator. [`tests/test_risk_models.py`]
- **VaR Output Stability Over Rolling Windows (Historical & EWMA Historical):** Period-to-period VaR jumps across three consecutive 60-day windows must remain below 100%. [`tests/test_risk_models.py`]

## Backtesting Tests

- Rolling backtest effectively generates the required schema and columns for daily evaluation across all six model methods. [`tests/test_backtesting.py`]
- Exception flags strictly return boolean structures. [`tests/test_backtesting.py`]
- Actual exception rate is bounded reasonably between 0 and 1. [`tests/test_backtesting.py`]
- No lookahead bias is introduced; the engine is strictly prohibited from accessing data beyond the rolling calibration window. [`tests/test_backtesting.py`]
- The summary function effectively captures unconditional coverage metrics and executes a clustering analysis to detect consecutive daily exceptions. [`tests/test_backtesting.py`]
- Summary output is automatically created and persisted to disk. [`tests/test_backtesting.py`]

## Plot Tests

Visual tests are implemented in `notebooks/plot_tests.ipynb`. They confirm that model implications hold graphically — as recommended in the model validation lecture — and provide a second, independent check on behaviours that automated assertion tests may miss (e.g. subtle pricing noise visible only in second-order discrete derivatives).

Each plot test generates a saved figure and includes an embedded programmatic assertion so the notebook can be run end-to-end and any failure is caught immediately.

- **Call/Put prices vs. spot — monotonicity and convexity:** Plots prices, 1st-difference derivatives (≈ delta), and 2nd-difference derivatives (≈ gamma) along a fine spot grid. Confirms call increases and put decreases with spot, and both curves are convex. [`notebooks/plot_tests.ipynb`, Test 1]
- **Call price vs. strike — monotonicity:** Plots call price and its 1st-difference derivative as a function of strike. Confirms call price is strictly decreasing in strike (higher hurdle → lower value). [`notebooks/plot_tests.ipynb`, Test 2]
- **Call price vs. implied volatility — monotonicity:** Plots call price and vega proxy across the vol range. Confirms call price is strictly increasing in implied volatility. [`notebooks/plot_tests.ipynb`, Test 3]
- **1st-difference derivative vs. analytic delta:** Central finite-difference `(C(S+h) − C(S−h)) / 2h` is overlaid on `black_scholes_delta` for both calls and puts. Maximum absolute error must be below 1e-4. Pricing noise would appear as erratic divergence from the analytic curve. [`notebooks/plot_tests.ipynb`, Test 4]
- **2nd-difference derivative vs. analytic gamma:** Finite-difference `(C(S+h) − 2C(S) + C(S−h)) / h²` is overlaid on `black_scholes_gamma`. Maximum absolute error must be below 1e-3; gamma must be non-negative everywhere. Spikes in the residual indicate pricing noise. [`notebooks/plot_tests.ipynb`, Test 5]
- **Option price decay over time (theta, no spikes):** ATM call and put prices are plotted against a dense time-to-maturity grid from near-zero to 2 years. The 1st- and 2nd-difference derivatives over time are plotted to detect any discontinuities. Price must be monotonically increasing with T and must converge to intrinsic value at expiry. [`notebooks/plot_tests.ipynb`, Test 6]
- **Pricing noise check on fine spot grid:** Prices, 1st-differences, and 2nd-differences are plotted for n=1,000 spot points. The maximum relative 2nd-difference must remain below 1e-3, confirming the formula is smooth enough not to introduce noise into VaR scenario repricing. [`notebooks/plot_tests.ipynb`, Test 7]
- **Put-Call Parity across strikes (residual ~0):** The empirical C − P is overlaid on the theoretical S − K·e^{−rT} curve, and the residual is plotted separately. Maximum absolute residual must be below 1e-8 — machine-precision confirmation of arbitrage-free pricing. [`notebooks/plot_tests.ipynb`, Test 8]
- **Forward rates consistency across tenors:** Call price curves for tenors of 0.25Y, 0.5Y, 1Y, 1.5Y, and 2Y are plotted on the same axes. ATM call prices are extracted and plotted as a bar chart to confirm strict monotone increase with tenor. [`notebooks/plot_tests.ipynb`, Test 9]
- **EWMA weight decay visualisation:** EWMA weight vectors for λ ∈ {0.80, 0.94, 0.97, 0.99} are plotted on linear and log scales. Confirms weights sum to 1, most recent observation carries the highest weight, and smaller λ produces faster decay. [`notebooks/plot_tests.ipynb`, Test 10]

## Known Limitations

- Parametric VaR (both equally-weighted and EWMA) relies strictly on a first-order delta-normal approximation, ignoring gamma, vega, and higher-order convex option risks. Expected Shortfall is not computed for Parametric variants.
- The simple backtest pass/fail rule serves as a quick diagnostic tolerance check, not a fully comprehensive regulatory validation.
- Kupiec unconditional coverage is evaluated, but more complex independence testing (like Christoffersen's) is not implemented for the scope of this engine.

