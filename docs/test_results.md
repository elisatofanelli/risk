# Test Results

This document records the formal results of the automated test suite and the visual plot test notebook, verifying that the risk calculation system complies with the approved Test Plan and standard model risk management practices.

## Run Summary

- **Test command:** `pytest -v`
- **Status:** PASSING
- **Last observed result:** `63 passed in 351.39s` (equally-weighted and EWMA suite, all six model variants, including new parameter stability and output stability tests).
- **Plot tests:** All 10 visual tests in `notebooks/plot_tests.ipynb` pass with embedded assertions. Figures saved to `notebooks/plot_test_*.png`.
- **Notes:** The suite utilises real historical market data (AAPL, MSFT, SPY) to ensure the mathematical defenses and statistical models are validated against empirical market structures rather than synthetic generators.

## Results Table

The following table summarises the test execution mapped directly to the Test Plan.

| Test Category | Test Name / Focus | Expected Result | Actual Result | Interpretation / Status |
| --- | --- | --- | --- | --- |
| **Unit Tests** | Black-Scholes Positivity | Prices > 0 | Passed | Core pricing functions return valid positive floats. |
| **Unit Tests** | Dynamic Volatility Shock | Volatility increases on price drop | Passed | The leverage effect heuristic correctly dynamically adjusts implied volatility. |
| **Unit Tests** | Portfolio Valuation | Returns numeric total | Passed | Portfolio aggregation engine works correctly. |
| **Unit Tests** | EWMA Weight Vector | Sums to 1; most recent > oldest | Passed | For all λ ∈ (0,1) tested, the normalised weight vector sums to 1.0 and the final weight strictly exceeds the first. |
| **Unit Tests** | EWMA Covariance Symmetry | Symmetric, non-negative diagonal | Passed | EWMA covariance matrix is symmetric and all diagonal entries (variances) are non-negative. |
| **Unit Tests** | EWMA Decay Param Validation | Raises `ValueError` for λ ≤ 0 or ≥ 1 | Passed | `ewma_calibrate` raises `ValueError` for boundary and out-of-range lambda values. |
| **Unit Tests** | EWMA Schema Compatibility | Same keys as `calibrate_from_history` + EWMA keys | Passed | Returned dict contains all equally-weighted keys plus `ewma_lambda` and `ewma_weights`; downstream consumers are unaffected. |
| **Special Cases** | Put-Call Parity | Parity mathematically holds | Passed | Pricing formula exhibits correct internal consistency and discounting. |
| **Special Cases** | Time to Maturity (T=0) | Returns exact intrinsic value | Passed | System gracefully bypasses division-by-zero errors at expiration. |
| **Special Cases** | Delta Boundaries | Call ∈ [0,1], Put ∈ [-1,0] | Passed | Option sensitivities respect theoretical limits. |
| **Special Cases** | Volatility/Strike = 0 | Raises `ValueError` | Passed | System correctly rejects physically impossible states. |
| **Special Cases** | EWMA Recency Sensitivity | EWMA VaR > EW VaR after volatility spike | Passed | After appending a high-volatility period to the end of history, all three EWMA VaR variants exceed their equally-weighted counterparts, confirming the recency weighting is active. |
| **Data Testing** | Missing/Invalid Inputs | Raises `ValueError` | Passed | System rejects malformed CSVs and unsupported instrument types. |
| **Data Testing** | Negative Historical Prices | Raises `ValueError` | Passed | Defense mechanism prevents fatal log-return math errors. |
| **Integration** | End-to-End (3 models) | Models run & CSVs generated | Passed | Full equally-weighted workflow executes smoothly from input to report. |
| **Integration** | End-to-End (6 models) | All 6 models run & written to CSV | Passed | All six model variants (equally-weighted and EWMA) execute and are written to a single risk report CSV without error. |
| **Validation** | Covariance Dimensions | Matches asset count (N × N) | Passed | Calibration engine accurately maps multi-asset structures. |
| **Validation** | P&L Attribution | Sum of parts = Total P&L | Passed | Confirms no unexplained residuals or leakages during portfolio aggregation. |
| **Validation** | Gamma Convexity Benefit | Actual loss < Delta approx | Passed | Full repricing properly captures protective positive gamma during crashes. |
| **Validation** | Vega Effect (Leverage) | Adjusted VaR ≤ Unadjusted | Passed | Dynamic volatility shock correctly buffers long option losses. |
| **Validation** | EWMA vs. EW Ordering (calm regime) | EWMA VaR ≈ EW VaR | Passed | In a stationary, low-volatility synthetic return series, EWMA Historical VaR is within the specified tolerance of equally-weighted Historical VaR. |
| **Robustness** | Extreme Market Outliers | System processes without NaNs | Passed | Model survives severe single-day historical crashes without failing. |
| **Robustness** | Short Covariance Window | Matrix generates successfully | Passed | Calibration handles small historical samples gracefully. |
| **Robustness** | Monte Carlo Convergence | Large N stabilizes variance | Passed | Simulation internally converges to the true non-linear portfolio distribution. |
| **Robustness** | Calibration Window Sensitivity | VaR changes dynamically | Passed | Model actively responds to varying historical observation window lengths. |
| **Robustness** | EWMA Lambda Sensitivity | λ=0.80 vs. λ=0.97 produce different VaR | Passed | Running `ewma_historical_var_es` with λ=0.80 and λ=0.97 on identical data produces materially different VaR estimates; the decay parameter is a live, impactful input. |
| **Stability** | Covariance Stability — Input Perturbation (EW) | Max covariance change < 1e-6 under 0.1% price shock | Passed | Equally-weighted covariance matrix is numerically stable; a uniform 0.1% price perturbation causes no material change in any matrix entry. |
| **Stability** | Covariance Stability — Input Perturbation (EWMA) | Max covariance change < 1e-6 under 0.1% price shock | Passed | EWMA covariance matrix is at least as stable as the equally-weighted estimate under a uniform input shock, consistent with the down-weighting of old observations. |
| **Stability** | VaR Output Stability — Input Perturbation (all 6 models) | Relative VaR change < 2% under 0.1% spot shock | Passed | All six model variants produce VaR estimates that move by less than 2% when current spot prices are uniformly shifted by 0.1%, confirming the risk output is not chaotically sensitive to minor data revisions. |
| **Stability** | Covariance Stability Over Rolling Windows (EW & EWMA) | Max swing < 5× median magnitude across 60-day windows (EW); < 10× (EWMA) | Passed | Covariance estimates calibrated on three consecutive non-overlapping 60-day windows do not fluctuate wildly under normal market conditions for either calibration family. |
| **Stability** | VaR Output Stability Over Rolling Windows (Historical & EWMA Historical) | Period-to-period VaR jump < 100% | Passed | Historical VaR and EWMA Historical VaR estimates across three consecutive 60-day windows show no pathological instability (>100% jump), confirming a minimum sanity floor for output continuity over time. |
| **Backtesting** | No Lookahead Bias | Only past data is accessed | Passed | Rolling window correctly segregates out-of-sample data. |
| **Backtesting** | Exception Clustering | Identifies consecutive breaches | Passed | Statistical module accurately tracks sequential VaR failures. |
| **Backtesting** | EWMA Backtest Schema | Correct columns and method label | Passed | Backtests with `ewma_historical`, `ewma_parametric`, and `ewma_monte_carlo` produce DataFrames with the same required columns as equally-weighted counterparts; `method` column is correctly labelled. |
| **Backtesting** | EWMA Exception Clustering Improvement | EWMA pairs < EW pairs | Passed | Across the full 1,003-observation window, all three EWMA variants produce fewer consecutive exception pairs (5–6) than their equally-weighted counterparts (13–14), confirming faster adaptation to the 2022 regime shift. |
| **Backtesting** | VaR Stability | Max daily jump < 10% | Passed | Model outputs remain stable day-to-day; properly mitigates the historical drop-off effect. |
| **Plot Tests** | Call/Put prices vs. spot — monotonicity & convexity | Strict monotonicity; 2nd diff ≥ 0 | Passed | Call strictly increases, put strictly decreases with spot; both curves are convex. |
| **Plot Tests** | Call price vs. strike — monotonicity | Strictly decreasing in K | Passed | 1st-difference derivative is uniformly negative across the strike grid. |
| **Plot Tests** | Call price vs. implied vol — monotonicity | Strictly increasing in σ | Passed | Vega proxy (1st-difference wrt vol) is positive everywhere. |
| **Plot Tests** | 1st-difference derivative vs. analytic delta | Max error < 1e-4 | Passed | Central finite-difference delta matches `black_scholes_delta` for both calls and puts; no pricing noise detected. |
| **Plot Tests** | 2nd-difference derivative vs. analytic gamma | Max error < 1e-3; gamma ≥ 0 | Passed | Finite-difference gamma matches `black_scholes_gamma`; gamma is non-negative everywhere. |
| **Plot Tests** | Option price decay over time (theta, no spikes) | Monotone increase in T; converges to intrinsic | Passed | ATM call price increases monotonically with tenor; price is near zero as T → 0; no spikes visible in the 2nd-difference over the time grid. |
| **Plot Tests** | Pricing noise check on fine spot grid (n=1,000) | Relative 2nd-diff noise < 1e-3 | Passed | Second-order discrete derivatives on a 1,000-point grid are smooth; no pricing noise that would corrupt VaR scenario repricing. |
| **Plot Tests** | Put-Call Parity across strikes (residual ~0) | Max residual < 1e-8 | Passed | Parity holds to machine precision across the full strike range; arbitrage-free pricing confirmed visually and numerically. |
| **Plot Tests** | Forward rates consistency across tenors | ATM call strictly increasing with tenor | Passed | ATM call prices for 0.25Y, 0.5Y, 1Y, 1.5Y, 2Y are in strict ascending order; longer maturities carry higher time value. |
| **Plot Tests** | EWMA weight decay visualisation | Weights sum to 1; most recent highest | Passed | For λ ∈ {0.80, 0.94, 0.97, 0.99}, all weight vectors sum to 1.0 and the most recent observation has the highest weight; log-scale plot confirms qualitatively different decay rates. |

---

## Model Validation Analysis: VaR and ES Output Consistency

During the automated integration run using real market data, the system produced the following risk metrics for the mixed portfolio (Current Value: **$4,505.33**):

| Model | VaR | ES |
|---|---|---|
| Historical | 672.51 | 989.99 |
| Monte Carlo | 700.31 | 889.14 |
| Parametric | 717.05 | 900.22 |
| **EWMA Historical** | **485.15** | **536.94** |
| **EWMA Parametric** | **572.21** | **718.76** |
| **EWMA Monte Carlo** | **562.33** | **703.70** |
| MC (User Params) | 776.22 | 983.82 |
| Parametric (User Params) | 793.94 | 996.95 |

### Interpretation of Equally-Weighted Results

The equally-weighted family exhibits a theoretically sound ordering that validates the modeling choices:

1. **Parametric VaR (717.05) is the highest.** This model relies on a linear Delta-Normal approximation. Because our portfolio contains long options, the Parametric model completely ignores the protective positive Gamma effect, leading to an overestimation of losses during severe market shocks.
2. **Monte Carlo VaR (700.31) is lower.** By performing full Black-Scholes repricing across 10,000 scenarios, the Monte Carlo engine correctly captures the Gamma convexity benefit, which cushions the blow of simulated crashes.
3. **Historical VaR (672.51) is the lowest.** The Historical model not only captures Gamma but also dynamically incorporates the empirical leverage effect. As empirical equity prices drop, volatility spikes; this increases the value of long options, acting as an ultimate buffer against underlying stock losses.

### Interpretation of EWMA Results

All three EWMA variants produce materially lower VaR estimates than their equally-weighted counterparts. This is the expected and correct outcome given the current market data: the 250-day equally-weighted window includes the high-volatility 2022 bear market, while the EWMA weighting (λ = 0.94, half-life ≈ 12 days) concentrates mass on the most recent, lower-volatility period leading up to the May 2026 valuation date. The EWMA estimates therefore better reflect the current volatility regime rather than the crisis period embedded in the tail of the historical window.

The EWMA family preserves the same qualitative ordering as the equally-weighted family:

- **EWMA Parametric (572.21)** is the highest within the EWMA family, again reflecting the absence of Gamma in the delta-normal formula.
- **EWMA Monte Carlo (562.33)** captures Gamma through full repricing and sits below EWMA Parametric.
- **EWMA Historical (485.15)** is the lowest, incorporating both Gamma and the leverage-effect volatility buffering on top of the recency weighting.

The spread between equally-weighted and EWMA VaR — roughly $190–230 across all three model pairs — quantifies the risk premium attributable to including the 2022 regime in the calibration window. This divergence is itself useful model risk information: it signals that the equally-weighted estimates are partially backward-looking, while the EWMA estimates may underestimate tail risk if market conditions deteriorate.

### Interpretation of User-Supplied Parameter Results

When user-supplied mean and covariance parameters are provided via `data/user_params_mean.csv` and `data/user_params_cov.csv`, the system additionally runs the Monte Carlo and Parametric models using those externally calibrated inputs. The user-supplied results (MC: 776.22, Parametric: 793.94) are higher than the historically calibrated counterparts, reflecting a more conservative parameter set in the user-supplied inputs. This cross-check confirms that the system correctly substitutes the user parameters without affecting the main six-model workflow.

---

## Backtesting Analysis

| Model | Observations | Exceptions | Rate | Expected | Kupiec p-value | Consecutive Pairs | Verdict |
|---|---|---|---|---|---|---|---|
| **Historical** | 1,003 | 71 | 7.08% | 5.00% | 0.0044 | 14 | **Fails Kupiec** |
| **Parametric** | 1,003 | 62 | 6.18% | 5.00% | 0.0972 | 14 | **Passes** |
| **Monte Carlo** | 1,003 | 62 | 6.18% | 5.00% | 0.0972 | 13 | **Passes** |
| **EWMA Historical** | 1,003 | 60 | 5.98% | 5.00% | 0.1657 | 5 | **Passes** |
| **EWMA Parametric** | 1,003 | 51 | 5.08% | 5.00% | 0.9023 | 6 | **Passes** |
| **EWMA Monte Carlo** | 1,003 | 53 | 5.28% | 5.00% | 0.6823 | 6 | **Passes** |

### Historical VaR Kupiec Failure — Diagnosis

The equally-weighted Historical model fails the Kupiec unconditional coverage test (p = 0.0044 < 0.05), meaning we can statistically reject the hypothesis that the model correctly covers losses at the 95% level. The 14 consecutive exception pairs confirm significant clustering, concentrated in the 2022 bear market when major indices and tech equities fell 25–35% over approximately nine months.

This is a known structural limitation of rolling-window historical simulation: when a volatility regime shift occurs, the calibration window continues to include the preceding calm period, systematically underestimating the new risk environment until the old data rolls out. A 250-day window takes a full trading year to fully adapt.

### EWMA Models Pass — Key Improvement

All three EWMA variants pass the Kupiec test convincingly. The improvements are material:

- **EWMA Historical** reduces exceptions from 71 to 60 and consecutive pairs from 14 to 5, raising the Kupiec p-value from 0.0044 to 0.1657. The recency weighting allows the calibration to react faster to the 2022 regime shift, elevating VaR earlier during the stress period and thereby avoiding many of the clustered breaches that afflict the equally-weighted historical method.
- **EWMA Parametric** achieves the closest exception rate to the theoretical 5.00% target (5.08%), with a near-perfect Kupiec p-value of 0.9023. The EWMA covariance matrix updates daily and responds to rising volatility significantly faster than the equally-weighted sample covariance, which is anchored to the full 250-day window.
- **EWMA Monte Carlo** closely mirrors EWMA Parametric (5.28%, p = 0.6823), as expected given that both draw from the same EWMA calibration.

The reduction in consecutive exception pairs from 13–14 (equally-weighted) to 5–6 (EWMA) is the clearest empirical evidence that EWMA weighting addresses the primary failure mode identified in this engine: slow reaction to volatility regime shifts.

### Equally-Weighted Parametric and Monte Carlo Pass

Both the equally-weighted Parametric and Monte Carlo methodologies pass the Kupiec test (p = 0.0972 > 0.05). The exception rate of 6.18% is slightly above the 5% target but resides within acceptable diagnostic tolerance. The 13–14 consecutive exception pairs reflect the same 2022 stress cluster; however, because the covariance matrix is updated daily, these models adapted to rising volatility faster than the purely scenario-based equally-weighted Historical method.

### Action Plan / Remediation

The primary remediation for the equally-weighted Historical VaR failure is to replace it with the EWMA Historical variant in production reporting, as the backtest evidence shows a materially superior exception profile. As a secondary option, the equally-weighted calibration window can be dynamically reduced to 125 days, allowing the model to adapt approximately twice as fast — a standard practitioner adjustment documented in Pritsker (2006). The EWMA Parametric and EWMA Monte Carlo models are recommended as the primary VaR estimates going forward given their near-target exception rates and strong Kupiec results.

---

## Stability Test Results

Six new stability tests were added to `tests/test_risk_models.py` following the expanded test plan. These tests apply both input-perturbation and rolling-window stability checks across both calibration families.

**Input Perturbation — Covariance (EW and EWMA):** A uniform 0.1% multiplicative shock applied to all historical prices produces a maximum absolute change of less than 1e-6 in any covariance entry for both the equally-weighted and EWMA calibrators. This confirms numerical stability of the calibration layer and the absence of chaotic sensitivity to minor data revisions.

**Input Perturbation — VaR Output (all six models):** A 0.1% uniform shift in current spot prices causes a relative VaR change of less than 2% across all six model variants. Output stability is a key model validation criterion: if small input perturbations caused large output swings, the risk estimates would be unreliable in practice.

**Rolling Window Stability — Covariance:** Covariance matrices calibrated on three consecutive non-overlapping 60-day windows show no pathological swings: the maximum entry range across windows stays within 5× the median magnitude for the equally-weighted calibrator, and within 10× for the EWMA calibrator (the EWMA tolerance is wider because recency weighting inherently concentrates sensitivity on recent observations, making period-to-period variation slightly larger by design).

**Rolling Window Stability — VaR Output:** Historical VaR and EWMA Historical VaR estimates computed on three consecutive 60-day windows do not jump by more than 100% between adjacent periods under normal market conditions. This establishes a minimum sanity floor for temporal output continuity.

---

## Plot Test Results

Visual tests were executed by running `notebooks/plot_tests.ipynb` end-to-end. Each cell includes both a saved figure and an embedded programmatic assertion; the notebook completed with zero assertion errors. The ten figures are saved as `notebooks/plot_test_1_price_vs_spot.png` through `notebooks/plot_test_10_ewma_weights.png`.

Key findings from the visual inspection:

- **Pricing smoothness confirmed.** The 1st- and 2nd-difference discrete derivatives of option prices along a 1,000-point spot grid show no erratic spikes or discontinuities. The maximum relative 2nd-difference noise is well below the 1e-3 threshold for both calls and puts, confirming that Black-Scholes repricing during VaR scenario generation will not introduce numerical artefacts.
- **Analytic greeks are internally consistent with the pricing function.** The finite-difference delta and gamma match their analytic counterparts to 1e-4 and 1e-3 precision respectively, validating that `black_scholes_price`, `black_scholes_delta`, and `black_scholes_gamma` are mutually consistent.
- **Put-Call Parity holds to machine precision.** The maximum residual across 300 strikes is below 1e-8, confirming the pricing formula is arbitrage-free.
- **Temporal stability verified.** ATM call price increases monotonically from near-zero at expiry to the 2-year value, with no spikes in the discrete time-derivative, confirming correct T-to-expiry calculation throughout the engine.
- **EWMA weights behave as designed.** For all four λ values tested, the weight vector sums to 1.0 and concentrates mass on recent observations. The log-scale plot visually confirms that λ = 0.80 decays far more aggressively than λ = 0.99, validating the practical impact of the decay parameter.

---

## Update Instructions

When new tests are added or the suite changes:
1. Save all source and test files.
2. Run `pytest -v` from the project root and update the Results Table above.
3. Re-execute `notebooks/plot_tests.ipynb` via `jupyter nbconvert --to notebook --execute notebooks/plot_tests.ipynb` and confirm zero assertion errors.
4. Re-run `main.py` and update the VaR/ES table and Backtesting Analysis section with the latest CSV output values.
