# Model Documentation

## A. Model Purpose

The system calculates one-day portfolio VaR and ES for portfolios containing stocks and European options. The architecture is intentionally modular so the same pricing and valuation functions can support historical analysis, parametric approximation, Monte Carlo simulation, and backtesting.

## B. Portfolio Instruments

The portfolio currently supports:

- Stocks
- European calls
- European puts

Each instrument is represented in the standardized portfolio CSV and valued using a common portfolio engine.

## C. Market Data and Inputs

The system uses the following inputs:

- Historical price data for calibration and backtesting
- Portfolio file with positions and instrument parameters
- Implied volatility for each option
- Risk-free rate for each option
- User-provided valuation date and spot prices when needed

The portfolio file is the source of truth for option pricing parameters. Historical price data is used for calibration and scenario generation, not for option implied volatility.

## D. Pricing Models

### Stock Pricing

Stocks are valued at the current spot price multiplied by position size.

### Black-Scholes Option Pricing

European options are priced using the Black-Scholes formula. The implementation supports calls and puts and uses:

- spot price
- strike price
- time to maturity
- implied volatility
- risk-free rate

### Key Assumptions

- European exercise only
- constant implied volatility
- constant risk-free rate
- no dividends unless explicitly modeled elsewhere
- lognormal underlying dynamics

These assumptions keep the model mathematically tractable and appropriate for a course project, but they do not represent every real-market feature.

## E. Calibration Methodology

Calibration is based on historical log returns.

The workflow is:

1. Load historical price data.
2. Compute log returns using consecutive daily prices.
3. Estimate:
   - mean vector
   - covariance matrix
   - correlation matrix
   - annualized versions of mean, covariance, and volatility

Log returns are used because they are additive across time and align naturally with multiplicative price dynamics.

The historical calibration window is the set of observations used up to the current estimation date or rolling backtest date.

## F. Historical VaR and ES Methodology

### Historical Scenario Generation

Historical VaR uses the observed historical log-return vectors as scenarios. For each historical return vector, the system applies the return shock to current spot prices.

### Full Portfolio Repricing

Each scenario is passed through the full valuation engine:

- stocks are repriced directly from the shocked spot
- options are repriced using Black-Scholes with the original implied volatility and risk-free rate from the portfolio input

This preserves the nonlinear option payoff behavior and avoids the common mistake of approximating options as stocks.

### VaR and ES Definitions

- PnL = scenario value - current value
- Loss = -PnL
- VaR = confidence quantile of the loss distribution
- ES = average loss conditional on losses at or beyond VaR

### Strengths and Limitations

Strengths:

- uses actual historical co-movements
- captures nonlinear repricing of options
- straightforward to explain and audit

Limitations:

- depends heavily on historical sample quality
- assumes past return patterns remain informative
- can understate risk if the recent sample is calm or unrepresentative

## G. Parametric VaR Methodology

### Delta-Normal Approximation

Parametric VaR uses a first-order delta-normal approximation. The portfolio is reduced to a vector of delta exposures by underlying, and risk is estimated using the covariance of daily log returns.

### Delta Exposure Aggregation

For each underlying:

- stock positions contribute delta of 1
- option positions contribute Black-Scholes delta

The exposures are aggregated by underlying to produce a portfolio-level sensitivity vector.

### Covariance-Based Portfolio Standard Deviation

Portfolio variance is computed as:

`exposure_vector.T @ covariance_matrix @ exposure_vector`

The standard deviation is the square root of that variance. VaR is then estimated using the normal quantile.

### Strengths and Limitations

Strengths:

- fast
- simple
- useful for first-pass risk estimation
- easy to backtest and compare across portfolios

Limitations:

- gamma, vega, volatility smile, and nonlinear tail effects are not fully captured
- delta-only approximation is weakest for portfolios with large option convexity
- assumes approximately normal risk factor behavior

## H. Monte Carlo VaR and ES Methodology

### Multivariate Normal Simulation

Monte Carlo VaR simulates correlated daily log returns from the calibrated daily mean vector and covariance matrix.

### Correlated Log Returns

Simulated returns preserve cross-asset correlation structure estimated from historical data.

### Full Portfolio Repricing

Each simulated scenario produces a set of shocked spot prices, which are passed through the full valuation engine. Options remain priced with the original implied volatility and risk-free rate from the portfolio file.

### Random Seed for Reproducibility

A fixed random seed is used so repeated runs produce the same simulation path and the same VaR/ES estimates, which is important for grading and regression testing.

### Strengths and Limitations

Strengths:

- captures portfolio nonlinearities better than parametric VaR
- preserves correlation structure
- flexible and extensible

Limitations:

- depends on the multivariate normal assumption
- simulation error remains, especially for small `n_sims`
- computationally more expensive than parametric VaR

## I. Backtesting Methodology

Backtesting uses rolling one-day-ahead VaR.

For each date `t` after the calibration window:

- use prices up to `t` as the calibration/history set
- compute VaR for day `t+1`
- compute portfolio value at `t` and `t+1`
- actual_pnl = value_t_plus_1 - value_t
- actual_loss = -actual_pnl
- exception = actual_loss > VaR

The summary reports:

- number of observations
- number of exceptions
- expected exception rate
- actual exception rate
- a simple pass/fail diagnostic
- a Kupiec unconditional coverage p-value when available

### Simple Pass/Fail Tolerance

At 95% confidence, the expected exception rate is 5%. A simple diagnostic tolerance is used to flag clearly unusual results. This is not a substitute for a full regulatory test.

### Limitations of Small Samples

Backtesting can be noisy when the number of out-of-sample observations is small. Exception rates may vary materially due to sampling noise, especially with short histories or low-confidence data.

## J. Model Limitations and Risk Management Considerations

The system intentionally simplifies several real-world features:

- normality assumption for return distributions
- stationarity assumption in calibration
- correlation instability through time
- implied volatility is held constant for options
- option Greeks beyond delta are not included in parametric VaR
- liquidity, transaction costs, dividends, early exercise, and volatility surface dynamics are not modeled

These limitations should be understood when interpreting results.

## K. Justification of Modeling Choices

### Why options are repriced under scenarios

Options are nonlinear instruments. Scenario-based repricing captures the change in option value due to the shocked underlying price. This is more realistic than treating options as fixed linear exposures.

### Why implied volatility is used instead of historical volatility

The portfolio input is assumed to describe the option contract as currently observed in the market. Implied volatility is the relevant pricing input for the contract. Historical volatility describes the underlying asset’s past behavior and is not a direct substitute for pricing the option itself.

### Why log returns are used for calibration

Log returns are standard in quantitative finance because they are additive over time, easier to aggregate across horizons, and align with the multiplicative nature of asset prices.

### Why parametric VaR is included despite limitations

Parametric VaR is fast, interpretable, and widely used as a baseline approximation. Even though it is less accurate for nonlinear portfolios, it provides a useful benchmark and supports comparison against historical and Monte Carlo approaches.

### Why Monte Carlo and historical VaR complement each other

- Historical VaR is data-driven and grounded in observed scenarios.
- Monte Carlo VaR is model-driven and can generate many scenarios from the estimated covariance structure.

Using both methods improves robustness and gives the grader multiple perspectives on the same risk profile.

