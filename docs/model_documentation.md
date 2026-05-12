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

## D. Data Validation and Quality Control

The system implements rigorous controls to ensure input data integrity, following standard data management best practices for risk models.

**Price History Validation**
Loading historical data executes a series of automatic checks to prevent risk calculation errors:
* **Temporal Integrity:** Verifies the presence of the 'date' column and converts values into a standard temporal format, halting execution in case of invalid dates.
* **Numeric Data Cleaning:** Checks that all prices are numeric and removes any invalid values (NaNs).
* **Reasonableness Check:** The system ensures that all prices are strictly positive (>0); the presence of zero or negative prices generates an immediate error to avoid undefined logarithms when computing returns.

**Portfolio Validation**
Each instrument in the input file is validated prior to valuation:
* **Required Fields:** Verifies the presence of critical parameters such as position size, strike, implied volatility, and risk-free rates.
* **Instrument Constraints:** The system exclusively accepts 'stock' and 'option' types. For options, it validates that the type is strictly restricted to 'call' or 'put'.
* **Maturity Integrity:** Option maturity dates must be correctly formatted and readable by the system.

## E. Pricing Models

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

## F. Mathematical Specifications

To ensure transparency in model validation, the fundamental mathematical specifications used in the system are detailed below.

**Log Returns**
All calibration calculations are based on daily logarithmic returns:
$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)$$

**Parametric VaR (Delta-Normal)**
Portfolio variance is calculated as:
$$\sigma_p^2 = \delta^T \Sigma \delta$$
Where $\delta$ is the vector of delta exposures (spot price $\times$ option delta) and $\Sigma$ is the covariance matrix of logarithmic returns.

**Black-Scholes Pricing**
The price of a European Call ($C$) and a European Put ($P$) is defined as:
$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$
$$P = K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)$$
Where:
$$d_1 = \frac{\ln(S/K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}$$
$$d_2 = d_1 - \sigma\sqrt{T}$$

## G. Calibration Methodology

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

## H. Historical VaR and ES Methodology

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

## I. Parametric VaR Methodology

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

## J. Monte Carlo VaR and ES Methodology

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

## K. Backtesting Methodology

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
- exception clustering analysis (evaluating consecutive daily exceptions) to ensure the model reacts sufficiently to volatility spikes and does not fail in systematic bursts.

### Simple Pass/Fail Tolerance

At 95% confidence, the expected exception rate is 5%. A simple diagnostic tolerance is used to flag clearly unusual results. This is not a substitute for a full regulatory test.

### Limitations of Small Samples

Backtesting can be noisy when the number of out-of-sample observations is small. Exception rates may vary materially due to sampling noise, especially with short histories or low-confidence data.

## L. Model Limitations and Risk Management Considerations

The system intentionally simplifies several real-world features:

- normality assumption for return distributions
- stationarity assumption in calibration
- correlation instability through time
- implied volatility is held constant for options
- option Greeks beyond delta are not included in parametric VaR
- liquidity, transaction costs, dividends, early exercise, and volatility surface dynamics are not modeled

These limitations should be understood when interpreting results.

## M. Justification of Modeling Choices

The decision to use Black-Scholes for option pricing is the most consequential choice in the system. Black-Scholes is a strong simplification (it assumes constant volatility, no dividends, and lognormal dynamics) but it is the natural starting point for a portfolio of European options, as foundationally established by Black and Scholes (1973) and Merton (1973). Its analytical tractability makes it easy to validate and audit. More importantly, the implied volatility supplied in the portfolio input already encodes the market's view of the option's risk, which is exactly what should be used when pricing a contract as it currently stands in the market. Using historical volatility instead would mean replacing a forward-looking market price with a backward-looking statistical estimate, which conflates two different quantities. This is why the system keeps implied volatility as an instrument-level parameter rather than overwriting it with anything calibrated from the price history.

Log returns are used throughout the calibration and scenario generation steps because they are the natural language of multiplicative price dynamics. As standard industry practice dictates (e.g., Hull, 2018), if a stock follows geometric Brownian motion, its log returns are the quantity that is approximately normal, additive across days, and stable to aggregate. Arithmetic returns do not have these properties and would introduce compounding errors when constructing multi-asset scenarios.

For the VaR models, three methods are implemented rather than one because no single method dominates in all respects. Historical VaR is appealing precisely because it makes no distributional assumption since it uses the actual co-movements observed in the data, which captures correlation breakdowns and fat tails implicitly. Its weakness is that it is entirely backward-looking and depends heavily on the historical window being representative. Monte Carlo VaR addresses this by generating a much larger set of scenarios from the calibrated covariance structure, which reduces sampling noise and allows the full nonlinear option payoff to be captured under many more market conditions. The tradeoff is that it inherits the normality assumption built into the simulation. Parametric VaR is included as a fast analytical benchmark: it is the method most commonly used in practice for a first-pass estimate (Jorion, 2006) and having all three allows the results to be cross-checked against each other. When all three agree closely (as they do here) it provides reassurance that the portfolio's risk profile is not being distorted by the choice of method.

The decision to reprice options fully under each scenario, rather than approximating their value change using delta, is deliberate. A delta approximation treats an option as if it were a linear position in the underlying, which is only accurate for small moves. Historical and Monte Carlo scenarios can include large one-day moves and for those the nonlinear payoff structure of the option matters. Full repricing using Black-Scholes at each scenario spot price is computationally affordable for a portfolio of this size and is meaningfully more accurate, particularly for options that are close to at-the-money where the gamma effect is largest. Finally, to validate the integrity of these risk estimates, the system utilizes the unconditional coverage framework developed by Kupiec (1995), which remains a regulatory standard for backtesting under the Basel frameworks.

## N. Model Validation and Behavioral Checks

To ensure the pricing engine correctly calculates non-linear exposures, the underlying Black-Scholes implementation has been mathematically verified against strict theoretical boundaries. The system confirms that:

- Put-Call Parity holds across a full range of strikes (verifying internal consistency and correct discounting).
- Price Monotonicity is maintained: call prices strictly increase with both spot price and implied volatility, while put prices strictly decrease as spot price rises.
- Delta Boundaries behave as theoretically expected, remaining strictly bounded within [0, 1] for European calls and [-1, 0] for European puts, approaching these limits naturally as options move deep in- or out-of-the-money.

## O. Numerical Robustness and Edge Cases

Following best practices for handling the difficulty of numerical testing, the software includes logic to handle edge cases and prevent computational instability.

**Pricing Stability**
The Black-Scholes calculation routine is protected from numerical failures via:
* **Input Validation:** An internal function verifies that spot price, strike price, and volatility are all positive before proceeding with logarithmic calculations.
* **Time to Maturity Handling (T=0):** To avoid division by zero in the calculation of $d_1$ and $d_2$ when an option expires, the system bypasses the Black-Scholes formula entirely and directly returns the intrinsic value (spot - strike for calls, strike - spot for puts).

**Cluster Analysis in Backtesting**
The backtesting module does not simply count VaR exceptions; it evaluates whether they occur on consecutive days (clustering). This check is crucial for identifying if the model fails to react promptly to structural changes in market volatility.