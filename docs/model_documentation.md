# Model Documentation: Risk Engine and Valuation System

## A. Purpose and Scope
This document outlines the theoretical framework, mathematical models, and risk management assumptions underlying the Python-based risk calculation engine. The system is designed to value mixed portfolios of equities and European options and compute Value at Risk (VaR) and Expected Shortfall (ES) using three distinct methodologies: Historical Simulation, Parametric (Delta-Normal), and Monte Carlo Simulation. The model operates at a standard 95% confidence level.

## B. Governance and Model Risk Management
In alignment with standard Model Risk Management (MRM) practices and regulatory guidelines (e.g., OCC 2011-12, Basel FRTB), this system incorporates robust software design, continuous validation testing, and explicit handling of numerical limitations. Code is strictly segregated between data loading, calibration, pricing, and reporting to ensure independent testability and prevent lookahead bias during backtesting.

## C. Methodologies Overview
The engine calculates risk metrics utilizing three separate methodologies to provide a comprehensive view of portfolio risk and to allow for cross-validation:
1.  **Historical VaR:** Reprices the current portfolio using historical log-return scenarios.
2.  **Parametric VaR:** Uses an analytical delta-normal variance-covariance matrix approach.
3.  **Monte Carlo VaR:** Simulates 10,000 future market states using a calibrated covariance matrix and reprices the portfolio in each state.

## D. Data Quality and Sources

**Data Source and Assessment**
The historical dataset (`sample_historical_prices.csv`) consists of real-world adjusted closing prices for AAPL, MSFT, and SPY, sourced via Yahoo Finance.
- **Scope:** The data covers approximately 1,250 trading days (5 years), specifically from **January 2021 to December 2025**, providing a sufficient sample for both the 250-day rolling calibration and the 1,000-day out-of-sample backtest.
- **Statistical Characteristics:** The empirical returns exhibit typical equity market features: negative skewness and a kurtosis higher than the normal distribution (fat tails), which justifies the use of Historical and Monte Carlo simulation over a pure Delta-Normal approach.
- **Outliers:** Significant volatility clusters are present (notably the 2022 bear market), which serve as a rigorous stress test for the models.
- **Proxies:** No data proxies were required or utilized for this analysis, as complete historical time series were directly available for all underlying instruments.

**Validation Logic**
The system implements automated validation logic (`input_loader.py`) to verify structural integrity upon ingestion. Missing values or malformed instrument definitions raise explicit `ValueErrors`. The system actively checks for and rejects non-positive historical prices to prevent fatal mathematical errors during log-return calculations. Validation dates are strictly enforced to prevent lookahead bias.

## E. Pricing Engine: Black-Scholes Formula
European options are priced using the standard Black-Scholes-Merton model.
- **Mathematical Limits:** The code mathematically handles the $T \to 0$ expiration boundary by cleanly returning the option's intrinsic value, avoiding division-by-zero errors. Implied volatility or strike inputs of $0$ are structurally rejected.
- **Full Repricing vs. Delta-Gamma:** The Historical and Monte Carlo engines utilize **full Black-Scholes repricing** for every scenario rather than relying on Taylor series approximations (Delta/Gamma). Component accuracy tests prove that full repricing correctly captures the positive gamma of long options during severe market shocks, whereas linear approximations systematically overstate losses.

## F. Historical VaR Specifications
- **Confidence Level:** Set at 95%, the industry standard threshold for internal daily risk reporting.
- **Calibration Window:** 250 trading days. This represents one full trading year and strictly aligns with standard Basel regulatory frameworks for VaR calibration.
- **Dynamic Volatility Shock (Leverage Effect):** The historical repricing engine utilizes an inverse elasticity of -0.5. This models the well-documented asymmetric volatility phenomenon (leverage effect) in equity markets, originally observed by Black (1976) and Christie (1982), where negative returns yield proportional increases in implied volatility. This effect correctly buffs the value of long options during market crashes.

## G. Parametric VaR Specifications
- **Model:** First-order Delta-Normal.
- **Covariance Matrix:** Calibrated using the same 250-day historical window. The matrix maps an $N \times N$ structure accurately reflecting the number of underlying assets.
- **Limitation:** Assumes a linear relationship between asset price changes and option values. While computationally fast, this model ignores convexity (Gamma) and volatility sensitivity (Vega), making it less accurate for highly non-linear portfolios during large shocks.

## H. Monte Carlo VaR Specifications
- **Simulation Count:** Runs 10,000 paths by default. Numerical convergence testing confirms that VaR estimates stabilize significantly as the simulation count increases from 200 to 10,000.
- **Process:** Generates multivariate normal shocks using a Cholesky decomposition of the calibrated historical covariance matrix. 
- **Volatility Dynamics:** Incorporates the same dynamic volatility shock (inverse elasticity of -0.5, Black 1976, Christie 1982) as the Historical model to accurately reprice options in simulated stress events.

## I. Comparison of Modeling Alternatives
As part of the design process, alternative models were considered and evaluated against the chosen implementations:
- **Volatility Modeling (Static vs. GARCH):** A static 250-day rolling window was chosen for simplicity and computational efficiency. However, a GARCH(1,1) model would react much faster to regime shifts and volatility clustering.
- **Return Distribution (Normal vs. Student's T):** The Monte Carlo and Parametric models assume multivariate normal returns. While empirically convenient, replacing this with a Student's T-distribution would better capture the "fat tails" observed in our data assessment without relying solely on the Historical VaR model.
- **Simulation Method (Filtered Historical Simulation):** Standard Historical Simulation gives equal weight to all days in the 250-day window. FHS (Filtered Historical Simulation) was considered as an alternative to weight recent observations more heavily, but the standard rolling window was maintained for regulatory compliance and simplicity.

## J. Outcome Analysis and Validation
The model's outputs are continuously verified through rigorous outcome analysis:
- **P&L Attribution:** Testing proves that the aggregated portfolio P&L perfectly matches the sum of the individual instrument P&Ls, verifying the integrity of the portfolio valuation engine.
- **Backtesting (Kupiec Test):** The `backtesting.py` module evaluates the unconditional coverage of the VaR models over a rolling window, applying the Kupiec Proportion of Failures (POF) test to ensure the observed exception rate aligns statistically with the 95% confidence level. Exception clustering is actively tracked to monitor model reaction speed to regime shifts.