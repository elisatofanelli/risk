"""Historical calibration from price data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from a price history DataFrame."""

    log_returns = np.log(price_df / price_df.shift(1))
    return log_returns.dropna(how="any")


def calibrate_from_history(price_df: pd.DataFrame, annualization_factor: int = 252) -> dict:
    """Calibrate mean, covariance, volatility and correlation from historical log returns."""

    log_returns = compute_log_returns(price_df).dropna(how="any")

    daily_mean_returns = log_returns.mean()
    daily_cov_matrix = log_returns.cov()
    daily_volatility = log_returns.std()
    correlation_matrix = log_returns.corr()

    annualized_mean_returns = daily_mean_returns * annualization_factor
    annualized_cov_matrix = daily_cov_matrix * annualization_factor
    annualized_volatility = daily_volatility * (annualization_factor ** 0.5)

    return {
        "daily_mean_returns": daily_mean_returns,
        "daily_cov_matrix": daily_cov_matrix,
        "daily_volatility": daily_volatility,
        "annualized_mean_returns": annualized_mean_returns,
        "annualized_cov_matrix": annualized_cov_matrix,
        "annualized_volatility": annualized_volatility,
        "correlation_matrix": correlation_matrix,
    }

def ewma_calibrate(price_df: pd.DataFrame, lam: float = 0.94, annualization_factor: int = 252) -> dict:
    """
    Calibrate mean and covariance using an Exponentially Weighted Moving Average (EWMA).

    Weights decay geometrically: w_t ∝ λ^(T-1-t), with the most recent observation
    carrying the highest weight.  The decay factor λ controls how quickly
    older data loses influence:
      - λ = 0.94  is the RiskMetrics daily standard (strong recency bias)
      - λ = 0.97  gives a longer effective memory (~33 trading days half-life)
      - λ = 0.99  approximates a ~69-day half-life, closer to equal-weighting

    The EWMA covariance is computed as the weighted outer-product sum of demeaned
    log-returns (bias=False correction is NOT applied; EWMA is inherently a biased
    estimator and debiasing would distort the recency weighting).

    Parameters
    ----------
    price_df : pd.DataFrame
        Price history; columns are underlying tickers, rows are dates (sorted ascending).
    lam : float
        Decay factor λ ∈ (0, 1).  Defaults to 0.94 (RiskMetrics standard).
    annualization_factor : int
        Number of trading days per year used to annualise estimates.  Defaults to 252.

    Returns
    -------
    dict
        Same schema as ``calibrate_from_history``, plus:
        - ``ewma_lambda``: the λ used
        - ``ewma_weights``: pd.Series of normalised per-observation weights (index = dates)
    """
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lam must be in (0, 1), got {lam}")

    log_returns = compute_log_returns(price_df).dropna(how="any")
    if log_returns.empty:
        raise ValueError("Not enough observations to compute EWMA calibration.")

    n = len(log_returns)
  
    raw_weights = np.array([lam ** (n - 1 - i) for i in range(n)])
    weights = raw_weights / raw_weights.sum() 

    daily_mean_returns = pd.Series(
        np.average(log_returns.values, axis=0, weights=weights),
        index=log_returns.columns,
    )

    demeaned = log_returns.values - daily_mean_returns.values[np.newaxis, :]
    weighted_demeaned = demeaned * weights[:, np.newaxis]
    cov_values = weighted_demeaned.T @ demeaned 
    daily_cov_matrix = pd.DataFrame(
        cov_values, index=log_returns.columns, columns=log_returns.columns
    )

    daily_volatility = pd.Series(
        np.sqrt(np.diag(cov_values)), index=log_returns.columns
    )

    d_inv = np.where(daily_volatility.values > 0, 1.0 / daily_volatility.values, 0.0)
    corr_values = d_inv[:, None] * cov_values * d_inv[None, :]
    correlation_matrix = pd.DataFrame(
        corr_values, index=log_returns.columns, columns=log_returns.columns
    )

    ewma_weights = pd.Series(weights, index=log_returns.index, name="ewma_weight")

    return {
        "daily_mean_returns": daily_mean_returns,
        "daily_cov_matrix": daily_cov_matrix,
        "daily_volatility": daily_volatility,
        "annualized_mean_returns": daily_mean_returns * annualization_factor,
        "annualized_cov_matrix": daily_cov_matrix * annualization_factor,
        "annualized_volatility": daily_volatility * (annualization_factor ** 0.5),
        "correlation_matrix": correlation_matrix,
        "ewma_lambda": lam,
        "ewma_weights": ewma_weights,
    }


def load_user_params(mean_path: str, cov_path: str) -> dict:
    """
    Build a calibration_result dict from user-supplied mean and
    covariance CSV files, bypassing historical estimation entirely.
    This allows monte_carlo_var_es and parametric_var to be run
    without providing price history.

    mean_path: CSV with columns [underlying, daily_mean]
    cov_path:  CSV with 'underlying' as index and one column per ticker
    """
    mean_df = pd.read_csv(mean_path).set_index("underlying")["daily_mean"]
    cov_df  = pd.read_csv(cov_path).set_index("underlying")
    cov_df  = cov_df.astype(float)

    ann = 252
    daily_vol = pd.Series(
        np.sqrt(np.diag(cov_df.values)),
        index=cov_df.index,
    )
    
    d_inv = 1.0 / daily_vol.values
    corr_values = d_inv[:, None] * cov_df.values * d_inv[None, :]
    correlation = pd.DataFrame(
        corr_values, index=cov_df.index, columns=cov_df.columns
    )

    return {
        "daily_mean_returns":      mean_df,
        "daily_cov_matrix":        cov_df,
        "daily_volatility":        daily_vol,
        "annualized_mean_returns": mean_df * ann,
        "annualized_cov_matrix":   cov_df * ann,
        "annualized_volatility":   daily_vol * np.sqrt(ann),
        "correlation_matrix":      correlation,
    }