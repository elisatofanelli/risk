"""Historical calibration from price data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from a price history DataFrame."""

    log_returns = np.log(price_df / price_df.shift(1))
    return log_returns.dropna(how="any")


def calibrate_from_history(price_df: pd.DataFrame, annualization_factor: int = 252) -> dict:
    """Calibrate mean, covariance, volatility, and correlation from historical log returns."""

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
    # Correlation = D^{-1} @ Cov @ D^{-1} where D = diag(vol)
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