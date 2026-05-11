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
