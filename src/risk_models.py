"""Historical, Monte Carlo, and parametric risk models."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm

from .calibration import compute_log_returns
from .portfolio import portfolio_delta_exposures, value_portfolio


def _loss_summary(losses: np.ndarray, confidence_level: float, current_value: float) -> dict:
    var = float(np.quantile(losses, confidence_level))
    tail = losses[losses >= var]
    es = float(tail.mean()) if len(tail) else var
    return {
        "VaR": max(var, 0.0),
        "ES": max(es, 0.0),
        "confidence_level": confidence_level,
        "current_value": float(current_value),
    }


def historical_var_es(portfolio_df, price_history_df, valuation_date, confidence_level=0.95) -> dict:
    price_history = price_history_df.copy()
    log_returns = compute_log_returns(price_history).dropna(how="any")
    if log_returns.empty:
        raise ValueError("Not enough historical observations to compute risk.")

    current_prices = price_history.iloc[-1].to_dict()
    current_value = value_portfolio(portfolio_df, current_prices, valuation_date)

    losses = []
    for _, r in log_returns.iterrows():
        scenario_prices = {
            col: float(current_prices[col] * np.exp(r[col]))
            for col in price_history.columns
        }
        scenario_value = value_portfolio(portfolio_df, scenario_prices, valuation_date)
        losses.append(-(scenario_value - current_value))

    losses_arr = np.asarray(losses, dtype=float)
    result = _loss_summary(losses_arr, confidence_level, current_value)
    result.update({"number_of_scenarios": int(len(losses_arr)), "method": "historical"})
    return result


def monte_carlo_var_es(
    portfolio_df,
    spot_prices,
    calibration_result,
    valuation_date,
    confidence_level=0.95,
    n_sims=10000,
    random_seed=42,
) -> dict:
    daily_mean = calibration_result["daily_mean_returns"].reindex(spot_prices.keys()).fillna(0.0)
    daily_cov = calibration_result["daily_cov_matrix"].reindex(index=spot_prices.keys(), columns=spot_prices.keys()).fillna(0.0)

    current_value = value_portfolio(portfolio_df, spot_prices, valuation_date)
    rng = np.random.default_rng(random_seed)
    simulated_returns = rng.multivariate_normal(daily_mean.values, daily_cov.values, size=n_sims)

    losses = []
    cols = list(spot_prices.keys())
    for scenario in simulated_returns:
        scenario_prices = {
            col: float(spot_prices[col] * np.exp(scenario[i]))
            for i, col in enumerate(cols)
        }
        scenario_value = value_portfolio(portfolio_df, scenario_prices, valuation_date)
        losses.append(-(scenario_value - current_value))

    losses_arr = np.asarray(losses, dtype=float)
    result = _loss_summary(losses_arr, confidence_level, current_value)
    result.update(
        {
            "number_of_scenarios": int(n_sims),
            "n_sims": int(n_sims),
            "random_seed": int(random_seed),
            "method": "monte_carlo",
        }
    )
    return result


def parametric_var(
    portfolio_df,
    spot_prices,
    calibration_result,
    valuation_date,
    confidence_level=0.95,
) -> dict:
    exposures = portfolio_delta_exposures(portfolio_df, spot_prices, valuation_date)
    underlying_order = list(spot_prices.keys())
    exposure_vector = np.array([exposures.get(u, 0.0) * float(spot_prices[u]) for u in underlying_order], dtype=float)
    cov = calibration_result["daily_cov_matrix"].reindex(index=underlying_order, columns=underlying_order).fillna(0.0)
    mean = calibration_result["daily_mean_returns"].reindex(underlying_order).fillna(0.0).values

    portfolio_mean = float(exposure_vector @ mean)
    portfolio_var = float(exposure_vector @ cov.values @ exposure_vector)
    portfolio_std = sqrt(max(portfolio_var, 0.0))
    z_score = float(norm.ppf(confidence_level))
    var = max(z_score * portfolio_std - portfolio_mean, 0.0)

    current_value = value_portfolio(portfolio_df, spot_prices, valuation_date)
    return {
        "VaR": float(var),
        "confidence_level": confidence_level,
        "current_value": float(current_value),
        "portfolio_std": float(portfolio_std),
        "method": "parametric_delta_normal",
        "number_of_scenarios": None,
        "ES": None,
    }

