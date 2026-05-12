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


def _shock_portfolio_volatility(portfolio_df: pd.DataFrame, log_returns: dict[str, float]) -> pd.DataFrame:
    """
    Dynamically adjust option implied volatility based on underlying spot shocks.
    Uses a standard leverage effect heuristic: volatility increases when spot drops.
    Formula: new_vol = original_vol * exp(-0.5 * log_return)
    """
    shocked_df = portfolio_df.copy()
    option_mask = shocked_df["type"] == "option"

    for idx, row in shocked_df[option_mask].iterrows():
        underlying = row["underlying"]
        if underlying in log_returns:
            r = log_returns[underlying]
            original_vol = float(row["implied_vol"])
            # Apply an inverse elasticity of 0.5 to model the leverage effect
            shocked_vol = original_vol * np.exp(-0.5 * r)
            shocked_df.at[idx, "implied_vol"] = shocked_vol

    return shocked_df


def historical_var_es(portfolio_df, price_history_df, valuation_date, confidence_level=0.95) -> dict:
    price_history = price_history_df.copy()
    log_returns = compute_log_returns(price_history).dropna(how="any")
    if log_returns.empty:
        raise ValueError("Not enough historical observations to compute risk.")

    current_prices = price_history.iloc[-1].to_dict()
    current_value = value_portfolio(portfolio_df, current_prices, valuation_date)

    losses = []
    for _, r in log_returns.iterrows():
        r_dict = r.to_dict()
        scenario_prices = {
            col: float(current_prices[col] * np.exp(r_dict[col]))
            for col in price_history.columns
        }
        
        # Apply dynamic volatility shock to the options
        scenario_portfolio = _shock_portfolio_volatility(portfolio_df, r_dict)
        scenario_value = value_portfolio(scenario_portfolio, scenario_prices, valuation_date)
        
        losses.append(current_value - scenario_value)

    losses_array = np.array(losses)
    summary = _loss_summary(losses_array, confidence_level, current_value)
    summary["number_of_scenarios"] = len(losses_array)
    return summary


def monte_carlo_var_es(
    portfolio_df,
    spot_prices,
    calibration_result,
    valuation_date,
    confidence_level=0.95,
    n_sims=10000,
    random_seed=42,
) -> dict:
    underlying_order = list(spot_prices.keys())
    mean = calibration_result["daily_mean_returns"].reindex(underlying_order).fillna(0.0).values
    cov = calibration_result["daily_cov_matrix"].reindex(index=underlying_order, columns=underlying_order).fillna(0.0).values

    rng = np.random.default_rng(random_seed)
    simulated_returns = rng.multivariate_normal(mean, cov, size=n_sims)

    current_value = value_portfolio(portfolio_df, spot_prices, valuation_date)
    losses = []

    for i in range(n_sims):
        r_dict = {u: float(simulated_returns[i, j]) for j, u in enumerate(underlying_order)}
        scenario_prices = {
            u: float(spot_prices[u] * np.exp(r_dict[u]))
            for u in underlying_order
        }
        
        # Apply dynamic volatility shock to the options
        scenario_portfolio = _shock_portfolio_volatility(portfolio_df, r_dict)
        scenario_value = value_portfolio(scenario_portfolio, scenario_prices, valuation_date)
        
        losses.append(current_value - scenario_value)

    losses_array = np.array(losses)
    summary = _loss_summary(losses_array, confidence_level, current_value)
    summary["number_of_scenarios"] = n_sims
    return summary


def parametric_var(
    portfolio_df,
    spot_prices,
    calibration_result,
    valuation_date,
    confidence_level=0.95,
) -> dict:
    exposures = portfolio_delta_exposures(portfolio_df, spot_prices, valuation_date)
    underlying_order = list(spot_prices.keys())
    exposure_vector = np.array(
        [exposures.get(u, 0.0) * float(spot_prices[u]) for u in underlying_order],
        dtype=float,
    )
    cov = (
        calibration_result["daily_cov_matrix"]
        .reindex(index=underlying_order, columns=underlying_order)
        .fillna(0.0)
    )
    mean = (
        calibration_result["daily_mean_returns"]
        .reindex(underlying_order)
        .fillna(0.0)
        .values
    )

    portfolio_mean = float(exposure_vector @ mean)
    portfolio_var_scalar = float(exposure_vector @ cov.values @ exposure_vector)
    portfolio_std = sqrt(max(portfolio_var_scalar, 0.0))
    z_score = float(norm.ppf(confidence_level))
    var = max(z_score * portfolio_std - portfolio_mean, 0.0)
    es = float(-portfolio_mean + portfolio_std * norm.pdf(z_score) / (1.0 - confidence_level))
    es = max(es, 0.0)

    current_value = value_portfolio(portfolio_df, spot_prices, valuation_date)
    return {
        "VaR": var,
        "ES": es,
        "confidence_level": confidence_level,
        "current_value": current_value,
        "number_of_scenarios": 0,
    }