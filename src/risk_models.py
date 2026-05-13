"""Historical, Monte Carlo, and parametric risk models."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm

from .calibration import compute_log_returns, ewma_calibrate
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

    current_value = value_portfolio(portfolio_df, spot_prices, valuation_date)
    return {
        "VaR": var,
        "ES": None,  # ES omitted as not requested for parametric model
        "confidence_level": confidence_level,
        "current_value": float(current_value),
    }

def _weighted_loss_summary(
    losses: np.ndarray,
    weights: np.ndarray,
    confidence_level: float,
    current_value: float,
) -> dict:
    """
    Compute VaR and ES from a weighted loss distribution.

    VaR is the weighted quantile: the smallest loss L such that the cumulative
    EWMA weight of scenarios with loss ≤ L is ≥ confidence_level.
    ES is the weighted average of losses that exceed the VaR threshold.

    Parameters
    ----------
    losses : np.ndarray
        Per-scenario loss values (positive = loss).
    weights : np.ndarray
        Non-negative EWMA weights corresponding to each scenario; need not sum to 1
        (they are re-normalised internally).
    confidence_level : float
        E.g. 0.95 for 95% VaR.
    current_value : float
        Current mark-to-market portfolio value (stored for reporting).
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    sort_idx = np.argsort(losses)
    sorted_losses = losses[sort_idx]
    sorted_weights = weights[sort_idx]
    cum_weights = np.cumsum(sorted_weights)

    var_idx = int(np.searchsorted(cum_weights, confidence_level))
    var_idx = min(var_idx, len(sorted_losses) - 1)
    var = float(sorted_losses[var_idx])

    tail_mask = sorted_losses >= var
    tail_losses = sorted_losses[tail_mask]
    tail_weights = sorted_weights[tail_mask]
    tail_weight_sum = tail_weights.sum()
    es = float((tail_losses * tail_weights).sum() / tail_weight_sum) if tail_weight_sum > 0 else var

    return {
        "VaR": max(var, 0.0),
        "ES": max(es, 0.0),
        "confidence_level": confidence_level,
        "current_value": float(current_value),
    }


def ewma_historical_var_es(
    portfolio_df,
    price_history_df,
    valuation_date,
    confidence_level: float = 0.95,
    lam: float = 0.94,
) -> dict:
    """
    EWMA-weighted historical simulation VaR and ES.

    Identical to ``historical_var_es`` in scenario construction (each past
    log-return window is replayed against today's portfolio) but instead of
    assigning equal probability 1/T to every scenario, each scenario receives
    an EWMA weight λ^(T-1-t) (normalised).  More recent history therefore
    dominates the loss distribution, making VaR and ES more responsive to
    recent volatility regimes.

    Volatility surface shocks (leverage-effect adjustment on implied vol) are
    applied identically to the plain historical method.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Portfolio positions (same schema as used throughout the engine).
    price_history_df : pd.DataFrame
        Full price history; columns = underlyings, rows = dates (ascending).
    valuation_date : datetime
        Date used for option pricing (time-to-expiry calculation).
    confidence_level : float
        VaR/ES confidence level, e.g. 0.95.
    lam : float
        EWMA decay factor λ ∈ (0, 1).  Default 0.94 (RiskMetrics daily).
    """
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lam must be in (0, 1), got {lam}")

    price_history = price_history_df.copy()
    log_returns = compute_log_returns(price_history).dropna(how="any")
    if log_returns.empty:
        raise ValueError("Not enough historical observations to compute risk.")

    n = len(log_returns)
    raw_weights = np.array([lam ** (n - 1 - i) for i in range(n)])
    weights = raw_weights / raw_weights.sum()

    current_prices = price_history.iloc[-1].to_dict()
    current_value = value_portfolio(portfolio_df, current_prices, valuation_date)

    losses = []
    for _, r in log_returns.iterrows():
        r_dict = r.to_dict()
        scenario_prices = {
            col: float(current_prices[col] * np.exp(r_dict[col]))
            for col in price_history.columns
        }
        scenario_portfolio = _shock_portfolio_volatility(portfolio_df, r_dict)
        scenario_value = value_portfolio(scenario_portfolio, scenario_prices, valuation_date)
        losses.append(current_value - scenario_value)

    losses_array = np.array(losses)
    summary = _weighted_loss_summary(losses_array, weights, confidence_level, current_value)
    summary["number_of_scenarios"] = n
    summary["ewma_lambda"] = lam
    return summary


def ewma_parametric_var(
    portfolio_df,
    spot_prices,
    price_history_df,
    valuation_date,
    confidence_level: float = 0.95,
    lam: float = 0.94,
) -> dict:
    """
    Parametric (delta-normal) VaR and ES using an EWMA covariance matrix.

    Replaces the equally-weighted sample covariance in ``parametric_var`` with
    an EWMA covariance matrix, making the risk estimate more sensitive to
    recent volatility clustering.  The analytical delta-normal formula is
    otherwise unchanged:

        VaR = z_α · σ_P − μ_P
        ES  = −μ_P + σ_P · φ(z_α) / (1 − α)

    where σ_P and μ_P are the EWMA-based portfolio standard deviation and mean
    computed from the delta-exposure vector.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Portfolio positions.
    spot_prices : dict[str, float]
        Current spot prices keyed by underlying ticker.
    price_history_df : pd.DataFrame
        Price history used to compute the EWMA calibration.
    valuation_date : datetime
        Pricing date for options.
    confidence_level : float
        VaR/ES confidence level, e.g. 0.95.
    lam : float
        EWMA decay factor λ ∈ (0, 1).  Default 0.94.
    """
    calibration_result = ewma_calibrate(price_history_df, lam=lam)
    result = parametric_var(
        portfolio_df, spot_prices, calibration_result, valuation_date, confidence_level
    )
    result["ewma_lambda"] = lam
    return result


def ewma_monte_carlo_var_es(
    portfolio_df,
    spot_prices,
    price_history_df,
    valuation_date,
    confidence_level: float = 0.95,
    lam: float = 0.94,
    n_sims: int = 10_000,
    random_seed: int = 42,
) -> dict:
    """
    Monte Carlo VaR and ES driven by an EWMA covariance matrix.

    Identical in structure to ``monte_carlo_var_es``: multivariate normal
    log-returns are simulated and replayed against the current portfolio
    (with implied-vol shocks).  The key difference is that the mean vector
    and covariance matrix fed to the simulation are EWMA estimates rather
    than equally-weighted sample moments, so the simulation reflects recent
    volatility regimes more strongly.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Portfolio positions.
    spot_prices : dict[str, float]
        Current spot prices.
    price_history_df : pd.DataFrame
        Price history used to compute the EWMA calibration.
    valuation_date : datetime
        Pricing date for options.
    confidence_level : float
        VaR/ES confidence level, e.g. 0.95.
    lam : float
        EWMA decay factor λ ∈ (0, 1).  Default 0.94.
    n_sims : int
        Number of Monte Carlo paths.  Default 10 000.
    random_seed : int
        RNG seed for reproducibility.
    """
    calibration_result = ewma_calibrate(price_history_df, lam=lam)
    result = monte_carlo_var_es(
        portfolio_df,
        spot_prices,
        calibration_result,
        valuation_date,
        confidence_level=confidence_level,
        n_sims=n_sims,
        random_seed=random_seed,
    )
    result["ewma_lambda"] = lam
    return result
