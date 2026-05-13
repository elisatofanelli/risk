"""VaR backtesting utilities."""

from __future__ import annotations

from math import log

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .calibration import calibrate_from_history
from .portfolio import value_portfolio
from .risk_models import historical_var_es, monte_carlo_var_es, parametric_var, \
    ewma_historical_var_es, ewma_parametric_var, ewma_monte_carlo_var_es


def _method_var(
    portfolio_df,
    history_window: pd.DataFrame,
    valuation_date,
    method: str,
    confidence_level: float,
    n_sims: int,
    random_seed: int,
) -> dict:
    current_prices = history_window.iloc[-1].to_dict()
    calibration_result = calibrate_from_history(history_window)

    if method == "historical":
        return historical_var_es(portfolio_df, history_window, valuation_date, confidence_level=confidence_level)
    if method == "parametric":
        return parametric_var(portfolio_df, current_prices, calibration_result, valuation_date, confidence_level=confidence_level)
    if method == "monte_carlo":
        return monte_carlo_var_es(
            portfolio_df, current_prices, calibration_result, valuation_date,
            confidence_level=confidence_level, n_sims=n_sims, random_seed=random_seed,
        )
    if method == "ewma_historical":
        return ewma_historical_var_es(
            portfolio_df, history_window, valuation_date,
            confidence_level=confidence_level,
        )
    if method == "ewma_parametric":
        return ewma_parametric_var(
            portfolio_df, current_prices, history_window, valuation_date,
            confidence_level=confidence_level,
        )
    if method == "ewma_monte_carlo":
        return ewma_monte_carlo_var_es(
            portfolio_df, current_prices, history_window, valuation_date,
            confidence_level=confidence_level, n_sims=n_sims, random_seed=random_seed,
        )
    raise ValueError("method must be one of: historical, parametric, monte_carlo, "
                     "ewma_historical, ewma_parametric, ewma_monte_carlo")


def backtest_var(
    portfolio_df,
    price_history_df,
    method,
    confidence_level=0.95,
    calibration_window=250,
    n_sims=5000,
    random_seed=42,
) -> pd.DataFrame:
    """Run rolling one-day-ahead VaR backtesting."""

    _valid_methods = {"historical", "parametric", "monte_carlo",
                      "ewma_historical", "ewma_parametric", "ewma_monte_carlo"}
    if method not in _valid_methods:
        raise ValueError(f"method must be one of: {', '.join(sorted(_valid_methods))}")

    if len(price_history_df) <= calibration_window:
        raise ValueError("Not enough observations for the requested calibration window.")

    rows = []
    dates = list(price_history_df.index)

    for i in range(calibration_window, len(price_history_df) - 1):
        history_window = price_history_df.iloc[: i + 1]
        date_t = dates[i]
        date_t_plus_1 = dates[i + 1]

        spot_t = history_window.iloc[-1].to_dict()
        valuation_date = pd.Timestamp(date_t).to_pydatetime()
        current_value = value_portfolio(portfolio_df, spot_t, valuation_date)

        risk_result = _method_var(
            portfolio_df=portfolio_df,
            history_window=history_window,
            valuation_date=valuation_date,
            method=method,
            confidence_level=confidence_level,
            n_sims=n_sims,
            random_seed=random_seed + i,
        )
        var = float(risk_result["VaR"])

        spot_t_plus_1 = price_history_df.iloc[i + 1].to_dict()
        next_valuation_date = pd.Timestamp(date_t_plus_1).to_pydatetime()
        current_value_t_plus_1 = value_portfolio(portfolio_df, spot_t_plus_1, next_valuation_date)
        actual_pnl = float(current_value_t_plus_1 - current_value)
        actual_loss = float(-actual_pnl)
        exception = bool(actual_loss > var)

        rows.append(
            {
                "date": date_t_plus_1,
                "method": method,
                "confidence_level": confidence_level,
                "portfolio_value_t": float(current_value),
                "portfolio_value_t_plus_1": float(current_value_t_plus_1),
                "actual_pnl": actual_pnl,
                "actual_loss": actual_loss,
                "VaR": var,
                "exception": exception,
            }
        )

    return pd.DataFrame(rows)


def summarize_backtest(backtest_df: pd.DataFrame) -> dict:
    """Summarize backtest exceptions with diagnostic and clustering check."""

    if backtest_df.empty:
        raise ValueError("Backtest DataFrame is empty.")

    method = str(backtest_df["method"].iloc[0])
    confidence_level = float(backtest_df["confidence_level"].iloc[0])
    n_obs = int(len(backtest_df))
    n_exc = int(backtest_df["exception"].sum())
    actual_rate = n_exc / n_obs if n_obs else 0.0
    expected_rate = 1.0 - confidence_level
    pass_fail = (
        0.02 <= actual_rate <= 0.08
        if abs(confidence_level - 0.95) < 1e-12
        else abs(actual_rate - expected_rate) <= 0.03
    )

    p_value = None
    if 0 < actual_rate < 1 and 0 < expected_rate < 1:
        lr_uc = -2.0 * (
            (n_exc * log(expected_rate) + (n_obs - n_exc) * log(1.0 - expected_rate))
            - (n_exc * log(actual_rate) + (n_obs - n_exc) * log(1.0 - actual_rate))
        )
        p_value = float(1 - chi2.cdf(lr_uc, df=1))

    exc_flags = backtest_df["exception"].astype(int).values
    consecutive_pairs = int(((exc_flags[:-1] == 1) & (exc_flags[1:] == 1)).sum())
    clustering_concern = consecutive_pairs > 1

    interpretation = (
        "Observed exception rate is within the simple diagnostic tolerance."
        if pass_fail
        else "Observed exception rate falls outside the simple diagnostic tolerance."
    )
    clustering_note = (
        f"{consecutive_pairs} consecutive exception pair(s) detected — "
        + ("possible clustering, investigate further."
           if clustering_concern
           else "no material clustering observed.")
    )

    return {
        "method": method,
        "confidence_level": confidence_level,
        "number_of_observations": n_obs,
        "number_of_exceptions": n_exc,
        "expected_exception_rate": expected_rate,
        "actual_exception_rate": actual_rate,
        "pass_fail_simple": pass_fail,
        "interpretation": interpretation,
        "kupiec_p_value": p_value,
        "consecutive_exception_pairs": consecutive_pairs,
        "clustering_concern": clustering_concern,
        "clustering_note": clustering_note,
    }

