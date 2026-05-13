"""Portfolio valuation and risk exposure helpers."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from .pricing import black_scholes_delta, black_scholes_price


def get_underlyings(portfolio_df: pd.DataFrame) -> list[str]:
    """Extract a sorted list of unique underlying asset tickers from the portfolio."""
    return sorted(portfolio_df["underlying"].dropna().astype(str).unique().tolist())


def _time_to_maturity(valuation_date: datetime, maturity: pd.Timestamp) -> float:
    """Calculate the time to maturity in years between the valuation date and the maturity date."""
    if pd.isna(maturity):
        return 0.0
    delta_days = (pd.Timestamp(maturity) - pd.Timestamp(valuation_date)).days
    return max(delta_days / 365.0, 0.0)


def value_instrument(row: pd.Series, spot_prices: dict[str, float], valuation_date: datetime) -> float:
    """Calculate the monetary value of a single instrument (stock or option) based on current spot prices."""
    underlying = row["underlying"]
    if underlying not in spot_prices:
        raise ValueError(f"Missing spot price for underlying '{underlying}'.")

    spot = float(spot_prices[underlying])
    position = float(row["position"])

    if row["type"] == "stock":
        return position * spot

    T = _time_to_maturity(valuation_date, row["maturity"])
    strike = float(row["strike"])
    sigma = float(row["implied_vol"])
    r = float(row["risk_free_rate"])
    option_type = str(row["option_type"])
    price = black_scholes_price(spot, strike, T, sigma, r, option_type)
    return position * price


def value_portfolio(portfolio_df: pd.DataFrame, spot_prices: dict[str, float], valuation_date: datetime) -> float:
    """Calculate the total aggregate value of the entire portfolio."""
    return float(
        sum(value_instrument(row, spot_prices, valuation_date) for _, row in portfolio_df.iterrows())
    )


def portfolio_delta_exposures(portfolio_df: pd.DataFrame, spot_prices: dict[str, float], valuation_date: datetime) -> dict[str, float]:
    """Calculate the aggregate delta exposure for each underlying asset in the portfolio."""
    exposures: dict[str, float] = {}
    for _, row in portfolio_df.iterrows():
        underlying = row["underlying"]
        spot = float(spot_prices[underlying])
        position = float(row["position"])

        if row["type"] == "stock":
            delta = 1.0
        else:
            T = _time_to_maturity(valuation_date, row["maturity"])
            delta = black_scholes_delta(
                spot,
                float(row["strike"]),
                T,
                float(row["implied_vol"]),
                float(row["risk_free_rate"]),
                str(row["option_type"]),
            )

        exposures[underlying] = exposures.get(underlying, 0.0) + position * delta
    return exposures