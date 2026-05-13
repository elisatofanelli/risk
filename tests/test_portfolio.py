from datetime import datetime

import pandas as pd
import pytest
import numpy as np
from src.portfolio import value_instrument, value_portfolio
from src.pricing import black_scholes_price
from src.input_loader import load_portfolio
from src.portfolio import value_portfolio


def test_portfolio_valuation_works():
    """Verify that valuing a valid portfolio returns a float type."""
    df = load_portfolio("data/sample_portfolio.csv")
    spot_prices = {"AAPL": 210.0, "MSFT": 340.0}
    value = value_portfolio(df, spot_prices, datetime(2026, 5, 11))
    assert isinstance(value, float)


def test_input_validation_catches_missing_option_fields(tmp_path):
    """Ensure that loading a portfolio with missing required option fields raises a ValueError."""
    csv = tmp_path / "bad.csv"
    csv.write_text(
        "instrument_id,type,underlying,position,strike,maturity,option_type,implied_vol,risk_free_rate\n"
        "OPT1,option,AAPL,10,,,call,0.25,0.04\n"
    )

    with pytest.raises(ValueError, match="Missing required option field"):
        load_portfolio(str(csv))


def test_input_validation_catches_invalid_instrument_type(tmp_path):
    """Ensure that loading a portfolio with an unsupported instrument type raises a ValueError."""
    csv = tmp_path / "bad_type.csv"
    # Insert a non-supported type "bond"
    csv.write_text(
        "instrument_id,type,underlying,position,strike,maturity,option_type,implied_vol,risk_free_rate\n"
        "BND1,bond,AAPL,10,,,,,\n"
    )

    with pytest.raises(ValueError, match="Invalid instrument type"):
        load_portfolio(str(csv))

def test_pnl_attribution_stock_plus_option_equals_total():
    """
    P&L Attribution Test: For a mixed portfolio (one stock + one option),
    verify that the sum of the individual instrument P&Ls equals the total
    portfolio P&L under a price shock.

    This confirms the portfolio valuation engine correctly aggregates
    instrument-level changes with no unexplained residual.
    """

    valuation_date = datetime(2026, 5, 11)

    portfolio_df = pd.DataFrame([
        {
            "instrument_id": "STK1",
            "type": "stock",
            "underlying": "AAPL",
            "position": 10.0,
            "strike": pd.NA,
            "maturity": pd.NA,
            "option_type": pd.NA,
            "implied_vol": pd.NA,
            "risk_free_rate": pd.NA,
        },
        {
            "instrument_id": "OPT1",
            "type": "option",
            "underlying": "AAPL",
            "position": 1.0,
            "strike": 200.0,
            "maturity": pd.Timestamp("2027-05-11"),
            "option_type": "call",
            "implied_vol": 0.25,
            "risk_free_rate": 0.05,
        },
    ])

    spot_before = {"AAPL": 200.0}
    spot_after  = {"AAPL": 195.0}  # -2.5% shock

    value_before = value_portfolio(portfolio_df, spot_before, valuation_date)
    value_after  = value_portfolio(portfolio_df, spot_after,  valuation_date)
    total_pnl    = value_after - value_before

    stock_row  = portfolio_df[portfolio_df["type"] == "stock"].iloc[0]
    option_row = portfolio_df[portfolio_df["type"] == "option"].iloc[0]

    stock_pnl  = (value_instrument(stock_row,  spot_after,  valuation_date)
                - value_instrument(stock_row,  spot_before, valuation_date))
    option_pnl = (value_instrument(option_row, spot_after,  valuation_date)
                - value_instrument(option_row, spot_before, valuation_date))

    sum_of_parts = stock_pnl + option_pnl

    assert abs(total_pnl - sum_of_parts) < 1e-6, (
        f"Portfolio P&L ({total_pnl:.6f}) != "
        f"sum of instrument P&Ls ({sum_of_parts:.6f}). "
        "Unexplained residual detected."
    )

    assert stock_pnl < 0, "Stock should lose value when spot drops"
    assert option_pnl < 0, "Call option should lose value when spot drops"

    print(f"\nP&L Attribution:")
    print(f"  Stock  P&L: {stock_pnl:.4f}")
    print(f"  Option P&L: {option_pnl:.4f}")
    print(f"  Total  P&L: {total_pnl:.4f}  (residual: {abs(total_pnl - sum_of_parts):.2e})")