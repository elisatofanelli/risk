from datetime import datetime

import pandas as pd
import pytest

from src.input_loader import load_portfolio
from src.portfolio import value_portfolio


def test_portfolio_valuation_works():
    df = load_portfolio("data/sample_portfolio.csv")
    spot_prices = {"AAPL": 210.0, "MSFT": 340.0}
    value = value_portfolio(df, spot_prices, datetime(2026, 5, 11))
    assert isinstance(value, float)


def test_input_validation_catches_missing_option_fields(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text(
        "instrument_id,type,underlying,position,strike,maturity,option_type,implied_vol,risk_free_rate\n"
        "OPT1,option,AAPL,10,,,call,0.25,0.04\n"
    )

    with pytest.raises(ValueError, match="Missing required option field"):
        load_portfolio(str(csv))


def test_input_validation_catches_invalid_instrument_type(tmp_path):
    csv = tmp_path / "bad_type.csv"
    # Insert a non-supported type "bond"
    csv.write_text(
        "instrument_id,type,underlying,position,strike,maturity,option_type,implied_vol,risk_free_rate\n"
        "BND1,bond,AAPL,10,,,,,\n"
    )

    with pytest.raises(ValueError, match="Invalid instrument type"):
        load_portfolio(str(csv))