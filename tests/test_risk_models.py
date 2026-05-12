from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.calibration import calibrate_from_history, compute_log_returns
from src.input_loader import load_portfolio
from src.market_data import load_price_history
from src.reporting import create_risk_report
from src.risk_models import historical_var_es, monte_carlo_var_es, parametric_var
from src.risk_models import _shock_portfolio_volatility


def test_log_returns_shape_is_correct():
    prices = load_price_history("data/sample_historical_prices.csv")
    returns = compute_log_returns(prices)
    assert returns.shape[0] == prices.shape[0] - 1
    assert returns.shape[1] == prices.shape[1]


def test_covariance_matrix_has_correct_dimensions():
    prices = load_price_history("data/sample_historical_prices.csv")
    calib = calibrate_from_history(prices)
    cov = calib["daily_cov_matrix"]
    assert cov.shape == (prices.shape[1], prices.shape[1])


def test_risk_models_and_report(tmp_path):
    portfolio_df = load_portfolio("data/sample_portfolio.csv")
    prices = load_price_history("data/sample_historical_prices.csv")
    spot_prices = prices.iloc[-1].to_dict()
    calibration_result = calibrate_from_history(prices)
    valuation_date = datetime(2026, 5, 11)

    hist = historical_var_es(portfolio_df, prices, valuation_date)
    mc = monte_carlo_var_es(portfolio_df, spot_prices, calibration_result, valuation_date, n_sims=2000)
    param = parametric_var(portfolio_df, spot_prices, calibration_result, valuation_date)

    assert hist["VaR"] >= 0
    assert mc["VaR"] >= 0
    assert param["VaR"] >= 0

    report_path = tmp_path / "risk_report.csv"
    create_risk_report({"historical": hist, "mc": mc, "param": param}, str(report_path))
    assert report_path.exists()


def test_negative_prices_are_rejected(tmp_path):
    csv = tmp_path / "bad_prices.csv"
    # Insert a negative historical price (-150.0)
    csv.write_text(
        "date,AAPL\n"
        "2026-01-01,150.0\n"
        "2026-01-02,-150.0\n"
    )
    
    with pytest.raises(ValueError, match="contains non-positive prices"):
        load_price_history(str(csv))


def test_robustness_extreme_outlier_shock():
    """
    Data Robustness Test: Verify how the model behaves with an extreme
    historical outlier (e.g., a 90% crash in a single day).
    The system must not crash by calculating standard deviations of NaNs.
    """
    df_outlier = pd.DataFrame({
        "AAPL": [100.0, 10.0, 10.5]  # 3 prices
    }, index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]))
    
    calib = calibrate_from_history(df_outlier)
    
    assert not np.isnan(calib["daily_volatility"]["AAPL"])


def test_robustness_covariance_short_window():
    """
    Operational Limits Test: Verify the stability of the covariance matrix
    when the historical sample is extremely short (e.g., only 3 days of data).
    """
    df_short = pd.DataFrame({
        "AAPL": [100.0, 102.0, 101.0],
        "MSFT": [200.0, 205.0, 203.0]
    }, index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]))
    
    calib = calibrate_from_history(df_short)
    cov = calib["daily_cov_matrix"]
    
    # Must successfully create a 2x2 matrix without null values
    assert cov.shape == (2, 2)
    assert not cov.isnull().any().any()

def test_dynamic_volatility_leverage_effect():
    """Verify that a drop in stock price causes implied volatility to rise."""
    import pandas as pd
    import numpy as np
    
    # Create a dummy portfolio with one option
    portfolio_df = pd.DataFrame({
        "type": ["option"],
        "underlying": ["AAPL"],
        "implied_vol": [0.20]  # Starting vol at 20%
    })
    
    # Simulate a severe market crash (-10% log return)
    log_returns = {"AAPL": -0.10}
    
    shocked_df = _shock_portfolio_volatility(portfolio_df, log_returns)
    shocked_vol = shocked_df.loc[0, "implied_vol"]
    
    # Formula is: vol * exp(-0.5 * return) -> 0.20 * exp(-0.5 * -0.10)
    expected_vol = 0.20 * np.exp(-0.5 * -0.10)
    
    assert shocked_vol > 0.20  # Volatility must have increased
    assert abs(shocked_vol - expected_vol) < 1e-6