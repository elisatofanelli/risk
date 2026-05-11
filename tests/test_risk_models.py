from datetime import datetime

from src.calibration import calibrate_from_history, compute_log_returns
from src.input_loader import load_portfolio
from src.market_data import load_price_history
from src.reporting import create_risk_report
from src.risk_models import historical_var_es, monte_carlo_var_es, parametric_var


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
    create_risk_report({"historical": hist, "monte_carlo": mc, "parametric": param}, str(report_path))
    assert report_path.exists()

