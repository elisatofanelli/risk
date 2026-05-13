from pathlib import Path

from src.backtesting import backtest_var, summarize_backtest
from src.input_loader import load_portfolio
from src.market_data import load_price_history
from src.reporting import save_backtest_summary
import numpy as np


def test_backtest_output_has_required_columns():
    portfolio_df = load_portfolio("data/sample_portfolio.csv")
    prices = load_price_history("data/sample_historical_prices.csv")
    backtest_df = backtest_var(portfolio_df, prices, "historical", calibration_window=260, n_sims=500)
    required = {
        "date",
        "method",
        "confidence_level",
        "portfolio_value_t",
        "portfolio_value_t_plus_1",
        "actual_pnl",
        "actual_loss",
        "VaR",
        "exception",
    }
    assert required.issubset(backtest_df.columns)


def test_exception_column_is_boolean_and_rate_valid(tmp_path):
    portfolio_df = load_portfolio("data/sample_portfolio.csv")
    prices = load_price_history("data/sample_historical_prices.csv")
    backtest_df = backtest_var(portfolio_df, prices, "parametric", calibration_window=260)
    assert backtest_df["exception"].dtype == bool

    summary = summarize_backtest(backtest_df)
    assert 0.0 <= summary["actual_exception_rate"] <= 1.0

    summary_path = tmp_path / "backtest_summary.csv"
    save_backtest_summary([summary], str(summary_path))
    assert summary_path.exists()


def test_backtesting_does_not_look_ahead(monkeypatch):
    portfolio_df = load_portfolio("data/sample_portfolio.csv")
    prices = load_price_history("data/sample_historical_prices.csv")
    seen_lengths = []

    from src import backtesting as bt

    original = bt.calibrate_from_history

    def wrapped(history_window):
        seen_lengths.append(len(history_window))
        return original(history_window)

    monkeypatch.setattr(bt, "calibrate_from_history", wrapped)
    bt.backtest_var(portfolio_df, prices.iloc[:270], "historical", calibration_window=250)

    assert seen_lengths
    assert min(seen_lengths) == 251
    assert max(seen_lengths) == 269



    