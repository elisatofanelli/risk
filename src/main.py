"""Entry point for portfolio valuation and risk model demo."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # pragma: no cover - supports both module and script execution
    from .backtesting import backtest_var, summarize_backtest
    from .calibration import calibrate_from_history
    from .input_loader import load_portfolio
    from .market_data import load_price_history
    from .portfolio import portfolio_delta_exposures, value_portfolio
    from .reporting import create_risk_report, save_backtest_results, save_backtest_summary
    from .risk_models import historical_var_es, monte_carlo_var_es, parametric_var
except ImportError:  # pragma: no cover - script fallback
    from src.backtesting import backtest_var, summarize_backtest
    from src.calibration import calibrate_from_history
    from src.input_loader import load_portfolio
    from src.market_data import load_price_history
    from src.portfolio import portfolio_delta_exposures, value_portfolio
    from src.reporting import create_risk_report, save_backtest_results, save_backtest_summary
    from src.risk_models import historical_var_es, monte_carlo_var_es, parametric_var


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    portfolio_path = project_root / "data" / "sample_portfolio.csv"
    history_path = project_root / "data" / "sample_historical_prices.csv"
    portfolio_df = load_portfolio(str(portfolio_path))
    price_history_df = load_price_history(str(history_path))

    valuation_date = datetime(2026, 5, 11)
    spot_prices = price_history_df.iloc[-1].to_dict()
    calibration_result = calibrate_from_history(price_history_df)

    portfolio_value = value_portfolio(portfolio_df, spot_prices, valuation_date)
    deltas = portfolio_delta_exposures(portfolio_df, spot_prices, valuation_date)
    hist_result = historical_var_es(portfolio_df, price_history_df, valuation_date)
    mc_result = monte_carlo_var_es(portfolio_df, spot_prices, calibration_result, valuation_date)
    param_result = parametric_var(portfolio_df, spot_prices, calibration_result, valuation_date)
    report_path = project_root / "outputs" / "risk_report.csv"
    create_risk_report(
        {
            "historical": hist_result,
            "monte_carlo": mc_result,
            "parametric": param_result,
        },
        str(report_path),
    )

    bt_hist = backtest_var(portfolio_df, price_history_df, "historical")
    bt_param = backtest_var(portfolio_df, price_history_df, "parametric")
    bt_mc = backtest_var(portfolio_df, price_history_df, "monte_carlo")

    save_backtest_results(bt_hist, str(project_root / "outputs" / "backtest_historical.csv"))
    save_backtest_results(bt_param, str(project_root / "outputs" / "backtest_parametric.csv"))
    save_backtest_results(bt_mc, str(project_root / "outputs" / "backtest_monte_carlo.csv"))

    summary_df = save_backtest_summary(
        [
            summarize_backtest(bt_hist),
            summarize_backtest(bt_param),
            summarize_backtest(bt_mc),
        ],
        str(project_root / "outputs" / "backtest_summary.csv"),
    )

    print(f"Portfolio value: {portfolio_value:.2f}")
    print("Delta exposures:")
    for underlying, delta in deltas.items():
        print(f"  {underlying}: {delta:.4f}")
    print(f"Historical VaR: {hist_result['VaR']:.2f}, ES: {hist_result['ES']:.2f}")
    print(f"Monte Carlo VaR: {mc_result['VaR']:.2f}, ES: {mc_result['ES']:.2f}")
    print(f"Parametric VaR: {param_result['VaR']:.2f}")
    print(f"Risk report saved to: {report_path}")
    print("Backtest summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
