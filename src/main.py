"""Entry point for portfolio valuation and risk model demo."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from .backtesting import backtest_var, summarize_backtest
    from .calibration import calibrate_from_history, load_user_params
    from .input_loader import load_portfolio
    from .market_data import load_price_history
    from .portfolio import portfolio_delta_exposures, value_portfolio
    from .reporting import create_risk_report, save_backtest_results, save_backtest_summary
    from .risk_models import (
        historical_var_es, monte_carlo_var_es, parametric_var,
        ewma_historical_var_es, ewma_parametric_var, ewma_monte_carlo_var_es,
    )
except ImportError:
    from src.backtesting import backtest_var, summarize_backtest
    from src.calibration import calibrate_from_history, load_user_params
    from src.input_loader import load_portfolio
    from src.market_data import load_price_history
    from src.portfolio import portfolio_delta_exposures, value_portfolio
    from src.reporting import create_risk_report, save_backtest_results, save_backtest_summary
    from src.risk_models import (
        historical_var_es, monte_carlo_var_es, parametric_var,
        ewma_historical_var_es, ewma_parametric_var, ewma_monte_carlo_var_es,
    )


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

    hist_result  = historical_var_es(portfolio_df, price_history_df, valuation_date)
    mc_result    = monte_carlo_var_es(portfolio_df, spot_prices, calibration_result, valuation_date)
    param_result = parametric_var(portfolio_df, spot_prices, calibration_result, valuation_date)

    ewma_hist_result  = ewma_historical_var_es(portfolio_df, price_history_df, valuation_date)
    ewma_param_result = ewma_parametric_var(portfolio_df, spot_prices, price_history_df, valuation_date)
    ewma_mc_result    = ewma_monte_carlo_var_es(portfolio_df, spot_prices, price_history_df, valuation_date)

    user_mean_path = project_root / "data" / "user_params_mean.csv"
    user_cov_path  = project_root / "data" / "user_params_cov.csv"

    all_results = {
        "historical":       hist_result,
        "monte_carlo":      mc_result,
        "parametric":       param_result,
        "ewma_historical":  ewma_hist_result,
        "ewma_parametric":  ewma_param_result,
        "ewma_monte_carlo": ewma_mc_result,
    }

    mc_user = param_user = None
    if user_mean_path.exists() and user_cov_path.exists():
        user_params = load_user_params(str(user_mean_path), str(user_cov_path))

        mc_user    = monte_carlo_var_es(portfolio_df, spot_prices, user_params, valuation_date)
        param_user = parametric_var(portfolio_df, spot_prices, user_params, valuation_date)

        all_results["mc_user_params"]    = mc_user
        all_results["param_user_params"] = param_user

        create_risk_report(
            {"mc_user_params": mc_user, "param_user_params": param_user},
            str(project_root / "outputs" / "risk_report_user_params.csv"),
        )

    report_path = project_root / "outputs" / "risk_report.csv"
    create_risk_report(all_results, str(report_path))

    bt_hist       = backtest_var(portfolio_df, price_history_df, "historical",      calibration_window=250)
    bt_param      = backtest_var(portfolio_df, price_history_df, "parametric",      calibration_window=250)
    bt_mc         = backtest_var(portfolio_df, price_history_df, "monte_carlo",     calibration_window=250, n_sims=1000)
    bt_ewma_hist  = backtest_var(portfolio_df, price_history_df, "ewma_historical", calibration_window=250)
    bt_ewma_param = backtest_var(portfolio_df, price_history_df, "ewma_parametric", calibration_window=250)
    bt_ewma_mc    = backtest_var(portfolio_df, price_history_df, "ewma_monte_carlo",calibration_window=250, n_sims=1000)

    save_backtest_results(bt_hist,       str(project_root / "outputs" / "backtest_historical.csv"))
    save_backtest_results(bt_param,      str(project_root / "outputs" / "backtest_parametric.csv"))
    save_backtest_results(bt_mc,         str(project_root / "outputs" / "backtest_monte_carlo.csv"))
    save_backtest_results(bt_ewma_hist,  str(project_root / "outputs" / "backtest_ewma_historical.csv"))
    save_backtest_results(bt_ewma_param, str(project_root / "outputs" / "backtest_ewma_parametric.csv"))
    save_backtest_results(bt_ewma_mc,    str(project_root / "outputs" / "backtest_ewma_monte_carlo.csv"))

    summary_df = save_backtest_summary(
        [
            summarize_backtest(bt_hist),
            summarize_backtest(bt_param),
            summarize_backtest(bt_mc),
            summarize_backtest(bt_ewma_hist),
            summarize_backtest(bt_ewma_param),
            summarize_backtest(bt_ewma_mc),
        ],
        str(project_root / "outputs" / "backtest_summary.csv"),
    )

    print(f"\nPortfolio value: {portfolio_value:.2f}")
    print("Delta exposures:")
    for underlying, delta in deltas.items():
        print(f"  {underlying}: {delta:.4f}")

    print(f"\n--- Equally-weighted models ---")
    print(f"Historical      VaR: {hist_result['VaR']:.2f},  ES: {hist_result['ES']:.2f}")
    print(f"Monte Carlo     VaR: {mc_result['VaR']:.2f},  ES: {mc_result['ES']:.2f}")
    print(f"Parametric      VaR: {param_result['VaR']:.2f},  ES: {param_user['ES']}")

    print(f"\n--- EWMA models (λ=0.94) ---")
    print(f"EWMA Historical VaR: {ewma_hist_result['VaR']:.2f},  ES: {ewma_hist_result['ES']:.2f}")
    print(f"EWMA Monte Carlo VaR: {ewma_mc_result['VaR']:.2f},  ES: {ewma_mc_result['ES']:.2f}")
    print(f"EWMA Parametric VaR: {ewma_param_result['VaR']:.2f},  ES: {param_user['ES']}")

    if mc_user and param_user:
        print(f"\n--- User-supplied parameter models ---")
        print(f"Monte Carlo (user) VaR: {mc_user['VaR']:.2f},  ES: {mc_user['ES']:.2f}")
        print(f"Parametric  (user) VaR: {param_user['VaR']:.2f},  ES: {param_user['ES']}")
        print(f"  (user params: {user_mean_path.name}, {user_cov_path.name})")

    print(f"\nRisk report saved to: {report_path}")
    print("\nBacktest summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
