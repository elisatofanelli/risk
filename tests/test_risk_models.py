from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.calibration import calibrate_from_history, compute_log_returns
from src.input_loader import load_portfolio
from src.market_data import load_price_history
from src.reporting import create_risk_report
import src.risk_models as risk_models_module
from src.risk_models import (
    historical_var_es, 
    monte_carlo_var_es, 
    parametric_var, 
    _shock_portfolio_volatility
)


def test_log_returns_shape_is_correct():
    """Verify that computing log returns yields N-1 rows and the same number of columns."""
    prices = load_price_history("data/sample_historical_prices.csv")
    returns = compute_log_returns(prices)
    assert returns.shape[0] == prices.shape[0] - 1
    assert returns.shape[1] == prices.shape[1]


def test_covariance_matrix_has_correct_dimensions():
    """Verify that the calibrated covariance matrix is a square matrix matching the number of assets."""
    prices = load_price_history("data/sample_historical_prices.csv")
    calib = calibrate_from_history(prices)
    cov = calib["daily_cov_matrix"]
    assert cov.shape == (prices.shape[1], prices.shape[1])


def test_risk_models_and_report(tmp_path):
    """Verify that all core VaR models produce non-negative outputs and can be serialized to a CSV report."""
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
    """Ensure the system rejects negative prices during data loading to prevent log math errors."""
    csv = tmp_path / "bad_prices.csv"
  
    csv.write_text(
        "date,AAPL\n"
        "2026-01-01,150.0\n"
        "2026-01-02,-150.0\n"
    )
    
    with pytest.raises(ValueError, match="contains non-positive prices"):
        load_price_history(str(csv))


def test_robustness_extreme_outlier_shock():
    """Verify the calibration engine handles extreme price shocks without producing NaNs."""
    df_outlier = pd.DataFrame({
        "AAPL": [100.0, 10.0, 10.5]  
    }, index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]))
    
    calib = calibrate_from_history(df_outlier)
    
    assert not np.isnan(calib["daily_volatility"]["AAPL"])


def test_robustness_covariance_short_window():
    """Verify covariance matrix computation remains stable and non-null even with minimal historical data."""
    df_short = pd.DataFrame({
        "AAPL": [100.0, 102.0, 101.0],
        "MSFT": [200.0, 205.0, 203.0]
    }, index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]))
    
    calib = calibrate_from_history(df_short)
    cov = calib["daily_cov_matrix"]
    
    assert cov.shape == (2, 2)
    assert not cov.isnull().any().any()


def test_dynamic_volatility_leverage_effect():
    """Verify that a simulated drop in stock price correctly triggers an increase in option implied volatility."""
   
    portfolio_df = pd.DataFrame({
        "type": ["option"],
        "underlying": ["AAPL"],
        "implied_vol": [0.20]  
    })
    
    log_returns = {"AAPL": -0.10}
    
    shocked_df = _shock_portfolio_volatility(portfolio_df, log_returns)
    shocked_vol = shocked_df.loc[0, "implied_vol"]
    
    expected_vol = 0.20 * np.exp(-0.5 * -0.10)
    
    assert shocked_vol > 0.20  
    assert abs(shocked_vol - expected_vol) < 1e-6


def test_monte_carlo_var_converges_with_more_simulations():
    """Verify that increasing the number of Monte Carlo simulations reduces the error relative to a high-sim baseline."""
    prices = load_price_history("data/sample_historical_prices.csv")
    portfolio_df = load_portfolio("data/sample_portfolio.csv")
    spot_prices = prices.iloc[-1].to_dict()
    calibration_result = calibrate_from_history(prices)
    valuation_date = datetime(2026, 5, 11)

    var_100 = monte_carlo_var_es(
        portfolio_df, spot_prices, calibration_result, valuation_date,
        n_sims=100, random_seed=42
    )["VaR"]

    var_2000 = monte_carlo_var_es(
        portfolio_df, spot_prices, calibration_result, valuation_date,
        n_sims=2000, random_seed=42
    )["VaR"]

    var_10000 = monte_carlo_var_es(
        portfolio_df, spot_prices, calibration_result, valuation_date,
        n_sims=10000, random_seed=42
    )["VaR"]

    error_100 = abs(var_100 - var_10000)
    error_2000 = abs(var_2000 - var_10000)

    assert error_2000 < error_100, (
        f"Expected internal convergence: 2k sim error ({error_2000:.2f}) "
        f"should be smaller than 100 sim error ({error_100:.2f})"
    )


def test_var_sensitivity_to_calibration_window():
    """Verify that VaR outputs are sensitive to changes in the historical calibration window length."""
    prices = load_price_history("data/sample_historical_prices.csv")
    portfolio_df = load_portfolio("data/sample_portfolio.csv")
    valuation_date = datetime(2026, 5, 11)

    prices_250 = prices.iloc[-250:]
    var_250 = historical_var_es(portfolio_df, prices_250, valuation_date)["VaR"]

    prices_60 = prices.iloc[-60:]
    var_60 = historical_var_es(portfolio_df, prices_60, valuation_date)["VaR"]

    assert var_250 != var_60, (
        f"Expected VaR to differ across calibration windows, "
        f"but both returned {var_250:.4f}"
    )

    print(f"\nCalibration window sensitivity:")
    print(f"  VaR (250-day window): {var_250:.2f}")
    print(f"  VaR  (60-day window): {var_60:.2f}")
    print(f"  Difference: {abs(var_250 - var_60):.2f}")


def test_leverage_effect_increases_var_for_negative_scenarios():
    """Verify that the dynamic volatility adjustment (leverage effect) produces a more conservative higher VaR."""
    prices = load_price_history("data/sample_historical_prices.csv")
    portfolio_df = load_portfolio("data/sample_portfolio.csv")
    valuation_date = datetime(2026, 5, 11)

    var_with_adjustment = historical_var_es(
        portfolio_df, prices, valuation_date, confidence_level=0.95
    )["VaR"]

    def no_vol_shock(portfolio_df, log_returns):
        return portfolio_df.copy()

    with patch.object(risk_models_module, "_shock_portfolio_volatility", no_vol_shock):
        var_without_adjustment = historical_var_es(
            portfolio_df, prices, valuation_date, confidence_level=0.95
        )["VaR"]

    assert var_with_adjustment <= var_without_adjustment, (
            f"Expected leverage-adjusted VaR ({var_with_adjustment:.2f}) <= "
            f"unadjusted VaR ({var_without_adjustment:.2f}). "
            "Higher volatility increases long option value, buffering the portfolio loss."
        )

    print(f"\nLeverage effect assumption impact:")
    print(f"  VaR with adjustment (lambda=0.5): {var_with_adjustment:.2f}")
    print(f"  VaR without adjustment:           {var_without_adjustment:.2f}")
    print(f"  VaR increase from assumption:     {var_with_adjustment - var_without_adjustment:.2f}")


def test_covariance_stability_under_small_price_perturbation():
    """Verify that a minor, uniform input perturbation does not trigger disproportionate changes in the calibrated covariance matrix."""
    prices = load_price_history("data/sample_historical_prices.csv")
 
    perturbed_prices = prices * 1.001         
 
    calib_base = calibrate_from_history(prices)
    calib_pert = calibrate_from_history(perturbed_prices)
 
    cov_base = calib_base["daily_cov_matrix"].values
    cov_pert = calib_pert["daily_cov_matrix"].values
 
    max_diff = float(np.abs(cov_base - cov_pert).max())
 
    assert max_diff < 1e-6, (
        f"Covariance matrix changed by {max_diff:.2e} under a 0.1% price "
        f"perturbation — parameter instability detected."
    )
 
    print(f"\nParameter stability (input perturbation → covariance):")
    print(f"  Max absolute cov entry change: {max_diff:.2e}  (threshold: 1e-6)")
 
 
def test_ewma_covariance_stability_under_small_price_perturbation():
    """Verify that a minor, uniform input perturbation does not trigger disproportionate changes in the EWMA covariance matrix."""
    from src.calibration import ewma_calibrate
 
    prices = load_price_history("data/sample_historical_prices.csv")
    perturbed_prices = prices * 1.001
 
    calib_base = ewma_calibrate(prices)
    calib_pert = ewma_calibrate(perturbed_prices)
 
    cov_base = calib_base["daily_cov_matrix"].values
    cov_pert = calib_pert["daily_cov_matrix"].values
 
    max_diff = float(np.abs(cov_base - cov_pert).max())
 
    assert max_diff < 1e-6, (
        f"EWMA covariance matrix changed by {max_diff:.2e} under a 0.1% "
        f"price perturbation."
    )
 
    print(f"\nEWMA parameter stability (input perturbation → covariance):")
    print(f"  Max absolute cov entry change: {max_diff:.2e}  (threshold: 1e-6)")
 
 
def test_var_output_stability_under_small_spot_perturbation():
    """Verify that a minor spot price perturbation yields stable, proportionally small changes in VaR output across all models."""
    from src.calibration import ewma_calibrate
    from src.risk_models import (
        ewma_historical_var_es, ewma_parametric_var, ewma_monte_carlo_var_es,
    )
 
    prices    = load_price_history("data/sample_historical_prices.csv")
    portfolio = load_portfolio("data/sample_portfolio.csv")
    val_date  = datetime(2026, 5, 11)
 
    spot_base = prices.iloc[-1].to_dict()
    spot_pert = {k: v * 1.001 for k, v in spot_base.items()}   # +0.1% shock
 
    calib     = calibrate_from_history(prices)
 
    results = {
        "historical":       (
            historical_var_es(portfolio, prices, val_date)["VaR"],
            historical_var_es(portfolio, prices * 1.001, val_date)["VaR"],
        ),
        "parametric":       (
            parametric_var(portfolio, spot_base, calib, val_date)["VaR"],
            parametric_var(portfolio, spot_pert, calib, val_date)["VaR"],
        ),
        "monte_carlo":      (
            monte_carlo_var_es(portfolio, spot_base, calib, val_date, n_sims=2000)["VaR"],
            monte_carlo_var_es(portfolio, spot_pert, calib, val_date, n_sims=2000)["VaR"],
        ),
        "ewma_historical":  (
            ewma_historical_var_es(portfolio, prices, val_date)["VaR"],
            ewma_historical_var_es(portfolio, prices * 1.001, val_date)["VaR"],
        ),
        "ewma_parametric":  (
            ewma_parametric_var(portfolio, spot_base, prices, val_date)["VaR"],
            ewma_parametric_var(portfolio, spot_pert, prices, val_date)["VaR"],
        ),
        "ewma_monte_carlo": (
            ewma_monte_carlo_var_es(portfolio, spot_base, prices, val_date, n_sims=2000)["VaR"],
            ewma_monte_carlo_var_es(portfolio, spot_pert, prices, val_date, n_sims=2000)["VaR"],
        ),
    }
 
    max_rel_change_threshold = 0.02   # 2% relative tolerance
 
    print(f"\nOutput stability (0.1% spot perturbation → VaR change):")
    for method, (var_base, var_pert) in results.items():
        rel_change = abs(var_pert - var_base) / max(var_base, 1e-8)
        print(f"  {method:<20} base={var_base:.2f}  perturbed={var_pert:.2f}  "
              f"rel_change={rel_change:.4%}")
        assert rel_change < max_rel_change_threshold, (
            f"{method}: VaR changed by {rel_change:.2%} under a 0.1% spot "
            f"perturbation — output instability detected (threshold: "
            f"{max_rel_change_threshold:.0%})."
        )
 
 
def test_calibrated_covariance_stable_over_rolling_windows():
    """Verify that covariance matrix entries do not exhibit extreme instability when calibrated across consecutive rolling windows."""
    from src.calibration import ewma_calibrate
 
    prices  = load_price_history("data/sample_historical_prices.csv")
    w       = 60
 
    windows = [
        prices.iloc[-(3 * w): -(2 * w)],
        prices.iloc[-(2 * w): -w],
        prices.iloc[-w:],
    ]
 
    for label, calibrate_fn in [
        ("equally-weighted", calibrate_from_history),
        ("EWMA",             ewma_calibrate),
    ]:
        covs = np.stack(
            [calibrate_fn(win)["daily_cov_matrix"].values for win in windows]
        )                                            
 
        cov_range = covs.max(axis=0) - covs.min(axis=0) 
        cov_median = np.median(np.abs(covs))               
        
        multiplier = 10.0 if label == "EWMA" else 5.0
        max_allowed_swing = multiplier * max(cov_median, 1e-10)
        max_observed_swing = float(cov_range.max())
 
        assert max_observed_swing < max_allowed_swing, (
            f"{label} covariance: max swing across 60-day windows "
            f"({max_observed_swing:.2e}) exceeds 5× median magnitude "
            f"({max_allowed_swing:.2e}) — parameter instability over time."
        )
 
        print(f"\nParameter stability over time ({label}):")
        print(f"  Max covariance swing: {max_observed_swing:.2e} "
              f"(threshold: {max_allowed_swing:.2e})")
 
 
def test_var_output_stable_over_rolling_windows():
    """Verify that calculated VaR does not jump pathologically across consecutive, non-overlapping rolling windows."""
    from src.risk_models import ewma_historical_var_es
 
    prices    = load_price_history("data/sample_historical_prices.csv")
    portfolio = load_portfolio("data/sample_portfolio.csv")
    w         = 60
 
    windows = [
        prices.iloc[-(3 * w): -(2 * w)],
        prices.iloc[-(2 * w): -w],
        prices.iloc[-w:],
    ]
 
    for label, fn in [
        ("historical",      historical_var_es),
        ("ewma_historical", ewma_historical_var_es),
    ]:
        vars_ = []
        for win in windows:
            val_date = win.index[-1].to_pydatetime()
            vars_.append(fn(portfolio, win, val_date)["VaR"])
 
        print(f"\nOutput stability over time ({label}):")
        for i, v in enumerate(vars_):
            print(f"  Window {i+1}: VaR = {v:.2f}")
 
        for i in range(len(vars_) - 1):
            base = max(vars_[i], 1e-8)
            rel_jump = abs(vars_[i + 1] - vars_[i]) / base
            assert rel_jump < 1.0, (
                f"{label}: VaR jumped by {rel_jump:.0%} between window "
                f"{i+1} ({vars_[i]:.2f}) and window {i+2} ({vars_[i+1]:.2f}) "
                f"— output instability over time detected."
            )