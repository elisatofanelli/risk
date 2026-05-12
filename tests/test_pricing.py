from math import exp
import pytest

from src.pricing import black_scholes_price, black_scholes_delta


def test_black_scholes_call_put_positive():
    call = black_scholes_price(100, 100, 1.0, 0.2, 0.05, "call")
    put = black_scholes_price(100, 100, 1.0, 0.2, 0.05, "put")
    assert call > 0
    assert put > 0


def test_put_call_parity_approximately_holds():
    s = 100.0
    k = 100.0
    T = 1.0
    sigma = 0.2
    r = 0.05
    call = black_scholes_price(s, k, T, sigma, r, "call")
    put = black_scholes_price(s, k, T, sigma, r, "put")
    lhs = call - put
    rhs = s - k * exp(-r * T)
    assert abs(lhs - rhs) < 1e-2


def test_call_goes_to_intrinsic_as_vol_approaches_zero():
    """Price goes to intrinsic value as vol approaches 0."""
    spot, strike, r, T = 150.0, 100.0, 0.05, 1.0
    price = black_scholes_price(spot, strike, T, 0.0001, r, "call")
    intrinsic = spot - strike * exp(-r * T)
    assert abs(price - intrinsic) < 1e-2


def test_time_to_maturity_zero():
    """If T=0, the option must evaluate exactly to its intrinsic value."""
    S = 100.0; K = 90.0; T = 0.0; r = 0.05; sigma = 0.2
    call_price = black_scholes_price(S, K, T, sigma, r, "call")
    assert call_price == 10.0  # 100 - 90
    
    # If the Call is OTM at T=0, it is worth 0
    call_otm = black_scholes_price(90.0, 100.0, 0.0, sigma, r, "call")
    assert call_otm == 0.0


def test_delta_boundaries():
    """Test the mathematical limits of the Delta."""
    delta_call = black_scholes_delta(100.0, 100.0, 1.0, 0.2, 0.05, "call")
    delta_put = black_scholes_delta(100.0, 100.0, 1.0, 0.2, 0.05, "put")
    
    assert 0.0 < delta_call < 1.0
    assert -1.0 < delta_put < 0.0


def test_volatility_zero_raises_error():
    """Special Case Test: Zero volatility (with T>0) should raise an error."""
    with pytest.raises(ValueError, match="Volatility must be positive"):
        # Passing sigma = 0.0
        black_scholes_price(100.0, 100.0, 1.0, 0.0, 0.05, "call")


def test_strike_zero_raises_error():
    """Special Case Test: Zero or negative strike is physically impossible."""
    with pytest.raises(ValueError, match="Strike price must be positive"):
        # Passing strike = 0.0
        black_scholes_price(100.0, 0.0, 1.0, 0.2, 0.05, "put")