from math import exp

from src.pricing import black_scholes_price


def test_black_scholes_call_put_positive():
    call = black_scholes_price(100, 100, 1.0, 0.2, 0.05, "call")
    put = black_scholes_price(100, 100, 1.0, 0.2, 0.05, "put")
    assert call > 0
    assert put > 0


def test_put_call_parity_approximately_holds():
    s = 100
    k = 100
    T = 1.0
    sigma = 0.2
    r = 0.05
    call = black_scholes_price(s, k, T, sigma, r, "call")
    put = black_scholes_price(s, k, T, sigma, r, "put")
    lhs = call - put
    rhs = s - k * exp(-r * T)
    assert abs(lhs - rhs) < 1e-2


def test_call_goes_to_intrinsic_as_vol_approaches_zero():
    """Price goes to intrinsic value as vol → 0"""
    spot, strike, r, T = 150.0, 100.0, 0.05, 1.0
    price = black_scholes_price(spot, strike, T, 0.0001, r, "call")
    intrinsic = spot - strike * exp(-r * T)
    assert abs(price - intrinsic) < 0.5

def test_call_price_approaches_spot_as_strike_goes_to_zero():
    """Call price → spot as strike → 0"""
    price = black_scholes_price(100.0, 0.001, 1.0, 0.2, 0.05, "call")
    assert abs(price - 100.0) < 1.0