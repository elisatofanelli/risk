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
