"""European option pricing utilities."""

from __future__ import annotations

from math import exp, log, sqrt

from scipy.stats import norm


def _validate_inputs(spot: float, strike: float, T: float, sigma: float, r: float) -> None:
    if spot <= 0:
        raise ValueError("Spot price must be positive.")
    if strike <= 0:
        raise ValueError("Strike price must be positive.")
    if T < 0:
        raise ValueError("Time to maturity cannot be negative.")
    if sigma <= 0 and T > 0:
        raise ValueError("Volatility must be positive when time to maturity is greater than zero.")


def _intrinsic_value(spot: float, strike: float, option_type: str) -> float:
    if option_type == "call":
        return max(spot - strike, 0.0)
    if option_type == "put":
        return max(strike - spot, 0.0)
    raise ValueError("option_type must be 'call' or 'put'.")


def black_scholes_price(spot: float, strike: float, T: float, sigma: float, r: float, option_type: str) -> float:
    """Price a European option using the Black-Scholes formula."""

    option_type = option_type.lower().strip()
    _validate_inputs(spot, strike, T, sigma, r)

    if T <= 0:
        return _intrinsic_value(spot, strike, option_type)

    if sigma <= 0:
        raise ValueError("Volatility must be positive when time to maturity is greater than zero.")

    sqrt_T = sqrt(T)
    d1 = (log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        return spot * norm.cdf(d1) - strike * exp(-r * T) * norm.cdf(d2)
    if option_type == "put":
        return strike * exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'.")


def black_scholes_delta(spot: float, strike: float, T: float, sigma: float, r: float, option_type: str) -> float:
    """Delta of a European option using the Black-Scholes formula."""

    option_type = option_type.lower().strip()
    _validate_inputs(spot, strike, T, sigma, r)

    if T <= 0:
        if option_type == "call":
            return 1.0 if spot > strike else 0.0
        if option_type == "put":
            return -1.0 if spot < strike else 0.0
        raise ValueError("option_type must be 'call' or 'put'.")

    if sigma <= 0:
        raise ValueError("Volatility must be positive when time to maturity is greater than zero.")

    d1 = (log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    if option_type == "put":
        return norm.cdf(d1) - 1.0
    raise ValueError("option_type must be 'call' or 'put'.")

def black_scholes_gamma(spot: float, strike: float, T: float, sigma: float, r: float, option_type: str = "call") -> float:
    option_type = option_type.lower().strip()
    _validate_inputs(spot, strike, T, sigma, r)

    if T <= 0:
        return 0.0

    if sigma <= 0:
        raise ValueError("Volatility must be positive when time to maturity is greater than zero.")

    d1 = (log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    
    return norm.pdf(d1) / (spot * sigma * sqrt(T))


def implied_volatility(target_price: float, spot: float, strike: float, T: float, r: float, option_type: str, tol: float = 1e-6, max_iter: int = 100) -> float:
    option_type = option_type.lower().strip()
    
    intrinsic = _intrinsic_value(spot, strike, option_type)
    if target_price < intrinsic:
        raise ValueError("Target price is below intrinsic value; implied volatility is undefined.")
        
    low_vol = 1e-5
    high_vol = 5.0
    
    for i in range(max_iter):
        mid_vol = (low_vol + high_vol) / 2.0
        price = black_scholes_price(spot, strike, T, mid_vol, r, option_type)
        
        diff = price - target_price
        if abs(diff) < tol:
            return mid_vol
            
        if price < target_price:
            low_vol = mid_vol
        else:
            high_vol = mid_vol
            
    return (low_vol + high_vol) / 2.0