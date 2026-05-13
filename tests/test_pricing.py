from math import exp
import pytest
import math
from src.pricing import black_scholes_price, black_scholes_delta
from src.pricing import black_scholes_price, black_scholes_delta, black_scholes_gamma, implied_volatility


def test_black_scholes_call_put_positive():
    """Verify that valid Black-Scholes calls and puts always return positive prices."""
    call = black_scholes_price(100, 100, 1.0, 0.2, 0.05, "call")
    put = black_scholes_price(100, 100, 1.0, 0.2, 0.05, "put")
    assert call > 0
    assert put > 0


def test_put_call_parity_approximately_holds():
    """Ensure the computed call and put prices satisfy put-call parity."""
    s = 100.0
    k = 100.0
    T = 1.0
    sigma = 0.2
    r = 0.05
    call = black_scholes_price(s, k, T, sigma, r, "call")
    put = black_scholes_price(s, k, T, sigma, r, "put")
    lhs = call - put
    rhs = s - k * exp(-r * T)
    assert abs(lhs - rhs) < 1e-8


def test_call_goes_to_intrinsic_as_vol_approaches_zero():
    """Confirm that as volatility approaches zero, the option price converges to its intrinsic value."""
    spot, strike, r, T = 150.0, 100.0, 0.05, 1.0
    price = black_scholes_price(spot, strike, T, 0.0001, r, "call")
    intrinsic = spot - strike * exp(-r * T)
    assert abs(price - intrinsic) < 1e-2


def test_time_to_maturity_zero():
    """Verify that when time to maturity is zero, the option evaluates to its intrinsic payoff."""
    S = 100.0; K = 90.0; T = 0.0; r = 0.05; sigma = 0.2
    call_price = black_scholes_price(S, K, T, sigma, r, "call")
    assert call_price == 10.0  # 100 - 90
    
    call_otm = black_scholes_price(90.0, 100.0, 0.0, sigma, r, "call")
    assert call_otm == 0.0


def test_delta_boundaries():
    """Check that call Delta is in (0, 1) and put Delta is in (-1, 0)."""
    delta_call = black_scholes_delta(100.0, 100.0, 1.0, 0.2, 0.05, "call")
    delta_put = black_scholes_delta(100.0, 100.0, 1.0, 0.2, 0.05, "put")
    
    assert 0.0 < delta_call < 1.0
    assert -1.0 < delta_put < 0.0


def test_volatility_zero_raises_error():
    """Ensure the model explicitly rejects zero volatility inputs for non-expired options."""
    with pytest.raises(ValueError, match="Volatility must be positive"):
        black_scholes_price(100.0, 100.0, 1.0, 0.0, 0.05, "call")


def test_strike_zero_raises_error():
    """Ensure the model rejects zero or negative strike prices."""
    with pytest.raises(ValueError, match="Strike price must be positive"):
        black_scholes_price(100.0, 0.0, 1.0, 0.2, 0.05, "put")


def test_delta_gamma_approximation_vs_full_repricing():
    """Verify that the Delta-Gamma approximation closely matches the exact Black-Scholes repricing for a small spot price move."""
    spot = 100.0
    strike = 100.0
    T = 1.0
    sigma = 0.20
    r = 0.05
    option_type = "call"
    
    shock_pct = 0.01 
    spot_change = spot * shock_pct
    new_spot = spot + spot_change

    original_price = black_scholes_price(spot, strike, T, sigma, r, option_type)
    exact_new_price = black_scholes_price(new_spot, strike, T, sigma, r, option_type)
    
    delta = black_scholes_delta(spot, strike, T, sigma, r, option_type)
    gamma = black_scholes_gamma(spot, strike, T, sigma, r, option_type) 
    
    approx_new_price = original_price + (delta * spot_change) + (0.5 * gamma * (spot_change ** 2))
    
    approximation_error = abs(exact_new_price - approx_new_price)
    assert approximation_error < 5e-3, (
        f"Delta-Gamma approx ({approx_new_price:.4f}) deviates too much from "
        f"exact price ({exact_new_price:.4f}). Error: {approximation_error:.6f}"
    )


def test_implied_volatility_round_trip():
    """Test the invertibility of the pricing model by verifying that an implied volatility solver recovers the input volatility."""
    spot = 100.0
    strike = 105.0 
    T = 1.0
    r = 0.05
    input_vol = 0.25  
    option_type = "call"

    generated_price = black_scholes_price(spot, strike, T, input_vol, r, option_type)
    
    calculated_vol = implied_volatility(generated_price, spot, strike, T, r, option_type)
    
    assert abs(calculated_vol - input_vol) < 1e-4, (
        f"Implied volatility solver failed round-trip. "
        f"Input vol: {input_vol:.4f}, Calculated vol: {calculated_vol:.4f}"
    )


class TestMonotonicity:
    """Verify monotonic relationships of option prices with respect to key inputs."""
 
    def test_call_price_increases_with_spot(self):
        """Verify that a call option's value increases as the underlying spot price rises."""
        strike = 100.0
        T = 1.0
        sigma = 0.25
        r = 0.05
 
        spots = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
        prices = [black_scholes_price(s, strike, T, sigma, r, "call") for s in spots]
 
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1], (
                f"Call price not monotonically increasing in spot: "
                f"price({spots[i]}) = {prices[i]:.4f} >= "
                f"price({spots[i+1]}) = {prices[i+1]:.4f}"
            )
 
    def test_put_price_decreases_with_spot(self):
        """Verify that a put option's value decreases as the underlying spot price rises."""
        strike = 100.0
        T = 1.0
        sigma = 0.25
        r = 0.05
 
        spots = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
        prices = [black_scholes_price(s, strike, T, sigma, r, "put") for s in spots]
 
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1], (
                f"Put price not monotonically decreasing in spot: "
                f"price({spots[i]}) = {prices[i]:.4f} <= "
                f"price({spots[i+1]}) = {prices[i+1]:.4f}"
            )
 
    def test_call_price_increases_with_volatility(self):
        """Verify that a call option's value increases as volatility rises."""
        strike = 100.0
        T = 1.0
        r = 0.05
 
        vols = [0.05, 0.10, 0.20, 0.30, 0.40, 0.60]
        prices = [black_scholes_price(100.0, strike, T, v, r, "call") for v in vols]
 
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1], (
                f"Call price not increasing in vol: "
                f"price(vol={vols[i]}) = {prices[i]:.4f} >= "
                f"price(vol={vols[i+1]}) = {prices[i+1]:.4f}"
            )
 
    def test_call_price_decreases_with_strike(self):
        """Verify that a call option's value decreases as the strike price increases."""
        spot = 100.0
        T = 1.0
        sigma = 0.25
        r = 0.05
 
        strikes = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
        prices = [black_scholes_price(spot, k, T, sigma, r, "call") for k in strikes]
 
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1], (
                f"Call price not decreasing in strike: "
                f"price(K={strikes[i]}) = {prices[i]:.4f} <= "
                f"price(K={strikes[i+1]}) = {prices[i+1]:.4f}"
            )


class TestConvexity:
    """Verify the convexity of Black-Scholes option prices with respect to spot and strike."""
 
    def test_call_price_is_convex_in_spot(self):
        """Verify that the call price is convex with respect to the spot price."""
        strike = 100.0
        T = 1.0
        sigma = 0.25
        r = 0.05
        h = 1.0
 
        for s in [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]:
            p_up   = black_scholes_price(s + h, strike, T, sigma, r, "call")
            p_mid  = black_scholes_price(s,     strike, T, sigma, r, "call")
            p_down = black_scholes_price(s - h, strike, T, sigma, r, "call")
            second_diff = p_up - 2 * p_mid + p_down
 
            assert second_diff > 0, (
                f"Call price not convex at spot={s}: "
                f"finite-diff second derivative = {second_diff:.8f} (expected > 0)"
            )
 
    def test_put_price_is_convex_in_spot(self):
        """Verify that the put price is convex with respect to the spot price."""
        strike = 100.0
        T = 1.0
        sigma = 0.25
        r = 0.05
        h = 1.0
 
        for s in [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]:
            p_up   = black_scholes_price(s + h, strike, T, sigma, r, "put")
            p_mid  = black_scholes_price(s,     strike, T, sigma, r, "put")
            p_down = black_scholes_price(s - h, strike, T, sigma, r, "put")
            second_diff = p_up - 2 * p_mid + p_down
 
            assert second_diff > 0, (
                f"Put price not convex at spot={s}: "
                f"finite-diff second derivative = {second_diff:.8f} (expected > 0)"
            )
 
    def test_call_price_convex_in_strike(self):
        """Verify that the call price is convex with respect to the strike price."""
        spot = 100.0
        T = 1.0
        sigma = 0.25
        r = 0.05
        h = 5.0
 
        for k in [80.0, 90.0, 100.0, 110.0, 120.0]:
            p_up   = black_scholes_price(spot, k + h, T, sigma, r, "call")
            p_mid  = black_scholes_price(spot, k,     T, sigma, r, "call")
            p_down = black_scholes_price(spot, k - h, T, sigma, r, "call")
            second_diff = p_up - 2 * p_mid + p_down
 
            assert second_diff >= 0, (
                f"Call not convex in strike at K={k}: "
                f"finite-diff second derivative = {second_diff:.8f} (expected >= 0)"
            )


class TestDeltaSelfConsistency:
    """Verify that analytic Delta matches numerical finite differences."""
 
    @pytest.mark.parametrize("spot,strike,T,sigma,r,option_type", [
        (100.0, 100.0, 1.0, 0.20, 0.05, "call"),
        (100.0, 100.0, 1.0, 0.20, 0.05, "put"),
        (120.0, 100.0, 1.0, 0.25, 0.03, "call"),
        ( 80.0, 100.0, 0.5, 0.30, 0.02, "call"),
        ( 80.0, 100.0, 0.5, 0.30, 0.02, "put"),
    ])
    def test_delta_equals_finite_difference_derivative(
        self, spot, strike, T, sigma, r, option_type
    ):
        """Verify that the analytic Black-Scholes Delta matches the numerical first derivative with respect to spot."""
        h = 0.01
 
        analytic_delta  = black_scholes_delta(spot, strike, T, sigma, r, option_type)
        p_up            = black_scholes_price(spot + h, strike, T, sigma, r, option_type)
        p_down          = black_scholes_price(spot - h, strike, T, sigma, r, option_type)
        numerical_delta = (p_up - p_down) / (2 * h)
 
        assert abs(analytic_delta - numerical_delta) < 1e-5, (
            f"Delta mismatch for {option_type} S={spot} K={strike} T={T} "
            f"sigma={sigma} r={r}:\n"
            f"  analytic  delta = {analytic_delta:.8f}\n"
            f"  numerical delta = {numerical_delta:.8f}\n"
            f"  difference      = {abs(analytic_delta - numerical_delta):.2e}"
        )


class TestGammaSelfConsistency:
    """Verify that analytic Gamma matches numerical finite differences."""
 
    @pytest.mark.parametrize("spot,strike,T,sigma,r,option_type", [
        (100.0, 100.0, 1.0, 0.20, 0.05, "call"),
        (100.0, 100.0, 1.0, 0.20, 0.05, "put"),
        (120.0, 100.0, 1.0, 0.25, 0.03, "call"),
        ( 80.0, 100.0, 0.5, 0.30, 0.02, "call"),
    ])
    def test_gamma_equals_finite_difference_second_derivative(
        self, spot, strike, T, sigma, r, option_type
    ):
        """Verify that the analytic Black-Scholes Gamma matches the numerical second derivative with respect to spot."""
        h = 1.0
 
        analytic_gamma  = black_scholes_gamma(spot, strike, T, sigma, r, option_type)
        p_up            = black_scholes_price(spot + h, strike, T, sigma, r, option_type)
        p_mid           = black_scholes_price(spot,     strike, T, sigma, r, option_type)
        p_down          = black_scholes_price(spot - h, strike, T, sigma, r, option_type)
        numerical_gamma = (p_up - 2 * p_mid + p_down) / (h ** 2)
 
        assert abs(analytic_gamma - numerical_gamma) < 1e-4, (
            f"Gamma mismatch for {option_type} S={spot} K={strike} T={T} "
            f"sigma={sigma} r={r}:\n"
            f"  analytic  gamma = {analytic_gamma:.8f}\n"
            f"  numerical gamma = {numerical_gamma:.8f}\n"
            f"  difference      = {abs(analytic_gamma - numerical_gamma):.2e}"
        )
 
    def test_gamma_is_positive_everywhere(self):
        """Verify that the analytic Gamma is strictly positive for all valid inputs."""
        strike = 100.0
        T = 1.0
        sigma = 0.25
        r = 0.05
 
        for spot in [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]:
            g = black_scholes_gamma(spot, strike, T, sigma, r, "call")
            assert g > 0, f"Gamma not positive at spot={spot}: gamma={g:.6f}"


class TestFailureModes:
    """Verify the pricer handles extreme but valid inputs safely."""
 
    @pytest.mark.parametrize("spot,strike,T,sigma,r,option_type", [
        (100.0, 100.0, 1.0,   5.0,  0.05, "call"),
        (100.0, 100.0, 1.0,   5.0,  0.05, "put"),
        (100.0, 100.0, 1e-6, 0.25,  0.05, "call"),
        (100.0, 100.0, 1e-6, 0.25,  0.05, "put"),
        (1000.0, 100.0, 1.0, 0.25,  0.05, "call"),
        (  1.0, 100.0, 1.0,  0.25,  0.05, "call"),
        (1e6,   100.0, 1.0,  0.25,  0.05, "call"),
        (100.0, 100.0, 1.0,  0.25, -0.01, "call"),
        (100.0, 100.0, 1.0,  0.25, -0.01, "put"),
    ])
    def test_no_nan_or_inf_on_extreme_inputs(
        self, spot, strike, T, sigma, r, option_type
    ):
        """Verify that extreme but valid inputs produce finite, non-negative prices without errors."""
        price = black_scholes_price(spot, strike, T, sigma, r, option_type)
 
        assert math.isfinite(price), (
            f"Pricer returned non-finite value {price!r} for "
            f"spot={spot} K={strike} T={T} sigma={sigma} r={r} "
            f"type={option_type}"
        )
        assert price >= 0, (
            f"Pricer returned negative price {price:.6f} for "
            f"spot={spot} K={strike} T={T} sigma={sigma} r={r} "
            f"type={option_type}"
        )
 
    def test_delta_finite_on_extreme_inputs(self):
        """Verify that Delta remains finite and bounded within [-1, 1] for extreme valid inputs."""
        cases = [
            (100.0, 100.0, 1e-6, 0.25, 0.05, "call"),
            (1000.0, 100.0, 1.0, 0.25, 0.05, "call"),
            (1.0,   100.0, 1.0, 0.25, 0.05, "call"),
            (100.0, 100.0, 1.0, 5.0,  0.05, "call"),
        ]
        for spot, strike, T, sigma, r, option_type in cases:
            delta = black_scholes_delta(spot, strike, T, sigma, r, option_type)
            assert math.isfinite(delta), (
                f"Delta is not finite ({delta!r}) for "
                f"spot={spot} K={strike} T={T} sigma={sigma}"
            )
            assert -1.0 <= delta <= 1.0, (
                f"Delta out of [-1,1] bounds ({delta:.4f}) for "
                f"spot={spot} K={strike} T={T} sigma={sigma}"
            )
 
    def test_gamma_finite_and_non_negative_on_extreme_inputs(self):
        """Verify that Gamma remains finite and non-negative for extreme valid inputs."""
        cases = [
            (100.0, 100.0, 1e-6, 0.25, 0.05, "call"),
            (1000.0, 100.0, 1.0, 0.25, 0.05, "call"),
            (1.0,   100.0, 1.0, 0.25, 0.05, "call"),
            (100.0, 100.0, 1.0, 5.0,  0.05, "call"),
        ]
        for spot, strike, T, sigma, r, option_type in cases:
            gamma = black_scholes_gamma(spot, strike, T, sigma, r, option_type)
            assert math.isfinite(gamma), (
                f"Gamma is not finite ({gamma!r}) for "
                f"spot={spot} K={strike} T={T} sigma={sigma}"
            )
            assert gamma >= 0, (
                f"Gamma is negative ({gamma:.6f}) for "
                f"spot={spot} K={strike} T={T} sigma={sigma}"
            )


class TestTemporalStability:
    """Verify option prices decay smoothly towards intrinsic value as expiry approaches."""
 
    def test_atm_call_price_decreases_monotonically_toward_expiry(self):
        """Verify that an ATM call option loses time value monotonically as expiry approaches."""
        spot   = 100.0
        strike = 100.0
        sigma  = 0.25
        r      = 0.05
 
        times  = [2.0, 1.0, 0.5, 0.25, 1/12, 1/52]
        prices = [black_scholes_price(spot, strike, t, sigma, r, "call") for t in times]
 
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1], (
                f"ATM call not losing time value monotonically: "
                f"price(T={times[i]:.4f}) = {prices[i]:.4f} <= "
                f"price(T={times[i+1]:.4f}) = {prices[i+1]:.4f}"
            )
 
    def test_itm_call_converges_to_intrinsic_at_expiry(self):
        """Verify that an ITM call option price converges to its intrinsic value as time to maturity approaches zero."""
        spot      = 110.0
        strike    = 100.0
        sigma     = 0.25
        r         = 0.05
        intrinsic = max(spot - strike, 0.0)
 
        price_near_expiry = black_scholes_price(spot, strike, 1e-6, sigma, r, "call")
 
        assert abs(price_near_expiry - intrinsic) < 0.01, (
            f"ITM call at T≈0 ({price_near_expiry:.4f}) does not converge to "
            f"intrinsic ({intrinsic:.4f})"
        )
 
    def test_no_price_spikes_on_monthly_time_grid(self):
        """Verify that option prices decay smoothly without anomalous spikes as time to expiry decreases."""
        spot   = 100.0
        strike = 100.0
        sigma  = 0.25
        r      = 0.05

        times  = [i / 12 for i in range(12, 0, -1)]
        prices = [black_scholes_price(spot, strike, t, sigma, r, "call") for t in times]

        for i in range(len(prices) - 1):
            step = prices[i] - prices[i + 1]
            
            assert step > 0, (
                f"Positive spike detected! Price increased from "
                f"{prices[i]:.4f} to {prices[i+1]:.4f} as time decreased."
            )
            
            max_theoretical_step = 0.25 * prices[0]
            assert step < max_theoretical_step, (
                f"Negative spike detected between T={times[i]:.4f} and "
                f"T={times[i+1]:.4f}: step={step:.4f} > threshold={max_theoretical_step:.4f}"
            )