from math import exp
import pytest
import math
from src.pricing import black_scholes_price, black_scholes_delta
from src.pricing import black_scholes_price, black_scholes_delta, black_scholes_gamma, implied_volatility


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

def test_full_repricing_exceeds_delta_approximation_for_large_move():
    """
    Component Accuracy Test: For a large negative spot move (-10%), verify that
    the actual option loss computed by full Black-Scholes repricing is greater
    than the loss predicted by the linear delta approximation alone.

    This confirms that full repricing was the correct choice for historical and
    Monte Carlo scenarios: the delta approximation systematically understates
    losses for large moves due to option convexity (gamma effect).

    Method:
      - Price an ATM call at spot=100.
      - Apply a -10% spot shock (spot -> 90).
      - Compute actual loss: original_price - repriced_price.
      - Compute delta-approximated loss: delta * spot * 0.10
      - Assert: actual_loss > delta_approx_loss
    """
    spot = 100.0
    strike = 100.0
    T = 1.0
    sigma = 0.25
    r = 0.05
    shock = -0.10  

    
    shocked_spot = spot * math.exp(shock)  

    original_price = black_scholes_price(spot, strike, T, sigma, r, "call")
    repriced_price = black_scholes_price(shocked_spot, strike, T, sigma, r, "call")
    actual_loss = original_price - repriced_price

    delta = black_scholes_delta(spot, strike, T, sigma, r, "call")
    delta_approx_loss = delta * spot * abs(shock)

    assert actual_loss < delta_approx_loss, (
        f"Expected full repricing loss ({actual_loss:.4f}) < "
        f"delta approximation ({delta_approx_loss:.4f})"
    )


def test_delta_gamma_approximation_vs_full_repricing():
    """
    Delta-Gamma Approximation Test: Verifies that the 2nd-order Taylor series 
    expansion (Delta + Gamma) closely approximates the exact Black-Scholes 
    repricing for a small move in the underlying asset.
    
    Formula: Approx New Price ≈ Old Price + (Delta * dS) + (0.5 * Gamma * dS^2)
    """
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
    """
    Implied Volatility Round-Trip Test: Tests the invertibility of the pricing model.
    If we input Volatility X into Black-Scholes to get Price Y, putting Price Y 
    into an Implied Volatility solver must return Volatility X.
    """
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
    """Call price must be strictly increasing in spot; put price strictly decreasing."""
 
    def test_call_price_increases_with_spot(self):
        """
        Monotonicity Test: A call option's value must increase as the underlying
        spot price rises, all else equal.
 
        Fundamental no-arbitrage property: a higher spot makes in-the-money
        expiry more likely, so the call must be worth more.
        """
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
        """
        Monotonicity Test: A put option's value must decrease as spot rises —
        the mirror of call monotonicity, equally required by no-arbitrage.
        """
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
        """
        Monotonicity Test: Higher vol widens the distribution of future outcomes,
        increasing the expected payoff of a call. Price must rise strictly with vol.
        """
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
        """
        Monotonicity Test: A call with a higher strike pays off less in every
        scenario, so it must be worth strictly less.
        """
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
    """
    The Black-Scholes call (and put) price is a convex function of spot.
 
    Convexity means: price(S+h) + price(S-h) >= 2*price(S)  for any h > 0.
    This is the discrete version of gamma > 0 and must hold everywhere.
    """
 
    def test_call_price_is_convex_in_spot(self):
        """
        Convexity Test: Verify call price is convex in spot via the finite-
        difference second derivative at multiple spot levels.
 
        Uses h=$1 to stay clear of floating-point cancellation noise — per
        the lecture's guidance on choosing delta-x to balance convexity error
        and cancellation ('Beware addition' and 'Approximations' slides).
        """
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
        """
        Convexity Test: Put prices share the same gamma as calls (put-call
        parity), so convexity must hold for puts too.
        """
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
        """
        Convexity Test: The call price is convex in strike — a classic butterfly
        arbitrage argument. Violating this allows a costless butterfly to have
        a positive payoff.
        """
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
    """
    black_scholes_delta must equal the numerical first derivative of
    black_scholes_price with respect to spot.
 
    If they disagree, one function is wrong — a classic code-review catch
    described in the lecture.
    """
 
    @pytest.mark.parametrize("spot,strike,T,sigma,r,option_type", [
        (100.0, 100.0, 1.0, 0.20, 0.05, "call"),   # ATM call
        (100.0, 100.0, 1.0, 0.20, 0.05, "put"),    # ATM put
        (120.0, 100.0, 1.0, 0.25, 0.03, "call"),   # deep ITM call
        ( 80.0, 100.0, 0.5, 0.30, 0.02, "call"),   # OTM call
        ( 80.0, 100.0, 0.5, 0.30, 0.02, "put"),    # ITM put
    ])
    def test_delta_equals_finite_difference_derivative(
        self, spot, strike, T, sigma, r, option_type
    ):
        """
        Self-Consistency Test: black_scholes_delta(S) should match
        [price(S+h) - price(S-h)] / (2h) to within numerical precision.
 
        Uses a centred difference with h=0.01 — small enough for accuracy,
        large enough to avoid cancellation errors (per lecture's guidance on
        picking delta-x).
        """
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
    """
    black_scholes_gamma must equal the numerical second derivative of
    black_scholes_price with respect to spot.
 
    The lecture devoted several slides to how easy it is to get gamma wrong
    numerically. This test checks the analytic formula against a finite
    difference to confirm both compute the same quantity.
    """
 
    @pytest.mark.parametrize("spot,strike,T,sigma,r,option_type", [
        (100.0, 100.0, 1.0, 0.20, 0.05, "call"),
        (100.0, 100.0, 1.0, 0.20, 0.05, "put"),
        (120.0, 100.0, 1.0, 0.25, 0.03, "call"),
        ( 80.0, 100.0, 0.5, 0.30, 0.02, "call"),
    ])
    def test_gamma_equals_finite_difference_second_derivative(
        self, spot, strike, T, sigma, r, option_type
    ):
        """
        Self-Consistency Test: black_scholes_gamma(S) should match
        [price(S+h) - 2*price(S) + price(S-h)] / h^2.
 
        Uses h=1.0 ($1 bump). The lecture explicitly warns that h must be
        chosen carefully: too small causes cancellation; too large introduces
        truncation. h=$1 balances both for typical equity option inputs.
        """
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
        """
        Sanity Test: Analytic gamma must be strictly positive for all finite
        inputs — confirming the convexity established in the price tests is
        also reflected in the Greek itself.
        """
        strike = 100.0
        T = 1.0
        sigma = 0.25
        r = 0.05
 
        for spot in [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]:
            g = black_scholes_gamma(spot, strike, T, sigma, r, "call")
            assert g > 0, f"Gamma not positive at spot={spot}: gamma={g:.6f}"
 

 
class TestFailureModes:
    """
    Verify that the pricer handles extreme but legal inputs without producing
    NaN, inf, or raising unexpected exceptions.
 
    The lecture specifically asks: 'Where can underflows and overflows occur
    and how will they propagate through the calculation?'
    """
 
    @pytest.mark.parametrize("spot,strike,T,sigma,r,option_type", [
        (100.0, 100.0, 1.0,   5.0,  0.05, "call"),   # very high vol
        (100.0, 100.0, 1.0,   5.0,  0.05, "put"),    # very high vol, put
        (100.0, 100.0, 1e-6, 0.25,  0.05, "call"),   # near-expiry
        (100.0, 100.0, 1e-6, 0.25,  0.05, "put"),    # near-expiry, put
        (1000.0, 100.0, 1.0, 0.25,  0.05, "call"),   # deep ITM
        (  1.0, 100.0, 1.0,  0.25,  0.05, "call"),   # deep OTM
        (1e6,   100.0, 1.0,  0.25,  0.05, "call"),   # very large spot
        (100.0, 100.0, 1.0,  0.25, -0.01, "call"),   # negative rate
        (100.0, 100.0, 1.0,  0.25, -0.01, "put"),    # negative rate, put
    ])
    def test_no_nan_or_inf_on_extreme_inputs(
        self, spot, strike, T, sigma, r, option_type
    ):
        """
        Failure Mode Test: Extreme but valid inputs must produce a finite,
        non-NaN, non-negative price.
 
        A NaN or inf propagating silently through a risk system would corrupt
        every downstream calculation — the lecture warns this is the most
        dangerous class of numerical error.
        """
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
        """
        Failure Mode Test: Delta must be finite and within [-1, 1] for all
        valid inputs. Infinite or NaN greeks would break hedging and VaR.
        """
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
        """
        Failure Mode Test: Gamma must be finite and non-negative for all
        valid inputs.
        """
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
    """
    As time-to-expiry decreases, prices should decay smoothly to intrinsic
    value without spikes or discontinuities.
 
    The lecture's robustness slides state: 'Model outputs should be relatively
    stable over time' and 'Small perturbations of model inputs should not lead
    to large changes in model outputs.'
    """
 
    def test_atm_call_price_decreases_monotonically_toward_expiry(self):
        """
        Temporal Stability Test: An ATM call loses time value monotonically
        as expiry approaches. The model must produce a smooth term structure.
        """
        spot   = 100.0
        strike = 100.0
        sigma  = 0.25
        r      = 0.05
 
        # Decreasing T: 2yr → 1yr → 6m → 3m → 1m → 1wk
        times  = [2.0, 1.0, 0.5, 0.25, 1/12, 1/52]
        prices = [black_scholes_price(spot, strike, t, sigma, r, "call") for t in times]
 
        for i in range(len(prices) - 1):
            assert prices[i] > prices[i + 1], (
                f"ATM call not losing time value monotonically: "
                f"price(T={times[i]:.4f}) = {prices[i]:.4f} <= "
                f"price(T={times[i+1]:.4f}) = {prices[i+1]:.4f}"
            )
 
    def test_itm_call_converges_to_intrinsic_at_expiry(self):
        """
        Limiting Case / Temporal Stability Test: As T→0, the ITM call price
        must converge to max(S-K, 0).
 
        Combines the lecture's 'Limiting cases' and 'Test over time' themes:
        the model must not blow up or drift as it reaches the boundary.
        """
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
        """
        Temporal Stability Test: Price should smoothly decay as time to expiry decreases.
        
        Note: We do not use a flat absolute threshold because ATM option time 
        decay (Theta) scales with the square root of time, meaning absolute 
        price drops accelerate as expiry approaches. Instead, we check for 
        strict monotonicity (no upward spikes) and bound the maximum expected drop.
        """
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