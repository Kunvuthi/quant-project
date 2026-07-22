import numpy as np
from models.heston import simulate_heston_paths
from pricing.fourier import (
    cos_call_price, cos_put_price, cos_smile
)
from pricing.carr_madan import (
    carr_madan_call_prices, carr_madan_put_prices
)

S0, v0 = 100.0, 0.04
kappa, theta, xi, rho, r = 2.0, 0.04, 0.3, -0.7, 0.05
T, K = 0.5, 100.0

def test_cos_call_price_vs_monte_carlo():
    """
    Cross-check the COS pricer against Week 4's independently-validated MC
    simulator, ATM call. Tolerance from the MC standard error of the payoff
    itself (not a guessed constant), same discipline as prior validation tests.
    """
    n_steps, n_paths = 126, 500_000

    rng = np.random.default_rng(0)
    S, _ = simulate_heston_paths(S0, v0, kappa, theta, xi, rho, r, T, n_steps, n_paths, rng=rng)
    S_T = S[:, -1]

    payoff = np.maximum(S_T - K, 0.0)
    disc_payoff = np.exp(-r * T) * payoff
    price_mc = disc_payoff.mean()
    se_mc = disc_payoff.std(ddof=1) / np.sqrt(n_paths)  # standard error of the MC mean itself

    price_cos = cos_call_price(S0, v0, kappa, theta, xi, rho, r, T, K)

    assert abs(price_cos - price_mc) < 4 * se_mc, \
        f"COS price {price_cos:.4f} vs MC {price_mc:.4f} +/- {se_mc:.4f}"

def test_cos_put_vs_parity():
    """
    Native put should agree with a parity-derived put to near machine precision,
    since COS carries essentially no numerical noise (~1e-13, per the Day 22-23
    convergence check), unlike MC where parity vs native made a real difference.
    Tight tolerance is deliberate: any larger gap signals a genuine bug in one
    of the two independent code paths, not floating-point noise.
    """
    put_native = cos_put_price(S0, v0, kappa, theta, xi, rho, r, T, K)
    call_price = cos_call_price(S0, v0, kappa, theta, xi, rho, r, T, K)
    put_parity = call_price - S0 + K * np.exp(-r * T)

    assert np.isclose(put_native, put_parity, atol=1e-6)


def test_cos_smile_matches_individual_calls():
    """cos_smile's batched A_k reuse shouldn't change results vs one-at-a-time calls."""
    strikes = np.linspace(70, 130, 13)
    smile_prices = cos_smile(S0, v0, kappa, theta, xi, rho, r, T, strikes, option_type='call')
    individual_prices = np.array([
        cos_call_price(S0, v0, kappa, theta, xi, rho, r, T, k) for k in strikes
    ])
    assert np.allclose(smile_prices, individual_prices, atol=1e-8)


def test_cos_smile_matches_individual_puts():
    """Same check on the put side, exercises cos_put_coefficients through cos_smile."""
    strikes = np.linspace(70, 130, 13)
    smile_prices = cos_smile(S0, v0, kappa, theta, xi, rho, r, T, strikes, option_type='put')
    individual_prices = np.array([
        cos_put_price(S0, v0, kappa, theta, xi, rho, r, T, k) for k in strikes
    ])
    assert np.allclose(smile_prices, individual_prices, atol=1e-8)


def test_cos_smile_dtype_safety():
    """
    Regression guard for the int-strikes silent-truncation bug: integer strikes
    should not corrupt prices via integer array assignment.
    """
    strikes_int = np.array([90, 100, 110])  # deliberately int64, not float
    prices = cos_smile(S0, v0, kappa, theta, xi, rho, r, T, strikes_int, option_type='call')
    assert prices.dtype == np.float64
    assert np.all(prices > 0)  # would be silently zero/wrong if truncated to int
    
# ---- Car-Madan ---- #
    
def test_carr_madan_vs_cos():
    """
    Carr-Madan FFT strip vs the already-validated COS pricer, at a spread of
    strikes. Looser tolerance than COS-vs-itself convergence checks, since
    Carr-Madan's FFT/Simpson discretization has real truncation error, unlike
    COS's exponential convergence.
    """
    strikes, cm_prices = carr_madan_call_prices(S0, v0, kappa, theta, xi, rho, r, T)

    mask = (strikes > 70) & (strikes < 130)
    for K, cm_price in zip(strikes[mask][::50], cm_prices[mask][::50]):
        cos_price = cos_call_price(S0, v0, kappa, theta, xi, rho, r, T, K)
        assert abs(cm_price - cos_price) < 1e-2, f"K={K}: CM={cm_price}, COS={cos_price}"
        
def test_carr_madan_put_vs_cos_put():
    """
    CM put (parity-derived from the CM call strip) vs COS's natively-derived
    put. Agreement here is real cross-validation signal: one path is
    FFT+parity, the other is an independent cosine-series put derivation,
    not just parity algebra checking itself.
    """
    strikes, cm_puts = carr_madan_put_prices(S0, v0, kappa, theta, xi, rho, r, T)

    mask = (strikes > 70) & (strikes < 130)
    for K, cm_put in zip(strikes[mask][::50], cm_puts[mask][::50]):
        cos_put = cos_put_price(S0, v0, kappa, theta, xi, rho, r, T, K)
        assert abs(cm_put - cos_put) < 1e-2, f"K={K}: CM put={cm_put}, COS put={cos_put}"