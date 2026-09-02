import numpy as np
import pytest
from models.heston import simulate_heston_paths, heston_cumulants
from models.bsm import bsm_price
from models.merton import (
    merton_char_func, merton_cumulants
)
from models.kou import (
    kou_char_func, kou_cumulants
)
from models.bates import (
bates_char_func, bates_cumulants, bates_simulate_terminal
)
from pricing.fourier import (
    cos_call_price, cos_put_price, cos_smile, bates_cos_price, merton_cos_price, kou_cos_price
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
        
class TestMerton:
    # Reference parameters for Merton tests. q nonzero to exercise the dividend term,
    # mu_j negative for the equity-skew direction, all distinct so no coincidental pass.
    S0, R, Q, T = 100.0, 0.03, 0.01, 0.5
    SIGMA, LAM, MU_J, DELTA_J = 0.20, 0.5, -0.10, 0.15

    def test_bsm_limit_put(self):
        """Same BSM-limit check on the put side (independent payoff coefficients)."""
        for K in (80.0, 100.0, 120.0):
            merton = merton_cos_price(self.S0, self.R, self.Q, self.T, self.SIGMA, lam=0.0,
                                      mu_j=self.MU_J, delta_j=self.DELTA_J, K=K,
                                      option_type="put")
            bsm = bsm_price(self.S0, K, self.T, self.R, self.SIGMA, "put", b=self.R - self.Q)
            assert np.isclose(merton, bsm, atol=1e-6), (K, merton, bsm)

    def test_jump_params_irrelevant_when_lam_zero(self):
        """With lam=0, changing mu_j and delta_j must not move the price.
        A direct guard that no jump effect leaks through the lam=0 gate."""
        base = merton_cos_price(self.S0, self.R, self.Q, self.T, self.SIGMA, 0.0, -0.2, 0.30, K=100.0)
        alt = merton_cos_price(self.S0, self.R, self.Q, self.T, self.SIGMA, 0.0, 0.4, 0.05, K=100.0)
        assert np.isclose(base, alt, atol=1e-10)

    def test_martingale_char_func(self):
        """phi(-i) = S0 e^{(r-q)T}, certifying the kappa compensator.

        Uses nonzero jumps: the point is that the compensator cancels the mean
        jump growth. Evaluated with jumps on, so a wrong kappa fails here.
        """
        phi = merton_char_func(np.array([-1j]), self.S0, self.R, self.Q, self.T,
                               self.SIGMA, self.LAM, self.MU_J, self.DELTA_J)[0]
        expected = self.S0 * np.exp((self.R - self.Q) * self.T)
        assert np.isclose(phi.real, expected, atol=1e-8)
        assert np.isclose(phi.imag, 0.0, atol=1e-8)

    def test_char_func_at_zero(self):
        """phi(0) = 1 for any valid char func (normalisation)."""
        phi = merton_char_func(np.array([0.0 + 0j]), self.S0, self.R, self.Q, self.T,
                               self.SIGMA, self.LAM, self.MU_J, self.DELTA_J)[0]
        assert np.isclose(phi, 1.0, atol=1e-12)

    def test_cumulants_reduce_to_bsm_when_lam_zero(self):
        """With lam=0, cumulants must be the pure BSM log-price cumulants:
        c1 = log S0 + (r - q - 0.5 sigma^2) T,  c2 = sigma^2 T."""
        c1, c2 = merton_cumulants(self.S0, self.R, self.Q, self.T, self.SIGMA, lam=0.0,
                                  mu_j=self.MU_J, delta_j=self.DELTA_J)
        c1_bsm = np.log(self.S0) + (self.R - self.Q - 0.5 * self.SIGMA**2) * self.T
        c2_bsm = self.SIGMA**2 * self.T
        assert np.isclose(c1, c1_bsm, atol=1e-12)
        assert np.isclose(c2, c2_bsm, atol=1e-12)

    def test_put_call_parity(self):
        """COS call and put on the same strike must satisfy parity:
        C - P = S0 e^{-qT} - K e^{-rT}. A cross-check the two payoff sides agree."""
        K = 100.0
        call = merton_cos_price(self.S0, self.R, self.Q, self.T, self.SIGMA,
                                self.LAM, self.MU_J, self.DELTA_J, K, "call")
        put = merton_cos_price(self.S0, self.R, self.Q, self.T, self.SIGMA,
                               self.LAM, self.MU_J, self.DELTA_J, K, "put")
        lhs = call - put
        rhs = self.S0 * np.exp(-self.Q * self.T) - K * np.exp(-self.R * self.T)
        assert np.isclose(lhs, rhs, atol=1e-4)
        
    def test_cos_matches_mc(self):
        """COS price must agree with independent Monte Carlo (z-score < ~3).

        MC simulates the actual Merton process (diffusion + Poisson jumps) with
        no shared machinery, so agreement certifies the jump pricing, not just
        the analytic limits.
        """
        from models.merton import merton_simulate_terminal
        K = 100.0
        cos_p = merton_cos_price(self.S0, self.R, self.Q, self.T, self.SIGMA,
                                 self.LAM, self.MU_J, self.DELTA_J, K, "call")
        ST = merton_simulate_terminal(self.S0, self.R, self.Q, self.T, self.SIGMA,
                                      self.LAM, self.MU_J, self.DELTA_J,
                                      n_paths=2_000_000, seed=42)
        disc_payoff = np.exp(-self.R * self.T) * np.maximum(ST - K, 0.0)
        mc_price = disc_payoff.mean()
        mc_se = disc_payoff.std(ddof=1) / np.sqrt(len(ST))
        z = abs(cos_p - mc_price) / mc_se
        assert z < 3.0, f"COS={cos_p:.4f}, MC={mc_price:.4f}+/-{mc_se:.4f}, z={z:.2f}"
        
class TestKou:
    S0, R, Q, T = 100.0, 0.03, 0.01, 0.5
    SIGMA, LAM, P, ETA1, ETA2 = 0.20, 0.5, 0.4, 10.0, 5.0

    def test_bsm_limit_call(self):
        """lam=0 recovers BSM (jump params set nonzero, must not leak)."""
        for K in (80.0, 100.0, 120.0):
            kou = kou_cos_price(self.S0, self.R, self.Q, self.T, self.SIGMA,
                                lam=0.0, p=self.P, eta1=self.ETA1, eta2=self.ETA2,
                                K=K, option_type="call")
            bsm = bsm_price(self.S0, K, self.T, self.R, self.SIGMA, "call", b=self.R - self.Q)
            assert np.isclose(kou, bsm, atol=1e-6), (K, kou, bsm)

    def test_bsm_limit_put(self):
        for K in (80.0, 100.0, 120.0):
            kou = kou_cos_price(self.S0, self.R, self.Q, self.T, self.SIGMA,
                                lam=0.0, p=self.P, eta1=self.ETA1, eta2=self.ETA2,
                                K=K, option_type="put")
            bsm = bsm_price(self.S0, K, self.T, self.R, self.SIGMA, "put", b=self.R - self.Q)
            assert np.isclose(kou, bsm, atol=1e-6), (K, kou, bsm)

    def test_martingale_char_func(self):
        """phi(-i) = S0 e^{(r-q)T}, jumps on, certifies kappa."""
        phi = kou_char_func(np.array([-1j]), self.S0, self.R, self.Q, self.T,
                            self.SIGMA, self.LAM, self.P, self.ETA1, self.ETA2)[0]
        expected = self.S0 * np.exp((self.R - self.Q) * self.T)
        assert np.isclose(phi.real, expected, atol=1e-8)
        assert np.isclose(phi.imag, 0.0, atol=1e-8)

    def test_char_func_at_zero(self):
        phi = kou_char_func(np.array([0.0 + 0j]), self.S0, self.R, self.Q, self.T,
                            self.SIGMA, self.LAM, self.P, self.ETA1, self.ETA2)[0]
        assert np.isclose(phi, 1.0, atol=1e-12)

    def test_cumulants_reduce_to_bsm_when_lam_zero(self):
        c1, c2 = kou_cumulants(self.S0, self.R, self.Q, self.T, self.SIGMA,
                               lam=0.0, p=self.P, eta1=self.ETA1, eta2=self.ETA2)
        assert np.isclose(c1, np.log(self.S0) + (self.R - self.Q - 0.5 * self.SIGMA**2) * self.T, atol=1e-12)
        assert np.isclose(c2, self.SIGMA**2 * self.T, atol=1e-12)

    def test_eta1_constraint_raises(self):
        """eta1 <= 1 must raise (compensator divergence)."""
        with pytest.raises(ValueError):
            kou_char_func(np.array([1.0 + 0j]), self.S0, self.R, self.Q, self.T,
                          self.SIGMA, self.LAM, self.P, eta1=0.8, eta2=self.ETA2)

    def test_put_call_parity(self):
        K = 100.0
        call = kou_cos_price(self.S0, self.R, self.Q, self.T, self.SIGMA,
                             self.LAM, self.P, self.ETA1, self.ETA2, K, "call", L=14)
        put = kou_cos_price(self.S0, self.R, self.Q, self.T, self.SIGMA,
                            self.LAM, self.P, self.ETA1, self.ETA2, K, "put", L=14)
        lhs = call - put
        rhs = self.S0 * np.exp(-self.Q * self.T) - K * np.exp(-self.R * self.T)
        assert np.isclose(lhs, rhs, atol=1e-4)

    def test_cos_matches_mc(self):
        """COS vs independent Monte Carlo, z-score < 3. The real jump-pricing test."""
        from models.kou import kou_simulate_terminal
        K = 100.0
        cos_p = kou_cos_price(self.S0, self.R, self.Q, self.T, self.SIGMA,
                              self.LAM, self.P, self.ETA1, self.ETA2, K, "call")
        ST = kou_simulate_terminal(self.S0, self.R, self.Q, self.T, self.SIGMA,
                                   self.LAM, self.P, self.ETA1, self.ETA2,
                                   n_paths=2_000_000, seed=42)
        disc = np.exp(-self.R * self.T) * np.maximum(ST - K, 0.0)
        mc_price, mc_se = disc.mean(), disc.std(ddof=1) / np.sqrt(len(ST))
        z = abs(cos_p - mc_price) / mc_se
        assert z < 3.0, f"COS={cos_p:.4f}, MC={mc_price:.4f}+/-{mc_se:.4f}, z={z:.2f}"
    
class TestBates:
    # distinct values, q and lam nonzero to exercise dividends and jumps,
    # mu_j < 0 for equity-skew direction
    S0, v0 = 100.0, 0.04
    KAPPA_V, THETA, XI, RHO = 2.0, 0.04, 0.3, -0.7
    R, Q, TAU = 0.03, 0.01, 0.5
    LAM, MU_J, DELTA_J = 0.5, -0.10, 0.15

    def test_martingale(self):
        """phi(-i) = S0 e^{(r-q)tau}: certifies (r-q) once, -lam*kappa_j once."""
        phi = bates_char_func(np.array([-1j]), self.S0, self.v0, self.KAPPA_V,
                              self.THETA, self.XI, self.RHO, self.R, self.Q,
                              self.TAU, self.LAM, self.MU_J, self.DELTA_J)[0]
        expected = self.S0 * np.exp((self.R - self.Q) * self.TAU)
        assert np.isclose(phi.real, expected, atol=1e-8)
        assert np.isclose(phi.imag, 0.0, atol=1e-8)

    def test_char_func_at_zero(self):
        phi = bates_char_func(np.array([0.0 + 0j]), self.S0, self.v0, self.KAPPA_V,
                              self.THETA, self.XI, self.RHO, self.R, self.Q,
                              self.TAU, self.LAM, self.MU_J, self.DELTA_J)[0]
        assert np.isclose(phi, 1.0, atol=1e-12)

    def test_heston_limit(self):
        """lam -> 0 recovers Heston. Compared at q=0 so drift and discount both
        equal r, matching cos_call_price's single-r convention (a dividend-aware
        model cannot be checked against a pricer that conflates drift and
        discount, so q=0 is the fair comparison)."""
        for K in (90.0, 100.0, 110.0):
            bates = bates_cos_price(self.S0, self.v0, self.KAPPA_V, self.THETA,
                                    self.XI, self.RHO, self.R, 0.0, self.TAU, K,
                                    lam=0.0, mu_j=self.MU_J, delta_j=self.DELTA_J)
            heston = cos_call_price(self.S0, self.v0, self.KAPPA_V, self.THETA,
                                    self.XI, self.RHO, self.R, self.TAU, K)
            assert np.isclose(bates, heston, atol=1e-8), (K, bates, heston)

    def test_merton_limit(self):
        """xi -> 0, v0 = theta recovers Merton at sigma = sqrt(theta): the
        stochastic-vol side collapses to constant vol, leaving the jumps."""
        xi_small = 1e-4
        for K in (90.0, 100.0, 110.0):
            bates = bates_cos_price(self.S0, self.THETA, self.KAPPA_V, self.THETA,
                                    xi_small, self.RHO, self.R, self.Q, self.TAU, K,
                                    self.LAM, self.MU_J, self.DELTA_J)
            merton = merton_cos_price(self.S0, self.R, self.Q, self.TAU,
                                      np.sqrt(self.THETA), self.LAM, self.MU_J,
                                      self.DELTA_J, K, "call")
            assert np.isclose(bates, merton, atol=1e-3), (K, bates, merton)

    def test_cumulants_reduce_to_heston_when_lam_zero(self):
        """At lam=0 the jump cumulant terms vanish, leaving Heston(r-q)."""
        c1_b, c2_b = bates_cumulants(self.S0, self.v0, self.KAPPA_V, self.THETA,
                                     self.XI, self.RHO, self.R, self.Q, self.TAU,
                                     0.0, self.MU_J, self.DELTA_J)
        c1_h, c2_h = heston_cumulants(self.S0, self.v0, self.KAPPA_V, self.THETA,
                                      self.XI, self.RHO, self.R - self.Q, self.TAU)
        assert np.isclose(c1_b, c1_h, atol=1e-12)
        assert np.isclose(c2_b, c2_h, atol=1e-12)

    def test_put_call_parity(self):
        K = 100.0
        call = bates_cos_price(self.S0, self.v0, self.KAPPA_V, self.THETA, self.XI,
                               self.RHO, self.R, self.Q, self.TAU, K, self.LAM,
                               self.MU_J, self.DELTA_J, "call")
        put = bates_cos_price(self.S0, self.v0, self.KAPPA_V, self.THETA, self.XI,
                              self.RHO, self.R, self.Q, self.TAU, K, self.LAM,
                              self.MU_J, self.DELTA_J, "put")
        lhs = call - put
        rhs = self.S0 * np.exp(-self.Q * self.TAU) - K * np.exp(-self.R * self.TAU)
        assert np.isclose(lhs, rhs, atol=1e-4)

    def test_cos_matches_mc(self):
        """COS vs independent Monte Carlo (Heston QE variance + Poisson jumps),
        z-score < 3. The full stochastic-vol-plus-jumps pricing test."""
        K = 100.0
        cos_p = bates_cos_price(self.S0, self.v0, self.KAPPA_V, self.THETA, self.XI,
                                self.RHO, self.R, self.Q, self.TAU, K, self.LAM,
                                self.MU_J, self.DELTA_J, "call")
        ST = bates_simulate_terminal(self.S0, self.v0, self.KAPPA_V, self.THETA,
                                     self.XI, self.RHO, self.R, self.Q, self.TAU,
                                     self.LAM, self.MU_J, self.DELTA_J,
                                     n_paths=1_000_000, n_steps=100, seed=42)
        disc = np.exp(-self.R * self.TAU) * np.maximum(ST - K, 0.0)
        mc_p, mc_se = disc.mean(), disc.std(ddof=1) / np.sqrt(len(ST))
        z = abs(cos_p - mc_p) / mc_se
        assert z < 3.0, f"COS={cos_p:.4f}, MC={mc_p:.4f}+/-{mc_se:.4f}, z={z:.2f}"