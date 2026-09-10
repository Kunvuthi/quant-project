import numpy as np
import pytest
from models.rbergomi import (
    RBergomiParams, volterra_cov, cross_cov, simulate_rbergomi
)
# from pricing.monte_carlo import european_price_from_samples -> migrate tmr

# Assumed API contract (pins tomorrow's implementation):
#   volterra_cov(s, t, H) -> E[W_tilde_s W_tilde_t], elementwise over broadcast s, t
#   cross_cov(t_vol, s_bm, H) -> E[W_tilde_{t_vol} W_{s_bm}], elementwise
#   simulate_rbergomi(t_grid, params, n_paths, rng) -> (S, v, W_tilde), each (n_paths, N)
# rBergomi has NO char func and NO cumulants by construction (non-Markovian), so
# there is no phi(0)=1 / phi(-i) analytic test here; the moment checks below are
# the simulation-side equivalents.


# ---- Analytic building blocks (fast, deterministic, no simulation) ---- #

class TestVolterraCovariance:
    """The covariance kernel is the ground truth the simulator is built from,
    so certify it in isolation first, exactly as the char-func/cumulant tests
    certify the Fourier building blocks before any pricing runs."""

    H = 0.1  # rough; distinct from the 0.5 limit below so neither test passes by coincidence

    def test_diagonal_is_roughness_scaling(self):
        """E[W_tilde_t^2] = t^(2H). This single identity is the normalisation
        the sqrt(2H) prefactor exists to enforce; if it fails, every downstream
        variance and the martingale correction are wrong."""
        t = np.array([0.05, 0.25, 0.5, 1.0, 2.0])
        diag = volterra_cov(t, t, self.H)
        assert np.allclose(diag, t ** (2.0 * self.H), atol=1e-8), (diag, t ** (2.0 * self.H))

    def test_reduces_to_brownian_covariance_at_half(self):
        """H=0.5 collapses the Volterra process to plain BM (kernel = 1,
        sqrt(2H)=1, W_tilde = W), so the covariance must equal min(s,t). This is
        the rBergomi analogue of the BSM/Heston-limit tests: a known closed form
        the general expression must reproduce. Also exercises the 2F1 (or
        quadrature) path at the boundary, where it is most fragile."""
        s = np.array([0.1, 0.3, 0.7, 0.7, 1.0])
        t = np.array([0.2, 0.3, 0.5, 1.2, 1.0])
        cov = volterra_cov(s, t, 0.5)
        assert np.allclose(cov, np.minimum(s, t), atol=1e-6), (cov, np.minimum(s, t))

    def test_cross_cov_reduces_to_min_at_half(self):
        """At H=0.5, W_tilde = W, so E[W_tilde_t W_s] = min(s,t) as well.
        Certifies the leverage-channel covariance block independently of the
        auto-covariance, since a sign or index slip here silently breaks skew."""
        t_vol = np.array([0.2, 0.5, 1.0, 1.0])
        s_bm = np.array([0.5, 0.5, 0.4, 1.5])
        cross = cross_cov(t_vol, s_bm, 0.5)
        assert np.allclose(cross, np.minimum(t_vol, s_bm), atol=1e-6), (cross, np.minimum(t_vol, s_bm))


# ---- Simulator moment checks (z-score, SE-derived tolerances) ---- #

class TestRBergomiSimulator:
    # Rough H, O(1) vol-of-vol as is standard for rBergomi, strong negative rho
    # for the equity leverage direction, flat forward variance at 0.04 (matches
    # the Heston tests' v0 so the level is familiar). All distinct.
    H, ETA, RHO, V0 = 0.1, 1.5, -0.9, 0.04
    T, N_STEPS = 1.0, 100

    def _params(self, rho=None):
        rho = self.RHO if rho is None else rho
        return RBergomiParams(H=self.H, eta=self.ETA, rho=rho,
                              xi0=lambda t: np.full_like(np.asarray(t, float), self.V0))

    def _grid(self):
        return np.linspace(self.T / self.N_STEPS, self.T, self.N_STEPS)

    def test_volterra_variance_scaling(self):
        """Empirical Var(W_tilde_t) must follow t^(2H). Checked as a log-log
        slope (~2H) rather than pointwise: a slope is a convergence-rate
        certification and dodges the multiple-comparisons flakiness of asserting
        |z|<k at every one of N grid points. This gates everything downstream."""
        n_paths = 300_000
        rng = np.random.default_rng(0)
        t = self._grid()
        _, _, W_tilde = simulate_rbergomi(t, self._params(), n_paths, rng)

        emp_var = W_tilde.var(axis=0, ddof=1)
        slope = np.polyfit(np.log(t), np.log(emp_var), 1)[0]
        assert np.isclose(slope, 2.0 * self.H, atol=0.02), f"log-log slope {slope:.4f} vs 2H={2*self.H}"

    def test_forward_variance_is_xi0(self):
        """E[v_t] = xi0(t). Certifies the -0.5*eta^2*t^(2H) martingale
        correction: with it, the lognormal mean is pinned to xi0; without it,
        E[v_t] drifts up by exp(0.5*eta^2*t^(2H)). z-score at a spread of
        maturities, SE from the sample mean of v itself."""
        n_paths = 300_000
        rng = np.random.default_rng(1)
        t = self._grid()
        _, v, _ = simulate_rbergomi(t, self._params(), n_paths, rng)

        theo = self.V0
        for i in (self.N_STEPS // 4, self.N_STEPS // 2, self.N_STEPS - 1):
            emp = v[:, i].mean()
            se = v[:, i].std(ddof=1) / np.sqrt(n_paths)
            z = abs(emp - theo) / se
            assert z < 3.0, f"t={t[i]:.3f}: E[v]={emp:.5f} vs xi0={theo:.5f} +/- {se:.5f}, z={z:.2f}"

    def test_underlying_is_martingale(self):
        """Under the forward measure (r=0), E[S_T] = S_0 = 1. This certifies the
        -0.5*v*dt Ito drift correction in the log-price, the simulation-side
        analogue of the phi(-i)=S0 e^{(r-q)T} martingale test for the Fourier
        models. A wrong or missing drift term fails here even when v is correct."""
        n_paths = 500_000
        rng = np.random.default_rng(2)
        t = self._grid()
        S, _, _ = simulate_rbergomi(t, self._params(), n_paths, rng)
        S_T = S[:, -1]

        emp = S_T.mean()
        se = S_T.std(ddof=1) / np.sqrt(n_paths)
        z = abs(emp - 1.0) / se
        assert z < 3.0, f"E[S_T]={emp:.5f} vs 1.0 +/- {se:.5f}, z={z:.2f}"

    def test_positivity(self):
        """v = xi0 * exp(...) and S = exp(...) are positive by construction;
        a nonpositive value means an indexing/assembly bug, not a numerical one."""
        rng = np.random.default_rng(3)
        t = self._grid()
        S, v, _ = simulate_rbergomi(t, self._params(), 10_000, rng)
        assert np.all(v > 0.0)
        assert np.all(S > 0.0)


# ---- Leverage channel: the skew-sign certification ---- #

class TestRBergomiLeverage:
    """The skew is produced by the price sharing the SAME Brownian increments
    that drive W_tilde. Nothing in TestRBergomiSimulator touches that coupling
    (variance-scaling and E[v_t] inspect only W_tilde and v; the martingale test
    holds for any rho), so all of those stay green even if the price is driven by
    an independently drawn BM. These two tests are the only guard on the leverage
    wiring. Certified via the sample skewness of log S_T, which maps directly to
    the sign of the IV skew and needs no pricer or IV inversion."""

    H, ETA, V0, T, N_STEPS = 0.1, 1.5, 0.04, 1.0, 100

    def _params(self, rho):
        return RBergomiParams(H=self.H, eta=self.ETA, rho=rho,
                              xi0=lambda t: np.full_like(np.asarray(t, float), self.V0))

    @staticmethod
    def _sample_skewness(x):
        xc = x - x.mean()
        return (xc ** 3).mean() / (xc.var() ** 1.5)

    def test_negative_correlation_gives_negative_skew(self):
        """rho < 0: down moves in S coincide with up moves in v, the left wing
        lifts, log-return distribution is left-skewed. Assert skewness clearly
        negative (many SE below zero). A symmetric result here is the classic
        broken-leverage bug: the price consuming a fresh BM instead of the shared
        one, which severs the channel while leaving every moment test above green."""
        n_paths = 400_000
        rng = np.random.default_rng(10)
        t = np.linspace(self.T / self.N_STEPS, self.T, self.N_STEPS)
        S, _, _ = simulate_rbergomi(t, self._params(rho=-0.9), n_paths, rng)
        g1 = self._sample_skewness(np.log(S[:, -1]))
        se = np.sqrt(6.0 / n_paths)  # asymptotic SE of the skewness estimator
        assert g1 < -5.0 * se, f"log S_T skewness {g1:.4f} not clearly negative (SE {se:.4f})"

    def test_zero_correlation_is_symmetric(self):
        """rho = 0: vol moves independ of price direction, so there is still a
        smile (kurtosis) but NO skew. Skewness must be consistent with zero.
        This is the control that proves the negative skew above is caused by rho,
        not by some unconditional asymmetry in the simulator."""
        n_paths = 400_000
        rng = np.random.default_rng(11)
        t = np.linspace(self.T / self.N_STEPS, self.T, self.N_STEPS)
        S, _, _ = simulate_rbergomi(t, self._params(rho=0.0), n_paths, rng)
        g1 = self._sample_skewness(np.log(S[:, -1]))
        se = np.sqrt(6.0 / n_paths)
        assert abs(g1) < 4.0 * se, f"log S_T skewness {g1:.4f} not consistent with 0 (SE {se:.4f})"