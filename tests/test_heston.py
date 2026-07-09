import numpy as np
import pytest
from models.heston import sample_cir_qe_step, simulate_heston_paths, cir_qe_regime_params, heston_char_func


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _closed_form_moments(kappa, theta, xi, v0, dt):
    ekt = np.exp(-kappa * dt)
    m = theta + (v0 - theta) * ekt
    s2 = (xi**2 * v0 * ekt / kappa) * (1 - ekt) + (xi**2 * theta / (2 * kappa)) * (1 - ekt) ** 2
    return m, s2


def test_qe_moments_feller_satisfied(rng):
    """2*kappa*theta >= xi^2, comfortably away from the boundary -> mostly regime 1."""
    kappa, theta, xi, v0, dt = 2.0, 0.04, 0.3, 0.04, 1 / 252
    n_paths = 200_000

    v_t = np.full(n_paths, v0)
    v_next = sample_cir_qe_step(v_t, kappa, theta, xi, dt, rng=rng)

    m_true, s2_true = _closed_form_moments(kappa, theta, xi, v0, dt)

    assert np.isclose(v_next.mean(), m_true, rtol=1e-2)
    assert np.isclose(v_next.var(), s2_true, rtol=5e-2)
    assert v_next.min() >= 0.0


def test_qe_moments_feller_violated(rng):
    """2*kappa*theta < xi^2, forces psi into regime 2 (point-mass + exponential)."""
    kappa, theta, xi, v0, dt = 1.0, 0.04, 1.5, 0.005, 0.5
    n_paths = 200_000

    v_t = np.full(n_paths, v0)
    v_next = sample_cir_qe_step(v_t, kappa, theta, xi, dt, rng=rng)

    m_true, s2_true = _closed_form_moments(kappa, theta, xi, v0, dt)
    psi_true = s2_true / m_true**2

    # sanity: this parameter set should actually be exercising regime 2
    assert psi_true > 1.5

    assert np.isclose(v_next.mean(), m_true, rtol=1e-2)
    assert np.isclose(v_next.var(), s2_true, rtol=5e-2)
    assert v_next.min() >= 0.0


def test_qe_never_negative(rng):
    """Regression guard: this is the entire reason QE exists over naive Euler."""
    kappa, theta, xi, v0, dt = 0.5, 0.02, 1.5, 0.001, 1 / 252  # deliberately nasty
    v_t = np.full(50_000, v0)
    v_next = sample_cir_qe_step(v_t, kappa, theta, xi, dt, rng=rng)
    assert v_next.min() >= 0.0
    
    
def test_heston_martingale_property():
    """
    E[S_T] = S0 * exp(r*T) under the risk-neutral measure, always, for any
    correctly-implemented model. This isolates the K0 correction specifically,
    if K0 is wrong, this drifts off; the variance-moments tests from Day 17
    wouldn't catch a K0 bug since they never touch the spot process at all.
    """
    S0, v0 = 100.0, 0.04
    kappa, theta, xi, rho, r = 2.0, 0.04, 0.3, -0.7, 0.05
    T, n_steps, n_paths = 1.0, 252, 200_000

    rng = np.random.default_rng(0)
    S, v = simulate_heston_paths(S0, v0, kappa, theta, xi, rho, r,
                                   T, n_steps, n_paths, rng=rng)

    S_T = S[:, -1]
    true_mean = S0 * np.exp(r * T)
    empirical_mean = S_T.mean()

    se_approx = np.sqrt(theta) * S0 * np.exp(r * T) * np.sqrt(T) / np.sqrt(n_paths)

    assert abs(empirical_mean - true_mean) < 4 * se_approx
    
def test_qe_regime2_zero_fraction():
    """
    Regression guard against a regime-1/regime-2 swap bug. Moments-only tests
    (test_qe_moments_feller_violated) can't distinguish 'correct regime-2 sampling'
    from 'branches swapped but moments still roughly match', since both branches
    are moment-matched to the same (m, s2) by construction. This checks the
    fraction of exact zeros in v_next against p, computed from the SAME
    cir_qe_regime_params call the sampler itself uses, so a swap or branch bug
    shows up directly, independent of whether the Day 17 moment formulas are right.
    """
    kappa, theta, xi, v0, dt = 1.0, 0.04, 1.5, 0.005, 0.5  # same as feller_violated test
    n_paths = 200_000

    v_t = np.full(n_paths, v0)
    rng = np.random.default_rng(0)

    mask_1, a, b2, p, beta = cir_qe_regime_params(v_t, kappa, theta, xi, dt)

    # sanity: this parameter set should be deep enough in regime 2 that the
    # zero-fraction is actually large enough to be a meaningful check, not noise
    assert mask_1.mean() < 0.05  # almost all paths in regime 2
    assert p[0] > 0.1            # p is the same scalar across all paths here, since v_t is constant

    v_next = sample_cir_qe_step(v_t, kappa, theta, xi, dt, rng=rng)

    empirical_zero_frac = (v_next == 0.0).mean()
    expected_p = p[0]

    # binomial proportion standard error, same logic as any MC convergence check
    se = np.sqrt(expected_p * (1 - expected_p) / n_paths)

    assert abs(empirical_zero_frac - expected_p) < 4 * se


def test_heston_char_func_long_maturity_stability():
    """
    Regression guard against the 'Little Heston Trap' (Albrecher et al. 2007):
    the naive Riccati root produces a discontinuous complex log for long
    maturities, this checks phi is smooth (no jump) across a fine tau grid
    at parameters known to trigger the naive-formula discontinuity.
    """
    S0, v0 = 100.0, 0.04
    kappa, theta, xi, rho, r = 1.5, 0.04, 0.9, -0.7, 0.05  # high xi, long horizon stresses this
    u_test = 5.0  # a single, moderately large u also stresses this more than u near 0

    taus = np.linspace(0.1, 10.0, 200)
    phi_vals = np.array([heston_char_func(u_test, S0, v0, kappa, theta, xi, rho, r, tau)
                          for tau in taus])

    # phi should vary smoothly; a branch-cut jump shows up as an abrupt
    # discontinuity in |phi| or its phase between adjacent tau values
    jumps = np.abs(np.diff(phi_vals))
    assert jumps.max() < 10 * np.median(jumps), "discontinuity detected, possible branch-cut issue"