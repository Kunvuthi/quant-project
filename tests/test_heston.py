import numpy as np
import pytest
from models.heston import sample_cir_qe_step, simulate_heston_paths


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