"""Validation tests for the SVI parametrization and its analytic derivatives.

Two independent checks on svi_raw, svi_raw_first, svi_raw_second:

1. Vertex identities at k = m, exact closed forms with no discretization error:
       w(m)  = a + b*sigma
       w'(m) = b*rho
       w''(m)= b/sigma
   Test parameters are all-distinct and none equal to 1, so a swapped b/sigma
   cannot pass by coincidence.

2. Finite-difference consistency: svi_raw_first must be the derivative of
   svi_raw and svi_raw_second the derivative of svi_raw_first. Checked by the
   convergence *rate* (order-2 slope on a log-log error-vs-h sweep), not a
   single-h tolerance a subtly wrong term could slip under.
"""

import numpy as np
import pytest

from calibration.svi import (
    SVIParams,
    svi_raw,
    svi_raw_first,
    svi_raw_second,
)


# All-distinct, none equal to 1, negative rho and m to reflect real equity skew.
PARAMS = SVIParams(a=0.04, b=0.4, rho=-0.3, m=-0.05, sigma=0.15)


# ---------------------------------------------------------------------------
# 1. Vertex identities (exact)
# ---------------------------------------------------------------------------


def test_vertex_value():
    """w(m) = a + b*sigma, exact."""
    p = PARAMS
    got = float(svi_raw(np.array([p.m]), p)[0])
    expected = p.a + p.b * p.sigma
    assert np.isclose(got, expected, rtol=0.0, atol=1e-14)


def test_vertex_first():
    """w'(m) = b*rho, exact."""
    p = PARAMS
    got = float(svi_raw_first(np.array([p.m]), p)[0])
    expected = p.b * p.rho
    assert np.isclose(got, expected, rtol=0.0, atol=1e-14)


def test_vertex_second():
    """w''(m) = b/sigma, exact. Fails loudly if b and sigma are swapped."""
    p = PARAMS
    got = float(svi_raw_second(np.array([p.m]), p)[0])
    expected = p.b / p.sigma
    assert np.isclose(got, expected, rtol=0.0, atol=1e-14)


def test_vertex_second_would_catch_swap():
    """Guard on the test itself: b/sigma and sigma/b must differ materially,
    otherwise test_vertex_second could pass a swapped implementation."""
    p = PARAMS
    assert not np.isclose(p.b / p.sigma, p.sigma / p.b, rtol=1e-6)


# ---------------------------------------------------------------------------
# 2. Finite-difference consistency and convergence rate
# ---------------------------------------------------------------------------

# Vertex, crossover (m +/- sigma), asymptotic wings (m +/- 10 sigma). Both
# signs, to catch a rho-sign error that at the vertex only flips w' but in the
# wings swaps which side is steep.
K_TEST = np.array([
    PARAMS.m,
    PARAMS.m + PARAMS.sigma,
    PARAMS.m - PARAMS.sigma,
    PARAMS.m + 10 * PARAMS.sigma,
    PARAMS.m - 10 * PARAMS.sigma,
])


def _central_first(f, k, h, params):
    return (f(k + h, params) - f(k - h, params)) / (2.0 * h)


def test_first_derivative_matches_fd():
    """svi_raw_first agrees with a central difference of svi_raw."""
    h = 1e-6
    analytic = svi_raw_first(K_TEST, PARAMS)
    fd = _central_first(svi_raw, K_TEST, h, PARAMS)
    # central-difference truncation ~ h^2 * w''' ; at h=1e-6 far below 1e-8,
    # so atol 1e-7 certifies agreement rather than luck.
    assert np.allclose(analytic, fd, rtol=0.0, atol=1e-7)


def test_second_derivative_matches_fd():
    """svi_raw_second agrees with a central difference of svi_raw_first."""
    h = 1e-5
    analytic = svi_raw_second(K_TEST, PARAMS)
    fd = _central_first(svi_raw_first, K_TEST, h, PARAMS)
    assert np.allclose(analytic, fd, rtol=0.0, atol=1e-6)


def test_first_derivative_convergence_rate():
    """Central-difference error in w' falls at order h^2 (slope ~2 log-log).

    A subtly wrong analytic derivative converges to the wrong value, so its
    error plateaus instead of dropping. Rate, not magnitude, is the real test.
    """
    hs = np.array([1e-2, 5e-3, 2.5e-3, 1.25e-3])
    k = PARAMS.m + 0.5 * PARAMS.sigma  # off-vertex, generic point
    analytic = svi_raw_first(np.array([k]), PARAMS)[0]
    errs = np.array([
        abs(_central_first(svi_raw, np.array([k]), h, PARAMS)[0] - analytic)
        for h in hs
    ])
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert np.isclose(slope, 2.0, atol=0.15)


def test_second_derivative_convergence_rate():
    """Central-difference error in w'' falls at order h^2.

    h floored well above the roundoff wall (~eps^(1/4) ~ 1e-4 for a second
    difference); the sweep stays truncation-dominated.
    """
    hs = np.array([2e-2, 1e-2, 5e-3, 2.5e-3])
    k = PARAMS.m + 0.15 * PARAMS.sigma
    analytic = svi_raw_second(np.array([k]), PARAMS)[0]
    errs = np.array([
        abs(_central_first(svi_raw_first, np.array([k]), h, PARAMS)[0] - analytic)
        for h in hs
    ])
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert np.isclose(slope, 2.0, atol=0.2)


# ---------------------------------------------------------------------------
# 3. Structural sanity
# ---------------------------------------------------------------------------


def test_second_derivative_strictly_positive():
    """w'' > 0 everywhere for admissible params, which is why the sign of
    Durrleman's g is driven entirely by its middle term."""
    k = np.linspace(PARAMS.m - 20 * PARAMS.sigma, PARAMS.m + 20 * PARAMS.sigma, 401)
    assert np.all(svi_raw_second(k, PARAMS) > 0.0)


def test_dtype_safety_integer_k():
    """Integer k input must not truncate, per the W5 int64 lesson."""
    k_int = np.array([-1, 0, 1])
    out = svi_raw(k_int, PARAMS)
    assert out.dtype == np.float64
    assert np.isfinite(out).all()
    
# ---------------------------------------------------------------------------
# 4. Durrleman g
# ---------------------------------------------------------------------------

from calibration.svi import durrleman_g

def test_g_flat_slice_is_one():
    """Flat total variance (b = 0) has w' = w'' = 0, so g == 1 everywhere.

    Catches a dropped factor or sign error in the first term, which would stop
    it reducing to 1.
    """
    flat = SVIParams(a=0.04, b=0.0, rho=-0.3, m=-0.05, sigma=0.15)
    k = np.linspace(-1.0, 1.0, 201)
    g = durrleman_g(k, flat)
    assert np.allclose(g, 1.0, rtol=0.0, atol=1e-14)


def test_g_positive_on_admissible_slice():
    """A sane arbitrage-free slice has no butterfly violations: g > 0 on a
    dense grid. Confirms the checker does not produce false positives."""
    k = np.linspace(-1.0, 1.0, 2001)
    g = durrleman_g(k, PARAMS)
    assert np.all(g > 0.0)


def test_g_negative_when_lee_bound_violated():
    """A checker that never fires is not a checker. A slice breaking the Lee
    wing bound b(1+|rho|) <= 2 must drive g negative somewhere."""
    p = SVIParams(a=0.04, b=5.0, rho=-0.7, m=-0.05, sigma=0.15)
    assert p.b * (1 + abs(p.rho)) > 2.0  # confirm the slice is actually illegal
    k = np.linspace(-1.0, 1.0, 2001)
    g = durrleman_g(k, p)
    assert np.min(g) < 0.0
    
# ---------------------------------------------------------------------------
# 5. Optimisation
# ---------------------------------------------------------------------------

from calibration.svi import fit_slice

def test_fit_slice_recovers_known_params():
    """Ground truth: synthesize w from known SVI params, fit, recover them.

    Certifies the whole inner-to-outer chain end to end. Noiseless, so the
    recovered params should match tightly; this is the gate before any real
    CBOE data touches the fitter.
    """
    true = SVIParams(a=0.04, b=0.4, rho=-0.3, m=-0.05, sigma=0.15)
    k = np.linspace(-0.4, 0.4, 25)
    w = svi_raw(k, true)  # noiseless synthetic market

    result = fit_slice(k, w, lam=0.0)
    p = result.params

    assert np.isclose(p.a, true.a, atol=1e-3)
    assert np.isclose(p.b, true.b, atol=1e-3)
    assert np.isclose(p.rho, true.rho, atol=1e-2)
    assert np.isclose(p.m, true.m, atol=1e-2)
    assert np.isclose(p.sigma, true.sigma, atol=1e-2)
    assert result.rmse < 1e-4
    assert result.arbitrage_free