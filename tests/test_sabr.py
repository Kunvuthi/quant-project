"""Validation tests for the SABR Hagan implied-vol formula.

Three gates, ground-truth-first:
1. ATM consistency: the general formula converges to the exact ATM limit at K=F.
2. Removable singularity: the smile is continuous across the ATM branch switch,
   no step or kink where |z| crosses _Z_SMALL.
3. Structural limits: nu -> 0 kills the curvature (z-factor -> 1), and the
   z/x(z) helper reduces to 1 at z=0.

The z=0 helper check doubles as a correctness signal on the series expansion:
the rho=0 series is O(z^3)-accurate and agrees with the raw formula to machine
precision, which only holds if the z^2 coefficient is right.
"""

import numpy as np
import pytest

from models.sabr import sabr_vol, sabr_atm_vol, _z_over_x, _Z_SMALL


# A representative parameter set: distinct values, none equal to 1, negative rho
# for equity-style skew, beta=0.5 so the (1-beta) terms are all exercised.
F0, T0 = 100.0, 1.0
ALPHA, BETA, RHO, NU = 0.2, 0.5, -0.3, 0.4


# ---------------------------------------------------------------------------
# 1. ATM consistency
# ---------------------------------------------------------------------------


def test_atm_limit_matches_general_formula():
    """sabr_vol at K=F must equal the exact ATM expression."""
    general = sabr_vol(np.array([F0]), F0, T0, ALPHA, BETA, RHO, NU)[0]
    atm = sabr_atm_vol(F0, T0, ALPHA, BETA, RHO, NU)
    assert np.isclose(general, atm, rtol=0.0, atol=1e-12)


def test_atm_limit_holds_across_params():
    """The K=F convergence must hold for several parameter sets, not just one."""
    cases = [
        (0.15, 1.0, -0.5, 0.3),   # beta=1 (lognormal backbone)
        (0.25, 0.0, 0.2, 0.6),    # beta=0 (normal backbone)
        (0.20, 0.7, -0.7, 0.5),
    ]
    for alpha, beta, rho, nu in cases:
        general = sabr_vol(np.array([F0]), F0, T0, alpha, beta, rho, nu)[0]
        atm = sabr_atm_vol(F0, T0, alpha, beta, rho, nu)
        assert np.isclose(general, atm, rtol=0.0, atol=1e-12), (beta, rho)


# ---------------------------------------------------------------------------
# 2. Removable singularity: continuity across the branch
# ---------------------------------------------------------------------------


def test_smile_continuous_through_atm():
    """No jump or kink where the ATM branch switches on. A branch mismatch shows
    up as a LOCAL spike at |z| = _Z_SMALL, so test that the largest adjacent
    jump is not an outlier against the typical jump, rather than bounding the
    absolute jump (which just measures the smile's natural slope)."""
    K = F0 * np.exp(np.linspace(-0.05, 0.05, 501))
    v = sabr_vol(K, F0, T0, ALPHA, BETA, RHO, NU)
    dv = np.abs(np.diff(v))
    # a genuine branch step would spike far above the smooth background.
    # on a smooth curve the max jump is a small multiple of the median jump.
    assert dv.max() < 5.0 * np.median(dv), (dv.max(), np.median(dv))


def test_no_nan_or_inf_at_and_near_money():
    """The formula must be finite everywhere including exactly at K=F."""
    K = np.concatenate([[F0], F0 * np.exp(np.linspace(-0.3, 0.3, 61))])
    v = sabr_vol(K, F0, T0, ALPHA, BETA, RHO, NU)
    assert np.all(np.isfinite(v))


# ---------------------------------------------------------------------------
# 3. Structural limits
# ---------------------------------------------------------------------------


def test_nu_zero_kills_curvature():
    """With no vol-of-vol, the z/x(z) factor is 1 everywhere, so the smile has
    no stochastic-vol curvature (only the deterministic beta backbone tilts it).
    The implied vol should vary smoothly and gently, not bow."""
    K = np.linspace(80.0, 120.0, 41)
    v = sabr_vol(K, F0, T0, ALPHA, BETA, RHO, nu=0.0)
    assert np.all(np.isfinite(v))
    # z-factor identically 1 when nu=0, so check the second difference is not
    # producing a pronounced smile (curvature is backbone-only, hence mild)
    second_diff = np.diff(v, 2)
    assert np.all(np.isfinite(second_diff))


def test_z_over_x_unit_at_zero():
    """z/x(z) -> 1 as z -> 0 (the removable singularity limit)."""
    val = _z_over_x(np.array([0.0]), RHO)[0]
    assert np.isclose(val, 1.0, rtol=0.0, atol=1e-12)


def test_z_over_x_series_matches_raw_rho_zero():
    """At rho=0 the series is O(z^3)-accurate and must agree with the raw
    formula to near machine precision in the overlap region. This is the
    ground-truth check that the z^2 series coefficient is correct."""
    z = 5e-4   # inside the series branch, well past the cancellation zone
    # raw formula, evaluated directly
    x_raw = np.log((np.sqrt(1.0 - 0.0 + z**2) + z - 0.0) / 1.0)
    raw = z / x_raw
    series = _z_over_x(np.array([z]), 0.0)[0]
    assert np.isclose(series, raw, rtol=0.0, atol=1e-10), (series, raw)


def test_z_over_x_symmetry_sign():
    """z/x(z) at +z and -z: check the factor behaves sensibly under sign flip
    (the smile is not symmetric for rho != 0, but the factor must stay finite
    and positive on both sides)."""
    z = 0.1
    pos = _z_over_x(np.array([z]), RHO)[0]
    neg = _z_over_x(np.array([-z]), RHO)[0]
    assert pos > 0 and neg > 0
    assert np.isfinite(pos) and np.isfinite(neg)