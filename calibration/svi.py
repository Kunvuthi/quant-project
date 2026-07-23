"""
SVI (Stochastic Volatility Inspired) smile parametrization and calibration.

Raw SVI in total implied variance against log-moneyness k = log(K / F):

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

Calibration uses the Zeliade two-stage reduction: for fixed (m, sigma) the
substitution y = (k - m) / sigma makes w linear in (a, d, c) where
d = rho * b * sigma and c = b * sigma, so the inner problem is a constrained
linear least squares and the outer problem is a 2D search over (m, sigma).
"""

from typing import Callable, Literal, NamedTuple
import warnings

import numpy as np
from scipy.optimize import minimize


class SVIParams(NamedTuple):
    """Raw SVI parameters for a single maturity slice."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float


class SVIFitResult(NamedTuple):
    """Outcome of a single-slice calibration."""

    params: SVIParams
    rmse: float
    max_violation: float
    arbitrage_free: bool
    n_outer_evals: int


# ---------------------------------------------------------------------------
# Parametrization and derivatives
# ---------------------------------------------------------------------------


def svi_raw(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """Total implied variance w(k) under raw SVI."""
    k = np.asarray(k, dtype=float)
    w_k = params.a + params.b*( params.rho*(k-params.m) + np.sqrt((k-params.m)**2 + params.sigma**2))
    return w_k


def svi_raw_first(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """First derivative dw/dk, analytic."""
    k = np.asarray(k, dtype=float)
    dw = params.b*( params.rho + (k-params.m)/np.sqrt((k-params.m)**2 + params.sigma**2) )
    return dw


def svi_raw_second(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """Second derivative d2w/dk2, analytic.

    Strictly positive for admissible parameters, which is why the sign of
    Durrleman's g is driven entirely by the -w'^2 (1/w + 1/4) / 4 term.
    """
    k = np.asarray(k, dtype=float)
    d2w = params.b*(params.sigma**2)/(((k-params.m)**2 + params.sigma**2))**(3/2)
    return d2w


# ---------------------------------------------------------------------------
# Inner problem: linear in (a, d, c) for fixed (m, sigma)
# ---------------------------------------------------------------------------


def reduced_design_matrix(k: np.ndarray, m: float, sigma: float) -> np.ndarray:
    """Design matrix for w = a + d * y + c * sqrt(y^2 + 1), y = (k - m) / sigma.

    Columns [1, y, sqrt(y^2 + 1)], shape (len(k), 3).
    """
    k = np.asarray(k, dtype=float)
    y = (k-m) / sigma
    X = np.column_stack([np.ones_like(len(k)), y, np.sqrt(y**2 + 1)])
    return X


def reduced_constraints(sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear inequalities on (a, d, c) as (A, ub) meaning A @ x <= ub. 

    Encodes b >= 0, |rho| <= 1, w >= 0 everywhere, and the Lee wing bound.
    Derive these in your own total-variance convention, the published factor
    differs by convention and a wrong tau scaling fails silently.
    """
    A = np.array([
    [0.0,  0.0, -1.0],   # b >= 0            ->  -c <= 0
    [0.0,  1.0, -1.0],   # rho <= 1          ->  d - c <= 0
    [0.0,  -1.0, -1.0],           # rho >= -1         -> -d - c <= 0
    [-1.0,  0.0, 0.0],           # w >= 0 (relaxed)  ->  -a <= 0
    [0.0,  1.0, 1.0],           # Lee, right wing   -> (c + d) / sigma <= 2
    [0.0,  -1.0, 1.0],           # Lee, left wing    -> (c - d) / sigma <= 2
    ], dtype=float)
    ub = np.array([0.0, 0.0, 0.0, 0.0, 2*sigma, 2*sigma], dtype=float)
    return A, ub


def solve_inner(
    k: np.ndarray,
    w_market: np.ndarray,
    m: float,
    sigma: float,
    tau: float,
    weights: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Constrained weighted linear least squares for (a, d, c) at fixed (m, sigma).

    Must return a feasible point, not merely near-feasible, since recovery
    divides by c.
    """
    X = reduced_design_matrix(k, m, sigma)
    w = np.asarray(w_market, dtype=float)

    # fold weights into the data, objective becomes plain least squares
    if weights is None:
        Xw, ww = X, w
    else:
        s = np.sqrt(np.asarray(weights, dtype=float))
        Xw, ww = X * s[:, None], w * s

    XtX = Xw.T @ Xw
    Xtw = Xw.T @ ww

    def obj(x: np.ndarray) -> float:
        r = Xw @ x - ww
        return float(r @ r)

    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * (XtX @ x - Xtw)

    A, ub = reduced_constraints(sigma, tau)

    x_ls, *_ = np.linalg.lstsq(Xw, ww, rcond=None)

    if np.all(A @ x_ls <= ub):
        a, d, c = x_ls
        return float(a), float(d), float(c)

    res = minimize(
        obj, x_ls,
        jac=grad,
        method="SLSQP",
        constraints=[{
            "type": "ineq",            # SLSQP wants g(x) >= 0
            "fun": lambda x: ub - A @ x,
            "jac": lambda x: -A,
        }],
        options={"ftol": 1e-12, "maxiter": 200},
    )

    if not res.success:
        warnings.warn(f"solve_inner: SLSQP failed ({res.message})", RuntimeWarning)

    a, d, c = res.x
    return float(a), float(d), float(c)


def recover_raw(a: float, d: float, c: float, m: float, sigma: float) -> SVIParams:
    """Map reduced parameters back to raw SVI: b = c / sigma, rho = d / c."""
    b = c / sigma
    rho = d / c
    return float(b), float(rho)


# ---------------------------------------------------------------------------
# Arbitrage diagnostics
# ---------------------------------------------------------------------------


def durrleman_g(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """Durrleman's g(k). Butterfly arbitrage is exactly g(k) < 0.

    Built from w, w' and w'' analytically, never from numerical differences of
    a priced call strip.
    """
    raise NotImplementedError


def svi_density(k: np.ndarray, params: SVIParams, tau: float) -> np.ndarray:
    """Risk-neutral density in log-moneyness implied by the slice.

    Validation only, not used by the constraint. Flat-vol input must reproduce
    the lognormal density exactly and integrate to one.
    """
    raise NotImplementedError


def calendar_violation(
    slices: dict[float, SVIParams],
    k_grid: np.ndarray,
) -> float:
    """Worst violation of monotonicity of w in maturity across fitted slices.

    Zero means calendar arbitrage free on k_grid. Slice-by-slice raw SVI does
    not guarantee this, it has to be checked after the fact.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Outer problem
# ---------------------------------------------------------------------------


def butterfly_penalty(params: SVIParams, k_grid: np.ndarray) -> float:
    """Squared hinge sum(max(0, -g(k_i))^2) on a grid fixed across iterations.

    Fixed so the penalty varies smoothly with (m, sigma). Density near the
    money matters more than range.
    """
    raise NotImplementedError


def outer_objective(
    x: np.ndarray,
    k: np.ndarray,
    w_market: np.ndarray,
    tau: float,
    k_grid: np.ndarray,
    lam: float,
    weights: np.ndarray | None = None,
) -> float:
    """Objective over x = (m, sigma): inner solve, then RMSE plus penalty.

    The composite is slightly inconsistent, the inner solve minimizes pure RMSE
    while this sees RMSE plus penalty. Deliberate, and the reason the accepted
    fit is gated on g >= 0 directly rather than on a small residual penalty.
    """
    raise NotImplementedError


def fit_slice(
    k: np.ndarray,
    w_market: np.ndarray,
    tau: float,
    weights: np.ndarray | None = None,
    lam: float = 0.0,
    k_grid: np.ndarray | None = None,
    n_starts: int = 5,
    method: Literal["nelder-mead", "powell"] = "nelder-mead",
) -> SVIFitResult:
    """Calibrate one maturity slice via the two-stage reduction.

    Default lam = 0 supports continuation: fit unpenalized, gate on g, re-fit
    with ramped lam warm-started from that solution only if the gate fails.
    """
    raise NotImplementedError


def verify_fit(
    params: SVIParams,
    k_data_range: tuple[float, float],
    k_extrap: float = 2.0,
    n_dense: int = 20000,
) -> tuple[bool, bool, float]:
    """Post-acceptance gate on a much denser grid than the optimizer used.

    Returns (clean_on_data, clean_on_extrapolation, min_g). Reported separately,
    a fit can be usable on quoted strikes and useless in the wings.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Surface level
# ---------------------------------------------------------------------------


def fit_surface(
    slices: dict[float, tuple[np.ndarray, np.ndarray]],
    lam: float = 0.0,
    order: Literal["ascending-tau", "descending-tau"] = "ascending-tau",
) -> dict[float, SVIFitResult]:
    """Fit each maturity independently, shortest first.

    Shortest first because small w with steep w' is where butterfly failures
    actually occur, and knowing early beats a tidy loop.
    """
    raise NotImplementedError


def svi_smile_interpolator(
    fit: dict[float, SVIParams],
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Return w(k, tau) interpolating in total variance across fitted slices.

    Interpolation is in w, not implied vol, so calendar monotonicity survives
    linear interpolation between slices.
    """
    raise NotImplementedError