"""
SVI (Stochastic Volatility Inspired) smile parametrization and calibration.

Raw SVI in total implied variance against log-moneyness k = log(K / F):

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

Calibration uses the Zeliade two-stage reduction: for fixed (m, sigma) the
substitution y = (k - m) / sigma makes w linear in (a, d, c) where
d = rho * b * sigma and c = b * sigma, so the inner problem is a constrained
linear least squares and the outer problem is a 2D search over (m, sigma).

Note: Tau not included as not in formula.
"""

from typing import Callable, Literal, NamedTuple
import warnings
import pandas as pd
from data.option_chain import build_slice

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
    X = np.column_stack([np.ones_like(k), y, np.sqrt(y**2 + 1)])
    return X


def reduced_constraints(sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Linear inequalities on (a, d, c) as (A, ub) meaning A @ x <= ub. 

    Encodes b >= 0, |rho| <= 1, w >= 0 everywhere, and the Lee wing bound.
    Derive these in your own total-variance convention, the published factor
    differs by convention.
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

    A, ub = reduced_constraints(sigma)

    x_ls, *_ = np.linalg.lstsq(Xw, ww, rcond=None)

    if np.all(A @ x_ls <= ub):
        a, d, c = x_ls
        return float(a), float(d), float(c), True

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

    a, d, c = res.x
    return float(a), float(d), float(c), bool(res.success)


def recover_raw(a: float, d: float, c: float, m: float, sigma: float) -> SVIParams:
    """Map reduced parameters back to raw SVI: b = c / sigma, rho = d / c."""
    c_floor = 1e-12
    if c < c_floor:
        # flat slice, rho undefined; documented convention rho = 0
        return SVIParams(a=float(a), b=0.0, rho=0.0, m=float(m), sigma=float(sigma))
    
    b = c / sigma
    rho = float(np.clip(d / c, -1.0, 1.0))
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)

# ---------------------------------------------------------------------------
# Arbitrage diagnostics
# ---------------------------------------------------------------------------

def durrleman_g(k: np.ndarray, params: SVIParams) -> np.ndarray:
    """Durrleman's g(k). Butterfly arbitrage is exactly g(k) < 0.

    Built from w, w' and w'' analytically, never from numerical differences of
    a priced call strip.
    """
    k = np.asarray(k, dtype=float)
    w = svi_raw(k, params)
    w1 = svi_raw_first(k, params)
    w2 = svi_raw_second(k, params)
    g = (1 - k*w1/(2*w))**2 - (w1**2 / 4)*(1/w + 1/4) + w2/2
    return g
    


def svi_density(k: np.ndarray, params: SVIParams, tau: float) -> np.ndarray:
    """Risk-neutral density in log-moneyness implied by the slice.

    Validation only, not used by the constraint. Flat-vol input must reproduce
    the lognormal density exactly and integrate to one.
    """
    raise NotImplementedError


def calendar_violation(
    surface: dict[float, SVIFitResult],
    k_grid: np.ndarray,
    data_ranges: dict[float, tuple[float, float]] | None = None,
) -> tuple[float, np.ndarray, float]:
    """Worst calendar-arbitrage violation across a fitted surface.

    Calendar arbitrage-free means total variance is non-decreasing in maturity
    at fixed k: w(k, tau_1) <= w(k, tau_2) for tau_1 < tau_2. Slices are fit
    independently, so two curves can cross (especially in the wings); this
    detects it.

    Returns (max_violation, W, max_violation_in_data). W is the (n_k, n_tau)
    total-variance grid for diagnostics. max_violation is the worst crossing
    anywhere on k_grid; max_violation_in_data is the worst crossing restricted
    to the region where BOTH slices in the crossing pair have data (their
    strike ranges overlap that k). A crossing in-data is a real arbitrage; one
    only in the extrapolation region is far less alarming, since SVI is
    guessing there. If data_ranges is None, max_violation_in_data is NaN.
    """
    taus = sorted(surface.keys())
    W = np.full((len(k_grid), len(taus)), np.nan)
    for j, tau in enumerate(taus):
        W[:, j] = svi_raw(k_grid, surface[tau].params)

    dW = np.diff(W, axis=1)                       # w(tau_{j+1}) - w(tau_j)
    violation = np.maximum(0.0, -dW)              # (n_k, n_tau-1), positive where w dropped
    max_violation = float(np.max(violation))

    if data_ranges is None:
        return max_violation, W, float("nan")

    # in-data mask: for each adjacent tau pair (column j of `violation`),
    # a k point counts only if BOTH slices tau_j and tau_{j+1} have data there
    in_data = np.zeros_like(violation, dtype=bool)
    for j in range(len(taus) - 1):
        lo_j, hi_j = data_ranges[taus[j]]
        lo_j1, hi_j1 = data_ranges[taus[j + 1]]
        lo = max(lo_j, lo_j1)                     # overlap of the two data ranges
        hi = min(hi_j, hi_j1)
        in_data[:, j] = (k_grid >= lo) & (k_grid <= hi)

    masked = np.where(in_data, violation, 0.0)
    max_violation_in_data = float(np.max(masked))
    return max_violation, W, max_violation_in_data


# ---------------------------------------------------------------------------
# Outer problem
# ---------------------------------------------------------------------------


def butterfly_penalty(k_grid: np.ndarray, params: SVIParams) -> float:
    """Squared hinge sum(max(0, -g(k_i))^2) on a grid fixed across iterations.

    Fixed so the penalty varies smoothly with (m, sigma). Density near the
    money matters more than range.
    """
    k = np.asarray(k_grid, dtype=float)
    g = durrleman_g(k, params)
    violation = np.maximum(0.0, -g)
    return float(np.sum(violation**2))


def outer_objective(
    x: np.ndarray,
    k: np.ndarray,
    w_market: np.ndarray,
    k_grid: np.ndarray,
    lam: float,
    weights: np.ndarray | None = None,
) -> float:
    """Objective over x = (m, sigma): inner solve, then RMSE plus penalty.

    The inner solve minimizes weighted RMSE. Deliberate, and the reason the accepted
    fit is gated on g >= 0 directly rather than on a small residual penalty.
    """
    # x is the outer search point, unpack the two parameters
    m, sigma = x

    # inner solve: best (a, d, c) for this fixed (m, sigma)
    a, d, c, ok = solve_inner(k, w_market, m, sigma, weights)

    # --- fit term: RMSE in reduced coordinates ---
    # rebuild the design matrix so we can reconstruct fitted w
    if not ok:
        return np.inf
    
    X = reduced_design_matrix(k, m, sigma)          # shape (n, 3)
    w_fit = X @ np.array([a, d, c])                        # fitted total variance
    resid = w_fit - w_market

    if weights is None:
        rmse = np.sqrt( np.mean( resid**2 ) )
    else:
        rmse = np.sqrt( np.sum(weights * resid**2) / np.sum(weights) )

    # --- penalty term: butterfly arbitrage ---
    params = recover_raw(a, d, c, m, sigma)         # need SVIParams for g
    penalty = butterfly_penalty(k_grid, params)

    # combined score handed back to Nelder-Mead
    return rmse + lam * penalty


def fit_slice(
    k: np.ndarray,
    w_market: np.ndarray,
    weights: np.ndarray | None = None,
    lam: float = 0.0,
    k_grid: np.ndarray | None = None,
    method: Literal["Nelder-Mead", "Powell"] = "Nelder-Mead",
) -> SVIFitResult:
    """Calibrate one maturity slice via the two-stage reduction.

    Default lam = 0 supports continuation: fit unpenalized, gate on g, re-fit
    with ramped lam warm-started from that solution only if the gate fails.
    """
    # --- 0. coerce inputs ---
    k = np.asarray(k, dtype=float)
    w_market = np.asarray(w_market, dtype=float)

    # --- 1. build the FIXED penalty grid, once ---
    # fixed across every outer iteration and every start, so the penalty
    # is a smooth function of (m, sigma). dense, denser near the money.
    if k_grid is None:
        k_lo, k_hi = k.min(), k.max()
        margin = 0.1 * (k_hi - k_lo)
        k_grid = np.linspace(k_lo - margin, k_hi + margin, 400)

    # --- 2. assemble spread-out starting points ---
    # deterministic grid over the plausible region, not jitter round one point.
    # every sigma start must sit above the ~1e-3 floor.
    m_starts     = np.array([ -0.10, -0.05, 0.0, 0.05 ])       # vertex left/centre/right
    sigma_starts = np.array([ 0.05, 0.20 ])                   # sharp / rounded
    starts = [(m0, s0) for m0 in m_starts for s0 in sigma_starts]          # or however many -> n_starts

    # --- 3. multi-start local search, keep the best ---
    best_result = None
    best_score  = np.inf
    total_evals = 0
    m_lo, m_hi = -0.5, 0.5
    sigma_floor, sigma_hi = 1e-3, 1.0

    for (m0, sigma0) in starts:
        res = minimize(
            outer_objective,
            x0 = (m0, sigma0),
            args = (k, w_market, k_grid, lam, weights),
            method = method,
            bounds = [ (m_lo, m_hi), (sigma_floor, sigma_hi) ],   # keep sigma > floor
        )
        total_evals += res.nfev
        if res.fun < best_score:
            best_score  = res.fun
            best_result = res

    if best_result is None:
        raise RuntimeError("fit_slice: all starts failed")
    
    # --- 4. recover the winning slice ---
    m, sigma   = best_result.x
    a, d, c, ok = solve_inner(k, w_market, m, sigma, weights)
    if not ok:
        raise RuntimeError("fit_slice: inner solve failed at the winning point")
    params      = recover_raw(a, d, c, m, sigma)

    # --- 5. package the result ---
    rmse            = outer_objective(best_result.x, k, w_market, k_grid, 0.0, weights)     # recompute, penalty stripped out
    g_vals = durrleman_g(k_grid, params)
    max_violation = float(np.maximum(0.0, -np.min(g_vals)))
    arbitrage_free = max_violation == 0.0

    return SVIFitResult(
        params         = params,
        rmse           = rmse,
        max_violation  = max_violation,
        arbitrage_free = arbitrage_free,
        n_outer_evals  = total_evals,
    )


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
    k_lo, k_hi = k_data_range

    # dense grid over the data range only
    k_data = np.linspace(k_lo, k_hi, n_dense)
    g_data = durrleman_g(k_data, params)
    clean_on_data = bool(np.min(g_data) >= -1e-12)

    # dense grid over the wider extrapolation range
    k_wide = np.linspace(-k_extrap, k_extrap, n_dense)
    g_wide = durrleman_g(k_wide, params)
    # the worst dip anywhere on the wide grid
    min_g = float(np.min(g_wide))
    clean_on_extrap = bool(min_g >= -1e-12)

    return clean_on_data, clean_on_extrap, float(min_g)


# ---------------------------------------------------------------------------
# Surface level
# ---------------------------------------------------------------------------

def fit_surface(
    all_opts: pd.DataFrame,
    spot: float,
    targets: list[int],
    r: float,
    bsm_implied_vol,
    bsm_vega,
    lam: float = 0.0,
    return_ranges: bool = False,
    max_staleness_days=None,
) -> dict[float, SVIFitResult]:
    """Fit an SVI slice per target maturity, skipping illiquid ones.

    Sweeps build_slice + fit_slice over targets, shortest first (butterfly
    failures surface on short slices). Keyed by real tau, not target-days,
    since the calendar condition is stated in tau and actual DTE differs from
    the requested target. Illiquid expiries (build_slice raises) are warned and
    skipped, so the surface is built from whatever liquid maturities succeed.
    """
    # sort targets ascending so we fit shortest maturities first
    surface = {}
    ranges = {}
    for target in sorted(targets):
        try:
            s = build_slice(all_opts, spot, target, r, bsm_implied_vol, bsm_vega,
                    max_staleness_days=max_staleness_days)
        except ValueError as e:
            warnings.warn(f"fit_surface: skipping target {target}, {e}")
            continue
        result = fit_slice(s["k"], s["w"], weights=s["weights"], lam=lam)
        tau = s["T"]
        surface[tau] = result
        ranges[tau] = (float(s["k"].min()), float(s["k"].max()))
    if len(surface) < 2:
        warnings.warn("fit_surface: fewer than 2 liquid slices, calendar check not meaningful")
    return (surface, ranges) if return_ranges else surface


def svi_smile_interpolator(
    fit: dict[float, SVIParams],
) -> Callable[[np.ndarray, float], np.ndarray]:
    """Return w(k, tau) interpolating in total variance across fitted slices.

    Interpolation is in w, not implied vol, so calendar monotonicity survives
    linear interpolation between slices.
    """
    raise NotImplementedError