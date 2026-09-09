"""Calibration of jump-diffusion models (Merton, Kou) to a market IV slice.

Both fit in implied-vol space: price the strike strip via COS, invert each price
to Black-Scholes implied vol, least-squares against the market IVs. Forward-
consistent pricing (S0=F, r=q=R so the internal forward equals the parity
forward F, invert with carry b=0), so the untrusted spot never enters and the
parity-implied forward flows straight through.

Merton and Kou share the residual-and-least-squares machinery; only the pricer,
parameter vector, and bounds differ. The shared part lives in _calibrate_jump.
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Callable, NamedTuple

from calibration.implied_vol import bsm_implied_vol
from pricing.fourier import cos_call_price, merton_cos_price, kou_cos_price, vg_cos_price, bates_cos_price

class HestonParams(NamedTuple):
    v0: float
    kappa_v: float
    theta: float
    xi: float
    rho: float


class MertonParams(NamedTuple):
    sigma: float
    lam: float
    mu_j: float
    delta_j: float


class KouParams(NamedTuple):
    sigma: float
    lam: float
    p: float
    eta1: float
    eta2: float
    

class BatesParams(NamedTuple):
    v0: float
    kappa_v: float
    theta: float
    xi: float
    rho: float
    lam: float
    mu_j: float
    delta_j: float
    
    
class VGParams(NamedTuple):
    sigma: float
    nu: float
    theta: float


# ---------------------------------------------------------------------------
# Shared machinery
# ---------------------------------------------------------------------------

def _model_ivs(
    price_fn: Callable[[float], float],
    strikes: np.ndarray,
    F: float,
    T: float,
    r: float,
) -> np.ndarray:
    """Price each strike (forward-consistent) and invert to BSM implied vol.

    price_fn(K) returns the model call price at strike K, priced with S0=F and
    r=q so the internal forward is F. Inversion uses b=0 (at the forward).
    Returns model IVs aligned with strikes; NaN where inversion fails.
    """
    ivs = np.full(len(strikes), np.nan)
    for i, K in enumerate(strikes):
        price = price_fn(K)
        ivs[i] = bsm_implied_vol(price, F, K, T, r, "call", b=0.0)
    return ivs


def _calibrate_jump(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    bounds: tuple[list[float], list[float]],
) -> tuple[np.ndarray, bool]:
    """Run the shared least-squares. Returns (fitted_theta, success).
    RMSE is computed by the caller, where the market and model IVs are in scope."""
    result = least_squares(residual_fn, x0, bounds=bounds, method="trf")
    return result.x, result.success


# ---------------------------------------------------------------------------
# Heston
# ---------------------------------------------------------------------------

def calibrate_heston(
    k: np.ndarray,
    iv_market: np.ndarray,
    F: float,
    T: float,
    r: float,
    weights: np.ndarray | None = None,
) -> tuple[HestonParams, float]:
    """Fit Heston (v0, kappa_v, theta, xi, rho) to a market IV slice.

    Forward-consistent pricing. cos_call_price uses a single rate for BOTH drift
    and discount (Heston has no separate q), so to keep the internal forward at
    F we price at rate 0 (S0 = F, zero drift -> forward = F, undiscounted) and
    invert at rate 0 to match. The implied vol is invariant to the discounting
    convention as long as pricing and inversion use the SAME rate; using 0 for
    both is the clean self-consistent choice here. Contrast Merton/Kou/Bates/VG,
    which carry a separate q and so use the (F, r, r) convention with rate-r
    inversion.
    """
    strikes = F * np.exp(k)
    s = np.ones_like(iv_market) if weights is None else np.sqrt(np.asarray(weights, float))

    def residual(p: np.ndarray) -> np.ndarray:
        v0, kappa_v, theta, xi, rho = p
        # rate 0: forward-measure, undiscounted, internal forward = F
        price_fn = lambda K: cos_call_price(F, v0, kappa_v, theta, xi, rho, 0.0, T, K)
        model_iv = _model_ivs(price_fn, strikes, F, T, r=0.0)   # invert at rate 0 to match
        return s * (model_iv - iv_market)

    #             v0     kappa_v  theta   xi     rho
    x0 = np.array([0.04, 2.0,     0.04,   0.3,  -0.6])
    bounds = ([1e-4, 0.1,  1e-4, 0.01, -0.999],
              [1.0,  10.0, 1.0,  2.0,   0.999])

    p_hat, ok = _calibrate_jump(residual, x0, bounds)

    v0, kappa_v, theta, xi, rho = p_hat
    price_fn = lambda K: cos_call_price(F, v0, kappa_v, theta, xi, rho, 0.0, T, K)
    model_iv = _model_ivs(price_fn, strikes, F, T, r=0.0)
    rmse = float(np.sqrt(np.nanmean((model_iv - iv_market) ** 2)))
    return HestonParams(*p_hat), rmse

# ---------------------------------------------------------------------------
# Merton
# ---------------------------------------------------------------------------

def calibrate_merton(
    k: np.ndarray,
    iv_market: np.ndarray,
    F: float,
    T: float,
    r: float,
    weights: np.ndarray | None = None,
) -> tuple[MertonParams, float]:
    """Fit Merton (sigma, lam, mu_j, delta_j) to a market IV slice.

    Residual in vol space, weighted by sqrt(weights) so least_squares' internal
    square-and-sum recovers the weighted SSQ (same trick as SABR).
    """
    strikes = F * np.exp(k)
    s = np.ones_like(iv_market) if weights is None else np.sqrt(np.asarray(weights, float))

    def residual(theta: np.ndarray) -> np.ndarray:
        sigma, lam, mu_j, delta_j = theta
        price_fn = lambda K: merton_cos_price(F, r, r, T, sigma, lam, mu_j, delta_j, K, "call")
        model_iv = _model_ivs(price_fn, strikes, F, T, r)
        return s * (model_iv - iv_market)

    # x0: sigma near ATM vol, modest jumps, mu_j < 0 for equity skew
    x0 = np.array([0.15, 0.5, -0.10, 0.15])
    # bounds: sigma>0, lam>=0, mu_j free-ish, delta_j>0
    bounds = ([0.01, 0.0, -1.0, 0.01], [1.0, 5.0, 0.5, 1.0])

    theta, ok = _calibrate_jump(residual, x0, bounds)

    # unweighted vol-point RMSE at the solution
    sigma, lam, mu_j, delta_j = theta
    price_fn = lambda K: merton_cos_price(F, r, r, T, sigma, lam, mu_j, delta_j, K, "call")
    model_iv = _model_ivs(price_fn, strikes, F, T, r)
    rmse = float(np.sqrt(np.nanmean((model_iv - iv_market) ** 2)))

    return MertonParams(*theta), rmse


# ---------------------------------------------------------------------------
# Kou
# ---------------------------------------------------------------------------

def calibrate_kou(
    k: np.ndarray,
    iv_market: np.ndarray,
    F: float,
    T: float,
    r: float,
    weights: np.ndarray | None = None,
    L: float = 14.0,
) -> tuple[KouParams, float]:
    """Fit Kou (sigma, lam, p, eta1, eta2) to a market IV slice.

    eta1 bounded > 1 (compensator convergence, the constraint from Wed). L
    defaults to 14 for the fat-tail wing accuracy (Kou needs a wider COS range
    than Merton).
    """
    strikes = F * np.exp(k)
    s = np.ones_like(iv_market) if weights is None else np.sqrt(np.asarray(weights, float))

    def residual(theta: np.ndarray) -> np.ndarray:
        sigma, lam, p, eta1, eta2 = theta
        price_fn = lambda K: kou_cos_price(F, r, r, T, sigma, lam, p, eta1, eta2, K, "call", L=L)
        model_iv = _model_ivs(price_fn, strikes, F, T, r)
        return s * (model_iv - iv_market)

    # x0: sigma near ATM, moderate jumps, p<0.5 (down-skew), eta1>1, eta2 fatter
    x0 = np.array([0.15, 0.5, 0.3, 5.0, 3.0])
    # bounds: eta1 strictly > 1 (constraint), p in [0,1]
    bounds = ([0.01, 0.0, 0.0, 1.01, 0.5], [1.0, 5.0, 1.0, 50.0, 50.0])

    theta, ok = _calibrate_jump(residual, x0, bounds)

    # unweighted vol-point RMSE at the solution
    sigma, lam, p, eta1, eta2 = theta
    price_fn = lambda K: kou_cos_price(F, r, r, T, sigma, lam, p, eta1, eta2, K, "call", L=L)
    model_iv = _model_ivs(price_fn, strikes, F, T, r)
    rmse = float(np.sqrt(np.nanmean((model_iv - iv_market) ** 2)))

    return KouParams(*theta), rmse


# ---------------------------------------------------------------------------
# Bates
# ---------------------------------------------------------------------------

def calibrate_bates(k, iv_market, F, T, r, weights=None):
    """Fit Bates (8 params) to a market IV slice.

    WARNING: 8 parameters. On a thin slice this is under-determined, jumps and
    diffusion both add short-dated variance, so the fit may not pin them
    uniquely. Read the result as expressive-but-possibly-overfit; unstable or
    boundary-pinned params signal the identifiability problem, not a bug.
    """
    strikes = F * np.exp(k)
    s = np.ones_like(iv_market) if weights is None else np.sqrt(np.asarray(weights, float))

    def residual(p):
        v0, kappa_v, theta, xi, rho, lam, mu_j, delta_j = p
        price_fn = lambda K: bates_cos_price(F, v0, kappa_v, theta, xi, rho, r, r, T, K,
                                             lam, mu_j, delta_j, "call")
        return s * (_model_ivs(price_fn, strikes, F, T, r) - iv_market)

    #      v0     kv    theta  xi    rho    lam   mu_j   delta_j
    x0 = np.array([0.04, 2.0, 0.04, 0.3, -0.6, 0.3, -0.10, 0.15])
    bounds = ([1e-4, 0.1, 1e-4, 0.01, -0.999, 0.0, -1.0, 0.01],
              [1.0, 10.0, 1.0, 2.0, 0.999, 3.0, 0.5, 1.0])

    p_hat, ok = _calibrate_jump(residual, x0, bounds)
    v0, kappa_v, theta, xi, rho, lam, mu_j, delta_j = p_hat
    price_fn = lambda K: bates_cos_price(F, v0, kappa_v, theta, xi, rho, r, r, T, K,
                                        lam, mu_j, delta_j, "call")
    model_iv = _model_ivs(price_fn, strikes, F, T, r)
    rmse = float(np.sqrt(np.nanmean((model_iv - iv_market) ** 2)))
    return BatesParams(*p_hat), rmse

# ---------------------------------------------------------------------------
# Variance Gamma (VG)
# ---------------------------------------------------------------------------

def calibrate_vg(k, iv_market, F, T, r, weights=None):
    """Fit VG (sigma, nu, theta) to a market IV slice."""
    strikes = F * np.exp(k)
    s = np.ones_like(iv_market) if weights is None else np.sqrt(np.asarray(weights, float))

    def residual(theta_vec):
        sigma, nu, theta = theta_vec
        price_fn = lambda K: vg_cos_price(F, r, r, T, sigma, nu, theta, K, "call")
        return s * (_model_ivs(price_fn, strikes, F, T, r) - iv_market)

    x0 = np.array([0.15, 0.3, -0.15])
    # bounds: sigma>0, nu>0, theta free. The VG constraint
    # 1 - theta*nu - 0.5*sigma^2*nu > 0 must hold; keep theta and nu bounded so
    # the optimizer stays feasible (vg_char_func raises otherwise).
    bounds = ([0.01, 0.01, -0.9], [1.0, 2.0, 0.5])

    theta_hat, ok = _calibrate_jump(residual, x0, bounds)
    sigma, nu, theta = theta_hat
    price_fn = lambda K: vg_cos_price(F, r, r, T, sigma, nu, theta, K, "call")
    model_iv = _model_ivs(price_fn, strikes, F, T, r)
    rmse = float(np.sqrt(np.nanmean((model_iv - iv_market) ** 2)))
    return VGParams(*theta_hat), rmse

