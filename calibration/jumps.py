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
from typing import NamedTuple

from pricing.fourier import merton_cos_price, kou_cos_price
from calibration._common import _model_ivs, _calibrate_jump


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

