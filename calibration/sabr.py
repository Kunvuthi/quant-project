import numpy as np
from scipy.optimize import least_squares
from typing import NamedTuple

from models.sabr import sabr_vol


class SABRParams(NamedTuple):
    """Calibrated SABR parameters for one maturity slice. beta is fixed, not fitted."""

    alpha: float
    beta: float
    rho: float
    nu: float

def calibrate_sabr(
    k: np.ndarray,
    iv_market: np.ndarray,
    F: float,
    T: float,
    beta: float = 1.0,
    weights: np.ndarray | None = None,
) -> tuple[SABRParams, float]:
    """Fit (alpha, rho, nu) to a market IV slice, beta fixed.

    Straight 3D nonlinear least squares on implied vols (no reduction like SVI).
    Returns (SABRParams, rmse).
    """
    k = np.asarray(k, dtype=float)
    iv_market = np.asarray(iv_market, dtype=float)

    # sqrt-weights folded into the residual, or ones if unweighted
    if weights is None:
        s = np.ones_like(iv_market)
    else:
        s = np.sqrt(np.asarray(weights, dtype=float))

    # strikes from log-moneyness. Note the sign convention: your slice k is
    # log(K/F), so K = F * exp(k). sabr_vol uses log(F/K) internally, which is
    # its own concern; you just hand it the correct K.
    K = F * np.exp(k)

    def residual(theta: np.ndarray) -> np.ndarray:
        alpha, rho, nu = theta
        # model IV at every strike, then weighted residual vector
        fit_sabr = sabr_vol(K, F, T, alpha, beta, rho, nu)
        res = s * (fit_sabr - iv_market)
        return res

    # starting point: alpha0 ~ ATM vol * F^(1-beta); with beta=1, alpha0 ~ ATM vol
    # use the near-money market IV as the ATM proxy
    atm_iv = iv_market[np.argmin(np.abs(k))]
    x0 = np.array([atm_iv * F**(1.0 - beta), -0.3, 0.5])

    alpha_scale = F ** (1.0 - beta)          # alpha lives in vol*F^(1-beta) units
    bounds = ([1e-6, -0.999, 1e-6],
              [5.0 * alpha_scale, 0.999, 10.0])

    result = least_squares(residual, x0, bounds=bounds, method="trf")

    alpha, rho, nu = result.x
    params = SABRParams(alpha=float(alpha), beta=float(beta),
                        rho=float(rho), nu=float(nu))

    # unweighted RMSE from a clean re-eval at the solution (result.fun is weighted)
    model_iv = sabr_vol(K, F, T, alpha, beta, rho, nu)
    rmse = float(np.sqrt(np.mean((model_iv - iv_market)**2)))

    return params, rmse