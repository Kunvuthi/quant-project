import numpy as np
from scipy.optimize import least_squares
from typing import Callable

from calibration.implied_vol import bsm_implied_vol

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