import numpy as np
from typing import Literal

def simulate_gbm_terminal(
    S: float, T: float, r: float, sigma: float, N: int,
    antithetic: bool = False, seed: int | None = None,
) -> np.ndarray:
    """Terminal GBM samples S_T via closed form (no time-stepping bias).
    antithetic=True returns the [S+ ; S-] layout. Returns S_T only."""
    if not isinstance(N, int) or N < 1:
        raise ValueError("N must be a positive integer")
    rng = np.random.default_rng(seed)
    if antithetic:
        half = rng.standard_normal(N // 2)
        Z = np.concatenate([half, -half])
    else:
        Z = rng.standard_normal(N)
    return S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
