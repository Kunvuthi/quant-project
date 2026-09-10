from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Literal


@dataclass(frozen=True)
class RBergomiParams:
    H: float            # roughness in (0, 0.5)
    eta: float          # vol-of-vol > 0
    rho: float          # price/vol correlation in (-1, 0]
    xi0: Callable[[np.ndarray], np.ndarray]  # forward variance curve t -> xi0(t); flat v0 for synthetic


def volterra_cov(s: np.ndarray, t: np.ndarray, H: float) -> np.ndarray:
    """E[W_tilde_s W_tilde_t] via the integral in the theory cell (2F1 or quadrature).
    Diagonal must return t**(2H); assert this in the harness."""
    ...  # TODO 


def cross_cov(t_vol: np.ndarray, s_bm: np.ndarray, H: float) -> np.ndarray:
    """E[W_tilde_{t_vol} W_{s_bm}] = sqrt(2H)/(H+0.5) * (t**(H+.5) - (t-min)**(H+.5))."""
    ...  # TODO 


def simulate_rbergomi(
    t_grid: np.ndarray,        # (N,), strictly increasing, positive
    params: RBergomiParams,
    n_paths: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Cholesky simulation.
    Returns (S, v, W_tilde), each (n_paths, N). S0 normalised to 1 (forward measure).
    Steps: build 2N x 2N Sigma from volterra_cov / min(s,t) / cross_cov; L = cholesky(Sigma);
    Z ~ N(0, I) shape (n_paths, 2N); map to (W_tilde, W); build dW increments from W;
    v = xi0(t)*exp(eta*W_tilde - .5*eta**2*t**(2H));
    dW_S = rho*dW + sqrt(1-rho**2)*dW_perp using the SAME dW increments;
    Euler log-S with -.5*v*dt drift.
    """
    ...  # TODO 