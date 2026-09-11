# First simulator-ONLY model module. No characteristic function and no cumulants,
# by construction: the Volterra kernel is non-Markovian, so there is no affine
# transform and no COS pricing. Monte Carlo only.
import numpy as np
from dataclasses import dataclass
from scipy.special import hyp2f1
from typing import Callable, Literal

@dataclass(frozen=True)
class RBergomiParams:
    H: float            # roughness in (0, 0.5)
    eta: float          # vol-of-vol > 0
    rho: float          # price/vol correlation in (-1, 0]
    xi0: Callable[[np.ndarray], np.ndarray]  # forward variance curve t -> xi0(t); flat v0 for synthetic


def volterra_cov(s: np.ndarray, t: np.ndarray, H: float) -> np.ndarray:
    """E[W_tilde_s W_tilde_t], elementwise over broadcast s, t.

    Closed form with a = min(s,t), b = max(s,t), x = b/a >= 1:
        2H * a**(2H) / (H+0.5) * x**(H-0.5) * hyp2f1(0.5-H, 1, H+1.5, 1/x)
    Diagonal (s=t, x=1) collapses to t**(2H). H=0.5 gives min(s,t).
    """
    s = np.asarray(s, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    a = np.minimum(s, t)          # smaller time = the integral's upper limit
    b = np.maximum(s, t)          # symmetrise so 1/x stays in [0,1] for hyp2f1
    x = b / a
    coeff = 2*H / (H + 0.5)
    # hyp2f1 is vectorised; arg order is (a, b, c, z)
    return coeff * a**(2*H) * x**(H-0.5) * hyp2f1(0.5-H, 1.0, H+1.5, 1.0/x)


def cross_cov(t_vol: np.ndarray, s_bm: np.ndarray, H: float) -> np.ndarray:
    """E[W_tilde_{t_vol} W_{s_bm}], elementwise. NOT symmetric: t_vol is the
    Volterra time, s_bm the plain-BM time.
        sqrt(2H)/(H+0.5) * ( t_vol**(H+0.5) - (t_vol - min(t_vol, s_bm))**(H+0.5) )
    H=0.5 gives min(t_vol, s_bm).
    """
    t_vol = np.asarray(t_vol, dtype=np.float64)
    s_bm = np.asarray(s_bm, dtype=np.float64)
    m = np.minimum(t_vol, s_bm)
    coeff = np.sqrt(2*H) / (H + 0.5)
    return coeff * (t_vol**(H+0.5) - (t_vol - m)**(H+0.5))


def build_joint_covariance(t_grid: np.ndarray, H: float) -> np.ndarray:
    """Assemble the 2N x 2N covariance of [W_tilde(t_1..t_N), W(t_1..t_N)].
    Block layout:
        top-left   (N,N): volterra_cov over the (t_i, t_j) mesh
        bot-right  (N,N): min(t_i, t_j)               [plain BM]
        top-right  (N,N): cross_cov(t_i as vol, t_j as bm)
        bot-left        : top-right.T   (cross_cov is not symmetric)
    """
    N = t_grid.shape[0]
    Ti, Tj = np.meshgrid(t_grid, t_grid, indexing="ij")   # Ti[i,j]=t_i, Tj[i,j]=t_j
    cov_vv = volterra_cov(Ti, Tj, H)
    cov_ww = np.minimum(Ti, Tj)
    cov_vw = cross_cov(Ti, Tj, H)   # (rows = vol time, cols = bm time)
    Sigma = np.empty((2 * N, 2 * N))
    Sigma[:N, :N] = cov_vv
    Sigma[N:, N:] = cov_ww
    Sigma[:N, N:] = cov_vw
    Sigma[N:, :N] = cov_vw.T
    return Sigma


def simulate_rbergomi(
    t_grid: np.ndarray,          # (N,), strictly increasing, strictly positive (excludes 0)
    params: RBergomiParams,
    n_paths: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Cholesky simulation. Returns (S, v, W_tilde), each (n_paths, N).
    S0 = 1 (forward measure, r = 0)."""
    H, eta, rho, xi0 = params.H, params.eta, params.rho, params.xi0
    N = t_grid.shape[0]

    # 1. Correlated Gaussian draw of (W_tilde, W) at the grid times.
    Sigma = build_joint_covariance(t_grid, H)
    L = np.linalg.cholesky(Sigma)                 # raises if Sigma not PD (a diagnostic in itself)
    Z = rng.standard_normal((n_paths, 2 * N))
    joint = Z @ L.T                               # each row ~ N(0, Sigma)
    W_tilde = joint[:, :N]                         # (n_paths, N)
    W = joint[:, N:]                               # (n_paths, N), BM LEVELS at t_1..t_N

    # 2. Variance path (the martingale-corrected lognormal).
    v = xi0(t_grid) * np.exp(eta*W_tilde - 0.5*eta**2 * t_grid**(2*H))

    # 3. Increments for the price. dt over intervals [t_{i-1}, t_i] with t_0 = 0.
    dt = np.diff(t_grid, prepend=0.0) # shape (N,)
    dW = np.diff(W, prepend=0.0, axis=1)  # [W_0 = 0]

    # BUG TRAP 1 (leverage): dW_S must reuse the SAME dW above, that is the
    #   shared-W channel. dW_perp is a FRESH independent draw with per-step
    #   variance dt (not variance 1): dW_perp = sqrt(dt) * standard_normal.
    Z_new = rng.standard_normal((n_paths, N))
    dW_perp = np.sqrt(dt) * Z_new                            
    dW_S = rho*dW + np.sqrt(1-rho**2)*dW_perp

    # BUG TRAP 2 (left-point v): the Euler step over [t_{i-1}, t_i] uses v at the
    #   LEFT endpoint t_{i-1}, so prepend v_0 = xi0(0) and drop the last column.
    #   Using v at the same index (right point) is the anticipating-integrand bug.
    v0 = xi0(np.array([0.0]))                    # xi0 must accept 0.0; shape (1,)
    v_left = np.concatenate([np.broadcast_to(v0, (n_paths, 1)), v[:, :-1]], axis=1)

    # 4. Log-Euler for log S, X_0 = 0.
    #    dX = sqrt(v_left)*dW_S - 0.5*v_left*dt
    log_incr = np.sqrt(v_left)*dW_S - 0.5*v_left*dt                                 # shape (n_paths, N)
    X = np.cumsum(log_incr, axis=1)
    S = np.exp(X)

    return S, v, W_tilde