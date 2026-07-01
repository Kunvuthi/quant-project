import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from collections.abc import Callable
from typing import Literal

def setup_grid(
    S0: float,
    T: float,
    sigma_max: float,
    n_S: int = 200,
    n_t: int = 200,
    n_std: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Log-space grid for the local-vol PDE.
    x = ln(S); uniform in x => geometric in S.
    Domain width set by n_std standard deviations of log-return at sigma_max.
    Returns x_grid, S_grid, t_grid, dx, dt.
    """
    # center the log-grid on ln(S0); width = n_std * sigma_max * sqrt(T) each side
    x_min = np.log(S0) - n_std*sigma_max*np.sqrt(T)
    x_max = np.log(S0) + n_std*sigma_max*np.sqrt(T)
    
    x_grid, dx = np.linspace(x_min, x_max, n_S, retstep=True)
    S_grid = np.exp(x_grid)
    
    t_grid, dt = np.linspace(0, T, n_t + 1, retstep=True)
    
    return x_grid, S_grid, t_grid, dx, dt

def build_operator_diagonals(
    sigma_nodes: np.ndarray,
    dx: float,
    r: float,
    q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tridiagonal diagonals of the log-space spatial operator L at one time level.

    Parameters
    ----------
    sigma_nodes : ndarray, shape (n_S,)
        Local vol sigma_loc(S_i, t) at each spatial node for THIS time level.
    dx, r, q : float

    Returns
    -------
    lower, diag, upper : ndarray
        The sub-, main, and super-diagonals of L.
    """
    nu = r - q - 0.5 * sigma_nodes**2                       # (per node)
    diffusion = sigma_nodes**2 / (2 * dx**2)               
    drift = nu / (2 * dx)             

    lower = diffusion - drift      # coefficient on u_{i-1}
    diag  = -2*diffusion - r       # coefficient on u_{i}         
    upper = diffusion + drift      # coefficient on u_{i+1}
    return lower, diag, upper

def cn_step(
    u_next: np.ndarray,
    sigma_nodes: np.ndarray,
    dx: float,
    dt: float,
    r: float,
    q: float,
) -> np.ndarray:
    """
    One backward Crank-Nicolson step: u^{n+1} -> u^n (interior nodes).
    Boundary conditions are applied by the caller, not here.

    Parameters
    ----------
    u_next : ndarray, shape (n_S,)
        Solution at the later time level (known).
    sigma_nodes : ndarray, shape (n_S,)
        Local vol at each node for this step.
    dx, dt, r, q : float

    Returns
    -------
    u_now : ndarray, shape (n_S,)
        Solution at the earlier time level (interior solved; edges set by caller).
    """
    lower, diag, upper = build_operator_diagonals(sigma_nodes, dx, r, q)

    half = 0.5 * dt
    # A = I - (dt/2) L   (implicit, solve with this)
    A_lower = -half * lower[1:]      # sub-diagonal: length n_S - 1
    A_diag  = 1.0 - half * diag      # main diagonal: length n_S
    A_upper = -half * upper[:-1]     # super-diagonal: length n_S - 1

    # B = I + (dt/2) L   (explicit, multiply with this)
    B_lower = +half * lower[1:]
    B_diag  = 1.0 + half * diag
    B_upper = +half * upper[:-1]

    A = diags([A_lower, A_diag, A_upper], offsets=[-1, 0, 1], format="csr")
    B = diags([B_lower, B_diag, B_upper], offsets=[-1, 0, 1], format="csr")

    rhs = B @ u_next                 # tridiagonal matvec
    u_now = spsolve(A, rhs)          # tridiagonal solve
    return u_now

def price_localvol(
    K: float,
    S0: float,
    T: float,
    r: float,
    q: float,
    local_vol_fn: Callable[[np.ndarray, float], np.ndarray],
    sigma_max: float,
    n_S: int = 200,
    n_t: int = 200,
    n_std: int = 5,
    option_type: Literal['call', 'put'] = 'call',
) -> float:
    """
    Price a European option under a local-vol surface via Crank-Nicolson.

    local_vol_fn : callable
        local_vol_fn(S_array, t) -> sigma at each spot for time t.
        (Constant vol: lambda S, t: 0.2 * np.ones_like(S).)
    Returns the price interpolated at S0.
    """
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    
    x_grid, S_grid, t_grid, dx, dt = setup_grid(S0, T, sigma_max, n_S, n_t, n_std)

    # --- terminal condition: payoff at t = T ---
    if option_type == 'call':
        u = np.maximum(S_grid - K, 0.0)           # V(S, T) = (S-K)+
    else:
        u = np.maximum(K - S_grid, 0.0)

    # --- march backward: from t_grid[-1] down to t_grid[0] ---
    for n in range(n_t, 0, -1):               # step from level n to n-1
        t_now = t_grid[n - 1]                  # the time we're solving FOR
        sigma_nodes = local_vol_fn(S_grid, t_now)
        
        u = cn_step(u, sigma_nodes, dx, dt, r, q)

        tau = T - t_now
        # --- boundaries: payoff and boundaries are a MATCHED PAIR ---
        if option_type == 'call':
            u[0]  = 0.0
            u[-1] = S_grid[-1]*np.exp(-q*tau) - K*np.exp(-r*tau)
        else:
            u[0]  = K*np.exp(-r*tau) - S_grid[0]*np.exp(-q*tau)
            u[-1] = 0.0

    # --- interpolate the price at S0 ---
    return np.interp(S0, S_grid, u)
