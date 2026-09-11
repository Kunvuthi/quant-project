import numpy as np
from typing import Literal

def simulate_gbm_paths(
    S: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate GBM price paths via exact log-space stepping.
    Returns array of shape (n_paths, n_steps) [or n_steps+1 if you include S_0].
    Monitoring dates assumed equally spaced: t_i = i*T/n_steps.
    """
    rng = np.random.default_rng(seed)

    dt = T/n_steps                      # T / n_steps  -- one step size

    # --- draw all increments at once: shape (n_paths, n_steps) ---
    Z = rng.standard_normal((n_paths, n_steps))                   

    # --- the cumulative-noise term: sigma * sqrt(dt) * cumsum of Z along TIME axis ---
    # each column i should hold sigma*sqrt(dt)*(Z_1 + ... + Z_i) for every path
    cum_noise = sigma * np.sqrt(dt) * np.cumsum(Z, axis=1) 

    # --- the deterministic drift term, one value per monitoring time ---
    # t_i = (i)*dt for i = 1..n_steps  -> a length-n_steps vector
    # drift_i = (r - 0.5 sigma^2) * t_i
    t_grid = np.arange(1, n_steps+1) * dt        
    drift = (r - 0.5*sigma**2) * t_grid                  

    # --- assemble log-prices, then exponentiate ---
    # log S_{t_i} = log S0 + drift_i + cum_noise[:, i]
    # drift is (n_steps,), cum_noise is (n_paths, n_steps) -> broadcasts over rows
    log_paths = np.log(S) + drift + cum_noise   
    paths = np.exp(log_paths)       

    return paths
        
