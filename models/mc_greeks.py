import numpy as np
from typing import Literal

def bsm_greeks_pw(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'] = 'call',
    n_paths: int = 100_000,
    seed: int | None = None,
) -> dict:
    """
    Pathwise Delta and Vega via Monte Carlo, antithetic.
    Returns {'delta': (est, se), 'vega': (est, se)}.
    """
    rng = np.random.default_rng(seed)
    
    n_pairs = n_paths//2

    # --- antithetic draws ---
    Z = rng.standard_normal(n_pairs)                      # shape (n_pairs,)
    Z_all = np.concatenate([Z, -Z])   # +Z block then -Z block

    # --- terminal prices for every path ---
    S_T = S*np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z_all)

    disc = np.exp(-r * T)
    itm = S_T > K                # the 1_{S_T > K} indicator, reused by both Greeks

    # --- pathwise integrands (your formulas) ---
    delta_paths = disc * itm * S_T / S
    vega_paths  = disc * itm * S_T * (np.sqrt(T) * Z_all - sigma * T)

    # --- antithetic pairing + SE (the Week 1 pattern) ---
    # for each integrand: reshape to pair the +Z path with its -Z partner,
    # average within the pair, then mean / std-over-pairs / sqrt(n_pairs)
    pair_means_delta = (delta_paths[:n_pairs] + delta_paths[n_pairs:]) / 2 
    mean_delta = pair_means_delta.mean()
    std_error_delta = pair_means_delta.std(ddof=1) / np.sqrt(n_pairs)
    
    pair_means_vega = (vega_paths[:n_pairs] + vega_paths[n_pairs:]) / 2 
    mean_vega = pair_means_vega.mean()
    std_error_vega = pair_means_vega.std(ddof=1) / np.sqrt(n_pairs)
    
    return {"delta": (mean_delta, std_error_delta), "vega": (mean_vega, std_error_vega)}


    