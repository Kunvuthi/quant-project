import numpy as np
from typing import Literal
from collections.abc import Callable
from numpy.typing import ArrayLike

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
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    
    if not isinstance(n_paths, int) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")
    
    rng = np.random.default_rng(seed)
    
    n_pairs = n_paths//2

    # --- antithetic draws ---
    Z = rng.standard_normal(n_pairs)                      # shape (n_pairs,)
    Z_all = np.concatenate([Z, -Z])   # +Z block then -Z block

    # --- terminal prices for every path ---
    S_T = S*np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z_all)

    disc = np.exp(-r * T)
    
    if option_type == "call":
        itm = S_T > K                # the 1_{S_T > K} indicator, reused by both Greeks
    else:
        itm = S_T < K

    # --- pathwise integrands ---
    if option_type == "call":
        delta_paths = disc * itm * S_T / S 
    else:
        delta_paths = -disc * itm * S_T / S

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

def bsm_greeks_lr(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    payoff: Callable[[ArrayLike], ArrayLike], 
    n_paths: int = 100_000,
    seed: int | None = None,
) -> dict:
    """
    Likelihood Ratio Delta and Vega via Monte Carlo, antithetic.
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
    payoff_value = payoff(S_T)

    # --- pathwise integrand ---
    delta_paths = disc * payoff_value  * Z_all / (S*sigma*np.sqrt(T))
    vega_paths  = disc * payoff_value * (((Z_all**2 - 1) / sigma) - Z_all * np.sqrt(T))

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