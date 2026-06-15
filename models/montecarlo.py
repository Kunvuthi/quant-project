import numpy as np
from typing import Literal

def bsm_price_mc(
    S: float, 
    K: float, 
    T: float, 
    r: float, 
    sigma: float,
    N: int = 10_000,
    option_type: Literal['call', 'put'] = 'call',
    antithetic: bool = False,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    European option price via direct Monte Carlo simulation of GBM.
    
    Uses the closed-form solution S_T = S * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),
    so no time-stepping bias. Only valid for path-independent payoffs
    
    Returns
    -------
    (price, standard_error) : tuple of floats
    """
    
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    
    if not isinstance(N, int) or N < 1:
        raise ValueError("N must be a positive integer")
    
    rng = np.random.default_rng(seed)

    if antithetic:
        Z = rng.standard_normal(N//2)
        Z = np.concatenate([Z, -Z])
    else:
        Z = rng.standard_normal(N)  
       
    ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
    
    if option_type == 'call':
        CT = np.maximum(ST - K, 0)
    else:
        CT = np.maximum(0, K-ST)
    dCT = np.exp(-r*T) * CT
    
    if antithetic:
        paired = dCT.reshape(2, N//2)
        pair_means = paired.mean(axis=0)  
        price = pair_means.mean(axis=0)
        std_error = pair_means.std(ddof=1) / np.sqrt(N//2)
    else:
        price = np.mean(dCT)
        std_error = np.std(dCT, ddof=1) / np.sqrt(N)
    
    return (price, std_error)
        
    
    