from scipy.optimize import brentq
from models.bsm import bsm_price
from typing import Literal
import numpy as np

def bsm_implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal['call', 'put'] = 'call',
    tol: float = 1e-8,
) -> float:
    """
    Implied volatility via Brent's method.
    
    Returns
    -------
    sigma : float
        The volatility such that bsm_price(S, K, T, r, sigma, option_type) ≈ price.
        Returns np.nan if no solution exists (e.g. price violates no-arbitrage bounds).
    """
    
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    
    # Arbitrage Bounds
    if option_type == 'call':
        LB = np.maximum(S - K*np.exp(-r*T), 0)
        UB = S
    else:
        LB = np.maximum(K*np.exp(-r*T) - S, 0)
        UB = K*np.exp(-r*T)
    
    if price < LB or price > UB:
        return np.nan
    
    def f(sigma):
        return bsm_price(S, K, T, r, sigma, option_type) - price
    
    sigma_imp = brentq(f, 1e-6, 5.0, xtol=tol)    
    
    return sigma_imp
    