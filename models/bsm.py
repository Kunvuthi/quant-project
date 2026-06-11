import numpy as np
from scipy.stats import norm
from typing import Literal
from numpy.typing import ArrayLike

def bsm_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    option_type: Literal['call', 'put'] = 'call'
) -> np.ndarray:
    """
    Black-Scholes-Merton European option price.
    
    Parameters
    ----------
    S : float or array
        Spot price(s).
    K : float or array
        Strike price(s).
    T : float or array
        Time to expiry in years.
    r : float
        Risk-free rate (continuously compounded).
    sigma : float or array
        Volatility (annualised, e.g. 0.2 for 20%).
    option_type : {'call', 'put'}, default 'call'
        
    Returns
    -------
    price : float or array
        Same shape as the broadcasted inputs.
        
    Notes
    -----
    Edge cases (T=0, sigma=0, S=0, K=0) are not handled in this version 
    and will produce NaN or inf. Will be addressed in Week 2.

    """
    
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Standard BSM formula
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Test cases to verify vectorisation and broadcasting
# # Single option
# print(bsm_price(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type='call'))

# # Vector of strikes
# strikes = np.array([80, 90, 100, 110, 120])
# print(bsm_price(S=100, K=strikes, T=1.0, r=0.05, sigma=0.20, option_type='call'))

# # 2D grid (this is the real test of vectorisation)
# strikes = np.array([90, 100, 110])
# maturities = np.array([[0.25], [0.5], [1.0], [2.0]])  # column vector
# print(bsm_price(S=100, K=strikes, T=maturities, r=0.05, sigma=0.20, option_type='call'))
# # Should return a 4x3 array
    
   