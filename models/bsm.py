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

def bsm_greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    option_type: Literal['call', 'put'] = 'call'
) -> dict:
    """
    Closed-form Black-Scholes-Merton Greeks for a European option.
    
    Returns
    -------
    dict with keys 'delta', 'gamma', 'vega', 'theta', 'rho'.
    Each value has the same shape as the broadcasted inputs.
    
    Convention
    ----------
    Vega is per unit of sigma (e.g. divide by 100 for "per vol point").
    Theta is per unit of T (per year). Divide by 365 for daily theta.
    Rho is per unit of r (per unit, not per basis point).
    """
    
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    
    if option_type == 'call':
        delta = cdf_d1
        theta = -S * pdf_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * cdf_d2
        rho = K * T * np.exp(-r * T) * cdf_d2
    else:
        delta = cdf_d1 - 1
        theta = -S * pdf_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * (1 - cdf_d2)
        rho = -K * T * np.exp(-r * T) * (1 - cdf_d2)

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    
    vega = S * pdf_d1 * np.sqrt(T) 
    
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

def bsm_greeks_fd(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'] = 'call',
    h: float = 1e-4
) -> dict:
    """
    Finite-difference Greeks via central differences.
    
    Used to verify the analytic Greeks. Accuracy is O(h^2);
    h=1e-4 is usually a good compromise between truncation and roundoff error.
    
    We are using the Central Difference formula for better accuracy:
    f'(x) + O(h^2) <- (f(x + h) - f(x - h)) / (2 * h) from Taylor expansion around x. 
    We use the 2nd central difference for gamma: f''(x) + O(h^2) <- (f(x + h) - 2 * f(x) + f(x - h)) / (h^2).
    This is more accurate than the Forward or Backward Difference formulas, which are O(h) accurate.
    """
    
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    
    price = bsm_price(S, K, T, r, sigma, option_type)
    
    # Delta: dPrice/dS
    price_up = bsm_price(S + h, K, T, r, sigma, option_type)
    price_down = bsm_price(S - h, K, T, r, sigma, option_type)
    delta = (price_up - price_down) / (2 * h)
    
    # Gamma: d^2Price/dS^2
    gamma = (price_up - 2 * price + price_down) / (h ** 2)
    
    # Vega: dPrice/dsigma
    price_up = bsm_price(S, K, T, r, sigma + h, option_type)
    price_down = bsm_price(S, K, T, r, sigma - h, option_type)
    vega = (price_up - price_down) / (2 * h)
    
    # Theta: dPrice/dT
    price_up = bsm_price(S, K, T + h, r, sigma, option_type)
    price_down = bsm_price(S, K, T - h, r, sigma, option_type)
    theta = -(price_up - price_down) / (2 * h)
    
    # Rho: dPrice/dr
    price_up = bsm_price(S, K, T, r + h, sigma, option_type)
    price_down = bsm_price(S, K, T, r - h, sigma, option_type)
    rho = (price_up - price_down) / (2 * h)
    
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}