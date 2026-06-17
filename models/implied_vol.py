from scipy.optimize import brentq
from models.bsm import bsm_price
from typing import Literal
import numpy as np
import pandas as pd

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

def compute_smile(
    df: pd.DataFrame,
    spot: float,
    T: float,
    r: float = 0.045,
    option_type: Literal['call', 'put'] = 'call',
) -> pd.DataFrame:
    """
    Compute implied volatility for each row in a cleaned chain.
    Adds 'iv' column. Drops rows where IV inversion fails (returns NaN).
    """
    df = df.copy()
    
    df['mid'] = (df['ask'] + df['bid']) / 2
    
    df['iv'] = df.apply(
    lambda row: bsm_implied_vol(
        price=row['mid'],
        S=spot,
        K=row['strike'],
        T=T,
        r=r,
        option_type=option_type,
    ),
    axis=1,
)
    
    n_before = len(df)
    df = df.dropna(subset=['iv'])
    n_after = len(df)
    if n_after < n_before:
        print(f"Dropped {n_before - n_after} rows with invalid IV")
    return df
    