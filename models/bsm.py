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
    option_type: Literal['call', 'put'] = 'call',
    b: float | None = None,
) -> np.ndarray:
    """
    Black-Scholes-Merton European option price, generalised with cost-of-carry b.

    b is the cost of carry on the underlying; r is the discount rate.
      b = r        -> plain BSM, no dividend (default)
      b = r - q    -> continuous dividend yield q
      b = 0        -> Black-76 (option on a future)
    Defaults to b = r, recovering the standard BSM formula exactly.
    """
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")

    if b is None:
        b = r

    # broadcast everything to a common shape so masks line up
    S, K, T, sigma = np.broadcast_arrays(
        np.asarray(S, dtype=float), np.asarray(K, dtype=float),
        np.asarray(T, dtype=float), np.asarray(sigma, dtype=float),
    )

    carry = np.exp((b - r) * T)
    fwd = S * carry                       # = S e^{(b-r)T}, the spot term
    true_fwd = S * np.exp(b * T) 

    # --- degenerate mask: where the standard formula divides by zero ---
    vol_time = sigma * np.sqrt(T)         # the denominator in d1
    degenerate = vol_time <= 0            # True where T==0 OR sigma==0

    # --- safe formula branch: substitute a dummy 1.0 in the denominator
    #     wherever it's degenerate, so no div-by-zero warning is emitted.
    #     The masked-out results are discarded by np.where below.
    safe_vt = np.where(degenerate, 1.0, vol_time)
    d1 = (np.log(S / K) + (b + 0.5 * sigma ** 2) * T) / safe_vt
    d2 = d1 - safe_vt

    if option_type == 'call':
        formula = fwd * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        intrinsic_T0 = np.maximum(S - K, 0.0)
        intrinsic_vol0 = np.exp(-r * T) * np.maximum(true_fwd - K, 0.0)
    else:
        formula = K * np.exp(-r * T) * norm.cdf(-d2) - fwd * norm.cdf(-d1)
        intrinsic_T0 = np.maximum(K - S, 0.0)
        intrinsic_vol0 = np.exp(-r * T) * np.maximum(K - true_fwd, 0.0)

    # --- select limits: T==0 takes priority, then sigma==0, else formula ---
    price = np.where(
        T <= 0, intrinsic_T0,
        np.where(sigma <= 0, intrinsic_vol0, formula),
    )

    return price

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

def bsm_vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    b: float | None = None,
) -> np.ndarray:
    """Analytic BSM vega with cost-of-carry b, matching bsm_price's convention.

    Vega is per unit of sigma. Uses the same d1 and the same e^{(b-r)T} spot
    factor as bsm_price, so it is forward-consistent for a parity-implied forward.
    """
    if b is None:
        b = r
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp((b - r) * T) * norm.pdf(d1) * np.sqrt(T)