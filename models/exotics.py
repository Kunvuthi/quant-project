import numpy as np
from scipy.stats import norm
from models.montecarlo import simulate_gbm_paths   # the primitive stays put
from typing import Literal

def geometric_asian_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    option_type: Literal['call', 'put'] = 'call',
) -> float:
    """
    Closed-form price of a geometric-average Asian call.
    Equally-spaced monitoring at t_i = i*T/n_steps, i = 1..n_steps.
    Used as the control-variate mean mu_X for the arithmetic Asian.
    """
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    
    n = n_steps

    # --- effective volatility ---
    sigma_hat = sigma * np.sqrt((n+1)*(2*n+1) / (6*n**2) )

    # --- average monitoring time ---
    t_bar = T*(n+1) / (2*n)

    # --- effective carry b_hat so the effective log-mean matches ---
    # b_hat = [ (r - 0.5 sigma^2) t_bar + 0.5 sigma_hat^2 T ] / T
    b_hat = ( (r - 0.5*sigma**2)*t_bar + ((0.5*sigma_hat**2)*T) ) / T

    # --- BS call with vol sigma_hat, forward grown at b_hat, discounted at r ---
    d1 = (np.log(S/K) + (b_hat + 0.5*sigma_hat**2)*T ) / (sigma_hat*np.sqrt(T))
    d2 = d1 - sigma_hat*np.sqrt(T)  

    if option_type == 'call':
        price = np.exp(-r*T) * (S*np.exp(b_hat*T)*norm.cdf(d1) - K*norm.cdf(d2))
    else:
        price = np.exp(-r*T) * (K*norm.cdf(-d2) - S*np.exp(b_hat*T)*norm.cdf(-d1))
    
    return price

def arithmetic_asian_price_cv(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_steps: int,
    n_paths: int = 100_000,
    seed: int | None = None,
    option_type: Literal['call','put'] = 'call',
) -> dict:
    
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    
    paths = simulate_gbm_paths(S, T, r, sigma, n_steps, n_paths=n_paths, seed=seed)
    disc = np.exp(-r * T)

    # target Y: arithmetic-average Asian call (no closed form)
    arith_mean = paths.mean(axis=1)                 # plain mean over time
    geo_mean = np.exp(np.mean(np.log(paths), axis=1))
    if option_type == 'call':
        Y = disc * np.maximum(arith_mean - K, 0.0)
        X = disc * np.maximum(geo_mean - K, 0.0)
    else:
        Y = disc * np.maximum(K - arith_mean, 0.0)
        X = disc * np.maximum(K - geo_mean, 0.0)
      
    mu_X = geometric_asian_price(S, K, T, r, sigma, n_steps, option_type)

    # optimal c from the SAME sample: c* = Cov(X,Y)/Var(X)
    cov = np.cov(X, Y, ddof=1)        # 2x2 matrix
    c_star = cov[0, 1] / cov[0, 0]

    # control-variate combination
    Y_cv = Y - c_star * (X - mu_X)

    return {
        "plain":   (Y.mean(),    Y.std(ddof=1)    / np.sqrt(n_paths)),
        "cv":      (Y_cv.mean(), Y_cv.std(ddof=1) / np.sqrt(n_paths)),
        "c_star":  c_star,
        "corr":    cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]),
    }
    
S, K, T, r, sigma, n = 100, 100, 1.0, 0.05, 0.20, 50
call = geometric_asian_price(S, K, T, r, sigma, n, 'call')
put  = geometric_asian_price(S, K, T, r, sigma, n, 'put')
# reconstruct b_hat for the parity RHS
n_ = n
sigma_hat = sigma*np.sqrt((n_+1)*(2*n_+1)/(6*n_**2))
t_bar = T*(n_+1)/(2*n_)
b_hat = ((r - 0.5*sigma**2)*t_bar + 0.5*sigma_hat**2*T)/T
parity_rhs = np.exp(-r*T)*(S*np.exp(b_hat*T) - K)
print(f"C-P {call-put:.4f} vs parity {parity_rhs:.4f}  diff {abs((call-put)-parity_rhs):.2e}")