import numpy as np
import scipy.stats as stats
from typing import Literal
from numpy.typing import ArrayLike

def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    N: int,
    option_type: Literal['call', 'put'] = 'call'
) -> float:
    """
    European option price via the Cox-Ross-Rubinstein binomial tree.
    
    Parameters
    ----------
    N : int
        Number of time steps. Larger N → more accurate but slower.
        Price converges to BSM as N → ∞ at rate O(1/N).
    
    Notes
    -----
    Uses the standard CRR parameterisation: u = exp(sigma*sqrt(dt)), d = 1/u.
    Tree is recombining: O(N) memory, O(N^2) time.
    """
    
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    
    if not isinstance(N, int) or N < 1:
        raise ValueError("N must be a positive integer")
    
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    
    q = (np.exp(r * dt) - d) / (u - d)
    
    # Compute terminal payoffs at maturity
    j = np.arange(N + 1)           # [0, 1, 2, 3]
    ST = S * (u ** j) * (d ** (N - j))   # ST[0] is all-down, ST[N] is all-up
    
    # Payoff at time T
    if option_type == "call":
        C = np.maximum(ST - K, 0)
    else:
        C = np.maximum(K - ST, 0)
        
    # Backward Induction
    discount = np.exp(-r * dt)
    for i in range(N-1, -1, -1):
        C = discount * (q * C[1:] + (1 - q) * C[:-1])
    
    return C[0]

def crr_richardson(S, K, T, r, sigma, N, option_type='call'):
    """Richardson-extrapolated CRR: 2*P(2N) - P(N)."""
    p_N = crr_price(S, K, T, r, sigma, N, option_type)
    p_2N = crr_price(S, K, T, r, sigma, 2*N, option_type)
    return 2 * p_2N - p_N