import numpy as np
from models.heston import heston_char_func  # you'll need this once you get to the COS pricer itself


def heston_cumulants(
    S0: float, v0: float,
    kappa: float, theta: float, xi: float, rho: float, r: float,
    tau: float,
) -> tuple[float, float]:
    """
    First and second cumulants of ln(S_T) under Heston, closed form.
    Used to set the COS truncation range [a, b].
    """
    ekt = np.exp(-kappa * tau)
    ekt2 = ekt ** 2

    c1 = np.log(S0) + (r-0.5*theta)*tau + (theta - v0) * (1-ekt) / (2*kappa)
    
    term1 = xi * tau * kappa * ekt * (v0 - theta) * (8 * kappa * rho - 4 * xi)
    term2 = kappa * rho * xi * (1 - ekt) * (16 * theta - 8 * v0)
    term3 = 2 * theta * kappa * tau * (-4 * kappa * rho * xi + xi**2 + 4 * kappa**2)
    term4 = xi**2 * ((theta - 2 * v0) * ekt2 + theta * (6 * ekt - 7) + 2 * v0)
    term5 = 8 * kappa**2 * (v0 - theta) * (1 - ekt)

    c2 = (term1 + term2 + term3 + term4 + term5) / (8 * kappa**3)
    
    return c1, c2

def cos_truncation_range(
    c1: float, c2: float, L: float = 10.0, c4: float = 0
) -> tuple[float, float]:
    """
    COS domain [a, b] from cumulants, c4 set to 0 (standard simplification,
    Fang & Oosterlee 2008).
    """
    a = c1 - L*np.sqrt(c2 + np.sqrt(c4))
    b = c1 + L*np.sqrt(c2 + np.sqrt(c4))
    return a, b

def cos_call_coefficients(K: float, a: float, b: float, N: int) -> np.ndarray:
    """
    Cosine-series coefficients V_k of a European call payoff on [a, b],
    vectorised over k = 0, ..., N-1 (Fang & Oosterlee 2008).
    """
    k = np.arange(N)
    u_k = k * np.pi / (b - a)

    x1, x2 = np.log(K), b  # payoff is zero below ln(K), so integrate [ln K, b] only

    def chi(x1, x2):
        denom = 1 + u_k**2
        return (1 / denom) * (
            np.cos(u_k * (x2 - a)) * np.exp(x2) - np.cos(u_k * (x1 - a)) * np.exp(x1)
            + u_k * np.sin(u_k * (x2 - a)) * np.exp(x2) - u_k * np.sin(u_k * (x1 - a)) * np.exp(x1)
        )

    def psi(x1, x2):
        out = np.empty(N)
        out[1:] = (np.sin(u_k[1:] * (x2 - a)) - np.sin(u_k[1:] * (x1 - a))) * (b - a) / (k[1:] * np.pi)
        out[0] = x2 - x1  # k=0 special case
        return out

    V = (chi(x1, x2) - K * psi(x1, x2))
    return V

def cos_call_price(
    S0: float, v0: float,
    kappa: float, theta: float, xi: float, rho: float, r: float,
    tau: float, K: float,
    N: int = 128, L: float = 10,
) -> float:
    """
    European call price under Heston via the COS method (Fang & Oosterlee 2008).
    """
    c1, c2 = heston_cumulants(S0, v0, kappa, theta, xi, rho, r, tau)
    a, b = cos_truncation_range(c1, c2, L)

    k = np.arange(N)
    u_k = k * np.pi / (b - a)

    phi_vals = heston_char_func(u_k, S0, v0, kappa, theta, xi, rho, r, tau)
    A_k = (2 / (b - a)) * np.real(phi_vals * np.exp(-1j * u_k * a))

    V_k = cos_call_coefficients(K, a, b, N)

    weights = np.ones(N)
    weights[0] = 0.5  # k=0 term carries half weight in the cosine series

    price = np.exp(-r * tau) * np.sum(weights * A_k * V_k)
    return price
    