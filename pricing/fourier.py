import numpy as np
from models.heston import heston_char_func, heston_cumulants  # you'll need this once you get to the COS pricer itself
from models.merton import merton_char_func, merton_cumulants
from models.kou import kou_char_func, kou_cumulants
from models.bates import bates_char_func, bates_cumulants
from models.vg import vg_char_func, vg_cumulants
from typing import Literal

def _cos_chi(a: float, b: float, u_k: np.ndarray, x1: float, x2: float) -> np.ndarray:
    """Shared cosine/exponential integral building block, payoff-agnostic."""
    denom = 1 + u_k**2
    return (1 / denom) * (
        np.cos(u_k * (x2 - a)) * np.exp(x2) - np.cos(u_k * (x1 - a)) * np.exp(x1)
        + u_k * np.sin(u_k * (x2 - a)) * np.exp(x2) - u_k * np.sin(u_k * (x1 - a)) * np.exp(x1)
    )

def _cos_psi(a: float, b: float, u_k: np.ndarray, x1: float, x2: float, N: int) -> np.ndarray:
    """Shared cosine integral building block, payoff-agnostic."""
    k = np.arange(N)
    out = np.zeros(N)   # was np.empty; zeros so a missed element is a benign 0, not garbage
    out[1:] = (np.sin(u_k[1:] * (x2 - a)) - np.sin(u_k[1:] * (x1 - a))) * (b - a) / (k[1:] * np.pi)
    out[0] = x2 - x1
    return out

def cos_call_coefficients(K: float, a: float, b: float, N: int) -> np.ndarray:
    """
    Cosine-series coefficients V_k of a European call payoff on [a, b],
    vectorised over k = 0, ..., N-1 (Fang & Oosterlee 2008).
    """
    k = np.arange(N)
    u_k = k * np.pi / (b - a)
    x1, x2 = np.log(K), b
    return (_cos_chi(a, b, u_k, x1, x2) - K * _cos_psi(a, b, u_k, x1, x2, N))

def cos_put_coefficients(K: float, a: float, b: float, N: int) -> np.ndarray:
    """
    V_k for a European put. Payoff g(x) = (K - e^x)^+ is nonzero on [a, ln K]
    instead of [ln K, b], the mirror image of the call's active region.
    """
    k = np.arange(N)
    u_k = k * np.pi / (b - a)
    x1, x2 = a, np.log(K)
    return (-1 *  _cos_chi(a, b, u_k, x1, x2) + K * _cos_psi(a, b, u_k, x1, x2, N) )


# ---------------------------------------------------------------------------
# MODEL-AGNOSTIC CORE. Knows nothing about Heston, Merton, or any model.
# Takes pre-evaluated char func values and cumulants as pure numbers.
# ---------------------------------------------------------------------------

def cos_density_coefficients_from_cf(
    phi_vals: np.ndarray, a: float, b: float, N: int,
) -> np.ndarray:
    """A_k from pre-evaluated char func values phi(u_k).

    phi_vals must already be phi evaluated at u_k = k*pi/(b-a) for k=0..N-1.
    Strike- and payoff-independent, computed once and reused across strikes.
    """
    k = np.arange(N)
    u_k = k * np.pi / (b - a)
    return (2 / (b - a)) * np.real(phi_vals * np.exp(-1j * u_k * a))


def cos_price_from_cf(
    phi_vals: np.ndarray,
    a: float, b: float,
    r: float, tau: float, K: float,
    N: int = 128,
    option_type: Literal['call', 'put'] = 'call',
) -> float:
    """European option price by COS from pre-evaluated char func values.

    Fully model-agnostic: phi_vals carries all model information. The caller
    (a per-model wrapper) computes phi at u_k and the truncation range, then
    calls this. Same core for Heston, Merton, Kou, anything with a char func.
    """
    A_k = cos_density_coefficients_from_cf(phi_vals, a, b, N)
    coeff_fn = cos_call_coefficients if option_type == 'call' else cos_put_coefficients
    V_k = coeff_fn(K, a, b, N)
    weights = np.ones(N); weights[0] = 0.5
    return np.exp(-r * tau) * np.sum(weights * A_k * V_k)


def cos_smile_from_cf(
    phi_vals: np.ndarray,
    a: float, b: float,
    r: float, tau: float, strikes: np.ndarray,
    N: int = 128,
    option_type: Literal['call', 'put'] = 'call',
) -> np.ndarray:
    """Strip of strikes from pre-evaluated char func values.

    A_k computed once (strike-independent), loop only the cheap per-strike V_k.
    """
    A_k = cos_density_coefficients_from_cf(phi_vals, a, b, N)
    weights = np.ones(N); weights[0] = 0.5
    coeff_fn = cos_call_coefficients if option_type == 'call' else cos_put_coefficients
    prices = np.empty(len(strikes), dtype=float)
    for i, K in enumerate(strikes):
        V_k = coeff_fn(K, a, b, N)
        prices[i] = np.exp(-r * tau) * np.sum(weights * A_k * V_k)
    return prices


def cos_truncation_range(c1: float, c2: float, L: float = 10.0, c4: float = 0) -> tuple[float, float]:
    """COS domain [a, b] from cumulants (Fang & Oosterlee 2008). Model-agnostic."""
    a = c1 - L * np.sqrt(c2 + np.sqrt(c4))
    b = c1 + L * np.sqrt(c2 + np.sqrt(c4))
    return a, b

# ---------------------------------------------------------------------------
# HESTON WRAPPERS. Thin: compute Heston's char func + cumulants, call the core.
# Signatures unchanged so existing callers (tests, notebooks) keep working.
# ---------------------------------------------------------------------------

def _heston_phi_vals(S0, v0, kappa, theta, xi, rho, r, tau, a, b, N):
    """Evaluate the Heston char func at the COS frequencies u_k."""
    k = np.arange(N)
    u_k = k * np.pi / (b - a)
    return heston_char_func(u_k, S0, v0, kappa, theta, xi, rho, r, tau)


def cos_call_price(S0, v0, kappa, theta, xi, rho, r, tau, K, N=128, L=10) -> float:
    """European call under Heston via COS. Thin wrapper over the model-agnostic core."""
    c1, c2 = heston_cumulants(S0, v0, kappa, theta, xi, rho, r, tau)
    a, b = cos_truncation_range(c1, c2, L)
    phi_vals = _heston_phi_vals(S0, v0, kappa, theta, xi, rho, r, tau, a, b, N)
    return cos_price_from_cf(phi_vals, a, b, r, tau, K, N, 'call')


def cos_put_price(S0, v0, kappa, theta, xi, rho, r, tau, K, N=128, L=10) -> float:
    """Native Heston put via COS, independent of the call pipeline."""
    c1, c2 = heston_cumulants(S0, v0, kappa, theta, xi, rho, r, tau)
    a, b = cos_truncation_range(c1, c2, L)
    phi_vals = _heston_phi_vals(S0, v0, kappa, theta, xi, rho, r, tau, a, b, N)
    return cos_price_from_cf(phi_vals, a, b, r, tau, K, N, 'put')


def cos_smile(S0, v0, kappa, theta, xi, rho, r, tau, strikes,
              option_type: Literal['call', 'put'] = 'call', N=128, L=10) -> np.ndarray:
    """Heston strike strip via COS."""
    c1, c2 = heston_cumulants(S0, v0, kappa, theta, xi, rho, r, tau)
    a, b = cos_truncation_range(c1, c2, L)
    phi_vals = _heston_phi_vals(S0, v0, kappa, theta, xi, rho, r, tau, a, b, N)
    return cos_smile_from_cf(phi_vals, a, b, r, tau, strikes, N, option_type)

# ---------------------------------------------------------------------------
# Merton WRAPPERS.
# ---------------------------------------------------------------------------

def merton_cos_price(
    S0: float, r: float, q: float, T: float,
    sigma: float, lam: float, mu_j: float, delta_j: float, K: float,
    option_type: Literal['call', 'put'] = 'call', N: int = 128, L: float = 10,
) -> float:
    c1, c2 = merton_cumulants(S0, r, q, T, sigma, lam, mu_j, delta_j)
    a, b = cos_truncation_range(c1, c2, L)
    k = np.arange(N)
    u_k = k * np.pi / (b - a)
    phi_vals = merton_char_func(u_k, S0, r, q, T, sigma, lam, mu_j, delta_j)
    return cos_price_from_cf(phi_vals, a, b, r, T, K, N, option_type)

# ---------------------------------------------------------------------------
# Kou WRAPPERS.
# ---------------------------------------------------------------------------
def kou_cos_price(
    S0: float, r: float, q: float, T: float,
    sigma: float, lam: float, p: float, eta1: float, eta2: float, K: float,
    option_type: Literal['call', 'put'] = 'call', N: int = 128, L: float = 10,
) -> float:
    c1, c2 = kou_cumulants(S0, r, q, T, sigma, lam, p, eta1, eta2)
    a, b = cos_truncation_range(c1, c2, L)
    k = np.arange(N)
    u_k = k * np.pi / (b - a)
    phi_vals = kou_char_func(u_k, S0, r, q, T, sigma, lam, p, eta1, eta2)
    return cos_price_from_cf(phi_vals, a, b, r, T, K, N, option_type)

# ---------------------------------------------------------------------------
# Bates WRAPPERS.
# ---------------------------------------------------------------------------

def bates_cos_price(
    S0: float,
    v0: float,
    kappa_v: float,
    theta: float,
    xi: float,
    rho: float,
    r: float,
    q: float,
    tau: float,
    K: float,
    lam: float,
    mu_j: float,
    delta_j: float,
    option_type: Literal['call', 'put'] = 'call',
    N: int = 256,
    L: float = 12.0,
) -> float:
    """European option price under Bates via COS.

    N=256, L=12 by default: Bates carries both stochastic-vol and jump fat
    tails, so it is more truncation-hungry than either alone. Validate the
    needed values rather than trusting these.
    """
    c1, c2 = bates_cumulants(S0, v0, kappa_v, theta, xi, rho, r, q, tau,
                             lam, mu_j, delta_j)
    a, b = cos_truncation_range(c1, c2, L)
    k = np.arange(N)
    u_k = k * np.pi / (b - a)
    phi_vals = bates_char_func(u_k, S0, v0, kappa_v, theta, xi, rho, r, q, tau,
                               lam, mu_j, delta_j)
    return cos_price_from_cf(phi_vals, a, b, r, tau, K, N, option_type)

# ---------------------------------------------------------------------------
# Variance Gamma Model WRAPPERS.
# ---------------------------------------------------------------------------

def vg_cos_price(
    S0: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    nu: float,
    theta: float,
    K: float,
    option_type: Literal['call', 'put'] = 'call',
    N: int = 256,
    L: float = 12.0,
) -> float:
    """European option price under Variance Gamma via COS."""
    c1, c2 = vg_cumulants(S0, r, q, T, sigma, nu, theta)
    a, b = cos_truncation_range(c1, c2, L)
    k = np.arange(N)
    u_k = k * np.pi / (b - a)
    phi_vals = vg_char_func(u_k, S0, r, q, T, sigma, nu, theta)
    return cos_price_from_cf(phi_vals, a, b, r, T, K, N, option_type)