import numpy as np
from models.heston import heston_char_func
from pricing.fourier import heston_cumulants


def bates_char_func(
    u: np.ndarray,
    S0: float,
    v0: float,
    kappa_v: float,      # Heston mean-reversion speed (NOT the jump compensator kappa_j)
    theta: float,
    xi: float,
    rho: float,
    r: float,
    q: float,
    tau: float,
    lam: float,          # jump intensity
    mu_j: float,         # mean log-jump size
    delta_j: float,      # log-jump std
) -> np.ndarray:
    """Bates (Heston + Merton jumps) characteristic function of log(S_T).

    phi_Bates = phi_Heston(r-q) * exp(iu * -lam*kappa_j*tau) * jump_factor

    Reuses heston_char_func for the diffusion + stochastic-vol part, passing
    (r - q) into its r-slot (Heston's drift is a pure r*iu*tau term, so this
    substitution gives the correct (r-q) forward and touches nothing else).
    The -lam*kappa_j compensator and the pure jump factor are multiplied in
    explicitly, since Heston carries no jump knowledge. This places (r-q) once
    (Heston) and -lam*kappa_j once (explicit compensator), so the martingale
    condition phi(-i) = S0 exp((r-q)tau) holds.

    kappa_j = exp(mu_j + 0.5 delta_j^2) - 1 is the Merton jump compensator,
    distinct from the Heston mean-reversion kappa_v.
    """
    kappa_j = np.exp(mu_j + 0.5 * delta_j**2) - 1.0

    phi_heston = heston_char_func(u, S0, v0, kappa_v, theta, xi, rho, r - q, tau)
    compensator = np.exp(1j * u * (-lam * kappa_j * tau))
    jump = np.exp(lam * tau * (np.exp(1j * u * mu_j - 0.5 * u**2 * delta_j**2) - 1.0))

    return phi_heston * compensator * jump


def bates_cumulants(
    S0: float,
    v0: float,
    kappa_v: float,
    theta: float,
    xi: float,
    rho: float,
    r: float,
    q: float,
    tau: float,
    lam: float,
    mu_j: float,
    delta_j: float,
) -> tuple[float, float]:
    """First and second cumulants of log(S_T) under Bates.

    Diffusion and jump cumulants add (independence). The Heston part is reused
    with (r - q); the jump part is Merton's:

        c1 = c1_Heston(r-q) + lam*tau*mu_j - lam*kappa_j*tau
        c2 = c2_Heston      + lam*tau*(mu_j^2 + delta_j^2)

    kappa_j = exp(mu_j + 0.5 delta_j^2) - 1. The -lam*kappa_j*tau in c1 is the
    compensator shifting the mean; lam*tau*(mu_j^2 + delta_j^2) is the jump
    variance via the compound-Poisson second moment E[Y^2] = mu_j^2 + delta_j^2.
    """
    kappa_j = np.exp(mu_j + 0.5 * delta_j**2) - 1.0

    c1_h, c2_h = heston_cumulants(S0, v0, kappa_v, theta, xi, rho, r - q, tau)

    c1 = c1_h + lam * tau * mu_j - lam * kappa_j * tau
    c2 = c2_h + lam * tau * (mu_j**2 + delta_j**2)

    return c1, c2