import numpy as np
from typing import Literal
from typing import Union


def cir_qe_regime_params(
    v_t: np.ndarray,
    kappa: float, theta: float, xi: float, dt: float,
    psi_c: float = 1.5,
) -> tuple[np.ndarray, ...]:
    """
    Compute QE regime-matching quantities for one step, per path.
    Returns everything both sample_cir_qe_step and the K0 correction need,
    so the moment/regime algebra lives in exactly one place.
    """
    m = theta + (v_t - theta)*np.exp(-kappa*dt)
    s2 = ((xi**2)/(2*kappa))*((2*v_t*np.exp(-kappa*dt)*(1-np.exp(-kappa*dt)))+(theta*(1-np.exp(-kappa*dt))**2)) 

    psi = s2 / m**2

    mask_1 = psi <= psi_c

    # --- Regime 1: squared Gaussian ---
    inv_psi = psi**-1
    b2 = 2*inv_psi - 1 + np.sqrt(2*inv_psi)*np.sqrt(2*inv_psi - 1)
    a = m/(1+b2)

    # --- Regime 2: point mass + exponential ---
    p = (psi - 1) / (psi + 1)
    beta = (1 - p) / m
    
    return mask_1, a, b2, p, beta


def sample_cir_qe_step(
    v_t: np.ndarray,       # current variance, shape (n_paths,)
    kappa: float,
    theta: float,
    xi: float,
    dt: float,
    psi_c: float = 1.5,
    rng: np.random.Generator = None,
    return_diagnostics: bool = False
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """One QE step: v_t -> v_{t+dt}, vectorised across paths."""
    mask_1, a, b2, p, beta = cir_qe_regime_params(v_t, kappa, theta, xi, dt, psi_c)

    v_next = np.empty_like(v_t)

    # --- Regime 1: squared Gaussian ---
    Z = rng.standard_normal(size=mask_1.sum())   # only for masked paths
    v_next[mask_1] = a[mask_1] * (np.sqrt(b2[mask_1]) + Z) ** 2

    # --- Regime 2: point mass + exponential ---
    U = rng.uniform(0, 1, len(v_t)-mask_1.sum())           # only for the other paths
    v_next[~mask_1] = np.where(U <= p[~mask_1], 0.0, np.log((1-p[~mask_1])/(1-U)) / beta[~mask_1])
    
    if return_diagnostics:
        return v_next, mask_1, a, b2, p, beta

    return v_next


def simulate_heston_paths(
    S0: float, v0: float,
    kappa: float, theta: float, xi: float, rho: float, r: float,
    T: float, n_steps: int, n_paths: int,
    psi_c: float = 1.5,
    gamma1: float = 0.5,   # trapezoidal weight on integrated variance
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (S_paths, v_paths), each shape (n_paths, n_steps+1)."""
    dt = T / n_steps
    gamma2 = 1 - gamma1

    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    # K1, K2, K3, K4 don't depend on v_t or the path, compute once outside the loop
    K1 = gamma1 * dt * ((kappa*rho/xi)-0.5) - (rho/xi)
    K2 = gamma2 * dt * ((kappa*rho/xi)-0.5) + (rho/xi)
    K3 = gamma1 * dt * (1-rho**2)
    K4 = gamma2 * dt * (1-rho**2)

    for i in range(n_steps):
        v_t = v[:, i]

        # 1. advance variance: call your existing sample_cir_qe_step and compute parameters
        v_next, mask_1, a, b2, p, beta = sample_cir_qe_step(v_t, kappa, theta, xi, dt, psi_c, rng, return_diagnostics=True)     

        # 2. s = K2 + 0.5*K4   (scalar, doesn't depend on path or regime)
        s = K2 + 0.5*K4 

        # 3. K0, computed per-path depending on which regime each path is in
        K0_regime1_formula = -(K1 + 0.5*K3)*v_t - (s*a*b2)/(1-2*s*a) + 0.5*np.log(1-2*s*a)
        K0_regime2_formula = -(K1 + 0.5*K3)*v_t - np.log(p+((1-p)*beta)/(beta-s))
        K0 = np.where(mask_1, K0_regime1_formula, K0_regime2_formula)

        # 4. independent Gaussian for the leftover spot noise
        Z_perp = rng.standard_normal(n_paths)

        # 5. log-spot update
        log_S_next = np.log(S[:, i]) + r*dt + K0 + K1*v_t + K2*v_next \
                     + np.sqrt(K3*v_t + K4*v_next) * Z_perp

        v[:, i+1] = v_next
        S[:, i+1] = np.exp(log_S_next)

    return S, v

def heston_char_func(
    u: np.ndarray,
    S0: float, v0: float,
    kappa: float, theta: float, xi: float, rho: float, r: float,
    tau: float,
) -> np.ndarray:
    """
    Heston characteristic function phi(u; tau) = E[e^{iu ln S_T}].
    Uses the branch-safe ("little trap"-avoiding) formulation: the Riccati
    equation has two equally valid roots differing by the sign of d, the
    naive root (Heston 1993 original) produces a discontinuous complex log
    for long maturities / certain parameter regimes (Albrecher et al. 2007,
    "The Little Heston Trap"); using the other root avoids the branch cut
    in practice.
    """
    u = np.asarray(u, dtype=np.complex128)

    d = np.sqrt((rho * xi * 1j * u - kappa)**2 + xi**2 * (1j * u + u**2))

    # branch-safe root: note the SIGN FLIP vs the naive g, this is the fix
    g = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

    exp_dt = np.exp(-d * tau)

    C = (r * 1j * u * tau
         + (kappa * theta / xi**2)
         * ((kappa - rho * xi * 1j * u - d) * tau
            - 2 * np.log((1 - g * exp_dt) / (1 - g)))
        )

    D = ((kappa - rho * xi * 1j * u - d) / xi**2) * ((1 - exp_dt) / (1 - g * exp_dt))

    return np.exp(1j * u * np.log(S0) + C + D * v0)