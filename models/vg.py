import numpy as np


def vg_char_func(
    u: np.ndarray,
    S0: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    nu: float,
    theta: float,
) -> np.ndarray:
    """Variance Gamma characteristic function of log(S_T).

    phi(u) = exp(iu(log S0 + (r - q + omega) T))
             * (1 - iu*theta*nu + 0.5*sigma^2*nu*u^2)^(-T/nu)

    with the martingale correction
        omega = (1/nu) * log(1 - theta*nu - 0.5*sigma^2*nu).

    Requires 1 - theta*nu - 0.5*sigma^2*nu > 0 (omega finite); this bounds
    theta, nu, sigma together, the VG analogue of Kou's eta1 > 1. Reduces to
    Black-Scholes as nu -> 0 (deterministic clock -> pure Brownian motion).
    """
    # constraint check for omega
    arg = 1 - theta*nu - 0.5*sigma**2*nu
    if arg <= 0:
        raise ValueError(
            f"VG constraint violated: 1 - theta*nu - 0.5*sigma^2*nu = {arg:.4f} <= 0 "
            f"(theta={theta}, nu={nu}, sigma={sigma}); omega diverges"
        )
    omega = (1/nu) * np.log(arg)
    drift = np.log(S0) + (r - q + omega) * T
    base = 1 - 1j*u*theta*nu + 0.5*sigma**2*nu*u**2
    return np.exp(1j*u*drift) * base**(-T/nu)

def vg_cumulants(
    S0: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    nu: float,
    theta: float,
) -> tuple[float, float]:
    """First and second cumulants of log(S_T) under Variance Gamma.

        c1 = (r - q + omega) T + theta T
        c2 = (sigma^2 + nu*theta^2) T

    with omega = (1/nu) log(1 - theta*nu - 0.5*sigma^2*nu), identical to the
    char func's. c1 is the drift plus the VG part's mean (theta*T); c2 is the
    diffusion-scale variance sigma^2*T plus nu*theta^2*T from the clock variance
    interacting with the drift.
    """
    arg = 1 - theta*nu - 0.5*sigma**2*nu
    if arg <= 0:
            raise ValueError(
                f"VG constraint violated: 1 - theta*nu - 0.5*sigma^2*nu = {arg:.4f} <= 0 "
                f"(theta={theta}, nu={nu}, sigma={sigma}); omega diverges"
            )
    omega = (1/nu) * np.log(arg)
    c1 = np.log(S0) + (r - q + omega) * T + theta * T
    c2 = (sigma**2 + nu * theta**2) * T
    return c1, c2

def vg_simulate_terminal(
    S0: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    nu: float,
    theta: float,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Terminal prices under VG by subordination sampling, for MC validation.

    Draw a Gamma clock g ~ Gamma(shape=T/nu, scale=nu) (mean T, var nu*T), then
    the log-return conditional on g is Normal(theta*g, sigma^2*g). Same omega
    correction as the char func so the simulated process matches it.
    """
    rng = np.random.default_rng(seed)
    omega = (1.0 / nu) * np.log(1.0 - theta * nu - 0.5 * sigma**2 * nu)

    # Gamma clock: mean T, variance nu*T  =>  shape T/nu, scale nu
    g = rng.gamma(shape=T / nu, scale=nu, size=n_paths)

    # conditional Gaussian return given the clock
    vg_return = theta * g + sigma * np.sqrt(g) * rng.standard_normal(n_paths)

    log_ST = np.log(S0) + (r - q + omega) * T + vg_return
    return np.exp(log_ST)
    
