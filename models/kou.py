import numpy as np

def kou_char_func(
    u: np.ndarray,
    S0: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
) -> np.ndarray:
    """Kou jump-diffusion characteristic function of log(S_T).

    Asymmetric double-exponential jumps. Requires eta1 > 1 for the compensator
    to converge (E[e^Y] finite on up-jumps).

    f_hat(u) = p*eta1/(eta1 - iu) + (1-p)*eta2/(eta2 + iu)
    kappa    = p*eta1/(eta1 - 1) + (1-p)*eta2/(eta2 + 1) - 1
    phi(u)   = exp( iu(log S0 + (r - q - 0.5 sigma^2 - lam*kappa) T) - 0.5 sigma^2 u^2 T )
               * exp( lam T (f_hat(u) - 1) )

    Reduces to BSM when lam = 0. Returns a complex array matching u.
    """
    if eta1 <= 1:
        raise ValueError(f"Kou requires eta1 > 1 for E[e^Y] to converge, got eta1={eta1}")
    
    f_hat = p*eta1/(eta1 - 1j * u) + (1-p)*eta2/(eta2 + 1j * u)           # single-jump char func, the two weighted terms
    kappa = p*eta1/(eta1 - 1) + (1-p)*eta2/(eta2 + 1) - 1            # f_hat(-i) - 1, i.e. the two terms with eta1-1, eta2+1
    drift = np.log(S0) + (r - q - 0.5*sigma**2 - lam*kappa)*T            
    diffusion = np.exp(1j*u*drift - 0.5*sigma**2*u**2*T)        
    jump = np.exp(lam*T*(f_hat - 1))            
    return diffusion * jump

def kou_cumulants(
    S0: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
) -> tuple[float, float]:
    """First and second cumulants of log(S_T) under Kou.

    Diffusion + jump cumulants add (independence). Single-jump moments of the
    double-exponential:
        E[Y]   = p/eta1 - (1-p)/eta2
        E[Y^2] = 2p/eta1^2 + 2(1-p)/eta2^2
    Then:
        c1 = log S0 + (r - q - 0.5 sigma^2 - lam*kappa) T + lam T E[Y]
        c2 = sigma^2 T + lam T E[Y^2]
    with kappa = p*eta1/(eta1-1) + (1-p)*eta2/(eta2+1) - 1.
    """
    kappa = p*eta1/(eta1-1) + (1-p)*eta2/(eta2+1) - 1   
    EY  = p/eta1 - (1-p)/eta2          
    EY2 = 2*p/eta1**2 + 2*(1-p)/eta2**2          
    c1 = np.log(S0) + (r - q - 0.5*sigma**2 - lam*kappa)*T + lam*T*EY
    c2 = sigma**2*T + lam*T*EY2
    return c1, c2

def kou_simulate_terminal(
    S0: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate terminal prices S_T under Kou, for MC validation of COS.

    One step to maturity. Same kappa compensator as kou_char_func, so it
    simulates exactly the process the char func describes.

    Jump draw: each jump is up with prob p (Exp(eta1)) or down with prob 1-p
    (Exp(eta2), negated). Unlike Merton the per-path jump sum has no closed
    normal form, so we draw the TOTAL number of jumps across all paths, sample
    each individual jump, and scatter-add them back to their paths.
    """
    rng = np.random.default_rng(seed)
    kappa = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1

    drift = (r - q - 0.5 * sigma**2 - lam * kappa) * T
    Z = rng.standard_normal(n_paths)
    diffusion = drift + sigma * np.sqrt(T) * Z

    # number of jumps per path, then total across all paths
    N = rng.poisson(lam * T, size=n_paths)
    total_jumps = int(N.sum())

    jump_sum = np.zeros(n_paths)
    if total_jumps > 0:
        # for each individual jump: up or down, then exponential magnitude
        is_up = rng.random(total_jumps) < p
        mags = np.where(
            is_up,
            rng.exponential(1.0 / eta1, size=total_jumps),      # up: +Exp(eta1)
            -rng.exponential(1.0 / eta2, size=total_jumps),     # down: -Exp(eta2)
        )
        # scatter each jump back to its path: path index repeated N[i] times
        path_idx = np.repeat(np.arange(n_paths), N)
        np.add.at(jump_sum, path_idx, mags)

    log_ST = np.log(S0) + diffusion + jump_sum
    return np.exp(log_ST)