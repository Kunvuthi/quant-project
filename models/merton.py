import numpy as np

def merton_char_func(
    u: np.ndarray,
    S0: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    lam: float,
    mu_j: float,
    delta_j: float,
) -> np.ndarray:
    """Merton jump-diffusion characteristic function of log(S_T).

    phi(u) = exp( iu(log S0 + (r - q - 0.5 sigma^2 - lam*kappa) T) - 0.5 sigma^2 u^2 T )
             * exp( lam T ( exp(iu mu_j - 0.5 u^2 delta_j^2) - 1 ) )
    with kappa = exp(mu_j + 0.5 delta_j^2) - 1, the martingale compensator.

    Reduces to the BSM log-price char func when lam = 0 (jump factor -> 1,
    drift correction vanishes). Returns a complex array matching u.
    """
    kappa = np.exp(mu_j + 0.5 * delta_j**2) - 1.0          # martingale compensator
    drift = np.log(S0) + (r - q - 0.5 * sigma**2 - lam * kappa) * T
    diffusion = np.exp(1j * u * drift - 0.5 * sigma**2 * u**2 * T)
    jump = np.exp(lam * T * (np.exp(1j * u * mu_j - 0.5 * u**2 * delta_j**2) - 1.0))
    return diffusion * jump

def merton_cumulants(
    S0: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    lam: float,
    mu_j: float,
    delta_j: float,
) -> tuple[float, float]:
    """First and second cumulants of log(S_T) under Merton, closed form.

    Diffusion and jump cumulants ADD (independence -> log char funcs add):

        c1 = log S0 + (r - q - 0.5 sigma^2 - lam*kappa) T + lam T mu_j
        c2 = sigma^2 T + lam T (mu_j^2 + delta_j^2)

    with kappa = exp(mu_j + 0.5 delta_j^2) - 1. The lam*T*mu_j in c1 is the mean
    total jump contribution; the lam*T*(mu_j^2 + delta_j^2) in c2 is the jump
    variance via the compound-Poisson second moment E[Y^2] = mu_j^2 + delta_j^2.
    Used to set the COS truncation range, so getting c2 right (including mu_j^2)
    keeps the wings accurate.
    """
    kappa = np.exp(mu_j + (delta_j**2 / 2)) - 1
    c1 = np.log(S0) + (r - q - 0.5*sigma**2 - lam*kappa)*T + lam*T*mu_j
    c2 = (sigma**2)*T + lam*T*(mu_j**2 + delta_j**2)
    return c1, c2

def merton_simulate_terminal(
    S0: float,
    r: float,
    q: float,
    T: float,
    sigma: float,
    lam: float,
    mu_j: float,
    delta_j: float,
    n_paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate terminal prices S_T under Merton, for MC validation of COS.

    One step to maturity (European payoff needs only S_T). Diffusion drift uses
    the SAME kappa compensator as merton_char_func, so this simulates exactly
    the process the char func describes, that identity is what makes the
    COS-vs-MC comparison a valid ground-truth check.

    Vectorized jump draw: given N jumps on a path, their sum is
    Normal(N*mu_j, N*delta_j^2), so draw one Poisson N per path then one normal
    per path with those per-path parameters, no inner loop over individual jumps.
    Returns an array of terminal prices, shape (n_paths,).
    """
    rng = np.random.default_rng(seed)
    kappa = np.exp(mu_j + 0.5 * delta_j**2) - 1.0
    
    # diffusion part (drift + brownian), shape (n_paths,):
    drift = (r - q - 0.5*sigma**2 - lam*kappa) * T
    Z = rng.standard_normal(n_paths)
    diffusion = drift + sigma * np.sqrt(T) * Z
    
    # jump part, vectorized:
    N = rng.poisson(lam * T, size=n_paths)       # jumps per path
    jump_mean_per_path  = N * mu_j
    jump_std_per_path  = np.sqrt(N) * delta_j      # sqrt of N*delta_j^2
    jump = rng.normal(loc=jump_mean_per_path, scale=jump_std_per_path)
    # note: when N=0 the scale is 0, np.random.normal(loc=0, scale=0) = 0, correct
    
    log_ST = np.log(S0) + diffusion + jump
    return np.exp(log_ST)