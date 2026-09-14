# params -> IV surface via the certified pricer (shared by dataset AND tomorrow's real-data path)
"""Parameters -> rBergomi implied-vol surface on the canonical grid.

The shared foundation of the deep-calibration pipeline. dataset.py calls this to
build training targets; calibrate.py calls it (tomorrow) on the real-data path.
Because both paths route through this one function, the network is always trained
against exactly the surface convention it is later calibrated against, with no
drift between the two.

Normalised, forward-relative world: S0 = 1, r = 0, so the forward is 1, ATM is
k = 0, and strikes are exp(k). This makes the surface spot-invariant, which is
why tomorrow's SPX quotes (converted to log-moneyness against the parity-implied
forward per maturity) drop straight onto this same grid.
"""
from __future__ import annotations
import numpy as np
from models.rbergomi import simulate_rbergomi, RBergomiParams
from pricing.monte_carlo import european_price_from_samples
from calibration.implied_vol import bsm_implied_vol

# --- Canonical grid. Defined ONCE here; every other deep_cal module imports it. ---
MATURITIES = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])   # (8,) years
LOG_MONEYNESS = np.linspace(-0.30, 0.30, 11)                          # (11,) k = log(K/F)
STRIKES = np.exp(LOG_MONEYNESS)                                       # forward = 1
N_MATURITIES = MATURITIES.size
N_STRIKES = LOG_MONEYNESS.size
SURFACE_SHAPE = (N_MATURITIES, N_STRIKES)                             # (8, 11) -> 88 nodes


def rbergomi_iv_surface(
    H: float,
    eta: float,
    rho: float,
    v0: float,
    n_paths: int = 50_000,
    rng: np.random.Generator | None = None,
    n_steps_per_year: int = 150,
    min_steps: int = 20,
) -> np.ndarray:
    """rBergomi IV surface on (MATURITIES x LOG_MONEYNESS). Returns (8, 11).

    v0 is the flat forward variance xi0(t) = v0 (this version). NaN entries mean
    IV inversion failed at that node (far-wing MC noise breaching no-arb bounds);
    dataset.py decides whether to keep or reject such a surface.
    """
    if rng is None:
        rng = np.random.default_rng()

    xi0 = lambda t: np.full_like(np.asarray(t, dtype=float), v0)   # single extension point
    params = RBergomiParams(H=H, eta=eta, rho=rho, xi0=xi0)

    surface = np.empty(SURFACE_SHAPE)
    for i, T in enumerate(MATURITIES):
        # Per-maturity grid, resolution scaled with T: short maturities need a
        # fine grid too (the v_left left-point bias shrinks like 1/n_steps), and
        # scaling avoids over-resolving the long end, which is both cheaper and
        # more uniform in bias across maturities than one global grid.
        n_steps = max(min_steps, round(n_steps_per_year * T))
        t_grid = np.linspace(T / n_steps, T, n_steps)

        S, _, _ = simulate_rbergomi(t_grid, params, n_paths, rng)
        S_T = S[:, -1]

        for j, K in enumerate(STRIKES):
            otype = "call" if K >= 1.0 else "put"      # price OTM: tighter IV, less MC noise
            price, _ = european_price_from_samples(S_T, K, r=0.0, T=T, option_type=otype)
            surface[i, j] = bsm_implied_vol(price, S=1.0, K=K, T=T, r=0.0, option_type=otype)

    return surface


def surface_is_complete(surface: np.ndarray) -> bool:
    """True if every node inverted (no NaNs). dataset.py uses this to filter."""
    return not np.isnan(surface).any()