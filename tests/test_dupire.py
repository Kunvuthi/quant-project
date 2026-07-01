import numpy as np
from calibration.dupire import dupire_local_vol
from models.bsm import bsm_price


def bsm_price_grid(S0, K_grid, T_grid, r, sigma):
    """Build a (n_T, n_K) call-price grid under constant-vol BSM."""
    return np.array([[bsm_price(S0, K, T, r, sigma, 'call') for K in K_grid]
                     for T in T_grid])


def test_dupire_recovers_constant_vol():
    """Dupire on constant-vol BSM prices must return flat sigma in the interior."""
    S0, r, sigma = 100.0, 0.05, 0.20
    K_grid = np.linspace(85, 115, 31)      # interior strikes (avoid wing instability)
    T_grid = np.linspace(0.3, 1.5, 25)
    C = bsm_price_grid(S0, K_grid, T_grid, r, sigma)

    sigma_loc = dupire_local_vol(C, K_grid, T_grid, r, q=0.0, verbose=False)

    interior = sigma_loc[2:-2, 2:-2]       # trim edges (one-sided derivs)
    assert np.isclose(np.nanmean(interior), sigma, atol=1e-3)
    assert np.nanstd(interior) < 5e-3      # flat: low spread
    
def test_dupire_masks_thin_density():
    """The mask must NaN points where density is too thin to trust (wings)."""
    S0, r, sigma = 100.0, 0.05, 0.20
    K_grid = np.linspace(50, 160, 45)      # WIDE strikes -> genuine thin-density wings
    T_grid = np.linspace(0.05, 1.5, 25)    # incl. short T (narrow density)
    C = bsm_price_grid(S0, K_grid, T_grid, r, sigma)

    sigma_loc = dupire_local_vol(C, K_grid, T_grid, r, q=0.0, verbose=False)

    assert np.isnan(sigma_loc).any()       # some points masked
    # but the central region should survive
    j_atm = np.argmin(np.abs(K_grid - 100))
    i_mid = len(T_grid) // 2
    assert not np.isnan(sigma_loc[i_mid, j_atm])
    
def test_dupire_no_sawtooth():
    """Recovered constant-vol surface must be smooth in K (guards the
    nested-np.gradient second-derivative bug that caused even/odd sawtooth)."""
    S0, r, sigma = 100.0, 0.05, 0.20
    K_grid = np.linspace(85, 115, 31)
    T_grid = np.linspace(0.3, 1.5, 25)
    C = bsm_price_grid(S0, K_grid, T_grid, r, sigma)
    sigma_loc = dupire_local_vol(C, K_grid, T_grid, r, q=0.0, verbose=False)

    interior = sigma_loc[2:-2, 2:-2]
    # second difference along the strike axis (axis=1): large for sawtooth, ~0 for smooth
    second_diff = interior[:, 2:] - 2*interior[:, 1:-1] + interior[:, :-2]
    assert np.nanmax(np.abs(second_diff)) < 1e-3