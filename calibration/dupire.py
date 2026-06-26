import numpy as np

def dupire_local_vol(C, K_grid, T_grid, r, q=0.0, density_floor=1e-6, verbose=True):
    """
    Extract the Dupire local-vol surface from a grid of call prices.

    Parameters
    ----------
    C : ndarray, shape (n_T, n_K)
        Call prices. Row = maturity, column = strike.
    K_grid : ndarray, shape (n_K,)
        Strikes (columns of C). Assumed sorted ascending.
    T_grid : ndarray, shape (n_T,)
        Maturities (rows of C). Assumed sorted ascending.
    r, q : float
        Rate and dividend yield.

    Returns
    -------
    sigma_loc : ndarray
        Local vol on the interior grid (edges dropped by central differencing).
    """
    # --- derivatives via central differences ---
    # dC/dT : differentiate along axis 0 (maturity, rows)
    dCdt = np.gradient(C, T_grid, axis=0)
    
    # dC/dK : differentiate along axis 1 (strike, columns)
    dCdK = np.gradient(C, K_grid, axis=1)
    
    # d2C/dK2 : second difference along axis 1
    dK = K_grid[1] - K_grid[0]                          # uniform spacing
    d2Cd2K = np.empty_like(C)
    d2Cd2K[:, 1:-1] = (C[:, 2:] - 2*C[:, 1:-1] + C[:, :-2]) / dK**2
    d2Cd2K[:, 0]  = d2Cd2K[:, 1]                         # edge fill (will be masked anyway)
    d2Cd2K[:, -1] = d2Cd2K[:, -2]
    
    # --- assemble Dupire's formula pointwise on the interior ---
    K = K_grid[np.newaxis, :]  # explicit broadcast over strikes
    numerator   = dCdt + (r - q) * K * dCdK + q * C
    denominator = 0.5 * K**2 * d2Cd2K
    
    # mask where the density (d2C/dK2) is too small to trust the division
    bad = d2Cd2K < density_floor
    sigma2_loc = np.full_like(C, np.nan)
    sigma2_loc[~bad] = numerator[~bad] / denominator[~bad]
    
    sigma_loc = np.sqrt(sigma2_loc)   # sqrt of negative -> nan (extra safety)
    
    if verbose:
        n_total = C.size
        n_density = int(bad.sum())                          # masked by density floor
        n_nan = int(np.isnan(sigma_loc).sum())              # total nan (density + neg variance)
        n_negvar = n_nan - n_density                        # additionally lost to negative variance
        print(f"Dupire extraction: {n_total} grid points")
        print(f"  masked (density < {density_floor:g}): {n_density} "
              f"({100*n_density/n_total:.1f}%) - untrustworthy wings")
        if n_negvar > 0:
            print(f"  additional NaN (negative variance from FD error): {n_negvar}")
        if n_density > 0:
            # report the strike/maturity extent of the masked region
            bad_T_idx, bad_K_idx = np.where(bad)
            print(f"  masked strikes span K=[{K_grid[bad_K_idx].min():.0f}, "
                  f"{K_grid[bad_K_idx].max():.0f}], "
                  f"maturities T=[{T_grid[bad_T_idx].min():.2f}, "
                  f"{T_grid[bad_T_idx].max():.2f}]")
        n_valid = n_total - n_nan
        print(f"  valid points: {n_valid} ({100*n_valid/n_total:.1f}%)")
    
    return sigma_loc