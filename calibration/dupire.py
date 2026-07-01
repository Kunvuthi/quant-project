import numpy as np

def dupire_local_vol(
    C: np.ndarray,
    K_grid: np.ndarray,
    T_grid: np.ndarray,
    r: float,
    q: float = 0.0,
    density_floor: float = 1e-6,
    epsilon: float = 1e-2,
    verbose: bool = True,
) -> np.ndarray:
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
    row_peak = np.max(d2Cd2K, axis=1, keepdims=True)   # shape (n_T, 1)
    relative_floor = epsilon * row_peak
    abs_bad = d2Cd2K < density_floor
    rel_bad = d2Cd2K < relative_floor
    bad = abs_bad | rel_bad
    sigma2_loc = np.full_like(C, np.nan)
    sigma2_loc[~bad] = numerator[~bad] / denominator[~bad]
    sigma_loc = np.sqrt(sigma2_loc)

    if verbose:
        n_total = C.size
        n_abs = int(abs_bad.sum())
        n_rel = int((rel_bad & ~abs_bad).sum())     # relative-only (not already caught by absolute)
        n_nan = int(np.isnan(sigma_loc).sum())
        n_negvar = n_nan - int(bad.sum())
        print(f"Dupire extraction: {n_total} grid points")
        print(f"  masked (absolute density < {density_floor:g}): {n_abs}")
        print(f"  masked (relative, < {epsilon:g} x row peak): {n_rel}")
        if n_negvar > 0:
            print(f"  additional NaN (negative variance): {n_negvar}")
        print(f"  valid points: {n_total - n_nan} ({100*(n_total-n_nan)/n_total:.1f}%)")
    
    return sigma_loc