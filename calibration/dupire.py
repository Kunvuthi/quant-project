import numpy as np

def dupire_local_vol(C, K_grid, T_grid, r, q=0.0):
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
    d2Cd2K = np.gradient(dCdK, K_grid, axis=1)
    
    # --- assemble Dupire's formula pointwise on the interior ---
    K = K_grid[np.newaxis, :]  # explicit broadcast over strikes
    sigma2_loc = (dCdt + (r-q)*K*dCdK + q*C) / (0.5 * (K**2) * d2Cd2K)
    
    return np.sqrt(sigma2_loc)