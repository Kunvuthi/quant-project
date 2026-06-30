# tests/test_pde.py
import numpy as np
from pricing.pde import price_localvol
from models.bsm import bsm_price

# --- module-level helpers ---
def const_vol(sigma):
    """Factory for a constant local-vol function."""
    return lambda S, t: sigma * np.ones_like(S)


def test_pde_reproduces_bsm():
    """Constant-vol PDE must match the closed-form BSM price."""
    S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    pde = price_localvol(K=K, S0=S0, T=T, r=r, q=q,
                         local_vol_fn=const_vol(sigma), sigma_max=sigma,
                         n_S=200, n_t=200, option_type='call')
    bsm = bsm_price(S0, K, T, r, sigma, 'call')
    assert np.isclose(pde, bsm, atol=5e-3)


def test_pde_second_order_convergence():
    """Error must quarter as the grid doubles (CN is O(dx^2, dt^2))."""
    S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    bsm = bsm_price(S0, K, T, r, sigma, 'call')
    errs = []
    for n in [50, 100, 200]:
        p = price_localvol(K=K, S0=S0, T=T, r=r, q=q,
                           local_vol_fn=const_vol(sigma), sigma_max=sigma,
                           n_S=n, n_t=n, option_type='call')
        errs.append(abs(p - bsm))
    ratio1 = errs[0] / errs[1]
    ratio2 = errs[1] / errs[2]
    assert 3.0 < ratio1 < 5.0
    assert 3.0 < ratio2 < 5.0


def test_pde_otm_and_itm():
    """PDE matches BSM across moneyness, not just ATM."""
    S0, T, r, q, sigma = 100.0, 1.0, 0.05, 0.0, 0.20
    for K in [85.0, 100.0, 115.0]:
        pde = price_localvol(K=K, S0=S0, T=T, r=r, q=q,
                             local_vol_fn=const_vol(sigma), sigma_max=sigma,
                             n_S=200, n_t=200, option_type='call')
        bsm = bsm_price(S0, K, T, r, sigma, 'call')
        assert np.isclose(pde, bsm, atol=5e-3)


def test_pde_put_call_parity():
    """C - P from the PDE must satisfy parity (model-free check on both pricers)."""
    S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    call = price_localvol(K=K, S0=S0, T=T, r=r, q=q,
                          local_vol_fn=const_vol(sigma), sigma_max=sigma,
                          n_S=200, n_t=200, option_type='call')
    put = price_localvol(K=K, S0=S0, T=T, r=r, q=q,
                         local_vol_fn=const_vol(sigma), sigma_max=sigma,
                         n_S=200, n_t=200, option_type='put')
    parity_rhs = S0 * np.exp((r - q - r) * T) - K * np.exp(-r * T)   # = S0 e^{-qT} - K e^{-rT}
    # note: forward grows at b=r-q, so spot term is S0 e^{(b-r)T} = S0 e^{-qT}
    assert np.isclose(call - put, parity_rhs, atol=5e-3)
