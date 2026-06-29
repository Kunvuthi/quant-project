# tests/test_pde.py
import numpy as np
from pricing.pde import price_call_localvol
from models.bsm import bsm_price

# --- module-level helpers ---
def const_vol(sigma):
    """Factory for a constant local-vol function."""
    return lambda S, t: sigma * np.ones_like(S)


def test_pde_reproduces_bsm():
    """Constant-vol PDE must match the closed-form BSM price."""
    S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    pde = price_call_localvol(K=K, S0=S0, T=T, r=r, q=q,
                              local_vol_fn=const_vol(sigma), sigma_max=sigma,
                              n_S=200, n_t=200)
    bsm = bsm_price(S0, K, T, r, sigma, 'call')
    assert np.isclose(pde, bsm, atol=5e-3)        # discretization error at 200x200


def test_pde_second_order_convergence():
    """Error must quarter as the grid doubles (CN is O(dx^2, dt^2))."""
    S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20
    bsm = bsm_price(S0, K, T, r, sigma, 'call')

    errs = []
    for n in [50, 100, 200]:
        p = price_call_localvol(K=K, S0=S0, T=T, r=r, q=q,
                                local_vol_fn=const_vol(sigma), sigma_max=sigma,
                                n_S=n, n_t=n)
        errs.append(abs(p - bsm))

    # each doubling should give ratio ~4; allow a generous band (kink perturbs it)
    ratio1 = errs[0] / errs[1]
    ratio2 = errs[1] / errs[2]
    assert 3.0 < ratio1 < 5.0
    assert 3.0 < ratio2 < 5.0


def test_pde_otm_and_itm():
    """PDE matches BSM across moneyness, not just ATM."""
    S0, T, r, q, sigma = 100.0, 1.0, 0.05, 0.0, 0.20
    for K in [85.0, 100.0, 115.0]:
        pde = price_call_localvol(K=K, S0=S0, T=T, r=r, q=q,
                                  local_vol_fn=const_vol(sigma), sigma_max=sigma,
                                  n_S=200, n_t=200)
        bsm = bsm_price(S0, K, T, r, sigma, 'call')
        assert np.isclose(pde, bsm, atol=5e-3)


# def test_pde_put_call_parity():
#     """Parity must hold (guards the boundary/sign logic)."""
#     # skipped as only have call-only
