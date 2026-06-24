import numpy as np
from models.bsm import bsm_price

def test_canonical_atm_call(atm_params):
    """The textbook value every BSM implementation must reproduce."""
    price = bsm_price(option_type="call", **atm_params)
    assert np.isclose(price, 10.4506, atol=1e-4)
    
def test_put_call_parity(atm_params):
    """C - P = S - K e^{-rT}, to machine precision."""
    call = bsm_price(option_type="call", **atm_params)
    put = bsm_price(option_type="put", **atm_params)
    S, K, r, T = atm_params["S"], atm_params["K"], atm_params["r"], atm_params["T"]
    parity_rhs = S - K * np.exp(-r * T)
    assert np.isclose(call - put, parity_rhs, atol=1e-10)

def test_put_value(atm_params):
    """Pin the ATM put price (frozen from current implementation)."""
    put = bsm_price(option_type="put", **atm_params)
    assert np.isclose(put, 5.5735, atol=1e-4)  

def test_call_decreasing_in_strike():
    """Call price must strictly decrease as strike rises."""
    strikes = np.array([90.0, 100.0, 110.0])
    prices = bsm_price(S=100.0, K=strikes, T=1.0, r=0.05, sigma=0.20,
                       option_type="call")
    assert np.all(np.diff(prices) < 0)
    
def test_carry_reduces_to_plain_bsm(atm_params):
    """b = r explicitly must equal the b=None default (the carry factor is 1)."""
    default = bsm_price(option_type="call", **atm_params)
    explicit = bsm_price(option_type="call", b=atm_params["r"], **atm_params)
    assert np.isclose(default, explicit, atol=1e-12)


def test_carry_parity(atm_params):
    """Generalised parity: C - P = S e^{(b-r)T} - K e^{-rT}."""
    p = {**atm_params, "b": 0.02}          # b != r: a dividend-like carry
    call = bsm_price(option_type="call", **p)
    put = bsm_price(option_type="put", **p)
    S, K, r, T, b = p["S"], p["K"], p["r"], p["T"], p["b"]
    rhs = S * np.exp((b - r) * T) - K * np.exp(-r * T)
    assert np.isclose(call - put, rhs, atol=1e-10)


def test_expiry_intrinsic_T0():
    """T=0 returns intrinsic value, not NaN."""
    itm_call = bsm_price(S=110.0, K=100.0, T=0.0, r=0.05, sigma=0.2, option_type="call")
    otm_call = bsm_price(S=90.0, K=100.0, T=0.0, r=0.05, sigma=0.2, option_type="call")
    assert np.isclose(itm_call, 10.0, atol=1e-12)
    assert np.isclose(otm_call, 0.0, atol=1e-12)


def test_zero_vol_forward_intrinsic():
    """sigma=0: discounted intrinsic of the forward S e^{bT}."""
    # b=r=0.05, T=1: forward = 100*e^0.05 = 105.127; call strike 100
    price = bsm_price(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.0, option_type="call")
    expected = np.exp(-0.05) * max(100 * np.exp(0.05) - 100, 0.0)
    assert np.isclose(price, expected, atol=1e-12)


def test_vectorised_with_edge():
    """A strike array straddling an edge: T=0 slot intrinsic, rest finite."""
    strikes = np.array([90.0, 100.0, 110.0])
    prices = bsm_price(S=100.0, K=strikes, T=0.0, r=0.05, sigma=0.2, option_type="call")
    # T=0 everywhere -> intrinsic (100-K)+ = [10, 0, 0]
    assert np.allclose(prices, [10.0, 0.0, 0.0], atol=1e-12)