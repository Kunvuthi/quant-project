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
    assert np.isclose(put, 5.5735, atol=1e-4)   # <-- VERIFY this number, see note

def test_call_decreasing_in_strike():
    """Call price must strictly decrease as strike rises."""
    strikes = np.array([90.0, 100.0, 110.0])
    prices = bsm_price(S=100.0, K=strikes, T=1.0, r=0.05, sigma=0.20,
                       option_type="call")
    assert np.all(np.diff(prices) < 0)