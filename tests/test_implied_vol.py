import numpy as np
import pytest
from calibration.implied_vol import bsm_implied_vol
from models.bsm import bsm_price


def test_roundtrip_call(atm_params):
    """Price at known sigma, invert, recover sigma."""
    true_sigma = atm_params["sigma"]
    price = bsm_price(option_type="call", **atm_params)
    recovered = bsm_implied_vol(price, S=atm_params["S"], K=atm_params["K"],
                                T=atm_params["T"], r=atm_params["r"],
                                option_type="call")
    assert np.isclose(recovered, true_sigma, atol=1e-6)


def test_roundtrip_put(atm_params):
    """Same round-trip for a put."""
    true_sigma = atm_params["sigma"]
    price = bsm_price(option_type="put", **atm_params)
    recovered = bsm_implied_vol(price, S=atm_params["S"], K=atm_params["K"],
                                T=atm_params["T"], r=atm_params["r"],
                                option_type="put")
    assert np.isclose(recovered, true_sigma, atol=1e-6)


def test_roundtrip_with_carry(atm_params):
    """Round-trip with b != r must recover sigma — the new carry path."""
    true_sigma = atm_params["sigma"]
    b = atm_params["r"] - 0.013          # dividend-like carry
    price = bsm_price(option_type="call", b=b, **atm_params)
    recovered = bsm_implied_vol(price, S=atm_params["S"], K=atm_params["K"],
                                T=atm_params["T"], r=atm_params["r"],
                                option_type="call", b=b)
    assert np.isclose(recovered, true_sigma, atol=1e-6)


def test_roundtrip_otm_strikes(atm_params):
    """Round-trip across a range of strikes, not just ATM."""
    base = {k: v for k, v in atm_params.items() if k != "K"}
    for K in [80.0, 90.0, 110.0, 120.0]:
        price = bsm_price(option_type="call", K=K, **base)
        recovered = bsm_implied_vol(price, S=base["S"], K=K, T=base["T"],
                                    r=base["r"], option_type="call")
        assert np.isclose(recovered, base["sigma"], atol=1e-6)


def test_arbitrage_violation_returns_nan(atm_params):
    """A price above the upper bound returns NaN, not a garbage sigma."""
    too_expensive = atm_params["S"] * 2          # way above the call UB (= fwd)
    result = bsm_implied_vol(too_expensive, S=atm_params["S"], K=atm_params["K"],
                             T=atm_params["T"], r=atm_params["r"], option_type="call")
    assert np.isnan(result)