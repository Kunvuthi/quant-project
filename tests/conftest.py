import pytest

@pytest.fixture
def atm_params():
    """Canonical ATM test case: the 10.4506 call."""
    return {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.20}