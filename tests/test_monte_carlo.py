import numpy as np
from pricing.monte_carlo import mc_price, european_price_from_samples


def test_antithetic_se_reduction():
    """The whole point of antithetic pairing: the SE from n negatively-correlated
    pairs must beat 2n iid draws. If _resolve_antithetic is wired wrong (averaging
    the wrong axis, or not pairing at all) the price still looks right, so ONLY the
    SE ratio catches it. Certifies the pairing survived the GBM relocation.

    Ground truth: GBM terminal, closed-form sampling, so both paths price the same
    option; we compare their reported SEs, not their prices.
    """
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
    n = 200_000
    rng = np.random.default_rng(0)

    # iid baseline: 2n independent draws
    Z = rng.standard_normal(2 * n)
    S_iid = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    _, se_iid = european_price_from_samples(S_iid, K, r, T, "call", antithetic=False)

    # antithetic: n pairs (Z, -Z) in the W2 [S+ ; S-] layout, same total 2n draws
    Zp = rng.standard_normal(n)
    S_plus = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Zp)
    S_minus = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * (-Zp))
    S_anti = np.concatenate([S_plus, S_minus])
    _, se_anti = european_price_from_samples(S_anti, K, r, T, "call", antithetic=True)

    # antithetic SE must be strictly smaller at equal simulation cost.
    assert se_anti < se_iid, f"no variance reduction: se_anti={se_anti:.5f} se_iid={se_iid:.5f}"


def test_mc_price_rejects_2d():
    """mc_price must refuse non-1-D input rather than silently averaging the wrong
    axis (the failure mode behind the W2 590k spurious 'reduction')."""
    import pytest
    with pytest.raises(ValueError):
        mc_price(np.ones((100, 2)), r=0.0, T=1.0)