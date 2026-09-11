import numpy as np
from typing import Literal, Callable
from models.montecarlo import simulate_gbm_terminal

# ---- The estimator (valid only on iid units) ---- #

def mc_price(payoffs: np.ndarray, r: float, T: float) -> tuple[float, float]:
    """Discounted mean and its standard error over INDEPENDENT samples.

    CONTRACT: payoffs is 1-D, one iid draw per element. SE = sd/sqrt(n) is a lie
    if the rows are correlated, so antithetic pairing is resolved upstream, never
    here. This is the W1 590k reshape ghost lifted up one layer: keep the estimator
    unable to be fooled by construction.
    """
    payoffs = np.asarray(payoffs, dtype=np.float64)
    if payoffs.ndim != 1:
        raise ValueError(f"mc_price expects 1-D iid payoffs, got shape {payoffs.shape}")
    n = payoffs.shape[0]
    disc = np.exp(-r * T) * payoffs
    price = disc.mean()
    se = disc.std(ddof=1) / np.sqrt(n)
    return price, se


# ---- Payoffs (thin; any exotic supplies its own and reuses mc_price) ---- #

def european_payoff(
    S_T: np.ndarray, K: float, option_type: Literal["call", "put"]
) -> np.ndarray:
    """(S_T - K)+ for a call, (K - S_T)+ for a put. Scalar K, vector S_T."""
    S_T = np.asarray(S_T, dtype=np.float64)
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - S_T, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _resolve_antithetic(payoffs: np.ndarray) -> np.ndarray:
    """Collapse antithetic payoff pairs into iid pair-averages.

    Layout convention (matches bsm_price_mc / mc_greeks since W2): the first half
    are f(S+), the second half f(S-), so unit i is (payoffs[i] + payoffs[i+n])/2.
    Pairing is on the PAYOFF because the payoff is nonlinear; averaging S_T first
    would be a different, wrong estimator. Explicit slice add, not reshape, so the
    pairing stays visually unmissable (the W2 lesson).
    """
    m = payoffs.shape[0]
    if m % 2 != 0:
        raise ValueError(f"antithetic payoffs need an even count, got {m}")
    n = m // 2
    return (payoffs[:n] + payoffs[n:]) / 2.0   # (n,) iid pair-averages


# ---- The common-case wrapper ---- #

def european_price_from_samples(
    S_T: np.ndarray,
    K: float,
    r: float,
    T: float,
    option_type: Literal["call", "put"],
    antithetic: bool = False,
) -> tuple[float, float]:
    """European price and SE from terminal samples.

    antithetic=False: S_T is n iid draws, priced directly.
    antithetic=True:  S_T is [f-half of S+ ; f-half of S-] in the W2 layout;
                      payoffs are pair-averaged into n/2 iid units before mc_price,
                      so the returned SE already reflects the variance reduction and
                      stays honest.
    """
    payoff = european_payoff(S_T, K, option_type)
    if antithetic:
        payoff = _resolve_antithetic(payoff)
    return mc_price(payoff, r, T)

def bsm_price_mc(
    S: float, K: float, T: float, r: float, sigma: float,
    N: int = 10_000, option_type: Literal["call", "put"] = "call",
    antithetic: bool = False, seed: int | None = None,
) -> tuple[float, float]:
    """European GBM price via MC. Simulate terminal, delegate to the agnostic
    core. Signature preserved from the pre-refactor version so call sites are
    unchanged; only the import path moves."""
    S_T = simulate_gbm_terminal(S, T, r, sigma, N, antithetic, seed)
    return european_price_from_samples(S_T, K, r, T, option_type, antithetic=antithetic)


# ---- Optional: generic exotic entry point (Phase 3 hook, leave stubbed) ---- #

def price_from_samples(
    paths_or_terminal: np.ndarray,
    payoff_fn: Callable[[np.ndarray], np.ndarray],
    r: float,
    T: float,
    antithetic: bool = False,
) -> tuple[float, float]:
    """Fully payoff-agnostic version: caller supplies payoff_fn (path- or
    terminal-dependent). Kept minimal now; fleshed out when deep hedging /
    path-dependent payoffs land in Phase 3. Same antithetic contract as above."""
    raise NotImplementedError