"""Generate the (params -> IV surface) training set for deep calibration.

Samples (H, eta, rho, v0) uniformly over the trained box, prices each via the
certified rBergomi pricer (surface.py), drops any surface with a failed inversion,
and saves params + surfaces + the sampling box to artifacts/. The network learns
this map; the box saved here is the region the network is valid in.
"""
from __future__ import annotations
import time
import numpy as np
from pathlib import Path

from calibration.deep_cal.surface import (
    rbergomi_iv_surface, surface_is_complete, SURFACE_SHAPE,
    MATURITIES, LOG_MONEYNESS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]   # deep_cal/ -> calibration/ -> repo root
ARTIFACTS = REPO_ROOT / "artifacts"

# --- Trained box. (H, eta, rho) as before; xi0 sampled as level + correlated pillar noise ---
PARAM_BOX = {
    "H":   (0.05, 0.25),
    "eta": (0.5,  4.0),
    "rho": (-0.95, -0.1),
    "xi0_level": (0.01, 0.16),   # base variance level; pillars are perturbations around it
}
PARAM_NAMES = ("H", "eta", "rho")          # scalar params; pillars appended after

# correlated-curve controls
PILLAR_CORR = 0.8      # adjacent-pillar correlation (AR(1)-style), smoother curves as ->1
PILLAR_SIGMA = 0.2     # log-perturbation size; larger = more curve-shape variety


def _pillar_chol() -> np.ndarray:
    """Cholesky of the AR(1) pillar correlation matrix C[i,j] = PILLAR_CORR**|i-j|.
    Built once per generate() call; maps iid normals to correlated pillar noise."""
    from calibration.deep_cal.surface import N_PILLARS
    idx = np.arange(N_PILLARS)
    C = PILLAR_CORR ** np.abs(idx[:, None] - idx[None, :])
    return np.linalg.cholesky(C)


def sample_params(n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 3 + N_PILLARS): [H, eta, rho, xi0_1..xi0_P].
    Pillars = level * exp(correlated log-noise), so positive and smoothly shaped."""
    from calibration.deep_cal.surface import N_PILLARS
    H = rng.uniform(*PARAM_BOX["H"], size=n)
    eta = rng.uniform(*PARAM_BOX["eta"], size=n)
    rho = rng.uniform(*PARAM_BOX["rho"], size=n)

    level = rng.uniform(*PARAM_BOX["xi0_level"], size=n)          # (n,)
    L = _pillar_chol()                                            # (P, P)
    z = rng.standard_normal((n, N_PILLARS))                      # iid
    corr_noise = z @ L.T                                         # (n, P) correlated
    pillars = level[:, None] * np.exp(PILLAR_SIGMA * corr_noise) # (n, P) positive, smooth

    return np.column_stack([H, eta, rho, pillars])              # (n, 3 + P)


def generate(
    n_surfaces: int,
    seed: int = 0,
    n_paths: int = 30_000,
    verbose_every: int = 100,
) -> dict:
    from calibration.deep_cal.surface import (
        rbergomi_iv_surface, surface_is_complete, SURFACE_SHAPE,
        MATURITIES, LOG_MONEYNESS, XI0_PILLARS, N_PILLARS,
    )
    rng = np.random.default_rng(seed)
    n_cols = 3 + N_PILLARS
    params_kept = np.empty((n_surfaces, n_cols))
    surfaces_kept = np.empty((n_surfaces, *SURFACE_SHAPE))

    filled, attempts, t0 = 0, 0, time.time()
    while filled < n_surfaces:
        p = sample_params(1, rng)[0]                             # (3 + P,)
        H, eta, rho = p[0], p[1], p[2]
        pillars = p[3:]
        surf = rbergomi_iv_surface(H, eta, rho, pillars, n_paths=n_paths,
                                   rng=np.random.default_rng(rng.integers(2**63)))
        attempts += 1
        if surface_is_complete(surf):
            params_kept[filled] = p
            surfaces_kept[filled] = surf
            filled += 1
            if verbose_every and filled % verbose_every == 0:
                dt = time.time() - t0
                print(f"{filled}/{n_surfaces}  ({dt/filled:.2f}s/surface, "
                      f"reject {1 - filled/attempts:.1%})")

    return {
        "params": params_kept,                                  # (n, 3 + P)
        "surfaces": surfaces_kept,
        "param_names": np.array(PARAM_NAMES + tuple(f"xi0_{i}" for i in range(N_PILLARS))),
        "param_box": np.array([PARAM_BOX[k] for k in PARAM_NAMES]),   # scalar bounds only
        "xi0_pillars": XI0_PILLARS,
        "pillar_corr": PILLAR_CORR,
        "pillar_sigma": PILLAR_SIGMA,
        "maturities": MATURITIES,
        "log_moneyness": LOG_MONEYNESS,
        "n_paths": n_paths,
        "reject_rate": 1 - filled / attempts,
    }

def save(data: dict, name: str = "rbergomi_flat_dataset.npz") -> Path:
    ARTIFACTS.mkdir(exist_ok=True)          # code owns the dir, no .gitkeep
    path = ARTIFACTS / name
    np.savez_compressed(path, **data)
    print(f"saved {data['params'].shape[0]} surfaces to {path}")
    return path


if __name__ == "__main__":
    data = generate(n_surfaces=4000, seed=0, n_paths=30_000)
    save(data, name="rbergomi_termstructure_dataset.npz")