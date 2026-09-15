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

ARTIFACTS = Path("artifacts")

# Trained box. The network is only trustworthy inside this hull, so calibration
# must land here with margin. Deliberately a bit wider than expected SPX values.
PARAM_BOX = {
    "H":   (0.05, 0.25),
    "eta": (0.5,  4.0),
    "rho": (-0.95, -0.1),
    "v0":  (0.01, 0.16),   # variance, so vol ~ [10%, 40%]
}
PARAM_NAMES = ("H", "eta", "rho", "v0")


def sample_params(n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 4) uniform draws over PARAM_BOX, columns ordered by PARAM_NAMES."""
    lows = np.array([PARAM_BOX[p][0] for p in PARAM_NAMES])
    highs = np.array([PARAM_BOX[p][1] for p in PARAM_NAMES])
    return rng.uniform(lows, highs, size=(n, 4))


def generate(
    n_surfaces: int,
    seed: int = 0,
    n_paths: int = 50_000,
    verbose_every: int = 100,
) -> dict:
    """Generate n_surfaces valid (param, surface) pairs. Rejects incompletes and
    resamples so the returned count is exactly n_surfaces. Returns a dict, does
    NOT save (caller saves, so timing runs can discard)."""
    rng = np.random.default_rng(seed)
    params_kept = np.empty((n_surfaces, 4))
    surfaces_kept = np.empty((n_surfaces, *SURFACE_SHAPE))

    filled, attempts, t0 = 0, 0, time.time()
    while filled < n_surfaces:
        p = sample_params(1, rng)[0]
        # each surface gets its own child rng so a specific one is reproducible
        surf = rbergomi_iv_surface(*p, n_paths=n_paths,
                                   rng=np.random.default_rng(rng.integers(2**63)))
        attempts += 1
        if surface_is_complete(surf):
            params_kept[filled] = p
            surfaces_kept[filled] = surf
            filled += 1
            if verbose_every and filled % verbose_every == 0:
                dt = time.time() - t0
                print(f"{filled}/{n_surfaces}  ({dt/filled:.2f}s/surface, "
                      f"reject rate {1 - filled/attempts:.1%})")

    return {
        "params": params_kept,                 # (n, 4)
        "surfaces": surfaces_kept,             # (n, 8, 11)
        "param_names": np.array(PARAM_NAMES),
        "param_box": np.array([PARAM_BOX[p] for p in PARAM_NAMES]),  # (4, 2)
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
    # --- stage 1: time one (DONE, 1.80s) ---
    # rng = np.random.default_rng(0)
    # p = sample_params(1, rng)[0]
    # t0 = time.time()
    # surf = rbergomi_iv_surface(*p, n_paths=30_000, rng=rng)
    # print(f"one surface: {time.time() - t0:.2f}s, complete={surface_is_complete(surf)}")

    # --- stage 2: probe reject rate ---
    # probe = generate(n_surfaces=100, seed=1, n_paths=30_000)
    # print("reject rate:", probe["reject_rate"])

    # --- stage 3: full run (uncomment after probe looks clean) ---
    data = generate(n_surfaces=3000, seed=0, n_paths=30_000)
    save(data)