# frozen-network input optimisation (the actual calibrator)
"""Calibrate rBergomi by optimising a frozen network's INPUTS to match a target
IV surface. The network is the fast differentiable forward map; calibration is
gradient descent on (H, eta, rho, v0), not on weights.

This is the inference path, so it applies the SAME scalers saved at train time.
Params are optimised in normalised space (well-conditioned, and the trained box
maps to a roughly unit cube), then un-normalised for the returned answer.
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path

from calibration.deep_cal.network import SurfaceNet

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO_ROOT / "artifacts"

def load_model(checkpoint_name: str = "surfacenet.pt"):
    """Rebuild the net and its scalers/mask/weights from the checkpoint.
    Everything the inference path needs travels together with the weights."""
    ckpt = torch.load(ARTIFACTS / checkpoint_name, weights_only=False)
    net = SurfaceNet(n_inputs=int(ckpt["n_inputs"]), n_outputs=int(ckpt["n_outputs"]))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)                     # freeze: differentiate wrt INPUTS only
    return net, ckpt

def _run_descent(net, target, w, x_mean, x_std, y_mean, y_std, x0, steps, lr):
    """One gradient-descent calibration from a single start x0 (raw space).
    Returns (theta_raw_np, final_loss)."""
    theta_raw = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta_raw], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        theta_norm = (theta_raw - x_mean) / x_std             # apply INPUT scaler
        pred_norm = net(theta_norm.unsqueeze(0)).squeeze(0)   # (88,) normalised
        pred_iv = pred_norm * y_std + y_mean                  # invert OUTPUT scaler -> real IV
        loss = (w * (pred_iv - target) ** 2).sum() / (w > 0).sum()
        loss.backward()
        opt.step()
    return theta_raw.detach().numpy(), float(loss.detach())



def calibrate(
    target_surface: np.ndarray,          # (8, 11) target IVs on the canonical grid
    checkpoint_name: str = "surfacenet_ts.pt",
    x0: np.ndarray | None = None,        # raw-space start (full param width); None -> auto
    steps: int = 500,
    lr: float = 1e-2,
    weights: np.ndarray | None = None,   # (8,11) override; default = training node weights
    n_restarts: int = 4,
    seed: int = 0,
    pillar_range: tuple[float, float] = (0.01, 0.16),   # plausible forward-variance level
) -> dict:
    net, ckpt = load_model(checkpoint_name)
    net = net.double()
    valid = np.asarray(ckpt["valid_nodes"])
    n_inputs = int(ckpt["n_inputs"])
    scalar_box = np.asarray(ckpt["param_box"], dtype=np.float64)   # (3, 2): H, eta, rho only
    n_scalars = scalar_box.shape[0]
    n_pillars = n_inputs - n_scalars                              # 5

    x_mean = torch.tensor(np.asarray(ckpt["x_mean"], dtype=np.float64))
    x_std = torch.tensor(np.asarray(ckpt["x_std"], dtype=np.float64))
    y_mean = torch.tensor(np.asarray(ckpt["y_mean"], dtype=np.float64))
    y_std = torch.tensor(np.asarray(ckpt["y_std"], dtype=np.float64))

    target = torch.tensor(target_surface.reshape(-1), dtype=torch.float64)

    if weights is None:
        w = torch.tensor(np.asarray(ckpt["node_weights"], dtype=np.float64))
    else:
        w = torch.tensor(weights.reshape(-1), dtype=np.float64)

    # --- full-width start + restarts: scalars from their box, pillars from pillar_range ---
    plo, phi = pillar_range
    scalar_lo, scalar_hi = scalar_box[:, 0], scalar_box[:, 1]

    def _default_start():                       # scalar box centre + flat mid pillars
        s0 = scalar_box.mean(axis=1)
        p0 = np.full(n_pillars, 0.5 * (plo + phi))
        return np.concatenate([s0, p0])

    starts = [_default_start() if x0 is None else np.asarray(x0, dtype=np.float64)]
    if n_restarts > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_restarts):
            s = rng.uniform(scalar_lo, scalar_hi)
            p = rng.uniform(plo, phi, size=n_pillars)
            starts.append(np.concatenate([s, p]))

    best = None
    for s in starts:
        theta, final_loss = _run_descent(net, target, w, x_mean, x_std,
                                          y_mean, y_std, s, steps, lr)
        with torch.no_grad():
            tn = (torch.tensor(theta) - x_mean) / x_std
            fit = (net(tn.unsqueeze(0)).squeeze(0) * y_std + y_mean).numpy()
        rmse = float(np.sqrt(np.mean(((fit - target_surface.reshape(-1))[valid]) ** 2)))
        if best is None or rmse < best["surface_rmse"]:
            best = {"theta": theta, "surface_rmse": rmse, "final_loss": final_loss}

    theta = best["theta"]
    scalars, pillars = theta[:n_scalars], theta[n_scalars:]

    # in-box: scalars against their box; pillars against positivity + plausible level
    scalars_ok = bool(np.all(scalars >= scalar_lo) and np.all(scalars <= scalar_hi))
    pillars_ok = bool(np.all(pillars > 0.0) and np.all(pillars <= 1.5 * phi))
    in_box = scalars_ok and pillars_ok

    return {
        "params": dict(zip([str(n) for n in ckpt["param_names"]], theta)),
        "theta": theta,
        "scalars": scalars,
        "pillars": pillars,
        "in_box": in_box,
        "scalars_ok": scalars_ok,
        "pillars_ok": pillars_ok,
        "surface_rmse": best["surface_rmse"],
        "final_loss": best["final_loss"],
    }