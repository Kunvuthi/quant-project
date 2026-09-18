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
    
def _to_theta(u: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Map unconstrained u -> theta strictly inside [lo, hi] via a sigmoid.
    theta = lo + (hi - lo) * sigmoid(u). Optimiser ranges over all of R; theta
    can never leave the box, so the network is never queried out of domain."""
    return lo + (hi - lo) * torch.sigmoid(u)


def _to_u(theta: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Inverse map for building a start: u = logit((theta - lo) / (hi - lo)).
    theta is clipped just inside the box first so the logit stays finite."""
    frac = (theta - lo) / (hi - lo)
    frac = np.clip(frac, 1e-4, 1 - 1e-4)
    return np.log(frac / (1 - frac))


def _run_descent(net, target, w, x_mean, x_std, y_mean, y_std,
                 u0, lo, hi, steps, lr):
    """Gradient descent in unconstrained u-space, box-safe. Returns (theta_np, loss)."""
    u = torch.tensor(u0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([u], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        theta = _to_theta(u, lo, hi)                          # always in-box
        theta_norm = (theta - x_mean) / x_std                # input scaler
        pred_norm = net(theta_norm.unsqueeze(0)).squeeze(0)
        pred_iv = pred_norm * y_std + y_mean                 # output scaler -> real IV
        loss = (w * (pred_iv - target) ** 2).sum() / (w > 0).sum()
        loss.backward()
        opt.step()
    theta_final = _to_theta(u.detach(), lo, hi).numpy()
    return theta_final, float(loss.detach())


def calibrate(
    target_surface: np.ndarray,
    checkpoint_name: str = "surfacenet_ts.pt",
    x0: np.ndarray | None = None,
    steps: int = 500,
    lr: float = 5e-2,                    # higher lr is fine in u-space (well-conditioned)
    weights: np.ndarray | None = None,
    n_restarts: int = 4,
    seed: int = 0,
    pillar_range: tuple[float, float] = (0.005, 0.25),   # bounds for the pillar variance
    market_valid: np.ndarray | None = None,
) -> dict:
    net, ckpt = load_model(checkpoint_name)
    net = net.double()
    valid = np.asarray(ckpt["valid_nodes"])
    n_inputs = int(ckpt["n_inputs"])
    scalar_box = np.asarray(ckpt["param_box"], dtype=np.float64)   # (3, 2)
    n_scalars = scalar_box.shape[0]
    n_pillars = n_inputs - n_scalars

    # full parameter box: scalar bounds from the checkpoint, pillar bounds from pillar_range
    lo = np.concatenate([scalar_box[:, 0], np.full(n_pillars, pillar_range[0])])
    hi = np.concatenate([scalar_box[:, 1], np.full(n_pillars, pillar_range[1])])
    lo_t = torch.tensor(lo); hi_t = torch.tensor(hi)

    x_mean = torch.tensor(np.asarray(ckpt["x_mean"], dtype=np.float64))
    x_std = torch.tensor(np.asarray(ckpt["x_std"], dtype=np.float64))
    y_mean = torch.tensor(np.asarray(ckpt["y_mean"], dtype=np.float64))
    y_std = torch.tensor(np.asarray(ckpt["y_std"], dtype=np.float64))

    # weights: network node_weights, intersected with the market mask
    if weights is None:
        w_np = np.asarray(ckpt["node_weights"], dtype=np.float64)
    else:
        w_np = np.asarray(weights.reshape(-1), dtype=np.float64)
    if market_valid is not None:
        w_np = w_np * market_valid.reshape(-1).astype(np.float64)
    w = torch.tensor(w_np, dtype=torch.float64)

    # sanitise NaN target nodes (0 * NaN = NaN would poison the loss)
    tgt = target_surface.reshape(-1).astype(np.float64)
    tgt_safe = np.where(w_np > 0, tgt, 0.0)
    target = torch.tensor(tgt_safe, dtype=torch.float64)

    # starts in RAW theta-space, converted to u-space; box centre + random restarts
    theta_starts = [scalar_box.mean(axis=1).tolist() + [0.5 * (pillar_range[0] + pillar_range[1])] * n_pillars
                    if x0 is None else list(np.asarray(x0, dtype=np.float64))]
    if n_restarts > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_restarts):
            theta_starts.append(list(rng.uniform(lo, hi)))

    best = None
    for th0 in theta_starts:
        u0 = _to_u(np.asarray(th0, dtype=np.float64), lo, hi)
        theta, final_loss = _run_descent(net, target, w, x_mean, x_std, y_mean, y_std,
                                          u0, lo_t, hi_t, steps, lr)
        with torch.no_grad():
            tn = (torch.tensor(theta) - x_mean) / x_std
            fit = (net(tn.unsqueeze(0)).squeeze(0) * y_std + y_mean).numpy()
        score_mask = valid & (w_np > 0) & np.isfinite(tgt)
        rmse = float(np.sqrt(np.mean(((fit - tgt)[score_mask]) ** 2)))
        if best is None or rmse < best["surface_rmse"]:
            best = {"theta": theta, "surface_rmse": rmse, "final_loss": final_loss}

    theta = best["theta"]
    scalars, pillars = theta[:n_scalars], theta[n_scalars:]

    # box-edge report: how close each param sits to its bound (pinned = straining)
    frac = (theta - lo) / (hi - lo)
    pinned = [str(n) for n, f in zip(ckpt["param_names"], frac) if f < 0.02 or f > 0.98]

    return {
        "params": dict(zip([str(n) for n in ckpt["param_names"]], theta)),
        "theta": theta, "scalars": scalars, "pillars": pillars,
        "surface_rmse": best["surface_rmse"],
        "final_loss": best["final_loss"],
        "n_nodes": int((w_np > 0).sum()),
        "pinned": pinned,                       # params sitting at a box edge
    }