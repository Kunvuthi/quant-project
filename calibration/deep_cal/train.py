"""Train SurfaceNet on the generated dataset, checkpoint weights WITH scalers.

The scaler statistics are part of the model: the net maps normalised params to
normalised surfaces, so calibrate.py must apply the identical transform. They are
saved in the same checkpoint as the weights, never separately.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from calibration.deep_cal.network import SurfaceNet, N_INPUTS, N_OUTPUTS

REPO_ROOT = Path(__file__).resolve().parents[2]   # deep_cal/ -> calibration/ -> repo root
ARTIFACTS = REPO_ROOT / "artifacts"


class Scaler:
    """Standardise to zero mean, unit variance. Fit on TRAIN only; the same
    stats transform val, test, and tomorrow's calibration inputs."""
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = np.where(std < 1e-12, 1.0, std)   # guard constant columns

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        return cls(x.mean(axis=0), x.std(axis=0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def load_dataset(name: str = "rbergomi_flat_dataset.npz"):
    d = np.load(ARTIFACTS / name, allow_pickle=True)
    X = d["params"].astype(np.float64)                     # (n, 4)
    Y = d["surfaces"].reshape(d["surfaces"].shape[0], -1)  # (n, 88) flattened
    return X, Y, d


def train(
    dataset_name: str = "rbergomi_flat_dataset.npz",
    checkpoint_name: str = "surfacenet.pt",
    val_frac: float = 0.15,
    epochs: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
    noise: np.ndarray | None = None,   # (8,11) per-node MC noise for weighting; required
    noise_max: float = 0.03,           # nodes noisier than this are dropped
    weight_clip: float = 10.0,         # cap any node weight at this multiple of the median
) -> dict:
    if noise is None:
        raise ValueError("pass the measured per-node noise grid for inverse-variance weighting")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X, Y, meta = load_dataset(dataset_name)
    n = X.shape[0]

    # split BEFORE fitting scalers: scalers see train only, no val leakage
    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    xs = Scaler.fit(X[tr_idx])
    ys = Scaler.fit(Y[tr_idx])

    Xtr = torch.tensor(xs.transform(X[tr_idx]), dtype=torch.float32)
    Ytr = torch.tensor(ys.transform(Y[tr_idx]), dtype=torch.float32)
    Xva = torch.tensor(xs.transform(X[val_idx]), dtype=torch.float32)
    Yva = torch.tensor(ys.transform(Y[val_idx]), dtype=torch.float32)

    net = SurfaceNet(n_inputs=N_INPUTS, n_outputs=N_OUTPUTS)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # --- node validity mask + weights from measured MC noise ---
    noise_flat = noise.reshape(-1)
    noise_std = noise_flat / ys.std                        # noise in standardised target units
    valid = (noise_flat > 1e-4) & (noise_flat < noise_max)    # drop dead (==0) and too-noisy nodes
    valid_count = int(valid.sum())
    print(f"kept {valid_count}/{noise_flat.size} nodes")

    w_np = np.zeros_like(noise_flat)
    w_np[valid] = 1.0 / noise_std[valid]                   # inverse-std in standardised space
    med = np.median(w_np[valid])                           # cap the range: no node dominates
    w_np[valid] = np.clip(w_np[valid], 0.0, weight_clip * med)
    w_np = w_np / w_np[valid].mean()                       # kept-node weights average ~1
    print(f"weight min/mean/max: {w_np[valid].min():.2f} / "
          f"{w_np[valid].mean():.2f} / {w_np[valid].max():.2f}")
    w = torch.tensor(w_np, dtype=torch.float32)

    def wmse(pred, targ):                                  # weighted MSE over valid nodes only
        return (w * (pred - targ) ** 2).sum() / valid_count

    n_tr = Xtr.shape[0]
    best_val, best_state, patience, since_improved = np.inf, None, 150, 0
    history = {"train": [], "val": []}

    for ep in range(epochs):
        net.train()
        for b in range(0, n_tr, batch_size):
            bi = slice(b, b + batch_size)
            opt.zero_grad()
            loss = wmse(net(Xtr[bi]), Ytr[bi])
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            tr_loss = wmse(net(Xtr), Ytr).item()
            va_loss = wmse(net(Xva), Yva).item()
        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        if va_loss < best_val - 1e-6:
            best_val, best_state, since_improved = va_loss, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            since_improved += 1
            if since_improved >= patience:
                print(f"early stop at epoch {ep}, best val {best_val:.3e}")
                break

        if ep % 50 == 0 or ep == epochs - 1:
            print(f"epoch {ep:4d}  train {tr_loss:.3e}  val {va_loss:.3e}")

    net.load_state_dict(best_state)

    # --- validation in VOL POINTS, over VALID nodes only ---
    net.eval()
    with torch.no_grad():
        pred_norm = net(Xva).numpy()
    pred_iv = ys.inverse(pred_norm)                        # (n_val, 88) real IVs
    true_iv = Y[val_idx]
    diff = (pred_iv - true_iv)[:, valid]                   # surviving nodes only
    rmse_vol = float(np.sqrt(np.mean(diff ** 2)))
    max_err = float(np.max(np.abs(diff)))
    print(f"\nheld-out surface RMSE: {rmse_vol:.5f} vol pts   max node err: {max_err:.5f}  (valid nodes)")

    ARTIFACTS.mkdir(exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "x_mean": xs.mean, "x_std": xs.std,
        "y_mean": ys.mean, "y_std": ys.std,
        "n_inputs": N_INPUTS, "n_outputs": N_OUTPUTS,
        "param_names": meta["param_names"],
        "param_box": meta["param_box"],
        "valid_nodes": valid,
        "node_weights": w_np,
        "noise_max": noise_max,
        "rmse_vol": rmse_vol,
    }, ARTIFACTS / checkpoint_name)
    print(f"saved checkpoint to {ARTIFACTS / checkpoint_name}")

    return {"history": history, "rmse_vol": rmse_vol, "max_err": max_err,
            "valid_nodes": valid, "pred_iv": pred_iv, "true_iv": true_iv}

if __name__ == "__main__":
    train()