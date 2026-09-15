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

ARTIFACTS = Path("artifacts")


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
    epochs: int = 400,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
) -> dict:
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
    loss_fn = nn.MSELoss()

    n_tr = Xtr.shape[0]
    best_val, best_state = np.inf, None
    history = {"train": [], "val": []}

    for ep in range(epochs):
        net.train()
        for b in range(0, n_tr, batch_size):
            bi = slice(b, b + batch_size)
            opt.zero_grad()
            loss = loss_fn(net(Xtr[bi]), Ytr[bi])
            loss.backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            tr_loss = loss_fn(net(Xtr), Ytr).item()
            va_loss = loss_fn(net(Xva), Yva).item()
        history["train"].append(tr_loss)
        history["val"].append(va_loss)

        if va_loss < best_val:                      # keep the best-val weights, not the last
            best_val = va_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if ep % 50 == 0 or ep == epochs - 1:
            print(f"epoch {ep:4d}  train {tr_loss:.3e}  val {va_loss:.3e}")

    net.load_state_dict(best_state)

    # --- validation in VOL POINTS, the number that actually means something ---
    net.eval()
    with torch.no_grad():
        pred_norm = net(Xva).numpy()
    pred_iv = ys.inverse(pred_norm)                 # back to real IVs
    true_iv = Y[val_idx]
    rmse_vol = float(np.sqrt(np.mean((pred_iv - true_iv) ** 2)))
    max_err = float(np.max(np.abs(pred_iv - true_iv)))
    print(f"\nheld-out surface RMSE: {rmse_vol:.5f} vol pts   max node err: {max_err:.5f}")

    ARTIFACTS.mkdir(exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "x_mean": xs.mean, "x_std": xs.std,       # scalers travel WITH the weights
        "y_mean": ys.mean, "y_std": ys.std,
        "n_inputs": N_INPUTS, "n_outputs": N_OUTPUTS,
        "param_names": meta["param_names"],
        "param_box": meta["param_box"],
        "rmse_vol": rmse_vol,
    }, ARTIFACTS / checkpoint_name)
    print(f"saved checkpoint to {ARTIFACTS / checkpoint_name}")

    return {"history": history, "rmse_vol": rmse_vol, "max_err": max_err}


if __name__ == "__main__":
    train()