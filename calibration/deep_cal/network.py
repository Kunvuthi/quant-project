# the MLP: architecture only, no training loop
"""The forward-map network: (H, eta, rho, xi0_pillars) -> flattened IV surface (88,).

Architecture only. No training loop, no I/O, no normalisation stats. Pure
nn.Module so it imports without dragging in torch training machinery and is
trivially unit-testable (shape in, shape out). Normalisation lives in train.py
and is saved with the weights, because calibrate.py must apply the identical
transform at inference.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from calibration.deep_cal.surface import N_MATURITIES, N_STRIKES, N_PILLARS

N_INPUTS = 3 + N_PILLARS                  # (H, eta, rho) + forward-variance pillars
N_OUTPUTS = N_MATURITIES * N_STRIKES      # 88, the flattened surface


class SurfaceNet(nn.Module):
    """MLP approximating the rBergomi forward map on the canonical grid.

    Grid-based (Horvath-Muguruza-Tomas): one network outputs the whole surface
    at once, so surface structure is learned jointly rather than pointwise.
    """
    def __init__(self, n_inputs: int = N_INPUTS, n_outputs: int = N_OUTPUTS,
                 width: int = 64, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = n_inputs
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.ELU()]
            d_in = width
        layers += [nn.Linear(d_in, n_outputs)]   # linear head: IVs are unbounded positive reals
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_inputs) normalised params. Returns (batch, n_outputs)
        normalised surface. Caller un-normalises with the saved output scaler."""
        return self.net(x)