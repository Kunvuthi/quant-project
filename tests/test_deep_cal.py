import numpy as np
import pytest
import torch

torch.manual_seed(0)


class TestSurface:
    def test_shape_and_grid(self):
        from calibration.deep_cal.surface import (
            rbergomi_iv_surface, SURFACE_SHAPE, MATURITIES, LOG_MONEYNESS,
        )
        assert (MATURITIES.size, LOG_MONEYNESS.size) == SURFACE_SHAPE
        surf = rbergomi_iv_surface(0.12, 1.8, -0.7, 0.04, n_paths=20_000,
                                   rng=np.random.default_rng(0))
        assert surf.shape == SURFACE_SHAPE
        assert not np.isnan(surf).any()


class TestSurfaceNet:
    def test_shapes(self):
        from calibration.deep_cal.network import SurfaceNet, N_INPUTS, N_OUTPUTS
        net = SurfaceNet()
        y = net(torch.randn(16, N_INPUTS))
        assert y.shape == (16, N_OUTPUTS)


@pytest.mark.slow
class TestRecovery:
    """Synthetic recovery on a self-contained tiny run. Fills in once train.py
    and calibrate.py exist this afternoon."""
    def test_recovers_known_params(self):
        pytest.skip("pending train.py + calibrate.py")