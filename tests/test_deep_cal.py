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
    """Synthetic recovery: make a surface from KNOWN interior params, calibrate,
    assert the params come back. Self-contained (trains a small net inline) so it
    does not depend on the big offline dataset or checkpoint."""
    def test_recovers_known_params(self, tmp_path):
        import numpy as np
        from calibration.deep_cal.dataset import generate, save
        from calibration.deep_cal.train import train
        from calibration.deep_cal.surface import rbergomi_iv_surface
        from calibration.deep_cal.calibrate import calibrate

        # tiny self-contained pipeline (loose, just certifies the logic)
        data = generate(n_surfaces=300, seed=2, n_paths=20_000)
        save(data, name="recovery_test.npz")
        noise = np.std([rbergomi_iv_surface(0.11, 1.8, -0.7, 0.04, n_paths=20_000,
                        rng=np.random.default_rng(s)) for s in range(8)], axis=0)
        train(dataset_name="recovery_test.npz", checkpoint_name="recovery_test.pt",
              noise=noise, epochs=300, seed=2)

        true = np.array([0.11, 1.8, -0.7, 0.04])          # interior of the box
        surf = rbergomi_iv_surface(*true, n_paths=80_000,  # more paths: clean target
                                   rng=np.random.default_rng(99))
        res = calibrate(surf, checkpoint_name="recovery_test.pt", steps=800)

        rec = res["theta"]
        assert res["in_box"]
        # loose per-param tolerances: small net + 300 surfaces + MC noise
        assert abs(rec[0] - true[0]) < 0.03, f"H {rec[0]:.3f} vs {true[0]}"
        assert abs(rec[2] - true[2]) < 0.10, f"rho {rec[2]:.3f} vs {true[2]}"
        assert res["surface_rmse"] < 0.01, res["surface_rmse"]