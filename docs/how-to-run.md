# How to run

This guide covers environment setup, the test suite, and reproducing the
deep-calibration capstone end to end. For what the pieces are, see
[architecture.md](architecture.md).

## Environment

Requires miniforge or mamba.

```bash
mamba env create -f environment.yml
conda activate quant
pip install -e .
```

The editable install matters: the code imports by absolute package path
(`from models.rbergomi import ...`), so the project root must be importable.

## Tests

The suite is the fastest way to confirm a working install. Fast tests only:

```bash
pytest -m "not slow"
```

The full suite, including the slow simulation, training, and synthetic-recovery
tests:

```bash
pytest
```

The `slow` marker covers anything that simulates paths, trains a network, or
generates a dataset. The synthetic-recovery test in `tests/test_deep_cal.py` is
worth reading on its own: it builds a surface from known interior parameters,
trains a small network inline, calibrates, and asserts the parameters come back.
It is the whole validation philosophy in one test.

## The notebooks

`notebooks/01` through `notebooks/18` are the canonical, ordered walkthrough. Each
one is self-contained and shows the validation for its model before using it.
Notebooks `16`, `17`, and `18` produce the headline results:

- `16_cross_model_benchmark.ipynb` — the five-model benchmark and the
  pinned-parameter finding.
- `17_rbergomi_model.ipynb` — the rBergomi build and the `H = 0.109` roughness
  recovery.
- `18_rbergomi_deep_calibration.ipynb` — the capstone: dataset, training,
  synthetic recovery, and the real SPX fit.

Notebook 18 orchestrates the full pipeline and is the recommended way to reproduce
the capstone, because it keeps the dataset, checkpoint, and calibration names
consistent across the steps.

## Reproducing the capstone from the modules

The pipeline can also be driven directly from the `calibration/deep_cal/` modules.
The one thing to keep straight is that the dataset name, the checkpoint name, and
the calibration checkpoint name must match across the three steps. The module
defaults use two different naming schemes (a `flat` set and a `termstructure` /
`_ts` set), so pass names explicitly rather than relying on the defaults.

Everything writes to and reads from `artifacts/`, which is not tracked.

**1. Generate the training set.** Sample the parameter box, price each sample
through the certified rBergomi surface pricer, and save.

```bash
python -m calibration.deep_cal.dataset
```

This runs the `__main__` block (4000 surfaces, 30k paths each) and writes
`artifacts/rbergomi_termstructure_dataset.npz`. Dataset generation is the slow
step, since every surface is a fresh Monte Carlo run.

**2. Measure the per-node Monte Carlo noise.** Price one fixed parameter set under
many seeds and take the node-wise spread. This noise floor drives the inverse-noise
weighting and node dropping in training. The notebook shows the exact call; the
result is saved as `artifacts/node_noise_ts.npy`.

**3. Train the network.** Fit `SurfaceNet` on the dataset, passing the measured
noise as the weighting. Scalers are checkpointed with the weights.

```python
from calibration.deep_cal.train import train
train(dataset_name="rbergomi_termstructure_dataset.npz",
      checkpoint_name="surfacenet_ts.pt")
```

**4. Calibrate.** Load a target surface and optimise the frozen network's inputs.
For a synthetic check, build the target from known parameters with
`rbergomi_iv_surface` and confirm recovery. For the real fit, build the target from
an SPX chain with `calibration.deep_cal.spx_data.load_spx_surface`.

```python
from calibration.deep_cal.calibrate import calibrate
result = calibrate(target_surface, checkpoint_name="surfacenet_ts.pt",
                   market_valid=valid_mask)
```

`result` reports the recovered parameters, which nodes were valid, whether any
parameter pinned to a box edge (it should not), and the surface RMSE.

## Fetching a fresh SPX chain

`data/option_chain.py` pulls a CBOE delayed-quote chain. The quoted spot on this
free feed is unreliable for index options, so the code computes and uses the
parity-implied forward instead. Any saved chains live under `data/raw/`, which is
not tracked; regenerate rather than commit them.
