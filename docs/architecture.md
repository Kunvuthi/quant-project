# Architecture

This document describes how the pieces fit together and why the boundaries fall
where they do. For the project narrative and results, see the
[README](../README.md).

## Layering

The stack is four layers, each depending only on the ones below it.

```
data/         raw quotes -> clean smiles, parity-implied forward
   |
models/       mathematical cores: char funcs, cumulants, path simulators
   |
pricing/      payoff-agnostic engines that consume a model
   |
calibration/  fit model parameters to a market or synthetic surface
```

Nothing in `models/` imports from `pricing/` or `calibration/`. A model knows its
own mathematics and nothing about how it will be priced or fitted. This is what
lets the same characteristic function feed both the COS pricer and the Carr-Madan
FFT pricer without either knowing about the other.

## The forward-relative convention

One convention runs through the entire stack: the parity-implied forward is the
canonical reference, and the quoted spot is treated as an untrusted diagnostic.

The free CBOE feed does not carry a reliable index spot for SPX, so
`data/option_chain.py` computes the forward from put-call parity per maturity and
everything downstream prices in log-moneyness `k = log(K / F)` against that
forward. Two consequences:

- Implied vols are inverted with cost-of-carry `b = 0` (pricing at the forward),
  so the untrusted spot never enters the calibration objective.
- Surfaces become spot-invariant. The deep-calibration grid is defined in a
  normalised world (`S0 = 1`, `r = 0`, forward `= 1`), and real SPX quotes drop
  onto that same grid once converted to log-moneyness against the parity forward.

## Models

Two families, split by whether the model is Fourier-tractable.

**Affine / Fourier-tractable** (`bsm`, `heston`, `merton`, `kou`, `bates`, `vg`):
each exposes a characteristic function of the log price and its first two
cumulants. The cumulants set the COS truncation range; the characteristic function
is the pricer's only input. These models are certified by two analytic identities
before any pricing: `phi(0) = 1` (normalisation) and `phi(-i) = S0 e^{(r-q)T}`
(the martingale condition).

**Simulation-only** (`montecarlo`, `rbergomi`, plus `exotics`, `mc_greeks`):
these expose path simulators, not characteristic functions. `rbergomi` is the
deliberate outlier. Its variance is driven by a Volterra process whose kernel
integrates the entire path history, so it is non-Markovian, has no affine
transform, and admits no COS or Carr-Madan pricing. It is certified instead
through its Gaussian covariance structure: the Volterra auto-covariance against
`t^(2H)` on the diagonal and `min(s,t)` at `H = 0.5`, and the price-vol cross
covariance against the same Brownian limit. The simulator itself is an exact
Cholesky factorisation of the joint covariance matrix.

## Pricing engines

- `pricing/fourier.py` — the COS method (Fang and Oosterlee). Payoff coefficients
  (`cos_call_coefficients`) are separated from the density expansion, so the same
  cosine machinery prices any model that supplies a characteristic function.
- `pricing/carr_madan.py` — the Carr-Madan damped FFT pricer, used as an
  independent cross-check on the COS Heston prices. Two pricers built from the same
  characteristic function via different transforms should agree, and where they do
  the characteristic function is almost certainly right.
- `pricing/monte_carlo.py` — the estimator (`mc_price`) accepts only
  one-dimensional iid payoffs and raises otherwise. This is a deliberate
  constraint: the standard error `sd / sqrt(n)` is only valid on independent draws,
  so antithetic pairing and any other variance reduction is resolved by the caller
  before the payoffs reach the estimator. The estimator cannot be tricked into
  understating its own error.
- `pricing/pde.py` — a log-space finite-difference solver for the local-vol PDE,
  with the grid width set by a number of standard deviations of log-return at the
  maximum volatility.

## Calibration

Classical calibrators all fit in implied-vol space against the parity forward:

- `implied_vol.py` — Brent inversion, returning NaN on no-arbitrage violations
  rather than a bad root.
- `svi.py` — raw SVI in total variance, fitted with the Zeliade two-stage
  reduction (the inner problem is linear for fixed `(m, sigma)`).
- `sabr.py` — Hagan implied vol, with a series branch near the money to avoid
  catastrophic cancellation, fitted by direct nonlinear least squares with `beta`
  fixed.
- `dupire.py` — the local-vol surface from a call-price grid, with a density floor
  guarding the denominator.
- `jumps.py`, `fourier_models.py` — Merton, Kou, Heston, Bates, and VG, each priced
  through the COS engine and inverted to implied vol, sharing one residual-and-
  least-squares core.

## The deep-calibration pipeline

`calibration/deep_cal/` is the capstone. Its central design decision is that the
training targets and the real-data inputs pass through one and only one surface
function, so the network is always calibrated against exactly the convention it
was trained on.

```
             surface.py  (params -> IV surface, certified MC pricer)
              /       \
     dataset.py        spx_data.py
     (sample box,      (real SPX chain ->
      price, save)      canonical grid + valid mask)
          |                     |
       train.py                 |
   (fit SurfaceNet,             |
    save scalers with           |
    the weights)                |
          |                     |
        network.py              |
     (SurfaceNet MLP) --------- calibrate.py
                          (freeze net, optimise INPUTS
                           in a sigmoid-bounded box)
```

Design points that matter:

- **One surface convention.** `surface.py` is the single source of the `8 x 11`
  grid and the pricing convention. Both the synthetic and real paths import the grid
  from here, so there is no possibility of a train/inference mismatch.
- **Scalers travel with the weights.** The network maps normalised parameters to a
  normalised surface. `train.py` saves the scaler statistics inside the same
  checkpoint as the weights, so `calibrate.py` cannot apply a different
  normalisation at inference.
- **Calibration optimises inputs, not weights.** The trained network is a fast,
  differentiable forward map. Calibration freezes it and runs gradient descent on
  the four physical parameters. The sigmoid reparameterisation
  (`theta = lo + (hi - lo) * sigmoid(u)`) lets the optimiser range over all of R
  while the parameters stay strictly inside the trained box, so the network is never
  evaluated out of domain.
- **A validity mask, not silent extrapolation.** `spx_data.py` marks every grid
  node that required extrapolation beyond the market's quoted strike range, and the
  calibration objective weights those nodes out. The short-dated deep-put wing is
  reported as a limitation rather than fitted with invented data.

## Artifacts and data

Neither `artifacts/` (datasets, checkpoints, measured noise) nor `data/raw/`
(licensed and downloaded market data) is tracked. Both are reproducible: the
datasets from `dataset.py`, the checkpoints from `train.py`, the noise from the
per-node measurement described in the README. See
[how-to-run.md](how-to-run.md) for the regeneration steps.
