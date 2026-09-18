# Quant Project - Phase 1: Options Pricing and Calibration

A math-first, self-directed build of a certified options-pricing and calibration
toolkit, from Black-Scholes through rough volatility, ending in a neural
deep-calibration capstone that fits the rough Bergomi model to live SPX option
surfaces in milliseconds.

Phase 1 is complete. This README documents what I built and, more importantly,
how I made myself trust it.

## What this project is really about

The list of models is not the point. Plenty of repositories implement Heston or a
COS pricer. What I set out to build is the discipline that tells me whether an
implementation is correct, because in numerical finance a wrong model rarely
crashes. It returns a plausible number that is quietly off.

Every component in this repository was validated before I used it. The habits
that run through the whole codebase:

- **Ground truth first.** Before a model prices anything, its building blocks are
  certified in isolation against a known closed form or identity. The rBergomi
  covariance kernel is checked against `t^(2H)` on the diagonal and against
  `min(s,t)` at the `H = 0.5` Brownian limit before a single path is simulated.
  Characteristic functions are checked for `phi(0) = 1` and the martingale
  condition `phi(-i) = S0 e^{(r-q)T}` before any Fourier pricing runs.
- **Synthetic recovery before real data.** Every calibrator is first pointed at a
  surface generated from known parameters, and required to recover those
  parameters, before it is ever shown a market quote. If it cannot invert its own
  forward map, it has no business touching SPX.
- **Convergence rates, not single points.** A binomial tree is not accepted
  because one price is close. It is accepted because the error falls at the
  `O(1/N)` rate the theory predicts. A single-point match can be a coincidence; a
  convergence slope cannot.
- **Tolerances derived from standard error.** Monte Carlo tests do not use magic
  constants. Tolerances come from the estimator's own standard error, so a test
  fails when the result is statistically inconsistent with truth, not when it
  drifts past an arbitrary threshold.
- **Forward-relative pricing with an untrusted spot.** The quoted index spot on a
  free feed is unreliable, so the parity-implied forward is the canonical
  reference throughout. Everything is priced in log-moneyness against that
  forward, which also makes surfaces spot-invariant.

The debugging stories below are the real signal. They are the bugs this
discipline caught that a single sanity check would have missed.

## The Phase 1 arc

Phase 1 follows one thread: each model fixes a specific thing the previous one
gets wrong about the volatility surface.

1. **Black-Scholes** gives a flat surface. Real markets show a skew and a term
   structure, so a constant-vol model is only a coordinate system (implied vol)
   for talking about prices, not a description of dynamics.
2. **Local volatility (Dupire)** reprices any static surface exactly, but its
   forward dynamics are wrong: the skew flattens as spot moves in a way markets do
   not.
3. **Heston** fixes the level of the problem by making variance stochastic and
   mean-reverting. It produces a genuine skew. Because it is Markovian, though, its
   skew term structure decays too fast, so it cannot match short-dated and
   long-dated skew at the same time.
4. **Jumps (Merton, Kou)** add the short-dated skew a pure diffusion cannot
   generate, because a diffusion needs time to build asymmetry while a jump is
   asymmetric instantly.
5. **Bates** combines stochastic volatility with jumps, getting both the
   long-dated smile and the short-dated skew, at the cost of eight parameters.
6. **Variance Gamma** takes the pure-jump route instead, a time-changed Brownian
   motion, as a contrast to the diffusion-plus-jump family.
7. **Rough volatility (rBergomi)** addresses the persistent term-structure
   failure directly. Making the variance driver rough (Hurst `H` near 0.1)
   produces a power-law ATM skew `tau^(H - 1/2)`, which is exactly the empirical
   `tau^(-0.4)` behaviour and is something no Markovian model can reproduce. The
   price is that rBergomi is non-Markovian: it has no characteristic function and
   no COS pricer, so it must be simulated.
8. **The capstone: deep calibration.** rBergomi is slow to calibrate because every
   surface needs a fresh Monte Carlo run. I trained a neural network to
   approximate the rBergomi parameter-to-surface map, validated it against the
   certified pricer, and then calibrated by gradient descent on the network's
   inputs. Calibration drops from a slow Monte Carlo loop to milliseconds, and the
   final fit is against real SPX quotes.

Each step is one weekly notebook in `notebooks/`, in order, `01` through `18`.

## Headline results

All figures below are reproduced from the notebooks, not quoted from memory.

**Rough-volatility recovery (notebook 17).** From option smiles simulated with a
true `H = 0.1`, a log-log fit of ATM skew against maturity recovers `H = 0.109`.
The fitted skew slope is `-0.391` against the theoretical `-0.400`. The roughness
signature is measurable and it matches.

**Five-model benchmark (notebook 16).** Five models fit to the same 82-day SPX
slice (52 OTM strikes), RMSE in vol points:

| model  | params | RMSE (vol pts) |
|--------|:------:|:--------------:|
| VG     | 3      | 1.816          |
| Merton | 4      | 1.348          |
| Heston | 5      | 0.948          |
| Kou    | 5      | 0.751          |
| Bates  | 8      | 0.337          |

RMSE falls almost monotonically with parameter count, which is mostly the "more
parameters fit better" law and not one model being truer. The real finding is
underneath the table. The two lowest-RMSE fits, Bates and Kou, both bought their
low RMSE by pinning parameters to their box edges (Bates pinned vol-of-vol and
jump intensity; Kou collapsed to a one-sided down-jump-only degeneracy). The only
fits with fully interior, interpretable parameters were the simpler models. So on
this slice the RMSE ranking and the trustworthiness ranking are nearly inverted:
Heston, all-interior and mid-table on RMSE, is the fit I would actually believe. A
lower RMSE is not an improvement when it is bought with pinned parameters.

**Real SPX deep calibration (notebook 18).** The trained network calibrated to a
live SPX surface returns `H = 0.107`, `rho = -0.896`, `eta = 2.46`, with no
parameter pinned to a box edge, at a surface RMSE of about `0.0067` in absolute
implied-vol units (roughly 0.67 vol points) over the valid nodes. Interior
maturities fit tightly. The honest limitation is the short-dated deep-put wing: a
single-`H` rBergomi undershoots the steepest short-maturity skew, which is a known
structural property of the one-factor model rather than a fitting failure.

## Debugging stories worth telling

These are the catches that show the validation discipline earning its keep.

- **Isolating Monte Carlo noise from signal.** Before trusting per-node surface
  errors in the deep-calibration pipeline, I measured the pure Monte Carlo noise at
  each grid node directly, by pricing the same fixed parameters under many seeds
  and taking the spread. That measured noise floor is what drove node dropping and
  inverse-noise weighting in the training loss, instead of guessing which nodes
  were reliable.
- **Zero times NaN does not vanish.** A zero weight on a bad grid node does not
  neutralise a NaN target, because `0 * NaN = NaN` poisons the whole loss. Both
  masking (drop the node) and sanitising (replace the NaN) are required. Either one
  alone silently breaks training.
- **The box escape.** Optimising the network's inputs without constraints let the
  calibrator walk parameters out of the trained domain, including correlation below
  `-1`, and still report a low RMSE by extrapolating the network into nonsense. The
  fix is a sigmoid reparameterisation: the optimiser ranges over all of R while the
  actual parameters can never leave the trained box, so the network is never queried
  out of domain.
- **Earlier Phase 1 catches.** A COS normalisation bug (the first cosine
  coefficient carries a half-weight that is easy to drop). A put sign error that
  passed at the money and only showed up in the wings. The Heston validation trap of
  comparing a single-rate reference against a dividend-aware pricer, where the models
  agree in form but disagree by exactly the carry term.

## Architecture

```
models/        stochastic models: closed forms, characteristic functions, simulators
pricing/       pricing engines: Fourier (COS, Carr-Madan), Monte Carlo, PDE
calibration/   surface calibrators, implied vol, local vol, and the deep-cal capstone
data/          option-chain ingestion and the parity-implied forward
notebooks/     the weekly arc, 01 through 18, in order
tests/         pytest suite with standard-error-derived tolerances
```

- `models/` holds each model's mathematical core. Fourier-tractable models
  (Heston, Merton, Kou, Bates, VG) expose a characteristic function and cumulants;
  simulation models (Monte Carlo GBM, rBergomi) expose path simulators. rBergomi is
  deliberately simulator-only, because its non-Markovian Volterra kernel admits no
  characteristic function.
- `pricing/` holds payoff-agnostic engines that consume the models. The COS pricer
  (Fang and Oosterlee) and the Carr-Madan FFT pricer both take a characteristic
  function; the Monte Carlo engine keeps its estimator honest by refusing anything
  but one-dimensional iid payoffs, so the standard error cannot be understated by
  hidden correlation; the PDE engine solves the local-vol equation on a log-space
  grid.
- `calibration/` holds the fitters. Classical calibrators (SVI, SABR, Dupire,
  Merton/Kou jumps, Heston/Bates/VG via COS) fit in implied-vol space against the
  parity forward. `calibration/deep_cal/` is the capstone pipeline, described below.
- `data/` ingests a CBOE delayed-quote chain, parses OCC symbols, and computes the
  parity-implied forward that the rest of the stack treats as canonical.

### The deep-calibration pipeline

`calibration/deep_cal/` is organised so that training targets and real-data inputs
route through exactly one surface convention, which removes any drift between what
the network learns and what it is later asked to calibrate.

- `surface.py` - parameters to an implied-vol surface on the canonical `8 x 11`
  grid, via the certified rBergomi Monte Carlo pricer. Both the dataset builder and
  the real-data path call this one function.
- `dataset.py` - samples the trained parameter box, prices each sample through
  `surface.py`, and saves parameters, surfaces, and the box itself to `artifacts/`.
- `network.py` - `SurfaceNet`, a small MLP that outputs the whole surface at once
  (the grid-based approach of Horvath, Muguruza and Tomas), so surface structure is
  learned jointly.
- `train.py` - trains the network and checkpoints the normalisation scalers
  alongside the weights, because inference must apply the identical transform.
- `calibrate.py` - freezes the trained network and optimises its inputs, not its
  weights, to match a target surface. The sigmoid reparameterisation keeps every
  parameter inside the trained box.
- `spx_data.py` - resamples a real SPX chain onto the canonical grid, flagging any
  node that required extrapolation beyond the market's quoted range.

See [docs/architecture.md](docs/architecture.md) for the full data flow.

## Setup

Requires miniforge or mamba.

```bash
mamba env create -f environment.yml
conda activate quant
pip install -e .
```

Run the test suite (fast tests only; the `slow` marker covers simulation,
training, and dataset generation):

```bash
pytest -m "not slow"
```

A step-by-step guide to reproducing the capstone, from dataset generation through
real-data calibration, is in [docs/how-to-run.md](docs/how-to-run.md).

## References

- Bayer, Friz and Gatheral (2016), *Pricing under rough volatility*, Quantitative
  Finance 16(6), 887–904. The rBergomi model.
- Gatheral, Jaisson and Rosenbaum (2018), *Volatility is rough*, Quantitative
  Finance 18(6), 933–949. The empirical case for `H` near 0.1.
- Horvath, Muguruza and Tomas (2021), *Deep learning volatility*, Quantitative
  Finance 21(1), 11–27. The grid-based deep-calibration approach used in the
  capstone.
- Bennedsen, Lunde and Pakkanen (2017), *Hybrid scheme for Brownian
  semistationary processes*, Finance and Stochastics 21(4), 931–965. The fast
  simulation scheme this project cites as the production alternative to the exact
  Cholesky simulator, and does not yet implement.

## Roadmap

Phase 1 is wrapped: this documentation pass, the write-up, and making the
repository public.

Deferred technical items, none of them blocking:

- Fully split-out per-model calibrators (partly done).
- A Heston-Kou hybrid.
- A proper multi-maturity benchmark on a richer data feed.
- A business-day-aware staleness filter on the option chain.
- The Bennedsen-Lunde-Pakkanen hybrid simulation scheme, to replace the exact
  Cholesky rBergomi simulator for speed.
- Dupire refinements on real data.

Possible rough-volatility extension: the short-wing undershoot in the SPX fit
motivates a multi-factor or rough-Heston model, if term-structure-of-skew fit
becomes the priority.
