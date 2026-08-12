# Weekly Log

A running record of work completed each day as documentation.

## Week 1 (Jun 10–17): BSM Implementation

### Day 1 - Thu Jun 11
- Created feature branch `feature/week-01-bsm`
- Implemented vectorised Black-Scholes-Merton pricer in `models/bsm.py`
  - Accepts scalar or NumPy array inputs; broadcasts naturally over strikes × maturities
  - Type-hinted with `ArrayLike` and `Literal['call', 'put']`
  - Calls and puts share `d1`, `d2` computation; conditional return avoids redundant work
- Verified against three benchmarks:
  - Single ATM 1Y call (σ=20%, r=5%) returns canonical value 10.4506
  - Vector of 5 strikes returns monotonically decreasing prices as expected
  - 4×3 grid via broadcasting shows correct moneyness × maturity structure
- Put-call parity test on the 4×3 grid: max residual 1.42e-14 (machine precision)
- Implemented closed-form Greeks (Δ, Γ, 𝒱, Θ, ρ) in `bsm_greeks`
  - Raw values (no per-day or per-vol-point scaling)
  - Call and put theta and rho differ only in the rate-effect term
- Verified Greeks via BSM PDE identity: residual = -2.22e-16 (machine epsilon)
- Implemented finite-difference Greeks in `bsm_greeks_fd` using central differences
  - All five FD Greeks agree with analytic to better than 1e-6
  - Built confidence in bump-and-revalue methodology for use later with Heston

### Day 2 - Fri Jun 12
- CRR binomial tree (recombining, O(N) memory)
- Empirically verified O(1/N) convergence to BSM (slope = -1 on log-log)
- Observed strike-discretisation oscillation at OTM strikes
- Richardson extrapolation gives O(1/N²) on smooth (ATM) case, ~1000× error reduction
- CRR put-call parity holds to machine precision regardless of N
  (arbitrage-free model is internally consistent even when externally inaccurate)

### Day 3 - Mon Jun 15
- Vectorised MC pricer in `models/montecarlo.py`
  - Direct sampling from closed-form GBM solution (no time-stepping bias)
  - Returns (price, standard_error) tuple, never a bare point estimate
  - Reproducible via optional seed
- Verified O(1/√N) convergence over 4 orders of magnitude (N=100 to 1M)
- Implemented antithetic variates with correct pair-averaging
  - **Two pedagogical bugs caught**:
    1. First attempt averaged unrelated payoffs (wrong concatenation order) → no
       variance reduction at all
    2. Second attempt collapsed wrong reshape axis → spurious 590,000× "reduction",
       caught by z-score check showing 273 SEs from truth
  - Final implementation: SE ratio converges to 1/√2 ≈ 0.707 over 5 orders of N
- MC put-call parity behaves differently from CRR's:
  - CRR parity holds exactly (deterministic quadrature)
  - MC parity holds only in expectation; finite-sample residual has stddev
    σ·S·e^(rT)·√T/√N ≈ 0.21 at N=10⁴ - consistent with observed -0.18
- Lesson: report SE always; cross-check estimate against truth via z-score

### Day 4 - Tue Jun 17
- Implied volatility inverter in `models/implied_vol.py`
  - Brent's method via `scipy.optimize.brentq`; round-trip accuracy ~1e-13
  - Arbitrage-bound rejection returns `np.nan` (not None - for vectorisation)
  - Tested on ATM call, OTM call, OTM put - all round-trip cleanly
  - `compute_smile` - applies inverter row-by-row via `df.apply(axis=1)` 
- First real options data via yfinance (rate-limited mid-session; cached defensively)
  - SPX 30 DTE chain, 46 calls × 14 cols
  - **yfinance quirk: openInterest column is always 0 for SPX index options.**
    Filter relaxed to ignore OI; documented in clean_chain defaults.
- Data hygiene module `models/spx_chain.py`:
  - `fetch_chain_cached` - file-backed cache in `data/raw/`, immune to rate limits
  - `clean_chain` - filters on bid > 0, spread, volume, staleness; 46 → 21 rows
- **First SPX volatility smile produced.** ATM ~13.3%, left wing steepens
  faster than right (equity skew); minimum at k ≈ 0.05 (5% OTM).
- Visualised in log-moneyness coordinates - natural for SPX vol literature.
- This shape is the empirical motivation for everything in weeks 3-9.

### Day 5 - Wed Jun 17
- **Renamed** `models/spx_chain.py` → `models/option_chain.py` for multi-source generality (`git mv` preserves history)
- **CBOE data source** added as alternative to yfinance (which rate-limited again mid-session)
  - `fetch_chain_cboe(ticker)` pulls full chain (~32k contracts in one call) from CBOE's free delayed-quotes JSON endpoint
  - Returns `(DataFrame, spot)` - spot extracted from response for single source of truth at fetch time
  - Provides real `open_interest`, BSM-computed Greeks, theoretical prices - far richer than yfinance
  - CBOE's own `iv30 = 13.6%` independently validates yesterday's ATM IV computation (~13.3%)
- **OCC symbol parser** `_parse_occ_symbol`: decodes `SPXW260618C00200000` → `{root, expiry, option_type, strike}`
  - Strike encoding: last 8 chars = strike × 1000, integer (avoids floats for exact representation)
  - **Root extraction added** after discovering CBOE returns both SPX (AM-settled monthly) and SPXW (PM-settled weekly) for many expiries - caused duplicate strikes at first
  - Filter to `root == 'SPXW'` downstream; SPXW carries most current liquidity
- `find_closest_expiry()` helper - date-agnostic, uses `pd.Timestamp.now(tz='UTC').normalize()` rather than hardcoded today
- `filter_by_expiry()` helper splits the full chain into `(calls, puts)` for one expiry, returning yfinance-shaped DataFrames so `clean_chain` and `compute_smile` work unchanged
- **CBOE-tuned `clean_chain` defaults**: `min_open_interest=10`, `max_relative_spread=0.30`, `max_staleness_days=2` (tighter than yfinance defaults because CBOE data is reliably cleaner)
- **Dual-expiry term structure produced**:
  - 30 DTE (expiry 2026-07-17): 83 usable strikes after cleaning; smile range 11.3%–17.9%, ATM IV 14.7%
  - 75 DTE (expiry 2026-08-31): 20 usable strikes; smile range 12.2%–15.9%, ATM IV 14.7%
  - Initial 60 DTE attempt (2026-08-14) gave only 9 strikes - too sparse; pushed out to next high-liquidity expiry
- **Empirical findings**:
  - ATM IV roughly flat across maturities (calm regime - no immediate vol premium)
  - Per-unit-strike skew ~70% steeper at 30 DTE than 75 DTE - textbook term-structure flattening, empirically confirmed
  - Far-OTM wing data is noisy (e.g., 75 DTE K=9000 IV=14.8% - single $0.93 mid price, low liquidity)
- Week 1 summary markdown cell written at top of `03_first_spx_smile.ipynb`
- Branch `feature/week-01-bsm` merged to `main` via PR; tagged `v0.1-week1`

## Week 2 (Jun 18-24): Greeks Deep Dive + GARCH Intro

### Day 6 - Thu Jun 18
- New module `models/mc_greeks.py` for Monte Carlo sensitivity *estimators*
  (kept separate from `montecarlo.py`, which owns pricing - LR estimators and
  control-variate variants will land here next); tested in `04_mc_greeks.ipynb`
- **Pathwise (PW) Greek estimators** for European call in `bsm_greeks_pw`
  - Returns `{'delta': (est, se), 'vega': (est, se)}` - never a bare estimate
  - Antithetic, `n_paths` = total paths convention (matches `bsm_price_mc`);
    $n_{\text{pairs}} = n_{\text{paths}} // 2$, explicit slice pairing
    `(arr[:n] + arr[n:]) / 2` rather than reshape - pairing made visually
    unmissable after Week 1's 590k bug
  - Delta integrand: $e^{-rT}\,\mathbf 1_{\{S_T>K\}}\,S_T / S_0$
  - Vega integrand:  $e^{-rT}\,\mathbf 1_{\{S_T>K\}}\,S_T\,(\sqrt{T}\,Z - \sigma T)$
- **Interchange justified before coding** (the whiteboard bit):
  - PW needs the *payoff* $f(S_T(\theta, Z))$ Lipschitz in $\theta$ for fixed $Z$
  - Call payoff $(S_T - K)^+$ is continuous (kink at $K$, hit with prob 0; no jump)
    $\Rightarrow$ Lipschitz $\Rightarrow$ dominated difference quotient
    $\Rightarrow$ interchange valid
  - Load-bearing condition is Lipschitz domination, NOT the measure-zero kink
- **Analytic pre-validation**: showed $\Delta_{\text{PW}}$ has mean exactly
  $\Phi(d_1)$ by the martingale identity
  $\mathbb{E}[\mathbf 1_{\{S_T>K\}}\,S_T] = S_0 e^{rT}\Phi(d_1)$, so the discount
  cancels to leave $\Phi(d_1)$. MC is an unbiased estimator by construction, not
  an approximation that happens to land close.
- **Validation (z-score discipline, 1M paths, seed 42)**:
  - Delta: PW = 0.63692 +/- 0.00020 vs analytic 0.63683, $z = 0.43$ ok
  - Vega:  PW = 37.56368 +/- 0.06616 vs analytic 37.52403, $z = 0.60$ ok
- **Empirical antithetic asymmetry observed**: relative SE for Delta ~0.03%,
  Vega ~0.18% (6x noisier) at equal path count. Structural: Delta integrand is
  near-symmetric in $Z$ so antithetic cancels hard; Vega's
  $(\sqrt{T}\,Z - \sigma T)$ weight flips sign with $Z$ and cancels far less.
  Not a bug - inherent to the integrand symmetry.
- **Key takeaway**: PW is the unbiased Greek method that survives the jump to
  models with no closed form (Heston W4, rBergomi W9), where there is no
  $\Phi(d_1)$ to differentiate. Bump-and-revalue survives too but pays in bias
  and variance.
- One-line PW validity test carried to Day 7: does the payoff itself *jump* as
  you move the parameter? Kinks survivable, jumps fatal. Call = kink,
  digital = jump.

### Day 7 - Fri Jun 19
- **Likelihood-ratio (LR) Greek estimators** added to `models/mc_greeks.py`;
  derived, tested, and documented in `04_mc_greeks.ipynb` (full formula reference
  and lessons cells live there)
  - `bsm_greeks_lr` takes a passed-in `payoff: Callable` rather than an
    `option_type` switch - LR is payoff-agnostic by construction
  - Delta score $\dfrac{Z}{S_0\sigma\sqrt{T}}$, vega score
    $\dfrac{Z^2-1}{\sigma} - Z\sqrt{T}$ (trap: $\sigma$ sits in both $m$ and $s$)
- **Validated (z-score, 1M paths, seed 42)**: LR delta $z = 0.84$, LR vega
  $z = 1.09$ vs Week 1 analytics. Both unbiased.
- **PW vs LR**: LR noisier at equal $N$ (delta ~6.5x SE, vega ~4.2x). Antithetic
  helps integrands odd in $Z$ (LR delta) but not even ones (LR vega's $Z^2$ term).
  All estimators $O(N^{-1/2})$ - same rate, different constant.
- **Digital demonstration (the point of the week)**: LR nails
  $\Delta = e^{-rT}\varphi(d_2)/(S_0\sigma\sqrt{T}) = 0.018762$ ($z = 0.46$); PW
  returns **exactly** $0 \pm 0$ because the jump payoff has $f' = 0$ a.s. Cannot
  reuse `bsm_greeks_pw` - the pathwise derivative genuinely does not exist for a jump.
- **Key lesson (detail in notebook)**: a tiny SE is a red flag, not a green light.
  SE measures spread, not correctness - the PW digital's $0 \pm 0$ would look like
  perfect convergence in a model with no analytic truth. Verify payoff is Lipschitz
  before trusting the SE.

### Day 8 - Mon Jun 22
- **Control variates** for the arithmetic-Asian call; derived, built, validated in
  `05_asian_option_cv.ipynb` (full reference + lessons cells there)
  - $Y_{\text{cv}} = Y - c(X - \mu_X)$, unbiased for any $c$; optimal
    $c^* = \text{Cov}(X,Y)/\text{Var}(X)$ (the OLS slope), giving
    $\text{Var}(Y_{\text{cv}}^*) = \text{Var}(Y)(1-\rho^2)$
  - Control $X$ = geometric-Asian call (closed form, lognormal since
    $\log\bar S_{\text{geo}} = \tfrac1n\sum\log S_{t_i}$ is normal); target $Y$ =
    arithmetic-Asian (no closed form, forces MC)
- **New `simulate_gbm_paths`** in `montecarlo.py`: exact log-space stepping,
  `cumsum` along time, `(n_paths, n_steps)`. General path primitive, reused for all
  path-dependent payoffs (barriers, stochastic vol W4-9). Validated per-timestep vs
  martingale $S_0 e^{rt_i}$: max $z = 0.92$.
- **New `models/exotics.py`** (imports the path primitive): `geometric_asian_price`
  (closed form via effective $\hat\sigma = \sigma\sqrt{(n+1)(2n+1)/6n^2} \to \sigma/\sqrt3$
  and carry $\hat b$), `arithmetic_asian_cv` (target + CV estimator)
  - Geometric variance needs shared-path covariance
    $\text{Cov}(\log S_{t_i}, \log S_{t_j}) = \sigma^2\min(t_i,t_j)$ - log-prices NOT
    independent. Validated vs MC: $z = 0.44$. Directional: geo (5.94) < vanilla (10.45).
- **CV result ($S=K=100$, $T=1$, $r=0.05$, $\sigma=0.2$, $n=12$, 100k paths)**:
  $\rho = 0.9996$, plain SE 0.0269 -> CV SE 0.000752 (~36x). Measured variance ratio
  $7.8\times10^{-4}$ matched predicted $1-\rho^2$ to the digit.
- **Key lesson**: variance reduction buys *compute, not rate* - still $O(N^{-1/2})$.
  36x SE = ~1300x paths by brute force, bought with one closed form + a covariance.
  A control variate is a known-answer rehearsal of the same noise; $1-\rho^2$ is the
  residual variance ($\rho^2 = R^2$).
- **Refactor 1 - editable install**: added `pyproject.toml` + `models/__init__.py`,
  `pip install -e .` (own package only, conda-forge stack untouched). Removed
  `sys.path.append('..')` from all notebooks. Closes Week-1 open issue.
- **Refactor 2 - per-source clean config**: `CLEAN_DEFAULTS` dict (yfinance/cboe
  profiles) in `option_chain.py`; `clean_chain` now takes `source=` + optional
  per-threshold overrides (None-sentinel pattern). Existing explicit-arg calls
  unchanged; `03` smile counts reproduce (83 / 20 strikes). Closes Week-1 open issue.

### Day 9 - Tue Jun 23
- **New `tests/` folder + pytest scaffold** (good-practice item, pulled forward from
  W3): `tests/{__init__,conftest,test_bsm}.py`, `[tool.pytest.ini_options]` in
  `pyproject.toml`. `conftest.py` holds an `atm_params` fixture (the 10.4506 case).
  Float asserts via `np.isclose`, never `==`.
- **`bsm_price` generalised with cost-of-carry $b$** (deferred Option B):
  - $d_1$ uses $b$; spot term gets factor $e^{(b-r)T}$; $K$ term keeps pure $e^{-rT}$
    discounting. $b$ rides with the underlying (carry/forward); $r$ is always the
    discount rate.
  - $b = r$ default (None-sentinel, $b = r$ if None) recovers plain BSM exactly -
    carry factor $= 1$. $b = r - q$ gives dividend yield $q$ - **closes the SPX
    dividend open issue**. $b = 0$ is Black-76.
- **Edge cases handled, vectorised** (closes the "addressed in W2" docstring note):
  - $T = 0$ -> intrinsic $(S-K)^+$; $\sigma = 0$ -> discounted forward intrinsic
    $e^{-rT}(Se^{bT}-K)^+$. $S=0$/$K=0$ left to the formula limit (not special-cased).
  - `np.broadcast_arrays` to align masks; nested `np.where` with $T=0$ taking priority
    over $\sigma=0$. Denominator dummy-substituted ($1.0$ where degenerate) so the
    discarded formula branch emits no div-by-zero warning.
- **Bug caught by a pinned test** (the safety net working day one): zero-vol limit
  first used $fwd = Se^{(b-r)T}$ (carry-adjusted spot) where it needed the *true*
  forward $Se^{bT}$. At $b=r$ this collapsed to $e^{-rT}(S-K)^+ = 0$ - a plausible,
  non-crashing wrong answer. Test pinned to hand-derived 4.877 caught it; fixed with a
  separate `true_fwd`. Same lesson as the digital's zero-SE: a clean number is not a
  correct one.
- **9 tests passing**: canonical ATM, put-call parity (plain + carry), put value,
  monotonic-in-strike, carry-reduces-to-BSM, T=0 intrinsic, zero-vol forward,
  vectorised-with-edge.

### Day 10 - Wed Jun 24
- **GARCH(1,1)** introduced, derived, and fit to real SPX returns;
  theory + from-scratch methodology in `06_garch.ipynb` (full reference + (B)
  hand-rolled MLE write-up cells there)
  - Structure: GARCH(1,1) is **ARMA(1,1) on squared shocks** $\epsilon_t^2$ -
    persistence $\alpha+\beta$, stationary iff $<1$, long-run variance
    $\omega/(1-\alpha-\beta)$, geometric mean-reversion. Returns are white noise in
    *level*; their squares carry the predictable structure.
  - Fit by MLE (squared-shock innovation not iid -> OLS fails); conditional Gaussian
    log-lik $-\tfrac12\sum[\ln\sigma_t^2 + \epsilon_t^2/\sigma_t^2]$, $\sigma_t^2$
    unrolled from the recursion (irreducible loop), constrained $\alpha+\beta<1$.
  - Day-7 chi-squared thread closed: standardised residuals $\hat z_t^2$ as the
    *sample* fit diagnostic (Ljung-Box, QQ) - distinct from the constant BSM $\sigma$.
- **Fit results** (SPX daily, 2016-2026, 2511 returns, spans COVID; `arch` library,
  percent-scaled): $\omega=0.036$, $\alpha=0.162$, $\beta=0.809$. Persistence
  $\alpha+\beta=0.971$, half-life 23.4 days, long-run vol 17.7% annualised
  (matches sample std 1.14% -> internal consistency check passed).
- **Realised vs implied loop closed** (matched date Jun 24, matched 30-day horizon):
  - GARCH 30-day-ahead forecast (variance-path-averaged, un-scaled, annualised): **18.2%**
  - Implied ATM (fresh CBOE pull, 30 DTE expiry 2026-07-24, strike 7370 vs spot 7354): **16.5%**
  - Implied sits ~1.7 vol pts *below* realised forecast -> options mildly cheap on this
    signal; long-vol read (gamma P&L > theta if realised exceeds implied). The
    implied-vs-realised arbitrage from Week-1 IV notes, made concrete.
  - Caveats logged: no dividend in IV inversion (carry $b=r-q$ would fix, ~0.1-0.2 pt);
    Gaussian shocks (fat tails -> `dist="t"`); risk-neutral implied vs physical GARCH
    (variance risk premium - implied *below* realised is the less common config).
- **Data provenance**: yfinance API rate-limited, Yahoo/Stooq programmatic endpoints
  blocked for index symbols (^GSPC download licensing-restricted; SPY proxy or manual
  is the workaround). SPX history obtained via Stooq *manual* download (`^spx_d.csv`).
  Known cache limitation: `period`-keyed price cache goes stale as history grows -
  date-stamped key or freshness check needed for Phase 3.

## Week 3 (Jun 25 - Jul 1): Dupire Local Volatility

### Day 11 - Thu Jun 25
- Branch `feature/week-03-dupire` off `main`.
- **Closed the carry-in-IV loop** (deferred W2 item): threaded cost-of-carry $b$ through
  `bsm_implied_vol` and `compute_smile` so IV inverts against the true forward
  $F = Se^{(b-r)T}$, not the no-dividend $Se^{rT}$.
  - **Fixed arbitrage bounds for carry**: strike discounts at $r$, spot/forward grows at
    $b$. Call $\in [\max(Se^{(b-r)T} - Ke^{-rT}, 0),\ Se^{(b-r)T}]$; put upper bound
    $Ke^{-rT}$. Reduces exactly to the W1 no-dividend bounds when $b=r$.
- **First tests for the IV inverter** (`tests/test_implied_vol.py`, 5 tests): round-trip
  call/put, round-trip with carry ($b \ne r$ - guards today's change), OTM strikes,
  arbitrage-violation -> NaN. **14 tests passing** total.
- **Parity-implied forward** (`implied_forward_from_parity` in `option_chain.py`):
  $F = K + e^{rT}(C - P)$ across common strikes, median for robustness. The
  market-implied (option-3) approach - no external dividend input.
  - **Caught a bad data point**: CBOE `current_price` field (7554) was stale/wrong;
    actual SPX ~7410 (verified). Parity forward (7443.7) agreed across all 19 strikes to
    4 sig figs - **the parity forward is more trustworthy than the reported spot**. This
    is why desks use parity forwards. Real data-hygiene lesson.
- **Dupire setup + intuition** documented in `07_dupire.ipynb`:
  - Local vol = BSM with constant $\sigma$ promoted to a function $\sigma_{\text{loc}}(S,t)$.
    Still one-factor, arbitrage-free, complete - but NOT lognormal, NOT closed-form.
    BSM is the special case $\sigma_{\text{loc}} = $ const. Non-lognormality is the point
    (it is what fits the smile).
  - Derivation roadmap: **Fokker-Planck** (forward Kolmogorov - differentiates terminal
    $(K,T)$, the surface axes; vs backward, which prices one option) -> **Breeden-
    Litzenberger** ($\partial^2 C/\partial K^2 = e^{-rT}p$, density from prices) ->
    substitute + solve for $\sigma_{\text{loc}}^2$.
  - **Structural reading of Dupire's formula**: $\sigma^2$ lives only in the FP diffusion
    term, so isolating it = dividing by the density term -> density ($\partial^2 C/\partial K^2$)
    lands in the **denominator**. Practical curse: density -> 0 in the wings, so
    $\sigma_{\text{loc}}$ blows up there. Exact in theory, minefield in practice - needs a
    smooth arbitrage-free surface BEFORE differentiating.
- **PARKED** (data pipeline issues, rebuild when able):
  - CBOE `current_price` unreliable (use parity forward instead)
  - yfinance fetch broken at notebook top
  - full clean smile re-extraction
  - idea: parity-forward sanity guard (warn if |implied q| > 5%) in `implied_forward_from_parity`

### Day 12 - Fri Jun 26
- New packages `pricing/` and `calibration/` (added to `pyproject.toml`, reinstalled).
- **`calibration/dupire.py`** - `dupire_local_vol`: extracts $\sigma_{\text{loc}}(K,T)$
  from a call-price grid via FD derivatives + Dupire's formula. Density-floor mask
  (NaN where $\partial^2C/\partial K^2$ too small) + verbose masking report.
  - **Case (a) - constant-vol validation**: Dupire on BSM prices returns flat 0.20,
    interior mean 0.2000 std 0.0005. Wings blow up (worst K=120/T=0.1 -> 0.34) **on
    clean noiseless data** - the denominator curse, exactly as derived.
- **`pricing/pde.py`** - Crank-Nicolson local-vol PDE pricer (log-space grid,
  tridiagonal operator with node-dependent vol, asymptotic boundaries, backward march).
  - `setup_grid` (log-uniform in $x=\ln S$, uniform in $t$), `build_operator_diagonals`
    (the FD stencil coefficients), `cn_step` (sparse tridiagonal solve via
    `scipy.sparse`), `price_call_localvol` (assembles + marches + interpolates at $S_0$).
  - **Checkpoint 1**: reproduces `bsm_price` to second order - error quarters per grid
    doubling (ratios 4.11, 4.13, 4.63). Confirms $O(\Delta x^2, \Delta t^2)$.
- **Case (b) - Dupire round-trip** (the week's main result): price under known linear
  skew ($\beta=-0.5$) -> extract -> recover. Mean error 0.008 over surface 0.186-0.216;
  error concentrated only in high-$K$/long-$T$ corner (density thinning). Extractor
  recovers genuine curvature where density supports it.
- **Two bugs caught**:
  1. nested `np.gradient` for $\partial^2C/\partial K^2$ -> sawtooth (even/odd node
     decoupling); fixed with direct stencil $(C_{i+1}-2C_i+C_{i-1})/\Delta K^2$
  2. $\beta=-0.1$ test surface too flat to test anything; needed $\beta=-0.5$
- **TODO (real-data refinements)**: density floor too lax for mild corner instability
  (curvature-aware mask needed); non-uniform-strike second difference for real chains;
  forward-PDE pricer would give whole strike rows per solve (vs 837 backward solves here).
- Notebook: full setup/intuition + implementation + lessons in `07_dupire.ipynb`.

### Day 13 - Mon Jun 29
- **Curvature-aware mask** added to `dupire_local_vol`: per-maturity *relative* density
  floor (mask where $\partial^2_K C < \epsilon \cdot \max_K \partial^2_K C$ for that row,
  $\epsilon=0.01$) OR'd with the absolute floor. Per-maturity peak (`keepdims=True`
  broadcast) is fairer than global - density magnitude falls with maturity, so each row
  judged on its own peak. Verbose report now separates absolute vs relative masking.
- **Case-(b) round-trip properly validated** after a four-hypothesis debugging chase
  (detail in `07_dupire.ipynb` lessons). Root cause: coarse PDE grid (n=100) prices,
  differentiated by $\partial_T$, produced a long-T noise band - NOT density, NOT edges,
  NOT bad prices. Density at the "worst" point was 88% of peak; the error map showed a
  T-band, not a corner. **Fix: re-price at n=200.** Max report error 0.88 -> 0.026,
  mean 0.054 -> 0.003. Clean recovery of the $\beta=-0.5$ linear skew.
- **Key principles banked**: (1) FD amplifies the input's noise floor - extraction is far
  more price-noise-sensitive than direct price comparison; a grid fine enough to *price*
  can be too coarse to *differentiate*. (2) Visualize the whole error field early - its
  spatial structure IS the diagnosis.
- **Type hints**: `pricing/pde.py` annotated to house style (`Callable[[np.ndarray, float],
  np.ndarray]` for the local-vol fn, full tuple return types).
- **Stale-notebook-state caution**: hit a K_grid/sigma_recovered size mismatch from
  re-running cells out of order; resolved by kernel restart. Argues for the pytest
  coverage (clean execution every run) queued for tomorrow.

### Day 14 - Tue Jun 30
- **PDE put mode**: `price_call_localvol` -> `price_localvol` with
  `option_type: Literal['call','put']='call'`. Payoff AND boundaries flip as a matched
  pair (put: low-S boundary $Ke^{-r\tau}-Se^{-q\tau}$, high-S -> 0); `cn_step`,
  `setup_grid`, `build_operator_diagonals` untouched (operator is payoff-agnostic).
  Caught a self-review bug: dropped the `cn_step` call in the loop (would return intrinsic,
  not price). Validated: PUT PDE 5.5759 vs BSM 5.5735, diff 2.4e-3 (same as call - shared
  discretization). Call sites updated (`test_pde.py`, `07_dupire.ipynb`).
- **PDE put-call-parity test**: $C-P = Se^{-qT}-Ke^{-rT}$, atol 5e-3. Observed parity
  holds at 2.4e-3 - errors do NOT cancel (call/put share same-sign discretization error),
  contra my guess they'd subtract out. **18 tests passing.**
- **`test_dupire.py` (extractor went 0 -> 3 tests)**:
  - `recovers_constant_vol`: flat 0.20 interior, nanmean atol 1e-3, nanstd < 5e-3
  - `masks_thin_density`: wide grid -> NaNs appear, ATM survives
  - `no_sawtooth`: regression guard for the nested-`np.gradient` bug. Metric = max abs
    *second difference* along strike axis (sawtooth -> large alternating 2nd diff; smooth
    -> ~0). Clean surface 1.66e-4 vs threshold 1e-3 (~6x margin); sawtooth amplitude ~0.04
    would give 2nd diff ~240x threshold. **Magnitude (2nd diff) separates sawtooth from
    noise; sign-change *count* would not (noise alternates too).**
- **Asian put**: `geometric_asian_price` gains `option_type`; same $\hat\sigma,\hat b,
  \hat d_{1,2}$, only final assembly flips ($K\Phi(-\hat d_2)-Se^{\hat bT}\Phi(-\hat d_1)$).
  Geometric-Asian parity check: C-P vs $e^{-rT}(Se^{\hat bT}-K)$, diff 6.66e-15 (machine
  precision - closed form, no discretization). `arithmetic_asian_price_cv` threads
  `option_type` to all three (target payoff, control payoff, `mu_X` call); CV machinery
  payoff-agnostic. [verified put X-Y corr stays ~0.9996]
- **Type hints**: `price_localvol`, `arithmetic_asian_price_cv` annotated. Sweep complete.
- **Dupire stays call-only** (no `option_type`): it's a *calibrator*, not a pricer -
  Dupire's formula is defined on the call surface (Breeden-Litzenberger). Puts -> convert
  to calls via parity *upstream* in the data pipeline, not in the extractor.

### Day 15 - Wed Jul 1
- **Option chain finished end-to-end on live CBOE data**: parser locked via assert on
  known OCC symbol (SPXW260618C00200000). Live path fetch -> SPXW filter -> clean ->
  parity forward, all working. Expiry 2026-07-31 (30 DTE), cleaned to 105 calls / 196
  puts. Parity forward 7500.64, std 0.53 over 38 strikes (0.007%), near-perfect parity
  consistency. current_price 7499.36 fresh this time and agrees (cross-validation bonus).
- **Implied q diagnostic reads 3.85%** (high vs SPX's true ~1.3%). NOT a bug: q = r - b
  inherits the untrusted spot and assumed r; over short T small spot errors annualize into
  large q errors. The forward and carry b (what we actually use) are correct; q is a
  flagged sanity check only, not used downstream.
- **Notebook 03_option_vol_smile.ipynb rebuilt**: was an archaeological dig (Week 1 title,
  broken early smile with hardcoded 7554.29 spot, two fetches, two smile methods one
  wrong). Now 18 clean cells, future-proof (nothing hardcoded, all dates/spots derived at
  run time), reads as a data-workflow explainer. Two helpers (forward_for_expiry,
  smile_for_expiry) so the two-maturity logic isn't copy-pasted. Fixed the cell-14 bug
  (75 DTE used unfiltered all_options, reintroducing duplicate strikes).
- **Folder reorg**: option_chain.py moved models/ -> new data/ package (it is data
  infrastructure, not a pricing model). Created data/__init__.py, registered data in
  pyproject.toml, pip install -e ., fixed notebook imports, verified + pytest green.
  Cache-path logic (Path(__file__).parent.parent) survives the move unchanged (data/ is
  same depth as models/ was); stale comment updated.
  W3. Dupire validated on synthetic ground truth only; real-data smoothing deferred to
  where Heston lives longest (W4-W6) and calibration needs a clean target surface.
- **Week 3 branch closed**: PR merged to main, tagged v0.3-week3.
- Note: SVI smile smoothing (flagged W1 for W3) intentionally carried to W6, not done in
  W3. Dupire validated on synthetic ground truth only; real-data smoothing deferred to
  where Heston lives longest (W4-W6) and calibration needs a clean target surface.

## Week 4 (Jul 2 - Jul 8): Heston

### Day 16 - Thu Jul 2
- Branch `feature/week-04-heston` created off `main` (v0.3-week3)
- **Conceptual groundwork only, no code yet** - Heston SDE system, Feller condition, and the
  CIR-to-noncentral-chi-squared link (thread closed from Day 7/Day 10) worked through from
  first principles before touching simulation code
- Established why $(S_t, v_t)$ jointly Markovian but $S_t$ alone is not - Feynman-Kac gives a
  3D PDE $(t,S,v)$ here vs Dupire's 2D $(t,S)$, motivates going straight to characteristic-
  function pricing in W5 rather than a direct PDE solve
- Traced Euler discretization failure mode for CIR: unbounded Gaussian shock evaluated at
  interval start can drive $v$ negative for any $\Delta t > 0$; Feller condition reduces
  frequency but does not eliminate the failure mode structurally
- Compared exact CIR sampling (noncentral chi-squared via Poisson-mixture) vs Euler on cost,
  and separately identified why exact *joint* $(S,v)$ simulation (Broadie-Kaya) needs the
  integrated variance $\int_t^{t+\Delta t} v_s\,ds$ conditional on both endpoints, not just the
  endpoints themselves - no closed form, inverted numerically from Laplace transform
- Notebook `07_heston.ipynb` started: intro (motivation, SDE system, CIR/chi-squared thread
  payoff, week plan) and lessons-learned section written up
- **Decision**: implement Andersen QE tomorrow (Day 17) rather than full Broadie-Kaya - better
  bias/speed tradeoff than truncated Euler, avoids Broadie-Kaya's numerical Laplace inversion
  cost

### Day 17 - Mon Jul 6
- **Andersen QE variance sampler** implemented in `models/heston.py`
  - `cir_qe_regime_params(v_t, kappa, theta, xi, dt, psi_c)` - single source of
    truth for $m$, $s^2$, $\psi$, and both regimes' moment-matched quantities
    ($a, b^2$ for squared-Gaussian; $p, \beta$ for point-mass+exponential), shared
    by the sampler and (Day 18) the spot-side martingale correction
  - `sample_cir_qe_step` - thin wrapper: draws masked Gaussian/uniform per regime,
    assigns into `v_next`, never negative by construction
  - **Silent bugs caught and fixed across iterations**: $s^2$ formula missing a
    $(1-e^{-\kappa\Delta t})$ factor on the first term, then a sign flip on the
    second term (`-` instead of `+`, would have made $s^2$ go negative for some
    parameter regimes); regime-specific quantities (`a`, `b2`, `p`, `beta`) not
    masked to match random-draw shapes before combining, would have thrown or
    silently misbehaved on broadcast
  - **Test-parameter bug, not a code bug**: first attempt at forcing regime 2 used
    `dt=1/252` with badly-violated Feller, still landed in regime 1. Root cause:
    $\psi \approx \xi^2\Delta t/v_t$ to leading order, so *smaller* $\Delta t$ pushes
    toward regime 1 regardless of Feller, regime 2 needs large $\Delta t$/small
    $v_0$/large $\xi$ together
- **`tests/test_heston.py`**: moment-matching tests (Feller-satisfied and
  Feller-violated parameter sets), never-negative regression guard. 24 passed.
  Known blind spot flagged: moments-only tests can't rule out a regime-1/regime-2
  swap bug (TODO: add fraction-of-exact-zeros check in a deep regime-2 case)
- Cosmetic `RuntimeWarning: invalid value encountered in sqrt` on regime-2 paths
  (computing `b2` unmasked before use) - harmless, correctly masked before
  assignment, candidate `np.errstate` cleanup, not urgent

### Day 18 - Mon Jul 6
- **Spot-side simulation, `simulate_heston_paths`**: derived $K_1$-$K_4$ and the
  regime-specific martingale correction $K_0$ from first principles (integrate the
  variance SDE, isolate $\xi\int\sqrt{v_s}\,dW^v_s$ as a known quantity from the two
  QE endpoints, correlated spot noise term follows by dividing by $\xi$ and
  multiplying by $\rho$) rather than pattern-matching a formula
  - $K_0$ reuses `cir_qe_regime_params`'s $a,b^2,p,\beta$ directly via each regime's
    moment-generating function evaluated at $s=K_2+\tfrac12K_4$
  - Implementation correct on first full pass; only inefficiency flagged:
    `cir_qe_regime_params` called twice per step (once inside `sample_cir_qe_step`,
    once directly for $K_0$), wasteful not incorrect, TODO to thread through as a
    combined return
- **Martingale property test** (`test_heston_martingale_property`): $E[S_T] = S_0
  e^{rT}$ checked against a Day-3-style standard-error tolerance
  ($\sqrt\theta\,S_0e^{rT}\sqrt{T/N}$, $\sqrt\theta$ standing in for constant
  $\sigma$ since Heston vol mean-reverts rather than staying fixed), 4 SE bound.
  Passed. This isolates $K_0$ specifically - the Day 17 variance-moment tests never
  touch the spot process and couldn't have caught a $K_0$ bug
- **25 tests passing total**
- **Results generated in `08_heston.ipynb`**:
  - Leverage effect: largest $v_t$ spike coincides exactly in time (not lagged) with
    the sharpest $S_t$ drop across sample paths, $\rho=-0.7$ visibly at work
  - CIR transition density: simulated $v_T$ histogram vs theoretical noncentral
    $\chi^2$ (via `scipy.stats.ncx2`, parameter mapping $c,df,nc$ from CIR
    coefficients) - matches near zero; **TODO**: linear-scale plot can't confirm
    tail agreement past $v_T\approx0.05$, replot log-scale
  - Endogenous smile: downward-sloping IV vs strike from simulated $S_T$ + MC
    payoff averaging + `implied_vol` inversion, correct sign for $\rho<0$ (fattened
    left tail -> higher IV at low strikes), same shape as real Week-1 SPX smile,
    produced from one correlation parameter rather than fit pointwise
  - Term structure: long-$T$ smile visibly flatter than short-$T$ at matched
    strikes - the Week-3-flagged limitation now directly observed, not just asserted
- **Vega/inversion noise diagnosed at deep ITM strike** ($K=70$ vs $S_0=100$): small
  kink in the smile traced to $\delta\sigma \approx \delta C/\text{vega}$, low vega
  amplifies ordinary MC price noise into larger recovered-IV noise (initially
  guessed backwards - low sensitivity in the forward direction does not mean low
  noise in the inverse direction). **TODO**: always invert the OTM instrument per
  strike (call for $K>S_0$, put for $K<S_0$, parity-converted if needed), same
  principle as the Dupire put-to-call parity TODO, deferred rather than fixed today

### Day 19 - Tue Jul 7
- **TODO 1 closed**: `test_qe_regime2_zero_fraction` added to `test_heston.py`.
  Guards against a regime-1/regime-2 swap bug that the Day 17 moments-only tests
  couldn't catch (both branches are moment-matched to the same $m,s^2$, so a swap
  wouldn't necessarily move the mean/variance). Checks `(v_next == 0.0).mean()`
  against `p` from `cir_qe_regime_params` directly (not hand-recomputed, so the
  test tracks the actual formula rather than a stale hardcoded target), tolerance
  from the binomial proportion standard error. Caught one bug along the way: first
  attempt reused stale Feller-violated parameters from the original (pre-fix)
  version of `test_qe_moments_feller_violated` that never actually reached regime 2
  ($\psi=1.0$ regime-1 boundary, `mask_1.mean()==1.0`); corrected by reusing the
  actual working parameters from that test. 26 tests passing
- **TODO 2 closed**: `sample_cir_qe_step` gained `return_diagnostics: bool = False`
  (default off, all existing call sites and tests unaffected), returning
  `(v_next, mask_1, a, b2, p, beta)` when `True`. `simulate_heston_paths` now calls
  `cir_qe_regime_params` exactly once per step via this flag, removing the
  redundant direct call that previously duplicated the same computation for the
  $K_0$ correction. Return type hint updated to `Union[np.ndarray, tuple[...]]`.
  Full suite reconfirmed green after the signature change
- **TODO 3 closed**: OTM-instrument convention applied to both the smile (result
  #3) and term-structure (result #4) notebook cells: call priced/inverted for
  $K\ge S_0$, put for $K<S_0$, put payoff computed directly from simulated $S_T$
  (not via parity-converted call price, since that would just carry the same MC
  noise through algebra rather than fixing the conditioning). Confirmed visually:
  the $K=70$ kink present in Day 18's original smile plot is gone in both replots,
  clean monotone curves across the full strike range on both maturities
- Noted in passing: `implied_vol.py` now imported from `calibration.implied_vol`
  rather than `models.implied_vol` (the longer-deferred move flagged as high-risk
  due to `test_implied_vol.py` dependency) - confirm and log when/how that move
  happened if not already captured elsewhere
- **All three Week 4 open TODOs from Day 18 now closed.** Remaining for tomorrow:
  Heston characteristic function derivation (last piece of the original Week 4
  arc, sets up Week 5's Carr-Madan/COS Fourier pricing directly)

### Day 20 - Wed Jul 8
- **Heston characteristic function derived**, last piece of the Week 4 arc, sets up
  Week 5's Carr-Madan/COS directly
  - Backward Kolmogorov PDE (Feynman-Kac) rebuilt from first principles via Ito's
    product rule on the martingale $N_t=e^{-rt}u(t,X_t)$, since this hadn't been
    covered before (only the forward/Fokker-Planck side, from Dupire), collecting
    the $dt$-drift and setting it to zero
  - Extended to the Heston pair $(x=\ln S, v)$: three second-derivative terms (two
    pure, one $\rho$-correlation cross term), full PDE derived term by term,
    coefficients confirmed correctly by hand before assembly
  - Affine ansatz $\phi=\exp(iux+C(\tau)+D(\tau)v)$ substituted, PDE separated into
    a linear ODE for $C$ and a Riccati ODE for $D$; closed-form Riccati solution
    taken as reference (Heston 1993/Gatheral), consistent with house convention of
    deriving structural results but referencing genuinely involved closed-form
    algebra
  - Full derivation and closed-form written up in `08_heston.ipynb`
- **Week 4 (Heston) closed.** QE variance sampler, correlated path simulator with
  properly-derived martingale correction, four validated results (leverage effect,
  chi-squared tail, endogenous smile, term-structure flattening), OTM-instrument
  convention fixed, characteristic function derived. 26 tests passing
- Branch `feature/week-04-heston` ready to merge `--no-ff`, tag `v0.4-week4`
- **Next**: Week 5, Fourier pricing (Carr-Madan, COS), building directly on
  today's $\phi(u;\tau)$, new notebook `09_fourier_pricing.ipynb`


## Week 5 (Jul 9 - Jul 22): Fourier Pricing

### Day 21 - Thu Jul 9
- Week 4 (Heston) merged --no-ff into main, tagged v0.4-week4
- Week 5 (Fourier pricing) started. `heston_char_func` implemented in
  `models/heston.py`, using the branch-safe Riccati root (Albrecher et al. 2007,
  "The Little Heston Trap") rather than the naive Heston (1993) form, which
  produces a discontinuous complex log for long maturities/high vol-of-vol.
  Sign flip in the definition of `g` is the entire fix, everything else identical
  to the derivation from Day 20
  - Cross-domain note: Riccati equation naming connects directly to LQR/Kalman
    covariance propagation from control theory, same quadratic-in-the-unknown
    backward-in-time structure, both arising from a linear-quadratic cost/variance
    object propagated via Feynman-Kac-adjacent machinery
- COS method theory covered conceptually: truncate density to $[a,b]$, cosine-series
  coefficients $A_k$ recoverable directly from $\phi(u_k)$ (no numerical
  integration), price collapses to a finite sum $\sum A_k V_k$ against payoff
  cosine coefficients $V_k$. Implementation (choosing $[a,b]$ from cumulants,
  deriving $V_k$ for a call, assembly + MC cross-validation) deferred to next
  session
- **Schedule change**: Jul 13-19 off (graduation ceremony + holiday). Week 5
  continues tomorrow (early start planned), remainder resumes after the break,
  downstream weeks (6-9) shift accordingly, no fixed date target for now

### Day 22 - Fri Jul 10
- **`heston_char_func` validated** against two independent checks before building on it:
  - `phi(u=0)=1` exactly, for any tau (trivial but effective boundary sanity check)
  - Cross-checked against Week 4's Monte Carlo simulator directly:
    `mean(exp(iu*ln(S_T)))` over simulated paths vs `heston_char_func(u,...)`,
    tolerance derived from the bound Var(unit-modulus RV) <= 1, giving
    SE <= 1/sqrt(N), checked at 3 SE (~6.7e-3 at N=200k). Both passed
- **COS method implemented end to end** in `pricing/fourier.py`:
  - `heston_cumulants` (c1, c2, closed form, Fang & Oosterlee), validated against
    the xi->0, v0=theta limit collapsing exactly onto BSM's known mean and variance
    of ln(S_T), both c1 and c2 checked independently
  - `cos_truncation_range` (a, b from cumulants, c4=0 simplification)
  - `cos_call_coefficients` (V_k, payoff cosine coefficients, reference formula)
  - `cos_call_price` assembling A_k, V_k, and the k=0 half-weight into the final sum
- **Bug found and fixed**: COS price (4.75) disagreed sharply with an independent
  direct Fourier-inversion cross-check (6.84, matching MC's 6.83) built specifically
  to isolate whether the char function or the COS assembly was at fault, confirmed
  char function was fine. Diagnosed via an L-sweep (price should stabilize once the
  truncation range is wide enough; instead it kept halving as L doubled), a
  dimensional tell rather than a one-off wrong number. Root cause: `A_k` and `V_k`
  each independently carried a `2/(b-a)` normalization factor, but only `A_k`
  (the density's cosine coefficient) should carry it; `V_k` is a plain payoff
  integral against a raw cosine and picked up a spurious second copy. Removed the
  factor from `cos_call_coefficients`, confirmed price stabilizes across L after
  the fix
- **`test_cos_call_price_vs_monte_carlo`** passing, tolerance from the MC sample's
  own empirical standard error (not a guessed constant)
- **Week 5 status at the break**: characteristic function + COS method fully
  implemented and validated end to end. Carr-Madan and the convergence-rate
  validation suite (COS accuracy vs N, vs MC, across strikes) deferred to after
  the break
- **Break starts now**: Jul 13-19 off (graduation ceremony + holiday). Resume
  Week 5 wrap-up (Carr-Madan, convergence checks) whenever back, no fixed date

### Day 23 - Mon Jul 20
- Back from the Jul 13-19 break, resumed Week 5
- **COS convergence-rate validation**: rather than using MC as the reference
  (its own irreducible ~1/sqrt(N) noise floor would swamp COS's much smaller
  error at moderate N), used a high-N (2048) COS price as a converged reference
  instead, standard technique when no independent closed form exists but a
  method is known to converge. Confirmed exponential convergence (error dropped
  ~6 orders of magnitude from N=8 to N=64 on a semilog(error) vs N plot, straight
  line as spectral-accuracy theory predicts), flattening near ~1e-13 once both
  sides hit floating-point noise
- **Refactored `cos_call_price`** to separate the strike-independent `A_k`
  (density coefficients, needs `heston_char_func`) from the cheap per-strike
  `V_k`, added `cos_density_coefficients` and `cos_smile` for batched strike-strip
  pricing without recomputing `A_k` per strike, same pattern as Day 18's
  `cir_qe_regime_params` factoring
- **Added native put pricing**: `cos_put_coefficients`, derived independently
  (not via parity) specifically so it serves as a real cross-check on the call
  path rather than a guaranteed-to-agree algebraic restatement of a potential bug
- **Two real bugs caught by the new test suite** (`test_fourier.py`):
  1. The `A_k`/`V_k` double-`2/(b-a)` normalization regression (from Day 22-23)
     briefly reappeared during the refactor, caught immediately by
     `test_cos_call_price_vs_monte_carlo` reproducing the exact old wrong price
  2. `cos_put_coefficients` had two compounding transcription errors: `K`
     multiplying the wrong building block (`chi` instead of `psi`) and an overall
     sign flip, `chi - K*psi` where the derivation needs `K*psi - chi`. Root
     cause worth remembering: the put isn't just the call's formula over
     different bounds, the payoffs are algebraic negatives of each other
     ($K-e^x=-(e^x-K)$), so the whole expression's sign structure needs
     re-deriving, not just the integration bounds swapped. Caught via
     `test_cos_put_vs_parity` and confirmed step by step with direct numerical
     checks against the parity target until the sign landed correctly
  3. `cos_smile`'s `np.empty_like(strikes)` silently inherited int64 dtype when
     given integer strikes, truncating prices; fixed to
     `np.empty(len(strikes), dtype=float)`, regression-guarded by
     `test_cos_smile_dtype_safety`
- Removed duplicate `test_cos_call_price_vs_monte_carlo` left in `test_heston.py`
  after moving it to `test_fourier.py`
- **35 tests passing.** Full COS pricer (calls, puts, batched smile) validated:
  convergence rate, MC cross-check, put-call parity, dtype safety
- Full smile plotted in `09_fourier_pricing.ipynb`: COS vs MC, same shape, COS
  smooth/instant, MC visibly noisier at the same strikes
- **Week 5 core methods done.** Carr-Madan still open (lower priority now that
  COS is fully validated and will likely be the production method going forward);
  next up whenever picked up: Carr-Madan implementation, or move straight to
  Week 6 (calibration) if Carr-Madan is deprioritized entirely

### Day 24 - Tue Jul 21
- **Carr-Madan method derived and implemented**, `pricing/carr_madan.py`
  - Derivation: raw call price $C(k)$ doesn't decay as $k\to-\infty$ (approaches
    $S_0$, a constant), so it isn't Fourier-transformable as-is. Damping factor
    $e^{\alpha k}$ ($\alpha>0$) forces left-tail decay, constrained from above by
    not overpowering the right tail's already-existing decay, formal constraint
    $E[S_T^{\alpha+1}]<\infty$, $\alpha=1.5$ used as the standard practical default
  - Damped transform $\psi(u)$ derived, evaluates the existing `heston_char_func`
    at a shifted complex argument $u-(\alpha+1)i$, no new characteristic-function
    code needed
  - DFT sampling identity derived from first principles (matching
    `np.fft.fft`'s exponent against the integral's kernel term by term):
    $\Delta u\cdot\Delta k=2\pi/N$, frequency-grid and strike-grid spacing are not
    independent choices, a genuine resolution/accuracy trade-off, not a free
    parameter each
  - Implementation: grid construction (u, dk, k, strikes) written independently
    from the derived identity; Simpson's rule weights, the $k_0$ phase-shift
    correction, and damping removal taken as scaffolded reference (standard
    implementation mechanics of the method, not something to re-derive)
  - **Correct on first full run**, no debugging needed this time, validated
    directly against COS across a strike spread (K=80 to 116), agreement to 5
    decimal places
  - `test_carr_madan_vs_cos` added, looser tolerance (1e-2) than COS's internal
    convergence checks, appropriate given Carr-Madan's real FFT/Simpson
    truncation error vs COS's exponential convergence
  - Theory write-up drafted for `09_fourier_pricing.ipynb` (not yet pasted in)

### Day 25 - Wed Jul 22
- **Carr-Madan puts added** via put-call parity from the already-validated call
  strip. Deliberate choice over a native derivation (unlike COS's put), since
  Carr-Madan's call side was already cross-validated against COS, a native put
  would mostly re-confirm already-confirmed machinery rather than add signal
- `test_carr_madan_put_vs_cos_put`: parity-derived CM put vs COS's natively-derived
  put, two structurally independent paths, genuine cross-validation rather than
  parity checking itself. Passed
- Notebook demo cell: Carr-Madan (FFT) vs COS smile overlay, `np.interp` used to
  map Carr-Madan's fixed FFT strike grid onto the chosen comparison strikes,
  standard practice for consuming Carr-Madan output, not a workaround
- Lessons-learned write-up completed in `09_fourier_pricing.ipynb`: damping
  factor constraint, DFT sampling identity, the parity-vs-native put reasoning,
  and a practical COS-vs-Carr-Madan takeaway
- Extended Carr-Madan vs COS comparison beyond the IV overlay (indistinguishable
  at plotting scale): raw price error vs strike, and a second N-sweep isolating
  quadrature from grid resolution. Found a grid-alignment artifact at K=100 and
  a genuine two-error-source bottleneck in Carr-Madan (quadrature vs interpolation
  grid, linked by the sampling identity). Full reasoning and plots in
  `09_fourier_pricing.ipynb`
- Full test suite reconfirmed green before merge
- **Week 5 (Fourier pricing) closed.** heston_char_func (branch-safe), COS
  (calls, puts, batched smile, convergence validated), Carr-Madan (FFT calls,
  parity puts), all cross-validated against each other and against Week 4's MC.
  Three independent pricing routes agreeing on the same smile
- Branch `feature/week-05-fourier` ready to merge --no-ff, tag v0.5-week5
- **Next**: Week 6, Heston + SABR calibration, SVI smile smoothing (deferred
  since W1/W3)

## Week 6 (Jul 23-29): Calibration (SVI, Heston + SABR)

### Day 26 - Thu Jul 23
- Started Week 6, branch `feature/week-06-calibration`, new module
  `calibration/svi.py`. Theory and scaffolding day, no fitting run yet
- **Raw SVI parametrization** in total implied variance against log-moneyness
  $k=\log(K/F)$: $w(k)=a+b\left(\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}\right)$
  - Total variance not implied vol, because calendar arbitrage reduces to
    $\partial_T w \ge 0$, plain monotonicity; the same condition in vol space
    is uglier and easy to get wrong
  - Log-moneyness against the forward, not spot, so slices are
    maturity-comparable and centred on ATMF; parity-implied forward already
    canonical from W1
  - SVI is Stochastic Volatility *Inspired*, no SDE underneath. Static
    parametrization with no dynamics, so it is the clean arbitrage-free target
    Heston and SABR calibrate *against*, not a model in its own right
- **Vertex identities** at $k=m$: $w=a+b\sigma$, $w'=b\rho$, $w''=b/\sigma$
  - These are also the three observable combinations near the money (level,
    slope, curvature), which is exactly why raw SVI is badly identified:
    $\{a,b,\sigma\}$ collapse onto two of them, and $\rho$ only separates from
    $b$ once the wings pin $b$ down. On thin noisy CBOE wings, $b$ goes soft
- **Zeliade two-stage reduction** adopted for this reason. Substituting
  $y=(k-m)/\sigma$, $c=b\sigma$, $d=\rho b\sigma$ gives $w=a+dy+c\sqrt{y^2+1}$,
  linear in $(a,d,c)$ for fixed $(m,\sigma)$
  - Inner problem: constrained linear least squares, cheap and reliable
  - Outer problem: 2D Nelder-Mead over $(m,\sigma)$ only, the two geometrically
    interpretable parameters. Five correlated dimensions down to two
  - Recovery $b=c/\sigma$, $\rho=d/c$
- **Lee wing bound derived rather than copied.** As $k\to\infty$, $w\sim\beta k$
  with $\beta=b(1+\rho)$, so $g(\infty)=1/4-\beta^2/16$ and $g\ge 0$ forces
  $\beta\le 2$ in total-variance units
  - Claude initially quoted 4 from memory; that is either the two-wing sum form
    or a slack box constraint. Published constants here differ by convention
    (total variance vs vol, per-wing vs summed) and a wrong factor silently
    lets long-dated slices fit steeper wings than any martingale supports
  - Same lesson as the COS normalization: derive in your own convention
- **Durrleman condition**: butterfly arbitrage is exactly $g(k)<0$, since the
  rest of the density $\frac{1}{\sqrt{2\pi w}}e^{-d_2^2/2}$ is strictly positive
  - $g=\left(1-\frac{kw'}{2w}\right)^2-\frac{w'^2}{4}\left(\frac1w+\frac14\right)+\frac{w''}{2}$
  - $w''>0$ always for raw SVI, so only the middle term can drive $g$ negative
  - Splitting it is the useful bit: the $-w'^2/16$ piece is asymptotic and just
    reproduces Lee, while $-w'^2/(4w)$ is the *practical* failure mode. It blows
    up where $w$ is small and $w'$ steep, i.e. **short-dated slices on the steep
    put wing, not the far wing**
  - Consequences: fit shortest maturities first (failures surface before a whole
    surface is built on top), and make the $g$ grid densest near the money
- Analytic $g$ from analytic $w,w',w''$, never numerical second differences of a
  priced call strip. Same reasoning as W3's nested-`np.gradient` sawtooth

### Day 27 - Tue Aug 12
- Back after a ~3 week break. Reconciliation plus one new function, `durrleman_g`
- **Reconciled `svi.py` against the log.** The three W6D1 fixes had landed
  (radicand +sigma^2 in all three derivative funcs, `return` not `raise`,
  `column_stack`), but a fresh read caught four new fill-in slips:
  1. `np.ones_like(len(k))` - `len(k)` is a scalar so this returned a single 1,
     not a column. Fixed to `ones_like(k)`
  2. `reduced_constraints(sigma, tau)` called with two args after `tau` was
     dropped from the signature - instant TypeError. Fixed
  3. `recover_raw` returned a bare `(b, rho)` tuple, silently dropping a, m,
     sigma. Rewritten to full `SVIParams` with both guards: degenerate-c floor
     (1e-12, rho=0 for the flat-slice case where rho is genuinely undefined) and
     rho clip to [-1, 1] (SLSQP satisfies |d|<=c only to tolerance, overshoot
     kills sqrt(1-rho^2) downstream)
  4. Dropped vestigial `tau` from `solve_inner`
  - 1 and 3 were silent, the dangerous ones
- **`solve_inner` two-stage branch completed**: unconstrained `lstsq`,
  feasibility check, SLSQP only on failure warm-started from the lstsq point
- **`durrleman_g` written and certified.** g = (1 - k w'/(2w))^2
  - w'^2/4 (1/w + 1/4) + w''/2, each of w, w', w'' evaluated once
- **Wrote `tests/test_svi.py`, 12 tests, all green:**
  - Vertex identities at k=m against exact closed forms w=a+b*sigma, w'=b*rho,
    w''=b/sigma, atol 1e-14. Params all-distinct and none equal to 1 so a
    swapped b/sigma cannot pass by coincidence, plus a meta-test on that
  - FD consistency: w' vs central diff of w, w'' vs central diff of w', across
    vertex, crossover (m +/- sigma), wings (m +/- 10 sigma), both signs
  - Convergence-rate: log-log slope of FD error vs h must be ~2
  - Structural: w'' > 0 across a wide grid, integer-k dtype safety (W5 lesson)
  - Durrleman g gates: flat slice (b=0) gives g==1 everywhere (free correctness
    test, catches a dropped factor or sign error in term one); admissible slice
    stays g > 0 (no false positives); Lee-bound-violating slice
    (b(1+|rho|) > 2) drives g < 0 somewhere (a checker that never fires is not
    a checker)
- **Convergence-rate debug**: `test_second_derivative_convergence_rate` first
  read slope ~4, not ~2. Not a wrong derivative (that plateaus at slope 0) but
  an unlucky eval point (m + 0.5 sigma) where the leading h^2 coefficient
  ~w'''' is near zero, so h^4 dominates and slope reads high. Moved the sweep
  near the vertex (m + 0.15 sigma) where higher derivatives are large and
  generic. Slope back to ~2. Slope-too-high is as diagnostic as too-low

#### Still open
- `solve_inner` returns `res.x` even on `res.success == False` (just warns).
  Failure-path decision belongs with `outer_objective` returning inf for a
  poisoned candidate; not yet wired
- `reduced_constraints` docstring still references a stale "wrong tau scaling"
- `svi_density` (belt-and-suspenders density cross-check vs g), then the outer
  layer (`butterfly_penalty`, `outer_objective`, `fit_slice`), then first real
  CBOE slice fit

#### Pace note
Targeting full W6 (SVI fits, Heston calibration, SABR) by Wed next week.

#### Design decisions
- **Butterfly penalty lives in the outer objective**, squared hinge
  $\lambda\sum\max(0,-g(k_i))^2$, so the inner solve stays linear. $g\ge 0$ is
  emphatically not linear in $(a,d,c)$ and putting it inside would destroy the
  reason for the reduction
  - Accepted cost: the composite is inconsistent, inner minimizes pure RMSE
    while outer sees RMSE plus penalty. Documented, not swept under
  - **Not regularization.** Ridge expresses a preference; this is a hard
    admissibility condition. A slice with $g<0$ is not a worse fit, it is not a
    density. So $\lambda$ is not cross-validated, it is ramped until violations
    vanish and the accepted fit is gated on $g\ge0$ directly, pass/fail
  - Continuation strategy: fit unpenalized first ($\lambda=0$), gate, only refit
    with ramped $\lambda$ warm-started from that solution if the gate fails.
    Most slices should cost nothing
- **Penalty grid fixed in $k$-space** across outer iterations, not regenerated
  per candidate. A moving grid makes the penalty jump as violation dips fall
  between sample points, and Nelder-Mead reads those jumps as real structure.
  Resolution handles the real concern instead, with the accepted fit
  re-verified on a much denser grid afterwards
- **One quote per strike** by the OTM convention (puts $k<0$, calls $k\ge0$),
  never blended. The ITM leg is the illiquid one. Also, since the forward is
  parity-implied, call/put IV disagreement at other strikes signals the forward
  is slightly off at that strike, not two independent measurements to average.
  Log it as a diagnostic instead
- **Weights frozen from raw market IVs**, never updated inside the optimizer
  - Updating breaks inner linearity ($W$ would depend on $(a,d,c)$)
  - Worse, it opens a degenerate direction: $W_{ii}$ falls as fitted vol rises,
    so the objective can be reduced by *inflating* $w$ without fitting anything
    better. The correct likelihood carries a $\sum\log\sigma_i(\theta)$ term
    that penalizes exactly this, and dropping it is what opens the hole. Same
    reason a learned noise scale cannot be treated as fixed in PyMC
  - IRLS is the honest version if self-consistency is ever wanted
  - Weight construction: spread $\to$ vol via vega, vol $\to$ total variance via
    $2\sigma_{BS}\tau$. Reintroduces vega, so low-vega strikes are down-weighted
    automatically, consistent with the W4-W5 conditioning thread

#### Written
- `svi_raw`, `svi_raw_first`, `svi_raw_second`, `reduced_design_matrix`,
  `reduced_constraints` done; `solve_inner` nearly done

#### Bugs caught in review
1. **Sign error in the radicand**, $(k-m)^2-\sigma^2$ instead of $+$, in all
   three of `svi_raw` and both derivatives. Nasty failure mode: NaN only when
   $|k-m|<\sigma$, a hole centred exactly on the vertex, finite and plausible
   everywhere else, so a wing-only test passes clean. Loud where you look least
2. `raise dw` / `raise d2w` instead of `return`, from editing the
   `raise NotImplementedError` line rather than replacing it
3. `reduced_design_matrix` built with `np.array([...])` giving shape $(3,n)$
   instead of $(n,3)$. Silently square and wrong at exactly 3 strikes, and
   otherwise the shape error surfaces layers away in the outer objective. Fixed
   with `np.column_stack`
4. `np.linalg.lstsq(A, ub)` as the SLSQP start: least-squares fitting the
   *constraint* rows, which contain no market data at all. Lands on boundaries
   and knows nothing about the slice. Wanted `lstsq(Xw, ww)`, the unconstrained
   fit to the data
5. `Warning.add_note` on the class: silent no-op with nothing raising it, which
   was the exact failure the line was meant to prevent

#### Good call worth keeping
- Put $\sigma$ in `ub` (as $2\sigma$) rather than $1/\sigma$ in `A` for the Lee
  rows. Legal since $\sigma>0$, keeps `A` as pure $\pm1$ entries, and degrades
  gracefully as $\sigma$ shrinks instead of blowing up the coefficient matrix
  exactly where the vertex is sharpest
- Falls out of that: as $\sigma\to0$ the Lee rows force $c=d=0$, so with $c\ge0$
  the feasible set **collapses to a single point** (a flat slice). The outer
  search therefore needs a hard lower bound on $\sigma$ (~1e-3), and should log
  when the optimizer sits on it, since that signals the data wants a kink raw
  SVI cannot deliver
- `tau` now confirmed unused in `reduced_constraints` since everything is in
  total variance; drop from the signature or comment why it stays

#### Open for tomorrow
- Finish `solve_inner`: two-stage branch (unconstrained `lstsq`, feasibility
  check `np.all(A @ x <= ub)`, SLSQP only on failure). Decide strict `<= ub` vs
  `+ tol`, a choice coupled to the clip in `recover_raw`. Log *which*
  constraint row failed, not just that one did
- Failure-path signature: leaning towards a flag over raise or warn, with
  `outer_objective` returning `np.inf` for a candidate whose inner solve failed,
  so the search walks away rather than accepting a garbage fit. Raising is wrong
  inside a loop called thousands of times
- `recover_raw` guards: degenerate $c$ ($\rho$ genuinely undefined at $b=0$,
  pick and document a convention rather than letting 1e-14 decide the skew) and
  $\rho$ overshooting 1 by tolerance (kills $\sqrt{1-\rho^2}$ downstream). Both
  tolerance-boundary bugs, same family as the COS normalization
- `durrleman_g` plus validation gates, ground-truth-first as usual:
  - FD convergence-rate check on both derivatives (slope 2, stopping the $h$
    sweep before the roundoff floor at $h\sim\epsilon^{1/3}$, $\epsilon^{1/4}$)
  - Evaluated at the vertex $k=m$ (exact closed forms, no discretization error
    to hide behind) and at $m\pm\sigma$, $m\pm10\sigma$ for the crossover and
    asymptotic regimes, both signs to catch a $\rho$ flip
  - Test params all distinct and none equal to 1, else $b/\sigma$ and $b\sigma$
    are indistinguishable and a swapped $b,\sigma$ passes. Using
    $(a,b,\rho,m,\sigma)=(0.04,0.4,-0.3,-0.05,0.15)$
  - Flat vol must give $g\equiv1$ and reproduce the lognormal exactly; density
    integrates to 1; $b$ broken past the Lee bound must make $g$ go negative
    (a checker that never fires is not a checker)
- Then `outer_objective`, `fit_slice`, and first fit against real CBOE slices

## Notes
- Conda env: `quant` (Python 3.11, numpy 2.4.6, scipy 1.17.1)
- Known quirk: scipy shows as `pypi_0` in `conda list` despite conda-forge install;
  functional, cosmetic only
- All Day 1–5 work pushed to `feature/week-01-bsm` branch on GitHub
- All Day 6–10 work pushed to `feature/week-02-bsm` branch on GitHub
- All Day 11–15 work pushed to `feature/week-03-dupire` branch on GitHub
- All Day 16–20 work pushed to `feature/week-04-heston` branch on GitHub
- All Day 21–25 work pushed to `feature/week-05-fourier-pricing` branch on GitHub
- Risk-free rate hardcoded at 4.5% - should pull FRED 1M T-bill rate per maturity
- **Phase 3 note**: adopt QuantLib (conda-forge `quantlib`) as the production pricing
  reference - cross-validate own pricers against it, and lean on it for the pricing layer
  in backtesting so own code focuses on portfolio/strategy logic. Build-to-learn now,
  library-in-production later. (Large C++ dependency - add deliberately when needed, not
  before.)

  