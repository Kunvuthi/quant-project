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
  
## Notes
- Conda env: `quant` (Python 3.11, numpy 2.4.6, scipy 1.17.1)
- Known quirk: scipy shows as `pypi_0` in `conda list` despite conda-forge install;
  functional, cosmetic only
- All Day 1–5 work pushed to `feature/week-01-bsm` branch on GitHub
- All Day 6–10 work pushed to `feature/week-02-bsm` branch on GitHub
- All Day 11–15 work pushed to `feature/week-03-dupire` branch on GitHub
- Risk-free rate hardcoded at 4.5% - should pull FRED 1M T-bill rate per maturity
- Smile wing noise from bid-ask spreads - SVI smoothing planned for Week 3