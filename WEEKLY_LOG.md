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

## Notes
- Conda env: `quant` (Python 3.11, numpy 2.4.6, scipy 1.17.1)
- Known quirk: scipy shows as `pypi_0` in `conda list` despite conda-forge install;
  functional, cosmetic only
- All Day 1–5 work pushed to `feature/week-01-bsm` branch on GitHub
- Risk-free rate hardcoded at 4.5% - should pull FRED 1M T-bill rate per maturity
- SPX dividend yield (~1.3%) not modelled - affects forward, hence IV inversion
- Smile wing noise from bid-ask spreads - SVI smoothing planned for Week 3
- `sys.path.append('..')` still in notebooks - convert to `pyproject.toml` editable install when it becomes annoying
- Per-source data hygiene config (yfinance vs CBOE filter defaults) could be extracted from function signature into a config dict - small refactor opportunity