"""Run the momentum strategies over rolling 5-week windows and collect outcomes.

Non-overlapping windows are the primary comparison; overlapping (small step) is a
robustness diagnostic. For each window: form portfolios using prices up to the
formation date, then measure each portfolio's realised 5-week return and in-window
max drawdown on prices AFTER formation. A held name going NaN mid-window is treated
as a TOTAL LOSS (conservative; may slightly overstate a loss where the gap is a
data hole or a merger payout rather than a true zero, stated as a caveat).

Why the drawdown and worst-case tail matter, from the literature:
- Daniel & Moskowitz (2016), "Momentum Crashes", J. Financial Economics: momentum
  returns are negatively skewed, with infrequent but severe strings of losses that
  cluster in "panic" states (after market declines, high volatility, during
  rebounds). This is the fat left tail the concentrated satellite is expected to
  show, and the reason the study reports the full return DISTRIBUTION and the worst
  window, not just the mean.
- Barroso & Santa-Clara (2015), "Momentum Has Its Moments": momentum's risk is
  time-varying and predictable from its own realised volatility, and volatility
  scaling largely tames the crashes. This is the principled basis for the deferred
  volatility-proxy experiment (manage the satellite's variance rather than just
  measure it).

Caveat on transfer: the crash literature is long-short. The long-only strategies
here omit the short-loser leg that causes the worst documented crashes, so the
measured tail should be milder than the papers' long-short figures. The COVID
window (Feb-Mar 2020) is kept in the data as a real event; results are reported
full-sample and ex-COVID so one extreme window does not silently drive the
conclusion.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from strategies.momentum import momentum_scores, select_portfolios, LOOKBACK
from strategies.reversal import reversal_scores

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "data" / "equity_cache"
WINDOW = 25        # ~5 trading weeks


def _portfolio_path(prices_after: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Daily portfolio value over the window, starting at 1.0.
    Rebased to each held name's formation price. A name that goes NaN mid-window
    is treated as a total loss from the point it disappears (its contribution ->0)."""
    held = prices_after[weights.index]
    p0 = held.iloc[0]
    rel = held.divide(p0)                     # each name's growth factor from formation
    # total-loss-on-NaN: once a name is NaN it contributes 0 for the rest of the window
    rel_filled = rel.copy()
    for col in rel_filled.columns:
        s = rel_filled[col]
        if s.isna().any():
            first_nan = s.isna().idxmax() if s.isna().any() else None
            if first_nan is not None:
                s.loc[first_nan:] = 0.0        # zero from first NaN onward (total loss)
            rel_filled[col] = s
    port = (rel_filled * weights).sum(axis=1)  # weighted portfolio value path
    return port


def run_windows(prices: pd.DataFrame, step: int, signal_fn, signal_name: str,
                min_names: int = 30) -> pd.DataFrame:
    """Roll windows with the given step (WINDOW for non-overlapping, smaller for
    overlapping). signal_fn scores the universe at each formation date (e.g.
    momentum_scores or reversal_scores); signal_name labels the output so multiple
    signals can be compared in one table. Returns a tidy table: one row per
    (window, signal, strategy)."""
    dates = prices.index
    rows = []
    first_form = LOOKBACK                       # need LOOKBACK history before forming
    last_form = len(dates) - WINDOW - 1         # need WINDOW days after forming

    for f in range(first_form, last_form, step):
        form_date = dates[f]
        prices_to_t = prices.iloc[: f + 1]      # up to and INCLUDING formation (no leak)
        prices_after = prices.iloc[f : f + WINDOW + 1]   # formation + window ahead

        try:
            scores = signal_fn(prices_to_t)
        except ValueError:
            continue
        if len(scores) < min_names:             # too few eligible names this window
            continue
        ports = select_portfolios(scores, prices_to_t)

        for name, w in ports.items():
            path = _portfolio_path(prices_after, w)
            total_ret = path.iloc[-1] / path.iloc[0] - 1.0
            running_max = path.cummax()
            max_dd = (path / running_max - 1.0).min()
            rows.append({
                "form_date": form_date, "signal": signal_name, "strategy": name,
                "total_return": float(total_ret), "max_drawdown": float(max_dd),
            })

    return pd.DataFrame(rows)

def tail_metrics(returns: pd.Series, alpha: float = 0.05) -> dict:
    """VaR and CVaR at the alpha tail (default 5%), computed empirically from the
    window return distribution. Reported as POSITIVE loss magnitudes.

    VaR: the alpha-quantile loss, the threshold the worst alpha of windows breach.
    CVaR (Expected Shortfall): the AVERAGE loss among that worst alpha of windows,
    so it measures how deep the tail goes, not just where it starts. CVaR >= VaR
    always; it is the more informative and more principled (coherent) tail measure,
    and the one that most distinguishes the concentrated satellite from the core.
    """
    q = returns.quantile(alpha)                    # e.g. 5th-percentile return (negative)
    var = -q                                       # as a positive loss
    cvar = -returns[returns <= q].mean()           # mean of the worst-alpha tail, positive
    return {"VaR": float(var), "CVaR": float(cvar)}

def print_summary(df: pd.DataFrame, label: str) -> None:
    print(f"\n=== {label}: {df['form_date'].nunique()} windows ===")
    print(f"{'signal':>10} {'strategy':>14}  {'mean':>7}  {'median':>7}  {'std':>6}  "
          f"{'worst':>7}  {'worstDD':>7}  {'VaR5':>6}  {'CVaR5':>6}")
    for sig in ["momentum", "reversal"]:
        for strat in ["core", "satellite", "blend", "core_volsized"]:
            r = df[(df["signal"] == sig) & (df["strategy"] == strat)]["total_return"]
            dd = df[(df["signal"] == sig) & (df["strategy"] == strat)]["max_drawdown"]
            if r.empty:
                continue
            t = tail_metrics(r)
            print(f"{sig:>10} {strat:>14}  {r.mean():+.3f}  {r.median():+.3f}  {r.std():.3f}  "
                  f"{r.min():+.3f}  {dd.min():+.3f}  {t['VaR']:.3f}  {t['CVaR']:.3f}")
            
def equity_curve(nonover_df: pd.DataFrame, signal: str, strategy: str) -> pd.Series:
    """Chain consecutive NON-OVERLAPPING windows into one continuous equity curve:
    reinvest fully every 5 weeks. Returns cumulative value indexed by form_date,
    starting at 1.0. This is the 'if I ran it continuously' total-return view, as
    opposed to the distribution of independent per-window returns. Only valid on
    non-overlapping windows (overlapping ones share days and cannot be chained)."""
    sub = nonover_df[(nonover_df["signal"] == signal) &
                     (nonover_df["strategy"] == strategy)].sort_values("form_date")
    growth = 1.0 + sub["total_return"].values          # per-window growth factors
    cumulative = np.cumprod(growth)
    return pd.Series(cumulative, index=sub["form_date"].values)



def main() -> None:
    prices = pd.read_parquet(CACHE / "close_panel_clean.parquet")
    print(f"panel {prices.shape}, {prices.index.min().date()} to {prices.index.max().date()}")

    from strategies.momentum import momentum_scores
    from strategies.reversal import reversal_scores

    # both signals over IDENTICAL windows, for a fair head-to-head
    mom_no = run_windows(prices, WINDOW, momentum_scores, "momentum")
    rev_no = run_windows(prices, WINDOW, reversal_scores, "reversal")
    nonover = pd.concat([mom_no, rev_no], ignore_index=True)

    mom_ov = run_windows(prices, 5, momentum_scores, "momentum")
    rev_ov = run_windows(prices, 5, reversal_scores, "reversal")
    over = pd.concat([mom_ov, rev_ov], ignore_index=True)

    covid = (nonover["form_date"] >= "2020-02-01") & (nonover["form_date"] <= "2020-05-31")
    nonover_excovid = nonover[~covid]

    print_summary(nonover, "NON-OVERLAPPING (primary)")
    print_summary(over, "OVERLAPPING (robustness)")
    print_summary(nonover_excovid, "NON-OVERLAPPING, EX-COVID")

    # chained equity-curve total return over the whole period, per signal x strategy
    print("\n=== CHAINED EQUITY CURVE: cumulative total return, full period ===")
    print(f"{'signal':>10} {'strategy':>14}  {'final_value':>11}  {'total_return':>12}")
    for sig in ["momentum", "reversal"]:
        for strat in ["core", "satellite", "blend", "core_volsized"]:
            curve = equity_curve(nonover, sig, strat)
            if curve.empty:
                continue
            final = curve.iloc[-1]
            print(f"{sig:>10} {strat:>14}  {final:>11.2f}  {final - 1.0:>+12.2%}")

    nonover.to_parquet(CACHE / "backtest_nonoverlap.parquet")
    over.to_parquet(CACHE / "backtest_overlap.parquet")
    print("\nwrote backtest_nonoverlap, backtest_overlap")

if __name__ == "__main__":
    main()