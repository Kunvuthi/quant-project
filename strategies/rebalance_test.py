"""Cadence experiment, generalised to any signal and construction. Gross of costs.
Within each 5-week window, re-rank on the given signal every rebal_step days with
the buffer rule. Answers whether frequent rebalancing rescues a fast-decaying signal
(reversal) that underperforms when held. NOTE: gross only; high turnover here is
cost-fatal and manually impractical live, so a positive gross result is a research
finding, not a usable strategy."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from strategies.momentum import momentum_scores, select_portfolios, CORE_N, LOOKBACK
from strategies.reversal import reversal_scores
from strategies.combo import combo_scores

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "data" / "equity_cache"
WINDOW = 25
BUFFER_MULT = 1.6      # hold a name until it falls out of top (basket_size * BUFFER_MULT)

SIGNALS = {"momentum": momentum_scores, "reversal": reversal_scores, "combo": combo_scores}


def _rebalanced_path(prices, f, weights0, held0, signal_fn, rebal_step, basket_size):
    """Walk one window day by day; re-rank on signal_fn every rebal_step days with a
    buffer. Returns (return, max_drawdown, total_turnover). Total loss on NaN."""
    dates = prices.index
    window_dates = dates[f: f + WINDOW + 1]
    held, weights = list(held0), weights0.copy()
    val, path, turnover = 1.0, [1.0], 0.0
    buffer_n = int(round(basket_size * BUFFER_MULT))

    for i in range(1, len(window_dates)):
        d_prev, d_now = window_dates[i - 1], window_dates[i]
        rel = (prices.loc[d_now, held] / prices.loc[d_prev, held]).fillna(0.0)  # NaN=total loss
        val *= float((weights * rel).sum())
        path.append(val)

        if rebal_step < WINDOW and i % rebal_step == 0 and i < len(window_dates) - 1:
            scores = signal_fn(prices.iloc[: f + i + 1])
            ranked = scores.sort_values(ascending=False)
            top_basket = list(ranked.index[:basket_size])
            top_buffer = set(ranked.index[:buffer_n])
            kept = [n for n in held if n in top_buffer]
            refill = [n for n in top_basket if n not in kept]
            new_held = (kept + refill)[:basket_size]
            turnover += len(set(new_held) - set(held)) / basket_size
            held = new_held
            weights = pd.Series(1.0 / len(held), index=held)

    run_max = np.maximum.accumulate(path)
    return path[-1] - 1.0, float((np.array(path) / run_max - 1.0).min()), turnover


def run_all(prices, rebal_step, window_step=WINDOW):
    """All signals x {core, satellite} at the given rebalance cadence. (Blend and
    core_volsized skipped here: blend is a fixed mix not a single re-rankable basket,
    and volsizing needs re-weighting logic; core and satellite are the clean cases
    for the cadence question.)"""
    dates = prices.index
    rows = []
    for f in range(LOOKBACK, len(dates) - WINDOW - 1, window_step):
        for sig_name, sig_fn in SIGNALS.items():
            try:
                scores = sig_fn(prices.iloc[: f + 1])
            except ValueError:
                continue
            if len(scores) < CORE_N + 5:
                continue
            ranked = scores.sort_values(ascending=False)
            for strat, size in [("core", CORE_N), ("satellite", 4)]:
                held0 = list(ranked.index[:size])
                w0 = pd.Series(1.0 / size, index=held0)
                ret, dd, to = _rebalanced_path(prices, f, w0, held0, sig_fn, rebal_step, size)
                rows.append({"form_date": dates[f], "signal": sig_name, "strategy": strat,
                             "total_return": ret, "max_drawdown": dd, "turnover": to})
    return pd.DataFrame(rows)


def main():
    prices = pd.read_parquet(CACHE / "close_panel_clean.parquet")
    cadences = {"hold": WINDOW, "fortnightly": 10, "weekly": 5, "twice_weekly": 2}

    for cad_name, step in cadences.items():
        df = run_all(prices, step)
        print(f"\n=== cadence: {cad_name} (step {step}) ===")
        print(f"{'signal':>10} {'strategy':>10}  {'mean':>7}  {'std':>6}  "
              f"{'worst':>7}  {'avg_turnover':>12}")
        for sig in ["momentum", "reversal", "combo"]:
            for strat in ["core", "satellite"]:
                s = df[(df.signal == sig) & (df.strategy == strat)]
                if s.empty:
                    continue
                r = s["total_return"]
                print(f"{sig:>10} {strat:>10}  {r.mean():+.3f}  {r.std():.3f}  "
                      f"{r.min():+.3f}  {s['turnover'].mean():>12.2f}")


if __name__ == "__main__":
    main()