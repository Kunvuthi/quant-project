"""Cadence experiment: does re-ranking momentum WITHIN the 5-week window help,
versus buy-and-hold? Tests weekly / fortnightly / hold on the CORE strategy only,
gross of costs. Everything is held fixed except rebalance frequency, so the
comparison isolates cadence.

Turnover control (buffer rule): at each rebalance, sell a held name only if it has
dropped out of the top BUFFER by momentum, and buy names newly in the top CORE_N to
refill to CORE_N. Names between CORE_N and BUFFER are held, not churned. This keeps
turnover realistic; rebalancing without a buffer is a strawman.

No look-ahead: each in-window rebalance re-ranks on prices up to that rebalance day
only. Gross of transaction costs (turnover is recorded so costs can be applied later).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from strategies.momentum import momentum_scores, CORE_N, LOOKBACK

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "data" / "equity_cache"
WINDOW = 25
BUFFER = 40          # hold a name until it falls out of the top 40; refill from top 25


def _equal_weight(names) -> pd.Series:
    return pd.Series(1.0 / len(names), index=list(names))


def run_cadence(prices: pd.DataFrame, rebal_step: int, window_step: int = WINDOW) -> pd.DataFrame:
    """Roll 5-week windows; within each, re-rank every rebal_step days with the
    buffer rule. rebal_step >= WINDOW means buy-and-hold (one selection at day 0).
    Records per-window total return, max drawdown, and total turnover."""
    dates = prices.index
    rows = []
    first_form = LOOKBACK
    last_form = len(dates) - WINDOW - 1

    for f in range(first_form, last_form, window_step):
        prices_to_form = prices.iloc[: f + 1]
        try:
            scores = momentum_scores(prices_to_form)
        except ValueError:
            continue
        if len(scores) < 50:
            continue

        held = list(scores.sort_values(ascending=False).index[:CORE_N])
        weights = _equal_weight(held)

        # walk the window day by day, rebalancing on the cadence
        window_dates = dates[f : f + WINDOW + 1]
        port_val = 1.0
        path = [port_val]
        turnover = 0.0

        for i in range(1, len(window_dates)):
            d_prev, d_now = window_dates[i - 1], window_dates[i]
            # daily return of the held equal-weight book (total loss on NaN)
            p_prev = prices.loc[d_prev, held]
            p_now = prices.loc[d_now, held]
            rel = (p_now / p_prev).fillna(0.0)          # NaN -> total loss that day
            # simpler: portfolio daily growth factor
            growth = float((weights * rel).sum())
            port_val *= growth
            path.append(port_val)

            # rebalance if this day is on the cadence (and not the last day)
            days_in = i
            if rebal_step < WINDOW and days_in % rebal_step == 0 and i < len(window_dates) - 1:
                scores_now = momentum_scores(prices.iloc[: f + i + 1])
                ranked = scores_now.sort_values(ascending=False)
                top_core = set(ranked.index[:CORE_N])
                top_buffer = set(ranked.index[:BUFFER])
                # keep held names still in the buffer; drop the rest
                kept = [n for n in held if n in top_buffer]
                # refill to CORE_N from the top-core names not already held
                refill = [n for n in ranked.index[:CORE_N] if n not in kept]
                new_held = (kept + refill)[:CORE_N]
                # turnover = fraction of book replaced this rebalance
                turnover += len(set(new_held) - set(held)) / CORE_N
                held = new_held
                weights = _equal_weight(held)

        total_ret = path[-1] / path[0] - 1.0
        run_max = np.maximum.accumulate(path)
        max_dd = float((np.array(path) / run_max - 1.0).min())
        rows.append({"form_date": dates[f], "total_return": float(total_ret),
                     "max_drawdown": max_dd, "turnover": float(turnover)})

    return pd.DataFrame(rows)


def main() -> None:
    prices = pd.read_parquet(CACHE / "close_panel_clean.parquet")
    cadences = {"hold": WINDOW, "fortnightly": 10, "weekly": 5}

    print(f"{'cadence':>12}  {'mean':>7}  {'median':>7}  {'std':>6}  "
          f"{'worst':>7}  {'worstDD':>7}  {'avg_turnover':>12}")
    results = {}
    for name, step in cadences.items():
        df = run_cadence(prices, rebal_step=step)
        results[name] = df
        r = df["total_return"]
        print(f"{name:>12}  {r.mean():+.3f}  {r.median():+.3f}  {r.std():.3f}  "
              f"{r.min():+.3f}  {df['max_drawdown'].min():+.3f}  "
              f"{df['turnover'].mean():>12.2f}")

    for name, df in results.items():
        df.to_parquet(CACHE / f"rebalance_{name}.parquet")
    print("\nwrote rebalance_hold, rebalance_fortnightly, rebalance_weekly")


if __name__ == "__main__":
    main()