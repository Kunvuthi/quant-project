"""Stage 3: turn the raw close/volume panels into a clean, backtest-ready panel.

Steps, in order:
  1. Trim to the study window (2016 on): most of the 1500 names have data here;
     the 1962 start is a handful of old names and mostly NaN.
  2. Split-adjust close per name: detect day-over-day ratios near a canonical
     split fraction, back-adjust prices before the split so the series is
     continuous. Clean integer-ratio splits are auto-adjusted; ambiguous large
     jumps are flagged for manual review, NOT auto-adjusted, so a real crash is
     never silently erased (that would delete the downside tail we care about).
  3. Screen out names with insufficient history in the window or too-thin
     liquidity (low average dollar volume), using the volume panel.
  4. Write close_panel_clean.parquet + split_report + flagged_for_review.

Run:  python data/clean_panel.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "data" / "equity_cache"

START = "2016-01-01"
CANONICAL = np.array([2, 3, 4, 5, 6, 7, 8, 10,           # reverse splits (ratio up)
                      1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/10])  # forward (ratio down)
RATIO_TOL = 0.02          # within 2% of an exact canonical ratio -> a split
JUMP_FLAG = 0.35          # |1 - ratio| beyond this but not clean -> flag for review
MIN_DAYS = 750            # ~3 trading years of history required in the window
MIN_DOLLAR_VOL = 1e6      # median daily dollar volume floor (liquidity screen)


def detect_and_adjust(close: pd.Series) -> tuple[pd.Series, list, list]:
    """Return (adjusted_close, splits_applied, flags). Walks the series, finds
    canonical-ratio jumps, back-adjusts prior prices. Ambiguous jumps flagged."""
    c = close.dropna()
    if len(c) < 2:
        return close, [], []
    ratio = c / c.shift(1)
    adj = c.copy()
    splits, flags = [], []

    for dt, r in ratio.items():
        if not np.isfinite(r) or r <= 0:
            continue
        near = CANONICAL[np.argmin(np.abs(CANONICAL - r))]
        if abs(r - near) / near <= RATIO_TOL:
            # clean split: back-adjust everything strictly before dt by r
            adj.loc[adj.index < dt] *= r
            splits.append((dt, round(float(r), 4), round(float(near), 4)))
        elif abs(1 - r) >= JUMP_FLAG:
            # large but not clean: could be a real crash/pop, flag don't touch
            flags.append((dt, round(float(r), 4)))

    return adj.reindex(close.index), splits, flags


def main() -> None:
    close = pd.read_parquet(CACHE / "close_panel_raw.parquet").loc[START:]
    vol = pd.read_parquet(CACHE / "vol_panel_raw.parquet").loc[START:]
    print(f"window {START} on: {close.shape[0]} dates, {close.shape[1]} names")

    adj_cols, split_rows, flag_rows = {}, [], []
    for tkr in close.columns:
        adj, splits, flags = detect_and_adjust(close[tkr])
        adj_cols[tkr] = adj
        for dt, r, near in splits:
            split_rows.append({"ticker": tkr, "date": dt, "ratio": r, "canonical": near})
        for dt, r in flags:
            flag_rows.append({"ticker": tkr, "date": dt, "ratio": r})

    adj_close = pd.DataFrame(adj_cols)

    # --- screens ---
    enough_history = adj_close.notna().sum() >= MIN_DAYS
    dollar_vol = (close * vol).median()                 # median daily $ volume per name
    liquid = dollar_vol >= MIN_DOLLAR_VOL
    keep = enough_history & liquid
    clean = adj_close.loc[:, keep[keep].index]

    print(f"dropped {int((~enough_history).sum())} for short history, "
          f"{int((~liquid).sum())} for low liquidity "
          f"(overlap counted once in the {int((~keep).sum())} total dropped)")
    print(f"clean panel: {clean.shape}")
    print(f"splits auto-adjusted: {len(split_rows)}   flagged for review: {len(flag_rows)}")

    clean.to_parquet(CACHE / "close_panel_clean.parquet")
    pd.DataFrame(split_rows).to_parquet(CACHE / "split_report.parquet")
    pd.DataFrame(flag_rows).to_parquet(CACHE / "flagged_for_review.parquet")
    print("wrote close_panel_clean, split_report, flagged_for_review")


if __name__ == "__main__":
    main()
    import pandas as pd, numpy as np
    f = pd.read_parquet("data/equity_cache/flagged_for_review.parquet")
    print(len(f), "flags")
    print(f["ratio"].describe())
    # how many flagged ratios are suspiciously close to a canonical split?
    canon = np.array([2,3,4,0.5,1/3,0.25,0.2])
    f["nearest"] = f["ratio"].apply(lambda r: canon[np.argmin(np.abs(canon-r))])
    f["dist"] = (f["ratio"]-f["nearest"]).abs()/f["nearest"]
    print("flagged ratios within 5% of a canonical split:", int((f["dist"]<0.05).sum()))
    print(f.sort_values("dist").head(15).to_string())