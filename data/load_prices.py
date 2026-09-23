"""Stage 2: walk the Stooq US archive, load the ~1500 universe names into a
single daily-close price panel, and cache it.

Reads universe.parquet (stage 1). Recurses the raw archive (handles the numbered
subfolders in nasdaq/nyse stocks and the flat nysemkt stocks folder), skips ETF
folders, parses the confirmed Stooq format (comma-delimited, YYYYMMDD dates,
.us.txt files), and writes a wide close-price panel (dates x tickers) to cache.

Split adjustment and liquidity/history screening are stage 3, not here. This
stage just gets the raw closes into one aligned frame and reports coverage.

Run:  python data/load_prices.py --raw "D:/stooq_raw/data/daily/us"
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE = REPO_ROOT / "data" / "equity_cache"

STOOQ_COLS = ["ticker", "per", "date", "time", "open", "high", "low",
              "close", "vol", "openint"]


def build_file_index(raw_root: Path) -> dict[str, Path]:
    """Map lowercase filename -> full path for every .txt under raw_root,
    skipping any path containing 'etf'. One recursive pass over the whole tree,
    so the numbered subfolders and the flat nysemkt folder are both covered."""
    index: dict[str, Path] = {}
    for p in raw_root.rglob("*.txt"):
        if "etf" in str(p).lower():
            continue
        index[p.name.lower()] = p
    return index


def load_one(path: Path) -> pd.DataFrame:
    """Parse one Stooq file -> DataFrame indexed by date with close and vol.
    Empty frame if malformed."""
    df = pd.read_csv(path, header=0, names=STOOQ_COLS, usecols=["date", "close", "vol"])
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date")[["close", "vol"]].astype(float)
    return df


def load_panel(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    uni = pd.read_parquet(CACHE / "universe.parquet")
    index = build_file_index(raw_root)
    print(f"archive: {len(index)} non-ETF .txt files indexed")

    close_series, vol_series, hits, misses = {}, {}, [], []
    for _, row in uni.iterrows():
        path = index.get(row["stooq_file"].lower())
        if path is None:
            misses.append(row["ticker"]); continue
        d = load_one(path)
        if d.empty:
            misses.append(row["ticker"]); continue
        close_series[row["ticker"]] = d["close"]
        vol_series[row["ticker"]] = d["vol"]
        hits.append(row["ticker"])

    close_panel = pd.DataFrame(close_series).sort_index()
    vol_panel = pd.DataFrame(vol_series).sort_index()
    coverage = uni.assign(found=uni["ticker"].isin(hits))
    print(f"matched {len(hits)}/{len(uni)} names; {len(misses)} misses")
    print("misses by tier:")
    print(coverage[~coverage["found"]]["tier"].value_counts().to_string())
    return close_panel, vol_panel, coverage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True,
                    help='path to unzipped .../data/daily/us')
    args = ap.parse_args()

    close_panel, vol_panel, coverage = load_panel(Path(args.raw))
    print(f"\nclose panel: {close_panel.shape}  "
          f"({close_panel.index.min().date()} to {close_panel.index.max().date()})")
    close_panel.to_parquet(CACHE / "close_panel_raw.parquet")
    vol_panel.to_parquet(CACHE / "vol_panel_raw.parquet")
    coverage.to_parquet(CACHE / "universe_coverage.parquet")
    print("wrote close_panel_raw, vol_panel_raw, universe_coverage")


if __name__ == "__main__":
    main()