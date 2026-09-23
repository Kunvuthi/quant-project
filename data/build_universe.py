"""Stage 1 of the equity data pipeline: build the S&P 500/400/600 universe and
map each ticker to its Stooq on-disk filename.

Standalone and runnable. Scrapes the three S&P constituent lists from Wikipedia,
unions them into one large/mid/small-cap universe, and derives the filename each
ticker should have in the Stooq US archive. Writes universe.parquet into the
equity cache; the archive walk (stage 2) reads it to know which files to pull.

SURVIVORSHIP CAVEAT: these are CURRENT constituents. Any backtest built on this
universe only sees companies that survived to today, which flatters returns and,
more importantly, understates the catastrophic-downside tail of the small-cap
(sp600) sleeve. State this in the write-up; do not read the small-cap drawdowns
as worst-case.

Run:  python data/build_universe.py
Deps: pandas, lxml  (pip install lxml --break-system-packages if read_html fails)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

import requests
from io import StringIO

REPO_ROOT = Path(__file__).resolve().parents[1]      # data/ -> repo root
CACHE = REPO_ROOT / "data" / "equity_cache"

# Wikipedia constituent pages. (url, table_index, ticker_column) per tier.
# VERIFY these on first run: the 400/600 pages sometimes shift table index or
# name the column "Ticker symbol" rather than "Symbol". The debug print below
# shows what actually parsed so you can correct this dict if a tier looks wrong.
WIKI = {
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 0, "Symbol"),
    "sp400": ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", 0, "Symbol"),
    "sp600": ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", 0, "Symbol"),
}

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0 Safari/537.36"}

def scrape_tier(url: str, table_idx: int, col: str) -> pd.Series:
    """Fetch with a browser UA (Wikipedia 403s urllib's default), then parse.
    Raises with a column dump if the expected column is absent."""
    resp = requests.get(url, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    if table_idx >= len(tables):
        raise ValueError(f"{url}: only {len(tables)} tables found, need index {table_idx}")
    df = tables[table_idx]
    if col not in df.columns:
        raise ValueError(
            f"{url}: column {col!r} not in table {table_idx}. "
            f"Columns present: {list(df.columns)}"
        )
    return df[col].astype(str).str.strip()


def scrape_constituents(verbose: bool = True) -> pd.DataFrame:
    frames = []
    for tier, (url, idx, col) in WIKI.items():
        tickers = scrape_tier(url, idx, col)
        if verbose:
            print(f"{tier}: {len(tickers)} tickers, sample {list(tickers.head(5))}")
        frames.append(pd.DataFrame({"ticker": tickers, "tier": tier}))
    out = pd.concat(frames, ignore_index=True)
    # a ticker can legitimately appear once; if it shows in two tiers keep the first
    out = out.drop_duplicates(subset="ticker", keep="first").reset_index(drop=True)
    return out


def ticker_to_stooq_filename(ticker: str) -> str:
    """AAPL -> aapl.us.txt, BRK.B -> brk_b.us.txt, BF-B -> bf_b.us.txt.
    Class separators (. or -) become underscores IN the ticker; the market
    suffix is a literal '.us.txt' (confirmed from aaap.us.txt on disk)."""
    t = ticker.lower().replace(".", "_").replace("-", "_")
    return f"{t}.us.txt"


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    uni = scrape_constituents()
    uni["stooq_file"] = uni["ticker"].map(ticker_to_stooq_filename)

    print(f"\ntotal unique constituents: {len(uni)}")
    print(uni["tier"].value_counts().to_string())
    print("\nsample mapping:")
    print(uni.head(10).to_string(index=False))

    out_path = CACHE / "universe.parquet"
    uni.to_parquet(out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
