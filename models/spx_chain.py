import os
import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np

# spx = yf.Ticker("^SPX")
# expirations = spx.options
# print(f"Number of expirations available: {len(expirations)}")
# print(f"First 10 expirations: {expirations[:10]}")
# print(f"Spot S = {spx.info.get('regularMarketPrice', 'N/A')}")

# today = datetime(2026, 6, 16)
# target = pd.Timestamp(today) + pd.Timedelta(days=30)
# print(f"Target ~30 days from today: {target.date()}")

# # Find closest available expiry
# available = [pd.Timestamp(e) for e in expirations]
# closest = min(available, key=lambda d: abs(d - target))
# print(f"Closest available: {closest.date()} ({(closest - pd.Timestamp(today)).days})")

def fetch_chain(ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch option chain for a given ticker and expiry.
    
    Parameters
    ----------
    ticker : str
        Yahoo ticker symbol (e.g. '^SPX').
    expiry : str
        Expiry date in 'YYYY-MM-DD' format. Must be in ticker.options.
    
    Returns
    -------
    (calls, puts) : tuple of DataFrames
        Raw option chain as returned by yfinance, no cleaning applied.
    """
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)
    return chain.calls, chain.puts

def fetch_chain_cached(
    ticker: str, 
    expiry: str, 
    cache_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch with file-based caching. Re-fetches only if file doesn't exist."""
    if cache_dir is None:
        # Default to <repo_root>/data/raw, independent of caller's CWD
        repo_root = Path(__file__).parent.parent  # spx_chain.py -> models/ -> repo
        cache_dir = repo_root / 'data' / 'raw'
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    calls_path = cache_dir / f"{ticker.lstrip('^')}_{expiry}_calls.csv"
    puts_path  = cache_dir / f"{ticker.lstrip('^')}_{expiry}_puts.csv"
    
    if calls_path.exists() and puts_path.exists():
        print(f"Loading from cache: {cache_dir}")
        return pd.read_csv(calls_path), pd.read_csv(puts_path)
    
    print("Fetching live...")
    calls, puts = fetch_chain(ticker, expiry)
    calls.to_csv(calls_path, index=False)
    puts.to_csv(puts_path, index=False)
    return calls, puts

def clean_chain(
    df: pd.DataFrame,
    min_volume: int = 10,
    min_open_interest: int = 1,
    max_relative_spread: float = 0.5,
    max_staleness_days: int = 2,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filter an option chain for tradeable, trustworthy quotes.
    
    Returns a copy with a new 'mid' column = (bid + ask) / 2.
    """
    df = df.copy()
    
   
    
    if verbose: print(f"Started: {len(df)} rows")
    df = df[df['bid'] > 0]
    if verbose: print(f"After bid > 0:               {len(df)}")
    
    df['mid'] = (df['bid'] + df['ask']) / 2
    
    df = df[(df['ask'] - df['bid']) / df['mid']  <= max_relative_spread]
    if verbose: print(f"After spread filter:         {len(df)}")
    df = df[df['volume'].fillna(0) >= min_volume]
    if verbose: print(f"After volume filter:         {len(df)}")
    df = df[df['openInterest'] >= min_open_interest]
    if verbose: print(f"After open interest filter:  {len(df)}")
    
    df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate'], utc=True)
    now = pd.Timestamp.now(tz='UTC')
    staleness_days = (now - df['lastTradeDate']).dt.total_seconds() / 86400
    df = df[staleness_days <= max_staleness_days]
    if verbose: print(f"After staleness filter:      {len(df)}")
    
    return df
    


        
    
    