import pandas as pd
from datetime import datetime
import numpy as np
from typing import Literal
import requests

CLEAN_DEFAULTS = {
    "cboe": {
        "min_volume": 5,
        "min_open_interest": 10,       # CBOE has real OI; tighter
        "max_relative_spread": 0.30,
        "max_staleness_days": 2,
    },
}

def _parse_occ_symbol(symbol: str) -> dict:
    """Parse OCC option symbol like 'SPXW260618C00200000'."""
    # The OCC format is fixed-width from the RIGHT:
    # last 8 chars  = strike × 1000
    # one char before that = C or P
    # six chars before that = YYMMDD
    # everything left over = root symbol (SPX, SPXW, AAPL, etc.)
    payload = symbol[-15:]
    root = symbol[:-15]
    
    date_str = payload[:6]
    option_type = payload[6]
    strike_int = int(payload[7:])
    
    return {
        'root': root,
        'expiry': f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:]}",
        'option_type': 'call' if option_type == 'C' else 'put',
        'strike': strike_int / 1000.0,
    }

def fetch_chain_cboe(ticker: str = '^SPX') -> tuple[pd.DataFrame, float]:
    """
    Fetch full option chain from CBOE delayed-quote JSON.
    NOTE: the returned spot is CBOE's `current_price`, which is UNRELIABLE for
    index options (no CGIF license on the free feed). Use implied_forward_from_parity
    as the canonical forward; treat this spot as a flagged diagnostic only.
    """
    cboe_ticker = ticker.lstrip('^')
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{cboe_ticker}.json"
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    resp.raise_for_status()
    data = resp.json()["data"]

    spot = float(data["current_price"])          # untrusted, see docstring
    df = pd.DataFrame(data["options"])
    parsed = df['option'].apply(_parse_occ_symbol).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    df = df.rename(columns={'open_interest': 'openInterest',
                            'last_trade_time': 'lastTradeDate'})
    return df, spot

def filter_by_expiry(
    all_options: pd.DataFrame, 
    expiry: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter a CBOE full chain to one expiry; return (calls, puts).
    """
    df = all_options[all_options['expiry'] == expiry]
    calls = df[df['option_type'] == 'call'].copy()
    puts = df[df['option_type'] == 'put'].copy()
    return calls, puts

def clean_chain(
    df: pd.DataFrame,
    source: Literal["yfinance", "cboe"] = "cboe",
    min_volume: int | None = None,
    min_open_interest: int | None = None,
    max_relative_spread: float | None = None,
    max_staleness_days: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filter an option chain for tradeable, trustworthy quotes.
    
    Returns a copy with a new 'mid' column = (bid + ask) / 2.
    """
    if source not in CLEAN_DEFAULTS:
        raise ValueError(
            f"source must be one of {list(CLEAN_DEFAULTS)}, got {source!r}"
        )

    # start from the source profile, override only explicitly-passed thresholds
    params = {**CLEAN_DEFAULTS[source]}
    overrides = {
        "min_volume": min_volume,
        "min_open_interest": min_open_interest,
        "max_relative_spread": max_relative_spread,
        "max_staleness_days": max_staleness_days,
    }
    for key, val in overrides.items():
        if val is not None:
            params[key] = val

    df = df.copy()

    if verbose: print(f"Started: {len(df)} rows")
    df = df[df['bid'] > 0]
    if verbose: print(f"After bid > 0:               {len(df)}")

    df['mid'] = (df['bid'] + df['ask']) / 2

    df = df[(df['ask'] - df['bid']) / df['mid'] <= params["max_relative_spread"]]
    if verbose: print(f"After spread filter:         {len(df)}")
    df = df[df['volume'].fillna(0) >= params["min_volume"]]
    if verbose: print(f"After volume filter:         {len(df)}")
    df = df[df['openInterest'] >= params["min_open_interest"]]
    if verbose: print(f"After open interest filter:  {len(df)}")

    df['lastTradeDate'] = pd.to_datetime(df['lastTradeDate'], utc=True)
    now = pd.Timestamp.now(tz='UTC')
    staleness_days = (now - df['lastTradeDate']).dt.total_seconds() / 86400
    df = df[staleness_days <= params["max_staleness_days"]]
    if verbose: print(f"After staleness filter:      {len(df)}")

    return df

def find_closest_expiry(
    available_expiries: list[str],
    target_days: int,
    today: pd.Timestamp | None = None,
) -> tuple[str, int]:
    """
    Find the listed expiry closest to a target number of days out.
    
    Parameters
    ----------
    available_expiries : list of str
        Expiry dates as 'YYYY-MM-DD' strings.
    target_days : int
        Desired days to expiry (e.g. 30, 60, 90).
    today : Timestamp or None
        Reference date. Defaults to current UTC date.
    
    Returns
    -------
    (expiry, actual_days) : tuple
        The chosen expiry string and the actual number of days out.
    """
    if today is None:
        today = pd.Timestamp.now(tz='UTC').normalize()
    elif today.tz is None:
        today = today.tz_localize('UTC')
    
    target = today + pd.Timedelta(days=target_days)
    available = [pd.Timestamp(e).tz_localize('UTC') for e in available_expiries]
    closest = min(available, key=lambda d: abs(d - target))
    actual_days = (closest - today).days
    return closest.strftime('%Y-%m-%d'), actual_days

def implied_vol_with_forward(price, S, K, T, r, F, option_type, bsm_implied_vol):
    """
    Invert implied vol using a parity-implied forward F instead of assuming F = S e^{rT}.
    The carry b is set so the pricer's forward matches F: F = S e^{bT} => b = ln(F/S)/T.
    """
    b = np.log(F / S) / T
    # bsm_implied_vol must accept and pass through b to bsm_price(..., b=b)
    return bsm_implied_vol(price, S, K, T, r, option_type, b=b)

def implied_forward_from_parity(calls, puts, r, T):
    """
    Extract the forward F from put-call parity across common strikes.
    F = K + e^{rT} (C - P), averaged over strikes (robust to per-strike noise).

    Parameters
    ----------
    calls, puts : DataFrame
        Cleaned chains for ONE expiry, each with 'strike' and 'mid' columns.
    r : float
        Risk-free rate (continuously compounded).
    T : float
        Time to expiry in years.

    Returns
    -------
    F : float
        Implied forward price.
    q : float
        Implied continuous dividend yield, backed out via F = S e^{(r-q)T}.
        (Returned as None here; computed by the caller who knows spot.)
    """
    # align calls and puts on common strikes
    merged = pd.merge(
        calls[['strike', 'mid']].rename(columns={'mid': 'call_mid'}),
        puts[['strike', 'mid']].rename(columns={'mid': 'put_mid'}),
        on='strike',
    )

    # per-strike forward estimate from parity
    merged['F_est'] = merged['strike'] + np.exp(r * T) * (
        merged['call_mid'] - merged['put_mid']
    )

    # robust central estimate: the parity forward is most reliable near ATM,
    # where |C - P| is large vs the spread. Use the strikes closest to where
    # C - P changes sign (the ATM-forward cross), or just take the median as
    # a noise-robust summary.
    F = merged['F_est'].median()

    return F, merged

def build_slice(
    all_options: pd.DataFrame,
    spot: float,
    target_days: int,
    r: float,
    bsm_implied_vol,
    bsm_vega,
    k_band: float = 0.4,
    max_staleness_days=None, 
) -> dict:
    """Build a fittable (k, iv, w, weights) slice for one target maturity.

    Ties the pipeline together: pick expiry, clean, parity forward, OTM
    convention, carry-consistent IV, total variance, and inverse-variance
    weights from spreads. Returns everything downstream fitters need.
    """
    expiries = sorted(all_options["expiry"].unique())
    expiry, dte = find_closest_expiry(expiries, target_days=target_days)
    T = dte / 365.0

    calls_raw, puts_raw = filter_by_expiry(all_options, expiry)
    calls = clean_chain(calls_raw, source="cboe", verbose=False, max_staleness_days=max_staleness_days)
    puts = clean_chain(puts_raw, source="cboe", verbose=False, max_staleness_days=max_staleness_days)

    F, merged = implied_forward_from_parity(calls, puts, r=r, T=T)
    if len(merged) < 5 or not np.isfinite(F):
        raise ValueError(
            f"build_slice: expiry {expiry} too illiquid, "
            f"{len(merged)} common strikes, F={F}"
        )
    b = np.log(F / spot) / T

    def side_to_kw(df, option_type):
        d = df.copy()
        d["k"] = np.log(d["strike"] / F)
        d["iv"] = d.apply(
            lambda row: bsm_implied_vol(
                price=row["mid"], S=spot, K=row["strike"], T=T, r=r,
                option_type=option_type, b=b,
            ), axis=1,
        )
        return d.dropna(subset=["iv"])

    calls_kw = side_to_kw(calls, "call")
    puts_kw = side_to_kw(puts, "put")
    otm = pd.concat([puts_kw[puts_kw["k"] < 0], calls_kw[calls_kw["k"] >= 0]]).sort_values("k")
    otm = otm[otm["k"].abs() < k_band]

    otm["iv_unc"] = ((otm["ask"] - otm["bid"]) / 2.0) / otm.apply(
        lambda row: bsm_vega(S=spot, K=row["strike"], T=T, r=r, sigma=row["iv"], b=b),
        axis=1,
    )
    otm["w"] = otm["iv"]**2 * T
    otm["w_unc"] = 2.0 * otm["iv"] * T * otm["iv_unc"]
    otm["weight"] = 1.0 / otm["w_unc"]**2

    wr = otm["weight"].to_numpy()
    fin = np.isfinite(wr)
    weights = np.clip(wr, wr[fin].max() * 1e-3, None)
    weights = weights / np.median(weights[fin])

    return {
        "k": otm["k"].to_numpy(), "iv": otm["iv"].to_numpy(),
        "w": otm["w"].to_numpy(), "weights": weights,
        "F": F, "T": T, "dte": dte, "expiry": expiry, "otm": otm,
    }
    
    