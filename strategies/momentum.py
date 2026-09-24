"""Momentum signal and the three portfolio constructions (core, satellite, blend).

Signal: trailing return over [t-LOOKBACK, t-SKIP], cross-sectionally ranked; the
top names are bought (long-only, per the competition ruleset). Skipping the most
recent SKIP days avoids the short-term reversal that would contaminate the signal.
Selection uses ONLY data strictly before the formation date, so there is no
look-ahead.

Literature:
- Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers",
  J. Finance. The seminal cross-sectional momentum result: past 3-12 month
  winners keep outperforming losers over the following months. Basis for the
  ranking signal and the ~6-month lookback.
- Jegadeesh & Titman (2001), "Profitability of Momentum Strategies": much of the
  momentum return accrues in the first few months after formation, which supports
  a ~5-week holding window capturing the signal rather than being too short.
- Carhart (1997): momentum as the 4th factor alongside market, size, value.
   Establishes it as a recognised factor instead of just a curiosity.
- Asness, Moskowitz & Pedersen (2013), "Value and Momentum Everywhere": the
  effect generalises across markets and asset classes.

IMPORTANT scope difference from the literature: the papers above study LONG-SHORT
momentum (buy winners, short losers). This competition is LONG-ONLY, so only the
winner leg is held. The long-only version has a milder risk profile than the
academic long-short strategy, because the short-loser leg (which drives the worst
momentum crashes, see backtest.py) is never held. Read results accordingly.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# Lookback and holding choices follow the momentum literature:
# 3-12 month formation works (Jegadeesh & Titman 1993); a ~6-month lookback sits
# in that range and suits a 5-week hold. SKIP drops the most recent week to avoid
# the well-documented short-term reversal at the 1-week horizon.
LOOKBACK = 126     # ~6 trading months
SKIP = 5           # ~1 week, dropped to avoid short-term reversal
CORE_N = 25        # diversified core basket (diversification tempers momentum crash risk)
SAT_N = 4          # concentrated satellite (deliberately high variance)
SAT_WEIGHT = 0.20  # satellite fraction of the blend
VOL_LOOKBACK = 63     # ~3 months of daily returns for the volatility estimate


def _inverse_vol_weights(prices_to_t: pd.DataFrame, names) -> pd.Series:
    """Inverse-volatility weights for `names`, from each stock's own daily-return
    std over the last VOL_LOOKBACK days (pre-formation, no look-ahead). Steadier
    names get more weight. Normalised to sum to 1. Grounded in Barroso &
    Santa-Clara (2015): scaling by own realised vol tames momentum's risk."""
    recent = prices_to_t.iloc[-(VOL_LOOKBACK + 1):][list(names)]
    daily_ret = recent.pct_change().dropna(how="all")
    vol = daily_ret.std()
    vol = vol.replace(0.0, np.nan)                  # guard: a zero-vol name would blow up
    inv = 1.0 / vol
    inv = inv.fillna(inv.median())                  # a name lacking vol data gets median weight
    return inv / inv.sum()


def momentum_scores(prices_to_t: pd.DataFrame) -> pd.Series:
    """Momentum score per eligible stock at formation date t (the last row).
    prices_to_t: close panel with rows up to and INCLUDING t, columns = tickers.
    Score = return over [t-LOOKBACK, t-SKIP]. Names lacking data at either
    endpoint are dropped (NaN score -> excluded)."""
    if len(prices_to_t) < LOOKBACK + 1:
        raise ValueError("not enough history before t for the momentum lookback")
    p_start = prices_to_t.iloc[-(LOOKBACK + 1)]      # price at t-LOOKBACK
    p_end = prices_to_t.iloc[-(SKIP + 1)]            # price at t-SKIP
    score = p_end / p_start - 1.0
    return score.dropna()


def select_portfolios(scores: pd.Series, prices_to_t: pd.DataFrame) -> dict[str, pd.Series]:
    """Build the four portfolios as weight Series from momentum scores.
    core: top CORE_N equal-weighted.
    satellite: top SAT_N equal-weighted (concentrated).
    blend: (1-SAT_WEIGHT)*core + SAT_WEIGHT*satellite.
    core_volsized: SAME top CORE_N names, inverse-volatility weighted, so the ONLY
      difference from `core` is the weighting scheme (isolates that one variable)."""
    ranked = scores.sort_values(ascending=False)
    core_names = ranked.index[:CORE_N]
    sat_names = ranked.index[:SAT_N]

    core = pd.Series(1.0 / len(core_names), index=core_names)
    sat = pd.Series(1.0 / len(sat_names), index=sat_names)

    blend = (1 - SAT_WEIGHT) * core.reindex(core.index.union(sat_names), fill_value=0.0)
    blend = blend.add(SAT_WEIGHT * sat.reindex(blend.index, fill_value=0.0), fill_value=0.0)

    core_volsized = _inverse_vol_weights(prices_to_t, core_names)

    return {"core": core, "satellite": sat, "blend": blend,
            "core_volsized": core_volsized}