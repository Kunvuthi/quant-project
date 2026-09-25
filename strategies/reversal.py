"""Short-term reversal signal: rank by RECENT return, buy the LOSERS.

The mirror of momentum. Where momentum buys medium-term winners (6-month lookback,
skip last week), short-term reversal buys last week's biggest losers, betting the
sharp recent drop bounces back. Documented factor (Jegadeesh-Titman note the
1-week horizon reverses; short-term reversal literature formalises it). Same
long-only, same constructions, so it drops into the existing backtest by swapping
only the score.

No look-ahead: uses only returns up to the formation date.
"""
from __future__ import annotations
import pandas as pd

REV_LOOKBACK = 5      # ~1 week; the horizon at which reversal (not momentum) dominates


def reversal_scores(prices_to_t: pd.DataFrame) -> pd.Series:
    """Reversal score at formation t (last row). Higher score = bigger recent LOSER
    = more attractive under reversal. Score = NEGATIVE of the last-week return, so
    the worst recent performers rank highest and get bought."""
    if len(prices_to_t) < REV_LOOKBACK + 1:
        raise ValueError("not enough history for the reversal lookback")
    p_start = prices_to_t.iloc[-(REV_LOOKBACK + 1)]
    p_end = prices_to_t.iloc[-1]
    recent_return = p_end / p_start - 1.0
    return (-recent_return).dropna()      # negate: biggest losers -> highest score