"""Conditional momentum-reversal combination.

Momentum is the dominant signal (established here: it beats pure reversal on every
metric), so it decides ELIGIBILITY: take a wide pool of the top momentum names.
Reversal is the weaker refinement, used only to pick ENTRY within that pool: among
the momentum winners, prefer the ones that dipped last week over the ones that
spiked, since a just-spiked name is the most likely to reverse against you next week.

This is not a blend of two equal signals (which would dilute momentum with the
weaker reversal). Momentum selects who is eligible; reversal picks which of the
eligible to actually buy. No look-ahead: both use only data up to formation.
"""
from __future__ import annotations
import pandas as pd
from strategies.momentum import momentum_scores
from strategies.reversal import reversal_scores

MOM_POOL = 50      # momentum eligibility pool (wider than the final basket of CORE_N)


def combo_scores(prices_to_t: pd.DataFrame) -> pd.Series:
    """Combined score: only the top MOM_POOL momentum names are eligible (all others
    get -inf so they never rank), and within that pool the score IS the reversal
    score (biggest recent losers rank highest). select_portfolios then takes the top
    CORE_N/SAT_N of these, i.e. the biggest dips among the momentum winners."""
    mom = momentum_scores(prices_to_t)
    rev = reversal_scores(prices_to_t)                 # higher = bigger recent loser

    pool = mom.sort_values(ascending=False).index[:MOM_POOL]   # momentum-eligible
    # score within the pool = reversal; everything outside the pool is excluded
    score = rev.reindex(mom.index)                     # align to the momentum universe
    score = score[score.index.isin(pool)]              # keep only eligible names
    return score.dropna()