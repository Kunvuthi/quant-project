"""Load a real SPX chain into the canonical (8 x 11) IV surface the network expects.

Reuses the Phase-1 pipeline: build_slice already delivers a clean (k, iv) smile per
maturity in the network's coordinates (log-moneyness vs the parity-implied forward,
OTM convention, carry-consistent IV). This layer only resamples each smile onto the
canonical LOG_MONEYNESS grid and stacks across the canonical MATURITIES, flagging any
grid node that required extrapolation beyond the market's quoted range.
"""
from __future__ import annotations
import numpy as np

from calibration.deep_cal.surface import MATURITIES, LOG_MONEYNESS, SURFACE_SHAPE
from data.option_chain import build_slice
from calibration.implied_vol import bsm_implied_vol
from models.bsm import bsm_vega


def load_spx_surface(
    all_options,
    spot: float,
    r: float,
    k_band: float = 0.4,
    max_staleness_days: int = 7,
) -> dict:
    """Build the canonical surface + a per-node valid mask from a raw CBOE chain.

    Returns dict with:
      surface       (8, 11) market IVs on the canonical grid (NaN where a whole
                    maturity slice failed to build)
      market_valid  (8, 11) bool: True where the node is INTERPOLATED from real
                    quotes, False where extrapolated beyond the market k-range or
                    the maturity slice was unavailable
      forwards      (8,) parity forward per canonical maturity (NaN if slice failed)
      dtes, expiries per maturity for the record
    """
    surface = np.full(SURFACE_SHAPE, np.nan)
    market_valid = np.zeros(SURFACE_SHAPE, dtype=bool)
    forwards = np.full(MATURITIES.size, np.nan)
    expiries = [None] * MATURITIES.size
    dtes = np.full(MATURITIES.size, np.nan)

    for i, T in enumerate(MATURITIES):
        target_days = round(T * 365)
        try:
            sl = build_slice(all_options, spot=spot, target_days=target_days, r=r,
                             bsm_implied_vol=bsm_implied_vol, bsm_vega=bsm_vega,
                             k_band=k_band, max_staleness_days=max_staleness_days)
        except ValueError:
            # slice too illiquid to build; leave this maturity row NaN/invalid
            continue
        
        # GUARD
        if abs(sl["dte"] - target_days) > 0.15 * target_days:
            continue # snapped too far; do not mislabel this smile onto this grid row

        k_mkt, iv_mkt = sl["k"], sl["iv"]
        order = np.argsort(k_mkt)
        k_mkt, iv_mkt = k_mkt[order], iv_mkt[order]
        if k_mkt.size < 3:
            continue

        # interpolate the smile onto the canonical log-moneyness nodes
        iv_row = np.interp(LOG_MONEYNESS, k_mkt, iv_mkt)   # flat-extrapolates past ends
        # a node is TRUSTWORTHY only if it lies within the market's quoted k-range
        in_range = (LOG_MONEYNESS >= k_mkt.min()) & (LOG_MONEYNESS <= k_mkt.max())

        surface[i] = iv_row
        market_valid[i] = in_range
        forwards[i] = sl["F"]
        expiries[i] = sl["expiry"]
        dtes[i] = sl["dte"]

    return {
        "surface": surface,
        "market_valid": market_valid,
        "forwards": forwards,
        "expiries": expiries,
        "dtes": dtes,
    }