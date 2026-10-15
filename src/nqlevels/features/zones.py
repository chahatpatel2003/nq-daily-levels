"""Zone universe for Phase 4: confirmed zigzag pivots with lifecycles.

A zone is born when a pivot is *confirmed* — the moment the reversal
threshold is crossed — never at the pivot extreme itself, which is only
identifiable in hindsight. Everything recorded at formation uses data up to
the confirmation minute and nothing later; features that grow with the
zone's life (penetration, taps, volume traded through, maximum excursion
since formation) are updated incrementally so any tap can read them
causally.

Three reversal scales stand in for "significance": a 0.75-ATR pivot is a
different structural object from a 0.15-ATR wiggle, and nesting (the same
price being a pivot at several scales) is itself a feature. Named concepts
(profile nodes, round numbers, HTF extremes) attach later as features — the
universe itself is purely mechanical.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

SCALES_ATR = (0.15, 0.35, 0.75)
BASE_MIN = 10                       # minutes of pre-pivot base
HW_FLOOR_T = 4.0
HW_CAP_ATR = 0.03                   # half-width cap, ATR fraction


def epoch_s(ts_like) -> np.ndarray:
    """Epoch seconds from any datetime-like, robust to storage resolution
    (parquet layers are ns; synthetic frames may be s)."""
    return (pd.DatetimeIndex(ts_like).as_unit("ns").asi8 // 10**9).astype("int64")


def zigzag_confirmed(hi: np.ndarray, lo: np.ndarray, threshold: float
                     ) -> List[Tuple[int, float, int, int]]:
    """Alternating pivots with their confirmation index:
    ``(i_pivot, price, +1 high / -1 low, i_confirm)``.

    Pivot indices and prices match ``features.impulses.zigzag_pivots``
    exactly (pinned by test); the confirmation index is the first bar at
    which the reversal threshold from the pivot extreme is met — the moment
    the pivot becomes knowable.
    """
    n = hi.size
    if n == 0:
        return []
    piv: List[Tuple[int, float, int, int]] = []
    up = None
    imax, imin = 0, 0
    for i in range(1, n):
        if hi[i] > hi[imax]:
            imax = i
        if lo[i] < lo[imin]:
            imin = i
        if up is None:
            if hi[i] - lo[imin] >= threshold:
                piv.append((imin, lo[imin], -1, i))
                up, imax = True, i
                for j in range(imin, i + 1):
                    if hi[j] > hi[imax]:
                        imax = j
            elif hi[imax] - lo[i] >= threshold:
                piv.append((imax, hi[imax], +1, i))
                up, imin = False, i
                for j in range(imax, i + 1):
                    if lo[j] < lo[imin]:
                        imin = j
        elif up:
            if hi[imax] - lo[i] >= threshold:
                piv.append((imax, hi[imax], +1, i))
                up, imin = False, imax
                for j in range(imax, i + 1):
                    if lo[j] < lo[imin]:
                        imin = j
        else:
            if hi[i] - lo[imin] >= threshold:
                piv.append((imin, lo[imin], -1, i))
                up, imax = True, imin
                for j in range(imin, i + 1):
                    if hi[j] > hi[imax]:
                        imax = j
    return piv


@dataclass
class Zone:
    """One zone with formation record and mutable interaction state.

    Prices are stored roll-free (own-scale minus that session's adj); the
    session engine shifts into the current contract's scale on use.
    """
    zid: int
    scale: int                       # index into SCALES_ATR
    dir: int                         # +1 pivot high, -1 pivot low
    price_rf: float                  # pivot price, roll-free
    hw_t: float                      # half width, ticks
    form_ord: int                    # session ordinal of confirmation
    form_min: int                    # minute index of confirmation
    form_ts_val: float               # epoch seconds of confirmation
    atr_form: float
    # formation features (frozen at confirmation)
    base_range_t: float
    dep_mag_t: float
    dep_vel_tpm: float
    dep_volx: float
    dep_delta: float
    broke_prior: int
    form_seg: int
    # mutable lifecycle state
    taps: int = 0
    armed: bool = True
    last_tap_ts: float = -np.inf
    max_pen_t: float = 0.0
    n_A: int = 0
    n_B: int = 0
    n_C: int = 0
    vol_through: float = 0.0
    mins_inside: float = 0.0
    max_away_t: float = 0.0
    flipped_at: Optional[float] = None    # epoch seconds of first acceptance
    retired: bool = False

    def band_rf(self) -> Tuple[float, float]:
        return self.price_rf - self.hw_t, self.price_rf + self.hw_t


def make_zones(bars_1m: pd.DataFrame, atr_t: float, adj_t: float,
               sess_ord: int, seg_of_min: np.ndarray, zid0: int,
               scales=SCALES_ATR) -> List[Zone]:
    """All zones confirmed inside one session, from its 1-minute bars.

    ``bars_1m`` is one session, one (front) contract, time-sorted. Departure
    features use only [pivot .. confirmation]; volume normalisation uses the
    trailing 3 hours before confirmation, never the whole session.
    """
    hi = bars_1m["high_t"].to_numpy("float64")
    lo = bars_1m["low_t"].to_numpy("float64")
    vol = bars_1m["volume"].to_numpy("float64")
    dlt = (bars_1m["buy_vol"].to_numpy("float64")
           - bars_1m["sell_vol"].to_numpy("float64"))
    ts = epoch_s(bars_1m["ts"]).astype("float64")
    out: List[Zone] = []
    zid = zid0
    for si, s_atr in enumerate(scales):
        piv = zigzag_confirmed(hi, lo, s_atr * atr_t)
        for k, (ip, px, d, ic) in enumerate(piv):
            a = max(0, ip - BASE_MIN)
            base = float(hi[a:ip + 1].max() - lo[a:ip + 1].min())
            hw = float(np.clip(base / 2.0, HW_FLOOR_T, HW_CAP_ATR * atr_t))
            seg_vol = vol[ip:ic + 1]
            trail = vol[max(0, ic - 180):ic + 1]
            med = float(np.median(trail)) if trail.size else np.nan
            dep = float((px - lo[ip:ic + 1].min()) if d == 1
                        else (hi[ip:ic + 1].max() - px))
            # structure break: the confirming leg took out the previous
            # opposite pivot (a knowable, strictly-prior price)
            brk = 0
            for kp in range(k - 1, -1, -1):
                pp = piv[kp]
                if pp[2] == -d:
                    if d == 1:
                        brk = int(lo[ip:ic + 1].min() < pp[1])
                    else:
                        brk = int(hi[ip:ic + 1].max() > pp[1])
                    break
            out.append(Zone(
                zid=zid, scale=si, dir=d, price_rf=px - adj_t, hw_t=hw,
                form_ord=sess_ord, form_min=ic, form_ts_val=float(ts[ic]),
                atr_form=atr_t, base_range_t=base, dep_mag_t=dep,
                dep_vel_tpm=dep / max(1, ic - ip),
                dep_volx=float(seg_vol.mean() / med) if med and med > 0 else np.nan,
                dep_delta=float(dlt[ip:ic + 1].sum() / seg_vol.sum())
                if seg_vol.sum() > 0 else np.nan,
                broke_prior=brk, form_seg=int(seg_of_min[min(ic, seg_of_min.size - 1)])))
            zid += 1
    return out
