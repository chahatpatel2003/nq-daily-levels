"""Displacement legs and their origin zones (supply/demand raw material).

A *leg* is a zigzag swing between confirmed pivots with reversal threshold
``thresh_atr`` × ATR. Its *origin* is the price region the move launched from:
the pivot price plus the base formed just before departure, capped so a wild
base cannot manufacture an enormous zone.

The origin of an up-leg is demand-flavoured (price left upward, fast); of a
down-leg, supply-flavoured. Whether such an origin behaves as a strong zone
*later* — and under what conditions (fresh vs mitigated, strong vs weak
displacement) — is exactly the Phase-3 question, so nothing here scores
anything; this module only detects and describes.

Confirmed legs only: the final swing to the last bar has no confirmed pivot
and is dropped. That rule holds identically on a full session and on a
pre-cut prefix, which keeps same-morning use truncation-safe.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

LEG_COLS = ["i0", "i1", "dir", "p0_t", "p1_t", "mag_t", "mag_atr", "dur_min",
            "vel_atr_h", "vol_x", "delta_share", "base_range_t", "zlo_t", "zhi_t"]


def zigzag_pivots(hi: np.ndarray, lo: np.ndarray, threshold: float) -> List[tuple]:
    """Alternating swing pivots: ``(index, price, +1 high / -1 low)``.

    A high pivot is confirmed when price falls ``threshold`` below the running
    maximum since the last low pivot; the pivot lands on the bar that set the
    maximum, not the bar that confirmed it.
    """
    n = hi.size
    if n == 0:
        return []
    piv: List[tuple] = []
    up = None                     # current swing direction, unknown at start
    imax, imin = 0, 0
    for i in range(1, n):
        if hi[i] > hi[imax]:
            imax = i
        if lo[i] < lo[imin]:
            imin = i
        if up is None:
            if hi[i] - lo[imin] >= threshold:
                piv.append((imin, lo[imin], -1))
                up, imax = True, i
                for j in range(imin, i + 1):
                    if hi[j] > hi[imax]:
                        imax = j
            elif hi[imax] - lo[i] >= threshold:
                piv.append((imax, hi[imax], +1))
                up, imin = False, i
                for j in range(imax, i + 1):
                    if lo[j] < lo[imin]:
                        imin = j
        elif up:
            if hi[imax] - lo[i] >= threshold:
                piv.append((imax, hi[imax], +1))
                up, imin = False, imax
                for j in range(imax, i + 1):
                    if lo[j] < lo[imin]:
                        imin = j
        else:
            if hi[i] - lo[imin] >= threshold:
                piv.append((imin, lo[imin], -1))
                up, imax = True, imin
                for j in range(imin, i + 1):
                    if hi[j] > hi[imax]:
                        imax = j
    return piv


def detect_impulses(bars: pd.DataFrame, atr_t: float, thresh_atr: float = 0.35,
                    base_min: int = 10, cap_atr: float = 0.06) -> pd.DataFrame:
    """Confirmed legs with origin zones, for one session of one contract.

    ``bars`` needs ``high_t, low_t, volume, buy_vol, sell_vol`` sorted by time.
    Origin zone: from the launch pivot into the pre-departure base, width
    ``min(base range, cap_atr × ATR)`` and at least one point.
    """
    if bars is None or len(bars) < base_min + 5 or not np.isfinite(atr_t) or atr_t <= 0:
        return pd.DataFrame(columns=LEG_COLS)
    hi = bars["high_t"].to_numpy("float64")
    lo = bars["low_t"].to_numpy("float64")
    vol = bars["volume"].to_numpy("float64")
    dlt = bars["buy_vol"].to_numpy("float64") - bars["sell_vol"].to_numpy("float64")
    piv = zigzag_pivots(hi, lo, thresh_atr * atr_t)
    if len(piv) < 2:
        return pd.DataFrame(columns=LEG_COLS)
    vol_pm = float(np.median(vol)) or np.nan
    cap = max(4.0, cap_atr * atr_t)
    rows = []
    for (i0, p0, s0), (i1, p1, s1) in zip(piv[:-1], piv[1:]):
        if i1 <= i0:
            continue
        d = 1 if p1 > p0 else -1
        # transit, not pivot-to-pivot: departure is the LAST touch of the
        # origin extreme, arrival the FIRST touch of the end extreme — a long
        # flat base must not make the displacement look slow
        seg_h, seg_l = hi[i0:i1 + 1], lo[i0:i1 + 1]
        if d == 1:
            dep = i0 + int(np.flatnonzero(seg_l <= p0 + 1e-9)[-1])
            arr = i0 + int(np.flatnonzero(seg_h >= p1 - 1e-9)[0])
        else:
            dep = i0 + int(np.flatnonzero(seg_h >= p0 - 1e-9)[-1])
            arr = i0 + int(np.flatnonzero(seg_l <= p1 + 1e-9)[0])
        dur = float(max(1, arr - dep))
        a = max(0, dep - base_min)
        base_hi, base_lo = float(hi[a:dep + 1].max()), float(lo[a:dep + 1].min())
        w = min(base_hi - base_lo, cap)
        w = max(w, 4.0)
        zlo, zhi = (p0, p0 + w) if d == 1 else (p0 - w, p0)
        v = float(vol[dep:arr + 1].sum())
        rows.append({"i0": int(dep), "i1": int(arr), "dir": d,
                     "p0_t": float(p0), "p1_t": float(p1),
                     "mag_t": abs(p1 - p0), "mag_atr": abs(p1 - p0) / atr_t,
                     "dur_min": dur,
                     "vel_atr_h": (abs(p1 - p0) / atr_t) / (dur / 60.0),
                     "vol_x": (v / max(dur, 1.0)) / vol_pm if vol_pm else np.nan,
                     "delta_share": float(dlt[dep:arr + 1].sum()) / v if v > 0 else np.nan,
                     "base_range_t": base_hi - base_lo,
                     "zlo_t": float(zlo), "zhi_t": float(zhi)})
    return pd.DataFrame(rows, columns=LEG_COLS)
