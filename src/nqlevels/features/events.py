"""Rejection-event detection — the raw material of reaction memory.

A *rejection event* is the thing Phase 2 never measured about history: not
"price visited this area" (visit counts were tested and rejected) but "price
arrived here and left fast". Concretely, for highs:

* the minute's high is the maximum over a ±``w_min`` neighbourhood;
* within the next ``confirm_min`` minutes — truncated the moment price trades
  back above the extreme, because a later excursion is not a reaction to a
  level that has already been taken out — price falls at least
  ``theta_atr`` × ATR from the extreme.

Lows mirror. The detector is a pure function of a bar frame, so the same code
runs on a full historical session (to build the memory layer) and on today's
pre-cut bars (for same-morning events) — which is what makes the same-day
features truncation-invariant by construction: they are computed *from* a
truncated tape, not filtered after the fact.

Magnitudes are normalised by the session's own pre-open ATR so a 2022 event
and a 2025 event mean the same thing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

EVENT_COLS = ["minute", "price_t", "dir", "mag_t", "mag_atr", "vel30_atr",
              "t_confirm_min", "t_max_min", "vol_x", "delta_x"]


def _local_extreme_idx(x: np.ndarray, w: int, is_high: bool) -> np.ndarray:
    """Indices that are the extreme of their ±w window, first index on ties."""
    s = pd.Series(x if is_high else -x)
    rmax = s.rolling(2 * w + 1, center=True, min_periods=1).max().to_numpy()
    cand = np.flatnonzero((x if is_high else -x) >= rmax - 1e-9)
    keep = []
    for i in cand:
        if keep and i - keep[-1] <= w and x[i] == x[keep[-1]]:
            continue                      # the same flat extreme, keep the first
        keep.append(int(i))
    return np.asarray(keep, dtype="int64")


def _one_side(bars: pd.DataFrame, atr_t: float, is_high: bool, w_min: int,
              confirm_min: int, theta_atr: float, vol_med5: float) -> list:
    hi = bars["high_t"].to_numpy("float64")
    lo = bars["low_t"].to_numpy("float64")
    vol = bars["volume"].to_numpy("float64")
    dlt = (bars["buy_vol"].to_numpy("float64") - bars["sell_vol"].to_numpy("float64"))
    n = hi.size
    need = theta_atr * atr_t
    out = []
    for i in _local_extreme_idx(hi if is_high else lo, w_min, is_high):
        px = hi[i] if is_high else lo[i]
        j0, j1 = i + 1, min(n, i + 1 + confirm_min)
        if j0 >= j1:
            continue
        if is_high:
            # stop the moment the extreme is exceeded — beyond that the area is
            # taken out and the excursion belongs to a different story
            brk = np.flatnonzero(hi[j0:j1] > px)
            j1 = j0 + int(brk[0]) if brk.size else j1
            if j0 >= j1:
                continue
            run = px - np.minimum.accumulate(lo[j0:j1])
        else:
            brk = np.flatnonzero(lo[j0:j1] < px)
            j1 = j0 + int(brk[0]) if brk.size else j1
            if j0 >= j1:
                continue
            run = np.maximum.accumulate(hi[j0:j1]) - px
        conf = np.flatnonzero(run >= need)
        if not conf.size:
            continue
        t_conf = int(conf[0]) + 1
        mag = float(run[-1])
        t_max = int(np.argmax(run)) + 1
        v30 = float(run[: min(30, run.size)].max())
        a, b = max(0, i - 2), min(n, i + 3)
        vwin = float(vol[a:b].sum())
        out.append({"minute": int(i), "price_t": float(px),
                    "dir": 1 if is_high else -1,
                    "mag_t": mag, "mag_atr": mag / atr_t,
                    "vel30_atr": v30 / atr_t,
                    "t_confirm_min": t_conf, "t_max_min": t_max,
                    "vol_x": vwin / vol_med5 if vol_med5 > 0 else np.nan,
                    "delta_x": float(dlt[a:b].sum()) / vwin if vwin > 0 else np.nan})
    return out


def detect_events(bars: pd.DataFrame, atr_t: float, w_min: int = 30,
                  confirm_min: int = 120, theta_atr: float = 0.25) -> pd.DataFrame:
    """All confirmed rejection events in one contract's bars, both directions.

    ``bars`` must be one session, one symbol, sorted by time, with columns
    ``high_t, low_t, volume, buy_vol, sell_vol``. Row position stands in for
    the minute clock; NQ trades essentially every minute inside a session, so
    the approximation is a few minutes at worst and identical across uses.
    """
    if bars is None or len(bars) < 2 * w_min + 2 or not np.isfinite(atr_t) or atr_t <= 0:
        return pd.DataFrame(columns=EVENT_COLS)
    v5 = pd.Series(bars["volume"].to_numpy("float64")).rolling(5).sum().dropna()
    vol_med5 = float(v5.median()) if len(v5) else np.nan
    rows = (_one_side(bars, atr_t, True, w_min, confirm_min, theta_atr, vol_med5)
            + _one_side(bars, atr_t, False, w_min, confirm_min, theta_atr, vol_med5))
    if not rows:
        return pd.DataFrame(columns=EVENT_COLS)
    return (pd.DataFrame(rows)[EVENT_COLS]
            .sort_values("minute").reset_index(drop=True))


# --------------------------------------------------------------------------- #
# cross-roll price mapping
# --------------------------------------------------------------------------- #


def roll_adjustment(bars_1m: pd.DataFrame, front_by_session: pd.Series,
                    min_overlap_minutes: int = 200) -> pd.DataFrame:
    """Per-session cumulative adjustment for expressing an old contract's price
    in a newer contract's scale.

    At each front-month change the spread is measured from the bars overlap:
    sessions where both contracts have bars, minute closes matched on
    timestamp, median difference, then the median over the last three overlap
    sessions before the switch. Cumulative sums give ``adj``, with
    ``price_in_scale_b = price_in_scale_a + adj[b] - adj[a]``.

    A spread that cannot be measured leaves NaN, which downstream must treat
    as "cannot carry", never as zero — the same contract as `build.roll`.
    """
    sess = front_by_session.sort_index()
    sessions = sess.index.to_numpy()
    syms = sess.to_numpy()
    change = np.flatnonzero(syms[1:] != syms[:-1]) + 1
    spreads = {}
    for k in change:
        out_sym, in_sym = syms[k - 1], syms[k]
        vals = []
        for s in sessions[max(0, k - 10): k + 3][::-1]:
            g = bars_1m[bars_1m["session"] == s]
            a = g[g["symbol"] == out_sym][["ts", "close_t"]]
            b = g[g["symbol"] == in_sym][["ts", "close_t"]]
            if len(a) < min_overlap_minutes or len(b) < min_overlap_minutes:
                continue
            m = a.merge(b, on="ts", suffixes=("_out", "_in"))
            if len(m) < min_overlap_minutes:
                continue
            vals.append(float((m["close_t_in"] - m["close_t_out"]).median()))
            if len(vals) >= 3:
                break
        spreads[k] = float(np.median(vals)) if vals else np.nan
    epoch = np.zeros(sessions.size, dtype="int64")
    for k in change:
        epoch[k:] += 1
    # a roll whose spread cannot be measured breaks the chain: sessions on
    # opposite sides of it are unmappable to each other, but mappings that stay
    # within one chain remain exact — so an unmeasurable roll costs only the
    # lookbacks that cross it, not everything after it
    adj = np.zeros(sessions.size, dtype="float64")
    chain = np.zeros(sessions.size, dtype="int64")
    cum, ch = 0.0, 0
    for k in change:
        if np.isfinite(spreads[k]):
            cum += spreads[k]
        else:
            cum, ch = 0.0, ch + 1
        adj[k:] = cum
        chain[k:] = ch
    return pd.DataFrame({"session": sessions, "symbol": syms, "epoch": epoch,
                         "chain": chain, "adj_t": adj})
