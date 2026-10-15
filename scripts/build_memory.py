#!/usr/bin/env python
"""Reaction-memory features for every measured axis price.

For each (session, grid price): the decayed history of rejection events near
that price over the prior ``--lookback`` sessions, in the current contract's
scale via the roll adjustment table, plus the same morning's overnight events
recomputed from pre-cut bars only (never read from the stored layer, so the
same-day columns are truncation-invariant by construction).

Bands are ATR-scaled: 0.02 ATR (~5-7 points) and 0.04 ATR. Features per band:

  mem{b}_n        raw event count in band, prior sessions
  mem{b}_w        decayed count (half-life ``--half-life`` sessions)
  mem{b}_magw     decayed sum of event magnitudes (ATR units)
  mem{b}_magmax   largest single event magnitude
  mem{b}_velmax   fastest 30-minute excursion
  mem{b}_last     sessions since the most recent event
  mem{b}_wup/wdn  decayed count split by event direction
  mem{b}_both     events from both directions exist
  mem0{b}_n       today's overnight events in band (confirmed pre-cut)
  mem0{b}_mag     largest overnight event magnitude, measured pre-cut only
  mem0{b}_recent  minutes from the latest overnight event's extreme to the cut

plus mem_cov (sessions actually available in the window) and mem_xroll
(window crosses a contract roll — carry error is possible there).
"""
from __future__ import annotations

import argparse
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.clock import decision_time_utc, load_calendar                    # noqa: E402
from nqlevels.features.events import detect_events                             # noqa: E402
from nqlevels.io.raw import assert_not_raw                                     # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402

CONFIGS = str(ROOT / "configs")
BANDS_ATR = {"2": 0.02, "4": 0.04}


def _slice_stats(ev_off: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                 val: np.ndarray, how: str) -> np.ndarray:
    """Aggregate ``val`` over [lo[i], hi[i]) slices of a sorted event array."""
    if how == "sum":
        c = np.concatenate([[0.0], np.cumsum(val)])
        return c[hi] - c[lo]
    if ev_off.size == 0:
        return np.full(lo.size, np.nan)
    # sentinel pad so hi == size is a legal reduceat index and the final
    # element is never silently dropped from a slice
    vp = np.concatenate([val, [-np.inf]])
    idx = np.stack([lo, hi], axis=1).ravel()
    red = np.maximum.reduceat(vp, idx)[::2]
    return np.where(hi > lo, red, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--half-life", type=float, default=20.0)
    a = ap.parse_args()
    t0 = _time.time()

    cal = load_calendar(CONFIGS)
    d = layer_dir("daily", CONFIGS)
    ctx = pd.read_parquet(d / "context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    ctxi = ctx.set_index("session")
    ev = pd.read_parquet(d / "events.parquet")
    adj = pd.read_parquet(d / "roll_adjust.parquet").set_index("session")
    ax = pd.read_parquet(d / "axis_measured.parquet",
                         columns=["session", "off_t"])
    bars = pd.read_parquet(d / "bars_1m.parquet")
    bars["ts"] = pd.DatetimeIndex(bars["ts"]).tz_convert("UTC")
    fb_by = {s: g for s, g in bars[bars["is_front"]].groupby("session", sort=False)}

    # chronological ordinals over every session with context (event windows are
    # counted in sessions, not calendar days)
    order = {s: i for i, s in enumerate(sorted(ctxi.index))}
    ev = ev.sort_values("session").reset_index(drop=True)
    ev["ord"] = ev["session"].map(order)
    ev["adj_t"] = ev["session"].map(adj["adj_t"])
    ev["chain"] = ev["session"].map(adj["chain"])
    ev_ord = ev["ord"].to_numpy("int64")

    out = []
    sessions = sorted(ax["session"].unique())
    ax_by = {s: g for s, g in ax.groupby("session", sort=False)}
    for i, s in enumerate(sessions):
        g = ax_by[s]
        off = g["off_t"].to_numpy("float64")
        j = order[s]
        ref = float(ctxi.loc[s, "ref_t"])
        atr = float(ctxi.loc[s, "atr14_t"])
        adj_s, chain_s = float(adj.loc[s, "adj_t"]), int(adj.loc[s, "chain"])
        f = {"session": np.repeat(s, off.size), "off_t": off}

        a0, a1 = np.searchsorted(ev_ord, j - a.lookback), np.searchsorted(ev_ord, j)
        W = ev.iloc[a0:a1]
        W = W[W["chain"] == chain_s]
        f["mem_cov"] = np.repeat(min(j, a.lookback), off.size)
        f["mem_xroll"] = np.repeat(int(W["adj_t"].nunique() > 1) if len(W) else 0,
                                  off.size)
        if len(W):
            e_off = (W["price_t"].to_numpy("float64")
                     + adj_s - W["adj_t"].to_numpy("float64")) - ref
            wgt = 0.5 ** ((j - W["ord"].to_numpy("float64")) / a.half_life)
            mag = W["mag_atr"].to_numpy("float64")
            vel = W["vel30_atr"].to_numpy("float64")
            eup = (W["dir"].to_numpy() == 1).astype("float64")
            eord = W["ord"].to_numpy("float64")
            o = np.argsort(e_off, kind="mergesort")
            e_off, wgt, mag, vel, eup, eord = (x[o] for x in
                                               (e_off, wgt, mag, vel, eup, eord))
        for tag, b_atr in BANDS_ATR.items():
            band = b_atr * atr
            if len(W):
                lo = np.searchsorted(e_off, off - band)
                hi = np.searchsorted(e_off, off + band)
                n = (hi - lo).astype("float64")
                f[f"mem{tag}_n"] = n
                f[f"mem{tag}_w"] = _slice_stats(e_off, lo, hi, wgt, "sum")
                f[f"mem{tag}_magw"] = _slice_stats(e_off, lo, hi, wgt * mag, "sum")
                f[f"mem{tag}_magmax"] = _slice_stats(e_off, lo, hi, mag, "max")
                f[f"mem{tag}_velmax"] = _slice_stats(e_off, lo, hi, vel, "max")
                last = _slice_stats(e_off, lo, hi, eord, "max")
                f[f"mem{tag}_last"] = np.where(n > 0, j - last, np.nan)
                wup = _slice_stats(e_off, lo, hi, wgt * eup, "sum")
                wdn = _slice_stats(e_off, lo, hi, wgt * (1 - eup), "sum")
                f[f"mem{tag}_wup"] = wup
                f[f"mem{tag}_wdn"] = wdn
                f[f"mem{tag}_both"] = ((wup > 0) & (wdn > 0)).astype("float64")
            else:
                for c in ("n", "w", "magw", "wup", "wdn", "both"):
                    f[f"mem{tag}_{c}"] = np.zeros(off.size)
                for c in ("magmax", "velmax", "last"):
                    f[f"mem{tag}_{c}"] = np.full(off.size, np.nan)

        # ---- today's overnight events, from pre-cut bars only --------------
        cut = pd.Timestamp(decision_time_utc(s, cal))
        gb = fb_by.get(s)
        pre = gb[gb["ts"] < cut].sort_values("ts").reset_index(drop=True) \
            if gb is not None else None
        e0 = detect_events(pre, atr) if pre is not None and len(pre) else \
            pd.DataFrame()
        for tag, b_atr in BANDS_ATR.items():
            band = b_atr * atr
            if len(e0):
                o0 = (e0["price_t"].to_numpy("float64") - ref)
                so = np.argsort(o0, kind="mergesort")
                o0 = o0[so]
                m0 = e0["mag_atr"].to_numpy("float64")[so]
                mn0 = e0["minute"].to_numpy("float64")[so]
                lo = np.searchsorted(o0, off - band)
                hi = np.searchsorted(o0, off + band)
                n0 = (hi - lo).astype("float64")
                f[f"mem0{tag}_n"] = n0
                f[f"mem0{tag}_mag"] = _slice_stats(o0, lo, hi, m0, "max")
                rec = _slice_stats(o0, lo, hi, mn0, "max")
                f[f"mem0{tag}_recent"] = np.where(n0 > 0, len(pre) - rec, np.nan)
            else:
                f[f"mem0{tag}_n"] = np.zeros(off.size)
                for c in ("mag", "recent"):
                    f[f"mem0{tag}_{c}"] = np.full(off.size, np.nan)
        out.append(pd.DataFrame(f))
        if i % 200 == 0:
            print(f"  {s}: {sum(len(x) for x in out):,} rows, "
                  f"{_time.time()-t0:.0f}s", flush=True)

    m = pd.concat(out, ignore_index=True)
    for c in m.columns:
        if m[c].dtype == "float64" and c != "off_t":
            m[c] = m[c].astype("float32")
    dst = assert_not_raw(d / "memory.parquet")
    m.to_parquet(dst, index=False, compression="zstd")
    nz = float((m["mem2_n"] > 0).mean())
    print(f"\nmemory: {len(m):,} rows, {m.shape[1]} cols | "
          f"share with any 0.02-ATR memory {nz:.3f} | "
          f"overnight-event share {float((m['mem02_n'] > 0).mean()):.4f} | "
          f"{dst.stat().st_size/1e6:.0f} MB, {(_time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
