#!/usr/bin/env python
"""Displacement-origin zones, tracked to each later morning.

Two passes:

1. Detect confirmed legs on every full session tape (`features/impulses.py`)
   and store their origin zones with strength features (magnitude, velocity,
   volume, delta, base) plus the same-session revisit depth after departure.
2. For every measured morning: collect origins from the prior ``--lookback``
   sessions in the same roll chain, express them in the morning's contract
   scale, compute age and mitigation (how deeply any later session — or this
   morning's overnight, from pre-open context — re-entered the zone), snap the
   zone centre to the measured 1-point grid, and keep the strongest origin per
   grid price. Same-morning origins are detected on pre-cut bars only.

Nothing here reads an outcome. The output joins axis_measured on
(session, off_t) so the honest grid machinery does all scoring.
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
from nqlevels.features.impulses import detect_impulses                         # noqa: E402
from nqlevels.io.raw import assert_not_raw                                     # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402

CONFIGS = str(ROOT / "configs")
STEP_T = 4.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=40)
    ap.add_argument("--thresh-atr", type=float, default=0.35)
    a = ap.parse_args()
    t0 = _time.time()

    cal = load_calendar(CONFIGS)
    d0 = layer_dir("daily", CONFIGS)
    ctx = pd.read_parquet(d0 / "context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    ctxi = ctx.set_index("session")
    adj = pd.read_parquet(d0 / "roll_adjust.parquet").set_index("session")
    ax = pd.read_parquet(d0 / "axis_measured.parquet", columns=["session", "off_t"])
    bars = pd.read_parquet(d0 / "bars_1m.parquet")
    bars["ts"] = pd.DatetimeIndex(bars["ts"]).tz_convert("UTC")
    fb_by = {s: g.sort_values("ts").reset_index(drop=True)
             for s, g in bars[bars["is_front"]].groupby("session", sort=False)}

    order = {s: i for i, s in enumerate(sorted(ctxi.index))}
    n_ord = len(order)

    # ---- pass 1: legs per session, session extremes ------------------------
    legs_all, ext = [], {}
    for s, g in fb_by.items():
        if s not in ctxi.index:
            continue
        atr = float(ctxi.loc[s, "atr14_t"])
        ext[order[s]] = (float(g["high_t"].max()), float(g["low_t"].min()))
        L = detect_impulses(g, atr, thresh_atr=a.thresh_atr)
        if L.empty:
            continue
        hi = g["high_t"].to_numpy("float64")
        lo = g["low_t"].to_numpy("float64")
        # same-session revisit after the leg arrived: how deep price came back
        # into the origin zone before the day ended
        pr = np.empty(len(L))
        for k, r in enumerate(L.itertuples()):
            j = int(r.i1) + 1
            if r.dir == 1:
                pr[k] = max(0.0, r.zhi_t - lo[j:].min()) if j < lo.size else 0.0
            else:
                pr[k] = max(0.0, hi[j:].max() - r.zlo_t) if j < hi.size else 0.0
        L = L.assign(post_rev_t=pr, session=s, ord=order[s], atr_t=atr,
                     adj_t=float(adj.loc[s, "adj_t"]),
                     chain=int(adj.loc[s, "chain"]))
        legs_all.append(L)
    legs = pd.concat(legs_all, ignore_index=True).sort_values("ord").reset_index(drop=True)
    print(f"legs: {len(legs):,} over {legs['session'].nunique()} sessions "
          f"(median {legs.groupby('session').size().median():.0f}/session), "
          f"{_time.time()-t0:.0f}s", flush=True)

    ext_hi = np.full(n_ord, np.nan)
    ext_lo = np.full(n_ord, np.nan)
    for k, (h, l) in ext.items():
        ext_hi[k], ext_lo[k] = h, l
    adj_by_ord = np.full(n_ord, np.nan)
    for s, i in order.items():
        if s in adj.index:
            adj_by_ord[i] = float(adj.loc[s, "adj_t"])
    hi_adj = ext_hi - adj_by_ord            # roll-free scale
    lo_adj = ext_lo - adj_by_ord

    legs_ord = legs["ord"].to_numpy("int64")

    # ---- pass 2: origins alive at each measured morning --------------------
    out = []
    sessions = sorted(ax["session"].unique())
    grid_by = {s: np.sort(g["off_t"].to_numpy("float64"))
               for s, g in ax.groupby("session", sort=False)}
    for i, s in enumerate(sessions):
        m = order[s]
        ref = float(ctxi.loc[s, "ref_t"])
        atr = float(ctxi.loc[s, "atr14_t"])
        adj_s, chain_s = float(adj.loc[s, "adj_t"]), int(adj.loc[s, "chain"])
        on_hi = float(ctxi.loc[s, "globex_high_pre"])
        on_lo = float(ctxi.loc[s, "globex_low_pre"])
        grid = grid_by[s]
        gmax = grid.max() + STEP_T

        a0, a1 = np.searchsorted(legs_ord, m - a.lookback), np.searchsorted(legs_ord, m)
        W = legs.iloc[a0:a1]
        W = W[W["chain"] == chain_s]
        rows = []
        # running minima/maxima of later-session extremes, in roll-free scale,
        # so mitigation between origin and today is two array lookups
        if len(W):
            lo_min = np.full(n_ord + 1, np.inf)
            hi_max = np.full(n_ord + 1, -np.inf)
            for k in range(m - 1, max(-1, m - a.lookback - 1), -1):
                lo_min[k] = min(lo_min[k + 1], lo_adj[k]) if np.isfinite(lo_adj[k]) else lo_min[k + 1]
                hi_max[k] = max(hi_max[k + 1], hi_adj[k]) if np.isfinite(hi_adj[k]) else hi_max[k + 1]
            for r in W.itertuples():
                shift = adj_s - r.adj_t
                zlo, zhi = r.zlo_t + shift, r.zhi_t + shift
                k = int(r.ord) + 1
                low_aft = (lo_min[k] + adj_s) if k <= m - 1 else np.inf
                high_aft = (hi_max[k] + adj_s) if k <= m - 1 else -np.inf
                low_aft = min(low_aft, on_lo)
                high_aft = max(high_aft, on_hi)
                if r.dir == 1:
                    mit = max(0.0, zhi - low_aft)
                else:
                    mit = max(0.0, high_aft - zlo)
                mit = max(mit, float(r.post_rev_t))
                width = zhi - zlo
                mid_off = (zlo + zhi) / 2.0 - ref
                if abs(mid_off) > gmax:
                    continue
                # a zone covers every grid price inside it, not just its centre
                base = {"orig_dir": int(r.dir), "orig_mag": float(r.mag_atr),
                        "orig_vel": float(r.vel_atr_h), "orig_volx": float(r.vol_x),
                        "orig_delta": float(r.delta_share),
                        "orig_age": int(m - r.ord),
                        "orig_mit_t": float(mit), "orig_width_t": float(width),
                        "orig_fresh": int(mit <= 0.0),
                        "orig_consumed": int(mit >= width),
                        "orig_intact": int((r.dir == 1) == (mid_off < 0)),
                        "orig_sameday": 0}
                g0 = np.ceil((zlo - ref) / STEP_T) * STEP_T
                g1 = np.floor((zhi - ref) / STEP_T) * STEP_T
                cov = np.arange(g0, g1 + STEP_T, STEP_T) if g1 >= g0 else \
                    np.array([np.round(mid_off / STEP_T) * STEP_T])
                for off in cov:
                    if abs(off) > gmax:
                        continue
                    rows.append({**base, "off_t": float(off)})
        # same-morning origins from pre-cut bars
        gb = fb_by.get(s)
        if gb is not None:
            cut = pd.Timestamp(decision_time_utc(s, cal))
            pre = gb[gb["ts"] < cut].reset_index(drop=True)
            L0 = detect_impulses(pre, atr, thresh_atr=a.thresh_atr)
            if len(L0):
                lo0 = pre["low_t"].to_numpy("float64")
                hi0 = pre["high_t"].to_numpy("float64")
                for r in L0.itertuples():
                    j = int(r.i1) + 1
                    if r.dir == 1:
                        mit = max(0.0, r.zhi_t - lo0[j:].min()) if j < lo0.size else 0.0
                    else:
                        mit = max(0.0, hi0[j:].max() - r.zlo_t) if j < hi0.size else 0.0
                    width = r.zhi_t - r.zlo_t
                    mid_off = (r.zlo_t + r.zhi_t) / 2.0 - ref
                    if abs(mid_off) > gmax:
                        continue
                    base = {"orig_dir": int(r.dir), "orig_mag": float(r.mag_atr),
                            "orig_vel": float(r.vel_atr_h), "orig_volx": float(r.vol_x),
                            "orig_delta": float(r.delta_share),
                            "orig_age": 0, "orig_mit_t": float(mit),
                            "orig_width_t": float(width),
                            "orig_fresh": int(mit <= 0.0),
                            "orig_consumed": int(mit >= width),
                            "orig_intact": int((r.dir == 1) == (mid_off < 0)),
                            "orig_sameday": 1}
                    g0 = np.ceil((r.zlo_t - ref) / STEP_T) * STEP_T
                    g1 = np.floor((r.zhi_t - ref) / STEP_T) * STEP_T
                    cov = np.arange(g0, g1 + STEP_T, STEP_T) if g1 >= g0 else \
                        np.array([np.round(mid_off / STEP_T) * STEP_T])
                    for off in cov:
                        if abs(off) > gmax:
                            continue
                        rows.append({**base, "off_t": float(off)})
        if not rows:
            continue
        f = pd.DataFrame(rows)
        f["session"] = s
        # per grid price: a live (fresh) origin outranks a stronger dead one,
        # then strength decides; count the rest
        f = f.sort_values(["orig_fresh", "orig_mag"], ascending=False)
        n_at = f.groupby("off_t").size().rename("orig_n")
        f = f.drop_duplicates(subset=["off_t"], keep="first").merge(
            n_at, left_on="off_t", right_index=True)
        out.append(f)
        if i % 200 == 0:
            print(f"  {s}: {sum(len(x) for x in out):,} origin-prices, "
                  f"{_time.time()-t0:.0f}s", flush=True)

    o = pd.concat(out, ignore_index=True)
    dst = assert_not_raw(d0 / "origins_at.parquet")
    o.to_parquet(dst, index=False, compression="zstd")
    per = o.groupby("session").size()
    print(f"\norigins_at: {len(o):,} rows over {o['session'].nunique()} sessions "
          f"(median {per.median():.0f} origin-prices/morning) | fresh share "
          f"{float(o['orig_fresh'].mean()):.3f} | consumed {float(o['orig_consumed'].mean()):.3f} | "
          f"same-day share {float(o['orig_sameday'].mean()):.3f} | "
          f"{dst.stat().st_size/1e6:.1f} MB, {(_time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
