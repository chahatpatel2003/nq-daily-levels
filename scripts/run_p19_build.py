#!/usr/bin/env python
"""P19: build the resistance→support flip/retest dataset from FROZEN
layers (dev + opened holdout). One row per (zone, flip-definition,
above-side tap after acceptance), with the frozen sweep outcomes and
paired shadow keys. No engine changes; original band unchanged."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEAL = "/Volumes/T9/nq_levels/sealed_holdout_1/derived/daily"
DEV = "/Volumes/T9/nq_levels/daily"
OUT = "/Volumes/T9/nq_levels/daily/p19_flips.parquet"


def load(daily, lo, hi):
    it = pd.read_parquet(f"{daily}/interactions.parquet",
                         columns=["session", "zid", "tap_sec", "tap_ts",
                                  "app_side_above", "tap_no", "zlo_t",
                                  "zhi_t", "atr_t", "flipped",
                                  "o_depth_max_full", "o_dwell_beyond_full",
                                  "o_first_beyond", "o_n_fwd"])
    p6 = pd.read_parquet(f"{daily}/p6_real.parquet")
    for d in (it, p6):
        d["session"] = d["session"].astype(str)
    k = ["session", "zid", "tap_sec"]
    keep = [c for c in p6.columns if c.startswith("t_rev")
            or c.startswith("depth_before") or c == "depth_max_full"]
    m = it.drop_duplicates(k).merge(p6.drop_duplicates(k)[k + keep],
                                    on=k, how="left")
    m = m[(m["session"] >= lo) & (m["session"] <= hi)]
    return m


def onem_close_above(zone_taps, b1m_ts, b1m_cl_adj):
    """FLIP A: first 1m close above far edge within 2 sessions of the
    first-beyond moment of the earliest clearing tap."""
    r = zone_taps.iloc[0]
    t0 = r["tap_ts"] + (r["o_first_beyond"] if np.isfinite(r["o_first_beyond"]) else 0)
    t1 = t0 + 2 * 86400
    i0, i1 = np.searchsorted(b1m_ts, [t0, t1])
    seg = b1m_cl_adj[i0:i1]
    hit = np.flatnonzero(seg > r["zhi_t"])
    return b1m_ts[i0 + hit[0]] if hit.size else np.nan


def main() -> int:
    frames = [load(DEV, "2021-01-01", "2026-06-17"),
              load(SEAL, "2026-06-18", "2026-08-14")]
    it = pd.concat(frames, ignore_index=True) \
        .sort_values(["zid", "tap_ts"]).reset_index(drop=True)
    it["width_t"] = it["zhi_t"] - it["zlo_t"]
    print(f"taps: {len(it):,} | zones: {it['zid'].nunique():,} | "
          f"above-side taps: {(it['app_side_above']==1).sum():,}")

    # global adjusted front 1m closes for FLIP A
    b1m = []
    for daily, lo, hi in ((DEV, "2021-01-01", "2026-06-17"),
                          (SEAL, "2026-06-18", "2026-08-14")):
        b = pd.read_parquet(f"{daily}/bars_1m.parquet",
                            columns=["session", "ts", "close_t",
                                     "is_front"])
        b = b[b["is_front"]].copy()
        b["session"] = b["session"].astype(str)
        b = b[(b["session"] >= lo) & (b["session"] <= hi)]
        adj = pd.read_parquet(f"{daily}/roll_adjust.parquet")
        adj["session"] = adj["session"].astype(str)
        b = b.merge(adj[["session", "adj_t"]], on="session", how="left")
        b["cl_adj"] = b["close_t"] - b["adj_t"].fillna(0.0)
        b1m.append(b[["ts", "cl_adj"]])
    b1m = pd.concat(b1m).sort_values("ts")
    b1m_ts = pd.DatetimeIndex(b1m["ts"]).asi8 // 10**9
    b1m_cl = b1m["cl_adj"].to_numpy()
    print(f"1m closes: {len(b1m_ts):,}")

    rows = []
    for zid, g in it.groupby("zid", sort=False):
        g = g.sort_values("tap_ts")
        res = g[g["app_side_above"] == 0]
        sup = g[g["app_side_above"] == 1]
        if not len(sup):
            continue
        w = g["width_t"].iloc[0]
        # break taps per definition (first qualifying resistance-side tap)
        cleared = res[res["o_depth_max_full"] > w]
        defs = {}
        defs["B"] = res[res["o_depth_max_full"] >= w + 20]
        defs["C"] = res[res["o_depth_max_full"] >= w + 40]
        defs["D"] = res[res["o_dwell_beyond_full"] >= 60]
        defs["E"] = sup[sup["flipped"] == 1]     # engine state at tap
        first_a = np.nan
        if len(cleared):
            first_a = onem_close_above(cleared, b1m_ts, b1m_cl)
        for dname in ("A", "B", "C", "D", "E"):
            if dname == "A":
                if not np.isfinite(first_a):
                    continue
                acc_ts = first_a
            elif dname == "E":
                ss = defs["E"]
                if not len(ss):
                    continue
                acc_ts = float(ss["tap_ts"].iloc[0]) - 1   # state already on
            else:
                dd = defs[dname]
                if not len(dd):
                    continue
                r0 = dd.iloc[0]
                acc_ts = float(r0["tap_ts"]) + (
                    float(r0["o_first_beyond"])
                    if np.isfinite(r0["o_first_beyond"]) else 0.0) + 1
            rts = sup[sup["tap_ts"] > acc_ts]
            for j, (_, rt) in enumerate(rts.iterrows(), start=1):
                if j > 6:
                    break
                rows.append(dict(zid=zid, definition=dname,
                                 sup_tap_no=j, acc_ts=acc_ts,
                                 session=rt["session"],
                                 tap_sec=rt["tap_sec"],
                                 tap_ts=rt["tap_ts"],
                                 width_t=w, atr_t=rt["atr_t"],
                                 dt_break=rt["tap_ts"] - acc_ts,
                                 **{c: rt[c] for c in rt.index
                                    if c.startswith("t_rev")
                                    or c.startswith("depth_before")
                                    or c == "depth_max_full"}))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    print(f"flip-retest rows: {len(df):,} → {OUT}")
    for dname, g in df[df["sup_tap_no"] == 1].groupby("definition"):
        print(f"  {dname}: flipped zones with ≥1 retest: {len(g):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
