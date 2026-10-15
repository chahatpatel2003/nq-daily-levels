#!/usr/bin/env python
"""P20 workload analytics: how many usable resistance opportunities does
the frozen zone engine produce for a human? Snapshots, counts, alerts,
overlaps, widths. No strategy metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEAL = "/Volumes/T9/nq_levels/sealed_holdout_1/derived/daily"
DEV = "/Volumes/T9/nq_levels/daily"
LO, HI = "2025-01-01", "2026-08-14"      # primary era (current regime)
SEGN = {0: "asia", 1: "london", 2: "premkt", 3: "ny-am", 4: "ny-pm"}


def load():
    its, zss, b1s = [], [], []
    for daily, lo, hi in ((DEV, "2021-01-01", "2026-06-17"),
                          (SEAL, "2026-06-18", "2026-08-14")):
        it = pd.read_parquet(f"{daily}/interactions.parquet",
                             columns=["session", "zid", "tap_sec", "tap_ts",
                                      "app_side_above", "tap_no", "zlo_t",
                                      "zhi_t", "seg", "age_min", "flipped",
                                      "o_depth_max_full"])
        it["session"] = it["session"].astype(str)
        it = it[(it["session"] >= lo) & (it["session"] <= hi)]
        if daily != DEV:
            it["zid"] = it["zid"] + 10_000_000
        its.append(it)
        zs = pd.read_parquet(f"{daily}/zones.parquet")
        zs["session"] = zs["session"].astype(str)
        zs = zs[(zs["session"] >= lo) & (zs["session"] <= hi)]
        if daily != DEV:
            zs["zid"] = zs["zid"] + 10_000_000
        zs["corpus"] = daily
        zss.append(zs)
        b = pd.read_parquet(f"{daily}/bars_1m.parquet",
                            columns=["session", "ts", "close_t", "is_front"])
        b = b[b["is_front"]].copy()
        b["session"] = b["session"].astype(str)
        b = b[(b["session"] >= lo) & (b["session"] <= hi)]
        adj = pd.read_parquet(f"{daily}/roll_adjust.parquet")
        adj["session"] = adj["session"].astype(str)
        b = b.merge(adj[["session", "adj_t"]], on="session", how="left")
        b["cl_adj"] = b["close_t"] - b["adj_t"].fillna(0.0)
        b1s.append(b[["session", "ts", "cl_adj"]])
    return (pd.concat(its, ignore_index=True),
            pd.concat(zss, ignore_index=True),
            pd.concat(b1s, ignore_index=True))


def main() -> int:
    it, zs, b1m = load()
    sessions = sorted(set(b1m["session"]))
    s_ord = {s: i for i, s in enumerate(sessions)}
    res = it[it["app_side_above"] == 0]
    prim = res[(res["session"] >= LO)]
    ns_prim = len([s for s in sessions if s >= LO])
    print(f"sessions total {len(sessions)} | primary era {ns_prim} "
          f"({LO}..)")

    # zone state table: form ord, edges, break ts (B: +5pt beyond far)
    z = zs.drop_duplicates("zid").set_index("zid")
    z["form_ord"] = z["session"].map(s_ord)
    lo_edge = z["price_rf"] - z["hw_t"]
    hi_edge = z["price_rf"] + z["hw_t"]
    # first B-acceptance per zone from resistance-side taps
    brk = res[res["o_depth_max_full"] >= (res["zhi_t"] - res["zlo_t"]) + 20]
    brk1 = brk.sort_values("tap_ts").groupby("zid")["tap_ts"].first()
    # taps per zone cumulative for liveness (approx: all taps count)
    tap_times = it.sort_values("tap_ts").groupby("zid")["tap_ts"] \
        .apply(list).to_dict()

    print("\n=== PART C/D — per-session opportunity counts (primary era) ===")
    g = prim.groupby("session")
    uniq = g["zid"].nunique()
    first = prim[prim["tap_no"] == 1].groupby("session").size()
    taps = g.size()
    live_mask = prim["flipped"] == 0
    mdo = prim[live_mask].groupby("session").size()
    for nm, s in (("unique zones approached", uniq),
                  ("first touches", first), ("all taps", taps),
                  ("manually distinct opps (unflipped taps)", mdo)):
        s = s.reindex([x for x in sessions if x >= LO]).fillna(0)
        q = np.percentile(s, [10, 25, 50, 75, 90, 95])
        print(f"  {nm:>38}: mean {s.mean():5.1f} med {q[2]:4.0f} "
              f"p10 {q[0]:3.0f} p25 {q[1]:3.0f} p75 {q[3]:3.0f} "
              f"p90 {q[4]:3.0f} p95 {q[5]:3.0f} max {s.max():4.0f}")
    m = mdo.reindex([x for x in sessions if x >= LO]).fillna(0)
    for k in (0, 1, 2, 3):
        print(f"    sessions with {k} MDO: {(m==k).mean():.1%}", end="")
    print(f" | 4+: {(m>=4).mean():.1%} | 8+: {(m>=8).mean():.1%} | "
          f"10+: {(m>=10).mean():.1%}")

    print("\n=== PART E/F — segments (primary era, MDO basis) ===")
    for sgi in range(5):
        s = prim[live_mask & (prim["seg"] == sgi)].groupby("session").size()
        s = s.reindex([x for x in sessions if x >= LO]).fillna(0)
        f1 = prim[(prim["seg"] == sgi) & (prim["tap_no"] == 1)] \
            .groupby("session").size().reindex(
            [x for x in sessions if x >= LO]).fillna(0)
        print(f"  {SEGN[sgi]:>7}: MDO mean {s.mean():4.1f} med "
              f"{s.median():3.0f} p90 {np.percentile(s,90):3.0f} "
              f"zero-days {(s==0).mean():4.1%} | first-touches "
              f"{f1.mean():4.1f}/day")

    print("\n=== PART G/H — tap number & zone age at first touch ===")
    for k in (1, 2, 3):
        n = int((prim["tap_no"] == k).sum())
        print(f"  tap{k}: {n:,} ({n/len(prim):.0%})", end="")
    n4 = int((prim["tap_no"] >= 4).sum())
    n6 = int((prim["tap_no"] >= 6).sum())
    print(f" | tap4+: {n4:,} ({n4/len(prim):.0%}) | tap6+: {n6:,}")
    ft = prim[prim["tap_no"] == 1]
    age_h = ft["age_min"] / 60.0
    for lo_, hi_, lab in ((0, 8, "same session"), (8, 23, "<1 day"),
                          (23, 46, "1-2 sess"), (46, 115, "3-5"),
                          (115, 230, "6-10"), (230, 460, "11-20"),
                          (460, 9e9, ">20")):
        m2 = (age_h >= lo_) & (age_h < hi_)
        print(f"  age {lab:>12}: {m2.mean():5.1%}", end="")
    print()

    print("\n=== PART I/J/K — snapshots: active zones, distance, alerts, overlap ===")
    snap_rows = []
    alert_stats = {x: [] for x in (10, 20, 25, 50, 75)}
    warn = {x: [] for x in (10, 20, 25, 50, 75)}
    prim_sessions = [s for s in sessions if s >= LO]
    b1m_prim = b1m[b1m["session"] >= LO]
    tapmap = prim.groupby(["session", "zid"])["tap_sec"].min()
    for sess, bars in b1m_prim.groupby("session"):
        so = s_ord[sess]
        cand = z[(z["form_ord"] <= so) & (z["form_ord"] >= so - 30)]
        if not len(cand):
            continue
        # liveness: fewer than 6 taps before this session, unbroken
        alive = []
        for zid, r in cand.iterrows():
            tt = tap_times.get(zid, [])
            sess_start_ts = bars["ts"].iloc[0].timestamp()
            n_prior = sum(1 for t in tt if t < sess_start_ts)
            if n_prior >= 6:
                continue
            b_ts = brk1.get(zid, np.inf)
            alive.append((zid, r["price_rf"] - r["hw_t"], b_ts))
        if not alive:
            continue
        edges = np.array([a[1] for a in alive])
        bts = np.array([a[2] for a in alive])
        zids = [a[0] for a in alive]
        cl = bars["cl_adj"].to_numpy()
        ts_e = pd.DatetimeIndex(bars["ts"]).asi8 / 1e9
        hourly = range(0, len(cl), 60)
        n_above, dists = [], []
        for i in hourly:
            live = (edges > cl[i]) & (bts > ts_e[i])
            n_above.append(int(live.sum()))
            if live.any():
                dists.append(float((edges[live] - cl[i]).min() / 4.0))
        seg_i = ((ts_e - ts_e[0]) // 3600).astype(int)
        snap_rows.append(dict(session=sess, start=n_above[0],
                              mean=float(np.mean(n_above)),
                              mx=int(np.max(n_above)),
                              med_dist=float(np.median(dists)) if dists
                              else np.nan,
                              dists=dists))
        # alerts at 1m resolution: first bar within X of each zone edge
        for X in alert_stats:
            xt = X * 4
            fired = 0
            for j, e in enumerate(edges):
                if bts[j] < ts_e[0]:
                    continue
                within = np.flatnonzero((cl <= e) & (cl >= e - xt))
                if not within.size:
                    continue
                fired += 1
                t_alert = ts_e[within[0]]
                tsec = tapmap.get((sess, zids[j]), None)
                if tsec is not None:
                    t_touch = ts_e[0] + tsec
                    if t_touch >= t_alert:
                        warn[X].append(t_touch - t_alert)
            alert_stats[X].append(fired)
    sn = pd.DataFrame(snap_rows)
    print(f"  active zones above price: start mean {sn['start'].mean():.1f}"
          f" | intraday mean {sn['mean'].mean():.1f} | max mean "
          f"{sn['mx'].mean():.1f}")
    alld = np.concatenate([r for r in sn["dists"] if r])
    print(f"  nearest-resistance distance (pts): med "
          f"{np.median(alld):.0f} p25 {np.percentile(alld,25):.0f} "
          f"p75 {np.percentile(alld,75):.0f} p90 "
          f"{np.percentile(alld,90):.0f}")
    for X in (10, 20, 25, 50, 75, 100, 150):
        print(f"    within {X:>3}pt: {(alld<=X).mean():.1%}", end="")
    print()
    print(f"  {'alert-X':>8} {'alerts/day':>10} {'%->touch':>9} "
          f"{'med warn':>9} {'p10':>6} {'p90':>7}")
    for X in alert_stats:
        a = np.array(alert_stats[X])
        w = np.array(warn[X])
        touch_rate = len(w) / max(1, a.sum())
        print(f"  {X:>8} {a.mean():>10.1f} {touch_rate:>9.1%} "
              f"{np.median(w)/60 if len(w) else -1:>8.0f}m "
              f"{np.percentile(w,10)/60 if len(w) else -1:>5.0f}m "
              f"{np.percentile(w,90)/60 if len(w) else -1:>6.0f}m")

    print("\n=== PART L — zone width distribution (full corpus) ===")
    w_pts = (zs["hw_t"] * 2) / 4.0
    q = np.percentile(w_pts, [10, 25, 50, 75, 90])
    print(f"  full width pts: mean {w_pts.mean():.1f} med {q[2]:.1f} "
          f"p10 {q[0]:.1f} p25 {q[1]:.1f} p75 {q[3]:.1f} p90 {q[4]:.1f} "
          f"min {w_pts.min():.1f} max {w_pts.max():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
