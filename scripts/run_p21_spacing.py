#!/usr/bin/env python
"""P21: spacing/redundancy audit of the FROZEN resistance-zone map.
Causal 5-minute snapshots; edge-to-edge gaps; overlap/stack/coverage;
diagnostic clustering at fixed thresholds. No outcomes, no changes."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEAL = "/Volumes/T9/nq_levels/sealed_holdout_1/derived/daily"
DEV = "/Volumes/T9/nq_levels/daily"
LO = "2025-01-01"
THRESH = (0.0, 5.0, 10.0, 15.0, 20.0)


def load():
    its, zss, b1s = [], [], []
    for daily, lo, hi in ((DEV, "2021-01-01", "2026-06-17"),
                          (SEAL, "2026-06-18", "2026-08-14")):
        it = pd.read_parquet(f"{daily}/interactions.parquet",
                             columns=["session", "zid", "tap_ts", "tap_no",
                                      "app_side_above", "zlo_t", "zhi_t",
                                      "o_depth_max_full"])
        it["session"] = it["session"].astype(str)
        it = it[(it["session"] >= lo) & (it["session"] <= hi)]
        zs = pd.read_parquet(f"{daily}/zones.parquet",
                             columns=["session", "zid", "price_rf", "hw_t",
                                      "scale", "form_min", "form_ts_val"])
        zs["session"] = zs["session"].astype(str)
        zs = zs[(zs["session"] >= lo) & (zs["session"] <= hi)]
        if daily != DEV:
            it["zid"] += 10_000_000
            zs["zid"] += 10_000_000
        b = pd.read_parquet(f"{daily}/bars_1m.parquet",
                            columns=["session", "ts", "close_t",
                                     "is_front"])
        b = b[b["is_front"]].copy()
        b["session"] = b["session"].astype(str)
        b = b[(b["session"] >= lo) & (b["session"] <= hi)]
        adj = pd.read_parquet(f"{daily}/roll_adjust.parquet")
        adj["session"] = adj["session"].astype(str)
        b = b.merge(adj[["session", "adj_t"]], on="session", how="left")
        b["cl"] = b["close_t"] - b["adj_t"].fillna(0.0)
        its.append(it); zss.append(zs); b1s.append(b[["session", "ts", "cl"]])
    return (pd.concat(its, ignore_index=True),
            pd.concat(zss, ignore_index=True),
            pd.concat(b1s, ignore_index=True))


def clusters(lo_e, hi_e, thr_t):
    """Union count of price-sorted zones with edge gap <= thr (ticks)."""
    order = np.argsort(lo_e)
    lo_s, hi_s = lo_e[order], hi_e[order]
    n = 1
    cur_hi = hi_s[0]
    widths = []
    start = lo_s[0]
    for i in range(1, len(lo_s)):
        if lo_s[i] - cur_hi <= thr_t:
            cur_hi = max(cur_hi, hi_s[i])
        else:
            widths.append(cur_hi - start)
            n += 1
            start, cur_hi = lo_s[i], hi_s[i]
    widths.append(cur_hi - start)
    return n, np.array(widths)


def main() -> int:
    it, zs, b1m = load()
    sessions = sorted(set(b1m["session"]))
    s_ord = {s: i for i, s in enumerate(sessions)}
    z = zs.drop_duplicates("zid").set_index("zid")
    z["ford"] = z["session"].map(s_ord)
    z["lo_e"] = z["price_rf"] - z["hw_t"]
    z["hi_e"] = z["price_rf"] + z["hw_t"]
    res = it[it["app_side_above"] == 0]
    brk = res[res["o_depth_max_full"] >= (res["zhi_t"] - res["zlo_t"]) + 20]
    brk1 = brk.sort_values("tap_ts").groupby("zid")["tap_ts"].first()
    tap_lists = it.sort_values("tap_ts").groupby("zid")["tap_ts"] \
        .apply(lambda x: np.array(x)).to_dict()

    acc = dict(n_above=[], n_win={25: [], 50: [], 100: [], 200: []},
               gaps=[], gaps_n5=[[] for _ in range(5)], d1=[],
               overlap_any=[], contained=[], nb={2: [], 5: [], 10: [],
                                                 15: [], 20: [], 30: []},
               stack=[], clus={t: [] for t in THRESH},
               clus100={t: [] for t in THRESH},
               cw={t: [] for t in THRESH},
               cov={50: [], 100: [], 200: [], 500: []}, walls=[],
               clean=[], ft_n=[], ft_gaps=[], ft_clus10=[],
               nyam=dict(n_above=[], gaps=[], overlap=[], nb10=[],
                         n100=[], clus10=[]))
    prim_sessions = [s for s in sessions if s >= LO]
    for sess, bars in b1m[b1m["session"] >= LO].groupby("session"):
        so = s_ord[sess]
        cand = z[(z["ford"] <= so) & (z["ford"] >= so - 30)]
        if not len(cand):
            continue
        ts0 = bars["ts"].iloc[0].timestamp()
        cl = bars["cl"].to_numpy()
        ts_e = pd.DatetimeIndex(bars["ts"]).asi8 / 1e9
        zids = cand.index.to_numpy()
        lo_e = cand["lo_e"].to_numpy(); hi_e = cand["hi_e"].to_numpy()
        ford = cand["ford"].to_numpy()
        form_ts = cand["form_ts_val"].to_numpy()
        b_ts = np.array([brk1.get(zid, np.inf) for zid in zids])
        taps_arr = [tap_lists.get(zid, np.empty(0)) for zid in zids]
        for i in range(0, len(cl), 5):
            t = ts_e[i]
            px = cl[i]
            ntaps = np.array([int((ta < t).sum()) for ta in taps_arr])
            live = (form_ts <= t) & (b_ts > t) & (ntaps < 6)
            above = live & (lo_e > px)
            if above.sum() < 1:
                continue
            L, H = lo_e[above], hi_e[above]
            nt_a = ntaps[above]
            order = np.argsort(L)
            Ls, Hs = L[order], H[order]
            acc["n_above"].append(int(above.sum()))
            for W in acc["n_win"]:
                acc["n_win"][W].append(int((Ls - px <= W * 4).sum()))
            acc["d1"].append(float((Ls[0] - px) / 4))
            gaps = (Ls[1:] - Hs[:-1]) / 4.0 if len(Ls) > 1 else np.empty(0)
            if i % 30 == 0:                    # thin heavy aggregations
                acc["gaps"].append(gaps)
                for kk in range(min(4, len(gaps))):
                    acc["gaps_n5"][kk].append(float(gaps[kk]))
                ov = np.zeros(len(Ls), bool)
                cont = np.zeros(len(Ls), bool)
                for a in range(len(Ls)):
                    o = (Ls < Hs[a]) & (Hs > Ls[a])
                    o[a] = False
                    ov[a] = o.any()
                    cont[a] = ((Ls <= Ls[a]) & (Hs >= Hs[a]) & o).any()
                acc["overlap_any"].append(float(ov.mean()))
                acc["contained"].append(float(cont.mean()))
                for X in acc["nb"]:
                    hasnb = np.zeros(len(Ls), bool)
                    for a in range(len(Ls)):
                        g2 = np.maximum(Ls - Hs[a], Ls[a] - Hs)
                        g2[a] = 9e9
                        hasnb[a] = (g2 <= X * 4).any()
                    acc["nb"][X].append(float(hasnb.mean()))
                ev = np.concatenate([np.stack([Ls, np.ones(len(Ls))], 1),
                                     np.stack([Hs, -np.ones(len(Hs))], 1)])
                ev = ev[np.argsort(ev[:, 0], kind="stable")]
                depth = np.cumsum(ev[:, 1])
                acc["stack"].append(int(depth.max()))
                for thr in THRESH:
                    ncl, widths = clusters(Ls, Hs, thr * 4)
                    acc["clus"][thr].append(ncl)
                    acc["cw"][thr].append(widths / 4.0)
                    in100 = Ls - px <= 400
                    if in100.any():
                        n1, _ = clusters(Ls[in100], Hs[in100], thr * 4)
                        acc["clus100"][thr].append(n1)
                for W in acc["cov"]:
                    hi_w = px + W * 4
                    seg_l = np.clip(Ls, px, hi_w)
                    seg_h = np.clip(Hs, px, hi_w)
                    _, widths = clusters(seg_l[seg_h > seg_l],
                                         seg_h[seg_h > seg_l], 0.0) \
                        if (seg_h > seg_l).any() else (0, np.array([0.0]))
                    acc["cov"][W].append(float(widths.sum() / (W * 4)))
                _, walls = clusters(Ls, Hs, 0.0)
                acc["walls"].append(walls / 4.0)
                covered = np.sort(np.stack([Ls, Hs], 1), 0)
                ncl0, w0 = clusters(Ls, Hs, 0.0)
                # clean gaps within 200pts
                m200 = Ls - px <= 800
                if m200.sum() > 1:
                    g200 = (Ls[m200][np.argsort(Ls[m200])][1:]
                            - np.sort(Hs[m200])[:-1]) / 4.0
                    acc["clean"].append(g200[g200 > 0])
                # first-touch-eligible subset
                ftm = nt_a == 0
                acc["ft_n"].append(int(ftm.sum()))
                if ftm.sum() > 1:
                    Lf, Hf = Ls[np.argsort(Ls)][ftm[np.argsort(Ls)]], None
                    Lf = np.sort(L[order][ftm])
                    Hf = H[order][ftm][np.argsort(L[order][ftm])]
                    acc["ft_gaps"].append((Lf[1:] - Hf[:-1]) / 4.0)
                    n10, _ = clusters(Lf, Hf, 40.0)
                    acc["ft_clus10"].append(n10)
            # NY-AM window
            mpt = ((t - 8 * 3600) % 86400) // 60
            if 390 <= mpt < 540 and i % 30 == 0:
                ny = acc["nyam"]
                ny["n_above"].append(int(above.sum()))
                ny["n100"].append(int((Ls - px <= 400).sum()))
                if len(Ls) > 1:
                    ny["gaps"].append(gaps)
                n10, _ = clusters(Ls, Hs, 40.0)
                ny["clus10"].append(n10)

    print("=== RESULTS (2025-01..2026-08-14, 5-min snapshots; heavy "
          "aggregations every 30 min) ===")
    print(f"active above: mean {np.mean(acc['n_above']):.1f} med "
          f"{np.median(acc['n_above']):.0f}")
    for W in (25, 50, 100, 200):
        print(f"  within {W:>3}pt: mean {np.mean(acc['n_win'][W]):.1f}")
    g = np.concatenate(acc["gaps"])
    q = np.percentile(g, [10, 25, 50, 75, 90, 95])
    print(f"\nedge-to-edge gaps (all adjacent pairs, n {len(g):,}): mean "
          f"{g.mean():.1f} med {q[2]:.1f} p25 {q[1]:.1f} p75 {q[3]:.1f} "
          f"p90 {q[4]:.1f}")
    bins = [(-9e9, 0, "overlap"), (0, 2, "0-2"), (2, 5, "2-5"),
            (5, 10, "5-10"), (10, 15, "10-15"), (15, 20, "15-20"),
            (20, 30, "20-30"), (30, 50, "30-50"), (50, 75, "50-75"),
            (75, 100, "75-100"), (100, 9e9, "100+")]
    print("  " + " | ".join(f"{lab} {(100*((g>lo)&(g<=hi)).mean()):.0f}%"
                            for lo, hi, lab in bins))
    print(f"\nprice→Z1 median {np.median(acc['d1']):.1f}pt | nearest-5 "
          "gap medians: " +
          " ".join(f"Z{k+1}→Z{k+2} {np.median(acc['gaps_n5'][k]):.1f}"
                   for k in range(4) if acc["gaps_n5"][k]))
    print(f"\noverlap-any {np.mean(acc['overlap_any']):.1%} | contained "
          f"{np.mean(acc['contained']):.1%} | stack max mean "
          f"{np.mean(acc['stack']):.1f} max {np.max(acc['stack'])}")
    for X in acc["nb"]:
        print(f"  neighbor ≤{X}pt: {np.mean(acc['nb'][X]):.1%}", end="")
    print()
    print("\nclusters (mean raw → distinct | zones/cluster | med width):")
    for thr in THRESH:
        c = np.array(acc["clus"][thr])
        cw = np.concatenate(acc["cw"][thr])
        c100 = np.array(acc["clus100"][thr])
        print(f"  ≤{thr:>4.0f}pt: {np.mean(acc['n_above']):.0f} → "
              f"{c.mean():5.1f} | {np.mean(acc['n_above'])/c.mean():4.1f} "
              f"| med width {np.median(cw):5.1f} p90 "
              f"{np.percentile(cw,90):5.1f} | within100pt "
              f"{c100.mean():4.1f}")
    print("\ncoverage of next window: " +
          " ".join(f"{W}pt {np.mean(acc['cov'][W]):.0%}"
                   for W in acc["cov"]))
    walls = np.concatenate(acc["walls"])
    print(f"walls (contiguous covered runs): med {np.median(walls):.1f}pt "
          f"p75 {np.percentile(walls,75):.1f} p90 "
          f"{np.percentile(walls,90):.1f} | share ≥40pt "
          f"{(walls>=40).mean():.1%} ≥75pt {(walls>=75).mean():.1%}")
    clean = np.concatenate(acc["clean"])
    print(f"clean gaps within 200pt: med {np.median(clean):.1f} p25 "
          f"{np.percentile(clean,25):.1f} p75 {np.percentile(clean,75):.1f}"
          f" p90 {np.percentile(clean,90):.1f}")
    ftg = np.concatenate(acc["ft_gaps"]) if acc["ft_gaps"] else np.empty(0)
    print(f"\nfirst-touch-eligible: active above mean "
          f"{np.mean(acc['ft_n']):.1f} | med gap {np.median(ftg):.1f}pt | "
          f"overlap {(ftg<0).mean():.1%} | ≤10pt-clusters mean "
          f"{np.mean(acc['ft_clus10']):.1f}")
    ny = acc["nyam"]
    nyg = np.concatenate(ny["gaps"])
    print(f"\nNY-AM: active above {np.mean(ny['n_above']):.1f} | within "
          f"100pt {np.mean(ny['n100']):.1f} | med gap {np.median(nyg):.1f}"
          f" | overlap {(nyg<0).mean():.1%} | ≤10pt clusters "
          f"{np.mean(ny['clus10']):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
