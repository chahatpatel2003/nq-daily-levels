#!/usr/bin/env python
"""P22: presentation-layer diagnostics. Overlap-connected parent regions
over the FROZEN zones (read-only), tick-grid overlap depth, CORE bands,
composition, alert collapse. No outcomes, no engine changes, no new
price levels (bounds provably derived from child min/max)."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SEAL = "/Volumes/T9/nq_levels/sealed_holdout_1/derived/daily"
DEV = "/Volumes/T9/nq_levels/daily"
LO = "2025-01-01"


def load():
    its, zss, b1s = [], [], []
    for daily, lo, hi in ((DEV, "2021-01-01", "2026-06-17"),
                          (SEAL, "2026-06-18", "2026-08-14")):
        it = pd.read_parquet(f"{daily}/interactions.parquet",
                             columns=["session", "zid", "tap_ts", "tap_no",
                                      "app_side_above", "zlo_t", "zhi_t",
                                      "seg", "o_depth_max_full"])
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
                            columns=["session", "ts", "close_t", "is_front"])
        b = b[b["is_front"]].copy()
        b["session"] = b["session"].astype(str)
        b = b[(b["session"] >= LO) & (b["session"] <= hi)]
        adj = pd.read_parquet(f"{daily}/roll_adjust.parquet")
        adj["session"] = adj["session"].astype(str)
        b = b.merge(adj[["session", "adj_t"]], on="session", how="left")
        b["cl"] = b["close_t"] - b["adj_t"].fillna(0.0)
        its.append(it); zss.append(zs); b1s.append(b[["session", "ts", "cl"]])
    return (pd.concat(its, ignore_index=True),
            pd.concat(zss, ignore_index=True),
            pd.concat(b1s, ignore_index=True))


def regions_of(L, H, meta):
    """Overlap-connected components of intervals (transitive). Returns
    list of dicts. Proof-of-derivation: bounds are min/max of children."""
    order = np.argsort(L)
    out = []
    cur = None
    for i in order:
        if cur is None or L[i] > cur["hi"]:
            if cur is not None:
                out.append(cur)
            cur = dict(lo=L[i], hi=H[i], members=[i])
        else:
            cur["hi"] = max(cur["hi"], H[i])
            cur["members"].append(i)
        assert cur["lo"] == min(L[m] for m in cur["members"])
        assert cur["hi"] == max(H[m] for m in cur["members"])
    if cur is not None:
        out.append(cur)
    return out


def depth_profile(L, H):
    """Tick-grid overlap depth via sweep. Returns (edges, depth) where
    depth[k] holds on [edges[k], edges[k+1])."""
    ev = sorted([(x, +1) for x in L] + [(x, -1) for x in H])
    edges, depth = [], []
    d = 0
    for x, s in ev:
        if edges and x == edges[-1]:
            d += s
            depth[-1] = d
        else:
            edges.append(x)
            d += s
            depth.append(d)
    return np.array(edges), np.array(depth[:-1]) if len(depth) > 1 \
        else np.array([])


def core_bands(edges, depth, k):
    """Intervals with depth >= k: (total width, longest, count)."""
    if not len(depth):
        return 0.0, 0.0, 0
    m = depth >= k
    tot, longest, count = 0.0, 0.0, 0
    run = 0.0
    for i in range(len(depth)):
        w = edges[i + 1] - edges[i]
        if m[i]:
            run += w
            tot += w
        else:
            if run > 0:
                count += 1
                longest = max(longest, run)
            run = 0.0
    if run > 0:
        count += 1
        longest = max(longest, run)
    return tot, longest, count


def main() -> int:
    it, zs, b1m = load()
    sessions = sorted(set(b1m["session"]))
    s_ord = {s: i for i, s in enumerate(sessions)}
    all_sess = sorted(set(zs["session"]))
    ford_map = {s: i for i, s in enumerate(all_sess)}
    z = zs.drop_duplicates("zid").set_index("zid")
    z["ford"] = z["session"].map(ford_map)
    z["lo"] = z["price_rf"] - z["hw_t"]
    z["hi"] = z["price_rf"] + z["hw_t"]
    res = it[it["app_side_above"] == 0]
    brk = res[res["o_depth_max_full"] >= (res["zhi_t"] - res["zlo_t"]) + 20]
    brk1 = brk.sort_values("tap_ts").groupby("zid")["tap_ts"].first()
    tap_lists = it.sort_values("tap_ts").groupby("zid")["tap_ts"] \
        .apply(lambda x: np.array(x)).to_dict()

    R = dict(width=[], nz=[], upiv=[], maxdep=[], c2w=[], c2l=[], c2n=[],
             c3w=[], c3l=[], c3n=[], c4=[], comp=Counter(), untouched=[],
             cores=[], wide_multi={30: [0, 0], 50: [0, 0], 75: [0, 0]},
             n_regions=[], nearest=dict(d=[], w=[], c2=[], nz=[], up=[],
                                        smax=[]),
             ny=dict(nreg=[], n25=[], n50=[], n100=[], nw=[], nc2=[],
                     nz=[]))
    viol = 0
    for sess, bars in b1m.groupby("session"):
        so = ford_map.get(sess)
        if so is None:
            continue
        cand = z[(z["ford"] <= so) & (z["ford"] >= so - 30)]
        if not len(cand):
            continue
        cl = bars["cl"].to_numpy()
        ts_e = pd.DatetimeIndex(bars["ts"]).asi8 / 1e9
        zid_arr = cand.index.to_numpy()
        L0 = cand["lo"].to_numpy(); H0 = cand["hi"].to_numpy()
        F0 = cand["form_ts_val"].to_numpy()
        SC = cand["scale"].to_numpy()
        FM = cand["form_min"].to_numpy()
        SESS0 = cand["session"].to_numpy()
        b_ts = np.array([brk1.get(x, np.inf) for x in zid_arr])
        taps = [tap_lists.get(x, np.empty(0)) for x in zid_arr]
        for i in range(0, len(cl), 30):
            t, px = ts_e[i], cl[i]
            ntaps = np.array([int((ta < t).sum()) for ta in taps])
            live = (F0 <= t) & (b_ts > t) & (ntaps < 6) & (L0 > px)
            if live.sum() < 1:
                continue
            L, H = L0[live], H0[live]
            sc, fm, ss = SC[live], FM[live], SESS0[live]
            nt = ntaps[live]
            regs = regions_of(L, H, None)
            R["n_regions"].append(len(regs))
            # NY-AM window bookkeeping
            mpt = ((t - 8 * 3600) % 86400) // 60
            is_ny = 390 <= mpt < 540
            if is_ny:
                R["ny"]["nreg"].append(len(regs))
                dists = np.array([r["lo"] - px for r in regs]) / 4
                for W, key in ((25, "n25"), (50, "n50"), (100, "n100")):
                    R["ny"][key].append(int((dists <= W).sum()))
            for j, r in enumerate(regs):
                mem = r["members"]
                w = (r["hi"] - r["lo"]) / 4
                edges, depth = depth_profile(L[mem], H[mem])
                if len(depth):
                    md = int(depth.max())
                else:
                    md = 1
                c2 = core_bands(edges, depth, 2)
                c3 = core_bands(edges, depth, 3)
                c4 = core_bands(edges, depth, 4)
                if i % 120 == 0:          # thin the heavy region stats
                    R["width"].append(w)
                    R["nz"].append(len(mem))
                    # unique pivots: cluster members by (session, form_min ±10)
                    piv = set()
                    for m in mem:
                        piv.add((ss[m], int(fm[m] // 10)))
                    R["upiv"].append(len(piv))
                    R["maxdep"].append(md)
                    R["c2w"].append(c2[0] / 4); R["c2l"].append(c2[1] / 4)
                    R["c2n"].append(c2[2])
                    R["c3w"].append(c3[0] / 4); R["c3l"].append(c3[1] / 4)
                    R["c3n"].append(c3[2])
                    R["c4"].append(c4[2] > 0)
                    R["untouched"].append(int((nt[mem] == 0).sum()))
                    scset = frozenset(sc[mem])
                    R["comp"][tuple(sorted(scset))] += 1
                    ncore = c2[2]
                    R["cores"].append(ncore)
                    for Wt in R["wide_multi"]:
                        if w > Wt:
                            R["wide_multi"][Wt][0] += 1
                            if ncore >= 2:
                                R["wide_multi"][Wt][1] += 1
                    if viol == 0:
                        ok = (abs(r["lo"] - L[mem].min()) < 1e-9
                              and abs(r["hi"] - H[mem].max()) < 1e-9)
                        if not ok:
                            viol += 1
            # nearest region stats
            r0 = regs[0]
            if i % 60 == 0:
                mem = r0["members"]
                edges, depth = depth_profile(L[mem], H[mem])
                c2 = core_bands(edges, depth, 2)
                R["nearest"]["d"].append((r0["lo"] - px) / 4)
                R["nearest"]["w"].append((r0["hi"] - r0["lo"]) / 4)
                R["nearest"]["c2"].append(c2[1] / 4)
                R["nearest"]["nz"].append(len(mem))
                piv = set((ss[m], int(fm[m] // 10)) for m in mem)
                R["nearest"]["up"].append(len(piv))
                R["nearest"]["smax"].append(int(sc[mem].max()))
                if is_ny:
                    R["ny"]["nw"].append((r0["hi"] - r0["lo"]) / 4)
                    R["ny"]["nc2"].append(c2[1] / 4)
                    R["ny"]["nz"].append(len(mem))

    def q(x, ps=(10, 25, 50, 75, 90, 95)):
        x = np.array(x)
        return {p: float(np.percentile(x, p)) for p in ps}

    print("=== PART A/E/F — parent regions ===")
    print(f"regions above price: mean {np.mean(R['n_regions']):.1f} med "
          f"{np.median(R['n_regions']):.0f}")
    wq = q(R["width"])
    print(f"region width: mean {np.mean(R['width']):.1f} med {wq[50]:.1f} "
          f"p75 {wq[75]:.1f} p90 {wq[90]:.1f} p95 {wq[95]:.1f}")
    nzq = q(R["nz"])
    print(f"zones/region: mean {np.mean(R['nz']):.1f} med {nzq[50]:.0f} "
          f"p75 {nzq[75]:.0f} p90 {nzq[90]:.0f} max {max(R['nz'])}")
    hist = Counter()
    for n in R["nz"]:
        hist["1" if n == 1 else "2" if n == 2 else "3" if n == 3 else
             "4" if n == 4 else "5" if n == 5 else "6-10" if n <= 10 else
             "11-20" if n <= 20 else "20+"] += 1
    tot = sum(hist.values())
    print("  zones/region hist: " + " ".join(
        f"{k}:{hist[k]/tot:.0%}" for k in
        ("1", "2", "3", "4", "5", "6-10", "11-20", "20+")))
    print(f"unique pivots/region: mean {np.mean(R['upiv']):.1f} med "
          f"{np.median(R['upiv']):.0f} | zones:pivots ratio "
          f"{np.mean(R['nz'])/np.mean(R['upiv']):.2f}")

    print("\n=== PART B/C/D — internal confluence ===")
    print(f"max overlap depth: med {np.median(R['maxdep']):.0f} p90 "
          f"{np.percentile(R['maxdep'],90):.0f} max {max(R['maxdep'])}")
    c2p = np.array(R["c2n"]) > 0
    c3p = np.array(R["c3n"]) > 0
    print(f"CORE-2 present: {c2p.mean():.0%} | longest-band med "
          f"{np.median(np.array(R['c2l'])[c2p]):.1f}pt p75 "
          f"{np.percentile(np.array(R['c2l'])[c2p],75):.1f} | "
          f"CORE-3 present: {c3p.mean():.0%} med "
          f"{np.median(np.array(R['c3l'])[c3p]):.1f}pt | CORE-4 present: "
          f"{np.mean(R['c4']):.0%}")
    cores = np.array(R["cores"])
    print(f"cores per region: none {(cores==0).mean():.0%} one "
          f"{(cores==1).mean():.0%} two {(cores==2).mean():.0%} 3+ "
          f"{(cores>=3).mean():.0%}")
    for Wt, (a, b) in R["wide_multi"].items():
        print(f"  regions >{Wt}pt: {a:,} — multi-core {b/max(a,1):.0%}")

    print("\n=== PART G — scale composition ===")
    tot = sum(R["comp"].values())
    for k, v in sorted(R["comp"].items(), key=lambda x: -x[1]):
        print(f"  scales {k}: {v/tot:.0%}", end="")
    print()

    print("\n=== PART I — untouched zones per region ===")
    u = np.array(R["untouched"])
    print(f"untouched/region: mean {u.mean():.1f} | regions with ≥1 "
          f"untouched: {(u>0).mean():.0%}")

    print("\n=== PART M — nearest region ===")
    for k, lab in (("d", "distance"), ("w", "parent width"),
                   ("c2", "CORE-2 longest"), ("nz", "zones"),
                   ("up", "unique pivots")):
        x = np.array(R["nearest"][k])
        print(f"  {lab:>15}: med {np.median(x):.1f} p75 "
              f"{np.percentile(x,75):.1f} p90 {np.percentile(x,90):.1f}")
    smax = Counter(R["nearest"]["smax"])
    ts = sum(smax.values())
    print("  nearest-region max scale: " +
          " ".join(f"s{k}:{v/ts:.0%}" for k, v in sorted(smax.items())))

    print("\n=== PART N — NY-AM view ===")
    ny = R["ny"]
    print(f"regions above price: mean {np.mean(ny['nreg']):.1f} med "
          f"{np.median(ny['nreg']):.0f}")
    for key, lab in (("n25", "within 25pt"), ("n50", "50pt"),
                     ("n100", "100pt")):
        print(f"  {lab}: mean {np.mean(ny[key]):.1f}")
    print(f"  nearest width med {np.median(ny['nw']):.1f} p90 "
          f"{np.percentile(ny['nw'],90):.1f} | nearest CORE-2 med "
          f"{np.median(ny['nc2']):.1f} | zones/nearest med "
          f"{np.median(ny['nz']):.0f}")
    print(f"\nPART P — derivation violations: {viol} (bounds provably "
          "min/max of children; cores from coverage sweep only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
