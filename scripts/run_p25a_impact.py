#!/usr/bin/env python
"""P25A Parts 3-4: impact of the causal break rule on P20-22 product
statistics. Mirrors run_p21_spacing.py / run_p22_regions.py conventions
(5-min snapshots 2025-01..2026-08-14; heavy aggregations thinned; NY-AM
= pt_min in [390,540)) with ONE lever: the zone break time b_ts.

  OLD    b_ts = tap_ts of the first breaking tap   (backdated)
  CAUSAL b_ts = first 1-second crossing of zhi + width + 20t
                (outputs/p25a_break_crossings.parquet)

Everything else (zone universe, tap counts, grids, region functions)
is identical between the two columns, so any delta is attributable to
the rule alone. Harness note: unlike run_p22_regions.py, the sealed 1m
bars are lower-bounded at 2026-06-18 here (P22 accidentally kept
2026-01..06-17 sealed bars alongside dev bars, double-weighting 2026-H1
snapshots for both rules equally).

Event metrics: resistance-side tap rows (dev <=2026-06-17, sealed
>=2026-06-18 to avoid double events); liveness at event time = b_ts >
tap_ts under each rule; region collapse per P21 = per-day overlap
components of the day's event zones (static geometry, P21 convention).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_p22_regions import regions_of, depth_profile, core_bands  # noqa

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
        b = b[(b["session"] >= max(LO, lo)) & (b["session"] <= hi)]
        adj = pd.read_parquet(f"{daily}/roll_adjust.parquet")
        adj["session"] = adj["session"].astype(str)
        b = b.merge(adj[["session", "adj_t"]], on="session", how="left")
        b["cl"] = b["close_t"] - b["adj_t"].fillna(0.0)
        its.append(it); zss.append(zs); b1s.append(b[["session", "ts",
                                                      "cl"]])
    return (pd.concat(its, ignore_index=True),
            pd.concat(zss, ignore_index=True),
            pd.concat(b1s, ignore_index=True))


def day_components(L, H):
    """Number of overlap components among a day's event zones (P21)."""
    if len(L) == 0:
        return 0
    o = np.argsort(L)
    n, cur = 1, H[o[0]]
    for i in o[1:]:
        if L[i] > cur:
            n += 1
            cur = H[i]
        else:
            cur = max(cur, H[i])
    return n


def main() -> int:
    it, zs, b1m = load()
    cx = pd.read_parquet(ROOT / "outputs" / "p25a_break_crossings.parquet")
    b_old = cx.set_index("zid")["tap_ts"].to_dict()
    b_cau = cx.set_index("zid")["cross_ts"].to_dict()
    all_sess = sorted(set(zs["session"]))
    ford_map = {s: i for i, s in enumerate(all_sess)}
    z = zs.drop_duplicates("zid").set_index("zid")
    z["ford"] = z["session"].map(ford_map)
    z["lo"] = z["price_rf"] - z["hw_t"]
    z["hi"] = z["price_rf"] + z["hw_t"]
    tap_lists = it.sort_values("tap_ts").groupby("zid")["tap_ts"] \
        .apply(lambda x: np.array(x)).to_dict()

    RULES = ("old", "cau")
    A = {r: dict(n_above=[], n_win={25: [], 50: [], 100: [], 200: []},
                 n_reg=[], reg_win={25: [], 50: [], 100: [], 200: []},
                 overlap=[], rw=[], rz=[], c2l=[], nw=[], nd=[],
                 ny=dict(nreg=[], n25=[], n100=[], nch=[], nch100=[],
                         nw=[]))
         for r in RULES}

    for sess, bars in b1m[b1m["session"] >= LO].groupby("session"):
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
        bo = np.array([b_old.get(x, np.inf) for x in zid_arr])
        bc = np.array([b_cau.get(x, np.inf) for x in zid_arr])
        taps = [tap_lists.get(x, np.empty(0)) for x in zid_arr]
        for i in range(0, len(cl), 5):
            t, px = ts_e[i], cl[i]
            ntaps = np.array([int((ta < t).sum()) for ta in taps])
            base = (F0 <= t) & (ntaps < 6)
            mpt = ((t - 8 * 3600) % 86400) // 60
            is_ny = 390 <= mpt < 540
            for rule, bts in (("old", bo), ("cau", bc)):
                acc = A[rule]
                live = base & (bts > t)
                above = live & (L0 > px)
                if above.sum() < 1:
                    continue
                L, H = L0[above], H0[above]
                o = np.argsort(L)
                Ls, Hs = L[o], H[o]
                acc["n_above"].append(int(above.sum()))
                for W in acc["n_win"]:
                    acc["n_win"][W].append(int((Ls - px <= W * 4).sum()))
                if i % 30 == 0:
                    regs = regions_of(Ls, Hs, None)
                    acc["n_reg"].append(len(regs))
                    dists = np.array([r["lo"] - px for r in regs]) / 4
                    for W in acc["reg_win"]:
                        acc["reg_win"][W].append(int((dists <= W).sum()))
                    ov = np.zeros(len(Ls), bool)
                    for a in range(len(Ls)):
                        m = (Ls < Hs[a]) & (Hs > Ls[a])
                        m[a] = False
                        ov[a] = m.any()
                    acc["overlap"].append(float(ov.mean()))
                    if is_ny:
                        acc["ny"]["nreg"].append(len(regs))
                        acc["ny"]["n25"].append(int((dists <= 25).sum()))
                        acc["ny"]["n100"].append(int((dists <= 100).sum()))
                        acc["ny"]["nch"].append(int(above.sum()))
                        acc["ny"]["nch100"].append(
                            int((Ls - px <= 400).sum()))
                    if i % 60 == 0:
                        r0 = regs[0]
                        mem = r0["members"]
                        edges, depth = depth_profile(Ls[mem], Hs[mem])
                        c2 = core_bands(edges, depth, 2)
                        acc["nw"].append((r0["hi"] - r0["lo"]) / 4)
                        acc["nd"].append((r0["lo"] - px) / 4)
                        acc["c2l"].append(c2[1] / 4)
                        if is_ny:
                            acc["ny"]["nw"].append(
                                (r0["hi"] - r0["lo"]) / 4)
                    if i % 120 == 0:
                        for r0 in regs:
                            acc["rw"].append(
                                (r0["hi"] - r0["lo"]) / 4)
                            acc["rz"].append(len(r0["members"]))

    # ---------- event / workload metrics ----------
    ev = it[it["app_side_above"] == 0].copy()
    ev = ev[ev["session"] >= LO]
    ev["b_old"] = ev["zid"].map(b_old).fillna(np.inf)
    ev["b_cau"] = ev["zid"].map(b_cau).fillna(np.inf)
    ev["lo"] = ev["zid"].map(z["lo"])
    ev["hi_e"] = ev["zid"].map(z["hi"])
    mpt = ((ev["tap_ts"] - 8 * 3600) % 86400) // 60
    ev["is_ny"] = (mpt >= 390) & (mpt < 540)
    ns = ev["session"].nunique()
    E = {}
    for rule in RULES:
        col = "b_old" if rule == "old" else "b_cau"
        lv = ev[ev[col] > ev["tap_ts"]]
        ft = lv[lv["tap_no"] == 1]
        E[rule] = dict(
            taps_day=len(lv) / ns,
            uniq_day=float(lv.groupby("session")["zid"].nunique().sum())
            / ns,
            ft_day=len(ft) / ns,
            reg_app_day=float(np.sum([day_components(
                g.drop_duplicates("zid")["lo"].to_numpy(),
                g.drop_duplicates("zid")["hi_e"].to_numpy())
                for _, g in lv.groupby("session")])) / ns,
            reg_ft_day=float(np.sum([day_components(
                g.drop_duplicates("zid")["lo"].to_numpy(),
                g.drop_duplicates("zid")["hi_e"].to_numpy())
                for _, g in ft.groupby("session")])) / ns,
            ny_ft_day=len(ft[ft["is_ny"]]) / ns,
            ny_reg_ft_day=float(np.sum([day_components(
                g.drop_duplicates("zid")["lo"].to_numpy(),
                g.drop_duplicates("zid")["hi_e"].to_numpy())
                for _, g in ft[ft["is_ny"]].groupby("session")])) / ns,
            ny_reg_app_day=float(np.sum([day_components(
                g.drop_duplicates("zid")["lo"].to_numpy(),
                g.drop_duplicates("zid")["hi_e"].to_numpy())
                for _, g in lv[lv["is_ny"]].groupby("session")])) / ns,
        )

    out = dict(sessions=int(ns))
    for rule in RULES:
        acc = A[rule]
        out[rule] = dict(
            child_above_mean=float(np.mean(acc["n_above"])),
            child_win={W: float(np.mean(v))
                       for W, v in acc["n_win"].items()},
            reg_above_mean=float(np.mean(acc["n_reg"])),
            reg_above_med=float(np.median(acc["n_reg"])),
            reg_win={W: float(np.mean(v))
                     for W, v in acc["reg_win"].items()},
            overlap_share=float(np.mean(acc["overlap"])),
            reg_width_med=float(np.median(acc["rw"])),
            reg_width_p75=float(np.percentile(acc["rw"], 75)),
            zones_per_reg=float(np.mean(acc["rz"])),
            nearest_w_med=float(np.median(acc["nw"])),
            nearest_d_med=float(np.median(acc["nd"])),
            nearest_c2_med=float(np.median(acc["c2l"])),
            raw_to_reg=float(np.mean(acc["n_above"])
                             / np.mean(acc["n_reg"])),
            ny=dict(nreg=float(np.mean(acc["ny"]["nreg"])),
                    n25=float(np.mean(acc["ny"]["n25"])),
                    n100=float(np.mean(acc["ny"]["n100"])),
                    nch=float(np.mean(acc["ny"]["nch"])),
                    nch100=float(np.mean(acc["ny"]["nch100"])),
                    nw_med=float(np.median(acc["ny"]["nw"]))),
            events=E[rule])
    (ROOT / "outputs" / "p25a_causal_break_impact.json") \
        .write_text(json.dumps(out, indent=1))

    def row(lab, o, c, f=1):
        d = c - o
        pc = (d / o * 100) if o else np.nan
        print(f"  {lab:<38} {o:>8.{f}f} {c:>8.{f}f} {d:>+8.{f}f} "
              f"{pc:>+7.1f}%")

    o, c = out["old"], out["cau"]
    print(f"=== P25A impact (2025-01..2026-08-14, {ns} sessions) ===")
    print(f"  {'metric':<38} {'OLD':>8} {'CAUSAL':>8} {'diff':>8} "
          f"{'%':>8}")
    row("child zones above price", o["child_above_mean"],
        c["child_above_mean"])
    for W in (25, 50, 100, 200):
        row(f"  children within {W}pt", o["child_win"][str(W)]
            if isinstance(list(o["child_win"])[0], str)
            else o["child_win"][W],
            c["child_win"][str(W)] if isinstance(
                list(c["child_win"])[0], str) else c["child_win"][W])
    row("regions above price", o["reg_above_mean"], c["reg_above_mean"])
    for W in (25, 50, 100, 200):
        row(f"  regions within {W}pt", o["reg_win"][W], c["reg_win"][W])
    row("overlap share (children)", o["overlap_share"] * 100,
        c["overlap_share"] * 100)
    row("region width med (pt)", o["reg_width_med"], c["reg_width_med"])
    row("zones/region", o["zones_per_reg"], c["zones_per_reg"])
    row("nearest region width med", o["nearest_w_med"],
        c["nearest_w_med"])
    row("nearest region dist med", o["nearest_d_med"],
        c["nearest_d_med"])
    row("nearest CORE-2 med", o["nearest_c2_med"], c["nearest_c2_med"])
    row("raw zones -> regions ratio", o["raw_to_reg"], c["raw_to_reg"])
    print("  --- NY-AM ---")
    row("NY-AM regions above", o["ny"]["nreg"], c["ny"]["nreg"])
    row("NY-AM regions within 25pt", o["ny"]["n25"], c["ny"]["n25"])
    row("NY-AM regions within 100pt", o["ny"]["n100"], c["ny"]["n100"])
    row("NY-AM children above", o["ny"]["nch"], c["ny"]["nch"])
    row("NY-AM nearest width med", o["ny"]["nw_med"], c["ny"]["nw_med"])
    print("  --- events/day ---")
    for k, lab in (("taps_day", "live-zone taps/day"),
                   ("uniq_day", "unique zones approached/day"),
                   ("ft_day", "first-touch child events/day"),
                   ("reg_app_day", "distinct region approaches/day"),
                   ("reg_ft_day", "distinct FT region events/day"),
                   ("ny_ft_day", "NY-AM FT child events/day"),
                   ("ny_reg_ft_day", "NY-AM FT region events/day"),
                   ("ny_reg_app_day", "NY-AM region approaches/day")):
        row(lab, o["events"][k], c["events"][k], 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
