#!/usr/bin/env python
"""P19 analysis: does broken-and-accepted resistance act as support on
first retest? Stages C-O per the registered plan. LEVEL research only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEAL = "/Volumes/T9/nq_levels/sealed_holdout_1/derived/daily"
DEV = "/Volumes/T9/nq_levels/daily"

BLOCKS = [("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
          ("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31"),
          ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-06-17"),
          ("2026-06-18", "2026-08-14")]


def pX_beforeY(d, X, Y):
    t = d[f"t_rev{X}"]
    dep = d[f"depth_before_rev{X}"]
    return float((t.notna() & (dep <= 4 * Y)).mean())


def react_row(d, label, show=True):
    out = {}
    for X in (5, 10, 15, 20, 30):
        out[f"P+{X}"] = float(d[f"t_rev{X}"].notna().mean())
    for T, lab in ((60, "1m"), (300, "5m"), (900, "15m")):
        mfe = np.zeros(len(d))
        for X in (5, 10, 15, 20, 30):
            mfe = np.where(d[f"t_rev{X}"].fillna(9e9) <= T, X, mfe)
        out[f"medMFE{lab}"] = float(np.median(mfe))
    out["P10<300"] = float((d["t_rev10"] <= 300).mean())
    out["med_t10"] = float(d["t_rev10"].median())
    out["P+10b4-10"] = pX_beforeY(d, 10, 10)
    out["P+20b4-10"] = pX_beforeY(d, 20, 10)
    if show:
        print(f"  {label:>26}: n {len(d):6,} " +
              " ".join(f"P+{X} {out[f'P+{X}']:.2f}" for X in (5, 10, 20, 30))
              + f" | P(+10<5m) {out['P10<300']:.2f} med_t10 "
              f"{out['med_t10']:.0f}s | MFE5m {out['medMFE5m']:.0f} | "
              f"+10b4-10 {out['P+10b4-10']:.2f}")
    return out


def main() -> int:
    df = pd.read_parquet("/Volumes/T9/nq_levels/daily/p19_flips.parquet")
    first = df[df["sup_tap_no"] == 1]

    print("=== STAGE C/F — first-retest reactions per registered definition ===")
    for dname in ("A", "B", "C", "D", "E"):
        react_row(first[first["definition"] == dname], f"FLIP {dname}")
    # baseline: ALL above-side taps (incl. non-flipped) for contrast
    frames = []
    for daily, lo, hi in ((DEV, "2021-01-01", "2026-06-17"),
                          (SEAL, "2026-06-18", "2026-08-14")):
        p6 = pd.read_parquet(f"{daily}/p6_real.parquet")
        p6["session"] = p6["session"].astype(str)
        p6 = p6[(p6["session"] >= lo) & (p6["session"] <= hi)]
        frames.append(p6[p6["side_above"] == 1])
    allsup = pd.concat(frames, ignore_index=True)
    react_row(allsup, "ALL above-side taps")

    print("\n=== STAGE D — exact location: real first retests vs PAIRED shadows ===")
    sh = []
    for daily, lo, hi in ((DEV, "2021-01-01", "2026-06-17"),
                          (SEAL, "2026-06-18", "2026-08-14")):
        s = pd.read_parquet(f"{daily}/p6_shadow.parquet")
        s["session"] = s["session"].astype(str)
        s = s[(s["session"] >= lo) & (s["session"] <= hi)
              & (s["side_above"] == 1)]
        sh.append(s)
    sh = pd.concat(sh, ignore_index=True)
    k = ["session", "zid", "tap_sec"]
    for dname in ("B", "D", "A"):
        f1 = first[first["definition"] == dname]
        m = f1[k].merge(sh, on=k, how="inner")
        rr = react_row(f1, "", show=False)
        ss = react_row(m, "", show=False)
        n_pairs = m.groupby(k).ngroups
        print(f"  FLIP {dname}: real P(+10<5m) {rr['P10<300']:.3f} vs shadow "
              f"{ss['P10<300']:.3f} (Δ {rr['P10<300']-ss['P10<300']:+.3f}) | "
              f"P+20 {rr['P+20']:.3f} vs {ss['P+20']:.3f} | medMFE5m "
              f"{rr['medMFE5m']:.0f} vs {ss['medMFE5m']:.0f} | paired taps "
              f"{n_pairs:,} shadows {len(m):,}")

    print("\n=== STAGE G — false breaks (cleared far edge but <B threshold) ===")
    # reconstruct from interactions: side0 taps with depth>w but <w+20
    frames = []
    for daily, lo, hi in ((DEV, "2021-01-01", "2026-06-17"),
                          (SEAL, "2026-06-18", "2026-08-14")):
        it = pd.read_parquet(f"{daily}/interactions.parquet",
                             columns=["session", "zid", "tap_sec",
                                      "app_side_above", "zlo_t", "zhi_t",
                                      "o_depth_max_full"])
        p6 = pd.read_parquet(f"{daily}/p6_real.parquet")
        for d in (it, p6):
            d["session"] = d["session"].astype(str)
        m = it.merge(p6[["session", "zid", "tap_sec", "t_rev10",
                         "t_rev20"]], on=["session", "zid", "tap_sec"],
                     how="left")
        m = m[(m["session"] >= lo) & (m["session"] <= hi)]
        frames.append(m)
    itall = pd.concat(frames, ignore_index=True)
    itall["w"] = itall["zhi_t"] - itall["zlo_t"]
    res = itall[itall["app_side_above"] == 0]
    false_brk = res[(res["o_depth_max_full"] > res["w"])
                    & (res["o_depth_max_full"] < res["w"] + 20)]
    true_brk = res[res["o_depth_max_full"] >= res["w"] + 20]
    print(f"  cleared-but-<+5pt (sweep-above): {len(false_brk):,} | "
          f"immediate 10pt DOWNWARD rejection {false_brk['t_rev10'].notna().mean():.2f}")
    later_true = false_brk.merge(true_brk[["zid"]].drop_duplicates(),
                                 on="zid", how="inner")
    print(f"  sweep-above zones later truly breaking (+5): "
          f"{later_true['zid'].nunique():,}/{false_brk['zid'].nunique():,}")

    print("\n=== STAGE H — retest depth inside old zone (FLIP B firsts) ===")
    fB = first[first["definition"] == "B"].copy()
    rel = fB["depth_max_full"] * 4 / fB["width_t"]   # depth pts→ticks? already ticks/ w
    rel = fB["depth_max_full"] / (fB["width_t"] / 4)  # both pts? normalize robustly
    rel = (fB["depth_max_full"] * 4) / fB["width_t"]
    fB["relpen"] = np.clip((fB["depth_max_full"] * 4) / fB["width_t"], 0, 3)
    for lo, hi, lab in ((0, 0.01, "touch only"), (0.01, 0.34, "upper third"),
                        (0.34, 0.67, "middle"), (0.67, 1.0, "lower third"),
                        (1.0, 3.1, "breach below")):
        g = fB[(fB["relpen"] >= lo) & (fB["relpen"] < hi)]
        if len(g) > 100:
            print(f"  {lab:>12}: n {len(g):6,} P(+10) "
                  f"{g['t_rev10'].notna().mean():.2f} P(+20) "
                  f"{g['t_rev20'].notna().mean():.2f}")

    print("\n=== STAGE J — time since acceptance (FLIP B firsts) ===")
    for lo, hi, lab in ((0, 300, "<5m"), (300, 1800, "5-30m"),
                        (1800, 7200, "30-120m"), (7200, 86400, "same/next"),
                        (86400, 5*86400, "1-5 sessions"),
                        (5*86400, 99*86400, ">5 sessions")):
        g = fB[(fB["dt_break"] >= lo) & (fB["dt_break"] < hi)]
        if len(g) > 100:
            print(f"  {lab:>12}: n {len(g):6,} P(+10<5m) "
                  f"{(g['t_rev10']<=300).mean():.2f} P(+20) "
                  f"{g['t_rev20'].notna().mean():.2f}")

    print("\n=== STAGE L — support tap lifecycle (FLIP B) ===")
    dB = df[df["definition"] == "B"]
    for j in (1, 2, 3, 4):
        g = dB[dB["sup_tap_no"] == j] if j < 4 else dB[dB["sup_tap_no"] >= 4]
        print(f"  sup tap {j}{'+' if j == 4 else ''}: n {len(g):6,} "
              f"P(+10<5m) {(g['t_rev10']<=300).mean():.2f} P(+20) "
              f"{g['t_rev20'].notna().mean():.2f}")

    print("\n=== STAGE N/O — chronological blocks (FLIP B firsts, real vs shadow) ===")
    for lo, hi in BLOCKS:
        g = fB[(fB["session"] >= lo) & (fB["session"] <= hi)]
        if len(g) < 150:
            continue
        m = g[k].merge(sh, on=k, how="inner")
        rp = float((g["t_rev10"] <= 300).mean())
        sp = float((m["t_rev10"] <= 300).mean()) if len(m) else np.nan
        print(f"  {lo[:7]}: n {len(g):5,} P(+10<5m) {rp:.3f} vs shadow "
              f"{sp:.3f} (Δ {rp-sp:+.3f}) | P(+20) "
              f"{g['t_rev20'].notna().mean():.2f} | "
              f"medMAE {g['depth_before_rev10'].median()/4:.1f}pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
