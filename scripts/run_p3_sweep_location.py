#!/usr/bin/env python
"""P3-208 stage 2 — does LOCATION predict sweep-aware reaction quality?

The dependent variable changed (y_any = direct rejection OR valid
sweep+reclaim), so previously rejected structural families are legitimately
re-scored — under exactly the Phase-2/3 protections: side x distance strata,
session-clustered bootstrap, |d| < 0.005 noise floor, touched-conditioned
contrasts at matched geometry, train first, registered gates before any
validation look.

Three taxonomy cells are carried everywhere (the depth cap has no plateau, so
a conclusion that flips across cells is not a conclusion):
  strict   D = 24t,      T = 900s
  mid      D = 0.05 ATR, T = 900s
  primary  D = 0.10 ATR, T = 1800s   (the letter of the stage-1 criteria)
All at R = 15 pts, d0 = 4t, reclaim = near edge held 60 s.

Families re-scored: classical core levels, untested prior-day support and
resistance, HTF extremes (+ pwh alone), displacement origins (fresh /
same-morning / consumed), reaction memory (strong / overnight-event), and
macro-release structures.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.clock import decision_time_utc, load_calendar                    # noqa: E402
from nqlevels.eval.quality import (STRAT_STATS, add_distance_bands,            # noqa: E402
                                   stratified_contrast)
from nqlevels.io.store import layer_dir                                        # noqa: E402
from nqlevels.labels.sweep import classify                                     # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from run_p3_priorweek import build_htf_levels                                  # noqa: E402

CONFIGS = str(ROOT / "configs")
NEAR_T = 6.0
CELLS = {"strict": dict(D="24t", T=900), "mid": dict(D="5%atr", T=900),
         "primary": dict(D="10%atr", T=1800)}
SUP = ["dk_pdl", "dk_pd_globex_low", "dk_pd_val"]
RES = ["dk_pdh", "dk_pd_globex_high", "dk_pd_vah"]

for c in ("y_any", "cat_A", "cat_B", "cat_C"):
    STRAT_STATS[c] = (c, None)
    for cell in ("strict", "mid", "primary"):
        STRAT_STATS[f"{c}_{cell}"] = (f"{c}_{cell}", None)


def build_frame(split: str):
    d0 = layer_dir("daily", CONFIGS)
    sw = pd.read_parquet(d0 / "sweep_labels.parquet")
    ax_cols = ["session", "off_t", "atr_t", "dist_atr", "split", "p_base",
               "near_core", "on_touches"] + SUP + RES
    ax = pd.read_parquet(d0 / "axis_measured.parquet", columns=ax_cols)
    m = sw.merge(ax, on=["session", "off_t"], how="inner", validate="1:1")
    m = m[(m["split"] == split) & (m["dist_atr"] <= 1.2) & m["p_base"].notna()
          & (m["n_fwd"] >= 1800)].reset_index(drop=True)
    m["above"] = (m["off_t"] > 0).astype("int8")
    m = add_distance_bands(m, n=5)

    for name, cell in CELLS.items():
        D = (m["atr_t"] * (0.05 if cell["D"] == "5%atr" else 0.10)) \
            if "atr" in cell["D"] else 24.0
        c = classify(m, R=15, d0_t=4.0, D_t=D, T_s=cell["T"], level="ne")
        for k in ("y_any", "cat_A", "cat_B", "cat_C"):
            m[f"{k}_{name}"] = c[k]

    # ---- family flags ------------------------------------------------------
    ctx = pd.read_parquet(d0 / "context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    ctxi = ctx.set_index("session")
    adj = pd.read_parquet(d0 / "roll_adjust.parquet")
    adji = adj.set_index("session")["adj_t"]
    bars = pd.read_parquet(d0 / "bars_1m.parquet")
    bars["ts"] = pd.DatetimeIndex(bars["ts"]).tz_convert("UTC")
    htf = build_htf_levels(bars, adj).set_index("session")

    flags = {}
    flags["core"] = m["near_core"] == 1
    unt = m["on_touches"].fillna(0) == 0
    flags["untested_support"] = (m[SUP].min(axis=1) <= NEAR_T) & unt
    flags["untested_resist"] = (m[RES].min(axis=1) <= NEAR_T) & unt
    hit = np.zeros(len(m), dtype=bool)
    for k in ("pwh", "pwl", "pmh", "pml"):
        lv = m["session"].map(htf[k]) + m["session"].map(adji)
        ref = m["session"].map(ctxi["ref_t"])
        kk = ((m["off_t"] - (lv - ref)).abs() <= NEAR_T).fillna(False).to_numpy()
        if k == "pwh":
            flags["pwh"] = pd.Series(kk, index=m.index)
        hit |= kk
    flags["htf_ext"] = pd.Series(hit, index=m.index)

    org = pd.read_parquet(d0 / "origins_at.parquet",
                          columns=["session", "off_t", "orig_fresh", "orig_intact",
                                   "orig_consumed", "orig_sameday"])
    m2 = m.merge(org, on=["session", "off_t"], how="left")
    flags["orig_fresh"] = ((m2["orig_fresh"] == 1) & (m2["orig_intact"] == 1)
                           & (m2["orig_sameday"] == 0)).fillna(False)
    flags["orig_sameday"] = ((m2["orig_sameday"] == 1)
                             & (m2["orig_fresh"] == 1)).fillna(False)
    flags["orig_consumed"] = (m2["orig_consumed"] == 1).fillna(False)

    mem = pd.read_parquet(d0 / "memory.parquet",
                          columns=["session", "off_t", "mem2_w", "mem02_n"])
    m3 = m.merge(mem, on=["session", "off_t"], how="left")
    thr = json.loads((ROOT / "outputs" / "p3_memory_thresholds.json").read_text())
    flags["mem_strong"] = (m3["mem2_w"] >= thr["mem2_w"]).fillna(False)
    flags["mem0_any"] = (m3["mem02_n"] > 0).fillna(False)

    # macro-release structures (tape-detected release mornings, volx >= 4)
    cal = load_calendar(CONFIGS)
    rel = set(ctx.loc[ctx["macro_volx_pre"] >= 4.0, "session"])
    fb_by = {s: g for s, g in bars[bars["is_front"]].groupby("session", sort=False)
             if s in rel}
    st = {}
    for s, g in fb_by.items():
        cut = pd.Timestamp(decision_time_utc(s, cal))
        w = g[(g["ts"] >= cut - timedelta(minutes=55))
              & (g["ts"] < cut - timedelta(minutes=25))]
        if len(w) < 15:
            continue
        ref = float(ctxi.loc[s, "ref_t"])
        st[s] = (float(w["high_t"].max()) - ref, float(w["low_t"].min()) - ref)
    sh = m["session"].map({s: v[0] for s, v in st.items()})
    sl = m["session"].map({s: v[1] for s, v in st.items()})
    flags["rel_ext"] = (((m["off_t"] - sh).abs() <= NEAR_T)
                        | ((m["off_t"] - sl).abs() <= NEAR_T)).fillna(False)
    return m, flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--families", default="",
                    help="comma list to restrict (validation looks are narrow)")
    a = ap.parse_args()
    t0 = _time.time()
    m, flags = build_frame(a.split)
    if a.families:
        keep = set(a.families.split(","))
        flags = {k: v for k, v in flags.items() if k in keep}
    print(f"{a.split}: {len(m):,} touched+resolved zones, "
          f"{m['session'].nunique()} sessions")
    for name, cell in CELLS.items():
        print(f"  base y_any[{name}]: {float(m[f'y_any_{name}'].mean()):.3f} "
              f"(A {float(m[f'cat_A_{name}'].mean()):.3f}, "
              f"B {float(m[f'cat_B_{name}'].mean()):.3f})")

    res = {"split": a.split, "contrasts": {}}
    hdr = (f"\n  {'family':>18} {'cell':>8} {'outcome':>7} {'diff':>8} {'lo':>8} "
           f"{'hi':>8} {'p>0':>6} {'n_flag':>8}")
    print(hdr)
    for fam, fl in flags.items():
        m["_f"] = fl.astype("int8")
        n1 = int(fl.sum())
        if n1 < 250:
            print(f"  {fam:>18}: underpowered ({n1})")
            continue
        for cell in CELLS:
            for oc in ("y_any", "cat_B", "cat_A"):
                r = stratified_contrast(m, "_f", stat=f"{oc}_{cell}",
                                        strata=("side", "_dband"),
                                        n_boot=a.boot, min_cell=20)
                res["contrasts"][f"{fam}.{cell}.{oc}"] = r
                star = "*" if (r.get("lo") or 0) > 0 or (r.get("hi") or 0) < 0 else " "
                if oc == "y_any" or star == "*":
                    print(f"  {fam:>18} {cell:>8} {oc:>7} {r['diff']:8.4f} "
                          f"{r['lo']:8.4f} {r['hi']:8.4f} {r['p_gt0']:6.3f} "
                          f"{n1:8,}{star}")
    out = ROOT / "outputs" / f"p3_sweep_location_{a.split}.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
