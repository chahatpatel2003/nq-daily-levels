#!/usr/bin/env python
"""E2 + E3 on the price-axis grid — the properly controlled test.

At a given distance and side there is exactly one price, so "a control level at
the same distance" cannot be drawn. Instead: lay a uniform 1-point grid over
the reachable axis, mark which prices are at a named level, measure every one,
and compare **within distance-and-side strata**. Distance is then held fixed by
construction rather than adjusted for afterwards.

Answers three questions in one pass:

  E2  what zone width, and does the answer survive the width being varied
  E3  do the classical named families beat matched non-level prices
  --  and, as a by-product, what the honest per-family numbers actually are

Train only. Validation is held for the selection stage; the holdout is sealed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.eval.baseline import E1_FEATURES, fit_baseline                   # noqa: E402
from nqlevels.eval.labeling import build_paths                                 # noqa: E402
from nqlevels.eval.quality import (add_distance_bands, clean_rate,             # noqa: E402
                                   mean_attraction, measure_candidates,
                                   reaction_ratio, stratified_contrast,
                                   touch_rate)
from nqlevels.eval.splits import assign, load_split_plan                        # noqa: E402
from nqlevels.eval.universe import CORE_KINDS                                   # noqa: E402
from nqlevels.io.store import layer_dir                                         # noqa: E402

CONFIGS = str(ROOT / "configs")
TICK = 0.25
pd.set_option("display.width", 230)


def hdr(s: str) -> None:
    print(f"\n{'='*100}\n{s}\n{'='*100}")


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype="float64")
    n = p.size
    o = np.argsort(p)
    r = p[o] * n / (np.arange(n) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(r, 0, 1)
    return out


def main() -> int:
    t0 = _time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=250)
    ap.add_argument("--near-t", type=float, default=6.0)
    a = ap.parse_args()

    d = layer_dir("daily", CONFIGS)
    ctx = pd.read_parquet(d / "context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    bk = assign(list(ctx["session"]), load_split_plan(CONFIGS))
    tr_days = set(bk["train"])

    ax = pd.read_parquet(d / "axis.parquet")
    ax = ax[ax["session"].isin(tr_days)].reset_index(drop=True)
    ax = ax.merge(ctx[["session", "atr14_t", "atr_pctile"]], on="session", how="left") \
           .rename(columns={"atr14_t": "atr_t"})
    ax = ax[np.isfinite(ax["atr_t"])].reset_index(drop=True)
    trctx = ctx[ctx["session"].isin(tr_days)].reset_index(drop=True)
    baseline = fit_baseline(trctx, features=list(E1_FEATURES))
    paths = build_paths(pd.read_parquet(d / "paths.parquet"))
    print(f"axis grid: {len(ax):,} prices over {ax['session'].nunique()} train sessions "
          f"({len(ax)/ax['session'].nunique():.0f}/session); setup {_time.time()-t0:.0f}s")
    print(f"near_core = within {a.near_t:.0f} ticks ({a.near_t*TICK:.1f} pts) of a core level "
          f"-> {100*ax['near_core'].mean():.1f}% of the axis")

    # ---------------------------------------------------------------- E2
    hdr("E2. Width sweep, stratified by distance and side (TRAIN)")
    print("    'd_clean' = clean-rejection rate at a core level minus the rate at a")
    print("    non-level price in the SAME distance band and side.\n")
    print(f"{'width':>13} {'w(pts)':>7} {'touch':>7} {'attr':>8} {'clean':>7} "
          f"{'d_att':>9} {'d_clean':>9} {'95% CI':>20} {'p>0':>6} {'revRat':>7}")
    widths = {"fixed_4t": np.full(len(ax), 4.0), "fixed_6t": np.full(len(ax), 6.0),
              "fixed_8t": np.full(len(ax), 8.0), "fixed_12t": np.full(len(ax), 12.0),
              "fixed_20t": np.full(len(ax), 20.0)}
    for k in (0.010, 0.020, 0.035, 0.060):
        widths[f"atr_{k:.3f}"] = np.clip(k * ax["atr_t"].to_numpy("float64"), 1.0, 200.0)
    rows, frames = [], {}
    for name, w in widths.items():
        m = measure_candidates(ax, paths, w, baseline, trctx,
                               point_grid=(5, 10, 20), clean_specs=((10, 5), (20, 10)))
        m = add_distance_bands(m)
        ca = stratified_contrast(m, "near_core", "clean_rate", n_boot=a.boot)
        aa = stratified_contrast(m, "near_core", "attraction", n_boot=a.boot)
        r = {"width": name, "w_pts": float(np.median(w) * TICK), "touch": touch_rate(m),
             "attraction": mean_attraction(m), "clean": clean_rate(m),
             "d_att": aa["diff"], "d_att_p": 1 - (aa["p_gt0"] or 0),
             "d_clean": ca["diff"], "lo": ca["lo"], "hi": ca["hi"],
             "p_gt0": ca["p_gt0"], "rev_ratio": reaction_ratio(m), "n": len(m)}
        print(f"{name:>13} {r['w_pts']:7.2f} {r['touch']:7.3f} {r['attraction']:+8.4f} "
              f"{r['clean']:7.3f} {r['d_att']:+9.5f} {r['d_clean']:+9.5f} "
              f"[{r['lo']:+.4f},{r['hi']:+.4f}] {r['p_gt0']:6.3f} {r['rev_ratio']:7.3f}",
              flush=True)
        rows.append(r)
        frames[name] = m
    sweep = pd.DataFrame(rows)

    best = sweep.loc[sweep["d_clean"].idxmax(), "width"]
    print(f"\n  strongest contrast at width = {best}")
    M = frames[best]

    hdr("E2b. Regime homogeneity of the chosen width")
    M2 = M[np.isfinite(M["atr_pctile"])].copy()
    M2["_terc"] = pd.qcut(M2["atr_pctile"], 3, labels=["low", "mid", "high"])
    for t, g in M2.groupby("_terc", observed=True):
        gg = add_distance_bands(g)
        c = stratified_contrast(gg, "near_core", "clean_rate", n_boot=150)
        print(f"    {t:>5} vol: clean {clean_rate(g):.4f}  d_clean {c['diff']:+.5f} "
              f"[{c['lo']:+.4f},{c['hi']:+.4f}]  n={len(g):,}")

    hdr("E3. Per-family: each core kind vs matched-distance non-level prices")
    print("    a grid price is 'at' a kind if within 6 ticks of it AND the")
    print("    comparison set is prices not near ANY core level, same distance band\n")
    rows = []
    for kind in CORE_KINDS:
        col = f"dk_{kind}"
        if col not in M.columns:
            continue
        sub = M[(M[col] <= a.near_t) | (M["near_core"] == 0)].copy()
        sub["_f"] = (sub[col] <= a.near_t).astype("int8")
        if sub["_f"].sum() < 300:
            continue
        sub = add_distance_bands(sub)
        ca = stratified_contrast(sub, "_f", "clean_rate", n_boot=a.boot)
        aa = stratified_contrast(sub, "_f", "attraction", n_boot=a.boot)
        f = sub[sub["_f"] == 1]
        t = f[f["touched"] == True]                                    # noqa: E712
        rows.append({"kind": kind, "n": int(len(f)),
                     "sessions": int(f["session"].nunique()),
                     "touch": touch_rate(f), "attraction": mean_attraction(f),
                     "d_att": aa["diff"], "d_att_p": 1 - (aa["p_gt0"] or 0),
                     "clean": clean_rate(f), "d_clean": ca["diff"],
                     "lo": ca["lo"], "hi": ca["hi"], "d_clean_p": 1 - (ca["p_gt0"] or 0),
                     "rev30m_atr": float(t["rev30m_atr"].median()) if len(t) else np.nan,
                     "slice60": float(t["slice_through_60s"].mean()) if len(t) else np.nan,
                     "accept": float(t["accepted"].mean()) if len(t) else np.nan})
    fam = pd.DataFrame(rows)
    fam["d_clean_fdr"] = bh_fdr(fam["d_clean_p"].fillna(1).to_numpy())
    fam["d_att_fdr"] = bh_fdr(fam["d_att_p"].fillna(1).to_numpy())
    print(fam.sort_values("d_clean", ascending=False)[
        ["kind", "sessions", "n", "touch", "attraction", "d_att", "d_att_fdr",
         "clean", "d_clean", "lo", "hi", "d_clean_fdr", "rev30m_atr", "slice60"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    hdr("E3b. Survivors at FDR < 0.10")
    s1 = fam[fam["d_clean_fdr"] < 0.10]
    s2 = fam[fam["d_att_fdr"] < 0.10]
    print(f"  reaction edge : {list(s1['kind']) if len(s1) else 'NONE'}")
    print(f"  destination   : {list(s2['kind']) if len(s2) else 'NONE'}")

    hdr("E3c. Sanity — the dense families, same treatment")
    sub = M.copy()
    sub = add_distance_bands(sub)
    for flag in ("near_core", "near_dense"):
        c = stratified_contrast(sub, flag, "clean_rate", n_boot=a.boot)
        aa = stratified_contrast(sub, flag, "attraction", n_boot=a.boot)
        print(f"    {flag:11s} share {100*sub[flag].mean():5.1f}%  "
              f"d_clean {c['diff']:+.5f} [{c['lo']:+.4f},{c['hi']:+.4f}]  "
              f"d_att {aa['diff']:+.5f}")

    out = {"sweep": sweep.to_dict("records"), "families": fam.to_dict("records"),
           "best_width": best, "n_sessions": int(ax["session"].nunique())}
    (ROOT / "outputs" / "e2e3_axis.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote outputs/e2e3_axis.json  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
