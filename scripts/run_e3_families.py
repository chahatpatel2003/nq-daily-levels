#!/usr/bin/env python
"""E3 — do the classical named level families carry any information?

Each family is measured on two separate questions, never mixed:

**Destination** — attraction, i.e. touch rate minus the distance-and-side
matched E1 baseline. A 70 % touch rate at a distance where an arbitrary price
is touched 71 % of the time is worth nothing, and most named levels sit close
to the market precisely because they are derived from recent price.

**Reaction** — behaviour conditional on arriving, against a placebo of
identical width at near-identical distance. Reported at several thresholds so
no single arbitrary definition drives the verdict.

Two controls that matter as much as the statistics:

* Every interval is a **session-level block bootstrap**. Four levels within ten
  points of each other on one morning are one observation, not four.
* 86 families are tested, so raw p-values are meaningless on their own.
  Benjamini-Hochberg FDR is reported alongside.

A family is also measured in a **pure** form — clusters where it is the only
kind present — so its own effect is separated from whatever it happens to
coincide with.

Train sessions only.
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.eval.baseline import E1_FEATURES, fit_baseline                   # noqa: E402
from nqlevels.eval.labeling import build_paths, clean_rejection, measure_zone   # noqa: E402
from nqlevels.eval.quality import (add_attraction, apply_width, block_bootstrap,  # noqa: E402
                                   clean_rate, make_placebos, mean_attraction,
                                   paired_contrast, reaction_ratio, touch_rate)
from nqlevels.eval.splits import assign, load_split_plan                        # noqa: E402
from nqlevels.io.store import layer_dir                                         # noqa: E402

CONFIGS = str(ROOT / "configs")
TICK = 0.25
pd.set_option("display.width", 240)


def hdr(s: str) -> None:
    print(f"\n{'='*104}\n{s}\n{'='*104}")


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype="float64")
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def measure_set(cands: pd.DataFrame, paths, w: np.ndarray, baseline, ctx) -> pd.DataFrame:
    z = apply_width(cands, w).reset_index(drop=True)
    rows = []
    for s, g in z.groupby("session", sort=False):
        p = paths.get(s)
        if p is None:
            continue
        zlo, zhi = g["zlo_t"].to_numpy("float64"), g["zhi_t"].to_numpy("float64")
        idx = g.index.to_numpy()
        for j in range(len(g)):
            r = measure_zone(p, zlo[j], zhi[j], point_grid=(5, 10, 15, 20, 30))
            r["_r"] = idx[j]
            rows.append(r)
    m = pd.DataFrame(rows).set_index("_r")
    out = m.join(z[[c for c in z.columns if c not in m.columns]]).reset_index(drop=True)
    for k, a in ((10, 5), (15, 10), (20, 10), (30, 15)):
        out[f"clean_{k}_{a}"] = clean_rejection(out, k, a)
    out["clean"] = out["clean_10_5"]
    out["rev30m_atr"] = out["rev_1800s"] / out["atr_t"]
    out["pen_atr"] = out["max_penetration_t"] / out["atr_t"]
    return add_attraction(out, baseline, ctx)


def main() -> int:
    t0 = _time.time()
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--width-k", type=float, default=0.03, help="half-width = k * ATR")
    ap.add_argument("--min-sessions", type=int, default=120)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()

    d = layer_dir("daily", CONFIGS)
    ctx = pd.read_parquet(d / "context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    bk = assign(list(ctx["session"]), load_split_plan(CONFIGS))
    tr_days = set(bk["train"])

    c = pd.read_parquet(d / "candidates_named.parquet")
    c = c[c["session"].isin(tr_days)]
    c = c.merge(ctx[["session", "atr14_t", "atr_pctile"]], on="session", how="left") \
         .rename(columns={"atr14_t": "atr_t"})
    c = c[np.isfinite(c["atr_t"]) & (c["dist_atr"] <= 1.5)].reset_index(drop=True)
    trctx = ctx[ctx["session"].isin(tr_days)].reset_index(drop=True)
    baseline = fit_baseline(trctx, features=list(E1_FEATURES))
    paths = build_paths(pd.read_parquet(d / "paths.parquet"))
    w = np.clip(a.width_k * c["atr_t"].to_numpy("float64"), 1.0, 200.0)
    print(f"train candidates {len(c):,} / {c['session'].nunique()} sessions; "
          f"half-width = {a.width_k}*ATR (median {np.median(w)*TICK:.2f} pts)")

    real = measure_set(c, paths, w, baseline, trctx)
    plac = make_placebos(c, kinds=("shift",), shift_atr=0.35)
    pl = measure_set(plac, paths, np.clip(a.width_k * plac["atr_t"].to_numpy("float64"),
                                          1.0, 200.0), baseline, trctx)
    print(f"measured {len(real):,} real + {len(pl):,} placebo zones "
          f"({_time.time()-t0:.0f}s)")

    hdr("0. Pooled reality check — all named levels together")
    for nm, f in (("real", real), ("placebo", pl)):
        t = f[f["touched"] == True]                                    # noqa: E712
        print(f"  {nm:8s} n={len(f):,}  touch {touch_rate(f):.3f}  "
              f"attraction {mean_attraction(f):+.4f}  clean10/5 {clean_rate(f):.3f}  "
              f"slice60s {t['slice_through_60s'].mean():.3f}  "
              f"accept {t['accepted'].mean():.3f}  revRatio {reaction_ratio(f):.3f}")

    # ---------------------------------------------------------------- families
    hdr("1. Per-family destination (attraction) and reaction, vs matched placebo")
    real_x = real.assign(kind=real["kinds"].str.split(",")).explode("kind")
    pl_x = pl.assign(kind=pl["kinds"].str.split(",")).explode("kind")
    rows = []
    for kind, g in real_x.groupby("kind"):
        if g["session"].nunique() < a.min_sessions:
            continue
        q = pl_x[pl_x["kind"] == kind]
        att = block_bootstrap(g, mean_attraction, n_boot=a.boot)
        con = paired_contrast(g, q, clean_rate, n_boot=a.boot)
        acon = paired_contrast(g, q, mean_attraction, n_boot=a.boot)
        t = g[g["touched"] == True]                                    # noqa: E712
        rows.append({
            "kind": kind, "n": len(g), "sessions": int(g["session"].nunique()),
            "med_dist_atr": float(g["dist_atr"].median()),
            "touch": touch_rate(g), "p_base": float(g["p_base"].mean()),
            "attraction": att[0], "att_lo": att[1], "att_hi": att[2],
            "d_att": acon["diff"], "d_att_p": 1 - acon["p_gt0"],
            "clean": clean_rate(g), "plac_clean": clean_rate(q),
            "d_clean": con["diff"], "dc_lo": con["lo"], "dc_hi": con["hi"],
            "d_clean_p": 1 - con["p_gt0"],
            "clean_15_10": float(t["clean_15_10"].dropna().mean()) if len(t) else np.nan,
            "clean_20_10": float(t["clean_20_10"].dropna().mean()) if len(t) else np.nan,
            "rev30m_atr": float(t["rev30m_atr"].median()) if len(t) else np.nan,
            "pen_atr": float(t["pen_atr"].median()) if len(t) else np.nan,
            "slice60": float(t["slice_through_60s"].mean()) if len(t) else np.nan,
            "accept": float(t["accepted"].mean()) if len(t) else np.nan,
            "n_touched": int(len(t)),
        })
    fam = pd.DataFrame(rows)
    fam["d_clean_fdr"] = bh_fdr(fam["d_clean_p"].fillna(1.0).to_numpy())
    fam["d_att_fdr"] = bh_fdr(fam["d_att_p"].fillna(1.0).to_numpy())

    show = ["kind", "sessions", "n", "med_dist_atr", "touch", "p_base", "attraction",
            "att_lo", "att_hi", "d_att", "d_att_fdr"]
    print("\n  -- DESTINATION: sorted by attraction (touch minus distance-matched baseline)")
    print(fam.sort_values("attraction", ascending=False)[show].head(18)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  -- worst:")
    print(fam.sort_values("attraction")[show].head(6)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    show2 = ["kind", "sessions", "n_touched", "clean", "plac_clean", "d_clean",
             "dc_lo", "dc_hi", "d_clean_fdr", "clean_20_10", "rev30m_atr",
             "pen_atr", "slice60", "accept"]
    print("\n  -- REACTION: sorted by clean-rejection edge over placebo")
    print(fam.sort_values("d_clean", ascending=False)[show2].head(20)
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    hdr("2. Survivors — FDR < 0.10 on either axis")
    surv = fam[(fam["d_clean_fdr"] < 0.10) | (fam["d_att_fdr"] < 0.10)]
    if len(surv):
        print(surv.sort_values("d_clean", ascending=False)[
            ["kind", "sessions", "n_touched", "attraction", "d_att_fdr",
             "clean", "d_clean", "d_clean_fdr", "rev30m_atr"]]
            .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    else:
        print("  NONE. No named family beats its matched placebo after FDR control.")

    hdr("3. Pure clusters only (this kind and nothing else)")
    pure = real_x[real_x["n_kinds"] == 1]
    pure_p = pl_x[pl_x["n_kinds"] == 1]
    prow = []
    for kind, g in pure.groupby("kind"):
        if g["session"].nunique() < a.min_sessions:
            continue
        q = pure_p[pure_p["kind"] == kind]
        con = paired_contrast(g, q, clean_rate, n_boot=300)
        att = block_bootstrap(g, mean_attraction, n_boot=300)
        prow.append({"kind": kind, "sessions": int(g["session"].nunique()), "n": len(g),
                     "attraction": att[0], "att_lo": att[1], "att_hi": att[2],
                     "clean": clean_rate(g), "d_clean": con["diff"],
                     "dc_lo": con["lo"], "dc_hi": con["hi"]})
    pu = pd.DataFrame(prow)
    if len(pu):
        print(pu.sort_values("d_clean", ascending=False).head(15)
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    hdr("4. Distance is the whole story? touch vs baseline by distance band")
    real["_b"] = pd.cut(real["dist_atr"], [0, .15, .3, .5, .8, 1.5])
    g = real.groupby("_b", observed=True).agg(
        n=("touched", "size"), touch=("touched", "mean"),
        base=("p_base", "mean"), attraction=("attraction", "mean"))
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))

    out = {"width_k": a.width_k, "families": fam.to_dict("records"),
           "pure": pu.to_dict("records") if len(pu) else [],
           "n_sessions": int(c["session"].nunique())}
    (ROOT / "outputs" / "e3_families.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote outputs/e3_families.json  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
