#!/usr/bin/env python
"""P3-208 stage 1 — the sweep/reclaim taxonomy itself (train, unconditional).

No location hypothesis is examined here. This stage answers, in order:

1. **Is sweep->reclaim->reaction a real, separable behaviour?** Among touched
   zones that penetrated beyond the far edge: reclaim probability and
   post-reclaim reaction probability as functions of penetration depth and
   time beyond. If "temporary sweep" differs from "acceptance", these curves
   bend somewhere; if they are flat, the distinction is arbitrary.
2. **Taxonomy frequencies over a definition grid** (R x d0 x D x T x level),
   looking for plateaus, not peaks.
3. **Year stability** of the frequencies at the primary definition.

Definition-selection criteria, WRITTEN BEFORE RESULTS (the primary cell is
chosen by these, then frozen for every later location test):
  * reclaim level = near edge (``ne``) — the trading concept being tested is
    "reclaimed the level", chosen a priori;
  * R = 15 points — mid-grid of the friction-motivated sizes;
  * d0 = 4 ticks — one point of boundary noise is not a sweep;
  * D and T from the stage-1 curves: the largest caps that keep
    P(reaction | reclaimed) within ~90 % of its shallow/fast plateau —
    with the neighbouring cells carried as sensitivity everywhere;
  * B-rate at the chosen cell must be year-stable on train (each year within
    ±30 % relative of pooled), else the nearest stable neighbour is taken.
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

from nqlevels.io.store import layer_dir                                        # noqa: E402
from nqlevels.labels.sweep import classify                                     # noqa: E402

CONFIGS = str(ROOT / "configs")
TICKS = 4  # per point


def load(split: str) -> pd.DataFrame:
    d0 = layer_dir("daily", CONFIGS)
    sw = pd.read_parquet(d0 / "sweep_labels.parquet")
    ax = pd.read_parquet(d0 / "axis_measured.parquet",
                         columns=["session", "off_t", "atr_t", "dist_atr",
                                  "split", "p_base", "near_core"])
    m = sw.merge(ax, on=["session", "off_t"], how="inner", validate="1:1")
    m = m[(m["split"] == split) & (m["dist_atr"] <= 1.2) & m["p_base"].notna()]
    return m.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    t0 = _time.time()
    m = load(a.split)
    res = {"split": a.split}
    print(f"{a.split}: {len(m):,} touched zones, {m['session'].nunique()} sessions | "
          f"resolved {float((m['n_fwd'] >= 1800).mean()):.3f}")
    m = m[m["n_fwd"] >= 1800].reset_index(drop=True)

    # ---------------------------------------------------------------- part 1
    print("\nPART 1 — is a temporary sweep separable from acceptance? "
          "(zones that went beyond the far edge by > 1 pt)")
    b = m[m["depth_max_full"] > 4.0].copy()
    print(f"  {len(b):,} of {len(m):,} touched zones went beyond ({len(b)/len(m):.1%})")
    b["depth_atr"] = b["depth_max_full"] / b["atr_t"]
    b["rec_ne"] = b["t_reclaim_ne"].notna()
    b["t_rec_rel"] = b["t_reclaim_ne"] - b["first_beyond"]
    b["react_after"] = (b["rev_after_ne_1800"] >= 15 * TICKS)

    depth_bins = [0, .01, .02, .035, .05, .075, .10, .15, 10.0]
    b["_db"] = pd.cut(b["depth_atr"], depth_bins)
    g = b.groupby("_db", observed=True).agg(
        n=("rec_ne", "size"), p_reclaim=("rec_ne", "mean"),
        p_react_if_rec=("react_after", lambda v: v[b.loc[v.index, "rec_ne"]].mean()),
        med_t_rec_min=("t_rec_rel", lambda v: np.nanmedian(v) / 60.0))
    print("\n  by max penetration depth (ATR fractions):")
    print(g.to_string(float_format=lambda v: f"{v:.3f}"))
    res["by_depth"] = g.reset_index().astype(str).to_dict("records")

    rec = b[b["rec_ne"]].copy()
    tbins = [0, 120, 300, 900, 1800, 3600, 1e9]
    rec["_tb"] = pd.cut(rec["t_rec_rel"], tbins)
    g2 = rec.groupby("_tb", observed=True).agg(
        n=("react_after", "size"), p_react=("react_after", "mean"),
        med_depth_atr=("depth_atr", "median"))
    print("\n  among reclaimed: reaction probability by time-to-reclaim:")
    print(g2.to_string(float_format=lambda v: f"{v:.3f}"))
    res["by_time"] = g2.reset_index().astype(str).to_dict("records")

    g3 = rec.groupby("_tb", observed=True)["re_swept_ne"].mean()
    print("\n  re-sweep rate after reclaim, by time-to-reclaim: "
          + " ".join(f"{v:.2f}" for v in g3))

    # ---------------------------------------------------------------- part 2
    print("\nPART 2 — taxonomy frequencies over the definition grid "
          "(shares of touched+resolved)")
    grid_rows = []
    Dspec = [("12t", 12.0), ("24t", 24.0), ("5%atr", None), ("10%atr", None)]
    for R in (10, 15, 20):
        for d0 in (0.0, 4.0):
            for dn, dv in Dspec:
                D = (m["atr_t"] * (0.05 if dn == "5%atr" else 0.10)) if dv is None else dv
                for T in (300, 900, 1800):
                    for lvl in ("en", "mid", "ne"):
                        c = classify(m, R=R, d0_t=d0, D_t=D, T_s=T, level=lvl)
                        ok = c["resolved"]
                        row = {"R": R, "d0": d0, "D": dn, "T": T, "L": lvl,
                               "A": float(c.loc[ok, "cat_A"].mean()),
                               "B": float(c.loc[ok, "cat_B"].mean()),
                               "C": float(c.loc[ok, "cat_C"].mean()),
                               "Dch": float(c.loc[ok, "cat_D"].mean()),
                               "any": float(c.loc[ok, "y_any"].mean())}
                        grid_rows.append(row)
    gr = pd.DataFrame(grid_rows)
    res["grid"] = gr.to_dict("records")
    print("\n  reclaim level = ne (a-priori primary), d0 = 4t:")
    sub = gr[(gr["L"] == "ne") & (gr["d0"] == 4.0)]
    print(sub.pivot_table(index=["R", "D"], columns="T",
                          values=["A", "B", "any"]).round(3).to_string())
    print("\n  B-rate by reclaim level (R=15, d0=4t):")
    sub2 = gr[(gr["R"] == 15) & (gr["d0"] == 4.0)]
    print(sub2.pivot_table(index="D", columns=["L", "T"], values="B").round(3).to_string())

    # ---------------------------------------------------------------- part 3
    print("\nPART 3 — year stability at candidate primary cells "
          "(R=15, d0=4t, level=ne)")
    m["_y"] = pd.to_datetime(m["session"].astype(str)).dt.year
    for dn in ("24t", "5%atr", "10%atr"):
        D = (m["atr_t"] * (0.05 if dn == "5%atr" else 0.10)) if "atr" in dn else 24.0
        for T in (900, 1800):
            c = classify(m, R=15, d0_t=4.0, D_t=D, T_s=T, level="ne")
            cc = pd.concat([m[["_y"]], c], axis=1)
            cc = cc[cc["resolved"]]
            yr = cc.groupby("_y")[["cat_A", "cat_B", "cat_C", "cat_D", "y_any"]].mean()
            print(f"\n  D={dn}, T={T}s:")
            print(yr.round(3).to_string())
            res[f"yr_D{dn}_T{T}"] = yr.reset_index().to_dict("records")

    out = ROOT / "outputs" / f"p3_sweep_taxonomy_{a.split}.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
