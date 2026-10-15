#!/usr/bin/env python
"""P3-208 stage 3 — the frozen P3-207 engine under sweep-aware outcomes.

Nothing is refit. The engine's stored score (E1 P(touch) x walk-forward
P(clean|touch), context-only) selects the same zones as before; only the
outcome the selection is judged on changes: y_any = direct rejection OR valid
sweep+reclaim, at the three frozen taxonomy cells. Matched same-morning
controls, top-1..4, train and validation, and the practical block the brief
asks for: among reached displayed zones — % direct, % sweep+reclaim,
% combined, % failure; typical sweep depth, time to reclaim, move after
reclaim.
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

from nqlevels.eval import topk                                                 # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402
from nqlevels.labels.sweep import classify                                     # noqa: E402

CONFIGS = str(ROOT / "configs")
CELLS = {"strict": dict(D="24t", T=900), "mid": dict(D="5%atr", T=900),
         "primary": dict(D="10%atr", T=1800)}
TICKS = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist-lo", type=float, default=0.15)
    ap.add_argument("--dist-hi", type=float, default=1.0)
    ap.add_argument("--min-sep-t", type=float, default=24.0)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    t0 = _time.time()

    d0 = layer_dir("daily", CONFIGS)
    ax = pd.read_parquet(d0 / "axis_measured.parquet",
                         columns=["session", "off_t", "atr_t", "dist_atr",
                                  "touched", "split", "p_base", "first_touch_min"])
    sc = pd.read_parquet(d0 / "scored_axis.parquet",
                         columns=["session", "off_t", "score_A", "p_clean_A"])
    sw = pd.read_parquet(d0 / "sweep_labels.parquet")
    m = ax.merge(sc, on=["session", "off_t"], how="inner", validate="1:1")
    m = m.merge(sw.drop(columns=["side"]), on=["session", "off_t"], how="left",
                validate="1:1")
    m = m[m["p_base"].notna() & (m["dist_atr"] <= 1.5)].reset_index(drop=True)
    m["above"] = (m["off_t"] > 0).astype("int8")

    for name, cell in CELLS.items():
        D = (m["atr_t"] * (0.05 if cell["D"] == "5%atr" else 0.10)) \
            if "atr" in cell["D"] else 24.0
        c = classify(m, R=15, d0_t=4.0, D_t=D, T_s=cell["T"], level="ne")
        for k in ("y_any", "cat_A", "cat_B", "cat_C"):
            m[f"{k}_{name}"] = c[k]

    res = {"policy": {"dist": [a.dist_lo, a.dist_hi], "min_sep_t": a.min_sep_t},
           "cells": CELLS}
    for split in ("train", "validation"):
        f = m[m["split"] == split]
        print(f"\n{'='*92}\n{split}: frozen engine top-K under sweep-aware outcomes"
              f"\n{'='*92}")
        for name in CELLS:
            r = topk.evaluate(f, "score_A", ks=(1, 2, 3, 4),
                              outcome=f"y_any_{name}", min_sep_t=a.min_sep_t,
                              dist_window=(a.dist_lo, a.dist_hi),
                              n_draws=8, n_boot=a.boot, seed=23,
                              by_year=(split == "validation"))
            res[f"topk_{split}_{name}"] = r
            print(f"\n--- cell {name} ---")
            print(topk.format_report(r))

        # practical block: composition among reached picks, K=4, primary cell
        picks = []
        for s, g in f.groupby("session"):
            picks.append(topk.select_topk(g, "score_A", 4, a.min_sep_t,
                                          (a.dist_lo, a.dist_hi)))
        P = pd.concat(picks, ignore_index=True)
        T = P[(P["touched"] == True) & P["y_any_primary"].notna()]    # noqa: E712
        B = T[T["cat_B_primary"] == 1]
        prac = {
            "zones_per_morning": float(P.groupby("session").size().mean()),
            "touch_rate": float(P["touched"].mean()),
            "resolved_share_of_touched": float(
                (P.loc[P["touched"] == True, "y_any_primary"].notna()).mean()),  # noqa: E712
            "pct_direct": float(T["cat_A_primary"].mean()),
            "pct_sweep_reclaim": float(T["cat_B_primary"].mean()),
            "pct_combined": float(T["y_any_primary"].mean()),
            "pct_fail": float(T["cat_C_primary"].mean()),
            "combined_strict": float(T["y_any_strict"].mean()),
            "combined_mid": float(T["y_any_mid"].mean()),
            "med_sweep_depth_pts_B": float((B["depth_before_ne"] / TICKS).median()),
            "med_time_to_reclaim_min_B": float(
                ((B["t_reclaim_ne"] - B["first_beyond"]) / 60.0).median()),
            "med_move_after_reclaim_pts_B": float(
                (B["rev_after_ne_1800"] / TICKS).median()),
            "resweep_rate_B": float(B["re_swept_ne"].mean()),
        }
        res[f"practical_{split}"] = prac
        print(f"\npractical (K=4, primary cell), {split}:")
        for k, v in prac.items():
            print(f"  {k:>30}: {v:.4f}")

    out = ROOT / "outputs" / "p3_sweep_topk.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
