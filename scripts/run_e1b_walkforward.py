#!/usr/bin/env python
"""E1b — leaner feature sets and an expanding-window refit.

Two problems the fixed-window E1 fit exposed:

1. The **down side is under-predicted** at medium and far distances in
   validation (+0.05 to +0.07). The yearly table shows why: the down-tail p90
   of the normalised excursion runs 1.04-1.17 in 2022-23 and 1.45-1.49 in
   2024-25. A model frozen on 2021-24H1 cannot know that.
2. A frozen fit is not what a live system does anyway. Every morning it would
   have all history up to yesterday.

So this refits on an expanding window — for each month of validation, fit on
everything strictly before it minus an embargo — and compares against the
frozen fit and against leaner feature sets.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.eval.baseline import (DEFAULT_SCALE_FEATURES, calibration,     # noqa: E402
                                    fit_baseline, scores, touch_panel)
from nqlevels.eval.splits import assign, load_split_plan                      # noqa: E402
from nqlevels.io.store import layer_dir                                       # noqa: E402

CONFIGS = str(ROOT / "configs")

FEATURE_SETS = {
    "V3_atr_on":   ["log_atr14", "log_on_range"],
    "V4_full":     list(DEFAULT_SCALE_FEATURES),
    "V5_rv_lean":  ["log_rv_on", "log_prior_rv", "log_on_range"],
    "V6_rv_atr":   ["log_rv_on", "log_prior_rv", "log_on_range", "log_atr14"],
    "V7_rv_plus":  ["log_rv_on", "log_prior_rv", "log_on_range", "log_atr14",
                    "log_pre_range", "on_vol_rel_c"],
}


def hdr(s: str) -> None:
    print(f"\n{'='*78}\n{s}\n{'='*78}")


def walk_forward_panel(ctx: pd.DataFrame, eval_sessions, features,
                       embargo_days: int = 7, min_train: int = 400) -> pd.DataFrame:
    """Refit monthly on everything strictly before the month, minus embargo."""
    ev = ctx[ctx["session"].isin(eval_sessions)].copy()
    ev["_m"] = pd.to_datetime(ev["session"]).dt.to_period("M")
    out = []
    for m, part in ev.groupby("_m"):
        cutoff = pd.Timestamp(part["session"].min()) - pd.Timedelta(days=embargo_days)
        tr = ctx[ctx["session"] < cutoff]
        if len(tr) < min_train:
            continue
        b = fit_baseline(tr, features=features)
        out.append(touch_panel(part.drop(columns=["_m"]), b))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> int:
    ctx = pd.read_parquet(layer_dir("daily", CONFIGS) / "context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"])
    plan = load_split_plan(CONFIGS)
    sess = [s.date() for s in ctx["session"]]
    buckets = assign(sess, plan)
    tr_days = pd.to_datetime(buckets["train"])
    va_days = pd.to_datetime(buckets["validation"])
    tr = ctx[ctx["session"].isin(tr_days)].reset_index(drop=True)
    va = ctx[ctx["session"].isin(va_days)].reset_index(drop=True)
    print(f"train {len(tr)} | validation {len(va)} | holdout SEALED, untouched")

    hdr("1. Feature sets — FROZEN fit on train, scored on validation")
    frozen = {}
    for name, feats in FEATURE_SETS.items():
        b = fit_baseline(tr, features=feats)
        sc = scores(touch_panel(va, b))
        frozen[name] = sc
        print(f"  {name:12s} brier {sc['brier']:.4f}  skill {sc['brier_skill']:+.4f}  "
              f"logloss {sc['log_loss']:.4f}  ECE {sc['ece']:.4f}  maxgap {sc['max_gap']:+.3f}")

    hdr("2. Same feature sets — EXPANDING-WINDOW refit (monthly), on validation")
    wf = {}
    for name, feats in FEATURE_SETS.items():
        p = walk_forward_panel(ctx, va_days, feats)
        sc = scores(p)
        wf[name] = sc
        d = sc["brier_skill"] - frozen[name]["brier_skill"]
        print(f"  {name:12s} brier {sc['brier']:.4f}  skill {sc['brier_skill']:+.4f} "
              f"({d:+.4f} vs frozen)  logloss {sc['log_loss']:.4f}  ECE {sc['ece']:.4f}")

    best = max(wf, key=lambda k: wf[k]["brier_skill"])
    print(f"\n  best walk-forward: {best}")

    hdr(f"3. Where the remaining error is — {best}, walk-forward, validation")
    p = walk_forward_panel(ctx, va_days, FEATURE_SETS[best])
    g = p.groupby(["side", "dist_atr"]).agg(n=("touched", "size"), pred=("p_pred", "mean"),
                                            obs=("touched", "mean"))
    g["gap"] = g["obs"] - g["pred"]
    print(g.reset_index().to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  by side:")
    bs = p.groupby("side").agg(n=("touched", "size"), pred=("p_pred", "mean"), obs=("touched", "mean"))
    bs["gap"] = bs["obs"] - bs["pred"]
    print(bs.to_string(float_format=lambda v: f"{v:.4f}"))

    hdr("4. Full-history walk-forward (train+validation) — yearly stability")
    allday = sorted(set(buckets["train"]) | set(buckets["validation"]))
    pall = walk_forward_panel(ctx, pd.to_datetime(allday), FEATURE_SETS[best])
    pall["year"] = pd.to_datetime(pall["session"]).dt.year
    for yr, d in pall.groupby("year"):
        sc = scores(d)
        print(f"    {yr}  n={sc['n']:5d}  obs {sc['base_rate']:.3f}  brier {sc['brier']:.4f}  "
              f"skill {sc['brier_skill']:+.4f}  ECE {sc['ece']:.4f}  maxgap {sc['max_gap']:+.3f}")
    sc = scores(pall)
    print(f"    ALL   n={sc['n']:5d}  obs {sc['base_rate']:.3f}  brier {sc['brier']:.4f}  "
          f"skill {sc['brier_skill']:+.4f}  ECE {sc['ece']:.4f}")

    hdr("5. Sensitivity — does the answer depend on the refit cadence or ridge alpha?")
    for alpha in (0.3, 1.0, 3.0, 10.0):
        b = fit_baseline(tr, features=FEATURE_SETS[best], alpha=alpha)
        sc = scores(touch_panel(va, b))
        print(f"    alpha {alpha:5.1f}  VAL skill {sc['brier_skill']:+.4f}  ECE {sc['ece']:.4f}")
    for emb in (1, 7, 30):
        sc = scores(walk_forward_panel(ctx, va_days, FEATURE_SETS[best], embargo_days=emb))
        print(f"    embargo {emb:3d}d  WF skill {sc['brier_skill']:+.4f}  ECE {sc['ece']:.4f}")

    out = {"frozen": frozen, "walk_forward": wf, "best": best,
           "features": FEATURE_SETS[best],
           "wf_calibration_by_side_distance": g.reset_index().to_dict("records")}
    (ROOT / "outputs" / "e1b_walkforward.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote outputs/e1b_walkforward.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
