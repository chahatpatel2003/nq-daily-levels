#!/usr/bin/env python
"""P6 magnitude report — Questions 2 and 3, in NQ points.

When a level works: how far does price travel away, how far did it go
against the level first, where inside the zone did it turn, how do direct /
partial / sweep+reclaim modes differ, and does deeper adverse movement
predict larger or smaller reactions. All from the enriched real-interaction
outcomes; endpoint-based reaction is the headline magnitude with fixed
horizons as robustness.
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
sys.path.insert(0, str(ROOT / "scripts"))

from nqlevels.io.store import layer_dir                                        # noqa: E402
from run_p6_spatial import add_labels                                          # noqa: E402

CONFIGS = str(ROOT / "configs")
PCTS = [10, 25, 50, 75, 90]


def stats_pts(v: pd.Series) -> dict:
    v = v.dropna()
    if not len(v):
        return {}
    d = {"n": int(len(v)), "mean": float(v.mean())}
    for p in PCTS:
        d[f"p{p}"] = float(np.percentile(v, p))
    return d


def fmt(d: dict) -> str:
    if not d:
        return "(empty)"
    return (f"mean {d['mean']:6.1f} | p10 {d['p10']:5.1f} p25 {d['p25']:5.1f} "
            f"med {d['p50']:5.1f} p75 {d['p75']:5.1f} p90 {d['p90']:5.1f} "
            f"(n={d['n']:,})")


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    m = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    m = m[(m["n_fwd"] >= 1800) & m["y"].notna()].copy()
    m["cat"] = np.select(
        [m["y"] == 1, m["y"] == 0], ["S", "C"], default="?")
    # sub-split successes: B (sweep+reclaim) vs direct; direct split by turn
    from nqlevels.labels.sweep import classify
    c = classify(m, R=15, d0_t=4.0, D_t=m["atr_t"] * 0.10, T_s=1800, level="ne")
    m["is_B"] = c["cat_B"] == 1
    m["is_A"] = c["cat_A"] == 1
    m.loc[m["is_A"] & (m["turn_frac_rev15"] <= 0.5), "cat"] = "A_front"
    m.loc[m["is_A"] & (m["turn_frac_rev15"] > 0.5), "cat"] = "A_deep"
    m.loc[m["is_B"], "cat"] = "B"
    m["sweep_pts"] = m["depth_before_ne"] / 4.0
    m["reclaim_min"] = (m["t_reclaim_ne"] - m["first_beyond"]) / 60.0
    m["rev_after_pts"] = m["rev_after_ne_1800"] / 4.0
    m["t15_min"] = m["t_rev15_s"] / 60.0
    m["tmax_min"] = m["t_revmax_s"] / 60.0
    m["pen_full_pts"] = m["pen_max_full_t"] / 4.0

    res: dict = {"n": len(m)}
    print(f"resolved deduped interactions: {len(m):,} | success (primary) "
          f"{float((m['y']==1).mean()):.4f}")
    print(f"category shares of ALL touches: " + " ".join(
        f"{k}: {float((m['cat']==k).mean()):.3f}" for k in
        ("A_front", "A_deep", "B", "C")))
    S = m[m["y"] == 1]
    print(f"of successes: " + " ".join(
        f"{k}: {float((S['cat']==k).mean()):.3f}" for k in
        ("A_front", "A_deep", "B")))

    print("\n=== Q2 REACTION MAGNITUDE, NQ points (successful interactions) ===")
    res["reaction"] = {}
    for name, col in (("endpoint-based", "rev_ep_pts"),
                      ("30-min horizon", None), ("60-min", None),
                      ("full window", None)):
        v = S["rev_ep_pts"] if col else S[
            {"30-min horizon": "rev_pts_1800", "60-min": "rev_pts_3600",
             "full window": "rev_full_t"}[name]] / (1 if col else 4.0)
        st = stats_pts(v)
        res["reaction"][name] = st
        print(f"  {name:>15}: {fmt(st)}")
    for cat in ("A_front", "A_deep", "B"):
        st = stats_pts(S.loc[S["cat"] == cat, "rev_ep_pts"])
        res["reaction"][cat] = st
        print(f"  {cat:>15}: {fmt(st)}")

    print("\n=== Q3 ADVERSE MOVEMENT BEFORE REACTION, NQ points ===")
    res["adverse"] = {}
    st = stats_pts(S["adv15_pts"])
    res["adverse"]["all_success"] = st
    print(f"  {'all successes':>15}: {fmt(st)}")
    for cat in ("A_front", "A_deep", "B"):
        st = stats_pts(S.loc[S["cat"] == cat, "adv15_pts"])
        res["adverse"][cat] = st
        print(f"  {cat:>15}: {fmt(st)}")
    print(f"  {'penetration (max, all touches)':>30}: "
          f"{fmt(stats_pts(m['pen_full_pts']))}")
    print(f"  {'penetration (failures C)':>30}: "
          f"{fmt(stats_pts(m.loc[m['cat']=='C','pen_full_pts']))}")

    print("\n=== ratio and coupling ===")
    adv = S["adv15_pts"].clip(lower=0.25)
    ratio = S["rev_ep_pts"] / adv
    print(f"  reaction/adverse ratio: median {float(ratio.median()):.2f} | "
          f"p25 {float(ratio.quantile(.25)):.2f} p75 {float(ratio.quantile(.75)):.2f}")
    cc = float(S[["adv15_pts", "rev_ep_pts"]].corr().iloc[0, 1])
    print(f"  corr(adverse, reaction) among successes: {cc:+.3f}")
    B = S[S["cat"] == "B"]
    cb = float(B[["sweep_pts", "rev_after_pts"]].corr().iloc[0, 1]) if len(B) > 500 else np.nan
    print(f"  corr(sweep depth, post-reclaim reaction) among B: {cb:+.3f}")
    res["ratio_med"] = float(ratio.median())
    res["corr_adv_rev"] = cc
    res["corr_sweep_rev"] = cb
    # deeper sweep -> bigger reaction? quartiles of sweep depth vs reaction
    if len(B) > 1000:
        q = pd.qcut(B["sweep_pts"], 4)
        t = B.groupby(q, observed=True)["rev_after_pts"].agg(["median", "size"])
        print("  post-reclaim reaction by sweep-depth quartile:")
        print(t.round(1).to_string())

    print("\n=== sweep/reclaim shape (B) ===")
    print(f"  sweep depth pts: {fmt(stats_pts(B['sweep_pts']))}")
    print(f"  time to reclaim min: {fmt(stats_pts(B['reclaim_min']))}")
    print(f"  reaction after reclaim pts: {fmt(stats_pts(B['rev_after_pts']))}")

    print("\n=== timing ===")
    print(f"  time to 15-pt reaction (min): {fmt(stats_pts(S['t15_min']))}")
    print(f"  time to max reaction (min):   {fmt(stats_pts(S['tmax_min']))}")

    print("\n=== Q21 where do turns occur (successes, fraction of zone) ===")
    tf = S["turn_frac_rev15"].dropna()
    res["turn_frac"] = stats_pts(tf)
    print(f"  turn_frac: {fmt(stats_pts(tf))}")
    bins = [0, .25, .5, .75, 1.0, 1.5, 100]
    lab = ["0-25%", "25-50%", "50-75%", "75-100%", "1-1.5x", ">1.5x(sweep)"]
    sh = pd.cut(tf, bins, labels=lab).value_counts(normalize=True).reindex(lab)
    print("  " + " | ".join(f"{k}: {v:.3f}" for k, v in sh.items()))
    res["turn_bins"] = sh.to_dict()

    out = ROOT / "outputs" / "p6_magnitude.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
