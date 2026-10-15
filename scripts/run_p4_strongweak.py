#!/usr/bin/env python
"""P4-303 — what did strong zones have, before the tap, that weak ones lacked?

Population: deduplicated, resolved interactions. Outcome: the frozen taxonomy
(primary cell headline, strict as sensitivity), y = A ∪ B given touch.

For every pre-tap feature: top-vs-bottom tercile contrast on y (binary and
categorical features use their levels), with session-clustered bootstrap CIs,
computed pooled AND within (segment × approach-side) strata — because time of
day and side are the mechanical confounds of this design — plus per-year
contrasts. The REGISTERED robustness bar (RESEARCH_PLAN §15.1): pooled
|d| ≥ 2 pp, stratified CI excluding zero, same sign in ≥ 5 of 6 years.

This is a screen, not a verdict: ~45 features are examined and the count is
reported. Survivors go to the model stage; nothing here is headlined without
the era bar.
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

from nqlevels.eval.quality import STRAT_STATS, stratified_contrast             # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402
from nqlevels.labels.sweep import classify                                     # noqa: E402

CONFIGS = str(ROOT / "configs")
SWEEP_COLS = ["n_fwd", "t_rev15", "depth_before_rev15", "first_beyond",
              "depth_max_full", "dwell_beyond_full", "t_reclaim_ne",
              "depth_before_ne", "rev_after_ne_1800", "t_reclaim_en",
              "t_reclaim_mid", "depth_before_mid", "rev_after_mid_1800",
              "crossings_full", "reentries_full", "inside_share_full",
              "last_beyond"]

CONT = ["hw_atr", "base_atr", "dep_mag_atr", "dep_vel", "dep_volx", "dep_delta",
        "age_min", "age_sessions", "mins_since_last", "prior_pen_atr",
        "vol_through_log", "mins_inside", "max_away_atr", "d20_min_atr",
        "volz_share", "volz_rel", "vol_cliff", "tap_no",
        "app_toward_5m", "app_toward_15m", "app_toward_30m", "app_range_5m",
        "app_range_30m", "app_delta_5m", "app_delta_15m", "app_volx_5m",
        "app_speed_5m", "app_accel", "app_compress", "app_retraces_15m",
        "app_monotone_1m", "range_used_atr", "vwap_dist_atr", "rv60_atr",
        "atr_pctile", "tod_pt_min", "prior_succ_share"]
CATS = ["scale", "dir_match", "broke_prior", "flipped", "app_side_above",
        "seg", "form_seg", "release_slot", "dow", "fresh_first_tap"]

STRAT_STATS["y"] = ("y", None)
STRAT_STATS["y_strict"] = ("y_strict", None)


def load_frame(cell: str = "primary") -> pd.DataFrame:
    d0 = layer_dir("daily", CONFIGS)
    m = pd.read_parquet(d0 / "interactions.parquet")
    sw = m[[f"o_{c}" for c in SWEEP_COLS]].rename(
        columns={f"o_{c}": c for c in SWEEP_COLS})
    for name, D, T in (("primary", "10%", 1800), ("strict", 24.0, 900)):
        Dv = m["atr_t"] * 0.10 if D == "10%" else D
        c = classify(sw, R=15, d0_t=4.0, D_t=Dv, T_s=T, level="ne")
        m[f"y_{name}"] = c["y_any"]
        m[f"catA_{name}"] = c["cat_A"]
        m[f"catB_{name}"] = c["cat_B"]
    m = m[m["o_n_fwd"] >= 1800].reset_index(drop=True)
    m["y"] = m["y_primary"]
    m["y_strict"] = m["y_strict"]

    # ---- dedup: same session, side, within 600 s and 0.05 ATR -> one tap ----
    m = m.sort_values(["session", "tap_sec"]).reset_index(drop=True)
    cid = np.zeros(len(m), dtype="int64")
    k = 0
    prev = None
    for i, r in enumerate(m.itertuples()):
        if prev is None or r.session != prev.session \
                or (r.tap_sec - prev.tap_sec) > 600 \
                or abs(r.price_t - prev.price_t) > 0.05 * r.atr_t \
                or r.app_side_above != prev.app_side_above:
            k += 1
        cid[i] = k
        prev = r
    m["cluster"] = cid
    m["n_dup"] = m.groupby("cluster")["zid"].transform("size")
    m = m.sort_values(["cluster", "age_min"], ascending=[True, False]) \
         .drop_duplicates("cluster", keep="first").reset_index(drop=True)

    # derived screen features
    m["base_atr"] = m["base_range_t"] / m["atr_t"]
    m["prior_pen_atr"] = m["prior_pen_t"] / m["atr_t"]
    m["vol_through_log"] = np.log1p(m["vol_through"])
    m["d20_min_atr"] = m[["d20_hi_atr", "d20_lo_atr"]].abs().min(axis=1)
    npr = m["n_prior_A"] + m["n_prior_B"] + m["n_prior_C"]
    m["prior_succ_share"] = np.where(npr > 0,
                                     (m["n_prior_A"] + m["n_prior_B"]) / npr, np.nan)
    m["fresh_first_tap"] = ((m["tap_no"] == 1) & (m["n_pending"] == 0)).astype(int)
    m["year"] = pd.to_datetime(m["session"].astype(str)).dt.year
    return m


def tercile_contrast(m, col, boot=300):
    v = m[col]
    ok = v.notna()
    if ok.sum() < 3000:
        return None
    q1, q2 = v[ok].quantile([1 / 3, 2 / 3])
    if q1 == q2:
        return None
    sub = m[ok & ((v <= q1) | (v >= q2))].copy()
    sub["_f"] = (sub[col] >= q2).astype("int8")
    r = stratified_contrast(sub, "_f", stat="y", strata=("seg", "app_side_above"),
                            n_boot=boot, min_cell=30)
    t, b = sub[sub["_f"] == 1], sub[sub["_f"] == 0]
    raw = float(t["y"].mean() - b["y"].mean())
    yrs = []
    for y, g in sub.groupby("year"):
        a_, b_ = g[g["_f"] == 1]["y"].mean(), g[g["_f"] == 0]["y"].mean()
        yrs.append(float(a_ - b_))
    n_sign = sum(1 for x in yrs if np.sign(x) == np.sign(r["diff"]) and x != 0)
    return {"feature": col, "diff": r["diff"], "lo": r["lo"], "hi": r["hi"],
            "raw": raw, "n_hi": int(len(t)), "yr_agree": n_sign,
            "n_years": len(yrs), "per_year": [round(x, 4) for x in yrs],
            "robust": bool(abs(r["diff"]) >= 0.02 and
                           (r["lo"] > 0 or r["hi"] < 0) and n_sign >= 5)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    t0 = _time.time()
    m = load_frame()
    print(f"deduped resolved interactions: {len(m):,} over "
          f"{m['session'].nunique()} sessions | base y(primary) "
          f"{float(m['y'].mean()):.4f} | strict {float(m['y_strict'].mean()):.4f}")
    print(f"catA {float(m['catA_primary'].mean()):.3f} | "
          f"catB {float(m['catB_primary'].mean()):.3f}")
    print("\nbase y by year: " + " ".join(
        f"{y}:{g['y'].mean():.3f}" for y, g in m.groupby("year")))
    print("base y by seg:  " + " ".join(
        f"{s}:{g['y'].mean():.3f}(n={len(g)})" for s, g in m.groupby("seg")))
    print("base y by tap_no: " + " ".join(
        f"{t}:{g['y'].mean():.3f}(n={len(g)})" for t, g in m.groupby("tap_no")))

    res = {"n": len(m), "base": float(m["y"].mean()), "screen": [],
           "cats": {}}
    rows = []
    for col in CONT:
        r = tercile_contrast(m, col, a.boot)
        if r:
            rows.append(r)
    tab = pd.DataFrame(rows).sort_values("diff", key=lambda s: s.abs(),
                                         ascending=False)
    res["screen"] = tab.to_dict("records")
    print(f"\ncontinuous screen — top-vs-bottom tercile on y, "
          f"stratified (seg x side); {len(rows)} features tested")
    print(f"{'feature':>18} {'d_strat':>8} {'lo':>8} {'hi':>8} {'raw':>7} "
          f"{'yrs+':>5} {'ROBUST':>7}")
    for r in tab.itertuples():
        print(f"{r.feature:>18} {r.diff:8.4f} {r.lo:8.4f} {r.hi:8.4f} "
              f"{r.raw:7.4f} {r.yr_agree}/{r.n_years:>2}  "
              f"{'*ROBUST*' if r.robust else ''}")

    # ---- lifetime / invalidation (Q12/Q13) ---------------------------------
    print("\nlifetime: y by age bucket")
    ab = pd.cut(m["age_min"], [0, 120, 480, 1380, 1380 * 5, 1380 * 15, 1e9],
                labels=["<2h", "2-8h", "8h-1s", "1-5s", "5-15s", ">15s"])
    g = m.groupby(ab, observed=True)["y"].agg(["mean", "size"])
    res["lifetime_age"] = g.reset_index().astype(str).to_dict("records")
    print(g.to_string())
    print("\nlifetime: y by tap_no x fresh age (<=1 session vs older)")
    m["_old"] = (m["age_sessions"] >= 2).astype(int)
    g2 = m.pivot_table(index="tap_no", columns="_old", values="y",
                       aggfunc=["mean", "size"])
    print(g2.round(3).to_string())
    print("\ninvalidation: y after role flip vs intact (by tap_no)")
    g3 = m.pivot_table(index="flipped", columns="tap_no", values="y",
                       aggfunc="mean")
    print(g3.round(3).to_string())
    res["flip_table"] = g3.reset_index().to_dict("records")

    print("\ncategorical levels (raw y rate per level)")
    for col in CATS:
        if col not in m.columns:
            continue
        g = m.groupby(col)["y"].agg(["mean", "size"])
        res["cats"][col] = g.reset_index().to_dict("records")
        lv = " | ".join(f"{i}: {r['mean']:.3f} (n={int(r['size'])})"
                        for i, r in g.iterrows())
        print(f"  {col:>16}: {lv}")

    out = ROOT / "outputs" / "p4_strongweak.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
