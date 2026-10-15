#!/usr/bin/env python
"""P3-207 — the assembled product engine, evaluated end-to-end.

Everything Phase 2+3 established, composed into the morning selection the
brief asks for, and measured the only honest way — as a selection against
matched same-morning alternatives (eval/topk):

  score(zone) = P(touch | distance, side, vol)          [E1, frozen, calibrated]
              x P(clean | touch, distance, day context) [walk-forward logistic]

Two variants answer the remaining question head-on:
  A  context-only          (distance, side, day features)
  B  location-augmented    (A + reaction-memory, consumed-structure, energy
                            and same-morning-origin flags)
If B does not beat A in selection, location features add nothing even when a
model is allowed to use them — closing the question the honest way.

The reaction-probability annotation is walk-forward calibrated: refit monthly
on all data before the month, so the 2024-25 base-rate fall is tracked, not
averaged away. Reported: calibration, top-1..4 reached/reacted rates,
magnitudes, penetration, slice, per-year stability — the practical numbers.
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

CONFIGS = str(ROOT / "configs")

CTX_FEATS = ["dist_atr", "dist2", "above", "ad", "atr_pctile",
             "on_range_rel_atr", "gap_abs", "log_volx", "loc_pdr"]
LOC_FEATS = ["mem_strong", "mem0_any", "orig_consumed_f", "orig_sameday_f",
             "orig_fresh_f", "near_core"]


def design(m: pd.DataFrame) -> pd.DataFrame:
    d = m.copy()
    d["dist2"] = d["dist_atr"] ** 2
    d["ad"] = d["above"] * d["dist_atr"]
    d["gap_abs"] = d["gap_atr_prior"].abs().fillna(0.0)
    d["log_volx"] = np.log1p(d["macro_volx_pre"]).fillna(0.0)
    d["loc_pdr"] = d["loc_in_pdr_prior"].clip(-2, 3).fillna(0.5)
    d["atr_pctile"] = d["atr_pctile"].fillna(0.5)
    d["on_range_rel_atr"] = d["on_range_rel_atr"].fillna(0.5)
    return d


def walkforward_pclean(d: pd.DataFrame, feats: list) -> np.ndarray:
    """Monthly expanding refit: P(clean | touch) for every row, using only
    strictly earlier months. Train rows get an in-sample fit (reference only —
    validation is the report)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    t = d[(d["touched"] == True) & d["clean"].notna()]               # noqa: E712
    Xt_all = t[feats].to_numpy("float64")
    yt_all = t["clean"].to_numpy("float64")
    tr_mask = (t["split"] == "train").to_numpy()

    sc = StandardScaler().fit(Xt_all[tr_mask])
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(sc.transform(Xt_all[tr_mask]), yt_all[tr_mask])
    p = np.full(len(d), np.nan)
    is_tr = (d["split"] == "train").to_numpy()
    p[is_tr] = lr.predict_proba(sc.transform(d.loc[is_tr, feats].to_numpy("float64")))[:, 1]

    mon = pd.to_datetime(d["session"].astype(str)).dt.to_period("M")
    mon_t = pd.to_datetime(t["session"].astype(str)).dt.to_period("M")
    for mth in sorted(mon[d["split"] == "validation"].unique()):
        hist = (mon_t < mth).to_numpy()
        if hist.sum() < 5000:
            continue
        sc_m = StandardScaler().fit(Xt_all[hist])
        lr_m = LogisticRegression(C=1.0, max_iter=1000)
        lr_m.fit(sc_m.transform(Xt_all[hist]), yt_all[hist])
        sel = ((mon == mth) & (d["split"] == "validation")).to_numpy()
        p[sel] = lr_m.predict_proba(sc_m.transform(d.loc[sel, feats].to_numpy("float64")))[:, 1]
    return p


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    t = topk.reliability(y, p, bins)
    w = t["n"] / t["n"].sum()
    return float((w * (t["p_mean"] - t["y_rate"]).abs()).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist-lo", type=float, default=0.15)
    ap.add_argument("--dist-hi", type=float, default=1.0)
    ap.add_argument("--min-sep-t", type=float, default=24.0)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    t0 = _time.time()

    d0 = layer_dir("daily", CONFIGS)
    m = pd.read_parquet(d0 / "axis_measured.parquet")
    mem = pd.read_parquet(d0 / "memory.parquet",
                          columns=["session", "off_t", "mem2_w", "mem02_n"])
    org = pd.read_parquet(d0 / "origins_at.parquet",
                          columns=["session", "off_t", "orig_fresh", "orig_intact",
                                   "orig_consumed", "orig_sameday"])
    m = m.merge(mem, on=["session", "off_t"], how="left", validate="1:1")
    m = m.merge(org, on=["session", "off_t"], how="left", validate="1:1")
    m = m[(m["side"] != "inside") & m["p_base"].notna()
          & (m["dist_atr"] <= 1.5)].reset_index(drop=True)
    m = design(m)

    q90 = float(m.loc[m["split"] == "train", "mem2_w"].quantile(0.90))
    m["mem_strong"] = (m["mem2_w"] >= q90).fillna(False).astype("int8")
    m["mem0_any"] = (m["mem02_n"] > 0).fillna(False).astype("int8")
    m["orig_consumed_f"] = (m["orig_consumed"] == 1).fillna(False).astype("int8")
    m["orig_fresh_f"] = ((m["orig_fresh"] == 1) & (m["orig_intact"] == 1)
                         & (m["orig_sameday"] == 0)).fillna(False).astype("int8")
    m["orig_sameday_f"] = ((m["orig_sameday"] == 1) & (m["orig_fresh"] == 1)
                           ).fillna(False).astype("int8")

    print("fitting walk-forward P(clean|touch): context-only ...")
    m["p_clean_A"] = walkforward_pclean(m, CTX_FEATS)
    print("fitting walk-forward P(clean|touch): location-augmented ...")
    m["p_clean_B"] = walkforward_pclean(m, CTX_FEATS + LOC_FEATS)
    m["score_A"] = m["p_base"] * m["p_clean_A"]
    m["score_B"] = m["p_base"] * m["p_clean_B"]
    m["p_joint_A"] = m["score_A"]

    res = {"policy": {"dist_window": [a.dist_lo, a.dist_hi],
                      "min_sep_t": a.min_sep_t}, "calibration": {}}

    # ---- calibration of the annotation, validation only ---------------------
    va = m[m["split"] == "validation"]
    vt = va[(va["touched"] == True) & va["clean"].notna()]           # noqa: E712
    e_cond = ece(vt["clean"].to_numpy("float64"), vt["p_clean_A"].to_numpy("float64"))
    yj = ((va["touched"] == True)                                     # noqa: E712
          & (pd.to_numeric(va["clean"], errors="coerce").fillna(0) == 1)).astype(float)
    e_joint = ece(yj.to_numpy(), va["score_A"].to_numpy("float64"))
    res["calibration"]["cond_ece"] = e_cond
    res["calibration"]["joint_ece"] = e_joint
    res["calibration"]["cond_table"] = topk.reliability(
        vt["clean"].to_numpy("float64"), vt["p_clean_A"].to_numpy("float64")).to_dict("records")
    print(f"\nvalidation calibration: P(clean|touch) ECE {e_cond:.4f} | "
          f"joint P(touch&clean) ECE {e_joint:.4f}")
    by_y = vt.assign(y=pd.to_datetime(vt["session"].astype(str)).dt.year) \
             .groupby("y").agg(pred=("p_clean_A", "mean"), real=("clean", "mean"),
                               n=("clean", "size"))
    print(by_y.to_string(float_format=lambda v: f"{v:.4f}"))
    res["calibration"]["by_year"] = by_y.reset_index().to_dict("records")

    # ---- top-k selection, validation ---------------------------------------
    for tag in ("A", "B"):
        print(f"\n{'='*90}\nvariant {tag} — top-K selection on validation, "
              f"vs matched same-morning controls\n{'='*90}")
        r = topk.evaluate(va, f"score_{tag}", ks=(1, 2, 3, 4), outcome="clean",
                          min_sep_t=a.min_sep_t, dist_window=(a.dist_lo, a.dist_hi),
                          n_draws=8, n_boot=a.boot, seed=23)
        print(topk.format_report(r))
        res[f"topk_{tag}"] = r

    # ---- practical numbers for the report: what the product would show -----
    from nqlevels.io.raw import assert_not_raw
    keep = ["session", "off_t", "dist_atr", "above", "p_base", "p_clean_A",
            "score_A", "split"]
    assert_not_raw(d0 / "scored_axis.parquet")
    m[keep].to_parquet(d0 / "scored_axis.parquet", index=False, compression="zstd")

    print(f"\n{'='*90}\npractical numbers — variant A, K=4, validation\n{'='*90}")
    picks = []
    for s, g in va.groupby("session"):
        p = topk.select_topk(g, "score_A", 4, a.min_sep_t, (a.dist_lo, a.dist_hi))
        picks.append(p)
    P = pd.concat(picks, ignore_index=True)
    T = P[P["touched"] == True]                                       # noqa: E712
    prac = {
        "zones_per_morning": float(P.groupby("session").size().mean()),
        "mornings": int(P["session"].nunique()),
        "touch_rate_per_zone": float(P["touched"].mean()),
        "reached_per_morning": float(T.groupby("session").size()
                                     .reindex(P["session"].unique(), fill_value=0).mean()),
        "p_ge1_reached": float((T.groupby("session").size()
                                .reindex(P["session"].unique(), fill_value=0) >= 1).mean()),
        "clean_10_5_given_touch": float(T["clean"].dropna().mean()),
        "fp10_5_given_touch": float(T["fp_rev10_adv5"].dropna().mean()),
        "fp15_10_given_touch": float(T["fp_rev15_adv10"].dropna().mean()),
        "fp20_10_given_touch": float(T["fp_rev20_adv10"].dropna().mean()),
        "fp30_15_given_touch": float(T["fp_rev30_adv15"].dropna().mean()),
        "med_rev30m_atr": float(T["rev30m_atr"].median()),
        "med_rev30m_pts": float((T["rev_1800s"] / 4.0).median()),
        "med_penetration_pts": float((T["max_penetration_t"] / 4.0).median()),
        "slice_through_rate": float(T["slice_through_60s"].mean()),
        "med_t_rev10_min": float((T["t_rev10_s"].dropna() / 60.0).median()),
        "med_first_touch_min": float(T["first_touch_min"].median()),
        "med_pred_p_touch": float(P["p_base"].median()),
        "med_pred_p_clean": float(P["p_clean_A"].median()),
    }
    for k, v in prac.items():
        print(f"  {k:>26}: {v:.4f}" if isinstance(v, float) else f"  {k:>26}: {v}")
    res["practical_K4"] = prac

    # three example mornings, highest scores, what the display would say
    ex = []
    for s in list(P["session"].unique())[:3]:
        g = P[P["session"] == s].sort_values("score_A", ascending=False)
        ex.append({"session": str(s), "zones": [
            {"off_pts": float(r["off_t"] / 4.0), "dist_atr": float(r["dist_atr"]),
             "p_touch": float(r["p_base"]), "p_clean": float(r["p_clean_A"]),
             "p_joint": float(r["score_A"]), "touched": bool(r["touched"]),
             "clean": (None if pd.isna(r["clean"]) else float(r["clean"]))}
            for _, r in g.iterrows()]})
    res["examples"] = ex

    out = ROOT / "outputs" / "p3_product.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
