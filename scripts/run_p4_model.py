#!/usr/bin/env python
"""P4-304 — can pre-tap information predict interaction success, out of time?

Three nested tiers, so every increment is attributable:
  T0  context only        (session segment, time, side, vol regime, ...)
  T1  + zone-static       (formation, freshness, age, memory, volume, HTF)
  T2  + approach          (how price came in, last 5-30 minutes)

Expanding-year walk-forward: train on years < Y, test on Y, for Y in
2022..2026. HistGradientBoosting with a logistic benchmark. Permutation
importance measured on TEST data only. OOS scores are persisted per row for
the top-k product test (run_p4_topk).
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
sys.path.insert(0, str(ROOT / "scripts"))

from nqlevels.io.store import layer_dir                                        # noqa: E402
from run_p4_strongweak import load_frame                                       # noqa: E402

CONFIGS = str(ROOT / "configs")

T0_FEATS = ["seg", "tod_pt_min", "app_side_above", "dow", "atr_pctile",
            "range_used_atr", "vwap_dist_atr", "rv60_atr", "release_slot",
            "sess_elapsed_s"]
T1_FEATS = T0_FEATS + [
    "scale", "dir_match", "broke_prior", "flipped", "fresh_first_tap",
    "hw_atr", "base_atr", "dep_mag_atr", "dep_vel", "dep_volx", "dep_delta",
    "tap_no", "age_min", "age_sessions", "mins_since_last",
    "n_prior_A", "n_prior_B", "n_prior_C", "n_pending", "prior_pen_atr",
    "prior_succ_share", "vol_through_log", "mins_inside", "max_away_atr",
    "d20_min_atr", "volz_share", "volz_rel", "vol_cliff", "form_seg"]
T2_FEATS = T1_FEATS + [
    "app_toward_5m", "app_toward_15m", "app_toward_30m", "app_range_5m",
    "app_range_15m", "app_range_30m", "app_delta_5m", "app_delta_15m",
    "app_delta_30m", "app_volx_5m", "app_speed_5m", "app_accel",
    "app_compress", "app_retraces_15m", "app_monotone_1m"]


def auc(y, s):
    y = np.asarray(y, "float64")
    s = np.asarray(s, "float64")
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size < 100 or y.min() == y.max():
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def fit_predict(tr, te, feats, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    g = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                       max_depth=4, l2_regularization=1.0,
                                       random_state=seed)
    g.fit(tr[feats].to_numpy("float64"), tr["y"].to_numpy("float64"))
    return g, g.predict_proba(te[feats].to_numpy("float64"))[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm-reps", type=int, default=3)
    a = ap.parse_args()
    t0 = _time.time()
    m = load_frame()
    m = m[m["y"].notna()].reset_index(drop=True)
    years = sorted(m["year"].unique())
    print(f"frame: {len(m):,} interactions, years {years}, "
          f"base y {float(m['y'].mean()):.4f}")

    res = {"by_year": {}, "importance": {}, "n": len(m)}
    scores = m[["session", "zid", "tap_sec", "year", "y", "y_strict",
                "seg", "app_side_above", "price_t", "atr_t", "tap_no",
                "age_sessions", "scale"]].copy()
    for tier, feats in (("T0", T0_FEATS), ("T1", T1_FEATS), ("T2", T2_FEATS)):
        scores[f"p_{tier}"] = np.nan
    imp_acc: dict = {}

    for Y in years[1:]:
        tr = m[m["year"] < Y]
        te = m[m["year"] == Y]
        if len(tr) < 8000 or len(te) < 2000:
            continue
        row = {"n_tr": len(tr), "n_te": len(te),
               "base": float(te["y"].mean())}
        preds = {}
        for tier, feats in (("T0", T0_FEATS), ("T1", T1_FEATS), ("T2", T2_FEATS)):
            g, p = fit_predict(tr, te, feats)
            preds[tier] = (g, p, feats)
            row[f"auc_{tier}"] = auc(te["y"], p)
            row[f"brier_{tier}"] = float(np.mean((p - te["y"]) ** 2))
            scores.loc[te.index, f"p_{tier}"] = p
        # permutation importance on TEST, tier T2
        g2, p2, f2 = preds["T2"]
        base_auc = row["auc_T2"]
        rng = np.random.default_rng(int(Y))
        X = te[f2].to_numpy("float64")
        for j, f in enumerate(f2):
            d = 0.0
            for rep in range(a.perm_reps):
                Xp = X.copy()
                Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
                d += base_auc - auc(te["y"], g2.predict_proba(Xp)[:, 1])
            imp_acc[f] = imp_acc.get(f, 0.0) + d / a.perm_reps
        res["by_year"][int(Y)] = row
        print(f"  {Y}: n_te {len(te):,} base {row['base']:.3f} | AUC "
              f"T0 {row['auc_T0']:.4f}  T1 {row['auc_T1']:.4f}  "
              f"T2 {row['auc_T2']:.4f}  (T1-T0 {row['auc_T1']-row['auc_T0']:+.4f}, "
              f"T2-T1 {row['auc_T2']-row['auc_T1']:+.4f})", flush=True)

    n_folds = len(res["by_year"])
    imp = pd.Series({k: v / n_folds for k, v in imp_acc.items()}) \
        .sort_values(ascending=False)
    res["importance"] = imp.to_dict()
    print(f"\npermutation importance (test-only, mean ΔAUC over {n_folds} folds), top 20:")
    for k, v in imp.head(20).items():
        print(f"  {k:>18}: {v:+.4f}")

    # pooled OOS calibration and per-seg AUC (T1 and T2)
    oos = scores[scores["p_T2"].notna()]
    for tier in ("T0", "T1", "T2"):
        p = oos[f"p_{tier}"]
        e = float(np.mean(np.abs(p.groupby(pd.qcut(p, 10, duplicates="drop"),
                                           observed=True).transform("mean")
                                 - oos.groupby(pd.qcut(p, 10, duplicates="drop"),
                                               observed=True)["y"].transform("mean"))))
        print(f"pooled OOS {tier}: AUC {auc(oos['y'], p):.4f} | ECE~ {e:.4f}")
        res[f"oos_{tier}"] = {"auc": auc(oos["y"], p), "ece": e}
    print("per-seg pooled OOS AUC (T2): " + " | ".join(
        f"{s}:{auc(g['y'], g['p_T2']):.3f}" for s, g in oos.groupby("seg")))

    d0 = layer_dir("daily", CONFIGS)
    scores.to_parquet(d0 / "p4_scores.parquet", index=False, compression="zstd")
    out = ROOT / "outputs" / "p4_model.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)} + p4_scores.parquet "
          f"({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
