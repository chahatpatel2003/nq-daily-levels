#!/usr/bin/env python
"""P9 checkpoint models — when does the interaction become predictable, and
what does a calibrated confirmation probability buy at each threshold?

Per checkpoint τ: HGB on (pre-touch approach/zone/state features) +
(post-touch features at τ), walk-forward by year. Outputs: the AUC-vs-τ
information-arrival curve, pre/post ablations, calibration, and the full
threshold curve (retained share, realized strict rate, adverse so far,
median reaction remaining) at the working checkpoints.
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
KEY = ["session", "zid", "tap_sec"]
CPS = (15, 30, 60, 120, 180, 300, 600, 900)
PRE = ["app_speed_5m", "app_accel", "app_volx_5m", "app_delta_5m",
       "app_toward_30m", "app_range_5m", "app_retraces_15m", "rv60_atr",
       "atr_pctile", "range_used_atr", "vwap_dist_atr", "tod_pt_min", "seg",
       "dep_vel", "prior_succ_share_f", "mins_since_last", "tap_no",
       "age_min", "hw_atr", "dist_t0_t"]


def auc(y, s):
    y = np.asarray(y, "float64")
    s = np.asarray(s, "float64")
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size < 200 or y.min() == y.max():
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    tl = pd.read_parquet(d0 / "p9_timeline.parquet")
    re = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    it = pd.read_parquet(d0 / "interactions.parquet").drop_duplicates(KEY)
    npr = it["n_prior_A"] + it["n_prior_B"] + it["n_prior_C"]
    it["prior_succ_share_f"] = np.where(
        npr > 0, (it["n_prior_A"] + it["n_prior_B"]) / npr, 0.5)
    keep = [c for c in PRE if c in it.columns] + KEY
    m = tl.merge(re[KEY + ["y", "y_strict", "rev_ep_pts", "seg", "dist_t0_t"]],
                 on=KEY, how="inner", validate="1:1")
    m = m.merge(it[[c for c in keep if c not in ("seg", "dist_t0_t")]],
                on=KEY, how="left", validate="1:1")
    m["year"] = pd.to_datetime(m["session"].astype(str)).dt.year
    m = m[m["y_strict"].notna()].reset_index(drop=True)
    print(f"frame: {len(m):,} | base strict {float(m['y_strict'].mean()):.4f}")

    from sklearn.ensemble import HistGradientBoostingClassifier

    def fit_oos(feats, tag):
        p = np.full(len(m), np.nan)
        for Y in (2022, 2023, 2024, 2025, 2026):
            tr = m[m["year"] < Y]
            te = m[m["year"] == Y]
            if len(tr) < 8000 or len(te) < 1500:
                continue
            g = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=4,
                l2_regularization=1.0, random_state=0)
            g.fit(tr[feats].to_numpy("float64"), tr["y_strict"].to_numpy())
            p[te.index] = g.predict_proba(te[feats].to_numpy("float64"))[:, 1]
        m[f"p_{tag}"] = p
        return p

    pre_feats = [c for c in PRE if c in m.columns]
    res: dict = {"auc_curve": {}}
    print(f"\ninformation-arrival curve (walk-forward OOS AUC on strict):")
    print(f"{'tau':>6} {'pre-only':>9} {'post-only':>10} {'both':>7} "
          f"{'by-year(both)':>40}")
    a0 = auc(m["y_strict"], fit_oos(pre_feats, "pre"))
    for tau in CPS:
        post = [f"cp{tau}_{f}" for f in ("pen", "rev", "now", "above_s",
                "below_sh", "vel60", "delta_sh", "intx", "eff", "d60")]
        fit_oos(post, f"post{tau}")
        fit_oos(pre_feats + post, f"both{tau}")
        oos = m[m[f"p_both{tau}"].notna()]
        yr = " ".join(f"{y}:{auc(g['y_strict'], g[f'p_both{tau}']):.3f}"
                      for y, g in oos.groupby("year"))
        aP = auc(m["y_strict"], m[f"p_post{tau}"])
        aB = auc(m["y_strict"], m[f"p_both{tau}"])
        res["auc_curve"][int(tau)] = {"pre": a0, "post": aP, "both": aB}
        print(f"{tau:>6} {a0:9.4f} {aP:10.4f} {aB:7.4f} {yr:>40}")

    print(f"\nthreshold curves (both-model), remaining move accounting:")
    res["thresholds"] = {}
    for tau in (60, 180):
        col = f"p_both{tau}"
        oos = m[m[col].notna()].copy()
        oos["rem"] = (oos["rev_ep_pts"] - oos[f"cp{tau}_rev"]).clip(lower=0)
        print(f"  τ={tau}s  (share of touches with OOS score: "
              f"{len(oos)/len(m):.2f})")
        print(f"  {'thr':>5} {'share':>6} {'strict':>7} {'sweepY':>7} "
              f"{'adv_med':>8} {'rem_med':>8} {'rem_mean':>9}")
        for thr in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            q = oos[oos[col] >= thr]
            if len(q) < 300:
                continue
            S = q[q["y_strict"] == 1]
            row = {"share": float(len(q) / len(oos)),
                   "strict": float(q["y_strict"].mean()),
                   "sweep": float(q["y"].mean()),
                   "adv_med": float(q[f"cp{tau}_pen"].median()),
                   "rem_med": float(S["rem"].median()),
                   "rem_mean": float(S["rem"].mean()), "n": int(len(q))}
            res["thresholds"][f"t{tau}_{thr}"] = row
            print(f"  {thr:>5.2f} {row['share']:6.3f} {row['strict']:7.3f} "
                  f"{row['sweep']:7.3f} {row['adv_med']:8.1f} "
                  f"{row['rem_med']:8.1f} {row['rem_mean']:9.1f}")

    # calibration at tau=60
    oos = m[m["p_both60"].notna()]
    b = pd.qcut(oos["p_both60"], 10, duplicates="drop")
    cal = oos.groupby(b, observed=True).agg(p=("p_both60", "mean"),
                                            y=("y_strict", "mean"),
                                            n=("y_strict", "size"))
    ece = float((cal["n"] / cal["n"].sum() * (cal["p"] - cal["y"]).abs()).sum())
    print(f"\ncalibration τ=60: ECE {ece:.4f}")
    res["ece60"] = ece

    out = ROOT / "outputs" / "p9_model.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
