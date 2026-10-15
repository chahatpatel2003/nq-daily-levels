#!/usr/bin/env python
"""E1 — the distance/excursion baseline.

Fits and validates the null model that every later destination claim is scored
against: given pre-open context, how likely is NQ to travel a given distance
during 06:30–09:00 PT?

Train only for fitting and selection; validation for reporting. The holdout is
never read.
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

from nqlevels.eval.baseline import (DEFAULT_SCALE_FEATURES, HORIZONS,       # noqa: E402
                                    build_scale_features, calibration,
                                    fit_baseline, scores, touch_panel)
from nqlevels.eval.splits import assign, load_split_plan                     # noqa: E402
from nqlevels.io.store import layer_dir                                      # noqa: E402

CONFIGS = str(ROOT / "configs")
TICK = 0.25
pd.set_option("display.width", 200)


def hdr(s: str) -> None:
    print(f"\n{'='*78}\n{s}\n{'='*78}")


def main() -> int:
    ctx = pd.read_parquet(layer_dir("daily", CONFIGS) / "context.parquet")
    plan = load_split_plan(CONFIGS)
    buckets = assign(list(ctx["session"]), plan)
    tr = ctx[ctx["session"].isin(buckets["train"])].reset_index(drop=True)
    va = ctx[ctx["session"].isin(buckets["validation"])].reset_index(drop=True)
    print(f"context {len(ctx)} sessions | train {len(tr)} | validation {len(va)} "
          f"| holdout SEALED ({len(buckets['holdout'])} sessions, untouched)")

    # ---------------------------------------------------------------- shape
    hdr("1. What does the target window actually do?")
    for name, d in (("train", tr), ("validation", va)):
        r = (d["up150_out"] + d["dn150_out"]) * TICK
        u, dn = d["up150_out"] * TICK, d["dn150_out"] * TICK
        print(f"  {name:11s} range(pts)  med {r.median():6.1f}  "
              f"p10 {r.quantile(.1):6.1f}  p90 {r.quantile(.9):6.1f}   |  "
              f"up med {u.median():6.1f}  dn med {dn.median():6.1f}  |  "
              f"never-traded-above {100*(u<0).mean():4.1f}%  below {100*(dn<0).mean():4.1f}%")

    hdr("2. Which pre-open variable predicts the window range? (train only)")
    f = build_scale_features(tr)
    y = np.log(np.clip((tr["up150_out"] + tr["dn150_out"]).to_numpy("float64"), 1, None))
    rows = []
    for c in f.columns:
        v = f[c].to_numpy("float64")
        m = np.isfinite(v) & np.isfinite(y)
        if m.sum() > 100:
            rows.append({"feature": c, "corr_with_logR": float(np.corrcoef(v[m], y[m])[0, 1]),
                         "n": int(m.sum())})
    ranked = pd.DataFrame(rows).reindex(
        pd.DataFrame(rows)["corr_with_logR"].abs().sort_values(ascending=False).index)
    print(ranked.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ------------------------------------------------------- model variants
    hdr("3. Baseline variants — fitted on train, scored on validation")
    variants = {
        "V0_constant":  [],
        "V1_atr_only":  ["log_atr14"],
        "V2_on_range":  ["log_on_range"],
        "V3_atr_plus_on": ["log_atr14", "log_on_range"],
        "V4_full":      list(DEFAULT_SCALE_FEATURES),
    }
    results, models = {}, {}
    for name, feats in variants.items():
        if not feats:                      # constant scale: one dummy regressor
            trc, vac = tr.copy(), va.copy()
            trc["_const"] = 0.0
            vac["_const"] = 0.0
            import nqlevels.eval.baseline as B
            orig = B.build_scale_features
            B.build_scale_features = lambda d: pd.DataFrame({"_c": np.zeros(len(d))}, index=d.index)
            b = fit_baseline(trc, features=["_c"])
            pan_tr = touch_panel(trc, b)
            pan_va = touch_panel(vac, b)
            B.build_scale_features = orig
        else:
            b = fit_baseline(tr, features=feats)
            pan_tr, pan_va = touch_panel(tr, b), touch_panel(va, b)
        models[name] = b
        results[name] = {"train": scores(pan_tr), "validation": scores(pan_va)}
        s = results[name]["validation"]
        st = results[name]["train"]
        print(f"  {name:16s} train brier {st['brier']:.4f} skill {st['brier_skill']:+.3f} | "
              f"VAL brier {s['brier']:.4f} skill {s['brier_skill']:+.3f} "
              f"logloss {s['log_loss']:.4f} ECE {s['ece']:.4f} maxgap {s['max_gap']:+.3f} "
              f"(n={s['n']:,})")

    best = max(results, key=lambda k: results[k]["validation"]["brier_skill"])
    print(f"\n  best by validation Brier skill: {best}")

    # ------------------------------------------------------------ calibration
    hdr(f"4. Calibration of {best} on VALIDATION")
    b = models[best]
    pan = touch_panel(va, b)
    cal = calibration(pan, n_bins=12)
    print(cal.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    hdr("5. Calibration by distance and horizon (validation)")
    for h in HORIZONS:
        p = touch_panel(va, b, horizon=h)
        g = p.groupby(["side", "dist_atr"]).agg(n=("touched", "size"),
                                                pred=("p_pred", "mean"),
                                                obs=("touched", "mean"))
        g["gap"] = g["obs"] - g["pred"]
        print(f"\n  horizon {h} min   (overall pred {p['p_pred'].mean():.3f} "
              f"vs obs {p['touched'].mean():.3f})")
        print("    " + g.reset_index().to_string(index=False,
              float_format=lambda v: f"{v:.3f}").replace("\n", "\n    "))

    # ------------------------------------------------------------- stability
    hdr("6. Stability — is the normalised excursion distribution stationary?")
    allc = ctx[ctx["session"].isin(buckets["train"] + buckets["validation"])].copy()
    s_all = b.scale(allc)
    allc["z_up"] = allc["up150_out"] / s_all
    allc["z_dn"] = allc["dn150_out"] / s_all
    allc["year"] = pd.to_datetime(allc["session"]).dt.year
    g = allc.groupby("year").agg(n=("z_up", "size"),
                                 up_med=("z_up", "median"), up_p90=("z_up", lambda x: x.quantile(.9)),
                                 dn_med=("z_dn", "median"), dn_p90=("z_dn", lambda x: x.quantile(.9)))
    print(g.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n  per-year validation-style calibration of the same fitted model:")
    for yr, d in allc.groupby("year"):
        p = touch_panel(d, b)
        sc = scores(p)
        print(f"    {yr}  n={sc['n']:5d}  obs {sc['base_rate']:.3f}  "
              f"brier {sc['brier']:.4f}  skill {sc['brier_skill']:+.3f}  ECE {sc['ece']:.4f}")

    # ------------------------------------------- does regime shape help?
    hdr("7. Does letting the SHAPE vary by regime help? (validation)")
    tr2 = tr.copy()
    va2 = va.copy()
    for d in (tr2, va2):
        d["reg_gap"] = pd.cut(d["gap_atr_prior"], [-np.inf, -0.3, 0.3, np.inf],
                              labels=["gap_dn", "flat", "gap_up"]).astype(str)
        d["reg_onr"] = pd.cut(d["on_range_rel_atr"], [0, 0.5, 0.9, np.inf],
                              labels=["compressed", "normal", "expanded"]).astype(str)
    for col in ("reg_gap", "reg_onr"):
        bb = fit_baseline(tr2, features=list(DEFAULT_SCALE_FEATURES), bin_col=col)
        p = touch_panel(va2, bb, bin_col=col)
        sc = scores(p)
        print(f"  shape by {col:8s}: VAL brier {sc['brier']:.4f} skill {sc['brier_skill']:+.3f} "
              f"ECE {sc['ece']:.4f}   (pooled: {results[best]['validation']['brier']:.4f} / "
              f"{results[best]['validation']['brier_skill']:+.3f} / "
              f"{results[best]['validation']['ece']:.4f})")

    # ------------------------------------------------------------- artefact
    out = {"variants": results, "best": best,
           "scale_features": models[best].scale_model.features,
           "scale_coef": dict(zip(models[best].scale_model.features,
                                  map(float, models[best].scale_model.coef))),
           "resid_sd": models[best].scale_model.resid_sd,
           "n_train": len(tr), "n_validation": len(va),
           "calibration_validation": cal.to_dict("records")}
    p = ROOT / "outputs" / "e1_baseline.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
