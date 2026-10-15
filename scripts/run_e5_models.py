#!/usr/bin/env python
"""E5/E6 — does ANY pre-open information predict zone quality?

The decisive experiment. Named level families showed no reliable edge (E3), so
the question becomes whether the price axis carries usable information at all,
in any form, under any name.

Two targets, kept separate:

**Destination** — `touched`. The bar is not "beat chance", it is **beat E1**,
which already predicts touch from distance and session scale alone. So the
metric is Brier skill *relative to the E1 prediction*, and E1's probability is
a feature so the model starts from it rather than relearning it.

**Reaction** — `clean` among touched prices, against the constant rate.

And one question that decides what the product can be:

**Is the signal per-price or per-session?** A model can score well by learning
"this is a clean-reacting morning" without knowing anything about *which* price
matters. That would still be useful — as a confidence reading — but it would
not let anyone pick 2-4 zones. Three diagnostics separate them: the ICC of the
outcome, **within-session AUC** (ranking prices inside a morning, which removes
all session-level variation), and a session-shuffled control.

Discipline: interpretable logistic first, gradient boosting only as a check for
missed interactions; a shuffled-label control; validation used for reporting,
never for tuning; holdout untouched.
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

CONFIGS = str(ROOT / "configs")
pd.set_option("display.width", 220)

LOC_FEATURES = [
    "dist_atr", "above", "round25_d", "round50_d", "round100_d",
    "on_vol", "on_volrel", "on_grad", "pd_vol", "pd_volrel", "pd_grad",
    "lb20_vol", "lb20_volrel", "lb20_grad",
    "on_touches", "on_mins_at", "on_recency", "on_visitlen",
    "pd_touches", "pd_mins_at", "pd_recency", "pd_visitlen",
    "lb20_touches", "lb20_mins_at", "lb20_recency", "lb20_visitlen",
    "d_core_t", "d_dense_t", "near_core", "near_dense",
]
CTX_FEATURES = [
    "atr_pctile", "on_range_rel_atr", "gap_atr_prior", "loc_in_on_pre",
    "loc_in_pdr_prior", "on_vol_rel", "macro_volx_pre", "dow",
    "globex_ret_pre", "premarket_ret_pre", "last60m_ret_pre",
]


def hdr(s: str) -> None:
    print(f"\n{'='*94}\n{s}\n{'='*94}")


def brier_skill(y, p, ref) -> float:
    y = np.asarray(y, "float64")
    den = np.mean((np.asarray(ref, "float64") - y) ** 2)
    return float(1 - np.mean((np.asarray(p, "float64") - y) ** 2) / den) if den > 0 else np.nan


def auc(y, s) -> float:
    y = np.asarray(y, dtype="float64")
    s = np.asarray(s, dtype="float64")
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size == 0 or y.min() == y.max():
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_session_auc(d: pd.DataFrame, score: str, target: str,
                       min_n: int = 30) -> tuple:
    """Mean AUC computed inside each session, then averaged over sessions.

    This is the number that decides whether zones can be *ranked on a morning*.
    Pooled AUC can be high purely because the model recognises clean-reacting
    days; only within-session ranking says it knows which price matters.
    """
    vals, ns = [], []
    for s, g in d.groupby("session"):
        y = g[target].to_numpy("float64")
        if len(g) < min_n or np.nanmin(y) == np.nanmax(y):
            continue
        a = auc(y, g[score].to_numpy("float64"))
        if np.isfinite(a):
            vals.append(a)
            ns.append(len(g))
    if not vals:
        return np.nan, 0, np.nan
    vals = np.array(vals)
    return float(vals.mean()), len(vals), float(vals.std() / np.sqrt(len(vals)))


def icc(d: pd.DataFrame, col: str) -> float:
    """Share of outcome variance that is between sessions rather than within."""
    g = d.groupby("session")[col]
    mu = d[col].mean()
    n = g.size()
    between = float((n * (g.mean() - mu) ** 2).sum())
    total = float(((d[col] - mu) ** 2).sum())
    return between / total if total > 0 else np.nan


def prep(d, feats):
    x = d[feats].to_numpy("float64")
    med = np.nanmedian(x, axis=0)
    return np.where(np.isfinite(x), x, med)


def fit_logistic(x, y, l2=1.0, iters=40):
    mu, sd = x.mean(0), np.where(x.std(0) > 1e-9, x.std(0), 1.0)
    xs = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    b = np.zeros(xs.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(xs @ b, -30, 30)))
        g = xs.T @ (p - y) / len(y) + l2 * np.r_[0, b[1:]] / len(y)
        h = (xs * (p * (1 - p))[:, None]).T @ xs / len(y) + l2 * np.eye(xs.shape[1]) / len(y)
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        b -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return b, mu, sd


def pred_logistic(model, x):
    b, mu, sd = model
    xs = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    return 1 / (1 + np.exp(-np.clip(xs @ b, -30, 30)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dist-atr", type=float, default=1.2)
    a = ap.parse_args()
    t0 = _time.time()

    m = pd.read_parquet(layer_dir("daily", CONFIGS) / "axis_measured.parquet")
    # side == 'inside' is the grid price straddling the reference: there is no
    # arrival to predict, and E1 has no baseline for it.
    m = m[(m["dist_atr"] <= a.max_dist_atr) & (m["side"] != "inside")
          & m["p_base"].notna()].reset_index(drop=True)
    tr = m[m["split"] == "train"].reset_index(drop=True)
    va = m[m["split"] == "validation"].reset_index(drop=True)
    print(f"train {len(tr):,} rows / {tr['session'].nunique()} sessions | "
          f"validation {len(va):,} / {va['session'].nunique()}")

    floc = [c for c in LOC_FEATURES if c in m.columns]
    fctx = [c for c in CTX_FEATURES if c in m.columns]
    results = {}

    # ------------------------------------------------------------ DESTINATION
    hdr("1. DESTINATION — can any location feature beat E1 on touch?")
    y_tr, y_va = tr["touched"].to_numpy("float64"), va["touched"].to_numpy("float64")
    e1 = va["p_base"].to_numpy("float64")
    base = np.full_like(y_va, y_tr.mean())
    print(f"  base rate           : val Brier {np.mean((base-y_va)**2):.5f}")
    print(f"  E1 alone            : val Brier {np.mean((e1-y_va)**2):.5f}  "
          f"skill vs base {brier_skill(y_va, e1, base):+.4f}  AUC {auc(y_va, e1):.4f}")
    for name, fs in (("E1 + location", ["p_base"] + floc),
                     ("E1 + loc + ctx", ["p_base"] + floc + fctx)):
        mdl = fit_logistic(prep(tr, fs), y_tr)
        pv = pred_logistic(mdl, prep(va, fs))
        print(f"  {name:20s}: val Brier {np.mean((pv-y_va)**2):.5f}  "
              f"skill vs E1 {brier_skill(y_va, pv, e1):+.5f}  AUC {auc(y_va, pv):.4f}")
        results[f"dest::{name}"] = {"skill_vs_e1": brier_skill(y_va, pv, e1),
                                    "auc": auc(y_va, pv)}
    ws, nse, se = within_session_auc(va.assign(_s=e1), "_s", "touched")
    print(f"\n  E1 within-session AUC {ws:.4f} +-{1.96*se:.4f} over {nse} sessions "
          f"(ranking prices inside one morning)")

    # --------------------------------------------------------------- REACTION
    hdr("2. REACTION — predict clean rejection given the zone was reached")
    trt = tr[(tr["touched"] == True) & tr["clean"].notna()].reset_index(drop=True)  # noqa: E712
    vat = va[(va["touched"] == True) & va["clean"].notna()].reset_index(drop=True)  # noqa: E712
    yt, yv = trt["clean"].to_numpy("float64"), vat["clean"].to_numpy("float64")
    print(f"  resolved touches: train {len(trt):,} ({yt.mean():.3f} clean) | "
          f"validation {len(vat):,} ({yv.mean():.3f} clean)")
    print(f"\n  ICC of `clean` (share of variance BETWEEN sessions): "
          f"train {icc(trt, 'clean'):.4f}  val {icc(vat, 'clean'):.4f}")
    const = np.full_like(yv, yt.mean())
    combos = {"distance only": ["dist_atr", "above"], "location only": floc,
              "context only": fctx, "location+context": floc + fctx}
    print(f"\n  {'model':>18} {'valBrier':>9} {'skill':>9} {'AUCtr':>7} {'AUCval':>7} "
          f"{'withinAUC':>10} {'+-95%':>7}")
    print(f"  {'constant':>18} {np.mean((const-yv)**2):9.5f} {0.0:9.5f} "
          f"{0.5:7.4f} {0.5:7.4f} {0.5:10.4f} {0.0:7.4f}")
    scores = {}
    for name, fs in combos.items():
        fs = [c for c in fs if c in trt.columns]
        mdl = fit_logistic(prep(trt, fs), yt)
        pv, pt = pred_logistic(mdl, prep(vat, fs)), pred_logistic(mdl, prep(trt, fs))
        w, nw, sw = within_session_auc(vat.assign(_s=pv), "_s", "clean")
        print(f"  {name:>18} {np.mean((pv-yv)**2):9.5f} {brier_skill(yv,pv,const):+9.5f} "
              f"{auc(yt,pt):7.4f} {auc(yv,pv):7.4f} {w:10.4f} {1.96*sw:7.4f}")
        results[f"react::{name}"] = {"auc_val": auc(yv, pv), "within_auc": w,
                                     "skill": brier_skill(yv, pv, const)}
        scores[name] = pv

    hdr("3. Controls")
    rng = np.random.default_rng(3)
    fs = floc + fctx
    mdl = fit_logistic(prep(trt, fs), rng.permutation(yt))
    pv = pred_logistic(mdl, prep(vat, fs))
    w, _, sw = within_session_auc(vat.assign(_s=pv), "_s", "clean")
    print(f"  shuffled labels   : val AUC {auc(yv,pv):.4f}  within {w:.4f}   (expect 0.500)")
    # session-shuffled context: keep each session's context but attach it to a
    # different session's prices -> destroys any real session-level signal
    perm = {s: t for s, t in zip(sorted(vat["session"].unique()),
                                 rng.permutation(sorted(vat["session"].unique())))}
    fake = vat.copy()
    cmap = vat.groupby("session")[fctx].first()
    fake[fctx] = cmap.loc[[perm[s] for s in fake["session"]]].to_numpy()
    mdl = fit_logistic(prep(trt, fctx), yt)
    pv = pred_logistic(mdl, prep(fake, fctx))
    print(f"  context shuffled across sessions: val AUC {auc(yv,pv):.4f}   (expect 0.500)")

    hdr("4. Gradient boosting")
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        fs = floc + fctx
        g = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                           l2_regularization=1.0, random_state=0)
        g.fit(trt[fs].to_numpy("float32"), yt)
        pv = g.predict_proba(vat[fs].to_numpy("float32"))[:, 1]
        pt = g.predict_proba(trt[fs].to_numpy("float32"))[:, 1]
        w, nw, sw = within_session_auc(vat.assign(_s=pv), "_s", "clean")
        print(f"  HGB loc+ctx       : val Brier {np.mean((pv-yv)**2):.5f} "
              f"skill {brier_skill(yv,pv,const):+.5f}  AUC tr {auc(yt,pt):.4f} "
              f"val {auc(yv,pv):.4f}  within {w:.4f} +-{1.96*sw:.4f}")
        results["react::hgb"] = {"auc_val": auc(yv, pv), "within_auc": w}
        scores["hgb"] = pv
        try:
            from sklearn.inspection import permutation_importance
            sub = vat.sample(min(20000, len(vat)), random_state=0)
            pi = permutation_importance(g, sub[fs].to_numpy("float32"),
                                        sub["clean"].to_numpy("float64"),
                                        n_repeats=4, random_state=0, scoring="roc_auc")
            imp = pd.Series(pi.importances_mean, index=fs).sort_values(ascending=False)
            print("\n  top features by permutation AUC drop:")
            print("    " + imp.head(12).round(5).to_string().replace("\n", "\n    "))
        except Exception as e:                                    # noqa: BLE001
            print(f"  (importance unavailable: {e})")
    except ImportError:
        print("  sklearn unavailable")

    hdr("5. Selectivity — the product only needs the top few prices per morning")
    for name in ("location+context", "hgb"):
        if name not in scores:
            continue
        v = vat.assign(score=scores[name])
        print(f"\n  ranking = {name}   (validation base rate {yv.mean():.4f})")
        print(f"    {'top':>8} {'n':>8} {'sess':>6} {'clean':>7} {'lift':>7} "
              f"{'per-session picks':>18}")
        for q in (0.999, 0.995, 0.99, 0.95, 0.90):
            thr = v["score"].quantile(q)
            top = v[v["score"] >= thr]
            print(f"    {100*(1-q):7.2f}% {len(top):8,} {top['session'].nunique():6d} "
                  f"{top['clean'].mean():7.4f} {top['clean'].mean()-yv.mean():+7.4f} "
                  f"{len(top)/max(1,top['session'].nunique()):18.1f}")
        # per-session best pick only
        best = v.loc[v.groupby("session")["score"].idxmax()]
        print(f"    single best price per session: n={len(best)}  "
              f"clean {best['clean'].mean():.4f}  lift {best['clean'].mean()-yv.mean():+.4f}")

    (ROOT / "outputs" / "e5_models.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote outputs/e5_models.json  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
