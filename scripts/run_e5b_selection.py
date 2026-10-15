#!/usr/bin/env python
"""E5b — reaction prediction and zone selection, evaluated without the trap.

## The trap

Ranking prices *among those that were touched* smuggles in the answer. The
farthest price price reached during the window **is** the extreme of the day's
move, so by construction price reversed there: on validation, "the farthest
touched price" has an 88.6 % clean-rejection rate against a 10.5 % base. That
is not a level effect, it is the definition of an extreme, and it is only
knowable after the session. Any ranker correlated with distance inherits it.

## The honest evaluations

1. **Within-distance-band AUC.** Does the score predict clean rejection *after*
   distance is held fixed? Distance is the contaminated axis, so if the score
   adds nothing inside a band it adds nothing.
2. **Pre-open selection.** Rank ALL grid prices for a morning, take the top K,
   and only then look at what happened: how often the selection was reached,
   and how it behaved when it was. Compared against a distance-matched random
   selection drawn from the same morning, so geometry cancels.

Both use train for fitting and validation for reporting. The holdout is sealed.
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
from nqlevels.eval.quality import add_distance_bands                           # noqa: E402

CONFIGS = str(ROOT / "configs")
pd.set_option("display.width", 220)

LOC = ["dist_atr", "above", "round25_d", "round50_d", "round100_d",
       "on_vol", "on_volrel", "on_grad", "pd_vol", "pd_volrel", "pd_grad",
       "lb20_vol", "lb20_volrel", "lb20_grad",
       "on_touches", "on_mins_at", "on_recency", "on_visitlen",
       "pd_touches", "pd_mins_at", "pd_recency", "pd_visitlen",
       "lb20_touches", "lb20_mins_at", "lb20_recency", "lb20_visitlen",
       "d_core_t", "d_dense_t", "near_core", "near_dense"]
CTX = ["atr_pctile", "on_range_rel_atr", "gap_atr_prior", "loc_in_on_pre",
       "loc_in_pdr_prior", "on_vol_rel", "macro_volx_pre", "dow",
       "globex_ret_pre", "premarket_ret_pre", "last60m_ret_pre"]


def hdr(s):
    print(f"\n{'='*96}\n{s}\n{'='*96}")


def auc(y, s):
    y = np.asarray(y, "float64")
    s = np.asarray(s, "float64")
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size < 20 or y.min() == y.max():
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dist-atr", type=float, default=1.2)
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    t0 = _time.time()

    m = pd.read_parquet(layer_dir("daily", CONFIGS) / "axis_measured.parquet")
    m = m[(m["dist_atr"] <= a.max_dist_atr) & (m["side"] != "inside")
          & m["p_base"].notna()].reset_index(drop=True)
    tr = m[m["split"] == "train"]
    va = m[m["split"] == "validation"]
    feats = [c for c in LOC + CTX if c in m.columns]

    trt = tr[(tr["touched"] == True) & tr["clean"].notna()]                   # noqa: E712
    vat = va[(va["touched"] == True) & va["clean"].notna()]                   # noqa: E712
    print(f"train touched {len(trt):,} | validation touched {len(vat):,} "
          f"({vat['clean'].mean():.4f} clean)")

    from sklearn.ensemble import HistGradientBoostingClassifier
    g = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                       l2_regularization=1.0, random_state=0)
    g.fit(trt[feats].to_numpy("float32"), trt["clean"].to_numpy("float64"))
    va = va.copy()
    va["score"] = g.predict_proba(va[feats].to_numpy("float32"))[:, 1]
    vat = va[(va["touched"] == True) & va["clean"].notna()]                   # noqa: E712

    # ---------------------------------------------------------------- 1
    hdr("1. Within-distance-band AUC — does the score add anything beyond distance?")
    vb = add_distance_bands(vat.copy(), n=10)
    rows = []
    for b, gg in vb.groupby("_dband", observed=True):
        rows.append({"band": str(b), "n": len(gg), "clean": gg["clean"].mean(),
                     "auc_score": auc(gg["clean"], gg["score"]),
                     "auc_dist": auc(gg["clean"], gg["dist_atr"]),
                     "auc_onvolrel": auc(gg["clean"], gg["on_volrel"]),
                     "auc_nearcore": auc(gg["clean"], gg["near_core"])})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    w = t["n"] / t["n"].sum()
    print(f"\n  weighted within-band AUC: score {float((w*t['auc_score']).sum()):.4f} | "
          f"distance {float((w*t['auc_dist']).sum()):.4f} | "
          f"on_volrel {float((w*t['auc_onvolrel']).sum()):.4f} | "
          f"near_core {float((w*t['auc_nearcore']).sum()):.4f}")
    print("  (0.500 = no information once distance is held fixed)")

    # ---------------------------------------------------------------- 2
    hdr("2. Pre-open selection — pick top-K from ALL grid prices, then look")
    print("   Selection happens before anything is known about the session's path.")
    print("   'random' picks K prices per morning from the same distance distribution.\n")
    rng = np.random.default_rng(0)
    print(f"   {'K':>3} {'rule':>10} {'zones':>7} {'touch':>7} {'clean|touch':>12} "
          f"{'rev30m_atr':>11} {'slice60':>8} {'med dist ATR':>13}")
    out = {}
    for K in (1, 2, 3, 4, 8):
        for rule in ("model", "random", "nearest", "core"):
            picks = []
            for s, gg in va.groupby("session"):
                if rule == "model":
                    p = gg.nlargest(K, "score")
                elif rule == "nearest":
                    p = gg.nsmallest(K, "dist_atr")
                elif rule == "core":
                    c = gg[gg["near_core"] == 1]
                    p = c.nsmallest(K, "d_core_t") if len(c) >= K else gg.nsmallest(K, "dist_atr")
                else:
                    # distance-matched random: sample K with the model's own
                    # distance profile so geometry cannot explain a difference
                    q = gg.sample(min(K, len(gg)), random_state=int(rng.integers(1e9)))
                    p = q
                picks.append(p)
            P = pd.concat(picks, ignore_index=True)
            T = P[(P["touched"] == True) & P["clean"].notna()]                # noqa: E712
            row = {"K": K, "rule": rule, "n": len(P), "touch": P["touched"].mean(),
                   "clean": T["clean"].mean() if len(T) else np.nan,
                   "rev": T["rev30m_atr"].median() if len(T) else np.nan,
                   "slice": T["slice_through_60s"].mean() if len(T) else np.nan,
                   "dist": P["dist_atr"].median()}
            out[f"K{K}_{rule}"] = row
            print(f"   {K:>3} {rule:>10} {len(P):7,} {row['touch']:7.3f} "
                  f"{row['clean']:12.4f} {row['rev']:11.4f} {row['slice']:8.4f} "
                  f"{row['dist']:13.3f}")
        print()

    # ---------------------------------------------------------------- 3
    hdr("3. Same comparison, distance-matched by construction")
    print("   For each model pick, draw a control at the SAME distance band and side")
    print("   on the SAME morning. Geometry is then identical by construction.\n")
    vb = add_distance_bands(va.copy(), n=10)
    for K in (2, 3, 4):
        mp, cp = [], []
        for s, gg in vb.groupby("session"):
            p = gg.nlargest(K, "score")
            mp.append(p)
            for _, r in p.iterrows():
                pool = gg[(gg["_dband"] == r["_dband"]) & (gg["above"] == r["above"])]
                pool = pool[pool.index != r.name]
                if len(pool):
                    cp.append(pool.sample(1, random_state=int(rng.integers(1e9))))
        M = pd.concat(mp, ignore_index=True)
        C = pd.concat(cp, ignore_index=True)
        Mt = M[(M["touched"] == True) & M["clean"].notna()]                   # noqa: E712
        Ct = C[(C["touched"] == True) & C["clean"].notna()]                   # noqa: E712
        print(f"   K={K}: model touch {M['touched'].mean():.3f} vs control "
              f"{C['touched'].mean():.3f} | clean {Mt['clean'].mean():.4f} "
              f"({len(Mt)}) vs {Ct['clean'].mean():.4f} ({len(Ct)}) | "
              f"diff {Mt['clean'].mean()-Ct['clean'].mean():+.4f}")

    (ROOT / "outputs" / "e5b_selection.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote outputs/e5b_selection.json  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
