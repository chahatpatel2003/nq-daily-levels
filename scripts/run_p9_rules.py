#!/usr/bin/env python
"""P9 mechanical confirmation rules — reliability × timeliness for every
registered cell.

For each rule: fire share, precision (strict & sweep-inclusive), recall over
successes, FP rate over failures, latency, adverse-before-fire, reaction
completed before fire, and reaction REMAINING after fire (median/mean/%),
per year and per session block for the leaders. R6 is evaluated as a
failure predictor (disqualifier).
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
SEG_N = {0: "asia", 1: "london", 2: "premkt", 3: "ny-am", 4: "ny-pm"}


def load():
    d0 = layer_dir("daily", CONFIGS)
    tl = pd.read_parquet(d0 / "p9_timeline.parquet")
    re = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    lab = re[KEY + ["y", "y_strict", "rev_ep_pts", "seg"]].copy()
    m = tl.merge(lab, on=KEY, how="inner", validate="1:1")
    m["year"] = pd.to_datetime(m["session"].astype(str)).dt.year
    return m


def eval_rule(m: pd.DataFrame, name: str) -> dict:
    t = m[f"{name}_t"]
    fired = t.notna()
    if fired.sum() < 500:
        return {}
    F = m[fired]
    S = m[m["y_strict"] == 1]
    FS = F[F["y_strict"] == 1]
    rem = (FS["rev_ep_pts"] - FS[f"{name}_done"]).clip(lower=0)
    yr = {}
    for y_, gy in F.groupby("year"):
        yr[int(y_)] = float(gy["y_strict"].mean())
    return {"fire_share": float(fired.mean()),
            "prec_strict": float(F["y_strict"].mean()),
            "prec_sweep": float(F["y"].mean()),
            "recall": float(FS.shape[0] / max(1, len(S))),
            "fp_rate": float(m[(m["y_strict"] == 0)][f"{name}_t"].notna().mean()),
            "lat_med": float(F[f"{name}_t"].median()),
            "lat_p75": float(F[f"{name}_t"].quantile(.75)),
            "adv_med": float(F[f"{name}_adv"].median()),
            "done_med": float(FS[f"{name}_done"].median()),
            "rem_med": float(rem.median()), "rem_mean": float(rem.mean()),
            "rem_pct": float((rem / FS["rev_ep_pts"].clip(lower=0.1)).median()),
            "by_year": yr, "n_fired": int(fired.sum())}


def main() -> int:
    t0 = _time.time()
    m = load()
    base_s = float(m["y_strict"].mean())
    base_y = float(m["y"].mean())
    print(f"interactions: {len(m):,} | base strict {base_s:.4f} | "
          f"sweep-incl {base_y:.4f}")
    rules = [c[:-2] for c in m.columns if c.endswith("_t")
             and c not in ("fb_t",)]
    res = {"base_strict": base_s, "rules": {}}
    print(f"\n{'rule':>14} {'fire%':>6} {'precS':>6} {'precY':>6} {'recall':>7} "
          f"{'FP%':>6} {'lat':>5} {'adv':>5} {'done':>5} {'remM':>6} {'rem%':>5} "
          f"{'yr_min':>6}")
    rows = []
    for name in sorted(rules):
        r = eval_rule(m, name)
        if not r:
            continue
        res["rules"][name] = r
        yr_min = min(r["by_year"].values()) if r["by_year"] else np.nan
        rows.append((name, r))
        print(f"{name:>14} {r['fire_share']:6.3f} {r['prec_strict']:6.3f} "
              f"{r['prec_sweep']:6.3f} {r['recall']:7.3f} {r['fp_rate']:6.3f} "
              f"{r['lat_med']:5.0f} {r['adv_med']:5.1f} {r['done_med']:5.1f} "
              f"{r['rem_med']:6.1f} {r['rem_pct']:5.2f} {yr_min:6.3f}")

    # R6 as failure predictor
    print("\nR6 acceptance filter (fires -> FAILURE predictor):")
    for name in ("r6T120", "r6T300"):
        t = m[f"{name}_t"].notna()
        print(f"  {name}: fires {float(t.mean()):.3f} | P(fail|fired) "
              f"{float(1-m[t]['y_strict'].mean()):.3f} vs base fail {1-base_s:.3f}")

    # simple combined framework: direct = r1h60, sweep = r3nearh60,
    # first-of-either as THE confirmation
    print("\ncombined framework (first of R1h60 / R3near-h60):")
    ct = m[["r1h60_t", "r3nearh60_t"]].min(axis=1)
    which = np.where(m["r1h60_t"].fillna(9e9) <= m["r3nearh60_t"].fillna(9e9),
                     "r1h60", "r3near")
    F = m[ct.notna()].copy()
    F["_ct"] = ct[ct.notna()]
    F["_adv"] = np.where(which[ct.notna()] == "r1h60", F["r1h60_adv"],
                         F["r3nearh60_adv"])
    F["_done"] = np.where(which[ct.notna()] == "r1h60", F["r1h60_done"],
                          F["r3nearh60_done"])
    FS = F[F["y_strict"] == 1]
    rem = (FS["rev_ep_pts"] - FS["_done"]).clip(lower=0)
    print(f"  fire {float(ct.notna().mean()):.3f} | precS "
          f"{float(F['y_strict'].mean()):.3f} | precY {float(F['y'].mean()):.3f} | "
          f"recall {len(FS)/int((m['y_strict']==1).sum()):.3f} | lat med "
          f"{float(F['_ct'].median()):.0f}s | adv med {float(F['_adv'].median()):.1f} | "
          f"rem med {float(rem.median()):.1f} pts ({float((rem/FS['rev_ep_pts'].clip(lower=.1)).median()):.0%})")
    res["combined"] = {"fire": float(ct.notna().mean()),
                       "prec_strict": float(F["y_strict"].mean()),
                       "rem_med": float(rem.median())}
    print("  by year: " + " ".join(f"{y}:{g['y_strict'].mean():.3f}"
                                   for y, g in F.groupby("year")))
    print("  by seg:  " + " ".join(
        f"{SEG_N.get(int(s), s)}:{g['y_strict'].mean():.3f}(n={len(g)})"
        for s, g in F.groupby("seg")))

    out = ROOT / "outputs" / "p9_rules.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
