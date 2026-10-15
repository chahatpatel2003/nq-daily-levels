#!/usr/bin/env python
"""E4 + target sensitivity — before concluding, check the verdict is not an
artefact of one reaction definition or of only ever looking at the first touch.

Three things:

1. **Target sensitivity.** `clean_10_5` is one arbitrary choice. If levels carry
   information, it should show up under *some* reasonable definition — larger
   thresholds, volatility-normalised reaction, pure excursion magnitude, or
   simply "did not slice straight through". Each target is run through the same
   distance-and-side stratified contrast against non-level prices.

2. **First touch versus repeat touch.** A level may only work the first time it
   is tested. Interactions are counted with hysteresis (price must clear an edge
   by a tick before a new touch is counted) so a price oscillating on the
   boundary is one interaction, not twelve.

3. **Regime conditioning.** If levels work only on particular mornings, the
   contrast should appear inside a regime even if it vanishes on average.

Train for exploration, validation for confirmation. Holdout sealed.
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

from nqlevels.eval.labeling import build_paths, measure_touch_sequence         # noqa: E402
from nqlevels.eval.quality import add_distance_bands, stratified_contrast      # noqa: E402
from nqlevels.eval.universe import CORE_KINDS, explode_kinds                   # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402

CONFIGS = str(ROOT / "configs")
TICK = 0.25
pd.set_option("display.width", 220)


def hdr(s):
    print(f"\n{'='*96}\n{s}\n{'='*96}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    t0 = _time.time()
    d = layer_dir("daily", CONFIGS)

    m = pd.read_parquet(d / "axis_measured.parquet")
    m = m[(m["dist_atr"] <= 1.2) & (m["side"] != "inside")].reset_index(drop=True)

    # ------------------------------------------------------------------ 1
    hdr("1. Target sensitivity — near_core vs matched non-level prices, many targets")
    print("   Each row: does being at a core level change this outcome, holding")
    print("   distance band and side fixed? |diff| < 0.005 is inside the noise floor.\n")
    rows = []
    for split in ("train", "validation"):
        s = m[m["split"] == split].copy()
        s = add_distance_bands(s, n=8)
        t = s[s["touched"] == True]                                   # noqa: E712
        atr = t["atr_t"]
        # build alternative targets
        s["y_clean105"] = s["clean_10_5"]
        s["y_clean1510"] = s.get("clean_15_10")
        s["y_clean2010"] = s.get("clean_20_10")
        s["y_clean3015"] = s.get("clean_30_15")
        s["y_rev_gt_pen"] = np.where(s["touched"] == True,                # noqa: E712
                                     (s["rev_1800s"] > s["max_penetration_t"]).astype(float),
                                     np.nan)
        s["y_rev25atr"] = np.where(s["touched"] == True,                  # noqa: E712
                                   (s["rev_1800s"] / s["atr_t"] > 0.25).astype(float), np.nan)
        s["y_rev40atr"] = np.where(s["touched"] == True,                  # noqa: E712
                                   (s["rev_1800s"] / s["atr_t"] > 0.40).astype(float), np.nan)
        s["y_noslice"] = np.where(s["touched"] == True,                   # noqa: E712
                                  1.0 - s["slice_through_60s"].astype(float), np.nan)
        s["y_noaccept"] = np.where(s["touched"] == True,                  # noqa: E712
                                   1.0 - s["accepted"].astype(float), np.nan)
        s["y_fast"] = np.where(s["touched"] == True,                      # noqa: E712
                               (s["rev_300s"] / s["atr_t"] > 0.10).astype(float), np.nan)
        targets = ["y_clean105", "y_clean1510", "y_clean2010", "y_clean3015",
                   "y_rev_gt_pen", "y_rev25atr", "y_rev40atr", "y_noslice",
                   "y_noaccept", "y_fast"]
        import nqlevels.eval.quality as Q
        for tg in targets:
            if tg not in s or s[tg].notna().sum() < 5000:
                continue
            Q.STRAT_STATS[tg] = (tg, "touched")
            r = stratified_contrast(s, "near_core", tg, n_boot=a.boot)
            base = float(s.loc[s["touched"] == True, tg].mean())          # noqa: E712
            rows.append({"split": split, "target": tg, "base_rate": base,
                         "diff": r["diff"], "lo": r["lo"], "hi": r["hi"],
                         "p_gt0": r["p_gt0"]})
    T = pd.DataFrame(rows)
    for split in ("train", "validation"):
        print(f"  -- {split}")
        print("  " + T[T["split"] == split][["target", "base_rate", "diff", "lo", "hi", "p_gt0"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}").replace("\n", "\n  "))
        print()

    # ------------------------------------------------------------------ 2
    hdr("2. First touch vs repeat touch (core levels, train+validation)")
    ctx = pd.read_parquet(d / "context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    cand = pd.read_parquet(d / "candidates_named.parquet")
    cx = explode_kinds(cand)
    cx = cx[cx["kind"].isin(CORE_KINDS)]
    cx = cx.merge(ctx[["session", "atr14_t"]], on="session", how="left").rename(
        columns={"atr14_t": "atr_t"})
    cx = cx[np.isfinite(cx["atr_t"]) & (cx["dist_atr"] <= 1.2)]
    keep = set(m["session"].unique())
    cx = cx[cx["session"].isin(keep)]
    cx = cx.drop_duplicates(subset=["session", "off_t"]).reset_index(drop=True)
    paths = build_paths(pd.read_parquet(d / "paths.parquet"))
    print(f"  measuring touch sequences for {len(cx):,} core zones "
          f"over {cx['session'].nunique()} sessions ...")
    recs = []
    for s, g in cx.groupby("session", sort=False):
        p = paths.get(s)
        if p is None:
            continue
        off = g["off_t"].to_numpy("float64")
        atr = float(g["atr_t"].iloc[0])
        for j in range(len(g)):
            for r in measure_touch_sequence(p, off[j] - 6, off[j] + 6, max_touches=4):
                r["session"] = s
                r["atr_t"] = atr
                recs.append(r)
    S = pd.DataFrame(recs)
    S["rev_atr"] = S["rev_1800s"] / S["atr_t"]
    S["tn"] = S["touch_no"].clip(upper=4)
    g = S.groupby("tn").agg(
        n=("fp_rev10", "size"), resolved=("fp_rev10", lambda x: x.notna().mean()),
        clean=("fp_rev10", "mean"), rev_atr=("rev_atr", "median"),
        pen=("max_penetration_t", "median"),
        prev_pen=("prev_penetration_t", "median"),
        secs_away=("secs_away", "median"))
    print("\n  reaction quality by touch number:")
    print("  " + g.to_string(float_format=lambda v: f"{v:.4f}").replace("\n", "\n  "))
    print("\n  ^ if a level is 'spent' after being tested, `clean` should fall with touch_no")

    # ------------------------------------------------------------------ 3
    hdr("3. Regime conditioning — does near_core work on any particular morning?")
    s = m[m["split"] == "train"].copy()
    s["y"] = s["clean_10_5"]
    import nqlevels.eval.quality as Q
    Q.STRAT_STATS["y"] = ("y", "touched")
    for name, col, bins, labels in (
        ("gap", "gap_atr_prior", [-np.inf, -0.3, 0.3, np.inf], ["gap_dn", "flat", "gap_up"]),
        ("on range", "on_range_rel_atr", [0, 0.5, 0.9, np.inf], ["compressed", "normal", "expanded"]),
        ("vol pct", "atr_pctile", [0, .33, .67, 1.0], ["low", "mid", "high"]),
        ("macro", "macro_volx_pre", [0, 3, 10, np.inf], ["quiet", "mild", "shock"]),
    ):
        if col not in s:
            continue
        s["_r"] = pd.cut(s[col], bins, labels=labels)
        print(f"  {name}:")
        for lab, gg in s.groupby("_r", observed=True):
            gg = add_distance_bands(gg.copy(), n=6)
            r = stratified_contrast(gg, "near_core", "y", n_boot=200)
            print(f"     {str(lab):>12}  n={len(gg):7,}  d_clean {r['diff']:+.5f} "
                  f"[{r['lo']:+.4f},{r['hi']:+.4f}]  p>0 {r['p_gt0']:.3f}")

    (ROOT / "outputs" / "e4_targets.json").write_text(
        json.dumps({"targets": T.to_dict("records"),
                    "touch_no": g.reset_index().to_dict("records")}, indent=2, default=str))
    print(f"\nwrote outputs/e4_targets.json  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
