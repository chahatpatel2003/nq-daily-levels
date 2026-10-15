#!/usr/bin/env python
"""P3-203 — do displacement origins mark strong zones, and when?

Hypotheses (registered before outcomes are read):
  H1  grid prices holding a displacement origin react more cleanly than
      matched non-origin prices at the same distance and side;
  H2  the effect is conditional: FRESH (never re-entered) intact origins are
      the strong case, consumed origins are dead;
  H3  origin strength (displacement magnitude / velocity / volume / delta)
      separates strong origins from weak ones *within* the origin family;
  H4  same-morning pre-open origins are the freshest case of all.

Same discipline as P3-202: side x distance-band strata, session-clustered
bootstrap, |d| < 0.005 noise floor, train first, one later validation look.
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

from nqlevels.eval.quality import add_distance_bands, stratified_contrast      # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402

CONFIGS = str(ROOT / "configs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", action="store_true")
    ap.add_argument("--max-dist-atr", type=float, default=1.2)
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    t0 = _time.time()
    split = "validation" if a.validation else "train"

    d0 = layer_dir("daily", CONFIGS)
    m = pd.read_parquet(d0 / "axis_measured.parquet")
    o = pd.read_parquet(d0 / "origins_at.parquet")
    m = m.merge(o, on=["session", "off_t"], how="left", validate="1:1")
    m = m[(m["split"] == split) & (m["dist_atr"] <= a.max_dist_atr)
          & (m["side"] != "inside") & m["p_base"].notna()].reset_index(drop=True)
    m = add_distance_bands(m, n=10)

    thr_path = ROOT / "outputs" / "p3_origin_thresholds.json"
    has = m["orig_mag"].notna()
    if split == "train":
        thr = {"mag_q75": float(m.loc[has, "orig_mag"].quantile(.75)),
               "vel_med": float(m.loc[has, "orig_vel"].median()),
               "volx_med": float(m.loc[has, "orig_volx"].median())}
        thr_path.write_text(json.dumps(thr, indent=2))
    else:
        thr = json.loads(thr_path.read_text())

    fresh = has & (m["orig_fresh"] == 1) & (m["orig_intact"] == 1)
    strong = has & (m["orig_mag"] >= thr["mag_q75"])
    flags = {
        "orig_any": has,
        "orig_intact": has & (m["orig_intact"] == 1),
        "orig_fresh": fresh & (m["orig_sameday"] == 0),
        "orig_fresh_strong": fresh & strong & (m["orig_sameday"] == 0),
        "orig_recent": has & (m["orig_age"].between(1, 5)) & (m["orig_intact"] == 1),
        "orig_recent_fresh": fresh & (m["orig_age"].between(1, 5)),
        "orig_consumed": has & (m["orig_consumed"] == 1),
        "orig_broken": has & (m["orig_intact"] == 0),
        "orig_sameday": has & (m["orig_sameday"] == 1) & (m["orig_fresh"] == 1)
                        & (m["orig_intact"] == 1),
        # the pooled fresh concept: any-age unmitigated intact origin,
        # same-morning ones included — one hypothesis, maximum power
        "orig_freshall": has & (m["orig_fresh"] == 1) & (m["orig_intact"] == 1),
    }
    for k, v in flags.items():
        m[k] = v.fillna(False).astype("int8")

    print(f"{split}: {len(m):,} rows, {m['session'].nunique()} sessions")
    print("\nflag shares / composition")
    for k in flags:
        g = m[m[k] == 1]
        if not len(g):
            print(f"  {k:>18}: empty")
            continue
        print(f"  {k:>18}: n {len(g):7,} | sessions {g['session'].nunique():4} | "
              f"touch {float(g['touched'].mean()):.3f} | med dist "
              f"{float(g['dist_atr'].median()):.3f} | near_core {float(g['near_core'].mean()):.3f}")

    res = {"split": split, "thresholds": thr, "contrasts": {}}
    print(f"\nstratified contrasts vs non-origin prices (side x dband) "
          f"[noise floor |d|<0.005]")
    print(f"  {'flag':>18} {'stat':>11} {'diff':>8} {'lo':>8} {'hi':>8} {'p>0':>6} {'n_flag':>8}")
    for k in flags:
        if not m[k].sum():
            continue
        for stat in ("clean_rate", "attraction", "rev30m_atr", "slice_rate"):
            r = stratified_contrast(m, k, stat=stat, strata=("side", "_dband"),
                                    n_boot=a.boot, min_cell=25)
            res["contrasts"][f"{k}.{stat}"] = r
            if stat == "clean_rate" or abs(r.get("diff") or 0) > 0.006:
                print(f"  {k:>18} {stat:>11} {r['diff']:8.4f} {r['lo']:8.4f} "
                      f"{r['hi']:8.4f} {r['p_gt0']:6.3f} {r.get('n_flag', 0):8,}")

    # low-count flags again at coarser strata — same matching idea, cells that
    # actually fill; reported separately so the two variants are comparable
    print("\nlow-power flags at 5 distance bands, min_cell 20")
    m5 = add_distance_bands(m.drop(columns=["_dband"]), n=5)
    for k in flags:
        if not (0 < m[k].sum() < 8000):
            continue
        for stat in ("clean_rate", "slice_rate"):
            r = stratified_contrast(m5, k, stat=stat, strata=("side", "_dband"),
                                    n_boot=a.boot, min_cell=20)
            res["contrasts"][f"{k}.{stat}.b5"] = r
            print(f"  {k:>18} {stat:>11} {r['diff']:8.4f} {r['lo']:8.4f} "
                  f"{r['hi']:8.4f} {r['p_gt0']:6.3f} {r.get('n_flag', 0):8,}")

    # within-family conditionals: only origin rows, is fresh/strong better?
    print("\nwithin-origin conditionals (origin rows only)")
    oo = m[has.to_numpy()].copy()
    for k, mask in (("fresh_vs_rest", (oo["orig_fresh"] == 1) & (oo["orig_intact"] == 1)),
                    ("strong_vs_weak", oo["orig_mag"] >= thr["mag_q75"]),
                    ("fast_vs_slow", oo["orig_vel"] >= thr["vel_med"]),
                    ("hivol_vs_lo", oo["orig_volx"] >= thr["volx_med"]),
                    ("intact_vs_broken", oo["orig_intact"] == 1),
                    ("young_vs_old", oo["orig_age"] <= 5)):
        oo["_f"] = mask.astype("int8")
        r = stratified_contrast(oo, "_f", stat="clean_rate",
                                strata=("side", "_dband"), n_boot=a.boot, min_cell=25)
        res["contrasts"][f"cond.{k}"] = r
        print(f"  {k:>18} {'clean_rate':>11} {r['diff']:8.4f} {r['lo']:8.4f} "
              f"{r['hi']:8.4f} {r['p_gt0']:6.3f} {r.get('n_flag', 0):8,}")

    out = ROOT / "outputs" / f"p3_origins_{split}.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
