#!/usr/bin/env python
"""P3-204 — conditional named levels: not "is PDH good" but "when is it good".

Registered conditions, all causal at 06:25 and all computed within the rows AT
the relevant levels (so every contrast is within-family, level vs same level
under a different condition, at matched side and distance):

  C1  untested overnight — the level price was never visited tonight
      (on_touches == 0), for prior-day-formed levels; the freshness story.
  C2  formed by a confirmed rejection — a direction-matched event from the
      immediately prior session sits at the level (mem2_last == 1 and
      direction-matched weight dominates).
  C3  overnight extremes: just-set (few touches) vs heavily retested.
  C4  round-number confluence — a prior-day level lying on a 100-point round.

Phase-2 P2-105 rejected these families *unconditionally*; P2-111 rejected
day-level regimes. This is the remaining cell: zone-level conditions.
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
NEAR_T = 6.0
PD_KINDS = ["dk_pdh", "dk_pdl", "dk_pdc", "dk_pd_poc", "dk_pd_vah", "dk_pd_val",
            "dk_pd_vwap", "dk_pd_globex_high", "dk_pd_globex_low"]


def contrast_within(sub: pd.DataFrame, mask: pd.Series, boot: int,
                    stats=("clean_rate", "rev30m_atr", "slice_rate")) -> dict:
    sub = sub.copy()
    sub["_f"] = mask.astype("int8")
    out = {}
    for st in stats:
        out[st] = stratified_contrast(sub, "_f", stat=st,
                                      strata=("side", "_dband"),
                                      n_boot=boot, min_cell=20)
    return out


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
    mem = pd.read_parquet(d0 / "memory.parquet")
    m = m.merge(mem, on=["session", "off_t"], how="left", validate="1:1")
    m = m[(m["split"] == split) & (m["dist_atr"] <= a.max_dist_atr)
          & (m["side"] != "inside") & m["p_base"].notna()].reset_index(drop=True)
    m = add_distance_bands(m, n=5)
    print(f"{split}: {len(m):,} rows, {m['session'].nunique()} sessions")

    res = {"split": split, "contrasts": {}}

    def report(name, sub, mask):
        n1, n0 = int(mask.sum()), int((~mask).sum())
        if min(n1, n0) < 300:
            print(f"  {name:>34}: underpowered ({n1} vs {n0})")
            return
        r = contrast_within(sub, mask, a.boot)
        res["contrasts"][name] = r
        c = r["clean_rate"]
        s = r["slice_rate"]
        print(f"  {name:>34}: clean {c['diff']:+.4f} [{c['lo']:+.4f},{c['hi']:+.4f}] "
              f"p>0 {c['p_gt0']:.2f} | slice {s['diff']:+.4f} "
              f"[{s['lo']:+.4f},{s['hi']:+.4f}] | n {n1:,}/{n0:,}")

    # ---- C1: untested overnight, at prior-day levels -----------------------
    at_pd = m[[k for k in PD_KINDS if k in m.columns]].min(axis=1) <= NEAR_T
    sub = m[at_pd].copy()
    untested = sub["on_touches"].fillna(0) == 0
    print("\nC1 — untested overnight (prior-day levels pooled, then headline kinds)")
    report("pd_any.untested_vs_tested", sub, untested)
    for k in ("dk_pdh", "dk_pdl", "dk_pd_vwap", "dk_pd_poc"):
        s2 = m[m[k] <= NEAR_T].copy()
        report(f"{k[3:]}.untested_vs_tested", s2, s2["on_touches"].fillna(0) == 0)

    # ---- C2: formed by a confirmed rejection yesterday ---------------------
    print("\nC2 — formed by a direction-matched rejection event yesterday")
    up = sub["above"].to_numpy() == 1
    dirw = np.where(up, sub["mem2_wup"], sub["mem2_wdn"])
    oppw = np.where(up, sub["mem2_wdn"], sub["mem2_wup"])
    formed = (sub["mem2_last"] == 1) & (dirw > oppw)
    report("pd_any.formed_by_rejection", sub, formed.fillna(False))
    s2 = m[(m[["dk_pdh", "dk_pdl"]].min(axis=1)) <= NEAR_T].copy()
    up2 = s2["above"].to_numpy() == 1
    d2 = np.where(up2, s2["mem2_wup"], s2["mem2_wdn"])
    o2 = np.where(up2, s2["mem2_wdn"], s2["mem2_wup"])
    report("pdh_pdl.formed_by_rejection", s2,
           ((s2["mem2_last"] == 1) & (d2 > o2)).fillna(False))

    # ---- C3: overnight extremes, just-set vs heavily retested --------------
    print("\nC3 — overnight extremes: lightly vs heavily tested tonight")
    s3 = m[(m[["dk_on_high", "dk_on_low"]].min(axis=1)) <= NEAR_T].copy()
    report("on_ext.light_vs_heavy", s3, s3["on_touches"].fillna(0) <= 2)

    # ---- C4: prior-day level on a 100-round --------------------------------
    print("\nC4 — prior-day level with 100-round confluence")
    report("pd_any.on_round100", sub, sub["dk_round_100"] <= NEAR_T)

    out = ROOT / "outputs" / f"p3_conditional_{split}.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
