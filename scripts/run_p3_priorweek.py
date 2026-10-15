#!/usr/bin/env python
"""P3-206 — prior-week/month levels, the pre-registered higher-power retest.

P2-106 found prior-week high/close mildly positive on BOTH splits (pwh
d_clean +0.064 train / +0.070 validation) but each kind alone was underpowered
(0.1-0.4 % axis coverage). Registered design:

  * levels: prior COMPLETED week's high/low/close and prior completed month's
    high/low/close, built from carry-adjusted session extremes, expressed in
    the morning's contract scale;
  * primary hypothesis: pooled at-prior-week-level (h ∪ l ∪ c, within 6t)
    clean_rate beats matched non-level prices at side × distance strata;
  * secondary: per kind, prior-month pooled, and untested-tonight condition.

Train first; the validation look only if train clears CI > 0.
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


def build_htf_levels(bars: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    """Prior completed-week and completed-month H/L/C per session, in a
    roll-free scale (subtract adj to compare across contracts)."""
    fb = bars[bars["is_front"]]
    ext = fb.groupby("session").agg(hi=("high_t", "max"), lo=("low_t", "min"),
                                    cl=("close_t", "last")).reset_index()
    ext = ext.merge(adj[["session", "adj_t"]], on="session", how="left")
    for c in ("hi", "lo", "cl"):
        ext[c] = ext[c] - ext["adj_t"]           # roll-free scale
    d = pd.to_datetime(ext["session"].astype(str))
    ext["wk"] = d.dt.isocalendar().year.astype(int) * 100 + d.dt.isocalendar().week.astype(int)
    ext["mo"] = d.dt.year * 100 + d.dt.month
    wk = ext.groupby("wk").agg(pwh=("hi", "max"), pwl=("lo", "min"),
                               pwc=("cl", "last")).reset_index()
    mo = ext.groupby("mo").agg(pmh=("hi", "max"), pml=("lo", "min"),
                               pmc=("cl", "last")).reset_index()
    wk_sorted = wk.sort_values("wk").reset_index(drop=True)
    mo_sorted = mo.sort_values("mo").reset_index(drop=True)
    wk_idx = {w: i for i, w in enumerate(wk_sorted["wk"])}
    mo_idx = {m: i for i, m in enumerate(mo_sorted["mo"])}
    rows = []
    for r in ext.itertuples():
        wi, mi = wk_idx[r.wk], mo_idx[r.mo]
        row = {"session": r.session}
        if wi > 0:
            p = wk_sorted.iloc[wi - 1]
            row.update(pwh=p["pwh"], pwl=p["pwl"], pwc=p["pwc"])
        if mi > 0:
            p = mo_sorted.iloc[mi - 1]
            row.update(pmh=p["pmh"], pml=p["pml"], pmc=p["pmc"])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", action="store_true")
    ap.add_argument("--max-dist-atr", type=float, default=1.2)
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    t0 = _time.time()
    split = "validation" if a.validation else "train"

    d0 = layer_dir("daily", CONFIGS)
    ctx = pd.read_parquet(d0 / "context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    ctxi = ctx.set_index("session")
    adj = pd.read_parquet(d0 / "roll_adjust.parquet")
    bars = pd.read_parquet(d0 / "bars_1m.parquet")
    htf = build_htf_levels(bars, adj).set_index("session")
    adji = adj.set_index("session")["adj_t"]

    m = pd.read_parquet(d0 / "axis_measured.parquet")
    mem = pd.read_parquet(d0 / "memory.parquet",
                          columns=["session", "off_t"])  # placeholder join sanity
    m = m[(m["split"] == split) & (m["dist_atr"] <= a.max_dist_atr)
          & (m["side"] != "inside") & m["p_base"].notna()].reset_index(drop=True)
    m = add_distance_bands(m, n=5)

    # express each level in the morning's scale and mark nearby grid rows
    kinds = ["pwh", "pwl", "pwc", "pmh", "pml", "pmc"]
    for k in kinds:
        lv = m["session"].map(htf[k] if k in htf.columns else pd.Series(dtype=float))
        lv = lv + m["session"].map(adji)                       # back to contract scale
        ref = m["session"].map(ctxi["ref_t"])
        m[f"at_{k}"] = ((m["off_t"] - (lv - ref)).abs() <= NEAR_T).fillna(False).astype("int8")
    m["at_pw"] = (m[["at_pwh", "at_pwl", "at_pwc"]].sum(axis=1) > 0).astype("int8")
    m["at_pm"] = (m[["at_pmh", "at_pml", "at_pmc"]].sum(axis=1) > 0).astype("int8")

    print(f"{split}: {len(m):,} rows, {m['session'].nunique()} sessions")
    for k in ["at_pw", "at_pm"] + [f"at_{x}" for x in kinds]:
        print(f"  {k:>8}: n {int(m[k].sum()):6,} "
              f"({float(m[k].mean()):.4f} of grid), sessions "
              f"{m.loc[m[k] == 1, 'session'].nunique()}")

    res = {"split": split, "contrasts": {}}
    print(f"\n  {'flag':>8} {'stat':>11} {'diff':>8} {'lo':>8} {'hi':>8} {'p>0':>6} {'n':>7}")
    for k in ["at_pw", "at_pm"] + [f"at_{x}" for x in kinds]:
        if m[k].sum() < 250:
            continue
        for stat in ("clean_rate", "attraction", "rev30m_atr", "slice_rate"):
            r = stratified_contrast(m, k, stat=stat, strata=("side", "_dband"),
                                    n_boot=a.boot, min_cell=15)
            res["contrasts"][f"{k}.{stat}"] = r
            if stat == "clean_rate" or abs(r.get("diff") or 0) > 0.008:
                print(f"  {k:>8} {stat:>11} {r['diff']:8.4f} {r['lo']:8.4f} "
                      f"{r['hi']:8.4f} {r['p_gt0']:6.3f} {r.get('n_flag', 0):7,}")

    # untested-tonight condition on the pooled pw family
    sub = m[m["at_pw"] == 1].copy()
    if len(sub) > 500:
        sub["_f"] = (sub["on_touches"].fillna(0) == 0).astype("int8")
        r = stratified_contrast(sub, "_f", stat="clean_rate",
                                strata=("side", "_dband"), n_boot=a.boot, min_cell=15)
        res["contrasts"]["pw.untested"] = r
        print(f"  {'pw|untstd':>8} {'clean_rate':>11} {r['diff']:8.4f} {r['lo']:8.4f} "
              f"{r['hi']:8.4f} {r['p_gt0']:6.3f} {r.get('n_flag', 0):7,}")

    out = ROOT / "outputs" / f"p3_priorweek_{split}.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
