#!/usr/bin/env python
"""P3-202 — does reaction memory mark tomorrow's strong prices?

Hypothesis (registered before any outcome is read): grid prices whose recent
history contains confirmed rejection events — especially large, fast,
direction-matched, or same-morning ones — have a higher clean-reaction rate
when touched than matched non-memory prices at the same distance and side.

Phase-2 discipline applies unchanged: contrasts are within distance-and-side
strata (P2-103), never conditioned on a ranking among touched rows (P2-108),
with session-clustered bootstrap and the |d| < 0.005 noise floor (P2-104).
A second strata variant additionally holds *visit intensity* (lb20_mins_at
tercile) fixed, because Phase 2 already established that visits alone carry
nothing — memory must add information beyond having-been-there.

Runs on TRAIN by default. --validation is the single confirmatory look and is
only to be used after the train result is logged.
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


def auc(y, s):
    y = np.asarray(y, "float64")
    s = np.asarray(s, "float64")
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size < 50 or y.min() == y.max():
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def banded_auc(d: pd.DataFrame, score: str, y: str = "clean") -> float:
    t = d[(d["touched"] == True) & d[y].notna()]                     # noqa: E712
    if not len(t):
        return np.nan
    rows = []
    for b, g in t.groupby("_dband", observed=True):
        a = auc(g[y], g[score])
        if np.isfinite(a):
            rows.append((len(g), a))
    if not rows:
        return np.nan
    n = np.array([r[0] for r in rows], "float64")
    v = np.array([r[1] for r in rows], "float64")
    return float((n * v).sum() / n.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", action="store_true",
                    help="the single confirmatory look — train must be logged first")
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
    m = add_distance_bands(m, n=10)
    # visit-intensity tercile as an extra stratum: does memory beat "was there"?
    lb = m["lb20_mins_at"].fillna(0.0)
    m["_lbter"] = pd.qcut(lb.rank(method="first"), 3, labels=False).astype("int8")

    print(f"{split}: {len(m):,} rows, {m['session'].nunique()} sessions | "
          f"touched {float(m['touched'].mean()):.3f} | "
          f"clean|touch {float(m.loc[m['touched'] == True, 'clean'].mean()):.4f}")

    # direction-matched memory: resistance history for zones above, support below
    up = m["above"].to_numpy() == 1
    m["mem2_dirw"] = np.where(up, m["mem2_wup"], m["mem2_wdn"])
    m["mem2_oppw"] = np.where(up, m["mem2_wdn"], m["mem2_wup"])

    # thresholds are train quantiles, frozen for any later validation look
    thr_path = ROOT / "outputs" / "p3_memory_thresholds.json"
    if split == "train":
        thr = {c: float(np.nanquantile(m[c], 0.90)) for c in
               ("mem2_w", "mem2_magw", "mem2_dirw", "mem4_w", "mem4_magw")}
        thr["mem02_mag_med"] = float(np.nanmedian(m.loc[m["mem02_n"] > 0, "mem02_mag"]))
        thr_path.write_text(json.dumps(thr, indent=2))
    else:
        thr = json.loads(thr_path.read_text())

    flags = {
        "mem_any": (m["mem2_n"] > 0),
        "mem_strong": (m["mem2_w"] >= thr["mem2_w"]),
        "mem_mag_strong": (m["mem2_magw"] >= thr["mem2_magw"]),
        "mem_dir_strong": (m["mem2_dirw"] >= thr["mem2_dirw"]),
        "mem4_strong": (m["mem4_w"] >= thr["mem4_w"]),
        "mem0_any": (m["mem02_n"] > 0),
        "mem0_strong": (m["mem02_n"] > 0) & (m["mem02_mag"] >= thr["mem02_mag_med"]),
    }
    for k, v in flags.items():
        m[k] = v.fillna(False).astype("int8") if hasattr(v, "fillna") else v.astype("int8")

    print("\nflag shares / composition")
    for k in flags:
        g = m[m[k] == 1]
        print(f"  {k:>15}: share {float(m[k].mean()):.4f} | near_core "
              f"{float(g['near_core'].mean()):.3f} | med dist {float(g['dist_atr'].median()):.3f} "
              f"| sessions {g['session'].nunique()}")

    print("\nwithin-distance-band AUC (clean | touch) — 0.500 = nothing")
    aucs = {}
    for c in ("mem2_w", "mem2_magw", "mem2_magmax", "mem2_dirw", "mem2_oppw",
              "mem4_w", "mem4_magw", "mem02_n", "lb20_mins_at", "dist_atr"):
        aucs[c] = banded_auc(m, c)
        print(f"  {c:>13}: {aucs[c]:.4f}")

    res = {"split": split, "n": len(m), "aucs": aucs, "thresholds": thr,
           "contrasts": {}}
    print("\nstratified contrasts (side x distance band)  [noise floor |d|<0.005]")
    hdr = f"  {'flag':>15} {'stat':>11} {'diff':>8} {'lo':>8} {'hi':>8} {'p>0':>6} {'n_flag':>8}"
    print(hdr)
    for k in flags:
        for stat in ("clean_rate", "attraction", "rev30m_atr", "slice_rate"):
            r = stratified_contrast(m, k, stat=stat, strata=("side", "_dband"),
                                    n_boot=a.boot, min_cell=40)
            res["contrasts"][f"{k}.{stat}"] = r
            if stat == "clean_rate" or abs(r.get("diff") or 0) > 0.004:
                print(f"  {k:>15} {stat:>11} {r['diff']:8.4f} {r['lo']:8.4f} "
                      f"{r['hi']:8.4f} {r['p_gt0']:6.3f} {r.get('n_flag', 0):8,}")

    print("\nsame, holding visit intensity fixed too (side x dband x lb20 tercile)")
    for k in ("mem_strong", "mem_mag_strong", "mem_dir_strong", "mem0_any",
              "mem0_strong"):
        r = stratified_contrast(m, k, stat="clean_rate",
                                strata=("side", "_dband", "_lbter"),
                                n_boot=a.boot, min_cell=40)
        res["contrasts"][f"{k}.clean_rate.lbfix"] = r
        print(f"  {k:>15} {'clean_rate':>11} {r['diff']:8.4f} {r['lo']:8.4f} "
              f"{r['hi']:8.4f} {r['p_gt0']:6.3f} {r.get('n_flag', 0):8,}")

    out = ROOT / "outputs" / f"p3_memory_{split}.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
