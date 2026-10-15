#!/usr/bin/env python
"""Build sweep_labels.parquet: path-aware sweep/reclaim metrics for every
touched grid zone in axis_measured (train + validation; the holdout was never
measured and is not built).

One walk per zone produces the raw quantities; every taxonomy definition is
applied downstream (labels/sweep.classify), so the definition grid never
re-reads the paths.
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.eval.labeling import build_paths                                 # noqa: E402
from nqlevels.io.raw import assert_not_raw                                     # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402
from nqlevels.labels.sweep import sweep_metrics                                # noqa: E402

CONFIGS = str(ROOT / "configs")


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    ax = pd.read_parquet(d0 / "axis_measured.parquet",
                         columns=["session", "off_t", "zlo_t", "zhi_t",
                                  "side", "touched"])
    ax = ax[(ax["touched"] == True) & (ax["side"] != "inside")]      # noqa: E712
    print(f"touched zones to walk: {len(ax):,} over {ax['session'].nunique()} sessions")

    paths = build_paths(pd.read_parquet(d0 / "paths.parquet"))
    out, n_done = [], 0
    for s, g in ax.groupby("session", sort=True):
        p = paths.get(s)
        if p is None:
            continue
        zlo = g["zlo_t"].to_numpy("float64")
        zhi = g["zhi_t"].to_numpy("float64")
        off = g["off_t"].to_numpy("float64")
        rows = []
        for j in range(len(g)):
            i = p.first_touch(zlo[j], zhi[j])
            if i is None:                      # touched per stored label; guard anyway
                continue
            m = sweep_metrics(p, zlo[j], zhi[j], i)
            m["off_t"] = off[j]
            rows.append(m)
        f = pd.DataFrame(rows)
        f["session"] = s
        out.append(f)
        n_done += len(rows)
        if len(out) % 200 == 0:
            print(f"  {s}: {n_done:,} zones, {_time.time()-t0:.0f}s", flush=True)

    m = pd.concat(out, ignore_index=True)
    for c in m.columns:
        if c in ("session", "side"):
            continue
        m[c] = pd.to_numeric(m[c], errors="coerce").astype("float32")
    dst = assert_not_raw(d0 / "sweep_labels.parquet")
    m.to_parquet(dst, index=False, compression="zstd")
    print(f"\nsweep_labels: {len(m):,} rows x {m.shape[1]} cols | "
          f"{dst.stat().st_size/1e6:.0f} MB | {(_time.time()-t0)/60:.1f} min")
    fb = m["first_beyond"].notna().mean()
    ne = m["t_reclaim_ne"].notna().mean()
    print(f"went beyond far edge: {fb:.3f} | of all, reclaimed near edge at some point: {ne:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
