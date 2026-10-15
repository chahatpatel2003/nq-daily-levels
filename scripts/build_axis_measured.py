#!/usr/bin/env python
"""Measure every axis grid price once, at a fixed width, and store the result.

Downstream experiments (first-touch decay, regime conditioning, feature
modelling, ranking) all need the same measured table. Re-deriving it per
experiment costs a minute each time and — worse — invites the width or the
clean-label definition to drift silently between experiments.

Width comes from E2. Everything is measured on TRAIN + VALIDATION; the holdout
is not built here at all, so it cannot leak into an exploratory run by accident.
"""
from __future__ import annotations

import argparse
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.eval.baseline import E1_FEATURES, fit_baseline                   # noqa: E402
from nqlevels.eval.labeling import build_paths                                 # noqa: E402
from nqlevels.eval.quality import measure_candidates                           # noqa: E402
from nqlevels.eval.splits import assign, load_split_plan                        # noqa: E402
from nqlevels.io.raw import assert_not_raw                                      # noqa: E402
from nqlevels.io.store import layer_dir                                         # noqa: E402

CONFIGS = str(ROOT / "configs")
TICK = 0.25

# Context columns carried through for regime work. Pre-open only, by name.
CTX_COLS = [
    "session", "symbol", "ref_t", "atr14_t", "atr5_t", "atr_pctile",
    "globex_range_pre", "globex_rv_pre", "globex_ret_pre", "globex_volume_pre",
    "globex_delta_pre", "premarket_range_pre", "premarket_ret_pre",
    "london_range_pre", "last60m_ret_pre", "last30m_ret_pre", "last60m_range_pre",
    "prior_rth_range_prior", "prior_rth_rv_prior", "prior_rth_ret_prior",
    "gap_t_prior", "gap_atr_prior", "loc_in_on_pre", "loc_in_pdr_prior",
    "on_range_rel_atr", "on_vol_rel", "macro_volx_pre", "macro_range_pre",
    "dow", "month",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-mode", default="fixed", choices=("fixed", "atr"))
    ap.add_argument("--width", type=float, default=6.0,
                    help="half-width: ticks if fixed, multiple of ATR if atr")
    a = ap.parse_args()
    t0 = _time.time()

    d = layer_dir("daily", CONFIGS)
    ctx = pd.read_parquet(d / "context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    bk = assign(list(ctx["session"]), load_split_plan(CONFIGS))
    keep = set(bk["train"]) | set(bk["validation"])

    ax = pd.read_parquet(d / "axis.parquet")
    ax = ax[ax["session"].isin(keep)]
    have = [c for c in CTX_COLS if c in ctx.columns]
    ax = ax.merge(ctx[have], on="session", how="left").rename(columns={"atr14_t": "atr_t"})
    ax = ax[np.isfinite(ax["atr_t"])].reset_index(drop=True)

    trctx = ctx[ctx["session"].isin(bk["train"])].reset_index(drop=True)
    baseline = fit_baseline(trctx, features=list(E1_FEATURES))
    paths = build_paths(pd.read_parquet(d / "paths.parquet"))
    full_ctx = ctx[ctx["session"].isin(keep)].reset_index(drop=True)

    w = (np.full(len(ax), a.width) if a.width_mode == "fixed"
         else np.clip(a.width * ax["atr_t"].to_numpy("float64"), 1.0, 200.0))
    print(f"measuring {len(ax):,} grid prices over {ax['session'].nunique()} sessions "
          f"at half-width {np.median(w)*TICK:.2f} pts ({a.width_mode})")

    m = measure_candidates(ax, paths, w, baseline, full_ctx,
                           point_grid=(5, 10, 15, 20, 30),
                           clean_specs=((10, 5), (15, 10), (20, 10), (30, 15)))
    m["split"] = np.where(m["session"].isin(bk["train"]), "train", "validation")
    for c in m.columns:
        if m[c].dtype == "float64" and c not in ("off_t", "zlo_t", "zhi_t"):
            m[c] = m[c].astype("float32")
    dst = assert_not_raw(d / "axis_measured.parquet")
    m.to_parquet(dst, index=False, compression="zstd")
    print(f"\naxis_measured: {len(m):,} rows x {m.shape[1]} cols, "
          f"{dst.stat().st_size/1e6:.0f} MB, {(_time.time()-t0)/60:.1f} min")
    print(m.groupby("split").agg(n=("touched", "size"), touch=("touched", "mean"),
                                 clean=("clean", "mean")).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
