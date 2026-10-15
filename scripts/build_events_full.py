#!/usr/bin/env python
"""Rebuild the rejection-event layer over the FULL corpus (2021-2026).

Identical detector and parameters to build_events.py; the only change is
that no split filter is applied — the holdout is spent (P3-209) and Phase 6
needs 2026 events as spatial history for walk-forward features. Writes
events_full.parquet; the original events.parquet is left untouched.
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.features.events import detect_events                             # noqa: E402
from nqlevels.io.raw import assert_not_raw                                     # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402

CONFIGS = str(ROOT / "configs")


def main() -> int:
    t0 = _time.time()
    d = layer_dir("daily", CONFIGS)
    ctx = pd.read_parquet(d / "context.parquet", columns=["session", "atr14_t", "symbol"])
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    ctxi = ctx.set_index("session")
    bars = pd.read_parquet(d / "bars_1m.parquet")
    bars["ts"] = pd.DatetimeIndex(bars["ts"]).tz_convert("UTC")
    out = []
    for s, g in bars[bars["is_front"]].groupby("session", sort=True):
        if s not in ctxi.index:
            continue
        atr = float(ctxi.loc[s, "atr14_t"])
        g = g.sort_values("ts").reset_index(drop=True)
        ev = detect_events(g, atr)
        if ev.empty:
            continue
        ev["ts"] = g["ts"].to_numpy()[ev["minute"].to_numpy()]
        ev["session"] = s
        ev["atr_t"] = atr
        out.append(ev)
    e = pd.concat(out, ignore_index=True)
    dst = assert_not_raw(d / "events_full.parquet")
    e.to_parquet(dst, index=False, compression="zstd")
    print(f"events_full: {len(e):,} rows over {e['session'].nunique()} sessions "
          f"({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
