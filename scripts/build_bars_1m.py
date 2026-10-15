#!/usr/bin/env python
"""Derive a 1-minute bar layer from L1.

Level *construction* never needs sub-minute data: highs, lows, volumes and
VWAPs aggregate exactly, so an overnight high computed from 1-minute bars is
identical to one computed from 1-second bars. Only *reaction* measurement needs
the fine path, and that lives in `paths.parquet`.

Collapsing 72 M 1-second rows to 1.2 M 1-minute rows makes the multi-session
lookback that swing and profile levels depend on cheap enough to run over the
whole corpus, and it is also the natural base for any coarser resolution — 5 m,
15 m, hourly, or a custom rolling window — since all of those are groupbys on
this table rather than another pass over L1.
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.io.raw import assert_not_raw                                    # noqa: E402
from nqlevels.io.store import layer_dir, list_partitions, read_layer          # noqa: E402

CONFIGS = str(ROOT / "configs")


def main() -> int:
    months = list_partitions("bars_1s", CONFIGS)
    out, t0 = [], _time.time()
    for i, mk in enumerate(months):
        b = read_layer("bars_1s", CONFIGS, months=[mk])
        if b.empty:
            continue
        b["ts"] = pd.DatetimeIndex(b["ts"]).tz_convert("UTC")
        b["m"] = b["ts"].dt.floor("1min")
        g = b.groupby(["session", "symbol", "is_front", "m"], sort=True, observed=True).agg(
            open_t=("open_t", "first"), high_t=("high_t", "max"),
            low_t=("low_t", "min"), close_t=("close_t", "last"),
            volume=("volume", "sum"), n_trades=("n_trades", "sum"),
            buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
            notional=("notional", "sum"), n_sec=("close_t", "size")).reset_index()
        out.append(g.rename(columns={"m": "ts"}))
        if i % 12 == 0:
            print(f"  {mk}: {sum(len(x) for x in out):,} rows, {_time.time()-t0:.0f}s", flush=True)
    d = pd.concat(out, ignore_index=True)
    for c in ("open_t", "high_t", "low_t", "close_t"):
        d[c] = d[c].astype("int32")
    for c in ("volume", "n_trades", "buy_vol", "sell_vol", "n_sec"):
        d[c] = d[c].astype("int32")
    dst = assert_not_raw(layer_dir("daily", CONFIGS) / "bars_1m.parquet")
    d.to_parquet(dst, index=False, compression="zstd")
    print(f"\nbars_1m: {len(d):,} rows, {d['session'].nunique()} sessions, "
          f"{dst.stat().st_size/1e6:.0f} MB, {(_time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
