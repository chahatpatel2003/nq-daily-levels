#!/usr/bin/env python
"""Build the target-window path layer: 1-second high/low/close for 06:30-09:00 PT.

Touch labels come free from the excursion profile — reaching a price means
passing every price below it, so ``up_t`` already answers "was this zone
touched, and when" at any width. **Reaction** labels do not: what price does
after arriving is genuinely local and needs the path.

So this extracts just the 150 minutes that matter, 9,000 seconds per session,
~12 M rows for the whole corpus. That is small enough to hold in memory for a
whole experiment, versus 72 M rows in L1 — and it is the difference between a
width sweep taking seconds and taking an hour.

Prices are stored as offsets from the session's 06:25 reference, in ticks. A
zone is then a pair of offsets and every label is a vectorised comparison, with
no per-session price-scale bookkeeping.
"""
from __future__ import annotations

import argparse
import sys
import time as _time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.clock import load_calendar, target_window_utc, trade_dates      # noqa: E402
from nqlevels.io.raw import assert_not_raw                                     # noqa: E402
from nqlevels.io.store import layer_dir, month_key, read_layer                 # noqa: E402

CONFIGS = str(ROOT / "configs")
WINDOW_S = 150 * 60


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-05")
    ap.add_argument("--end", default="2026-06-17")
    a = ap.parse_args()
    cal = load_calendar(CONFIGS)
    ctx = pd.read_parquet(layer_dir("daily", CONFIGS) / "context.parquet",
                          columns=["session", "symbol", "ref_t"])
    ref = dict(zip(ctx["session"], ctx["ref_t"]))
    sym = dict(zip(ctx["session"], ctx["symbol"]))

    sessions = trade_dates(date.fromisoformat(a.start), date.fromisoformat(a.end), cal)
    months = sorted({month_key(s) for s in sessions})
    out, t0 = [], _time.time()

    for i, mk in enumerate(months):
        b = read_layer("bars_1s", CONFIGS, months=[mk],
                       columns=["session", "symbol", "ts", "high_t", "low_t", "close_t"])
        if b.empty:
            continue
        b["ts"] = pd.DatetimeIndex(b["ts"]).tz_convert("UTC")
        for s in [x for x in sessions if month_key(x) == mk]:
            if s not in ref or not np.isfinite(ref[s]):
                continue
            t_lo, _ = target_window_utc(s, cal)
            t_lo = pd.Timestamp(t_lo)
            w = b[(b["session"] == s) & (b["symbol"] == sym[s])
                  & (b["ts"] >= t_lo) & (b["ts"] < t_lo + pd.Timedelta(seconds=WINDOW_S))]
            if w.empty:
                continue
            r = ref[s]
            out.append(pd.DataFrame({
                "session": s,
                "sec": ((w["ts"] - t_lo).dt.total_seconds()).to_numpy("int32"),
                "hi": (w["high_t"].to_numpy("float64") - r).astype("int32"),
                "lo": (w["low_t"].to_numpy("float64") - r).astype("int32"),
                "cl": (w["close_t"].to_numpy("float64") - r).astype("int32"),
            }))
        if i % 12 == 0:
            print(f"  {mk}: {sum(len(x) for x in out):,} rows, {_time.time()-t0:.0f}s", flush=True)

    p = pd.concat(out, ignore_index=True)
    dst = assert_not_raw(layer_dir("daily", CONFIGS) / "paths.parquet")
    p.to_parquet(dst, index=False, compression="zstd")
    print(f"\npaths: {len(p):,} rows over {p['session'].nunique()} sessions, "
          f"{dst.stat().st_size/1e6:.0f} MB, {(_time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
