#!/usr/bin/env python
"""Build L2: per-session pre-open context + RTH excursion profiles.

    python scripts/build_l2.py

Writes `daily/context.parquet` (one row per session) and
`daily/excursions.parquet` (one row per session-minute). Both are small enough
to live in memory for every later experiment, which is the point: after this
runs, no destination question needs to reopen the 1-second layer.
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

from nqlevels.build.context import (add_scales, excursion_profile,          # noqa: E402
                                    session_context, summarize_excursions)
from nqlevels.build.tape import window_bars                                  # noqa: E402
from nqlevels.clock import (load_calendar, prior_rth_window_utc,             # noqa: E402
                            prior_trade_date, trade_dates)
from nqlevels.io.store import layer_dir, month_key, read_layer               # noqa: E402
from nqlevels.io.raw import assert_not_raw                                   # noqa: E402

CONFIGS = str(ROOT / "configs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-05")
    ap.add_argument("--end", default="2026-06-17")
    a = ap.parse_args()
    cal = load_calendar(CONFIGS)
    sessions = trade_dates(date.fromisoformat(a.start), date.fromisoformat(a.end), cal)
    months = sorted({month_key(s) for s in sessions})
    print(f"{len(sessions)} sessions over {len(months)} months")

    ctx_rows, ex_frames = [], []
    prev_month_bars = pd.DataFrame()
    t0 = _time.time()

    for i, mk in enumerate(months):
        cur = read_layer("bars_1s", CONFIGS, months=[mk])
        if cur.empty:
            print(f"{mk}: no L1")
            continue
        cur["ts"] = pd.DatetimeIndex(cur["ts"]).tz_convert("UTC")
        # keep the previous month in scope so the first sessions can see their
        # prior session's RTH without a second read
        bars = pd.concat([prev_month_bars, cur], ignore_index=True) if len(prev_month_bars) else cur
        fronts = bars[bars["is_front"]].groupby("session")["symbol"].first().to_dict()

        for s in [x for x in sessions if month_key(x) == mk]:
            sym = fronts.get(s)
            if sym is None:
                continue
            fb = bars[(bars["session"] == s) & (bars["symbol"] == sym)].sort_values("ts")
            ps = prior_trade_date(s, cal)
            pb = bars[(bars["session"] == ps) & (bars["symbol"] == fronts.get(ps, sym))].sort_values("ts")
            r_lo, r_hi = prior_rth_window_utc(s, cal)
            row = session_context(s, cal, fb, window_bars(pb, r_lo, r_hi), sym)
            if "ref_t" not in row:
                continue
            ex = excursion_profile(s, cal, fb, row["ref_t"])
            row.update(summarize_excursions(ex))
            ctx_rows.append(row)
            if not ex.empty:
                ex_frames.append(ex)

        prev_month_bars = cur[cur["session"] >= max(cur["session"])]  # last session only
        if i % 6 == 0:
            print(f"  {mk}: {len(ctx_rows)} sessions, {_time.time()-t0:.0f}s", flush=True)

    ctx = add_scales(pd.DataFrame(ctx_rows))
    ex = pd.concat(ex_frames, ignore_index=True)
    d = layer_dir("daily", CONFIGS)
    ctx.to_parquet(assert_not_raw(d / "context.parquet"), index=False)
    ex.to_parquet(assert_not_raw(d / "excursions.parquet"), index=False)
    print(f"\ncontext: {len(ctx)} sessions x {ctx.shape[1]} cols "
          f"({ctx['session'].min()} .. {ctx['session'].max()})")
    print(f"excursions: {len(ex):,} session-minutes")
    print(f"done in {(_time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
