#!/usr/bin/env python
"""Build the named-family candidate layer for every session.

Runs on the 1-minute layer plus the 30-minute volume-at-price profiles. Level
construction never needs sub-minute data — highs, lows, volumes and VWAPs
aggregate exactly — so this is ~50x cheaper than the same job on L1 and gives
identical prices.

Output is one row per (session, level cluster) with the level expressed as an
**offset from the session's 06:25 reference in ticks**, which is the same
coordinate system as the path layer, so labelling is a direct comparison with
no per-session price bookkeeping.

Zone *width* is deliberately NOT applied here. E2 studies width, and baking a
width in at generation time would mean rebuilding the whole layer for every
width tested. Widths are applied at measurement time instead.
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

from nqlevels.clock import load_calendar, prior_trade_date, trade_dates       # noqa: E402
from nqlevels.io.raw import assert_not_raw                                     # noqa: E402
from nqlevels.io.store import layer_dir, read_layer                            # noqa: E402
from nqlevels.levels.candidates import build_named, gather_inputs              # noqa: E402

CONFIGS = str(ROOT / "configs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-05")
    ap.add_argument("--end", default="2026-06-17")
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--swing-bars", type=int, default=30, help="1-min bars each side")
    a = ap.parse_args()

    cal = load_calendar(CONFIGS)
    d = layer_dir("daily", CONFIGS)
    print("loading layers ...")
    bars = pd.read_parquet(d / "bars_1m.parquet")
    bars["ts"] = pd.DatetimeIndex(bars["ts"]).tz_convert("UTC")
    prof = read_layer("profiles", CONFIGS)
    prof["bucket"] = pd.DatetimeIndex(prof["bucket"]).tz_convert("UTC")
    ctx = pd.read_parquet(d / "context.parquet", columns=["session", "symbol", "ref_t", "atr14_t"])
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    ref = dict(zip(ctx["session"], ctx["ref_t"]))
    atr = dict(zip(ctx["session"], ctx["atr14_t"]))
    fronts = bars[bars["is_front"]].groupby("session")["symbol"].first().to_dict()
    print(f"  bars_1m {len(bars):,} | profiles {len(prof):,} | context {len(ctx)}")

    # patch the swing detector to a 1-minute scale
    import nqlevels.levels.named as N
    _orig_swing = N.swing_levels
    N.swing_levels = lambda b, fractal_n=a.swing_bars, max_levels=12: _orig_swing(
        b, fractal_n=fractal_n, max_levels=max_levels)

    # Pre-slice by session. gather_inputs filters `bars` several times per
    # session; over 1,370 sessions that is ~10^10 row comparisons on a 2.2 M-row
    # frame. Handing it only the ~21 sessions it can legally see turns a 20-minute
    # build into a 2-minute one and makes the causal window explicit.
    bars_by = {s: g for s, g in bars.groupby("session", sort=False)}
    prof_by = {s: g for s, g in prof.groupby("session", sort=False)}

    def scope(s: date):
        need = [s]
        d0 = s
        for _ in range(a.lookback + 1):
            d0 = prior_trade_date(d0, cal)
            need.append(d0)
        bl = [bars_by[x] for x in need if x in bars_by]
        pl = [prof_by[x] for x in need if x in prof_by]
        return (pd.concat(bl, ignore_index=True) if bl else pd.DataFrame(),
                pd.concat(pl, ignore_index=True) if pl else pd.DataFrame())

    sessions = trade_dates(date.fromisoformat(a.start), date.fromisoformat(a.end), cal)
    out, t0, skipped = [], _time.time(), 0
    for i, s in enumerate(sessions):
        sym = fronts.get(s)
        if sym is None or s not in ref or not np.isfinite(ref[s]):
            skipped += 1
            continue
        psym = fronts.get(prior_trade_date(s, cal), sym)
        b_s, p_s = scope(s)
        if b_s.empty:
            skipped += 1
            continue
        si = gather_inputs(s, sym, cal, b_s, p_s, lookback=a.lookback, prior_symbol=psym)
        if si is None:
            skipped += 1
            continue
        nm = build_named(si)
        if nm.empty:
            skipped += 1
            continue
        nm = nm.copy()
        nm["session"] = s
        nm["symbol"] = sym
        nm["ref_t"] = si.ref_t
        nm["off_t"] = nm["price_t"] - si.ref_t          # ticks from the reference
        nm["dist_t"] = nm["off_t"].abs()
        nm["dist_atr"] = nm["dist_t"] / atr.get(s, np.nan)
        nm["carry"] = si.carry.get("prior_session", "")
        out.append(nm)
        if i % 150 == 0:
            print(f"  {s}: {len(nm)} clusters, {_time.time()-t0:.0f}s", flush=True)

    c = pd.concat(out, ignore_index=True)
    dst = assert_not_raw(d / "candidates_named.parquet")
    c.to_parquet(dst, index=False, compression="zstd")
    print(f"\ncandidates: {len(c):,} rows over {c['session'].nunique()} sessions "
          f"({len(c)/c['session'].nunique():.0f}/session), skipped {skipped}, "
          f"{dst.stat().st_size/1e6:.0f} MB, {(_time.time()-t0)/60:.1f} min")
    print("\ntop level kinds by frequency:")
    kinds = c["kinds"].str.split(",").explode().value_counts()
    print(kinds.head(30).to_string())
    print(f"\ndistinct kinds: {kinds.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
