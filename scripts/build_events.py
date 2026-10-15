#!/usr/bin/env python
"""Build the rejection-event layer: every confirmed reaction in 2021-2025.

One row per event: where price arrived and left fast, with magnitude,
velocity, volume and delta context, normalised by that session's own pre-open
ATR. Detected on full session tapes — legal because events are only ever used
as *history* for later sessions; the same-morning path re-runs the detector on
pre-cut bars instead of reading this layer.

Also writes the roll adjustment table (per-session cumulative spread, measured
from bars overlap at each front-month change) so an event from an old contract
can be expressed in a later contract's scale.

Holdout sessions are not built at all, mirroring axis_measured — the frozen
pipeline recomputes events on the fly if a holdout run is ever pre-registered.
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

from nqlevels.eval.splits import assign, load_split_plan                       # noqa: E402
from nqlevels.features.events import detect_events, roll_adjustment            # noqa: E402
from nqlevels.io.raw import assert_not_raw                                     # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402

CONFIGS = str(ROOT / "configs")
TICK = 0.25


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-min", type=int, default=30)
    ap.add_argument("--confirm-min", type=int, default=120)
    ap.add_argument("--theta-atr", type=float, default=0.25)
    a = ap.parse_args()
    t0 = _time.time()

    d = layer_dir("daily", CONFIGS)
    ctx = pd.read_parquet(d / "context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    bk = assign(list(ctx["session"]), load_split_plan(CONFIGS))
    keep = set(bk["train"]) | set(bk["validation"])

    bars = pd.read_parquet(d / "bars_1m.parquet")
    bars["ts"] = pd.DatetimeIndex(bars["ts"]).tz_convert("UTC")

    # roll adjustment uses every session with overlap bars, holdout included —
    # it is a property of the contracts, not an outcome
    front = ctx.set_index("session")["symbol"]
    adj = roll_adjustment(bars, front)
    dst_adj = assert_not_raw(d / "roll_adjust.parquet")
    adj.to_parquet(dst_adj, index=False)
    n_break = adj["chain"].nunique() - 1
    print(f"roll_adjust: {adj['epoch'].nunique()} epochs, {n_break} broken chains, "
          f"span of adj {adj['adj_t'].min():.0f}..{adj['adj_t'].max():.0f} ticks")

    ctxi = ctx.set_index("session")
    out = []
    fb = bars[bars["is_front"]]
    for i, (s, g) in enumerate(fb.groupby("session", sort=True)):
        if s not in keep or s not in ctxi.index:
            continue
        atr = float(ctxi.loc[s, "atr14_t"])
        g = g.sort_values("ts").reset_index(drop=True)
        ev = detect_events(g, atr, a.w_min, a.confirm_min, a.theta_atr)
        if ev.empty:
            continue
        ev["ts"] = g["ts"].to_numpy()[ev["minute"].to_numpy()]
        ev["session"] = s
        ev["symbol"] = str(ctxi.loc[s, "symbol"])
        ev["atr_t"] = atr
        out.append(ev)
        if i % 200 == 0:
            print(f"  {s}: {sum(len(x) for x in out):,} events, "
                  f"{_time.time()-t0:.0f}s", flush=True)

    e = pd.concat(out, ignore_index=True)
    dst = assert_not_raw(d / "events.parquet")
    e.to_parquet(dst, index=False, compression="zstd")

    per = e.groupby("session").size()
    yr = e.assign(y=pd.to_datetime(e["session"].astype(str)).dt.year) \
          .groupby("y").agg(n=("mag_atr", "size"), mag_med=("mag_atr", "median"),
                            mag_p90=("mag_atr", lambda v: float(np.quantile(v, .9))))
    print(f"\nevents: {len(e):,} rows over {e['session'].nunique()} sessions "
          f"(median {per.median():.0f}/session, p10 {per.quantile(.1):.0f}, "
          f"p90 {per.quantile(.9):.0f})")
    print(f"dir balance: {float((e['dir'] == 1).mean()):.3f} high-side")
    print(yr.to_string())
    print(f"params w={a.w_min} confirm={a.confirm_min} theta={a.theta_atr} | "
          f"{dst.stat().st_size/1e6:.1f} MB, {(_time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
