#!/usr/bin/env python
"""E3b — the higher-timeframe levels the first candidate build missed.

`build_candidates.py` called `build_named(si)` without week/month bars, so
prior-week and prior-month high/low/close were never generated and therefore
never tested. Rather than rebuild the whole candidate and axis layer for six
extra prices, this computes them (plus multi-session extrema, which were also
absent) directly from the 1-minute layer and tests them against the **already
measured** axis grid with the same stratified contrast.

All levels are computed from RTH sessions strictly before the target session,
so nothing crosses the decision cut.
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

from nqlevels.clock import (load_calendar, prior_trade_date,                   # noqa: E402
                            rth_window_utc)
from nqlevels.eval.quality import add_distance_bands, stratified_contrast      # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402

CONFIGS = str(ROOT / "configs")
TICK = 0.25
pd.set_option("display.width", 220)


def hdr(s):
    print(f"\n{'='*96}\n{s}\n{'='*96}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--near-t", type=float, default=6.0)
    a = ap.parse_args()
    t0 = _time.time()
    cal = load_calendar(CONFIGS)
    d = layer_dir("daily", CONFIGS)

    bars = pd.read_parquet(d / "bars_1m.parquet")
    bars["ts"] = pd.DatetimeIndex(bars["ts"]).tz_convert("UTC")
    bars = bars[bars["is_front"]]
    ctx = pd.read_parquet(d / "context.parquet", columns=["session", "ref_t", "atr14_t"])
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date

    # per-session RTH OHLC (the unit every horizon level is built from)
    print("building per-session RTH OHLC ...")
    rows = []
    for s, g in bars.groupby("session", sort=True):
        lo, hi = rth_window_utc(s, cal)
        w = g[(g["ts"] >= lo) & (g["ts"] < hi)]
        if w.empty:
            continue
        rows.append({"session": s, "high": float(w["high_t"].max()),
                     "low": float(w["low_t"].min()),
                     "close": float(w.sort_values("ts")["close_t"].iloc[-1])})
    S = pd.DataFrame(rows).sort_values("session").reset_index(drop=True)
    S["ts"] = pd.to_datetime(S["session"])
    S["iso_week"] = S["ts"].dt.isocalendar().year.astype(str) + "-" + \
        S["ts"].dt.isocalendar().week.astype(str).str.zfill(2)
    S["cal_month"] = S["ts"].dt.to_period("M").astype(str)
    print(f"  {len(S)} sessions")

    # prior completed week / month aggregates, shifted so the current period is excluded
    wk = S.groupby("iso_week").agg(high=("high", "max"), low=("low", "min"),
                                   close=("close", "last")).shift(1)
    mo = S.groupby("cal_month").agg(high=("high", "max"), low=("low", "min"),
                                    close=("close", "last")).shift(1)
    lv = pd.DataFrame({"session": S["session"]})
    lv["pwh"] = S["iso_week"].map(wk["high"]).to_numpy()
    lv["pwl"] = S["iso_week"].map(wk["low"]).to_numpy()
    lv["pwc"] = S["iso_week"].map(wk["close"]).to_numpy()
    lv["pmh"] = S["cal_month"].map(mo["high"]).to_numpy()
    lv["pml"] = S["cal_month"].map(mo["low"]).to_numpy()
    lv["pmc"] = S["cal_month"].map(mo["close"]).to_numpy()
    # multi-session extrema over strictly prior sessions
    for n in (5, 10, 20, 60):
        lv[f"hi{n}"] = S["high"].rolling(n, min_periods=max(3, n // 3)).max().shift(1).to_numpy()
        lv[f"lo{n}"] = S["low"].rolling(n, min_periods=max(3, n // 3)).min().shift(1).to_numpy()

    m = pd.read_parquet(d / "axis_measured.parquet")
    m = m[(m["dist_atr"] <= 1.2) & (m["side"] != "inside")].reset_index(drop=True)
    m = m.merge(lv, on="session", how="left").merge(
        ctx[["session", "ref_t"]].rename(columns={"ref_t": "_ref"}), on="session", how="left")

    kinds = [c for c in lv.columns if c != "session"]
    for k in kinds:
        m[f"dk_{k}"] = (m[k] - m["_ref"] - m["off_t"]).abs()

    hdr("E3b. Higher-timeframe and multi-session levels vs matched non-level prices")
    print("   prior week / prior month high-low-close, and N-session extrema.")
    print("   Same stratified contrast: distance band and side held fixed.\n")
    import nqlevels.eval.quality as Q
    out = []
    for split in ("train", "validation"):
        s0 = m[m["split"] == split].copy()
        s0["y"] = s0["clean_10_5"]
        Q.STRAT_STATS["y"] = ("y", "touched")
        print(f"  -- {split}  ({s0['session'].nunique()} sessions)")
        print(f"   {'kind':>6} {'coverage':>9} {'n_at':>7} {'touch@':>7} {'clean@':>7} "
              f"{'d_att':>9} {'d_clean':>9} {'95% CI':>19} {'p>0':>6}")
        for k in kinds:
            col = f"dk_{k}"
            s = s0[s0[col].notna()].copy()
            if s.empty:
                continue
            s["_f"] = (s[col] <= a.near_t).astype("int8")
            if s["_f"].sum() < 300:
                continue
            s = add_distance_bands(s, n=8)
            ca = stratified_contrast(s, "_f", "y", n_boot=a.boot)
            aa = stratified_contrast(s, "_f", "attraction", n_boot=a.boot)
            f = s[s["_f"] == 1]
            t = f[f["touched"] == True]                                # noqa: E712
            print(f"   {k:>6} {100*s['_f'].mean():8.2f}% {int(s['_f'].sum()):7,} "
                  f"{f['touched'].mean():7.3f} "
                  f"{(t['clean_10_5'].mean() if len(t) else np.nan):7.4f} "
                  f"{aa['diff']:+9.5f} {ca['diff']:+9.5f} "
                  f"[{ca['lo']:+.4f},{ca['hi']:+.4f}] {ca['p_gt0']:6.3f}")
            out.append({"split": split, "kind": k, "n_at": int(s["_f"].sum()),
                        "d_att": aa["diff"], "d_clean": ca["diff"],
                        "lo": ca["lo"], "hi": ca["hi"], "p_gt0": ca["p_gt0"]})
        print()

    (ROOT / "outputs" / "e3b_horizon.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote outputs/e3b_horizon.json  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
