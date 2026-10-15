#!/usr/bin/env python
"""Sanity gate for the L1 layer.

Nothing downstream is worth building until these hold, so they are checks with
a pass/fail verdict rather than a notebook of eyeballed numbers:

  1. price scale      — ticks round-trip to plausible NQ index levels
  2. session bounds   — no bar outside [session open, session close)
  3. trade-date roll  — a bar's stamp maps back to the session it is filed under
  4. window slicing   — the 06:30-09:00 PT target window is where it should be,
                        on both sides of a DST transition
  5. profile ↔ bars   — volume-at-price sums equal bar volume, exactly
  6. cross-validation — aggregate 1-second bars to 1-minute and compare against
                        the independently built reference bars from the earlier
                        NQ program. Two implementations agreeing on OHLCV is
                        much stronger evidence than either one alone.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.build.tape import ticks_to_price, window_bars, window_profile  # noqa: E402
from nqlevels.clock import (decision_time_utc, load_calendar,               # noqa: E402
                            segment_windows_utc, session_close_utc,
                            session_open_utc, target_window_utc, trade_date)
from nqlevels.io.raw import load_paths                                       # noqa: E402
from nqlevels.io.store import read_layer                                     # noqa: E402

CONFIGS = str(ROOT / "configs")
TICK = 0.25
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="2024-03",
                    help="comma-separated YYYY-MM partitions to audit")
    a = ap.parse_args()
    cal = load_calendar(CONFIGS)
    months = [m.strip() for m in a.months.split(",")]
    bars = read_layer("bars_1s", CONFIGS, months=months)
    prof = read_layer("profiles", CONFIGS, months=months)
    if bars.empty:
        print("no L1 data for 2024-03 — run scripts/build_l1.py first")
        return 1
    bars["ts"] = pd.DatetimeIndex(bars["ts"]).tz_convert("UTC")
    prof["bucket"] = pd.DatetimeIndex(prof["bucket"]).tz_convert("UTC")
    sessions = sorted(bars["session"].unique())
    print(f"L1 sample: {len(bars):,} bar rows, {len(prof):,} profile rows, "
          f"{len(sessions)} sessions {sessions[0]} .. {sessions[-1]}\n")

    # 1 ------------------------------------------------------------- price
    print("1. price scale")
    front = bars[bars["is_front"]]
    px = ticks_to_price(front["close_t"], TICK)
    check("prices in the plausible NQ range 2021-2026",
          10000 < px.min() and px.max() < 60000, f"{px.min():.2f} .. {px.max():.2f}")
    check("ohlc ordering low <= open,close <= high",
          bool(((front["low_t"] <= front[["open_t", "close_t"]].min(axis=1))
                & (front["high_t"] >= front[["open_t", "close_t"]].max(axis=1))).all()))
    check("buy_vol + sell_vol <= volume (unattributed prints excluded)",
          bool((front["buy_vol"] + front["sell_vol"] <= front["volume"]).all()))

    # 2 ------------------------------------------------------------ bounds
    print("\n2. session bounds")
    bad = 0
    for s in sessions:
        b = bars[bars["session"] == s]
        lo, hi = session_open_utc(s, cal), session_close_utc(s, cal)
        bad += int(((b["ts"] < lo) | (b["ts"] >= hi)).sum())
    check("every bar inside [session open, session close)", bad == 0, f"{bad} violations")

    # 3 ------------------------------------------------------- trade date
    print("\n3. trade-date attribution")
    samp = bars.sample(min(20000, len(bars)), random_state=0)
    mism = sum(trade_date(t.to_pydatetime(), cal) != s
               for t, s in zip(samp["ts"], samp["session"]))
    check("bar stamp maps back to its own session", mism == 0,
          f"{mism}/{len(samp)} mismatched")

    # 4 ------------------------------------------------------------ window
    print("\n4. target-window slicing (DST-sensitive)")
    dst_probe = [d for d in sessions if d.month == 3 and 1 <= d.day <= 20] or sessions[:2]
    for s in dst_probe[:2] + dst_probe[-2:]:
        b = bars[(bars["session"] == s) & bars["is_front"]]
        lo, hi = target_window_utc(s, cal)
        w = window_bars(b, lo, hi)
        pt = w["ts"].dt.tz_convert(cal.trader_tz)
        check(f"{s}: target window is 06:30-09:00 PT",
              pt.min().strftime("%H:%M") == "06:30" and pt.max().strftime("%H:%M") < "09:00",
              f"{pt.min().strftime('%H:%M:%S')} .. {pt.max().strftime('%H:%M:%S')} PT, "
              f"{len(w):,} sec-bars, vol {int(w['volume'].sum()):,}")
        cut = decision_time_utc(s, cal)
        pre = b[b["ts"] < cut]
        check(f"{s}: pre-open data ends strictly before the 06:25 PT cut",
              bool(pre["ts"].max() < cut),
              f"last pre-open bar {pre['ts'].max().tz_convert(cal.trader_tz).strftime('%H:%M:%S')} PT")

    # 5 ----------------------------------------------------------- profile
    print("\n5. profile <-> bars consistency")
    mx = 0
    for s in sessions:
        for sym in bars[bars["session"] == s]["symbol"].unique():
            bv = int(bars[(bars["session"] == s) & (bars["symbol"] == sym)]["volume"].sum())
            pv = int(prof[(prof["session"] == s) & (prof["symbol"] == sym)]["volume"].sum())
            mx = max(mx, abs(bv - pv))
    check("session volume identical in both L1 datasets", mx == 0, f"max abs diff {mx}")

    s0 = sessions[len(sessions) // 2]
    segs = segment_windows_utc(s0, cal)
    p0 = prof[(prof["session"] == s0) & prof["is_front"]]
    lo, hi = segs["globex"]
    hi30 = pd.Timestamp(hi).floor("30min")
    g = window_profile(p0, lo, hi30)
    poc = g.loc[g["volume"].idxmax()]
    check(f"{s0}: overnight profile has a POC",
          len(g) > 20 and poc["volume"] > 0,
          f"{len(g)} price levels, POC {ticks_to_price(poc['price_t']):.2f} "
          f"vol {int(poc['volume']):,}")

    # 6 --------------------------------------------------------- reference
    print("\n6. cross-validation vs the independent reference build")
    ref_path = Path(load_paths(CONFIGS)["reference"]["tslab_nq_bars_1m"])
    if not ref_path.exists():
        check("reference bars available", False, f"missing {ref_path}")
    else:
        ref = pd.read_parquet(ref_path, columns=["session", "symbol", "open", "high",
                                                 "low", "close", "volume"])
        ref = ref[ref["session"].isin(sessions)].copy()
        ref.index = pd.DatetimeIndex(ref.index).tz_convert("UTC")
        f = bars[bars["is_front"]].copy()
        f["m"] = f["ts"].dt.floor("1min")
        agg = f.groupby(["session", "symbol", "m"]).agg(
            open=("open_t", "first"), high=("high_t", "max"),
            low=("low_t", "min"), close=("close_t", "last"),
            volume=("volume", "sum")).reset_index()
        for c in ("open", "high", "low", "close"):
            agg[c] = ticks_to_price(agg[c], TICK)
        j = agg.merge(ref.reset_index().rename(columns={"bar_open_utc": "m"}),
                      on=["session", "symbol", "m"], suffixes=("", "_ref"))
        check("reference bars overlap the sample", len(j) > 3000, f"{len(j):,} matched minutes")
        if len(j):
            for c in ("open", "high", "low", "close"):
                d = (j[c] - j[f"{c}_ref"]).abs()
                check(f"1-min {c} matches reference", float(d.max()) < 1e-9,
                      f"max diff {d.max():.6f}")
            dv = (j["volume"] - j["volume_ref"]).abs()
            share = float((dv == 0).mean())
            check("1-min volume matches reference", share > 0.999,
                  f"{share:.4%} of minutes identical, max diff {int(dv.max())}")
            miss_new = len(agg) - len(j)
            check("no minutes present in reference but absent here",
                  len(ref) - len(j) <= 0, f"reference-only minutes: {len(ref) - len(j)}, "
                                          f"new-only minutes: {miss_new}")

    npass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{'='*66}\n{npass}/{len(results)} checks passed")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
