#!/usr/bin/env python
"""End-to-end Phase-1 demonstration on a small window.

Runs the whole chain for a handful of sessions — L1 -> L2 daily context ->
candidate zones at 06:25 PT -> touch labels -> reaction labels — and prints
what it found. This is a *machinery* check, not a result: the sample is far too
small to say anything about edge, and the numbers below must not be quoted as
findings.

    python scripts/demo_pipeline.py --start 2024-03-04 --end 2024-03-15
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.build.daily import add_regimes, add_rolling_scales, session_row   # noqa: E402
from nqlevels.build.tape import ticks_to_price, window_bars                     # noqa: E402
from nqlevels.clock import (decision_time_utc, load_calendar,                   # noqa: E402
                            prior_rth_window_utc, prior_trade_date,
                            session_open_utc, target_window_utc, trade_dates)
from nqlevels.labels.reaction import chop_score, clean_rejection, evaluate_reaction  # noqa: E402
from nqlevels.labels.touch import evaluate_touch, zone_side                     # noqa: E402
from nqlevels.levels.candidates import build_candidates, gather_inputs          # noqa: E402
from nqlevels.levels.width import AtrFraction                                   # noqa: E402
from nqlevels.io.store import read_layer                                        # noqa: E402

CONFIGS = str(ROOT / "configs")
TICK = 0.25


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-03-04")
    ap.add_argument("--end", default="2024-03-15")
    ap.add_argument("--lookback", type=int, default=20)
    a = ap.parse_args()

    cal = load_calendar(CONFIGS)
    s0, s1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    sessions = trade_dates(s0, s1, cal)

    # read enough history for the lookback + ATR
    hist_start = prior_trade_date(s0, cal, a.lookback + 20)
    months = sorted({f"{d.year:04d}-{d.month:02d}"
                     for d in trade_dates(hist_start, s1, cal)})
    print(f"reading L1 months {months[0]}..{months[-1]}")
    bars = read_layer("bars_1s", CONFIGS, months=months)
    prof = read_layer("profiles", CONFIGS, months=months)
    if bars.empty:
        print("no L1 data — run scripts/build_l1.py first")
        return 1
    bars["ts"] = pd.DatetimeIndex(bars["ts"]).tz_convert("UTC")
    prof["bucket"] = pd.DatetimeIndex(prof["bucket"]).tz_convert("UTC")
    print(f"  {len(bars):,} bar rows, {len(prof):,} profile rows\n")

    # ---- L2 daily context ------------------------------------------------
    fronts = (bars[bars["is_front"]].groupby("session")["symbol"].first().to_dict())
    rows = []
    for s in trade_dates(hist_start, s1, cal):
        sym = fronts.get(s)
        if sym is None:
            continue
        fb = bars[(bars["session"] == s) & (bars["symbol"] == sym)].sort_values("ts")
        ps = prior_trade_date(s, cal)
        psym = fronts.get(ps, sym)
        pb = bars[(bars["session"] == ps) & (bars["symbol"] == psym)].sort_values("ts")
        r_lo, r_hi = prior_rth_window_utc(s, cal)
        rows.append(session_row(s, cal, fb, window_bars(pb, r_lo, r_hi), pb, sym))
    daily = add_regimes(add_rolling_scales(pd.DataFrame(rows)))
    daily = daily.set_index("session")
    print(f"L2: {len(daily)} sessions; ATR available from "
          f"{daily['atr_t'].first_valid_index()}\n")

    width = AtrFraction(k=0.05)
    all_rows = []

    for s in sessions:
        sym = fronts.get(s)
        if sym is None or s not in daily.index:
            continue
        ctx = daily.loc[s]
        if not np.isfinite(ctx.get("atr_t", np.nan)):
            print(f"{s}: no ATR yet, skipping")
            continue
        psym = fronts.get(prior_trade_date(s, cal), sym)
        si = gather_inputs(s, sym, cal, bars, prof, lookback=a.lookback,
                           prior_symbol=psym)
        if si is None:
            print(f"{s}: no pre-open data")
            continue
        cand = build_candidates(si, width, ctx)
        cand = cand[cand["generator"] == "named"]

        t_lo, t_hi = target_window_utc(s, cal)
        tgt = window_bars(bars[(bars["session"] == s) & (bars["symbol"] == sym)]
                          .sort_values("ts"), t_lo, t_hi)

        print(f"=== {s}  front={sym}  ref={ticks_to_price(si.ref_t):.2f}  "
              f"ATR={ticks_to_price(ctx['atr_t']):.0f}pts  gap={ticks_to_price(ctx['gap_t_prior']):+.0f}pts  "
              f"regime={ctx['regime_gap']}/{ctx['regime_vol']}  carry={si.carry.get('prior_session','')}")
        print(f"    candidates: {len(cand)}   target-window seconds: {len(tgt):,}")

        from nqlevels.labels.touch import Zone
        recs = []
        for _, c in cand.iterrows():
            z = Zone(int(c["lo_t"]), int(c["hi_t"]), str(c["kinds"]), int(c["price_t"]))
            side = zone_side(z, si.ref_t)
            to = evaluate_touch(tgt, z, si.ref_t, pd.Timestamp(t_lo))
            rec = {"session": s, "kinds": c["kinds"], "n_kinds": c["n_kinds"],
                   "price": ticks_to_price(c["price_t"]), "side": side,
                   "dist_pts": c["dist_t"] * TICK, "dist_atr": c["dist_atr"],
                   "half_width_pts": c["half_width_t"] * TICK,
                   "touched": to.touched, "min_to_touch": to.minutes_to_touch}
            if to.touched and not to.open_inside:
                ro = evaluate_reaction(tgt, z, side, to.first_touch_idx)
                rec.update({
                    "mfe_rev_5m": ro.mfe_rev[5] * TICK, "mfe_rev_30m": ro.mfe_rev[30] * TICK,
                    "mfe_cont_30m": ro.mfe_cont[30] * TICK,
                    "pen_pts": ro.max_penetration_t * TICK,
                    "secs_in": ro.secs_inside, "xings": ro.n_crossings,
                    "dir_eff": ro.directional_efficiency,
                    "clean10_5": clean_rejection(ro, 10, 5), "chop": chop_score(ro)})
            recs.append(rec)
        df = pd.DataFrame(recs)
        all_rows.append(df)
        hit = df[df["touched"]]
        print(f"    touched {len(hit)}/{len(df)}  "
              f"({df[df.side=='above'].touched.sum()} above / "
              f"{df[df.side=='below'].touched.sum()} below)")
        if "clean10_5" in df and hit["clean10_5"].notna().any():
            show = (hit.dropna(subset=["clean10_5"])
                       .sort_values("mfe_rev_30m", ascending=False)
                       .head(4)[["kinds", "price", "side", "dist_pts", "min_to_touch",
                                 "mfe_rev_30m", "pen_pts", "secs_in", "xings",
                                 "dir_eff", "clean10_5", "chop"]])
            print(show.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print()

    if not all_rows:
        print("nothing produced")
        return 1
    allr = pd.concat(all_rows, ignore_index=True)
    print("=" * 78)
    print(f"MACHINERY SUMMARY (NOT A RESULT — {allr['session'].nunique()} sessions, "
          f"{len(allr)} candidate rows)")
    print(f"  overall touch rate            : {allr['touched'].mean():.1%}")
    byd = allr.assign(b=pd.cut(allr["dist_atr"], [0, .25, .5, 1, 2, 10])) \
              .groupby("b", observed=True)["touched"].agg(["mean", "size"])
    print("  touch rate by distance (ATR)  :")
    for b, r in byd.iterrows():
        print(f"      {str(b):<14s} {r['mean']:.1%}  (n={int(r['size'])})")
    hit = allr[allr["touched"] & allr["clean10_5"].notna()] if "clean10_5" in allr else pd.DataFrame()
    if len(hit):
        print(f"  clean 10pt-before-5pt rate    : {hit['clean10_5'].mean():.1%} "
              f"(n={len(hit)} resolved touches)")
        print(f"  median secs inside zone       : {hit['secs_in'].median():.0f}")
        print(f"  median crossings              : {hit['xings'].median():.0f}")
    out = ROOT / "outputs" / "demo_pipeline.csv"
    allr.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
