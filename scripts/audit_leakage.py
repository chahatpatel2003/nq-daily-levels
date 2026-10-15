#!/usr/bin/env python
"""Leakage audit against real data.

The decisive test is **truncation invariance**: build a session's candidate
zones twice — once from the complete session tape, once from a tape physically
cut at 06:25 PT — and require byte-identical zones. Reading the code and
concluding it is causal is not the same thing; deleting the future and checking
nothing moved is.

    python scripts/audit_leakage.py --months 2024-03,2024-11 --sessions 8
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.audit.leakage import (AuditReport, AuditResult, check_decision_cut,
                                    check_feature_columns, check_fit_on_train_only,
                                    check_prefix_invariance, check_roll_causality)
from nqlevels.build.daily import (add_regimes, add_rolling_scales,
                                  preopen_columns, session_row)
from nqlevels.build.tape import window_bars
from nqlevels.clock import (decision_time_utc, load_calendar,
                            prior_rth_window_utc, prior_trade_date, trade_dates)
from nqlevels.eval.splits import assign, load_split_plan
from nqlevels.io.store import read_layer
from nqlevels.levels.candidates import build_candidates, gather_inputs
from nqlevels.levels.width import AtrFraction

CONFIGS = str(ROOT / "configs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default="2024-03,2024-11")
    ap.add_argument("--sessions", type=int, default=8)
    a = ap.parse_args()
    cal = load_calendar(CONFIGS)
    rep = AuditReport()

    months = [m.strip() for m in a.months.split(",")]
    hist = sorted({months[0]})
    # pull two extra months of history so the 20-session lookback is real
    y, m = map(int, months[0].split("-"))
    for k in (1, 2):
        mm, yy = m - k, y
        while mm <= 0:
            mm += 12
            yy -= 1
        hist.append(f"{yy:04d}-{mm:02d}")
    load_months = sorted(set(months) | set(hist))
    bars = read_layer("bars_1s", CONFIGS, months=load_months)
    prof = read_layer("profiles", CONFIGS, months=load_months)
    if bars.empty:
        print("no L1 data")
        return 1
    bars["ts"] = pd.DatetimeIndex(bars["ts"]).tz_convert("UTC")
    prof["bucket"] = pd.DatetimeIndex(prof["bucket"]).tz_convert("UTC")
    fronts = bars[bars["is_front"]].groupby("session")["symbol"].first().to_dict()
    all_sessions = sorted(fronts)
    print(f"loaded {len(bars):,} bar rows over {len(all_sessions)} sessions\n")

    # ---- L2 and its causal scaling --------------------------------------
    rows = []
    for s in all_sessions:
        sym = fronts[s]
        fb = bars[(bars["session"] == s) & (bars["symbol"] == sym)].sort_values("ts")
        ps = prior_trade_date(s, cal)
        pb = bars[(bars["session"] == ps) & (bars["symbol"] == fronts.get(ps, sym))].sort_values("ts")
        r_lo, r_hi = prior_rth_window_utc(s, cal)
        rows.append(session_row(s, cal, fb, window_bars(pb, r_lo, r_hi), pb, sym))
    daily = pd.DataFrame(rows)

    print(rep.add(check_prefix_invariance(
        lambda d: add_rolling_scales(d), daily, ["atr_t", "atr_pctile"])))
    print(rep.add(check_feature_columns(preopen_columns(daily))))
    daily = add_regimes(add_rolling_scales(daily)).set_index("session")

    # ---- truncation invariance on real candidate generation --------------
    width = AtrFraction(k=0.05)
    probe = [s for s in all_sessions
             if f"{s.year:04d}-{s.month:02d}" in months
             and np.isfinite(daily.loc[s, "atr_t"])][-a.sessions:]
    print(f"\ntruncation-invariance probe on {len(probe)} sessions "
          f"({probe[0]} .. {probe[-1]})")

    n_moved = n_zones = 0
    per_session = []
    for s in probe:
        sym = fronts[s]
        cut = pd.Timestamp(decision_time_utc(s, cal))
        psym = fronts.get(prior_trade_date(s, cal), sym)

        def gen(b: pd.DataFrame) -> pd.DataFrame:
            si = gather_inputs(s, sym, cal, b, prof, lookback=20, prior_symbol=psym)
            if si is None:
                return pd.DataFrame()
            c = build_candidates(si, width, daily.loc[s])
            return (c[c["generator"] == "named"]
                    .sort_values("price_t")
                    .reset_index(drop=True)[["price_t", "lo_t", "hi_t", "kinds"]])

        full = gen(bars)
        trunc = gen(bars[bars["ts"] < cut])
        same_shape = len(full) == len(trunc)
        moved = (0 if not same_shape else
                 int((~np.isclose(full["price_t"], trunc["price_t"])).sum()
                     + (full["kinds"].to_numpy() != trunc["kinds"].to_numpy()).sum()))
        n_zones += len(full)
        n_moved += moved if same_shape else max(len(full), len(trunc))
        per_session.append({"session": str(s), "n_full": len(full),
                            "n_trunc": len(trunc), "moved": moved,
                            "shape_ok": same_shape})
        flag = "ok" if same_shape and moved == 0 else "DIFFERS"
        print(f"  {s}: {len(full):3d} zones full / {len(trunc):3d} truncated  -> {flag}")

    print(rep.add(AuditResult(
        "candidate zones identical from a truncated tape", n_moved == 0,
        f"{n_zones} zones over {len(probe)} sessions, {n_moved} differences",
        n_zones)))

    # ---- windowing + split hygiene ---------------------------------------
    s0 = probe[-1]
    si = gather_inputs(s0, fronts[s0], cal, bars, prof, lookback=20)
    print(rep.add(check_decision_cut(
        {"pre_bars": si.pre_bars, "lookback": si.lookback_bars,
         **{f"seg_{k}": v for k, v in si.segments.items()}}, s0, cal)))

    plan = load_split_plan(CONFIGS)
    buckets = assign(trade_dates(date(2021, 1, 5), date(2026, 6, 17), cal), plan)
    print(rep.add(check_fit_on_train_only(buckets["train"], buckets["validation"])))
    print(rep.add(check_fit_on_train_only(buckets["train"] + buckets["validation"],
                                          buckets["holdout"])))
    log = ROOT / "outputs" / "l1_build_2021-01-05_2026-06-17.json"
    if log.exists():
        print(rep.add(check_roll_causality(json.loads(log.read_text())["rolls"])))

    out = ROOT / "outputs" / "leakage_audit.json"
    out.write_text(json.dumps({**rep.to_dict(), "per_session": per_session,
                               "split_sizes": {k: len(v) for k, v in buckets.items()}},
                              indent=2, default=str))
    print(f"\n{'='*70}\n{sum(r.passed for r in rep.results)}/{len(rep.results)} audits passed")
    print(f"split sizes: " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))
    print(f"wrote {out}")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
