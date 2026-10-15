#!/usr/bin/env python
"""P14 drawdown anatomy: which causal states characterise the bleed
episodes, and do any predict next-trade expectancy prospectively?
Descriptive postmortem strictly separated from predictive checks."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

BASE = "/Volumes/T9/nq_levels/daily/p14_baseline_tap3.parquet"
FIRE = "/Volumes/T9/nq_levels/daily/p14_fire.parquet"

STATE = ["atr_pctile", "rv60_atr", "range_used_atr", "ret_1h", "ret_3h",
         "recent5", "recent10", "recent20", "recent50", "vwap_dist_atr",
         "tap_no", "fire_s", "eff_ratio", "pen_max", "s_above_far",
         "sell_share", "dlt_fire", "v_avg", "mono", "zprior_succ",
         "mins_since_last", "age_min", "beyond_far", "intx_fire",
         "buy_fail", "app_speed_5m", "app_toward_30m"]


def episodes(day):
    eq = np.concatenate(([0.0], np.cumsum(day.to_numpy())))
    eps, peak, trough, ti = [], 0, 0.0, 0
    for i in range(1, len(eq)):
        if eq[i] >= eq[peak]:
            if trough > 1e-9:
                eps.append((trough, peak, ti, i))
            peak, trough = i, 0.0
        else:
            d = eq[peak] - eq[i]
            if d > trough:
                trough, ti = d, i
    if trough > 1e-9:
        eps.append((trough, peak, ti, len(eq) - 1))
    return eps


def main() -> int:
    base = pd.read_parquet(BASE)
    fire = pd.read_parquet(FIRE)
    fire["session"] = fire["session"].astype(str)
    m = base.merge(fire, on=["session", "zid", "tap_sec"], how="left",
                   suffixes=("", "_f"))
    m["R"] = m["pnl"] / 12.5
    sessions = sorted(m["session"].unique())
    day = m.groupby("session")["R"].sum().reindex(sessions).fillna(0.0)
    eps = episodes(day)
    s_ax = list(day.index)
    big = [e for e in eps if e[0] >= 10.0]
    bleed_sessions = set()
    print(f"episodes >=10R: {len(big)}")
    for depth, p_, t_, rec in sorted(big, reverse=True):
        span = s_ax[p_ - 1:t_]
        bleed_sessions |= set(span[1:])
        print(f"  {depth:5.1f}R {s_ax[p_-1]}→{s_ax[t_-1]} "
              f"({t_-p_} sess, {sum(1 for s in span[1:] if s in set(m['session']))} act)")
    m["bleed"] = m["session"].isin(bleed_sessions)
    print(f"\nbleed trades {int(m['bleed'].sum()):,} "
          f"(expR {m.loc[m['bleed'],'R'].mean():+.3f}) vs normal "
          f"{int((~m['bleed']).sum()):,} "
          f"({m.loc[~m['bleed'],'R'].mean():+.3f})")

    print("\n=== descriptive: standardized mean diff (bleed - normal) ===")
    rows = []
    for f in STATE:
        if f not in m.columns:
            continue
        a = m.loc[m["bleed"], f].astype(float)
        b = m.loc[~m["bleed"], f].astype(float)
        sd = pd.concat([a, b]).std()
        if sd > 0:
            rows.append((f, float((a.mean() - b.mean()) / sd)))
    for f, d in sorted(rows, key=lambda x: -abs(x[1]))[:15]:
        print(f"  {f:>18}: {d:+.2f}")

    print("\n=== predictive: expectancy by causal-state quintile ===")
    for f in ("recent20", "recent50", "atr_pctile", "rv60_atr", "ret_3h",
              "fire_s", "eff_ratio", "s_above_far", "tap_no",
              "sell_share", "mono", "pen_max"):
        if f not in m.columns or m[f].notna().sum() < 2000:
            continue
        x = m[m[f].notna()]
        try:
            q = pd.qcut(x[f], 5, duplicates="drop")
        except ValueError:
            continue
        g = x.groupby(q, observed=True)["R"].agg(["mean", "size"])
        cells = " ".join(f"{v:+.2f}" for v in g["mean"])
        print(f"  {f:>14}: {cells}   (n≈{int(g['size'].mean())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
