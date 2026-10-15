#!/usr/bin/env python
"""P25A: causal break-crossing table.

For every zone with a breaking tap (resistance-side tap whose
o_depth_max_full >= (zhi-zlo)+20t), find the FIRST 1-second bar at or
after the first breaking tap whose high reaches zhi + (zhi-zlo) + 20t —
the moment the break became knowable. Only the first breaking tap is
walked: its window is a superset of every later tap's window within the
session, and cross-session ordering is monotone, so its first crossing
IS the zone's causal break time.

Output: outputs/p25a_break_crossings.parquet
  zid, session, tap_ts (old backdated break time), cross_ts (causal
  break time), gap_s, sess_end_ts.
No frozen logic is modified; this is a read-only derivation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEV = "/Volumes/T9/nq_levels"
SEAL = "/Volumes/T9/nq_levels/sealed_holdout_1/derived"


def epoch_s(ts_like) -> np.ndarray:
    return pd.DatetimeIndex(ts_like).as_unit("ns").asi8 // 10 ** 9


def load_first_breaks() -> pd.DataFrame:
    # NOTE: sealed corpus is NOT lower-date-filtered here — break STATE
    # must see a zone's full tap history (its layer spans 2026-01..08-14
    # for warm-up). Event double-count prevention is the consumer's job.
    frames = []
    for base, lo, hi, off in ((DEV, "2021-01-01", "2026-06-17", 0),
                              (SEAL, "2026-01-01", "2026-08-14",
                               10_000_000)):
        it = pd.read_parquet(
            f"{base}/daily/interactions.parquet",
            columns=["session", "zid", "tap_ts", "app_side_above",
                     "zlo_t", "zhi_t", "o_depth_max_full"])
        it["session"] = it["session"].astype(str)
        it = it[(it["session"] >= lo) & (it["session"] <= hi)]
        res = it[it["app_side_above"] == 0]
        brk = res[res["o_depth_max_full"]
                  >= (res["zhi_t"] - res["zlo_t"]) + 20].copy()
        brk["zid"] += off
        brk["root"] = base
        frames.append(
            brk.sort_values("tap_ts").groupby("zid", as_index=False)
            .first())
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    fb = load_first_breaks()
    print(f"zones with a break: {len(fb):,}")
    out = []
    anom = 0
    for (root, mon), g in fb.assign(mon=fb["session"].str[:7]) \
            .groupby(["root", "mon"]):
        b = pd.read_parquet(f"{root}/bars_1s/{mon}.parquet",
                            columns=["session", "ts", "high_t",
                                     "is_front"])
        b = b[b["is_front"]]
        b["session"] = b["session"].astype(str)
        for sess, gs in g.groupby("session"):
            sb = b[b["session"] == sess].sort_values("ts")
            ep = epoch_s(sb["ts"])
            hi = sb["high_t"].to_numpy("float64")
            end = int(ep[-1]) if len(ep) else -1
            for r in gs.itertuples():
                j0 = int(np.searchsorted(ep, r.tap_ts, "left"))
                thr = r.zhi_t + (r.zhi_t - r.zlo_t) + 20
                seg = hi[j0:] >= thr
                k = int(np.argmax(seg)) if seg.any() else -1
                if k < 0:
                    anom += 1
                    cross = np.nan
                else:
                    cross = float(ep[j0 + k])
                out.append((r.zid, sess, float(r.tap_ts), cross, end))
        del b
    df = pd.DataFrame(out, columns=["zid", "session", "tap_ts",
                                    "cross_ts", "sess_end_ts"])
    df["gap_s"] = df["cross_ts"] - df["tap_ts"]
    df.to_parquet(ROOT / "outputs" / "p25a_break_crossings.parquet",
                  index=False)
    g = df["gap_s"].dropna()
    print(f"walked {len(df):,} | crossing-not-found anomalies: {anom}")
    print(f"gap tap_ts->cross_ts (s): zero {(g == 0).mean():.1%} | "
          f"med {g.median():.0f} p75 {g.quantile(.75):.0f} p90 "
          f"{g.quantile(.90):.0f} p99 {g.quantile(.99):.0f} "
          f"max {g.max():.0f}")
    print(f"affected zones (gap>0): {(g > 0).sum():,} "
          f"({(g > 0).mean():.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
