#!/usr/bin/env python
"""P12 stream prep: per-session within-day trade-R sequences for each
execution stream. tap<=3 is rebuilt through the one-open-trade portfolio
pass on tap<=3 fires (a tap<=3 trader takes trades the raw portfolio
skipped due to overlap), NOT a post-hoc filter of the raw stream.
Cap-N streams take the FIRST N trades of each day chronologically.
Saves /Volumes/T9/nq_levels/daily/p12_streams.pkl:
  {"sessions": [date...], "streams": {name: [np.array(R per trade)...]}}
aligned over ALL 2022+ sessions (zero-trade sessions = empty arrays).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p10_eval import portfolio                                             # noqa: E402
from run_p10_stage1 import load_trades                                         # noqa: E402

RISK_PTS = 12.5


def per_session(df, sessions):
    """List of np.array trade-R (chronological) aligned to sessions."""
    df = df.sort_values(["session", "t_open"])
    g = {s: grp["R"].to_numpy() for s, grp in df.groupby("session")}
    return [g.get(s, np.empty(0)) for s in sessions]


def cap(seqs, n):
    return [a[:n] for a in seqs]


def stats(name, seqs):
    allr = np.concatenate([a for a in seqs if len(a)])
    day = np.array([a.sum() for a in seqs])
    eq = np.cumsum(day)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    print(f"  {name:>12}: n {len(allr):6,} tpd {len(allr)/len(seqs):4.2f} "
          f"expR {allr.mean():+.3f} totR {allr.sum():6.0f} "
          f"day mean {day.mean():+.2f} maxDD {dd:4.1f}R "
          f"PF {allr[allr>0].sum()/-allr[allr<=0].sum():.2f}")


def main() -> int:
    t = load_trades()
    t = t[t["year"] >= 2022]
    sessions = sorted(t["session"].unique())
    print(f"{len(sessions)} sessions 2022+")

    pf_raw = portfolio(t, "E1d", ("r4",), 0.85, 12.5, 30.0)
    pf_raw["R"] = pf_raw["pnl"] / RISK_PTS
    old = pd.read_parquet("/Volumes/T9/nq_levels/daily/p11_stream.parquet")
    same = (len(pf_raw) == len(old)
            and np.allclose(np.sort(pf_raw["pnl"].to_numpy()),
                            np.sort(old["pnl"].to_numpy())))
    print(f"raw rebuild matches p11_stream: {same} "
          f"({len(pf_raw):,} vs {len(old):,})")

    pf_tap = portfolio(t[t["tap_no"] <= 3], "E1d", ("r4",), 0.85, 12.5, 30.0)
    pf_tap["R"] = pf_tap["pnl"] / RISK_PTS
    post_hoc_n = int((old["tap_no"] <= 3).sum())
    print(f"tap<=3 portfolio: {len(pf_tap):,} trades "
          f"(post-hoc filter would give {post_hoc_n:,})")

    raw = per_session(pf_raw, sessions)
    tap = per_session(pf_tap, sessions)
    streams = {
        "raw": raw, "cap8": cap(raw, 8), "cap6": cap(raw, 6),
        "cap4": cap(raw, 4), "tap3": tap, "tap3cap8": cap(tap, 8),
    }
    print("\nstream metrics (R units, base cost):")
    for k, v in streams.items():
        stats(k, v)

    out = Path("/Volumes/T9/nq_levels/daily/p12_streams.pkl")
    with out.open("wb") as f:
        pickle.dump({"sessions": sessions, "streams": streams}, f)
    print(f"\nwrote {out}")

    print("\n=== PART 23: 2026-05-04 churn day / 2025-04-07 monster day ===")
    for lab in ("2026-05-04", "2025-04-07"):
        i = [j for j, s in enumerate(sessions) if str(s) == lab][0]
        row = [f"{k}: {len(v[i])}t {v[i].sum():+.1f}R" for k, v in streams.items()]
        print(f"  {lab}: " + " | ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
