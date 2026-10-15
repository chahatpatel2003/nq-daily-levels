#!/usr/bin/env python
"""Recover each zone's pivot-EXTREME minute (vs its stored confirmation
minute) by re-running the confirmed zigzag per session and matching zones on
(session, scale, price, confirmation minute). Gives the confirmation delay —
the causality quantity Phase 7 must characterise: a pivot exists at its
extreme but is only displayable from confirmation.

Writes zones_confirm.parquet: zid, ip_min (extreme), delay_min.
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.features.zones import SCALES_ATR, zigzag_confirmed               # noqa: E402
from nqlevels.io.raw import assert_not_raw                                     # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402

CONFIGS = str(ROOT / "configs")


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    zn = pd.read_parquet(d0 / "zones.parquet",
                         columns=["session", "zid", "scale", "dir", "price_rf",
                                  "form_min"])
    ctx = pd.read_parquet(d0 / "context.parquet", columns=["session", "atr14_t"])
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    ctxi = ctx.set_index("session")["atr14_t"]
    adj = pd.read_parquet(d0 / "roll_adjust.parquet").set_index("session")["adj_t"]
    b1m = pd.read_parquet(d0 / "bars_1m.parquet",
                          columns=["session", "ts", "high_t", "low_t", "is_front"])
    b1m = b1m[b1m["is_front"]]
    rows = []
    for s, g in b1m.groupby("session", sort=True):
        z = zn[zn["session"] == s]
        if not len(z) or s not in ctxi.index:
            continue
        g = g.sort_values("ts")
        hi = g["high_t"].to_numpy("float64")
        lo = g["low_t"].to_numpy("float64")
        atr = float(ctxi.loc[s])
        adj_s = float(adj.loc[s])
        piv_by = {}
        for si, sc in enumerate(SCALES_ATR):
            piv_by[si] = zigzag_confirmed(hi, lo, sc * atr)
        for r in z.itertuples():
            px = r.price_rf + adj_s
            hit = [p for p in piv_by[int(r.scale)]
                   if p[3] == r.form_min and abs(p[1] - px) < 0.5]
            if hit:
                rows.append({"zid": int(r.zid), "ip_min": int(hit[0][0]),
                             "delay_min": int(r.form_min - hit[0][0])})
    d = pd.DataFrame(rows)
    d.to_parquet(assert_not_raw(d0 / "zones_confirm.parquet"), index=False)
    print(f"matched {len(d):,} of {len(zn):,} zones | delay min: med "
          f"{d['delay_min'].median():.0f} p25 {d['delay_min'].quantile(.25):.0f} "
          f"p75 {d['delay_min'].quantile(.75):.0f} p90 {d['delay_min'].quantile(.9):.0f} "
          f"| {(_time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
