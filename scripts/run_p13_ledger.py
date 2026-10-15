#!/usr/bin/env python
"""P13 canonical immutable trade ledger for F4 + tap<=3.

One row per EXECUTED trade (one-open-trade portfolio pass over tap<=3
R4v10 fires gated at p_both60 >= 0.85, entry E1d = market fill at the
close of second fire+3, stop 12.5, target 30, base cost 1.0 pt RT).
Prices are roll-adjusted continuous NQ (quarter-tick ints /4 = points).
Source of truth for forward-test comparison and bot reproduction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_p10_eval import portfolio                                             # noqa: E402
from run_p10_stage1 import load_trades                                         # noqa: E402

OUT = Path("/Volumes/T9/nq_levels/daily/p13_ledger.parquet")
STOP, TGT, COST = 12.5, 30.0, 1.0


def main() -> int:
    t = load_trades()
    t = t[t["year"] >= 2022]
    pf = portfolio(t[t["tap_no"] <= 3], "E1d", ("r4",), 0.85, STOP, TGT)
    it = pd.read_parquet("/Volumes/T9/nq_levels/daily/interactions.parquet")
    it = it.drop_duplicates(["session", "zid", "tap_sec"])
    pf = pf.merge(it[["session", "zid", "tap_sec", "tap_ts"]],
                  on=["session", "zid", "tap_sec"], how="left")
    zs = pd.read_parquet("/Volumes/T9/nq_levels/daily/zones.parquet")
    zs = zs.drop_duplicates("zid")[["zid", "scale", "price_rf", "hw_t",
                                    "form_ts_val", "form_seg"]]
    pf = pf.merge(zs, on="zid", how="left")

    led = pd.DataFrame({
        "trade_id": [f"{s}-{z}-{ts}" for s, z, ts in
                     zip(pf["session"], pf["zid"], pf["tap_sec"])],
        "session": pf["session"].astype(str),
        "year": pf["year"],
        "seg": pf["seg"],
        "zid": pf["zid"],
        "zone_scale": pf["scale"],
        "zone_center": pf["price_rf"] / 4.0,
        "zone_lo": pf["zlo"] / 4.0,
        "zone_hi": pf["zhi"] / 4.0,
        "zone_created_ts": pf["form_ts_val"],
        "tap_no": pf["tap_no"],
        "tap_sec": pf["tap_sec"],
        "tap_ts": pf["tap_ts"],
        "r4_fire_sec": pf["tap_sec"] + pf["fire_s"],
        "p_both60": pf["p_both60"],
        "entry_sec": pf["t_open"].astype(int),
        "entry_px": pf["e_px"] / 4.0,
        "stop_px": pf["e_px"] / 4.0 + STOP,
        "target_px": pf["e_px"] / 4.0 - TGT,
        "exit_sec": pf["t_close"].astype(int),
        "gross_pnl_pts": pf["pnl"] + COST,
        "net_pnl_pts": pf["pnl"],
        "gross_R": (pf["pnl"] + COST) / STOP,
        "net_R": pf["pnl"] / STOP,
        "end_px": pf["end_px"] / 4.0,
    })
    led["entry_ts"] = led["tap_ts"] + (led["entry_sec"] - led["tap_sec"])
    led["exit_ts"] = led["tap_ts"] + (led["exit_sec"] - led["tap_sec"])
    led["outcome"] = np.where(np.isclose(led["net_pnl_pts"], TGT - COST),
                              "target",
                              np.where(np.isclose(led["net_pnl_pts"],
                                                  -STOP - COST), "stop",
                                       "session_close"))
    led["exit_px"] = np.where(led["outcome"] == "target", led["target_px"],
                              np.where(led["outcome"] == "stop",
                                       led["stop_px"], led["end_px"]))
    led["cost_pts_rt"] = COST
    led = led.sort_values(["session", "entry_sec"]).reset_index(drop=True)
    led.to_parquet(OUT)
    print(f"ledger: {len(led):,} trades → {OUT}")
    print(led["outcome"].value_counts().to_dict())
    print(f"expR {led['net_R'].mean():+.4f} | PF "
          f"{led.loc[led['net_R']>0,'net_R'].sum()/-led.loc[led['net_R']<=0,'net_R'].sum():.3f}"
          f" | WR {(led['net_R']>0).mean():.4f}")
    print(f"P12 reconcile: n 3,835 → {len(led):,}; expR +0.453 → "
          f"{led['net_R'].mean():+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
