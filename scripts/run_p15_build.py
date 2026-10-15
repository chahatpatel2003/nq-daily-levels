#!/usr/bin/env python
"""P15: build + verify the five legal Phase-14 strategy ledgers and save
per-session R sequences for the lifecycle engine.

A PRIMARY  tap<=3 & fire_s<=24 & mono>=0.27, ck0, S12.5/T30
B T40      same population, S12.5/T40
C LOWDD    tap<=2 variant, S12.5/T30
D ULTRA    tap==1 variant, S12.5/T30
E HYBRID   A + walk-forward HGB top-70% (scored years only: 2023+)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_p14_model import FIRE, BASE_PRE, FIRE_F, HEALTH, fit_score            # noqa: E402
from run_p14_final import trades_for                                           # noqa: E402

OUT = Path("/Volumes/T9/nq_levels/daily/p15_streams.pkl")


def main() -> int:
    df = pd.read_parquet(FIRE)
    df["session"] = df["session"].astype(str)
    df["year"] = pd.to_datetime(df["session"]).dt.year
    df = df.sort_values(["session", "tap_sec"]).reset_index(drop=True)
    q50 = float(df.loc[df["year"] <= 2023, "fire_s"].quantile(0.5))
    qm = float(df.loc[df["year"] <= 2023, "mono"].median())
    fast = (df["fire_s"] <= q50) & (df["mono"] >= qm)
    ev = df["year"] >= 2022

    feats = [f for f in BASE_PRE + FIRE_F + HEALTH if f in df.columns]
    sc, thr, _ = fit_score(df, feats, 0, "hgb")
    keep70 = np.zeros(len(df), bool)
    for Y, t_ in thr.items():
        keep70 |= ((df["year"] == Y).to_numpy() & np.isfinite(sc)
                   & (sc >= t_[0.7]))

    defs = {
        "A_primary": (((df["tap_no"] <= 3) & fast & ev).to_numpy(), 30.0),
        "B_t40": (((df["tap_no"] <= 3) & fast & ev).to_numpy(), 40.0),
        "C_tap2": (((df["tap_no"] <= 2) & fast & ev).to_numpy(), 30.0),
        "D_ultra": (((df["tap_no"] == 1) & fast & ev).to_numpy(), 30.0),
        "E_hybrid": (((df["tap_no"] <= 3) & fast & keep70).to_numpy(), 30.0),
    }
    out = {"streams": {}, "sessions": {}}
    print(f"{'ledger':>10} {'n':>6} {'t/d':>5} {'WR':>6} {'expR':>7} "
          f"{'PF':>5} {'totR':>6} {'DDt':>6} {'DDd':>6} {'S':>5} {'T':>5}")
    for nm, (mask, tgt) in defs.items():
        d = trades_for(df[mask], 0, 12.5, tgt)
        sess = sorted(d["session"].unique())
        first_y = sess[0][:4]
        axis = sorted(df.loc[df["year"] >= int(first_y), "session"]
                      .unique())
        r = d["R"].to_numpy()
        w, l = r[r > 0], r[r <= 0]
        eq = np.cumsum(r)
        ddt = float((np.maximum.accumulate(eq) - eq).max())
        day = d.groupby("session")["R"].sum().reindex(axis).fillna(0.0)
        eqd = np.cumsum(day.to_numpy())
        ddd = float((np.maximum.accumulate(eqd) - eqd).max())
        print(f"{nm:>10} {len(r):>6,} {len(r)/len(axis):>5.2f} "
              f"{(r>0).mean():>6.3f} {r.mean():>+7.3f} "
              f"{w.sum()/-l.sum():>5.2f} {r.sum():>6.0f} {ddt:>6.1f} "
              f"{ddd:>6.1f} {12.5:>5} {tgt:>5}")
        g = {s: grp["R"].to_numpy() for s, grp in
             d.sort_values(["session", "t_open"]).groupby("session")}
        out["streams"][nm] = [g.get(s, np.empty(0)) for s in axis]
        out["sessions"][nm] = axis
    with OUT.open("wb") as f:
        pickle.dump(out, f)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
