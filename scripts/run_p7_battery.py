#!/usr/bin/env python
"""P7 Part 1 — the adversarial battery on the resistance-pivot edge.

The paired strict@resistance shadow contrast (real pivot vs symmetric
±10-pt shadows) recomputed inside every registered slice. Survival bar:
same sign in every slice; any slice flipping with CI excluding zero kills
or conditions the claim. Confirmation causality is enforced throughout —
age is measured from the confirmation timestamp, and the battery includes
confirmation-delay buckets.
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nqlevels.io.store import layer_dir                                        # noqa: E402
from nqlevels.labels.sweep import classify                                     # noqa: E402
from run_p6_spatial import add_labels, paired                                  # noqa: E402

CONFIGS = str(ROOT / "configs")
KEY = ["session", "zid", "tap_sec"]


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    re = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    sh = add_labels(pd.read_parquet(d0 / "p6_shadow.parquet"))
    it = pd.read_parquet(d0 / "interactions.parquet",
                         columns=KEY + ["scale", "tap_no", "age_min", "flipped",
                                        "release_slot", "atr_pctile",
                                        "range_used_atr", "tod_pt_min", "hw_atr"])
    it = it.drop_duplicates(KEY)
    cf = pd.read_parquet(d0 / "zones_confirm.parquet")
    sp = pd.read_parquet(d0 / "p6_spatial_feats.parquet",
                         columns=KEY + ["nest"])
    re = re.merge(it, on=KEY, how="left", validate="1:1") \
           .merge(cf, on="zid", how="left").merge(sp, on=KEY, how="left")
    re = re[re["has_t0"] & ~re["inside_t0"] & (re["n_fwd"] >= 1800)
            & re["y_strict"].notna()].copy()
    re["session_c"] = re["session"]
    # mid-cell outcome as an extra reaction-definition variant
    for d_ in (re, sh):
        c = classify(d_, R=15, d0_t=4.0, D_t=d_["atr_t"] * 0.05, T_s=900,
                     level="ne")
        d_["y_mid"] = c["y_any"]
    # roll-adjacent sessions
    adj = pd.read_parquet(d0 / "roll_adjust.parquet")
    adj = adj.sort_values("session").reset_index(drop=True)
    chg = adj.index[adj["epoch"].diff().fillna(0) > 0]
    roll_near = set()
    for i in chg:
        for j in range(max(0, i - 3), min(len(adj), i + 4)):
            roll_near.add(adj.loc[j, "session"])
    re["roll_adj"] = re["session"].isin(roll_near)
    # day character from bars_1m
    b1m = pd.read_parquet(d0 / "bars_1m.parquet",
                          columns=["session", "high_t", "low_t", "close_t",
                                   "open_t", "volume", "is_front"])
    b1m = b1m[b1m["is_front"]]
    day = b1m.groupby("session").agg(hi=("high_t", "max"), lo=("low_t", "min"),
                                     op=("open_t", "first"), cl=("close_t", "last"),
                                     vol=("volume", "sum"))
    day["trendiness"] = (day["cl"] - day["op"]).abs() / (day["hi"] - day["lo"])
    day["vol_rel"] = day["vol"] / day["vol"].rolling(20).median().shift(1)
    re = re.merge(day[["trendiness", "vol_rel"]], left_on="session",
                  right_index=True, how="left")

    g0 = re[re["side_above"] == 0]
    s0 = sh[sh["side_above"] == 0]
    res: dict = {}
    print(f"resistance events: {len(g0):,}")
    print(f"{'slice':>22} {'level':>14} {'diff':>8} {'lo':>8} {'hi':>8} {'n':>8}")

    def run(name, mask_fn_levels, col="y_strict"):
        for label, mask in mask_fn_levels:
            g = g0[mask(g0)]
            r = paired(g, s0, col, 10.0, 200)   # paired() joins on KEY, so the
            if r is None:                        # slice restricts shadows too
                print(f"{name:>22} {label:>14}: underpowered ({len(g):,})")
                continue
            res.setdefault(name, {})[label] = r
            flag = "" if r["diff"] > 0 else ("  <-- SIGN FLIP" if r["hi"] < 0
                                             else "  (neg, ci spans 0)")
            print(f"{name:>22} {label:>14} {r['diff']:8.4f} {r['lo']:8.4f} "
                  f"{r['hi']:8.4f} {r['n_events']:8,}{flag}")

    def ter(c):
        q1, q2 = g0[c].quantile([1 / 3, 2 / 3])
        return [("T1", lambda d, a=q1: d[c] <= a),
                ("T2", lambda d, a=q1, b=q2: (d[c] > a) & (d[c] <= b)),
                ("T3", lambda d, b=q2: d[c] > b)]

    run("vol regime (atr pct)", ter("atr_pctile"))
    run("trendiness", ter("trendiness"))
    run("day volume rel", ter("vol_rel"))
    run("distance at t0", ter("dist_t0_t"))
    run("band half-width", ter("hw_t"))
    run("nesting", ter("nest"))
    run("confirm delay", [("<=5m", lambda d: d["delay_min"] <= 5),
                          ("6-30m", lambda d: d["delay_min"].between(6, 30)),
                          (">30m", lambda d: d["delay_min"] > 30)])
    run("age", [("<1h", lambda d: d["age_min"] < 60),
                ("1-4h", lambda d: d["age_min"].between(60, 240)),
                ("4-12h", lambda d: d["age_min"].between(240, 720)),
                ("12-24h", lambda d: d["age_min"].between(720, 1440)),
                ("1-2sess", lambda d: d["age_min"].between(1440, 2880)),
                (">2sess", lambda d: d["age_min"] > 2880)])
    run("tap number", [("first", lambda d: d["tap_no"] == 1),
                       ("2-3", lambda d: d["tap_no"].between(2, 3)),
                       ("4-6", lambda d: d["tap_no"] >= 4)])
    run("pivot scale", [("s0", lambda d: d["scale"] == 0),
                        ("s1+", lambda d: d["scale"] >= 1)])
    run("flip state", [("intact", lambda d: d["flipped"] == 0),
                       ("flipped", lambda d: d["flipped"] == 1)])
    run("roll-adjacent", [("near roll", lambda d: d["roll_adj"]),
                          ("normal", lambda d: ~d["roll_adj"])])
    run("release slot", [("in slot", lambda d: d["release_slot"] == 1),
                         ("outside", lambda d: d["release_slot"] == 0)])
    run("outcome def", [("strict", lambda d: d["y_strict"].notna())])
    run("outcome=mid", [("all", lambda d: d["y_mid"].notna())], col="y_mid")
    run("outcome=primary", [("all", lambda d: d["y"].notna())], col="y")
    run("magnitude", [("all", lambda d: d["rev_ep_pts"].notna())],
        col="rev_ep_pts")

    # lead time from confirmation to tap (resistance)
    q = g0["age_min"] / 60.0
    print(f"\nlead time confirm->tap (resistance): med {q.median():.1f}h "
          f"p25 {q.quantile(.25):.1f} p75 {q.quantile(.75):.1f} "
          f"p90 {q.quantile(.9):.1f}")
    res["lead_h"] = {"med": float(q.median()), "p25": float(q.quantile(.25)),
                     "p75": float(q.quantile(.75)), "p90": float(q.quantile(.9))}

    out = ROOT / "outputs" / "p7_battery.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
