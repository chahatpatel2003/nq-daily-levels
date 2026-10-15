#!/usr/bin/env python
"""P23 aggregation: raw + region-deduped ledgers, head-to-head, yearly
folds, sessions, taps, scales, entry quality."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

df = pd.read_parquet("/Volumes/T9/nq_levels/daily/p23_trades.parquet")
NS = df["session"].nunique()
NS_BY_YEAR = df.groupby("year")["session"].nunique()

# region ids per session: overlap components of traded zones
zg = df[["session", "zid", "L", "U"]].drop_duplicates(["session", "zid"])
reg_map = {}
for sess, g in zg.groupby("session"):
    g = g.sort_values("L")
    rid, cur_hi = 0, -1e18
    for r in g.itertuples():
        if r.L > cur_hi:
            rid += 1
            cur_hi = r.U
        else:
            cur_hi = max(cur_hi, r.U)
        reg_map[(sess, r.zid)] = rid
df["region"] = [reg_map[(s, z)] for s, z in zip(df["session"], df["zid"])]


def dedup(d):
    """One open trade per (session, region): chronological."""
    d = d.sort_values(["session", "e_idx"])
    keep = []
    open_until = {}
    for r in d.itertuples():
        k = (r.session, r.region)
        if r.e_idx >= open_until.get(k, -1):
            keep.append(True)
            open_until[k] = r.x_idx
        else:
            keep.append(False)
    return d[np.array(keep)]


def stats(d, label="", show=True, ny=True):
    if len(d) < 100:
        return None
    r = d["netR"].to_numpy()
    w = r[r > 0]; l_ = r[r <= 0]
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    out = dict(n=len(r), tpd=len(r) / NS, wr=float((d["res"] == 1).mean()),
               expR=float(r.mean()), pf=float(w.sum() / -l_.sum()),
               totR=float(r.sum()), dd=dd,
               mae=float(d["mae"].median()), mfe=float(d["mfe"].median()),
               rr=float(d["rr"].median()),
               hold=float((d["x_idx"] - d["e_idx"]).median()))
    if ny:
        dn = d[d["seg"] == 3]
        out["ny_tpd"] = len(dn) / NS
        out["ny_expR"] = float(dn["netR"].mean()) if len(dn) > 100 else np.nan
    if show:
        print(f"  {label:>16}: n {out['n']:7,} ({out['tpd']:5.2f}/d, NY "
              f"{out.get('ny_tpd', 0):4.2f}) WR {out['wr']:.3f} expR "
              f"{out['expR']:+.4f} PF {out['pf']:.3f} totR "
              f"{out['totR']:8.0f} DD {out['dd']:6.0f} | medRR "
              f"{out['rr']:.2f} MAE {out['mae']:.2f} MFE {out['mfe']:.2f}"
              f" | NY expR {out.get('ny_expR', float('nan')):+.3f}")
    return out


def main() -> int:
    print(f"sessions {NS} | total trades {len(df):,}")
    variants = [("A_mid", 0)] + [(s, tf) for s in
                ("B_body_inz", "B_body_p5", "B_rng_inz", "B_rng_p5",
                 "C_rej", "D1_inz", "D2_below", "E_lowbrk")
                for tf in (30, 60, 120, 180, 300)]
    res_raw, res_dd = {}, {}
    print("\n=== RAW LEDGER (every child zone) ===")
    for s, tf in variants:
        d = df[(df["strat"] == s) & (df["tf"] == tf)]
        res_raw[(s, tf)] = stats(d, f"{s}@{tf}")
    print("\n=== REGION-DEDUPED LEDGER ===")
    for s, tf in variants:
        d = df[(df["strat"] == s) & (df["tf"] == tf)]
        res_dd[(s, tf)] = stats(dedup(d), f"{s}@{tf}")

    print("\n=== YEARLY expR (deduped; key variants) ===")
    keys = [("A_mid", 0), ("B_body_inz", 60), ("B_body_p5", 60),
            ("B_rng_inz", 60), ("C_rej", 60), ("D2_below", 60),
            ("E_lowbrk", 60), ("C_rej", 120), ("D2_below", 120)]
    for s, tf in keys:
        d = dedup(df[(df["strat"] == s) & (df["tf"] == tf)])
        row = []
        for y in ("2021", "2022", "2023", "2024", "2025", "2026"):
            g = d[d["year"] == y]
            row.append(f"{g['netR'].mean():+.3f}" if len(g) > 100
                       else "   ·  ")
        print(f"  {s:>10}@{tf:>3}: " + " ".join(row))

    print("\n=== TAP / SCALE splits (deduped, key variants, expR) ===")
    for s, tf in [("A_mid", 0), ("C_rej", 60), ("B_body_inz", 60)]:
        d = dedup(df[(df["strat"] == s) & (df["tf"] == tf)])
        taps = " ".join(f"tap{k}:{d[d['tap_no']==k]['netR'].mean():+.3f}"
                        for k in (1, 2, 3))
        t4 = d[d["tap_no"] >= 4]["netR"].mean()
        scs = " ".join(f"s{k}:{d[d['scale']==k]['netR'].mean():+.3f}"
                       for k in (0, 1, 2)
                       if len(d[d["scale"] == k]) > 100)
        print(f"  {s:>10}@{tf:>3}: {taps} tap4+:{t4:+.3f} | {scs}")

    print("\n=== ENTRY QUALITY ===")
    for s, tf in [("A_mid", 0), ("B_body_inz", 60), ("B_body_p5", 60),
                  ("C_rej", 60), ("D2_below", 60)]:
        d = df[(df["strat"] == s) & (df["tf"] == tf)]
        below = float((d["e_px"] < d["L"]).mean())
        rrlt = float((d["rr"] < 0.5).mean())
        rr1 = float((d["rr"] >= 1.0).mean())
        print(f"  {s:>10}@{tf:>3}: entry-below-zone {below:.0%} | "
              f"RR<0.5 {rrlt:.0%} | RR≥1 {rr1:.0%} | med RR "
              f"{d['rr'].median():.2f}")
    # A breakeven
    d = df[df["strat"] == "A_mid"]
    win_net = (d["rr"] * d["risk"] - 4.0) / d["risk"]
    print(f"\nA_mid: breakeven WR after costs ≈ "
          f"{(1+4/d['risk'].mean())/2:.3f} vs actual "
          f"{(d['res']==1).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
