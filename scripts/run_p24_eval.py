#!/usr/bin/env python
"""P24 evaluator: bracket surfaces (raw + region-dedup), tap variants,
folds, NY-AM, MAE/MFE, survival matrix, FADE-1 reproduction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STOPS = (10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0)
TGTS = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0)
COST = 1.0
BLOCKS = [("2022-01-01", "2022-06-30"), ("2022-07-01", "2022-12-31"),
          ("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-31"),
          ("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"),
          ("2025-01-01", "2025-06-30"), ("2025-07-01", "2025-12-31"),
          ("2026-01-01", "2026-06-17"), ("2026-06-18", "2026-08-14")]

df = pd.read_parquet(ROOT / "outputs" / "p24_direct_entry_trades.parquet")
NS = df["session"].nunique()
# region ids per session (overlap components over traded zones)
zg = df[["session", "zid", "L", "U"]].drop_duplicates(["session", "zid"])
reg = {}
for sess, g in zg.groupby("session"):
    g = g.sort_values("L")
    rid, cur = 0, -1e18
    for r in g.itertuples():
        if r.L > cur:
            rid += 1
            cur = r.U
        else:
            cur = max(cur, r.U)
        reg[(sess, r.zid)] = rid
df["region"] = [reg[(s, z)] for s, z in zip(df["session"], df["zid"])]


def outcomes(d, S, T):
    up = d[f"up{S}"].to_numpy()
    dn = d[f"dn{T}"].to_numpy()
    win = (dn >= 0) & ((up < 0) | (dn < up))
    loss = (up >= 0) & ~win
    end = (d["L"] - d["end_px"]).to_numpy() / 4.0
    pnl = np.where(win, T, np.where(loss, -S, end)) - COST
    close_rel = np.where(win, dn, np.where(loss, up, 10 ** 9))
    return pnl / S, win, close_rel


def dedup_pass(d, close_rel, key_cols):
    d = d.assign(closex=d["e_idx"] + close_rel) \
        .sort_values(["session", "e_idx"])
    keep = []
    open_until = {}
    for r in d.itertuples():
        k = tuple(getattr(r, c) for c in key_cols)
        if r.e_idx >= open_until.get(k, -1):
            keep.append(True)
            open_until[k] = r.closex
        else:
            keep.append(False)
    return d[np.array(keep)]


def metrics(rr, d=None):
    w, l_ = rr[rr > 0], rr[rr <= 0]
    eq = np.cumsum(rr)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return dict(n=len(rr), wr=float((rr > 0).mean()),
                expR=float(rr.mean()),
                pf=float(w.sum() / -l_.sum()) if len(l_) else np.inf,
                totR=float(rr.sum()), dd=dd)


def cell_eval(d, S, T, mode="region"):
    r, win, cr = outcomes(d, S, T)
    d2 = d.assign(rnet=r)
    key = ["session", "region"] if mode == "region" else ["session", "zid"]
    dd2 = dedup_pass(d2, cr, key)
    return dd2["rnet"].to_numpy(), dd2


def main() -> int:
    res = {}
    print(f"rows {len(df):,} | sessions {NS}")
    for fill in ("B1", "B2"):
        for tapv, tmask in (("all", df["tap_no"] >= 1),
                            ("tap1", df["tap_no"] == 1)):
            sub = df[(df["fill"] == fill) & tmask]
            surf = []
            for S in STOPS:
                for T in TGTS:
                    rr, dd2 = cell_eval(sub, S, T)
                    m = metrics(rr)
                    m.update(S=S, T=T, fill=fill, tap=tapv,
                             tpd=m["n"] / NS,
                             ny_tpd=float((dd2["seg"] == 3).sum()) / NS,
                             ny_expR=float(dd2.loc[dd2["seg"] == 3, "rnet"]
                                           .mean()))
                    surf.append(m)
            sdf = pd.DataFrame(surf)
            sdf.to_csv(ROOT / "outputs" /
                       f"p24_surface_{fill}_{tapv}.csv", index=False)
            pos = sdf[sdf["expR"] > 0]
            print(f"\n[{fill} {tapv}] positive cells: {len(pos)}/56")
            if len(pos):
                print(pos.sort_values("expR", ascending=False)
                      [["S", "T", "n", "wr", "expR", "pf", "dd", "tpd",
                        "ny_expR"]].head(10).round(3).to_string(index=False))
            res[f"{fill}_{tapv}_pos"] = len(pos)

    print("\n=== yearly + folds for key cells (B2, region-dedup) ===")
    keys = [(20.0, 30.0, "tap1"), (20.0, 30.0, "all"),
            (17.5, 30.0, "tap1"), (25.0, 30.0, "tap1"),
            (20.0, 25.0, "tap1"), (20.0, 40.0, "tap1"),
            (15.0, 30.0, "tap1"), (25.0, 40.0, "tap1")]
    for S, T, tapv in keys:
        sub = df[(df["fill"] == "B2") &
                 ((df["tap_no"] == 1) if tapv == "tap1"
                  else (df["tap_no"] >= 1))]
        rr, dd2 = cell_eval(sub, S, T)
        yr = dd2.groupby(dd2["session"].str[:4])["rnet"].mean()
        folds = []
        for lo, hi in BLOCKS:
            g = dd2[(dd2["session"] >= lo) & (dd2["session"] <= hi)]
            folds.append(float(g["rnet"].mean()) if len(g) > 50 else np.nan)
        fl = [f for f in folds if np.isfinite(f)]
        print(f"  S{S}/T{T} {tapv:>4}: expR {rr.mean():+.3f} | years " +
              " ".join(f"{v:+.2f}" for v in yr) +
              f" | folds pos {sum(1 for f in fl if f>0)}/{len(fl)} "
              f"worst {min(fl):+.3f}")
        res[f"cell_{S}_{T}_{tapv}"] = dict(expR=float(rr.mean()),
                                           folds=fl)

    print("\n=== FADE-1 reproduction (tap1, B2, S20/T30, "
          "P18 conventions) ===")
    sub = df[(df["fill"] == "B2") & (df["tap_no"] == 1)]
    sub = sub.sort_values(["session", "tap_sec"]) \
        .groupby(["session", "zid"], as_index=False).first()
    r, win, cr = outcomes(sub, 20.0, 30.0)
    # P18 forced-loss variant: non-win = -S
    r_forced = np.where(win, (30.0 - COST) / 20.0, -(20.0 + COST) / 20.0)
    d2 = sub.assign(rnet=r, rforce=r_forced)
    dd2 = dedup_pass(d2, np.where(win, sub[f"dn30.0"],
                                  np.where(sub[f"up20.0"] >= 0,
                                           sub[f"up20.0"], 1800)),
                     ["session"])          # GLOBAL one-open (P18 style)
    print(f"  trades {len(dd2):,} | 1s-sim expR {dd2['rnet'].mean():+.3f} "
          f"WR {(dd2['rnet']>0).mean():.3f} | P18-outcome-model expR "
          f"{dd2['rforce'].mean():+.3f} WR {(dd2['rforce']>0).mean():.3f}")
    print("  (P18 reported +0.342 pooled on its grid-approximated model)")
    res["fade1"] = dict(n=len(dd2), sim=float(dd2["rnet"].mean()),
                        forced=float(dd2["rforce"].mean()))
    json.dump(res, open(ROOT / "outputs" / "p24_summary.json", "w"),
              indent=1, default=float)

    print("\n=== survival matrix (B2 tap1 entries): P(adv>=X before "
          "fav>=Y) ===")
    print(f"{'':>6}" + "".join(f"  +{int(t)}" for t in (10, 20, 30, 40)))
    for S in (10.0, 15.0, 20.0, 25.0, 30.0):
        rowp = []
        for T in (10.0, 20.0, 30.0, 40.0):
            up = sub[f"up{S}"].to_numpy()
            dn = sub[f"dn{T}"].to_numpy()
            first_adv = (up >= 0) & ((dn < 0) | (up < dn))
            rowp.append(first_adv.mean())
        print(f"  -{int(S):>3}: " + " ".join(f"{p:.2f}" for p in rowp))
    mae = sub["mae1h"]
    print(f"\nMAE(1h) med {mae.median():.1f} p75 "
          f"{mae.quantile(.75):.1f} p90 {mae.quantile(.90):.1f} | "
          f"MFE med {sub['mfe1h'].median():.1f}")
    for X in (10, 15, 20, 25, 30):
        print(f"  adverse≥{X}: {(mae>=X).mean():.0%}", end="")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
