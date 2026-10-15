#!/usr/bin/env python
"""P10 stage 1 — family selection on 2021-2023 ONLY (registered staging).

Compares entry families, stop families and target families for each frozen
confirmation source, on the selection era. Everything chosen here executes
unchanged on 2024-2026 in stage 2. Shared evaluation helpers live here.
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

CONFIGS = str(ROOT / "configs")
KEY = ["session", "zid", "tap_sec"]
S1_PTS = (5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0)
S2_B = (0.0, 5.0, 10.0, 15.0, 20.0)
S3_B = (1.0, 2.0, 4.0, 6.0, 8.0)
S4_ATR = (0.02, 0.04)
R_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
T_FIXED = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0)
T_ALL = tuple(sorted(set(T_FIXED)
                     | {round(s * r, 2) for s in S1_PTS for r in R_GRID}))
T_IX = {v: i for i, v in enumerate(T_ALL)}
COST_MKT = {"opt": 0.5, "base": 1.0, "stress": 2.0}
COST_LIM = {"opt": 0.25, "base": 0.75, "stress": 1.75}


def load_trades():
    d0 = layer_dir("daily", CONFIGS)
    t = pd.read_parquet(d0 / "p10_trades.parquet")
    sc = pd.read_parquet(d0 / "p10_scores.parquet")
    t = t.merge(sc, on=KEY, how="left")
    t["year"] = pd.to_datetime(t["session"].astype(str)).dt.year
    return t


def trade_pnl(df: pd.DataFrame, stop_col: str, s_pts, t_pts: float,
              cost: float):
    """Vector P&L for one (stop, target) config. s_pts scalar (S1) or a
    per-row column name for variable-stop families."""
    ts = df[stop_col].to_numpy()
    tt = df[f"tt_{T_IX[t_pts]}"].to_numpy()
    S = df[s_pts].to_numpy("float64") if isinstance(s_pts, str) \
        else np.full(len(df), float(s_pts))
    ok = S > 0.5
    end_pnl = (df["e_px"].to_numpy() - df["end_px"].to_numpy()) / 4.0
    win = (tt >= 0) & ((ts < 0) | (tt < ts))          # ties -> stop-first
    loss = (ts >= 0) & ~win
    pnl = np.where(win, t_pts, np.where(loss, -S, end_pnl)) - cost
    return pnl[ok], S[ok], win[ok], loss[ok], df.loc[ok, "session"].to_numpy()


def metrics(pnl, S, win, loss, sessions):
    if pnl.size < 30:
        return {}
    R = pnl / S
    wr = float((pnl > 0).mean())
    gw = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    eqr = np.cumsum(R)
    dd = float((np.maximum.accumulate(eqr) - eqr).max())
    ls, cur = 0, 0
    for x in pnl:
        cur = cur + 1 if x <= 0 else 0
        ls = max(ls, cur)
    return {"n": int(pnl.size), "wr": wr, "exp_pts": float(pnl.mean()),
            "exp_R": float(R.mean()), "pf": float(gw / gl) if gl > 0 else np.inf,
            "avg_win": float(pnl[pnl > 0].mean()) if (pnl > 0).any() else np.nan,
            "avg_loss": float(pnl[pnl <= 0].mean()) if (pnl <= 0).any() else np.nan,
            "maxdd_R": dd, "streak": ls,
            "tpd": float(pnl.size / max(1, pd.unique(sessions).size))}


def eval_cfg(df, entry, stop_col, s_pts, t_pts, cost_kind="base",
             src=None, thr=None):
    d = df[(df["entry"] == entry) & (df["filled"] == True)]        # noqa: E712
    if src:
        d = d[d["src"].isin(src)] if isinstance(src, (list, tuple)) \
            else d[d["src"] == src]
    if thr is not None:
        d = d[d["p_both60"] >= thr]
    cost = (COST_MKT if entry in ("E1", "E1d") else COST_LIM)[cost_kind]
    return metrics(*trade_pnl(d, stop_col, s_pts, t_pts, cost))


def main() -> int:
    t0 = _time.time()
    t = load_trades()
    sel = t[t["year"] <= 2023]
    res: dict = {}
    print(f"selection era 2021-2023: {len(sel):,} rows | filled "
          f"{float(sel['filled'].mean()):.3f}")

    # fill rates by entry (r4)
    print("\nfill rates (r4):")
    for e, g in sel[sel["src"] == "r4"].groupby("entry"):
        print(f"  {e:>7}: filled {float(g['filled'].mean()):.3f}")

    print("\nA. entry families (r4), net base costs, grid-averaged "
          "expectancy_R over S1 {7.5,10,15} x T {15,20,30}:")
    entries = ["E1", "E1d", "E2p2.0", "E2p4.0", "E2p6.0", "E2p8.0",
               "E2p10.0", "E3r25", "E3r33", "E3r50", "E3r66", "E4"]
    s1ix = {v: i for i, v in enumerate(S1_PTS)}
    for e in entries:
        vals, ns = [], []
        for sp in (7.5, 10.0, 15.0):
            for tp in (15.0, 20.0, 30.0):
                r = eval_cfg(sel, e, f"ts1_{s1ix[sp]}", sp, tp, src="r4")
                if r:
                    vals.append(r["exp_R"])
                    ns.append(r["n"])
        if vals:
            print(f"  {e:>8}: mean expR {np.mean(vals):+.3f} | n~{int(np.mean(ns)):,}")
            res.setdefault("entry_r4", {})[e] = float(np.mean(vals))

    print("\nB. stop families (r4, best entry from A applied), T=20:")
    best_e = max(res["entry_r4"], key=res["entry_r4"].get)
    print(f"  [selection-era best entry: {best_e}]")
    for fam, cols in (("S1", [(f"ts1_{i}", sp) for i, sp in enumerate(S1_PTS)]),
                      ("S2", [(f"ts2_{i}", f"s2v_{i}") for i in range(len(S2_B))]),
                      ("S3", [(f"ts3_{i}", f"s3v_{i}") for i in range(len(S3_B))]),
                      ("S4", [(f"ts4_{i}", f"s4v_{i}") for i in range(len(S4_ATR))])):
        for col, sv in cols:
            r = eval_cfg(sel, best_e, col, sv, 20.0, src="r4")
            if r:
                lab = sv if isinstance(sv, str) else f"{sv}pts"
                print(f"  {fam} {lab:>8}: WR {r['wr']:.3f} expR {r['exp_R']:+.3f} "
                      f"expPts {r['exp_pts']:+.2f} PF {r['pf']:.2f} n {r['n']:,}")
                res.setdefault("stops_r4", {})[f"{fam}_{lab}"] = r

    print("\nC. target grid (r4, best entry, S1=10):")
    for tp in T_FIXED:
        r = eval_cfg(sel, best_e, f"ts1_{s1ix[10.0]}", 10.0, tp, src="r4")
        if r:
            print(f"  T={tp:>5}: WR {r['wr']:.3f} expR {r['exp_R']:+.3f} "
                  f"expPts {r['exp_pts']:+.2f} PF {r['pf']:.2f}")
            res.setdefault("targets_r4", {})[tp] = r

    print("\nD. same for stall source (st), E1, S1 grid x T=20 + targets:")
    for sp in S1_PTS:
        r = eval_cfg(sel, "E1", f"ts1_{s1ix[sp]}", sp, 20.0, src="st")
        if r:
            print(f"  S={sp:>5}: WR {r['wr']:.3f} expR {r['exp_R']:+.3f} n {r['n']:,}")
    for tp in (10.0, 15.0, 20.0, 30.0, 40.0):
        r = eval_cfg(sel, "E1", f"ts1_{s1ix[10.0]}", 10.0, tp, src="st")
        if r:
            print(f"  T={tp:>5}: WR {r['wr']:.3f} expR {r['exp_R']:+.3f} "
                  f"expPts {r['exp_pts']:+.2f}")

    print("\nE. model thresholds (framework r4+st, E1, S=10, T=20):")
    for thr in (None, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        r = eval_cfg(sel, "E1", f"ts1_{s1ix[10.0]}", 10.0, 20.0,
                     src=("r4", "st"), thr=thr)
        if r:
            print(f"  thr={thr}: WR {r['wr']:.3f} expR {r['exp_R']:+.3f} "
                  f"expPts {r['exp_pts']:+.2f} n {r['n']:,} tpd {r['tpd']:.2f}")

    out = ROOT / "outputs" / "p10_stage1.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
