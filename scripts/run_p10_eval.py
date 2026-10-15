#!/usr/bin/env python
"""P10 stage 2 — finalists frozen from 2021-2023, executed unchanged on
2024-2026, with the full evaluation battery.

Finalists (selection rules in §21.1, applied to stage-1 output):
  F1 balanced   r4 / E1d / S1=12.5 / T=30
  F2 max-exp    r4 / E1d / S1=12.5 / T=40
  F3 high-WR    frozen from the 21-23 frontier scan (WR-max subject to
                expR ≥ 0.05 and ±neighbour plateau)
  F4 model-gated r4 ∩ p60 ≥ 0.85 / E1d / S=12.5 / T=30
  F5 frequency  (r4 ∪ st) ∩ p60 ≥ 0.70 / E1 / S=10 / T=20
Rejected upstream: stall standalone (negative), ungated framework
(negative), pullback/retest entries (negative or dominated).
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

from run_p10_stage1 import (COST_LIM, COST_MKT, KEY, S1_PTS, T_ALL, T_IX,     # noqa: E402
                            load_trades, metrics, trade_pnl)

S1IX = {v: i for i, v in enumerate(S1_PTS)}
SEG_N = {0: "asia", 1: "london", 2: "premkt", 3: "ny-am", 4: "ny-pm"}


def rows_for(t, entry, src, thr):
    d = t[(t["entry"] == entry) & (t["filled"] == True)]             # noqa: E712
    d = d[d["src"].isin(src if isinstance(src, tuple) else (src,))]
    if thr is not None:
        d = d[d["p_both60"] >= thr]
    return d


def config_eval(t, name, entry, src, thr, s_pts, t_pts, cost_kind="base",
                sub=None):
    d = rows_for(t, entry, src, thr)
    if sub is not None:
        d = d[sub(d)]
    cost = (COST_MKT if entry in ("E1", "E1d") else COST_LIM)[cost_kind]
    return metrics(*trade_pnl(d, f"ts1_{S1IX[s_pts]}", s_pts, t_pts, cost))


def portfolio(t, entry, src, thr, s_pts, t_pts, cost_kind="base"):
    """Chronological one-open-trade pass; returns per-trade frame."""
    d = rows_for(t, entry, src, thr).copy()
    cost = (COST_MKT if entry in ("E1", "E1d") else COST_LIM)[cost_kind]
    pnl, S, win, loss, sess = trade_pnl(d, f"ts1_{S1IX[s_pts]}", s_pts,
                                        t_pts, cost)
    dd = d.copy()          # fixed-S1 stops: trade_pnl keeps every row (S>0.5)
    dd["pnl"] = pnl
    ts = dd[f"ts1_{S1IX[s_pts]}"].to_numpy()
    tt = dd[f"tt_{T_IX[t_pts]}"].to_numpy()
    close_rel = np.where((tt >= 0) & ((ts < 0) | (tt < ts)), tt,
                         np.where(ts >= 0, ts, dd["n_left"].to_numpy()))
    dd["t_open"] = dd["tap_sec"] + dd["fire_s"] + dd["e_delay"]
    dd["t_close"] = dd["t_open"] + close_rel
    dd = dd.sort_values(["session", "t_open"])
    keep = []
    cur_sess, cur_close = None, -1
    for r in dd.itertuples():
        if r.session != cur_sess or r.t_open >= cur_close:
            keep.append(True)
            cur_sess, cur_close = r.session, r.t_close
        else:
            keep.append(False)
    return dd[np.array(keep)]


def show(name, m_, extra=""):
    if not m_:
        print(f"  {name}: underpowered")
        return
    print(f"  {name:>22}: n {m_['n']:6,} tpd {m_['tpd']:5.2f} | WR "
          f"{m_['wr']:.3f} | expR {m_['exp_R']:+.3f} expPts {m_['exp_pts']:+.2f} "
          f"| PF {m_['pf']:.2f} | DD {m_['maxdd_R']:.0f}R | streak "
          f"{m_['streak']}{extra}")


def main() -> int:
    t0 = _time.time()
    t = load_trades()
    res: dict = {}

    # F3 frontier scan on 21-23 (frozen selection rule)
    sel = t[t["year"] <= 2023]
    best = None
    for sp in (15.0, 20.0, 25.0, 30.0):
        for tp in (10.0, 15.0, 20.0):
            m_ = config_eval(sel, "f3scan", "E1d", "r4", None, sp, tp)
            if not m_ or m_["exp_R"] < 0.05:
                continue
            nb = []
            for sp2 in (sp - 2.5, sp + 2.5):
                for tp2 in (tp - 2.5, tp + 2.5):
                    if sp2 in S1_PTS and tp2 in T_ALL:
                        m2 = config_eval(sel, "", "E1d", "r4", None, sp2, tp2)
                        if m2:
                            nb.append(m2["exp_R"])
            if nb and min(nb) < m_["exp_R"] - 0.03:
                continue
            if best is None or m_["wr"] > best[2]["wr"]:
                best = (sp, tp, m_)
    f3_s, f3_t = (best[0], best[1]) if best else (20.0, 15.0)
    print(f"F3 frozen from 21-23 frontier: S={f3_s}, T={f3_t} "
          f"(21-23 WR {best[2]['wr']:.3f}, expR {best[2]['exp_R']:+.3f})")

    FINAL = {
        "F1 balanced": ("E1d", ("r4",), None, 12.5, 30.0),
        "F2 max-exp": ("E1d", ("r4",), None, 12.5, 40.0),
        "F3 high-WR": ("E1d", ("r4",), None, f3_s, f3_t),
        "F4 model-gated": ("E1d", ("r4",), 0.85, 12.5, 30.0),
        "F5 frequency": ("E1", ("r4", "st"), 0.70, 10.0, 20.0),
        "REJ stall-alone": ("E1", ("st",), None, 10.0, 20.0),
        "REJ framework-ungated": ("E1", ("r4", "st"), None, 10.0, 20.0),
    }
    for era, mask in (("2021-23 (selection)", t["year"] <= 2023),
                      ("2024-26 (OOS)", t["year"] >= 2024)):
        print(f"\n=== {era} (net base costs) ===")
        for name, (e, s_, th, sp, tp) in FINAL.items():
            m_ = config_eval(t[mask], name, e, s_, th, sp, tp)
            res.setdefault(era, {})[name] = m_
            show(name, m_)

    print("\n=== per-year, finalists ===")
    for name in ("F1 balanced", "F2 max-exp", "F3 high-WR", "F4 model-gated",
                 "F5 frequency"):
        e, s_, th, sp, tp = FINAL[name]
        line = []
        for y in (2021, 2022, 2023, 2024, 2025, 2026):
            m_ = config_eval(t[t["year"] == y], "", e, s_, th, sp, tp)
            if m_:
                line.append(f"{y}:{m_['exp_R']:+.2f}R/{m_['wr']:.2f}")
                res.setdefault("by_year", {}).setdefault(name, {})[y] = m_
        print(f"  {name:>15}: " + " ".join(line))

    print("\n=== sessions (F1, full period) ===")
    e, s_, th, sp, tp = FINAL["F1 balanced"]
    for sg, nm in SEG_N.items():
        m_ = config_eval(t, "", e, s_, th, sp, tp,
                         sub=lambda d, sg=sg: d["seg"] == sg)
        show(nm, m_)
        res.setdefault("seg_F1", {})[nm] = m_

    print("\n=== tap number (F1) ===")
    for lab, fn in (("first", lambda d: d["tap_no"] == 1),
                    ("2-3", lambda d: d["tap_no"].between(2, 3)),
                    ("4-6", lambda d: d["tap_no"] >= 4)):
        show(lab, config_eval(t, "", e, s_, th, sp, tp, sub=fn))

    print("\n=== costs (F1, OOS 24-26) ===")
    oos = t[t["year"] >= 2024]
    for ck in ("opt", "base", "stress"):
        show(ck, config_eval(oos, "", e, s_, th, sp, tp, cost_kind=ck))

    print("\n=== delay: E1 vs E1d (F1 geometry, OOS) ===")
    for ee in ("E1", "E1d"):
        show(ee, config_eval(oos, "", ee, s_, th, sp, tp))

    print("\n=== neighbour sensitivity (F1, OOS): S x T +/-2.5 ===")
    for sp2 in (10.0, 12.5, 15.0):
        line = []
        for tp2 in (25.0, 30.0, 40.0):
            m_ = config_eval(oos, "", e, s_, th, sp2, tp2)
            line.append(f"T{tp2}:{m_['exp_R']:+.3f}" if m_ else "-")
        print(f"  S={sp2}: " + " ".join(line))

    print("\n=== partial T2 (F1: half at 15, half at 40, stop 12.5, OOS) ===")
    d = rows_for(oos, "E1d", ("r4",), None)
    c = COST_MKT["base"]
    p1, S1v, w1, l1, ss = trade_pnl(d, f"ts1_{S1IX[12.5]}", 12.5, 15.0, c / 2)
    p2, S2v, w2, l2, _ = trade_pnl(d, f"ts1_{S1IX[12.5]}", 12.5, 40.0, c / 2)
    pnl = 0.5 * p1 + 0.5 * p2
    show("partial 15/40", metrics(pnl, S1v, pnl > 0, pnl <= 0, ss))

    print("\n=== portfolio one-open-trade (finalists, full period) ===")
    for name in ("F1 balanced", "F3 high-WR", "F4 model-gated", "F5 frequency"):
        e2, s2, th2, sp2, tp2 = FINAL[name]
        pf_ = portfolio(t, e2, s2, th2, sp2, tp2)
        R = pf_["pnl"] / sp2
        per = pf_.groupby("session").size()
        n_days = t["session"].nunique()
        m_ = metrics(pf_["pnl"].to_numpy(), np.full(len(pf_), sp2),
                     pf_["pnl"].to_numpy() > 0, pf_["pnl"].to_numpy() <= 0,
                     pf_["session"].to_numpy())
        z = 1 - per.reindex(t["session"].unique()).notna().mean()
        show(name, m_, f" | zero-trade days {float(z):.2f}")
        res.setdefault("portfolio", {})[name] = m_

    out = ROOT / "outputs" / "p10_eval.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
