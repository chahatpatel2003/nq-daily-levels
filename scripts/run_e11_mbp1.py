#!/usr/bin/env python
"""E11 — does pre-open ORDER BOOK history mark prices that matter?

Strictly a pre-open question. Touch-time book behaviour is confirmation
research and is out of scope for Phase 2; what is tested here is whether the
resting-liquidity history left at a price during the night predicts anything
about that price during 06:30-09:00 PT.

The prior is poor and should be stated up front: NQ's top of book averages ~3
contracts (Phase 1), and *traded* volume at price — a strictly richer signal
about where business actually happened — carries no information within distance
bands (within-band AUC 0.497). So this is a check, not a hope.

Bounded by design: a sample of sessions, the London + premarket window only, and
a hard stop at the 2025-06-30 MBP-1 coverage end. Features per price tick:
time-weighted resting size, seconds spent at the top of book, add and cancel
counts, a replenishment count, and traded volume.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from datetime import date, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.clock import (decision_time_utc, load_calendar, pt_instant)      # noqa: E402
from nqlevels.eval.quality import add_distance_bands                           # noqa: E402
from nqlevels.io.raw import assert_not_raw, read_mbp1                          # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402

CONFIGS = str(ROOT / "configs")
TICK = 0.25
MBP1_END = date(2025, 6, 30)


def book_features(q: pd.DataFrame, ref: float) -> pd.DataFrame:
    """Per-price-tick book history over one pre-open window.

    Time-weighting matters more than event counts here: a price that was the
    best bid for two hours with five contracts resting is a different object
    from one quoted for four seconds, and raw add/cancel counts cannot tell
    them apart.
    """
    if q.empty:
        return pd.DataFrame()
    q = q.sort_values("ts_event", kind="stable")
    dt = q["ts_event"].diff().dt.total_seconds().shift(-1).fillna(0.0).clip(0, 60).to_numpy()
    out = {}
    for side, pxc, szc in (("bid", "bid_px_00", "bid_sz_00"),
                           ("ask", "ask_px_00", "ask_sz_00")):
        px = np.rint(q[pxc].to_numpy("float64") / TICK) - ref
        sz = q[szc].to_numpy("float64")
        ok = np.isfinite(px) & np.isfinite(sz) & (sz > 0)
        df = pd.DataFrame({"t": px[ok].astype("int64"), "w": dt[ok],
                           "ws": dt[ok] * sz[ok]})
        g = df.groupby("t").agg(secs=("w", "sum"), wsize=("ws", "sum"))
        out[side] = g.rename(columns={"secs": f"{side}_secs", "wsize": f"{side}_wsize"})
    ev = q[q["action"].isin(["A", "C", "T"])].copy()
    ev["t"] = (np.rint(ev["price"].to_numpy("float64") / TICK) - ref).astype("int64")
    piv = ev.pivot_table(index="t", columns="action", values="size",
                         aggfunc=["count", "sum"], fill_value=0)
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    f = out["bid"].join(out["ask"], how="outer").join(piv, how="outer").fillna(0.0)
    f["bk_secs"] = f.get("bid_secs", 0) + f.get("ask_secs", 0)
    f["bk_twsize"] = ((f.get("bid_wsize", 0) + f.get("ask_wsize", 0))
                      / np.maximum(f["bk_secs"], 1e-9))
    f["bk_adds"] = f.get("count_A", 0)
    f["bk_cancels"] = f.get("count_C", 0)
    f["bk_tradevol"] = f.get("sum_T", 0)
    f["bk_replenish"] = np.minimum(f["bk_adds"], f["bk_cancels"])
    f["bk_absorb"] = f["bk_tradevol"] / np.maximum(f["bk_tradevol"] + f.get("sum_C", 0), 1.0)
    return f[["bk_secs", "bk_twsize", "bk_adds", "bk_cancels",
              "bk_tradevol", "bk_replenish", "bk_absorb"]].reset_index().rename(
        columns={"t": "off_t"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sessions", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = _time.time()
    cal = load_calendar(CONFIGS)
    d = layer_dir("daily", CONFIGS)

    m = pd.read_parquet(d / "axis_measured.parquet")
    m = m[(m["dist_atr"] <= 1.2) & (m["side"] != "inside") & (m["session"] <= MBP1_END)]
    ctx = pd.read_parquet(d / "context.parquet", columns=["session", "symbol", "ref_t"])
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    ref = dict(zip(ctx["session"], ctx["ref_t"]))
    sym = dict(zip(ctx["session"], ctx["symbol"]))

    rng = np.random.default_rng(a.seed)
    pool = np.array(sorted(m["session"].unique()))
    pick = np.sort(rng.choice(pool, size=min(a.n_sessions, pool.size), replace=False))
    print(f"MBP-1 pre-open test on {len(pick)} sampled sessions "
          f"({pick[0]} .. {pick[-1]}); coverage ends {MBP1_END}")

    frames, t_read = [], 0.0
    for i, s in enumerate(pick):
        lo = pd.Timestamp(pt_instant(s, dtime(0, 0), cal))
        hi = pd.Timestamp(decision_time_utc(s, cal))
        try:
            tr0 = _time.time()
            q = read_mbp1(lo, hi, CONFIGS, symbol=sym[s],
                          columns=("ts_event", "action", "side", "price", "size",
                                   "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"))
            t_read += _time.time() - tr0
        except Exception as e:                                       # noqa: BLE001
            print(f"  {s}: read failed ({e})")
            continue
        if q.empty:
            continue
        f = book_features(q, ref[s])
        f["session"] = s
        frames.append(f)
        if i % 20 == 0:
            print(f"  {s}: {len(q):,} book events -> {len(f):,} priced rows "
                  f"({_time.time()-t0:.0f}s, read {t_read:.0f}s)", flush=True)

    B = pd.concat(frames, ignore_index=True)
    print(f"\nbook features: {len(B):,} (session, price) rows over {B['session'].nunique()} sessions")
    j = m[m["session"].isin(B["session"].unique())].merge(B, on=["session", "off_t"], how="left")
    for c in ("bk_secs", "bk_twsize", "bk_adds", "bk_cancels", "bk_tradevol",
              "bk_replenish", "bk_absorb"):
        j[c] = j[c].fillna(0.0)
    print(f"joined to {len(j):,} grid prices; "
          f"{100*(j['bk_secs']>0).mean():.1f}% had any top-of-book time")

    def auc(y, s):
        y = np.asarray(y, "float64")
        s = np.asarray(s, "float64")
        ok = np.isfinite(y) & np.isfinite(s)
        y, s = y[ok], s[ok]
        if y.size < 50 or y.min() == y.max():
            return np.nan
        r = pd.Series(s).rank(method="average").to_numpy()
        n1 = y.sum()
        return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))

    print(f"\n{'='*90}\nWithin-distance-band AUC of each book feature "
          f"(0.500 = no information beyond distance)\n{'='*90}")
    jb = add_distance_bands(j.copy(), n=8)
    t = jb[jb["touched"] == True]                                    # noqa: E712
    feats = ["bk_secs", "bk_twsize", "bk_adds", "bk_cancels", "bk_tradevol",
             "bk_replenish", "bk_absorb"]
    print(f"  {'feature':>14} {'AUC touch':>10} {'AUC clean':>10}")
    res = {}
    for f in feats:
        at, ac = [], []
        wt, wc = [], []
        for b, g in jb.groupby("_dband", observed=True):
            v = auc(g["touched"], g[f])
            if np.isfinite(v):
                at.append(v)
                wt.append(len(g))
        for b, g in t.groupby("_dband", observed=True):
            v = auc(g["clean"].astype("float64"), g[f])
            if np.isfinite(v):
                ac.append(v)
                wc.append(len(g))
        A = float(np.average(at, weights=wt)) if at else np.nan
        C = float(np.average(ac, weights=wc)) if ac else np.nan
        res[f] = {"auc_touch": A, "auc_clean": C}
        print(f"  {f:>14} {A:10.4f} {C:10.4f}")
    # reference: traded volume-at-price from the profile layer, same treatment
    for f in ("on_volrel", "dist_atr", "near_core"):
        if f not in jb:
            continue
        ac, wc = [], []
        for b, g in t.groupby("_dband", observed=True):
            v = auc(g["clean"].astype("float64"), g[f])
            if np.isfinite(v):
                ac.append(v)
                wc.append(len(g))
        print(f"  {('[ref] '+f):>14} {'':>10} {float(np.average(ac, weights=wc)):10.4f}")

    (ROOT / "outputs" / "e11_mbp1.json").write_text(json.dumps(
        {"n_sessions": int(B["session"].nunique()), "features": res}, indent=2, default=str))
    print(f"\nwrote outputs/e11_mbp1.json  ({(_time.time()-t0)/60:.1f} min, "
          f"{t_read/60:.1f} min reading MBP-1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
