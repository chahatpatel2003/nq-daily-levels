#!/usr/bin/env python
"""P7 — pivot-definition robustness: do fractal pivots on conventional
timeframes name the same prices as the volatility-scaled zigzag universe?

For TF in {5m, 15m, 30m, 1H}: k=2 fractal highs (bar high is the max of a
5-bar window, confirmed 2 bars later). For each confirmed fractal resistance
pivot, distance to the nearest ACTIVE zigzag zone centre (formed within the
prior 30 sessions or earlier same session, any scale). And the reverse:
share of zigzag pivots with a fractal within 6t. High overlap ⇒ the
resistance effect is about the LOCATION, not the zigzag formula.
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.io.store import layer_dir                                        # noqa: E402

CONFIGS = str(ROOT / "configs")
TFS = {"5m": 5, "15m": 15, "30m": 30, "1H": 60}
K = 2


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    zn = pd.read_parquet(d0 / "zones.parquet",
                         columns=["session", "zid", "price_rf", "form_ord", "dir"])
    adj = pd.read_parquet(d0 / "roll_adjust.parquet").set_index("session")["adj_t"]
    ctx = pd.read_parquet(d0 / "context.parquet", columns=["session"])
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    ord_of = {s: i for i, s in enumerate(sorted(ctx["session"]))}
    zn = zn.sort_values("form_ord")
    zn_ord = zn["form_ord"].to_numpy()
    b1m = pd.read_parquet(d0 / "bars_1m.parquet",
                          columns=["session", "ts", "high_t", "low_t", "is_front"])
    b1m = b1m[b1m["is_front"]]

    stats = {tf: {"n": 0, "w6": 0, "w12": 0, "w24": 0} for tf in TFS}
    rev = {"n": 0, "hit": {tf: 0 for tf in TFS}}
    sessions = sorted(set(b1m["session"]) & set(ord_of) & set(adj.index))
    rng = np.random.default_rng(1)
    sample = set(rng.choice(len(sessions), 400, replace=False))
    for i, s in enumerate(sessions):
        if i not in sample:
            continue
        g = b1m[b1m["session"] == s].sort_values("ts")
        o = ord_of[s]
        a0, a1 = np.searchsorted(zn_ord, o - 30), np.searchsorted(zn_ord, o + 1)
        centers = np.sort(zn.iloc[a0:a1]["price_rf"].to_numpy("float64")
                          + float(adj.loc[s]))
        if not centers.size:
            continue
        hi1 = g["high_t"].to_numpy("float64")
        fr_all = []
        for tf, m in TFS.items():
            n = len(hi1) // m
            h = hi1[:n * m].reshape(n, m).max(axis=1)
            piv = [h[j] for j in range(K, n - K)
                   if h[j] == h[j - K:j + K + 1].max()
                   and h[j] > h[j - K:j].max()]
            for px in piv:
                stats[tf]["n"] += 1
                j = np.searchsorted(centers, px)
                dmin = min(abs(px - centers[max(0, j - 1)]),
                           abs(px - centers[min(len(centers) - 1, j)]))
                for w in (6, 12, 24):
                    if dmin <= w:
                        stats[tf][f"w{w}"] += 1
            fr_all.append((tf, np.sort(np.array(piv))))
        # reverse: today's-session zigzag HIGH pivots near any fractal pivot
        z_today = zn.iloc[a0:a1]
        z_today = z_today[(z_today["form_ord"] == o) & (z_today["dir"] == 1)]
        for px in z_today["price_rf"].to_numpy("float64") + float(adj.loc[s]):
            rev["n"] += 1
            for tf, arr in fr_all:
                if arr.size:
                    j = np.searchsorted(arr, px)
                    d = min(abs(px - arr[max(0, j - 1)]),
                            abs(px - arr[min(arr.size - 1, j)]))
                    if d <= 12:
                        rev["hit"][tf] += 1

    print(f"fractal resistance pivots vs active zigzag zone centres "
          f"(400 sampled sessions):")
    print(f"{'TF':>5} {'n_pivots':>9} {'<=1.5pt':>9} {'<=3pt':>7} {'<=6pt':>7}")
    for tf in TFS:
        st = stats[tf]
        if st["n"]:
            print(f"{tf:>5} {st['n']:9,} {st['w6']/st['n']:9.3f} "
                  f"{st['w12']/st['n']:7.3f} {st['w24']/st['n']:7.3f}")
    print(f"\nreverse: zigzag HIGH pivots with a same-session fractal within "
          f"3 pts (n={rev['n']:,}):")
    for tf in TFS:
        print(f"  {tf:>5}: {rev['hit'][tf]/max(1,rev['n']):.3f}")
    print(f"({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
