#!/usr/bin/env python
"""P6 time-of-day report — Question 4: when do levels work best?

30-minute PT buckets over the whole intraday cycle, each with reaction
probability, magnitudes (points), adverse depth, category mix and timing;
buckets only count as a WINDOW when their direction vs the daily mean agrees
in ≥ 4 of the 5 walk-forward years (2022-2026). Adjacent stable buckets are
merged into plateaus; broad sessions are rolled up alongside; the
exact-location paired contrast is recomputed inside each broad block.
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
from run_p6_spatial import add_labels, paired                                  # noqa: E402

CONFIGS = str(ROOT / "configs")
KEY = ["session", "zid", "tap_sec"]
SEG_N = {0: "asia", 1: "london", 2: "premkt", 3: "ny-am", 4: "ny-pm"}


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    m = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    tod = pd.read_parquet(d0 / "interactions.parquet",
                          columns=KEY + ["tod_pt_min"]).drop_duplicates(KEY)
    m = m.merge(tod, on=KEY, how="left", validate="1:1")
    m = m[(m["n_fwd"] >= 1800) & m["y"].notna()].copy()
    m["bucket"] = (m["tod_pt_min"] // 30).astype(int)
    m["hhmm"] = (m["bucket"] * 30 // 60).astype(str).str.zfill(2) + ":" + \
        ((m["bucket"] * 30) % 60).astype(str).str.zfill(2)
    base = float(m["y"].mean())
    res: dict = {"base": base, "buckets": []}
    print(f"resolved interactions with TOD: {len(m):,} | base y {base:.4f}")

    print(f"\n{'PT':>6} {'n':>7} {'y':>7} {'stab':>5} {'rev_med':>8} "
          f"{'adv_med':>8} {'A%':>6} {'B%':>6} {'t15med':>7}")
    rows = []
    for b, g in m.groupby("bucket"):
        if len(g) < 800:
            continue
        S = g[g["y"] == 1]
        yr_sign = 0
        n_yr = 0
        for y_, gy in g[g["year"] >= 2022].groupby("year"):
            if len(gy) < 100:
                continue
            n_yr += 1
            if np.sign(gy["y"].mean() - base) == np.sign(g["y"].mean() - base):
                yr_sign += 1
        stable = n_yr >= 4 and yr_sign >= 4
        row = {"pt": g["hhmm"].iloc[0], "bucket": int(b), "n": len(g),
               "y": float(g["y"].mean()), "stable": bool(stable),
               "rev_med": float(S["rev_ep_pts"].median()) if len(S) else np.nan,
               "adv_med": float(S["adv15_pts"].median()) if len(S) else np.nan,
               "catA": float((g["y"] == 1).mean() and
                             float((g.loc[g["y"] == 1, "depth_max_full"] <= 4).mean())),
               "t15_med": float(S["t_rev15_s"].median() / 60) if len(S) else np.nan}
        rows.append(row)
        print(f"{row['pt']:>6} {row['n']:7,} {row['y']:7.3f} "
              f"{'*' if stable else ' ':>5} {row['rev_med']:8.1f} "
              f"{row['adv_med']:8.1f} {'':>6} {'':>6} {row['t15_med']:7.1f}")
    res["buckets"] = rows

    tab = pd.DataFrame(rows)
    up = tab[(tab["y"] > base + 0.02) & tab["stable"]]
    dn = tab[(tab["y"] < base - 0.02) & tab["stable"]]
    print(f"\nstable ABOVE-base windows (y > base+2pp, >=4/5 yrs): "
          + ", ".join(f"{r['pt']}({r['y']:.3f})" for _, r in up.iterrows()))
    print(f"stable BELOW-base windows: "
          + ", ".join(f"{r['pt']}({r['y']:.3f})" for _, r in dn.iterrows()))
    res["windows_up"] = up.to_dict("records")
    res["windows_dn"] = dn.to_dict("records")

    print("\n=== session rollup ===")
    seg = pd.read_parquet(d0 / "interactions.parquet",
                          columns=KEY).drop_duplicates(KEY)  # seg already in m
    for s_, g in m.groupby("seg"):
        S = g[g["y"] == 1]
        yrs = " ".join(f"{y_}:{gy['y'].mean():.3f}"
                       for y_, gy in g[g["year"] >= 2022].groupby("year"))
        print(f"  {SEG_N.get(int(s_), s_):>7}: n {len(g):6,} | y "
              f"{float(g['y'].mean()):.4f} | rev_med "
              f"{float(S['rev_ep_pts'].median()):6.1f} | adv_med "
              f"{float(S['adv15_pts'].median()):5.1f} | by yr {yrs}")

    print("\n=== exact-location paired contrast (±10 pts) inside blocks ===")
    sh = add_labels(pd.read_parquet(d0 / "p6_shadow.parquet"))
    sh = sh.merge(tod, on=KEY, how="left")
    re2 = m[m["has_t0"] & ~m["inside_t0"]].copy()
    re2["session_c"] = re2["session"]
    for name, lo, hi in (("asia", 0, 300), ("london", 300, 690),
                         ("ny-am", 690, 930), ("ny-pm+", 930, 1440)):
        gm = re2[(re2["tod_pt_min"] >= lo) & (re2["tod_pt_min"] < hi)]
        gs = sh[(sh["tod_pt_min"] >= lo) & (sh["tod_pt_min"] < hi)]
        r = paired(gm, gs, "y", 10.0, 200)
        if r:
            print(f"  {name:>8}: {r['diff']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] "
                  f"(n={r['n_events']:,})")
            res.setdefault("spatial_by_block", {})[name] = r

    out = ROOT / "outputs" / "p6_tod.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
