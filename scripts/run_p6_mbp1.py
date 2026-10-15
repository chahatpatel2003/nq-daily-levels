#!/usr/bin/env python
"""P6 bounded MBP-1 spatial test (Question 8) — does order-book history add
LOCAL price information at the tap, beyond what the tape already gives?

Registered scope (RESEARCH_PLAN §17.1): a sampled-session screen, not a
corpus build. 30 random sessions 2022-01..2025-06 (MBP-1 coverage), the
deduplicated taps of those sessions, trailing-60-minute book events within
±15 pts of each tap price; features = per-tick book quantities aggregated in
the zone band (±1.5 pt) vs the ring (±5..10 pt), as prominence ratios.
Outcomes: strict tidy rejection (where the exact-location effect lives) and
primary. A coarse screen by design (~1-2 k taps): it can detect the ~8 pp
effects that would matter, not 2 pp ones — stated as such.
"""
from __future__ import annotations

import json
import sys
import time as _time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nqlevels.io.raw import read_mbp1                                          # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402
from run_e11_mbp1 import book_features                                         # noqa: E402
from run_p6_spatial import add_labels                                          # noqa: E402

CONFIGS = str(ROOT / "configs")
KEY = ["session", "zid", "tap_sec"]
N_SESS = 30
BAND_T = 6
RING_T = (20, 40)
FEATS = ["bk_secs", "bk_twsize", "bk_adds", "bk_cancels", "bk_tradevol",
         "bk_replenish", "bk_absorb"]


def auc(y, s):
    y = np.asarray(y, "float64")
    s = np.asarray(s, "float64")
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size < 150 or y.min() == y.max():
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    re = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    it = pd.read_parquet(d0 / "interactions.parquet",
                         columns=KEY + ["price_t", "tap_ts"])
    re = re.merge(it, on=KEY, how="left", validate="1:1")
    re = re[(re["n_fwd"] >= 1800) & re["y"].notna()]
    ctx = pd.read_parquet(d0 / "context.parquet", columns=["session", "symbol"])
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    sym = ctx.set_index("session")["symbol"]

    pool = [s for s in re["session"].unique()
            if pd.Timestamp(str(s)).year >= 2022
            and s <= pd.Timestamp("2025-06-15").date()]
    rng = np.random.default_rng(42)
    sample = sorted(rng.choice(pool, N_SESS, replace=False))
    rows = []
    for k, s in enumerate(sample):
        g = re[re["session"] == s]
        if not len(g):
            continue
        t_lo = pd.Timestamp(g["tap_ts"].min() - 3700, unit="s", tz="UTC")
        t_hi = pd.Timestamp(g["tap_ts"].max(), unit="s", tz="UTC")
        try:
            q = read_mbp1(t_lo.to_pydatetime(), t_hi.to_pydatetime(), CONFIGS,
                          symbol=str(sym.loc[s]))
        except Exception as e:                                        # noqa: BLE001
            print(f"  {s}: read failed ({e})")
            continue
        if q.empty:
            continue
        q = q.reset_index()
        ts_col = "ts_recv" if "ts_recv" in q.columns else q.columns[0]
        if "ts_event" not in q.columns:
            q = q.rename(columns={ts_col: "ts_event"})
        q["ts_event"] = pd.DatetimeIndex(q["ts_event"]).tz_convert("UTC")
        te = q["ts_event"].astype("int64").to_numpy() // 10**9
        for r in g.itertuples():
            a = np.searchsorted(te, r.tap_ts - 3600)
            b = np.searchsorted(te, r.tap_ts)
            if b - a < 1000:
                continue
            w = q.iloc[a:b]
            X = float(r.price_t)
            w = w[(np.abs(np.rint(w["price"].to_numpy("float64") / 0.25) - X)
                   <= 60) | w["price"].isna()]
            f = book_features(w, X)
            if f.empty:
                continue
            off = f["off_t"].to_numpy()
            row = {"session": s, "zid": int(r.zid), "tap_sec": int(r.tap_sec),
                   "y": r.y, "y_strict": r.y_strict,
                   "side_above": r.side_above}
            for c in FEATS:
                v = f[c].to_numpy("float64")
                band = float(v[np.abs(off) <= BAND_T].sum()) / (2 * BAND_T + 1)
                ring_vals = []
                for rr in RING_T:
                    m = (np.abs(off - rr) <= BAND_T) | (np.abs(off + rr) <= BAND_T)
                    ring_vals.append(float(v[m].sum()) / max(1, int(m.sum())))
                ring = np.mean(ring_vals)
                row[f"{c}_prom"] = band / (ring + 1e-9)
            rows.append(row)
        print(f"  [{k+1}/{len(sample)}] {s}: {len(rows):,} taps, "
              f"{_time.time()-t0:.0f}s", flush=True)

    d = pd.DataFrame(rows)
    res = {"n_taps": len(d), "n_sessions": int(d["session"].nunique())}
    print(f"\nMBP-1 spatial screen: {len(d):,} taps, "
          f"{d['session'].nunique()} sessions")
    print(f"{'feature':>18} {'AUC_strict':>11} {'AUC_primary':>12} "
          f"{'AUC_strict@resist':>18}")
    for c in FEATS:
        col = f"{c}_prom"
        a1 = auc(d["y_strict"], d[col])
        a2 = auc(d["y"], d[col])
        d0_ = d[d["side_above"] == 0]
        a3 = auc(d0_["y_strict"], d0_[col])
        res[col] = {"auc_strict": a1, "auc_primary": a2, "auc_strict_res": a3}
        print(f"{col:>18} {a1:11.4f} {a2:12.4f} {a3:18.4f}")

    out = ROOT / "outputs" / "p6_mbp1.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
