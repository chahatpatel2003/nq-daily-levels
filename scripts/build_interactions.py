#!/usr/bin/env python
"""Phase-4 corpus build: every zone, every tap, every outcome, 2021-2026.

Chronological single pass. The zone registry persists across sessions in
roll-free prices; each session contributes new confirmed pivots, taps of
anything active, causal features, and frozen-taxonomy outcomes. Trailing
structures (5-session volume profile, 20-session extremes) are maintained
strictly from prior sessions.

Outputs on T9: interactions.parquet (one row per tap) and zones.parquet
(one row per zone at formation, with final lifecycle state).
"""
from __future__ import annotations

import glob
import sys
import time as _time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.features.interactions import build_session_data, process_session  # noqa: E402
from nqlevels.features.zones import epoch_s, make_zones                         # noqa: E402
from nqlevels.io.raw import assert_not_raw                                      # noqa: E402
from nqlevels.io.store import layer_dir, read_layer                             # noqa: E402

CONFIGS = str(ROOT / "configs")
B1S_DIR = "/Volumes/T9/nq_levels/bars_1s"
NEIGH_ATR = 0.05


def seg_codes(pt_min: np.ndarray) -> np.ndarray:
    return np.select(
        [(pt_min >= 390) & (pt_min < 540), (pt_min >= 540) & (pt_min < 780),
         (pt_min >= 300) & (pt_min < 390), (pt_min >= 0) & (pt_min < 300)],
        [3, 4, 2, 1], default=0).astype("int8")


def make_vol_feats(prof_deque, adj_now: float, atr: float):
    """Closure over the trailing 5-session profile in roll-free ticks."""
    if not prof_deque:
        return lambda zlo, zhi: {"volz_share": np.nan, "volz_rel": np.nan,
                                 "vol_cliff": np.nan}
    px = np.concatenate([p for p, _ in prof_deque])
    vv = np.concatenate([v for _, v in prof_deque])
    o = np.argsort(px, kind="mergesort")
    px, vv = px[o], vv[o]
    cum = np.concatenate(([0.0], np.cumsum(vv)))
    neigh = NEIGH_ATR * atr

    def f(zlo_cur, zhi_cur):
        zl, zh = zlo_cur - adj_now, zhi_cur - adj_now
        def s(a, b):
            return float(cum[np.searchsorted(px, b, "right")]
                         - cum[np.searchsorted(px, a, "left")])
        band = s(zl, zh)
        above = s(zh, zh + neigh)
        below = s(zl - neigh, zl)
        tot = band + above + below
        w = max(1.0, zh - zl)
        return {"volz_share": band / tot if tot > 0 else np.nan,
                "volz_rel": (band / w) / ((above + below) / (2 * neigh))
                if (above + below) > 0 else np.nan,
                "vol_cliff": (above - below) / (above + below)
                if (above + below) > 0 else np.nan}
    return f


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    ctx = pd.read_parquet(d0 / "context.parquet",
                          columns=["session", "atr14_t", "atr_pctile", "dow"])
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    ctxi = ctx.set_index("session")
    adj = pd.read_parquet(d0 / "roll_adjust.parquet").set_index("session")
    b1m_all = pd.read_parquet(d0 / "bars_1m.parquet")
    b1m_all["ts"] = pd.DatetimeIndex(b1m_all["ts"]).tz_convert("UTC")
    b1m_by = {s: g.sort_values("ts").reset_index(drop=True)
              for s, g in b1m_all[b1m_all["is_front"]].groupby("session", sort=False)}
    del b1m_all
    prof_all = read_layer("profiles", CONFIGS)
    prof_all = prof_all[prof_all["is_front"]]
    prof_all = prof_all.groupby(["session", "price_t"], as_index=False)["volume"].sum()
    prof_by = {s: g for s, g in prof_all.groupby("session", sort=False)}
    del prof_all

    ord_of = {s: i for i, s in enumerate(sorted(ctxi.index))}
    registry: dict = {}
    rows_all, zones_all = [], []
    prof_deque, ext_deque = [], []
    zid_counter = 0
    cur_chain = None
    files = sorted(glob.glob(f"{B1S_DIR}/[!._]*.parquet"))
    n_done = 0
    for fpath in files:
        b1s_m = pd.read_parquet(fpath)
        b1s_m = b1s_m[b1s_m["is_front"]]
        by_sess = {s: g for s, g in b1s_m.groupby("session", sort=False)}
        for s in sorted(by_sess):
            if s not in ctxi.index or s not in adj.index or s not in b1m_by:
                continue
            atr = float(ctxi.loc[s, "atr14_t"])
            if not np.isfinite(atr) or atr <= 0:
                continue
            adj_s = float(adj.loc[s, "adj_t"])
            chain = int(adj.loc[s, "chain"])
            if cur_chain is not None and chain != cur_chain:
                registry.clear()
            cur_chain = chain
            sess_ord = ord_of[s]
            b1s = by_sess[s].sort_values("ts").reset_index(drop=True)
            b1m = b1m_by[s]
            if len(b1s) < 3600 or len(b1m) < 120:
                continue
            sd = build_session_data(b1s, b1m)
            m_pt = ((epoch_s(b1m["ts"]) - 8 * 3600) % 86400) // 60
            segm = seg_codes(m_pt)
            new = make_zones(b1m, atr, adj_s, sess_ord, segm, zid_counter)
            zid_counter += len(new)
            hi20 = max(x[0] for x in ext_deque) if ext_deque else np.nan
            lo20 = min(x[1] for x in ext_deque) if ext_deque else np.nan
            vf = make_vol_feats(prof_deque, adj_s, atr)
            rows = process_session(registry, new, sd, atr, adj_s, chain,
                                   sess_ord, int(ctxi.loc[s, "dow"]),
                                   float(ctxi.loc[s, "atr_pctile"]), vf,
                                   (hi20, lo20))
            for r in rows:
                r["session"] = s
            rows_all.extend(rows)
            for z in new:
                zones_all.append({"session": s, **{k: v for k, v in
                                  asdict(z).items() if k != "retired"}})
            # trailing structures update (after processing -> strictly prior)
            p = prof_by.get(s)
            if p is not None and len(p):
                prof_deque.append(((p["price_t"].to_numpy("float64") - adj_s),
                                   p["volume"].to_numpy("float64")))
                if len(prof_deque) > 5:
                    prof_deque.pop(0)
            ext_deque.append((float(b1m["high_t"].max()) - adj_s,
                              float(b1m["low_t"].min()) - adj_s))
            if len(ext_deque) > 20:
                ext_deque.pop(0)
            n_done += 1
            if n_done % 100 == 0:
                print(f"  {s}: {len(rows_all):,} interactions, "
                      f"{zid_counter:,} zones, {len(registry)} active, "
                      f"{_time.time()-t0:.0f}s", flush=True)
        del b1s_m, by_sess

    inter = pd.DataFrame(rows_all)
    for c in inter.columns:
        if inter[c].dtype == object and c != "session":
            inter[c] = pd.to_numeric(inter[c], errors="coerce")
    dst = assert_not_raw(d0 / "interactions.parquet")
    inter.to_parquet(dst, index=False, compression="zstd")
    zf = pd.DataFrame(zones_all)
    zf.to_parquet(assert_not_raw(d0 / "zones.parquet"), index=False,
                  compression="zstd")
    per = inter.groupby("session").size()
    print(f"\ninteractions: {len(inter):,} over {inter['session'].nunique()} "
          f"sessions (median {per.median():.0f}/session) | zones {len(zf):,} | "
          f"{dst.stat().st_size/1e6:.0f} MB | {(_time.time()-t0)/60:.1f} min")
    print(f"tap_no dist: {inter['tap_no'].value_counts().sort_index().to_dict()}")
    print(f"resolved (>=1800s fwd): "
          f"{float((inter['o_n_fwd'] >= 1800).mean()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
