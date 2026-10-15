#!/usr/bin/env python
"""Phase-10 trade simulator corpus.

For every frozen-confirmation event (R4v10, or gated 5-min stall) and every
registered entry variant: the causal entry fill, then first-passage times to
every registered stop level (above) and target level (below) on the
1-second path, plus time-exit checkpoint prices and the session-end price.
Same-second stop/target ambiguity is resolved later as stop-first; limit
entries require trading THROUGH the limit by one tick within 900 s.

Output p10_trades.parquet: one row per (interaction, conf source, entry
variant); any (stop, target) strategy config is then a pure table lookup.
"""
from __future__ import annotations

import glob
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nqlevels.features.interactions import build_session_data                  # noqa: E402
from nqlevels.io.raw import assert_not_raw                                     # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402
from run_p6_spatial import add_labels                                          # noqa: E402

CONFIGS = str(ROOT / "configs")
B1S_DIR = "/Volumes/T9/nq_levels/bars_1s"
KEY = ["session", "zid", "tap_sec"]
S1_PTS = (5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0)
S2_B = (0.0, 5.0, 10.0, 15.0, 20.0)
S3_B = (1.0, 2.0, 4.0, 6.0, 8.0)
S4_ATR = (0.02, 0.04)
R_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
T_FIXED = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0)
T_ALL = tuple(sorted(set(T_FIXED)
                     | {round(s * r, 2) for s in S1_PTS for r in R_GRID}))
CHECKS = (300, 600, 900, 1800, 3600)
LIMIT_TIMEOUT = 900
E2_P = (2.0, 4.0, 6.0, 8.0, 10.0)
E3_R = (25, 33, 50, 66)


def main() -> int:
    t0c = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    tl = pd.read_parquet(d0 / "p9_timeline.parquet")
    re = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    it = pd.read_parquet(d0 / "interactions.parquet",
                         columns=KEY + ["zhi_t", "zlo_t", "tap_no"]).drop_duplicates(KEY)
    m = tl.merge(re[KEY + ["y_strict", "y", "rev_ep_pts", "seg", "atr_t",
                           "side_above", "n_fwd"]], on=KEY, how="inner")
    m = m.merge(it, on=KEY, how="left", validate="1:1")
    m = m[(m["side_above"] == 0) & (m["n_fwd"] >= 1800)
          & m["y_strict"].notna()]
    m = m.rename(columns={"r4v10.0_t": "f_r4", "r4v10.0_done": "d_r4",
                          "r4v10.0_adv": "a_r4", "r2t300p5.0_t": "f_st",
                          "r2t300p5.0_done": "d_st", "r2t300p5.0_adv": "a_st",
                          "r6T120_t": "f_dq"})
    m_by = {s: g for s, g in m.groupby("session", sort=False)}
    print(f"interactions: {len(m):,} | target levels: {len(T_ALL)}")

    b1m_all = pd.read_parquet(d0 / "bars_1m.parquet")
    b1m_all["ts"] = pd.DatetimeIndex(b1m_all["ts"]).tz_convert("UTC")
    b1m_by = {s: g.sort_values("ts").reset_index(drop=True)
              for s, g in b1m_all[b1m_all["is_front"]].groupby("session", sort=False)}
    del b1m_all

    rows = []
    n_done = 0
    for fpath in sorted(glob.glob(f"{B1S_DIR}/[!._]*.parquet")):
        b1s_m = pd.read_parquet(fpath)
        b1s_m = b1s_m[b1s_m["is_front"]]
        for s, b1s in b1s_m.groupby("session", sort=True):
            g = m_by.get(s)
            if g is None or s not in b1m_by:
                continue
            sd = build_session_data(b1s.sort_values("ts").reset_index(drop=True),
                                    b1m_by[s])
            n = sd.cl.size
            for r in g.itertuples():
                t = int(r.tap_sec)
                zlo, zhi = float(r.zlo_t), float(r.zhi_t)
                confs = []
                if np.isfinite(r.f_r4):
                    confs.append(("r4", int(r.f_r4), float(r.d_r4),
                                  float(r.a_r4)))
                if np.isfinite(r.f_st) and not (np.isfinite(r.f_dq)
                                                and r.f_dq < r.f_st):
                    confs.append(("st", int(r.f_st), float(r.d_st),
                                  float(r.a_st)))
                for src, fire, done_pts, adv_pts in confs:
                    f = t + fire
                    if f + 5 >= n:
                        continue
                    ref_px = float(sd.cl[f + 1])
                    ext_low = zlo - done_pts * 4.0
                    ext_hi = zlo + adv_pts * 4.0
                    entries = [("E1", f + 1, ref_px, True),
                               ("E1d", min(f + 3, n - 2), float(sd.cl[min(f + 3, n - 2)]), True)]
                    for p_ in E2_P:
                        entries.append((f"E2p{p_}", None, ref_px + 4 * p_, False))
                    for rr_ in E3_R:
                        lim = ext_low + rr_ / 100.0 * (done_pts * 4.0)
                        entries.append((f"E3r{rr_}", None, lim, False))
                    entries.append(("E4", None, zlo, False))
                    for ename, e_idx, e_px, is_mkt in entries:
                        if not is_mkt:
                            w = sd.hi[f + 1: min(n, f + 1 + LIMIT_TIMEOUT)]
                            hitj = np.flatnonzero(w >= e_px + 1.0)
                            if not hitj.size:
                                rows.append({"session": s, "zid": int(r.zid),
                                             "tap_sec": t, "src": src,
                                             "entry": ename, "filled": False})
                                continue
                            e_idx = f + 1 + int(hitj[0])
                        if e_idx >= n - 5:
                            continue
                        hi_f = sd.hi[e_idx:]
                        lo_f = sd.lo[e_idx:]
                        adv_run = np.maximum.accumulate(hi_f) - e_px
                        fav_run = e_px - np.minimum.accumulate(lo_f)
                        row = {"session": s, "zid": int(r.zid), "tap_sec": t,
                               "src": src, "entry": ename, "filled": True,
                               "fire_s": fire, "e_delay": int(e_idx - f),
                               "e_px": e_px, "zlo": zlo, "zhi": zhi,
                               "ext_hi": ext_hi, "atr_t": float(r.atr_t),
                               "tap_no": int(r.tap_no), "seg": int(r.seg),
                               "done": done_pts, "adv": adv_pts,
                               "end_px": float(sd.cl[n - 1]),
                               "n_left": int(n - e_idx)}
                        for i, sp in enumerate(S1_PTS):
                            j = int(np.searchsorted(adv_run, sp * 4.0))
                            row[f"ts1_{i}"] = j if j < adv_run.size else -1
                        for i, b in enumerate(S2_B):
                            lvl = zhi + 4 * b - e_px
                            row[f"s2v_{i}"] = lvl / 4.0
                            j = int(np.searchsorted(adv_run, max(lvl, 1.0)))
                            row[f"ts2_{i}"] = j if j < adv_run.size else -1
                        for i, b in enumerate(S3_B):
                            lvl = ext_hi + 4 * b - e_px
                            row[f"s3v_{i}"] = lvl / 4.0
                            j = int(np.searchsorted(adv_run, max(lvl, 1.0)))
                            row[f"ts3_{i}"] = j if j < adv_run.size else -1
                        for i, a_ in enumerate(S4_ATR):
                            lvl = a_ * float(r.atr_t)
                            row[f"s4v_{i}"] = lvl / 4.0
                            j = int(np.searchsorted(adv_run, lvl))
                            row[f"ts4_{i}"] = j if j < adv_run.size else -1
                        for i, tp in enumerate(T_ALL):
                            j = int(np.searchsorted(fav_run, tp * 4.0))
                            row[f"tt_{i}"] = j if j < fav_run.size else -1
                        for c_ in CHECKS:
                            row[f"cl_{c_}"] = float(sd.cl[min(n - 1, e_idx + c_)])
                        rows.append(row)
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {s}: {len(rows):,} rows, {_time.time()-t0c:.0f}s",
                      flush=True)
        del b1s_m

    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == "float64" and c not in ("e_px", "zlo", "zhi",
                                                  "ext_hi", "end_px"):
            df[c] = df[c].astype("float32")
    df.to_parquet(assert_not_raw(d0 / "p10_trades.parquet"), index=False,
                  compression="zstd")
    print(f"\np10_trades: {len(df):,} rows x {df.shape[1]} cols | filled "
          f"{float(df['filled'].mean()):.3f} | {(_time.time()-t0c)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
