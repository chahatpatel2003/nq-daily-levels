#!/usr/bin/env python
"""Phase-6 measurement corpus: enriched outcomes for every deduped real
interaction and its 8 shadow anchors (X ± 2.5/5/10/20 pts, same width).

Per event: common decision time t0 = tap − 900 s; every anchor scored from
its OWN first touch after t0 with the same walker; anchors already inside
their band at t0 are marked ineligible; shadows within 1.5 pts of another
active zone centre are flagged (they are not 'empty' prices).

Outputs: p6_real.parquet, p6_shadow.parquet.
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
from nqlevels.labels.enrich import enriched_metrics, first_touch_after         # noqa: E402
from run_p4_strongweak import load_frame                                       # noqa: E402

CONFIGS = str(ROOT / "configs")
B1S_DIR = "/Volumes/T9/nq_levels/bars_1s"
OFFSETS_T = (-80, -40, -20, -10, 10, 20, 40, 80)
T0_S = 900
MAX_AGE = 30


def main() -> int:
    t0c = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    f = load_frame()
    keep_cols = ["session", "zid", "tap_sec", "zlo_t", "zhi_t", "hw_t",
                 "atr_t", "app_side_above", "seg", "year", "price_t",
                 "y", "y_strict", "tap_no", "age_sessions"]
    f = f[keep_cols].copy()
    by_sess = {s: g for s, g in f.groupby("session", sort=False)}
    print(f"deduped interactions to measure: {len(f):,} over {len(by_sess)} sessions")

    ctx = pd.read_parquet(d0 / "context.parquet", columns=["session", "atr14_t"])
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    adj = pd.read_parquet(d0 / "roll_adjust.parquet").set_index("session")
    zn = pd.read_parquet(d0 / "zones.parquet",
                         columns=["session", "zid", "price_rf", "form_ord"])
    ord_of = {s: i for i, s in enumerate(sorted(ctx["session"]))}
    zn = zn.sort_values("form_ord")
    zn_ord = zn["form_ord"].to_numpy()
    b1m_all = pd.read_parquet(d0 / "bars_1m.parquet")
    b1m_all["ts"] = pd.DatetimeIndex(b1m_all["ts"]).tz_convert("UTC")
    b1m_by = {s: g.sort_values("ts").reset_index(drop=True)
              for s, g in b1m_all[b1m_all["is_front"]].groupby("session", sort=False)}
    del b1m_all

    real_rows, sh_rows = [], []
    n_done = 0
    for fpath in sorted(glob.glob(f"{B1S_DIR}/[!._]*.parquet")):
        b1s_m = pd.read_parquet(fpath)
        b1s_m = b1s_m[b1s_m["is_front"]]
        for s, b1s in b1s_m.groupby("session", sort=True):
            g = by_sess.get(s)
            if g is None or s not in b1m_by or s not in adj.index:
                continue
            sd = build_session_data(b1s.sort_values("ts").reset_index(drop=True),
                                    b1m_by[s])
            p = sd.path
            adj_s = float(adj.loc[s, "adj_t"])
            o = ord_of[s]
            a0, a1 = np.searchsorted(zn_ord, o - MAX_AGE), np.searchsorted(zn_ord, o + 1)
            centers = np.sort(zn.iloc[a0:a1]["price_rf"].to_numpy("float64") + adj_s)

            for r in g.itertuples():
                key = {"session": s, "zid": int(r.zid), "tap_sec": int(r.tap_sec)}
                m = enriched_metrics(p, float(r.zlo_t), float(r.zhi_t),
                                     int(r.tap_sec))
                t0 = int(r.tap_sec) - T0_S
                p_t0 = float(sd.cl[t0 - 1]) if t0 >= 60 else np.nan
                real_rows.append({**key, **m, "atr_t": r.atr_t, "hw_t": r.hw_t,
                                  "seg": r.seg, "year": r.year,
                                  "side_above": r.app_side_above,
                                  "tap_no_r": r.tap_no,
                                  "has_t0": t0 >= 60,
                                  "inside_t0": bool(t0 >= 60 and
                                                    r.zlo_t <= p_t0 <= r.zhi_t),
                                  "dist_t0_t": abs(float(r.price_t) - p_t0)
                                  if t0 >= 60 else np.nan})
                if t0 < 60:
                    continue
                for off in OFFSETS_T:
                    zlo, zhi = float(r.zlo_t) + off, float(r.zhi_t) + off
                    row = {**key, "offset_t": off, "atr_t": r.atr_t,
                           "hw_t": r.hw_t, "seg": r.seg, "year": r.year,
                           "side_above": r.app_side_above,
                           "dist_t0_t": abs(float(r.price_t) + off - p_t0)}
                    ctr = float(r.price_t) + off
                    j = np.searchsorted(centers, ctr)
                    near = min([abs(ctr - centers[k])
                                for k in (max(0, j - 1), min(len(centers) - 1, j))]) \
                        if len(centers) else np.inf
                    row["overlaps_zone"] = bool(near <= 6.0)
                    if zlo <= p_t0 <= zhi:
                        row["elig"] = False
                        row["touched"] = False
                        sh_rows.append(row)
                        continue
                    row["elig"] = True
                    ft = first_touch_after(p, zlo, zhi, t0)
                    if ft is None:
                        row["touched"] = False
                        sh_rows.append(row)
                        continue
                    ms = enriched_metrics(p, zlo, zhi, ft)
                    row.update(ms)
                    row["touched"] = True
                    row["t_to_touch_s"] = ft - t0
                    sh_rows.append(row)
            n_done += 1
            if n_done % 150 == 0:
                print(f"  {s}: {len(real_rows):,} real, {len(sh_rows):,} shadow, "
                      f"{_time.time()-t0c:.0f}s", flush=True)
        del b1s_m

    def finish(rows, name):
        df = pd.DataFrame(rows)
        for c in df.columns:
            if df[c].dtype == object and c not in ("session", "side"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.to_parquet(assert_not_raw(d0 / name), index=False, compression="zstd")
        return df

    re = finish(real_rows, "p6_real.parquet")
    sh = finish(sh_rows, "p6_shadow.parquet")
    print(f"\nreal {len(re):,} | shadow {len(sh):,} "
          f"(touched {float(sh['touched'].mean()):.3f}, elig "
          f"{float(sh['elig'].mean()):.3f}, overlap {float(sh['overlaps_zone'].mean()):.3f}) "
          f"| {(_time.time()-t0c)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
