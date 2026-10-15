#!/usr/bin/env python
"""P13 execution variants for tap<=3 R4v10 fires, recomputed exactly from
1s paths. For each fire: entry fills at fire+3 (frozen E1d reference,
must reproduce the ledger), fire+4/+5/+8 (Part-14 delay stress), and the
LIVE-LEGAL timing max(fire+3, tap+62)/(tap+63): the p_both60 gate uses
cp60 features (data through tap+60s), so a live engine can act no earlier
than tap+61; fills modelled at the close of tap+62 (base) and tap+63
(conservative). Stop 12.5 / target 30 from each variant's own fill; ties
stop-first; session-close exit otherwise; base cost 1.0 applied later.
Output: /Volumes/T9/nq_levels/daily/p13_exec.parquet
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
from nqlevels.io.store import layer_dir                                        # noqa: E402

CONFIGS = str(ROOT / "configs")
B1S_DIR = "/Volumes/T9/nq_levels/bars_1s"
KEY = ["session", "zid", "tap_sec"]
STOP_T, TGT_T = 50.0, 120.0            # ticks


def passage(hi, lo, e_px):
    """First-passage times (rel to entry idx) for stop above / target
    below a short from e_px; -1 if never."""
    up = np.flatnonzero(hi >= e_px + STOP_T)
    dn = np.flatnonzero(lo <= e_px - TGT_T)
    return (int(up[0]) if up.size else -1,
            int(dn[0]) if dn.size else -1)


def main() -> int:
    t0c = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    t = pd.read_parquet(d0 / "p10_trades.parquet",
                        columns=KEY + ["src", "entry", "filled", "fire_s",
                                       "tap_no", "seg"])
    sc = pd.read_parquet(d0 / "p10_scores.parquet")
    t = t.merge(sc[KEY + ["p_both60"]].rename(
        columns={"p_both60": "p60"}), on=KEY, how="left")
    m = t[(t["src"] == "r4") & (t["entry"] == "E1d") & t["filled"]
          & (t["tap_no"] <= 3)].drop_duplicates(KEY)
    m["year"] = pd.to_datetime(m["session"].astype(str)).dt.year
    m = m[m["year"] >= 2022]
    print(f"tap<=3 r4 fires to recompute: {len(m):,}")
    m_by = {s: g for s, g in m.groupby("session", sort=False)}

    b1m_all = pd.read_parquet(d0 / "bars_1m.parquet")
    b1m_all["ts"] = pd.DatetimeIndex(b1m_all["ts"]).tz_convert("UTC")
    b1m_by = {s: g.sort_values("ts").reset_index(drop=True)
              for s, g in b1m_all[b1m_all["is_front"]]
              .groupby("session", sort=False)}
    del b1m_all

    VAR = ("d3", "d4", "d5", "d8", "L62", "L63")
    rows = []
    n_done = 0
    for fpath in sorted(glob.glob(f"{B1S_DIR}/[!._]*.parquet")):
        b1s_m = pd.read_parquet(fpath)
        b1s_m = b1s_m[b1s_m["is_front"]]
        for s, b1s in b1s_m.groupby("session", sort=True):
            g = m_by.get(s)
            if g is None or s not in b1m_by:
                continue
            sd = build_session_data(
                b1s.sort_values("ts").reset_index(drop=True), b1m_by[s])
            n = sd.cl.size
            for r in g.itertuples():
                tsec = int(r.tap_sec)
                f = tsec + int(r.fire_s)
                row = {"session": s, "zid": int(r.zid), "tap_sec": tsec,
                       "tap_no": int(r.tap_no), "seg": int(r.seg),
                       "p60": float(r.p60), "fire_s": int(r.fire_s)}
                for v, e_idx in (("d3", f + 3), ("d4", f + 4),
                                 ("d5", f + 5), ("d8", f + 8),
                                 ("L62", max(f + 3, tsec + 62)),
                                 ("L63", max(f + 3, tsec + 63))):
                    if e_idx >= n - 5:
                        row[f"{v}_epx"] = np.nan
                        continue
                    e_px = float(sd.cl[e_idx])
                    ts_, tt_ = passage(sd.hi[e_idx:], sd.lo[e_idx:], e_px)
                    row[f"{v}_epx"] = e_px
                    row[f"{v}_eidx"] = e_idx
                    row[f"{v}_ts"] = ts_
                    row[f"{v}_tt"] = tt_
                    row[f"{v}_endpx"] = float(sd.cl[n - 1])
                rows.append(row)
            n_done += 1
        print(f"  {Path(fpath).name}: cum sessions {n_done} "
              f"({(_time.time()-t0c)/60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    out = Path("/Volumes/T9/nq_levels/daily/p13_exec.parquet")
    df.to_parquet(out)
    print(f"wrote {out} ({len(df):,} rows, "
          f"{(_time.time()-t0c)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
