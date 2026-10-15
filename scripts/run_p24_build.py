#!/usr/bin/env python
"""P24 builder: direct near-edge entries at frozen resistance child
zones. One row per (tap, fill-variant) with first-crossing times for
the FIXED SL/TP grid, MAE/MFE, and tags. Conventions registered a
priori: limit at L; B1 fill = first trade at L (the tap second); B2
fill = first trade >= L+0.5pt within 900 s; crossings measured from
each variant's own fill second; stop-first ties; cost 1.0pt RT applied
at evaluation."""
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

SEAL = "/Volumes/T9/nq_levels/sealed_holdout_1/derived/daily"
DEV = "/Volumes/T9/nq_levels/daily"
B1S_DEV = "/Volumes/T9/nq_levels/bars_1s"
B1S_SEAL = "/Volumes/T9/nq_levels/sealed_holdout_1/derived/bars_1s"
OUT = str(ROOT / "outputs" / "p24_direct_entry_trades.parquet")
STOPS = (10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0)
TGTS = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0)


def load_taps():
    frames = []
    for daily, lo, hi in ((DEV, "2021-01-01", "2026-06-17"),
                          (SEAL, "2026-06-18", "2026-08-14")):
        it = pd.read_parquet(f"{daily}/interactions.parquet",
                             columns=["session", "zid", "tap_sec", "tap_no",
                                      "app_side_above", "zlo_t", "zhi_t",
                                      "seg"])
        it["session"] = it["session"].astype(str)
        it = it[(it["session"] >= lo) & (it["session"] <= hi)
                & (it["app_side_above"] == 0)]
        zs = pd.read_parquet(f"{daily}/zones.parquet",
                             columns=["zid", "scale"])
        if daily != DEV:
            it["zid"] += 10_000_000
            zs = zs.copy()
            zs["zid"] += 10_000_000
        it = it.merge(zs.drop_duplicates("zid"), on="zid", how="left")
        frames.append(it)
    return pd.concat(frames, ignore_index=True)


def first_ge(a, thr):
    j = np.flatnonzero(a >= thr)
    return int(j[0]) if j.size else -1


def main() -> int:
    t0c = _time.time()
    taps = load_taps()
    by = {s: g for s, g in taps.groupby("session", sort=False)}
    d0 = layer_dir("daily", str(ROOT / "configs"))
    b1m_by = {}
    for src in (str(d0 / "bars_1m.parquet"), f"{SEAL}/bars_1m.parquet"):
        b = pd.read_parquet(src)
        b["ts"] = pd.DatetimeIndex(b["ts"]).tz_convert("UTC")
        for s, g in b[b["is_front"]].groupby("session"):
            ss = str(s)
            if src.startswith(str(d0)) and ss > "2026-06-17":
                continue
            if not src.startswith(str(d0)) and ss < "2026-06-18":
                continue
            b1m_by[ss] = g.sort_values("ts").reset_index(drop=True)
        del b

    files = sorted(glob.glob(f"{B1S_DEV}/[!._]*.parquet")) + \
        sorted(glob.glob(f"{B1S_SEAL}/2026-0[678]*.parquet"))
    rows = []
    n_done = 0
    for fpath in files:
        is_seal = "sealed" in fpath
        b1s_m = pd.read_parquet(fpath)
        b1s_m = b1s_m[b1s_m["is_front"]]
        for s, b1s in b1s_m.groupby("session", sort=True):
            ss = str(s)
            if (is_seal and ss < "2026-06-18") or \
               (not is_seal and ss > "2026-06-17"):
                continue
            g = by.get(ss)
            if g is None or ss not in b1m_by:
                continue
            sd = build_session_data(
                b1s.sort_values("ts").reset_index(drop=True), b1m_by[ss])
            n = sd.cl.size
            cl, hi, lo = sd.cl, sd.hi, sd.lo
            for r in g.sort_values("tap_sec").itertuples():
                t = int(r.tap_sec)
                L, U = float(r.zlo_t), float(r.zhi_t)
                if t >= n - 10:
                    continue
                base = dict(session=ss, zid=int(r.zid), tap_sec=t,
                            tap_no=int(r.tap_no), seg=int(r.seg),
                            scale=int(r.scale) if np.isfinite(r.scale)
                            else -1, L=L, U=U, width=(U - L) / 4.0)
                j2 = first_ge(hi[t:min(n, t + 900)], L + 2.0)
                fills = [("B1", t), ("B2", t + j2 if j2 >= 0 else -1)]
                for fill, e_idx in fills:
                    if e_idx < 0 or e_idx >= n - 5:
                        continue
                    h2, l2 = hi[e_idx:], lo[e_idx:]
                    row = dict(base, fill=fill, e_idx=e_idx,
                               end_px=float(cl[n - 1]))
                    for s_ in STOPS:
                        row[f"up{s_}"] = first_ge(h2, L + 4 * s_)
                    for t_ in TGTS:
                        row[f"dn{t_}"] = first_ge(-l2, -(L - 4 * t_))
                    w = min(len(h2), 3600)
                    row["mae1h"] = float((h2[:w].max() - L) / 4)
                    row["mfe1h"] = float((L - l2[:w].min()) / 4)
                    rows.append(row)
            n_done += 1
        print(f"  {Path(fpath).name}: sessions {n_done} rows {len(rows):,} "
              f"({(_time.time()-t0c)/60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    df["year"] = df["session"].str[:4]
    df.to_parquet(OUT)
    print(f"wrote {OUT}: {len(df):,} rows | {(_time.time()-t0c)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
