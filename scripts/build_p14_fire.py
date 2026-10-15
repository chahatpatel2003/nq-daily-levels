#!/usr/bin/env python
"""Build the P14 fire-time dataset: one row per R4v10 event (ALL taps),
features causal at each checkpoint per p14_common conventions, entry
fills + S/T crossing-time grids as labels. Independently recomputes the
R4 fire second and cross-checks p9_timeline. Then joins the causally
clean pre-touch features and recomputes STRICTLY MATURED zone-prior
outcome features (fixes AUDIT FINDING #2), plus matured recent-health
features from the event stream itself."""
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
from p14_common import B1S_DIR, fire_rows_for_session, load_meta               # noqa: E402

OUT = Path("/Volumes/T9/nq_levels/daily/p14_fire.parquet")
MATURITY_S = 7200

PRE_KEEP = ["app_speed_5m", "app_accel", "app_volx_5m", "app_delta_5m",
            "app_toward_30m", "app_range_5m", "app_retraces_15m",
            "rv60_atr", "atr_pctile", "range_used_atr", "vwap_dist_atr",
            "tod_pt_min", "dep_vel", "mins_since_last", "age_min",
            "hw_atr", "n_fwd"]


def main() -> int:
    t0c = _time.time()
    it = load_meta()
    it["year"] = pd.to_datetime(it["session"].astype(str)).dt.year
    it = it[(it["year"] >= 2021) & (it["side_above"] == 0)
            & (it["n_fwd"] >= 1800)]
    need = ["session", "zid", "tap_sec", "zlo_t", "zhi_t", "tap_no", "seg",
            "tap_ts", "f_r4_ref", "y_strict"]
    have = [c for c in need if c in it.columns]
    taps = it[have + [c for c in PRE_KEEP if c in it.columns]].copy()
    by = {s: g for s, g in taps.groupby("session", sort=False)}
    d0 = layer_dir("daily", str(ROOT / "configs"))
    b1m_all = pd.read_parquet(d0 / "bars_1m.parquet")
    b1m_all["ts"] = pd.DatetimeIndex(b1m_all["ts"]).tz_convert("UTC")
    b1m_by = {s: g.sort_values("ts").reset_index(drop=True)
              for s, g in b1m_all[b1m_all["is_front"]]
              .groupby("session", sort=False)}
    del b1m_all

    rows = []
    n_done, mismatch = 0, 0
    for fpath in sorted(glob.glob(f"{B1S_DIR}/[!._]*.parquet")):
        b1s_m = pd.read_parquet(fpath)
        b1s_m = b1s_m[b1s_m["is_front"]]
        for s, b1s in b1s_m.groupby("session", sort=True):
            g = by.get(s)
            if g is None or s not in b1m_by:
                continue
            sd = build_session_data(
                b1s.sort_values("ts").reset_index(drop=True), b1m_by[s])
            rr = fire_rows_for_session(sd, g)
            ref = {(int(x.zid), int(x.tap_sec)): x.f_r4_ref
                   for x in g.itertuples() if np.isfinite(x.f_r4_ref)}
            for x in rr:
                k = (x["zid"], x["tap_sec"])
                if k in ref and int(ref[k]) != x["fire_s"]:
                    mismatch += 1
            rows.extend(rr)
            n_done += 1
        print(f"  {Path(fpath).name}: sessions {n_done} rows {len(rows):,} "
              f"({(_time.time()-t0c)/60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    print(f"fires {len(df):,} | R4-second mismatches vs p9_timeline: "
          f"{mismatch}")
    # join pre-touch features + tap_ts + per-tap outcome labels for priors
    meta = taps[["session", "zid", "tap_sec", "tap_ts", "y_strict"]
                + [c for c in PRE_KEEP if c in taps.columns]]
    df = df.merge(meta, on=["session", "zid", "tap_sec"], how="left")

    # STRICT matured zone-prior features (fix of finding #2): a prior
    # same-zone tap's outcome counts only when tap_ts_prior + 7200 <= now
    df = df.sort_values("tap_ts").reset_index(drop=True)
    zprior_n = np.zeros(len(df)); zprior_succ = np.full(len(df), 0.5)
    hist: dict = {}
    for i, r in enumerate(df.itertuples()):
        h = hist.get(r.zid, [])
        mat = [y for ts_, y in h if ts_ + MATURITY_S <= r.tap_ts]
        if mat:
            zprior_n[i] = len(mat)
            zprior_succ[i] = float(np.mean(mat))
        y = getattr(r, "y_strict", np.nan)
        if np.isfinite(y):
            hist.setdefault(r.zid, []).append((r.tap_ts, float(y)))
    df["zprior_n"] = zprior_n
    df["zprior_succ"] = zprior_succ

    # matured recent-health from the R4 event stream itself: outcome of a
    # prior R4 (12.5/30 from ck2 entry) is knowable at its exit second
    up = df["ck2_up12.5"].fillna(-1).to_numpy()
    dn = df["ck2_dn30.0"].fillna(-1).to_numpy()
    has = df["ck2_epx"].notna().to_numpy()
    win = (dn >= 0) & ((up < 0) | (dn < up))
    close_rel = np.where(win, dn, np.where(up >= 0, up, 1800))
    eidx = df["ck2_eidx"].fillna(df["fire_sec"] + 3).to_numpy()
    exit_ts = (df["tap_ts"].to_numpy()
               + (eidx - df["tap_sec"].to_numpy()) + close_rel)
    res_ts = exit_ts[has]
    res_R = np.where(win, 2.32, -1.08)[has]
    order = np.argsort(res_ts, kind="stable")
    sorted_ts, sorted_R = res_ts[order], res_R[order]
    tap_ts = df["tap_ts"].to_numpy()
    for k in (5, 10, 20, 50):
        out = np.full(len(df), np.nan)
        for i in range(len(df)):
            j = np.searchsorted(sorted_ts, tap_ts[i] - 30)
            if j >= k:
                out[i] = sorted_R[j - k:j].mean()
            elif j > 0:
                out[i] = sorted_R[:j].mean()
        df[f"recent{k}"] = out
    df.to_parquet(OUT)
    print(f"wrote {OUT} ({len(df):,} rows, {len(df.columns)} cols, "
          f"{(_time.time()-t0c)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
