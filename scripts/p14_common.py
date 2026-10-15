#!/usr/bin/env python
"""P14 fire-time feature computation (shared by builder, truncation test
and streaming replay).

Chronology convention (the whole point):
  1s bar x covers [x, x+1); its OHLCV/delta are knowable at x+1.
  Touch at second t = first band contact during bar t (knowable t+1).
  R4v10 fires at bar f = first bar whose low prints >= 10.0 pts below the
  zone near edge, f <= t+60. A checkpoint-k decision uses bars <= f+k
  only; the order goes out at the bar boundary and fills during bar
  f+k+1 at its close. Pre-touch features use bars <= t-1 only.
  `fire_rows_for_session(..., truncate_at=x)` physically slices every
  array at x+1 and must reproduce identical feature values for any
  decision with dec_sec <= x.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.features.interactions import build_session_data                  # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402

DAILY = Path("/Volumes/T9/nq_levels/daily")
B1S_DIR = "/Volumes/T9/nq_levels/bars_1s"
R4_TICKS = 40.0                       # 10.0 pts
R4_WIN = 60
CKPTS = (0, 1, 2, 3, 5, 10, 15)
S_GRID = (7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0)
T_GRID = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0)
FULL_VARIANTS = (0, 2, 5, 10, 15)     # store full S/T grids for these


def load_meta():
    """Interaction rows (tap features) + resistance-role labels, keyed."""
    d0 = layer_dir("daily", str(ROOT / "configs"))
    KEY = ["session", "zid", "tap_sec"]
    it = pd.read_parquet(DAILY / "interactions.parquet").drop_duplicates(KEY)
    tl = pd.read_parquet(d0 / "p9_timeline.parquet",
                         columns=KEY + ["r4v10.0_t"]).rename(
        columns={"r4v10.0_t": "f_r4_ref"})
    from run_p6_spatial import add_labels
    re = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    keep = [c for c in ("y_strict", "side_above", "n_fwd", "seg")
            if c in re.columns]
    re = re[KEY + keep].drop_duplicates(KEY)
    it = it.drop(columns=[c for c in keep if c in it.columns],
                 errors="ignore")
    it = it.merge(tl, on=KEY, how="left").merge(re, on=KEY, how="inner")
    return it


def first_cross(a, thr):
    j = np.flatnonzero(a >= thr)
    return int(j[0]) if j.size else -1


def fire_rows_for_session(sd, taps, truncate_at=None):
    """Compute fire-time rows for one session.

    sd: SessionData; taps: DataFrame rows with tap_sec, zlo_t, zhi_t,
    tap_no plus passthrough meta. Returns list of dicts. With
    truncate_at=x, arrays are sliced to [:x+1] and only decisions with
    dec_sec <= x are emitted (used by the invariance test)."""
    n_full = sd.cl.size
    lim = n_full if truncate_at is None else min(n_full, truncate_at + 1)
    cl, hi, lo = sd.cl[:lim], sd.hi[:lim], sd.lo[:lim]
    cvol, cdlt = sd.cum_vol[:lim + 1], sd.cum_dlt[:lim + 1]
    rows = []
    for r in taps.itertuples():
        t = int(r.tap_sec)
        zlo, zhi = float(r.zlo_t), float(r.zhi_t)
        hw = (zhi - zlo) / 2.0
        e = min(lim, t + R4_WIN + 1)
        if t >= lim or e - t < 2:
            continue
        w_lo = lo[t:e]; w_hi = hi[t:e]; w_cl = cl[t:e]
        rev = np.clip(zlo - np.minimum.accumulate(w_lo), 0, None)
        f_rel = first_cross(rev, R4_TICKS)
        if f_rel < 0:
            continue
        f = t + f_rel
        dec = f                                   # earliest decision bar
        if truncate_at is not None and dec > truncate_at:
            continue
        # ---- touch -> fire window (bars t..f inclusive) ----
        wf_cl = cl[t:f + 1]; wf_hi = hi[t:f + 1]; wf_lo = lo[t:f + 1]
        m = f_rel + 1
        revf = rev[:m]
        pen = np.clip(np.maximum.accumulate(wf_hi) - zlo, 0, None)
        d_cl = np.diff(wf_cl) if m > 1 else np.zeros(1)
        path = float(np.abs(d_cl).sum())
        net = float(wf_cl[-1] - wf_cl[0])
        vol_tf = float(cvol[f + 1] - cvol[t])
        dlt_tf = float(cdlt[f + 1] - cdlt[t])
        buy_tf = (vol_tf + dlt_tf) / 2.0
        sell_tf = (vol_tf - dlt_tf) / 2.0
        base_v = (cvol[t] - cvol[max(0, t - 10800)]) / max(1, t - max(0, t - 10800))
        above_far = wf_cl > zhi
        above_near = wf_cl > zlo
        # pre-touch approach (bars <= t-1)
        a0 = max(0, t - 60)
        app_net = float(cl[t - 1] - cl[max(0, t - 301)]) if t > 301 else 0.0
        vol_pre = float(cvol[t] - cvol[a0])
        dlt_pre = float(cdlt[t] - cdlt[a0])
        buy_pre = (vol_pre + dlt_pre) / 2.0
        prog_pre = float(cl[t - 1] - cl[a0]) if t > 0 else 0.0
        row = dict(session=r.session, zid=int(r.zid), tap_sec=t,
                   fire_sec=f, dec_sec=dec, fire_s=f_rel,
                   tap_no=int(r.tap_no), seg=int(r.seg),
                   # family 1 — speed
                   t_m2=first_cross(revf, 8.0), t_m4=first_cross(revf, 16.0),
                   t_m6=first_cross(revf, 24.0), t_m8=first_cross(revf, 32.0),
                   v_inst=float((revf[-1] - revf[max(0, m - 4)]) / 4.0 / min(3, m)),
                   v_avg=float(10.0 / max(1, f_rel)),
                   mono=float((np.diff(revf) > 0).mean()) if m > 1 else 1.0,
                   # family 2 — path efficiency
                   eff_ratio=float(abs(net) / path) if path > 0 else 1.0,
                   n_flips=int((np.diff(np.sign(d_cl[d_cl != 0])) != 0).sum())
                   if (d_cl != 0).any() else 0,
                   ret_above=float(above_near.mean()),
                   t_inzone=float(((wf_cl >= zlo) & (wf_cl <= zhi)).mean()),
                   # family 3 — penetration
                   pen_max=float(pen[-1] / 4.0),
                   pen_hw=float(pen[-1] / max(1.0, 2 * hw)),
                   hit_mid=int(pen[-1] >= hw), hit_far=int(pen[-1] >= 2 * hw),
                   beyond_far=float(np.clip(wf_hi.max() - zhi, 0, None) / 4.0),
                   # family 4 — acceptance
                   s_above_far=int(above_far.sum()),
                   max_run_far=int(_maxrun(above_far)),
                   # family 5/6 — flow / absorption
                   dlt_fire=dlt_tf, vol_fire=vol_tf,
                   sell_share=float(sell_tf / vol_tf) if vol_tf > 0 else 0.5,
                   intx_fire=float((vol_tf / m) / base_v) if base_v > 0 else np.nan,
                   sell_imp=float(revf[-1] / 4.0 / (sell_tf / 1000 + 0.1)),
                   buy_fail=float(pen[-1] / 4.0 / (buy_tf / 1000 + 0.1)),
                   buy_pre_prog=float(prog_pre / 4.0 / (buy_pre / 1000 + 0.1)),
                   app_net_5m=float(app_net / 4.0),
                   # family 7 — regime (session-causal)
                   rng_used=float((np.max(hi[:t]) - np.min(lo[:t])) / 4.0)
                   if t > 10 else np.nan,
                   ret_1h=float((cl[t - 1] - cl[max(0, t - 3600)]) / 4.0)
                   if t > 0 else 0.0,
                   ret_3h=float((cl[t - 1] - cl[max(0, t - 10800)]) / 4.0)
                   if t > 0 else 0.0,
                   tod_hr=float(t / 3600.0), dow=int(getattr(r, "dow", 0)))
        # checkpoint deltas (FEATURES: bars <= f+k only) + entry/outcome
        # (LABELS: full tape by design — they are what we predict)
        for k in CKPTS:
            j = f + k
            if j + 1 > lim - 1:          # feature bars through j unavailable
                row[f"ck{k}_ok"] = 0
                continue
            row[f"ck{k}_ok"] = 1
            row[f"ck{k}_rev"] = float(np.clip(zlo - lo[t:j + 1].min(),
                                              0, None) / 4.0)
            row[f"ck{k}_cl"] = float((zlo - cl[j]) / 4.0)
            row[f"ck{k}_dlt"] = float(cdlt[j + 1] - cdlt[f + 1])
            e_idx = j + 1
            if e_idx >= n_full - 5:
                continue
            e_px = float(sd.cl[e_idx])
            row[f"ck{k}_epx"] = e_px
            row[f"ck{k}_eidx"] = e_idx
            if k in FULL_VARIANTS:
                h2 = sd.hi[e_idx:]
                l2 = sd.lo[e_idx:]
                for s_ in S_GRID:
                    row[f"ck{k}_up{s_}"] = first_cross(h2, e_px + 4 * s_)
                for t_ in T_GRID:
                    row[f"ck{k}_dn{t_}"] = first_cross(-l2, -(e_px - 4 * t_))
                row[f"ck{k}_endpx"] = float(sd.cl[n_full - 1])
        rows.append(row)
    return rows


def _maxrun(b):
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def load_session_arrays(sess, b1s_cache=None):
    """Load SessionData + tap rows for one session (test helper)."""
    import glob
    d0 = layer_dir("daily", str(ROOT / "configs"))
    it = load_meta()
    g = it[it["session"].astype(str) == str(sess)]
    b1m = pd.read_parquet(d0 / "bars_1m.parquet")
    b1m["ts"] = pd.DatetimeIndex(b1m["ts"]).tz_convert("UTC")
    b1m = b1m[b1m["is_front"]]
    mon = str(sess)[:7]
    b1s = pd.read_parquet(f"{B1S_DIR}/{mon}.parquet")
    b1s = b1s[b1s["is_front"]]
    b1s = b1s[b1s["session"].astype(str) == str(sess)]
    b1m_s = b1m[b1m["session"].astype(str) == str(sess)]
    sd = build_session_data(b1s.sort_values("ts").reset_index(drop=True),
                            b1m_s.sort_values("ts").reset_index(drop=True))
    return sd, g
