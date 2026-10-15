"""The per-session interaction engine: taps, causal features, outcomes.

One session at a time, chronologically: active zones (carried across sessions
in roll-free prices) are screened against 1-minute bars, taps are pinned to
the exact second on density-filled 1-second bars, approach and state features
are computed strictly from data before the tap second, and the outcome is
measured forward with the frozen sweep machinery.

Causality rules enforced here, because they are the ones that silently break:

* A zone exists only from its **confirmation minute**; a tap in the same
  session must be at or after it.
* **Approach features end at the second before the tap.** Pinned by a
  truncation test: corrupting every second from the tap onward changes no
  feature.
* **Prior-tap outcome features obey a 2-hour maturity rule**: a previous
  interaction contributes its label and depth only when the current tap is at
  least ``MATURITY_S`` after it, and the label is computed from a window
  capped at ``MATURITY_S`` — so the feature never encodes data that postdates
  the tap being predicted. The *outcome record* of an interaction, by
  contrast, may look as far forward as the session allows; outcomes are
  labels, not features.
* Re-arm: a new tap requires closes out of the band by max(8t, half-width)
  after the previous tap, and at least ``REARM_S`` seconds of separation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..eval.labeling import SessionPath
from ..labels.sweep import sweep_metrics
from .zones import Zone, epoch_s

TICKS = 4
REARM_S = 900
MATURITY_S = 7200
MAX_TAPS = 6
MAX_AGE_SESSIONS = 30
ACCEPT_DWELL_S = 300


@dataclass
class SessionData:
    """Everything the engine needs for one session, prebuilt once."""
    sec_ts: np.ndarray          # epoch seconds per dense second
    hi: np.ndarray
    lo: np.ndarray
    cl: np.ndarray
    vol: np.ndarray
    dlt: np.ndarray
    path: SessionPath
    min_of_sec: np.ndarray      # dense second -> minute index
    m_hi: np.ndarray            # per-minute arrays (screening)
    m_lo: np.ndarray
    m_cl: np.ndarray
    m_vol: np.ndarray
    m_ts: np.ndarray            # epoch seconds of minute start
    seg_of_sec: np.ndarray      # int segment code per second
    pt_min_of_sec: np.ndarray   # minutes-past-midnight PT per second
    cum_vol: np.ndarray         # dense cumulative volume
    cum_dlt: np.ndarray
    cum_pv: np.ndarray          # cumulative close*vol for session vwap
    pre_max: np.ndarray         # running session high per second
    pre_min: np.ndarray
    m_pre_max: np.ndarray       # running extremes per minute (through m)
    m_pre_min: np.ndarray


def build_session_data(b1s: pd.DataFrame, b1m: pd.DataFrame) -> SessionData:
    """Dense per-second arrays via forward-fill; minute arrays for screening."""
    ts = epoch_s(b1s["ts"])
    t0, t1 = int(ts[0]), int(ts[-1])
    n = t1 - t0 + 1
    idx = ts - t0
    hi = np.full(n, np.nan)
    lo = np.full(n, np.nan)
    cl = np.full(n, np.nan)
    vol = np.zeros(n)
    dlt = np.zeros(n)
    cl[idx] = b1s["close_t"].to_numpy("float64")
    hi[idx] = b1s["high_t"].to_numpy("float64")
    lo[idx] = b1s["low_t"].to_numpy("float64")
    vol[idx] = b1s["volume"].to_numpy("float64")
    dlt[idx] = (b1s["buy_vol"].to_numpy("float64")
                - b1s["sell_vol"].to_numpy("float64"))
    c = pd.Series(cl).ffill().to_numpy()
    hi = np.where(np.isnan(hi), c, hi)
    lo = np.where(np.isnan(lo), c, lo)
    cl = c
    ok = ~np.isnan(cl)
    hi, lo, cl, vol, dlt = hi[ok], lo[ok], cl[ok], vol[ok], dlt[ok]
    sec_ts = (t0 + np.flatnonzero(ok)).astype("int64")

    path = SessionPath("s", np.arange(cl.size, dtype="int32"), hi, lo, cl)
    m_ts = epoch_s(b1m["ts"])
    min_of_sec = np.searchsorted(m_ts, sec_ts, side="right") - 1
    pt = ((sec_ts - 8 * 3600) % 86400) // 60      # approximate PT minutes; DST
    # handled coarsely — segment boundaries are features, not measurements
    seg = np.select(
        [(pt >= 390) & (pt < 540), (pt >= 540) & (pt < 780),
         (pt >= 300) & (pt < 390), (pt >= 0) & (pt < 300)],
        [3, 4, 2, 1], default=0)                  # 0 asia,1 london,2 premkt,3 ny-am,4 ny-pm
    return SessionData(
        sec_ts=sec_ts, hi=hi, lo=lo, cl=cl, vol=vol, dlt=dlt, path=path,
        min_of_sec=min_of_sec,
        m_hi=b1m["high_t"].to_numpy("float64"),
        m_lo=b1m["low_t"].to_numpy("float64"),
        m_cl=b1m["close_t"].to_numpy("float64"),
        m_vol=b1m["volume"].to_numpy("float64"), m_ts=m_ts,
        seg_of_sec=seg.astype("int8"), pt_min_of_sec=pt.astype("int32"),
        cum_vol=np.concatenate(([0.0], np.cumsum(vol))),
        cum_dlt=np.concatenate(([0.0], np.cumsum(dlt))),
        cum_pv=np.concatenate(([0.0], np.cumsum(cl * vol))),
        pre_max=np.maximum.accumulate(hi),
        pre_min=np.minimum.accumulate(lo),
        m_pre_max=np.maximum.accumulate(b1m["high_t"].to_numpy("float64")),
        m_pre_min=np.minimum.accumulate(b1m["low_t"].to_numpy("float64")))


def capped_state_label(sd: SessionData, zlo: float, zhi: float, t: int,
                       atr: float) -> Tuple[str, float]:
    """(category, depth) for STATE features, measured only inside
    [t, t + MATURITY_S] so it is knowable MATURITY_S after the tap."""
    j1 = min(sd.cl.size, t + MATURITY_S)
    hi_f, lo_f, cl_f = sd.hi[t:j1], sd.lo[t:j1], sd.cl[t:j1]
    if hi_f.size < 60:
        return "U", 0.0
    below = zhi < cl_f[0] or (zlo + zhi) / 2 < sd.cl[max(0, t - 1)]
    near, far = (zhi, zlo) if below else (zlo, zhi)
    rev_run = (np.maximum.accumulate(hi_f) - near) if below \
        else (near - np.minimum.accumulate(lo_f))
    depth_run = np.maximum.accumulate(
        np.clip((far - lo_f) if below else (hi_f - far), 0.0, None))
    j_rev = int(np.searchsorted(rev_run, 15 * TICKS))
    if j_rev < rev_run.size and depth_run[j_rev] <= 4.0:
        return "A", float(depth_run[-1])
    beyond_cl = (cl_f < far) if below else (cl_f > far)
    fb = np.flatnonzero(depth_run > 0)
    if not fb.size:
        return "D", 0.0
    D = 0.10 * atr
    inside_again = (cl_f >= near) if below else (cl_f <= near)
    rec = np.flatnonzero(inside_again[fb[0]:])
    if rec.size and depth_run[fb[0] + rec[0]] <= D \
            and (fb[0] + rec[0] - fb[0]) <= 1800 \
            and j_rev < rev_run.size:
        return "B", float(depth_run[-1])
    if beyond_cl.sum() >= ACCEPT_DWELL_S and not rec.size:
        return "F", float(depth_run[-1])          # decisive acceptance -> flip
    return "C", float(depth_run[-1])


def approach_features(sd: SessionData, t: int, zlo: float, zhi: float,
                      atr: float) -> Dict:
    """Everything about how price came into the zone, from [t-1800, t)."""
    mid = (zlo + zhi) / 2.0
    p0 = sd.cl[t - 1] if t > 0 else sd.cl[0]
    side = 1 if p0 > mid else -1                  # +1 approaching from above
    f: Dict = {"app_side_above": int(side == 1)}
    for w, tag in ((300, "5m"), (900, "15m"), (1800, "30m")):
        a = max(0, t - w)
        disp = sd.cl[t - 1] - sd.cl[a]
        f[f"app_toward_{tag}"] = float(-side * disp) / atr
        f[f"app_range_{tag}"] = float(sd.hi[a:t].max() - sd.lo[a:t].min()) / atr \
            if t > a else np.nan
        dv = sd.cum_vol[t] - sd.cum_vol[a]
        dd = sd.cum_dlt[t] - sd.cum_dlt[a]
        f[f"app_vol_{tag}"] = float(dv)
        f[f"app_delta_{tag}"] = float(dd / dv) if dv > 0 else np.nan
    b = max(0, t - 10800)
    base_rate = (sd.cum_vol[t] - sd.cum_vol[b]) / max(1, t - b)
    f["app_volx_5m"] = (f["app_vol_5m"] / 300) / base_rate if base_rate > 0 else np.nan
    v5 = f.pop("app_vol_5m")
    f.pop("app_vol_15m")
    f.pop("app_vol_30m")
    f["app_speed_5m"] = f["app_toward_5m"] * 12.0          # ATR per hour
    prev = max(0, t - 600)
    disp_prev = sd.cl[max(0, t - 300)] - sd.cl[prev]
    f["app_accel"] = f["app_speed_5m"] - (-side * disp_prev / atr) * 12.0
    f["app_compress"] = f["app_range_5m"] / f["app_range_30m"] \
        if f.get("app_range_30m") else np.nan
    # 1-minute texture of the approach
    m1 = sd.min_of_sec[t]
    a1 = max(0, m1 - 15)
    seq = sd.m_cl[a1:m1]
    if seq.size >= 3:
        d = np.sign(np.diff(-side * seq))
        f["app_retraces_15m"] = int((np.diff(d[d != 0]) != 0).sum()) \
            if (d != 0).sum() > 1 else 0
        toward = np.sign(np.diff(-side * seq)) > 0
        run = 0
        for x in toward[::-1]:
            if x:
                run += 1
            else:
                break
        f["app_monotone_1m"] = int(run)
    else:
        f["app_retraces_15m"] = np.nan
        f["app_monotone_1m"] = np.nan
    return f


def context_features(sd: SessionData, t: int, atr: float) -> Dict:
    v = sd.cum_vol[t]
    pv = sd.cum_pv[t]
    vwap = pv / v if v > 0 else sd.cl[t - 1]
    m1 = sd.min_of_sec[t]
    a1 = max(0, m1 - 60)
    rets = np.diff(sd.m_cl[a1:m1])
    return {"seg": int(sd.seg_of_sec[t]),
            "tod_pt_min": int(sd.pt_min_of_sec[t]),
            "sess_elapsed_s": int(t),
            "range_used_atr": float((sd.pre_max[t - 1] - sd.pre_min[t - 1]) / atr)
            if t > 0 else 0.0,
            "vwap_dist_atr": float((sd.cl[t - 1] - vwap) / atr) if t > 0 else 0.0,
            "rv60_atr": float(np.sqrt((rets ** 2).sum()) / atr) if rets.size >= 30 else np.nan,
            "release_slot": int(330 <= sd.pt_min_of_sec[t] < 360)}


def process_session(registry: Dict[int, Zone], new_zones: List[Zone],
                    sd: SessionData, atr: float, adj: float, chain: int,
                    sess_ord: int, dow: int, atr_pct: float,
                    vol_feats, d20: Tuple[float, float],
                    state_out: Optional[Dict] = None) -> List[Dict]:
    """Run one session: detect taps, emit interaction rows, update states.

    ``vol_feats(zlo_cur, zhi_cur)`` returns trailing-profile volume features.
    ``d20`` = prior-20-session (high, low) in roll-free ticks.
    ``state_out``, if given, receives the within-session tap state the display
    simulation needs: ``{"upd": {zid: [(ts, cat, depth), ...]}}`` — the same
    maturity-capped labels the tap features themselves use.
    """
    rows: List[Dict] = []
    for z in new_zones:
        registry[z.zid] = z
    # order zones for deterministic output
    zids = sorted(registry)
    n_min = sd.m_cl.size
    hi20_rf, lo20_rf = d20

    # per-zone screening masks on minutes
    events = []                                   # (tap_sec, zid)
    zinfo = {}
    for zid in zids:
        z = registry[zid]
        if z.retired or sess_ord - z.form_ord > MAX_AGE_SESSIONS:
            z.retired = True
            continue
        zlo_rf, zhi_rf = z.band_rf()
        zlo, zhi = zlo_rf + adj, zhi_rf + adj
        hyst = max(8.0, z.hw_t)
        m0 = z.form_min if z.form_ord == sess_ord else 0
        mask = (sd.m_lo <= zhi) & (sd.m_hi >= zlo)
        mask[:m0] = False
        cand = np.flatnonzero(mask)
        out_m = (sd.m_cl < zlo - hyst) | (sd.m_cl > zhi + hyst)
        out_m[:m0] = False
        if not cand.size:
            # no band contact this session: any out-of-band close re-arms
            if not z.armed and out_m.any():
                z.armed = True
            zinfo[zid] = (zlo, zhi, None, None, mask)
            continue
        zinfo[zid] = (zlo, zhi, cand, out_m, mask)
        armed, last_ts = z.armed, z.last_tap_ts
        taps_now = 0
        j_out = 0
        outs = np.flatnonzero(out_m)
        last_tap_min = -1
        for m in cand:
            if not armed:
                while j_out < outs.size and outs[j_out] <= last_tap_min:
                    j_out += 1
                if j_out < outs.size and outs[j_out] < m:
                    armed = True
            if not armed or z.taps + taps_now >= MAX_TAPS:
                continue
            if sd.m_ts[m] - last_ts < REARM_S:
                continue
            s0 = int(np.searchsorted(sd.min_of_sec, m))
            s1 = int(np.searchsorted(sd.min_of_sec, m, side="right"))
            hit = np.flatnonzero((sd.lo[s0:s1] <= zhi) & (sd.hi[s0:s1] >= zlo))
            if not hit.size:
                continue
            t = s0 + int(hit[0])
            events.append((t, zid))
            armed = False
            last_ts = float(sd.sec_ts[t])
            last_tap_min = m
            taps_now += 1
        # closes beyond hysteresis after the final tap of the session re-arm
        if not armed and out_m[last_tap_min + 1:].any():
            armed = True
        z.armed = armed

    # chronological, then features + outcome + state updates in tap order
    events.sort()
    upd: Dict[int, List[Tuple[float, str, float]]] = {}
    for t, zid in events:
        z = registry[zid]
        zlo, zhi, cand, out_m, mask = zinfo[zid]
        now = float(sd.sec_ts[t])
        m1 = sd.min_of_sec[t]
        # within-session partials strictly before the tap minute
        pm = np.flatnonzero(mask[:m1])
        vol_in_now = float(sd.m_vol[pm].sum()) if pm.size else 0.0
        mins_in_now = float(pm.size)
        pivot_cur = z.price_rf + adj
        if m1 > 0:
            away_now = (pivot_cur - sd.m_pre_min[m1 - 1]) if z.dir == 1 \
                else (sd.m_pre_max[m1 - 1] - pivot_cur)
        else:
            away_now = 0.0
        if z.form_ord == sess_ord and m1 > z.form_min:
            a = z.form_min
            away_now = (pivot_cur - sd.m_lo[a:m1].min()) if z.dir == 1 \
                else (sd.m_hi[a:m1].max() - pivot_cur)
        prev_ts = upd[zid][-1][0] if upd.get(zid) else z.last_tap_ts
        mature = [(ts_, c, d) for ts_, c, d in upd.get(zid, [])
                  if now - ts_ >= MATURITY_S]
        nA = z.n_A + sum(1 for _, c, _ in mature if c == "A")
        nB = z.n_B + sum(1 for _, c, _ in mature if c == "B")
        nC = z.n_C + sum(1 for _, c, _ in mature if c in ("C", "F"))
        pen = max([z.max_pen_t] + [d for _, _, d in mature])
        n_pending = len(upd.get(zid, [])) - len(mature)

        r = {"session_ord": sess_ord, "zid": zid, "tap_sec": int(t),
             "tap_ts": now, "scale": z.scale, "dir": z.dir,
             "price_t": pivot_cur, "zlo_t": zlo, "zhi_t": zhi,
             "hw_t": z.hw_t, "atr_t": atr,
             "tap_no": z.taps + len(upd.get(zid, [])) + 1,
             "age_min": (now - z.form_ts_val) / 60.0,
             "age_sessions": sess_ord - z.form_ord,
             "mins_since_last": (now - prev_ts) / 60.0
             if np.isfinite(prev_ts) else np.nan,
             "n_prior_A": nA, "n_prior_B": nB, "n_prior_C": nC,
             "n_pending": n_pending,
             "prior_pen_t": pen, "vol_through": z.vol_through + vol_in_now,
             "mins_inside": z.mins_inside + mins_in_now,
             "max_away_atr": max(z.max_away_t, float(away_now)) / atr,
             "flipped": int(z.flipped_at is not None and z.flipped_at <= now),
             "base_range_t": z.base_range_t, "dep_mag_atr": z.dep_mag_t / z.atr_form,
             "dep_vel": z.dep_vel_tpm / z.atr_form * 60.0,
             "dep_volx": z.dep_volx, "dep_delta": z.dep_delta,
             "broke_prior": z.broke_prior, "form_seg": z.form_seg,
             "hw_atr": z.hw_t / atr,
             "d20_hi_atr": (hi20_rf + adj - pivot_cur) / atr
             if np.isfinite(hi20_rf) else np.nan,
             "d20_lo_atr": (pivot_cur - (lo20_rf + adj)) / atr
             if np.isfinite(lo20_rf) else np.nan,
             "dow": dow, "atr_pctile": atr_pct}
        vz = vol_feats(zlo, zhi)
        r.update(vz)
        r.update(approach_features(sd, t, zlo, zhi, atr))
        r.update(context_features(sd, t, atr))
        r["dir_match"] = int((r["app_side_above"] == 1) == (z.dir == 1))
        m = sweep_metrics(sd.path, zlo, zhi, t)
        m.pop("side", None)
        r.update({f"o_{k}": v for k, v in m.items()})
        rows.append(r)

        cat, depth = capped_state_label(sd, zlo, zhi, t, atr)
        upd.setdefault(zid, []).append((now, cat, depth))
        if cat == "F" and z.flipped_at is None:
            z.flipped_at = now + MATURITY_S       # knowable only at maturity

    if state_out is not None:
        state_out["upd"] = {zid: list(lst) for zid, lst in upd.items()}
    # end-of-session state roll-up
    for zid, lst in upd.items():
        z = registry[zid]
        for _, c, d in lst:
            z.taps += 1
            z.max_pen_t = max(z.max_pen_t, d)
            if c == "A":
                z.n_A += 1
            elif c == "B":
                z.n_B += 1
            elif c in ("C", "F"):
                z.n_C += 1
        z.last_tap_ts = lst[-1][0]
        if z.taps >= MAX_TAPS:
            z.retired = True
    for zid in zids:
        z = registry[zid]
        if z.retired:
            continue
        info = zinfo.get(zid)
        if info is None:
            continue
        zlo, zhi, cand, out_m, mask = info
        pm = np.flatnonzero(mask)
        z.vol_through += float(sd.m_vol[pm].sum()) if pm.size else 0.0
        z.mins_inside += float(pm.size)
        pivot_cur = z.price_rf + adj
        m0 = z.form_min if z.form_ord == sess_ord else 0
        if m0 < n_min:
            away = (pivot_cur - sd.m_lo[m0:].min()) if z.dir == 1 \
                else (sd.m_hi[m0:].max() - pivot_cur)
            z.max_away_t = max(z.max_away_t, float(away))
    for zid in list(registry):
        if registry[zid].retired:
            del registry[zid]
    return rows
