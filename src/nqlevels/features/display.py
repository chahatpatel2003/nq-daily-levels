"""Display-time zone scoring for the live simulation (Phase 5).

At a refresh instant t, every eligible zone gets the SAME feature vector a
tap at t would get — computed from a pre-session state snapshot plus the
within-session tap ledger (the maturity-capped labels), never from the
end-of-session rollup. The guarantee is pinned by a test: display features
evaluated at the exact second of a real tap equal that tap's recorded
features, field for field.

Eligibility mirrors the tap-acceptance rules (armed, 900-s re-arm, not
currently inside the band), so "displayed and later touched" connects
one-to-one with the interaction ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .interactions import (MATURITY_S, MAX_AGE_SESSIONS, MAX_TAPS, REARM_S,
                           SessionData, approach_features, context_features)
from .zones import Zone


@dataclass
class ZoneSnap:
    """A zone's mutable state frozen at session start (pre-rollup)."""
    taps: int
    n_A: int
    n_B: int
    n_C: int
    max_pen_t: float
    vol_through: float
    mins_inside: float
    max_away_t: float
    last_tap_ts: float
    flipped_at: Optional[float]
    armed: bool


def snap_of(z: Zone) -> ZoneSnap:
    return ZoneSnap(z.taps, z.n_A, z.n_B, z.n_C, z.max_pen_t, z.vol_through,
                    z.mins_inside, z.max_away_t, z.last_tap_ts, z.flipped_at,
                    z.armed)


@dataclass
class ZoneAux:
    """Per-session helper arrays for one zone (band in current scale)."""
    zlo: float
    zhi: float
    mask: np.ndarray            # per-minute band intersect
    out_m: np.ndarray           # per-minute beyond-hysteresis close
    cum_vol: np.ndarray         # cumsum of minute volume where mask (len n+1)
    cum_cnt: np.ndarray


def build_aux(z: Zone, sd: SessionData, adj: float,
              sess_ord: Optional[int] = None) -> ZoneAux:
    zlo_rf, zhi_rf = z.band_rf()
    zlo, zhi = zlo_rf + adj, zhi_rf + adj
    hyst = max(8.0, z.hw_t)
    mask = (sd.m_lo <= zhi) & (sd.m_hi >= zlo)
    out_m = (sd.m_cl < zlo - hyst) | (sd.m_cl > zhi + hyst)
    # a zone confirmed this session exists only from its confirmation minute —
    # identical to the m0 zeroing in process_session
    if sess_ord is not None and z.form_ord == sess_ord:
        mask[:z.form_min] = False
        out_m[:z.form_min] = False
    mv = np.where(mask, sd.m_vol, 0.0)
    return ZoneAux(zlo, zhi, mask, out_m,
                   np.concatenate(([0.0], np.cumsum(mv))),
                   np.concatenate(([0.0], np.cumsum(mask.astype("float64")))))


def upd_before(upd: List[Tuple[float, str, float]], now: float):
    return [e for e in upd if e[0] < now]


def eligible(z: Zone, snap: ZoneSnap, aux: ZoneAux, upd: List, sd: SessionData,
             t: int, sess_ord: int, atr: float,
             d_lo: float = 0.02, d_hi: float = 0.75) -> bool:
    now = float(sd.sec_ts[t])
    prior = upd_before(upd, now)
    if snap.taps + len(prior) >= MAX_TAPS:
        return False
    if sess_ord - z.form_ord > MAX_AGE_SESSIONS:
        return False
    if z.form_ord == sess_ord and sd.min_of_sec[t] < z.form_min:
        return False
    p0 = sd.cl[t - 1] if t > 0 else sd.cl[0]
    if aux.zlo <= p0 <= aux.zhi:
        return False
    mid = (aux.zlo + aux.zhi) / 2.0
    d = abs(mid - p0) / atr
    if not (d_lo <= d <= d_hi):
        return False
    # armed: 900 s since last tap AND an out-of-band close since it
    m1 = int(sd.min_of_sec[t])
    if prior:
        last_ts = prior[-1][0]
        if now - last_ts < REARM_S:
            return False
        lm = int(np.searchsorted(sd.m_ts, last_ts, side="right") - 1)
        return bool(aux.out_m[lm + 1:m1].any())
    if np.isfinite(snap.last_tap_ts):
        if now - snap.last_tap_ts < REARM_S:
            return False
        if not snap.armed:
            return bool(aux.out_m[:m1].any())
        return True
    return True


def zone_row(z: Zone, snap: ZoneSnap, aux: ZoneAux, upd: List,
             sd: SessionData, t: int, atr: float, adj: float, sess_ord: int,
             dow: int, atr_pct: float, vol_feats, d20: Tuple[float, float],
             with_approach: bool = True) -> Dict:
    """The tap-feature vector, evaluated at an arbitrary second t.

    Field-for-field mirror of the tap block in ``process_session`` — pinned
    by tests/test_display.py.
    """
    now = float(sd.sec_ts[t])
    m1 = int(sd.min_of_sec[t])
    prior = upd_before(upd, now)
    hi20_rf, lo20_rf = d20
    vol_in_now = float(aux.cum_vol[m1])
    mins_in_now = float(aux.cum_cnt[m1])
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
    prev_ts = prior[-1][0] if prior else snap.last_tap_ts
    mature = [(ts_, c, dp) for ts_, c, dp in prior if now - ts_ >= MATURITY_S]
    nA = snap.n_A + sum(1 for _, c, _ in mature if c == "A")
    nB = snap.n_B + sum(1 for _, c, _ in mature if c == "B")
    nC = snap.n_C + sum(1 for _, c, _ in mature if c in ("C", "F"))
    pen = max([snap.max_pen_t] + [dp for _, _, dp in mature])
    flip = (snap.flipped_at is not None and snap.flipped_at <= now) or \
        any(c == "F" and ts_ + MATURITY_S <= now for ts_, c, _ in prior)

    r = {"zid": z.zid, "tap_sec": int(t), "tap_ts": now,
         "scale": z.scale, "dir": z.dir, "price_t": pivot_cur,
         "zlo_t": aux.zlo, "zhi_t": aux.zhi, "hw_t": z.hw_t, "atr_t": atr,
         "tap_no": snap.taps + len(prior) + 1,
         "age_min": (now - z.form_ts_val) / 60.0,
         "age_sessions": sess_ord - z.form_ord,
         "mins_since_last": (now - prev_ts) / 60.0
         if np.isfinite(prev_ts) else np.nan,
         "n_prior_A": nA, "n_prior_B": nB, "n_prior_C": nC,
         "n_pending": len(prior) - len(mature),
         "prior_pen_t": pen, "vol_through": snap.vol_through + vol_in_now,
         "mins_inside": snap.mins_inside + mins_in_now,
         "max_away_atr": max(snap.max_away_t, float(away_now)) / atr,
         "flipped": int(flip),
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
    r.update(vol_feats(aux.zlo, aux.zhi))
    if with_approach:
        r.update(approach_features(sd, t, aux.zlo, aux.zhi, atr))
    else:
        p0 = sd.cl[t - 1] if t > 0 else sd.cl[0]
        r["app_side_above"] = int(p0 > (aux.zlo + aux.zhi) / 2.0)
    r.update(context_features(sd, t, atr))
    r["dir_match"] = int((r["app_side_above"] == 1) == (z.dir == 1))
    return r


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """The load_frame-style derived columns, identical to training."""
    d = df.copy()
    d["base_atr"] = d["base_range_t"] / d["atr_t"]
    d["prior_pen_atr"] = d["prior_pen_t"] / d["atr_t"]
    d["vol_through_log"] = np.log1p(d["vol_through"])
    d["d20_min_atr"] = d[["d20_hi_atr", "d20_lo_atr"]].abs().min(axis=1)
    npr = d["n_prior_A"] + d["n_prior_B"] + d["n_prior_C"]
    d["prior_succ_share"] = np.where(
        npr > 0, (d["n_prior_A"] + d["n_prior_B"]) / npr, np.nan)
    d["fresh_first_tap"] = ((d["tap_no"] == 1)
                            & (d["n_pending"] == 0)).astype(int)
    return d
