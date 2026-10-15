"""L2: per-session pre-open context and the session's excursion profile.

Two products, kept in separate tables because they play opposite roles:

**Context** — everything knowable at 06:25 PT. Volatility scales, gap and
location, overnight direction and momentum, participation, calendar, and a
tape-derived macro-shock reading. These are the conditioning variables for the
E1 baseline and for every regime split later.

**Excursion profile** — the running maximum distance price travelled above and
below the 06:25 reference, sampled every minute through the RTH session. This
single object answers every destination question without ever touching the
1-second layer again:

    P(touch a price d above by minute t)  ==  P(up_t[t] >= d)

because reaching a price necessarily means passing every price in between.
That identity is why the E1 baseline is a *distribution model of session
excursions* rather than a per-price classifier — it is the same thing, but
exact, and about three orders of magnitude cheaper.

Nothing here reads a bar at or after the decision cut except when computing the
excursion profile, which is an outcome by construction and is suffixed to make
that unmistakable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..clock import (Calendar, decision_time_utc, prior_rth_window_utc,
                     prior_trade_date, pt_instant, rth_window_utc,
                     segment_windows_utc, session_open_utc, target_window_utc)
from .tape import window_bars

TICK = 0.25
# Minute checkpoints across RTH. 0..150 covers the target window; the tail is
# kept for "did it arrive later" diagnostics, never for labels.
RTH_MINUTES = 390


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _minute_closes(b: pd.DataFrame) -> np.ndarray:
    """1-minute closes in ticks — the scale for realised-volatility work.

    Realised variance from 1-second prints on NQ is dominated by bid-ask
    bounce and estimates the spread, not the volatility; one minute is the
    coarsest resolution that still resolves the overnight session well.
    """
    if b.empty:
        return np.empty(0)
    g = b.groupby(b["ts"].dt.floor("1min"))["close_t"].last()
    return g.to_numpy(dtype="float64")


def _rv(b: pd.DataFrame) -> float:
    """Realised volatility in ticks over a window (sum of squared 1-min log
    returns, rescaled to a price move)."""
    c = _minute_closes(b)
    if c.size < 3:
        return np.nan
    r = np.diff(np.log(c))
    return float(np.sqrt(np.nansum(r * r)) * np.nanmean(c))


def _ohlcv(b: pd.DataFrame) -> Dict[str, float]:
    if b.empty:
        return dict(open=np.nan, high=np.nan, low=np.nan, close=np.nan,
                    range=np.nan, volume=0.0, n_trades=0.0, delta=0.0)
    return dict(open=float(b["open_t"].iloc[0]), high=float(b["high_t"].max()),
                low=float(b["low_t"].min()), close=float(b["close_t"].iloc[-1]),
                range=float(b["high_t"].max() - b["low_t"].min()),
                volume=float(b["volume"].sum()), n_trades=float(b["n_trades"].sum()),
                delta=float(b["buy_vol"].sum() - b["sell_vol"].sum()))


def _dir_eff(b: pd.DataFrame) -> float:
    """Net displacement over path length at 1-minute resolution."""
    c = _minute_closes(b)
    if c.size < 3:
        return np.nan
    path = float(np.abs(np.diff(c)).sum())
    return float(abs(c[-1] - c[0]) / path) if path > 0 else 0.0


def _loc(x: float, lo: float, hi: float) -> float:
    """Where x sits inside [lo, hi]; <0 below, >1 above. Unclipped on purpose —
    'the open gapped a third of a range above the night' is information."""
    if not np.isfinite(x) or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan
    return float((x - lo) / (hi - lo))


# --------------------------------------------------------------------------- #
# context
# --------------------------------------------------------------------------- #


def session_context(sess: date, cal: Calendar, front: pd.DataFrame,
                    prior_front: pd.DataFrame, symbol: str) -> Dict:
    """Everything a 06:25 PT run would know. ``front`` is the whole session."""
    cut = pd.Timestamp(decision_time_utc(sess, cal))
    pre = front[front["ts"] < cut]
    row: Dict = {"session": sess, "symbol": symbol,
                 "pre_n_seconds": int(len(pre))}
    if pre.empty:
        return row

    row["ref_t"] = float(pre["close_t"].iloc[-1])
    row["ref_ts"] = pre["ts"].iloc[-1]

    # --- segments of the current night -----------------------------------
    segs = segment_windows_utc(sess, cal)
    for name, (lo, hi) in segs.items():
        w = window_bars(pre, pd.Timestamp(lo), pd.Timestamp(hi))
        o = _ohlcv(w)
        for k in ("high", "low", "range", "volume", "delta", "n_trades"):
            row[f"{name}_{k}_pre"] = o[k]
        if name in ("globex", "premarket", "london"):
            row[f"{name}_rv_pre"] = _rv(w)
            row[f"{name}_diref_pre"] = _dir_eff(w)
            row[f"{name}_ret_pre"] = (o["close"] - o["open"]) if np.isfinite(o["open"]) else np.nan

    # --- short-horizon pre-open momentum ---------------------------------
    for mins in (15, 30, 60):
        w = pre[pre["ts"] >= cut - pd.Timedelta(minutes=mins)]
        o = _ohlcv(w)
        row[f"last{mins}m_ret_pre"] = (o["close"] - o["open"]) if np.isfinite(o["open"]) else np.nan
        row[f"last{mins}m_range_pre"] = o["range"]
        row[f"last{mins}m_volume_pre"] = o["volume"]
        row[f"last{mins}m_rv_pre"] = _rv(w)

    # --- prior session ----------------------------------------------------
    p = _ohlcv(prior_front)
    for k in ("open", "high", "low", "close", "range", "volume"):
        row[f"prior_rth_{k}_prior"] = p[k]
    row["prior_rth_rv_prior"] = _rv(prior_front)
    row["prior_rth_diref_prior"] = _dir_eff(prior_front)
    row["prior_rth_ret_prior"] = (p["close"] - p["open"]) if np.isfinite(p["open"]) else np.nan

    # --- gap and location -------------------------------------------------
    ref = row["ref_t"]
    row["gap_t_prior"] = ref - p["close"] if np.isfinite(p["close"]) else np.nan
    row["loc_in_on_pre"] = _loc(ref, row.get("globex_low_pre", np.nan),
                                row.get("globex_high_pre", np.nan))
    row["loc_in_pdr_prior"] = _loc(ref, p["low"], p["high"])

    # --- tape-derived macro shock (05:30 PT release slot, 55 min pre-cut) --
    t0 = pd.Timestamp(pt_instant(sess, time(5, 30), cal))
    shock = pre[(pre["ts"] >= t0) & (pre["ts"] < t0 + pd.Timedelta(minutes=1))]
    base = pre[(pre["ts"] >= t0 - pd.Timedelta(minutes=20))
               & (pre["ts"] < t0 - pd.Timedelta(minutes=5))]
    if not shock.empty and not base.empty:
        bpm = max(1.0, float(base["volume"].sum()) / 15.0)
        row["macro_volx_pre"] = float(shock["volume"].sum()) / bpm
        row["macro_range_pre"] = float(shock["high_t"].max() - shock["low_t"].min())
        row["macro_absret_pre"] = abs(float(shock["close_t"].iloc[-1] - shock["open_t"].iloc[0]))
    else:
        row["macro_volx_pre"] = row["macro_range_pre"] = row["macro_absret_pre"] = np.nan

    row["dow"] = sess.weekday()
    row["month"] = sess.month
    return row


def add_scales(d: pd.DataFrame) -> pd.DataFrame:
    """Trailing volatility scales, all from strictly prior sessions.

    ``shift(1)`` before every rolling window is the whole point — a session's
    scale must not contain its own range, or every normalised distance in the
    project carries a slice of the answer.
    """
    d = d.sort_values("session").copy()
    r = d["prior_rth_range_prior"]
    for n in (5, 10, 14, 20, 60):
        d[f"atr{n}_t"] = r.rolling(n, min_periods=max(3, n // 3)).mean().shift(1)
    d["atr_t"] = d["atr14_t"]
    d["atr_pctile"] = d["atr_t"].rolling(252, min_periods=60).rank(pct=True).shift(1)
    for c, name in (("prior_rth_rv_prior", "rv_pd"), ("globex_rv_pre", "rv_on")):
        if c in d:
            d[f"{name}_ma20"] = d[c].rolling(20, min_periods=5).mean().shift(1)
    d["on_vol_ma20"] = d["globex_volume_pre"].rolling(20, min_periods=5).mean().shift(1)
    d["on_vol_rel"] = d["globex_volume_pre"] / d["on_vol_ma20"]
    d["on_range_rel_atr"] = d["globex_range_pre"] / d["atr_t"]
    d["gap_atr_prior"] = d["gap_t_prior"] / d["atr_t"]
    d["macro_volx_ma20"] = d["macro_volx_pre"].rolling(20, min_periods=5).mean().shift(1)
    return d


# --------------------------------------------------------------------------- #
# excursion profile (OUTCOME)
# --------------------------------------------------------------------------- #


def excursion_profile(sess: date, cal: Calendar, front: pd.DataFrame,
                      ref_t: float, minutes: int = RTH_MINUTES) -> pd.DataFrame:
    """Running max up/down excursion from the reference, per RTH minute.

    Minute ``m`` covers ``[rth_open + m, rth_open + m + 1)``; ``up_t[m]`` is the
    furthest price traded above the reference at any time up to the *end* of
    that minute. So "touched d above within the first 30 minutes" is exactly
    ``up_t[29] >= d``.
    """
    r_lo, _ = rth_window_utc(sess, cal)
    r_lo = pd.Timestamp(r_lo)
    b = front[(front["ts"] >= r_lo) & (front["ts"] < r_lo + pd.Timedelta(minutes=minutes))]
    if b.empty or not np.isfinite(ref_t):
        return pd.DataFrame()
    m = ((b["ts"] - r_lo).dt.total_seconds() // 60).astype("int32")
    g = pd.DataFrame({"m": m, "hi": b["high_t"].to_numpy(), "lo": b["low_t"].to_numpy()})
    agg = g.groupby("m").agg(hi=("hi", "max"), lo=("lo", "min"))
    full = agg.reindex(np.arange(minutes))
    up = np.maximum.accumulate(np.where(np.isnan(full["hi"]), -np.inf, full["hi"] - ref_t))
    dn = np.maximum.accumulate(np.where(np.isnan(full["lo"]), -np.inf, ref_t - full["lo"]))
    up = np.where(np.isfinite(up), up, np.nan)
    dn = np.where(np.isfinite(dn), dn, np.nan)
    return pd.DataFrame({"session": sess, "m": np.arange(minutes, dtype="int32"),
                         "up_t": up.astype("float32"), "dn_t": dn.astype("float32")})


def summarize_excursions(ex: pd.DataFrame,
                         horizons: Sequence[int] = (15, 30, 60, 90, 150)) -> Dict:
    """Excursion reached by each horizon, plus the full-window totals."""
    out: Dict = {}
    if ex.empty:
        return out
    s = ex.set_index("m")
    for h in horizons:
        if h - 1 in s.index:
            out[f"up{h}_out"] = float(s.at[h - 1, "up_t"])
            out[f"dn{h}_out"] = float(s.at[h - 1, "dn_t"])
    if RTH_MINUTES - 1 in s.index:
        out["up_rth_out"] = float(s["up_t"].max())
        out["dn_rth_out"] = float(s["dn_t"].max())
    return out


def time_to_distance(ex: pd.DataFrame, d_t: float, side: str,
                     horizon: int = 150) -> Optional[int]:
    """First RTH minute at which the excursion reaches ``d_t`` ticks."""
    if ex.empty or not np.isfinite(d_t):
        return None
    col = "up_t" if side == "above" else "dn_t"
    v = ex.loc[ex["m"] < horizon, col].to_numpy()
    hit = np.flatnonzero(v >= d_t)
    return int(hit[0]) if hit.size else None
