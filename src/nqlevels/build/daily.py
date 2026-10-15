"""L2: one row per session — reference prices, volatility scales, regimes.

Two kinds of column, and the distinction is load-bearing:

``*_prior`` / ``*_pre``
    Known at the 06:25 PT cut. These are the only columns a candidate zone or a
    pre-open feature may read. Volatility scales (``atr_t``, ``on_range_t``)
    live here and are what every distance and every zone width is normalised
    by, so that a 150-point day and a 500-point day are comparable.

``*_rth`` / ``*_target``
    Outcome columns describing what the session actually did. They exist for
    labelling and diagnostics and are **never** inputs. The column-name split
    is what makes an accidental leak visible in a feature list rather than
    invisible in a join, and :func:`preopen_columns` is the allow-list the
    audit checks against.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..build.tape import window_bars
from ..clock import (Calendar, decision_time_utc, prior_rth_window_utc,
                     prior_trade_date, rth_window_utc, segment_windows_utc,
                     session_open_utc, target_window_utc)

PREOPEN_SUFFIXES = ("_prior", "_pre", "_on", "_ref", "_atr", "_regime")
OUTCOME_SUFFIXES = ("_rth", "_target", "_out")


def preopen_columns(df: pd.DataFrame) -> List[str]:
    """Columns a pre-open model is allowed to see."""
    return [c for c in df.columns
            if not any(c.endswith(s) or f"{s}_" in c for s in OUTCOME_SUFFIXES)]


def _ohlcv(b: pd.DataFrame) -> Dict[str, float]:
    if b.empty:
        return {"open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
                "volume": 0.0, "range": np.nan}
    return {"open": float(b["open_t"].iloc[0]), "high": float(b["high_t"].max()),
            "low": float(b["low_t"].min()), "close": float(b["close_t"].iloc[-1]),
            "volume": float(b["volume"].sum()),
            "range": float(b["high_t"].max() - b["low_t"].min())}


def session_row(sess: date, cal: Calendar, front_bars: pd.DataFrame,
                prior_rth_bars: pd.DataFrame, prior_globex_bars: pd.DataFrame,
                symbol: str) -> Dict:
    """Build one L2 row. ``front_bars`` is the whole session; the pre-open
    columns are computed only from the part before the decision cut."""
    cut = decision_time_utc(sess, cal)
    t_lo, t_hi = target_window_utc(sess, cal)
    r_lo, r_hi = rth_window_utc(sess, cal)
    segs = segment_windows_utc(sess, cal)

    row: Dict = {"session": sess, "symbol": symbol}

    pre = front_bars[front_bars["ts"] < cut]
    for name, (lo, hi) in segs.items():
        o = _ohlcv(window_bars(pre, lo, hi))
        for k, v in o.items():
            row[f"{name}_{k}_pre"] = v

    row["ref_t"] = float(pre["close_t"].iloc[-1]) if not pre.empty else np.nan
    row["ref_ts"] = pre["ts"].iloc[-1] if not pre.empty else pd.NaT
    row["pre_n_seconds"] = int(len(pre))

    p = _ohlcv(prior_rth_bars)
    for k, v in p.items():
        row[f"prior_rth_{k}_prior"] = v
    g = _ohlcv(prior_globex_bars)
    for k, v in g.items():
        row[f"prior_globex_{k}_prior"] = v

    if np.isfinite(row["ref_t"]) and np.isfinite(p["close"]):
        row["gap_t_prior"] = row["ref_t"] - p["close"]

    # outcomes — never features
    tgt = window_bars(front_bars, t_lo, t_hi)
    o = _ohlcv(tgt)
    for k, v in o.items():
        row[f"target_{k}_out"] = v
    if not tgt.empty and np.isfinite(row["ref_t"]):
        row["target_max_up_out"] = float(tgt["high_t"].max() - row["ref_t"])
        row["target_max_dn_out"] = float(row["ref_t"] - tgt["low_t"].min())
    rth = window_bars(front_bars, r_lo, r_hi)
    o = _ohlcv(rth)
    for k, v in o.items():
        row[f"rth_{k}_out"] = v
    return row


def add_rolling_scales(daily: pd.DataFrame, atr_n: int = 14) -> pd.DataFrame:
    """Volatility scales from strictly prior sessions.

    ``shift(1)`` before the rolling mean is the whole point: a session's ATR
    must not include its own range. Without it every distance-normalised
    feature in the project would carry a slice of the answer.
    """
    d = daily.sort_values("session").copy()
    rng = d["prior_rth_range_prior"]
    d["atr_t"] = rng.rolling(atr_n, min_periods=5).mean().shift(1)
    d["atr_pctile"] = (d["atr_t"].rolling(252, min_periods=60)
                       .rank(pct=True).shift(1))
    d["on_range_t"] = d["globex_range_pre"]
    d["on_range_rel_atr"] = d["on_range_t"] / d["atr_t"]
    d["gap_atr_prior"] = d["gap_t_prior"] / d["atr_t"]
    return d


def add_regimes(d: pd.DataFrame) -> pd.DataFrame:
    """Coarse, interpretable conditioning variables.

    Deliberately few and deliberately crude. Regime definitions multiply the
    hypothesis count fast, and a finding that only survives inside a
    finely-tuned regime is usually a finding about the regime definition.
    """
    d = d.copy()
    g = d["gap_atr_prior"]
    d["regime_gap"] = pd.cut(g, [-np.inf, -0.5, -0.15, 0.15, 0.5, np.inf],
                             labels=["gap_dn_big", "gap_dn", "flat", "gap_up", "gap_up_big"])
    d["regime_vol"] = pd.cut(d["atr_pctile"], [0, 0.25, 0.75, 1.0],
                             labels=["low", "mid", "high"], include_lowest=True)
    d["regime_on_range"] = pd.cut(d["on_range_rel_atr"], [0, 0.4, 0.8, np.inf],
                                  labels=["compressed", "normal", "expanded"])
    hi, lo = d["prior_rth_high_prior"], d["prior_rth_low_prior"]
    ref = d["ref_t"]
    d["open_vs_prior_range"] = np.where(ref > hi, "above",
                                        np.where(ref < lo, "below", "inside"))
    d["dow"] = pd.to_datetime(d["session"]).dt.dayofweek
    return d
