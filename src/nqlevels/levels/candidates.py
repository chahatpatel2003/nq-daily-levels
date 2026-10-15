"""L3: assemble one session's candidate zones — and own the causal boundary.

Every other module in the levels layer takes already-windowed frames and asks
no questions. This one does the windowing, and it is the single place where the
06:25 PT cut is applied and asserted. If a leak ever gets into a pre-open
feature, it comes through here, which is why the assertion is unconditional
rather than a debug flag.

The order of operations matters:

1. read L1 for the session and a lookback of prior sessions;
2. pick the front contract from the roll log — *not* from this session's own
   volume, which is not known at 06:25;
3. cut everything at the decision instant and assert it;
4. build named levels and the price-axis grid from the cut frames;
5. widen into zones using a width model that also only sees pre-cut context.

Prior-session levels are taken from the **incoming contract's own prints**
whenever it traded enough on that session, so a roll does not shift a level by
the carry spread. When it did not, the level is carried by the measured spread
and tagged; when neither is possible the level is dropped rather than guessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..build.profile import profile_sigma, profile_vwap
from ..build.roll import carry_level, roll_spread
from ..build.tape import window_bars, window_profile
from ..clock import (Calendar, assert_before_decision, decision_time_utc,
                     prior_rth_window_utc, prior_trade_date,
                     rth_window_utc, segment_windows_utc, session_open_utc)
from ..labels.touch import Zone
from . import named as N
from .axis import AxisSpec, build_axis_frame
from .width import WidthModel


@dataclass
class SessionInputs:
    """Everything the generator is allowed to see, already cut."""
    session: date
    symbol: str
    ref_t: float
    cut: pd.Timestamp
    pre_bars: pd.DataFrame
    pre_profile: pd.DataFrame
    segments: Dict[str, pd.DataFrame]
    prior_rth_bars: pd.DataFrame
    prior_globex_bars: pd.DataFrame
    prior_rth_profile: pd.DataFrame
    lookback_bars: pd.DataFrame
    lookback_profiles: Dict[str, pd.DataFrame]
    carry: Dict[str, str] = field(default_factory=dict)


def _front_bars(bars: pd.DataFrame, sess: date, symbol: str) -> pd.DataFrame:
    b = bars[(bars["session"] == sess) & (bars["symbol"] == symbol)]
    return b.sort_values("ts", kind="stable")


def gather_inputs(sess: date, symbol: str, cal: Calendar, bars: pd.DataFrame,
                  profiles: pd.DataFrame, lookback: int = 20,
                  prior_symbol: Optional[str] = None) -> Optional[SessionInputs]:
    """Window L1 down to what a 06:25 run would have had.

    ``symbol`` is the front contract for ``sess`` as chosen from prior-session
    volume. ``prior_symbol`` is the front contract of the previous session; when
    they differ the session is roll-adjacent and prior-day levels are handled
    accordingly.
    """
    cut = pd.Timestamp(decision_time_utc(sess, cal))
    cur = _front_bars(bars, sess, symbol)
    if cur.empty:
        return None
    pre = cur[cur["ts"] < cut]
    if pre.empty:
        return None
    assert_before_decision(pre["ts"], sess, cal, "pre-open bars")

    prof_s = profiles[(profiles["session"] == sess) & (profiles["symbol"] == symbol)]
    cut30 = cut.floor("30min")
    pre_prof = window_profile(prof_s, pd.Timestamp(session_open_utc(sess, cal)).floor("30min"), cut30) \
        if not prof_s.empty else pd.DataFrame()

    segs = {}
    for name, (lo, hi) in segment_windows_utc(sess, cal).items():
        segs[name] = window_bars(pre, pd.Timestamp(lo), pd.Timestamp(hi))

    # ---- prior session, preferring the incoming contract's own prints -------
    pd_ = prior_trade_date(sess, cal)
    carry: Dict[str, str] = {}
    prior_sym = symbol
    prior_bars_all = bars[bars["session"] == pd_]
    if symbol not in set(prior_bars_all["symbol"]):
        prior_sym = prior_symbol or symbol
        carry["prior_session"] = "outgoing_contract"
    else:
        n_own = int(prior_bars_all[prior_bars_all["symbol"] == symbol]["volume"].sum())
        n_tot = int(prior_bars_all["volume"].sum())
        if n_tot > 0 and n_own / n_tot < 0.02:
            prior_sym = prior_symbol or symbol
            carry["prior_session"] = "outgoing_contract_thin_incoming"
        else:
            carry["prior_session"] = ("same_contract" if prior_sym == (prior_symbol or symbol)
                                      else "incoming_contract_own_prints")
    pb = _front_bars(bars, pd_, prior_sym)
    r_lo, r_hi = prior_rth_window_utc(sess, cal)
    prior_rth = window_bars(pb, pd.Timestamp(r_lo), pd.Timestamp(r_hi))
    prior_prof_s = profiles[(profiles["session"] == pd_) & (profiles["symbol"] == prior_sym)]
    prior_rth_prof = (window_profile(prior_prof_s, pd.Timestamp(r_lo), pd.Timestamp(r_hi))
                      if not prior_prof_s.empty else pd.DataFrame())

    # ---- multi-session lookback (front contract of each session) ------------
    lb_sessions = [pd_]
    d = pd_
    for _ in range(lookback - 1):
        d = prior_trade_date(d, cal)
        lb_sessions.append(d)
    lb = bars[bars["session"].isin(lb_sessions) & bars["is_front"]]
    lb = lb[lb["ts"] < cut].sort_values("ts", kind="stable")
    lp = profiles[profiles["session"].isin(lb_sessions) & profiles["is_front"]]

    lookback_profiles = {}
    for tag, k in (("lb5", 5), ("lb20", lookback)):
        keep = set(lb_sessions[:k])
        sel = lp[lp["session"].isin(keep)]
        lookback_profiles[tag] = (
            sel.groupby("price_t", as_index=False)[["volume", "n_trades", "buy_vol", "sell_vol"]]
               .sum().sort_values("price_t").reset_index(drop=True)
            if not sel.empty else pd.DataFrame())

    return SessionInputs(
        session=sess, symbol=symbol, ref_t=float(pre["close_t"].iloc[-1]), cut=cut,
        pre_bars=pre, pre_profile=pre_prof, segments=segs,
        prior_rth_bars=prior_rth, prior_globex_bars=pb, prior_rth_profile=prior_rth_prof,
        lookback_bars=lb, lookback_profiles=lookback_profiles, carry=carry)


def build_named(si: SessionInputs, week_bars: Optional[pd.DataFrame] = None,
                month_bars: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """All classical families for one session, deduplicated into clusters."""
    levels: List[N.Level] = []
    levels += N.prior_session_levels(si.prior_rth_bars, si.prior_globex_bars,
                                     si.prior_rth_profile)
    levels += N.overnight_levels(si.segments, si.pre_profile)
    if week_bars is not None or month_bars is not None:
        levels += N.horizon_levels(week_bars if week_bars is not None else pd.DataFrame(),
                                   month_bars if month_bars is not None else pd.DataFrame())
    levels += N.round_numbers(si.ref_t)
    prior_close = (float(si.prior_rth_bars["close_t"].iloc[-1])
                   if not si.prior_rth_bars.empty else None)
    levels += N.gap_levels(prior_close, si.ref_t)
    for tag, prof in (("on", si.pre_profile), ("pd", si.prior_rth_profile),
                      *si.lookback_profiles.items()):
        if prof is not None and not prof.empty:
            levels += N.node_levels(prof, tag)
            levels += N.vwap_bands(profile_vwap(prof), profile_sigma(prof), tag)
    levels += N.swing_levels(si.lookback_bars)
    return N.dedupe_levels(levels)


def build_candidates(si: SessionInputs, width: WidthModel, ctx: pd.Series,
                     axis: Optional[AxisSpec] = None,
                     max_named_distance_atr: float = 3.0) -> pd.DataFrame:
    """Named levels -> zones, plus the price-axis grid if requested.

    Levels further than ``max_named_distance_atr`` from the reference are
    dropped: they are unreachable in a 2.5-hour window and would pad the
    sample with rows whose label is a foregone "no touch", flattering any
    touch model that learns nothing but distance.
    """
    nm = build_named(si)
    if nm.empty:
        return pd.DataFrame()
    atr = float(ctx.get("atr_t", np.nan))
    d = np.abs(nm["price_t"] - si.ref_t)
    if np.isfinite(atr) and atr > 0:
        nm = nm[d <= max_named_distance_atr * atr]
    if nm.empty:
        return pd.DataFrame()

    rows = []
    for _, r in nm.iterrows():
        c = ctx.copy()
        c["kind"] = r["kinds"].split(",")[0]
        lo, hi = width.zone_edges(r["price_t"], c)
        rows.append({
            "session": si.session, "symbol": si.symbol, "generator": "named",
            "price_t": r["price_t"], "lo_t": lo, "hi_t": hi,
            "half_width_t": (hi - lo) / 2.0, "width_model": width.name,
            "kinds": r["kinds"], "families": r["families"],
            "n_kinds": r["n_kinds"], "n_families": r["n_families"],
            "ref_t": si.ref_t, "dist_t": abs(r["price_t"] - si.ref_t),
            "dist_atr": abs(r["price_t"] - si.ref_t) / atr if np.isfinite(atr) and atr > 0 else np.nan,
            "carry": si.carry.get("prior_session", ""),
        })
    out = pd.DataFrame(rows)

    if axis is not None:
        af = build_axis_frame(
            si.ref_t, atr if np.isfinite(atr) and atr > 0 else 1.0, axis,
            profiles={"on": si.pre_profile, "pd": si.prior_rth_profile,
                      **si.lookback_profiles},
            bar_windows={"on": si.pre_bars, "lb": si.lookback_bars},
            named=nm)
        # The axis frame is ~120 columns wide; adding the three key columns with
        # insert() would fragment it once per session and dominate build time.
        keys = pd.DataFrame({"session": si.session, "symbol": si.symbol,
                             "generator": "axis"}, index=af.index)
        out = pd.concat([out, pd.concat([keys, af], axis=1)], ignore_index=True)
    return out


def zones_from(candidates: pd.DataFrame) -> List[Zone]:
    return [Zone(int(r["lo_t"]), int(r["hi_t"]), str(r.get("kinds", "axis")),
                 int(r["price_t"]))
            for _, r in candidates.iterrows()
            if np.isfinite(r.get("lo_t", np.nan)) and np.isfinite(r.get("hi_t", np.nan))]
