"""L3a: the classical, named pre-open reference levels.

This is the *baseline* candidate generator, not the answer. Its job is to give
the algorithmic price-axis model something to beat and to let each traditional
concept be measured on its own rather than assumed. Any family here can be
rejected on evidence; none of them is privileged.

Every level returned is a function of data strictly before the decision cut.
The builder takes already-windowed frames rather than reaching for data itself,
so the causal boundary lives in one place — :func:`build_candidates`, which
asserts it — instead of being re-implemented per family.

Families, grouped by what they are made of:

*prior session*   pdh, pdl, pdc, prior RTH open/mid/vwap, prior POC/VAH/VAL,
                  prior globex high/low
*current night*   on_high, on_low, on_mid, on_vwap, on_poc/vah/val, and the
                  London / Asia / premarket extremes separately, because they
                  form at different times against different participants
*longer horizon*  prior week and prior month high/low/close, N-session swings
*derived*         round numbers, gap edges, VWAP deviation bands, volume nodes
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..build.profile import find_nodes, poc, profile_sigma, profile_vwap, value_area
from ..build.tape import window_bars, window_profile


@dataclass(frozen=True)
class Level:
    """A single un-widened reference price with its provenance."""
    kind: str
    price_t: float
    family: str
    source: str = ""          # which window/session it came from
    weight: float = 1.0       # family-native strength (node prominence, etc.)

    def as_row(self) -> Dict:
        return {"kind": self.kind, "price_t": float(self.price_t),
                "family": self.family, "source": self.source, "weight": self.weight}


def _ohlc(bars: pd.DataFrame) -> Optional[Dict[str, float]]:
    """OHLC of a window, in ticks. Assumes ``bars`` is time-sorted."""
    if bars.empty:
        return None
    return {"open": float(bars["open_t"].iloc[0]), "high": float(bars["high_t"].max()),
            "low": float(bars["low_t"].min()), "close": float(bars["close_t"].iloc[-1])}


def _vwap_t(bars: pd.DataFrame, tick_size: float = 0.25) -> Optional[float]:
    v = int(bars["volume"].sum())
    if v <= 0:
        return None
    return float(bars["notional"].sum() / v / tick_size)


# --------------------------------------------------------------------------- #
# families
# --------------------------------------------------------------------------- #


def prior_session_levels(prior_rth: pd.DataFrame, prior_globex: pd.DataFrame,
                         prior_rth_profile: pd.DataFrame, tag: str = "pd") -> List[Level]:
    """Prior trade date: the levels almost every NQ trader has on the chart."""
    out: List[Level] = []
    o = _ohlc(prior_rth)
    if o:
        out += [Level(f"{tag}h", o["high"], "prior_session", tag),
                Level(f"{tag}l", o["low"], "prior_session", tag),
                Level(f"{tag}c", o["close"], "prior_session", tag),
                Level(f"{tag}o", o["open"], "prior_session", tag),
                Level(f"{tag}mid", (o["high"] + o["low"]) / 2, "prior_session", tag)]
        v = _vwap_t(prior_rth)
        if v is not None:
            out.append(Level(f"{tag}_vwap", v, "prior_session", tag))
    g = _ohlc(prior_globex)
    if g:
        out += [Level(f"{tag}_globex_high", g["high"], "prior_session", tag),
                Level(f"{tag}_globex_low", g["low"], "prior_session", tag)]
    if prior_rth_profile is not None and not prior_rth_profile.empty:
        val, p, vah = value_area(prior_rth_profile)
        if p is not None:
            out += [Level(f"{tag}_poc", p, "prior_profile", tag),
                    Level(f"{tag}_vah", vah, "prior_profile", tag),
                    Level(f"{tag}_val", val, "prior_profile", tag)]
    return out


def overnight_levels(segments: Dict[str, pd.DataFrame],
                     on_profile: pd.DataFrame) -> List[Level]:
    """Current session's own night, split by the desk that traded it."""
    out: List[Level] = []
    for name, bars in segments.items():
        o = _ohlc(bars)
        if not o:
            continue
        pfx = {"globex": "on", "asia": "asia", "london": "ldn",
               "premarket": "pre", "extended": "ext"}.get(name, name)
        out += [Level(f"{pfx}_high", o["high"], "overnight", name),
                Level(f"{pfx}_low", o["low"], "overnight", name)]
        if name == "globex":
            out.append(Level("on_mid", (o["high"] + o["low"]) / 2, "overnight", name))
            v = _vwap_t(bars)
            if v is not None:
                out.append(Level("on_vwap", v, "overnight", name))
    if on_profile is not None and not on_profile.empty:
        val, p, vah = value_area(on_profile)
        if p is not None:
            out += [Level("on_poc", p, "overnight_profile", "globex"),
                    Level("on_vah", vah, "overnight_profile", "globex"),
                    Level("on_val", val, "overnight_profile", "globex")]
    return out


def horizon_levels(week_bars: pd.DataFrame, month_bars: pd.DataFrame) -> List[Level]:
    """Prior calendar week and prior calendar month extremes."""
    out: List[Level] = []
    for bars, pfx in ((week_bars, "pw"), (month_bars, "pm")):
        o = _ohlc(bars)
        if o:
            out += [Level(f"{pfx}h", o["high"], "horizon", pfx),
                    Level(f"{pfx}l", o["low"], "horizon", pfx),
                    Level(f"{pfx}c", o["close"], "horizon", pfx)]
    return out


def round_numbers(ref_t: float, tick_size: float = 0.25,
                  steps: Sequence[int] = (25, 50, 100, 250, 500),
                  span_points: float = 400.0) -> List[Level]:
    """Psychological round handles within reach of the reference price.

    Emitted per step size so the research can find out whether 100s matter and
    25s do not, instead of collapsing them into one undifferentiated family.
    """
    out: List[Level] = []
    ref_px = ref_t * tick_size
    for step in steps:
        lo = int(np.floor((ref_px - span_points) / step)) * step
        hi = int(np.ceil((ref_px + span_points) / step)) * step
        for px in range(lo, hi + step, step):
            if abs(px - ref_px) <= span_points:
                out.append(Level(f"round_{step}", px / tick_size, "round", str(step)))
    return out


def gap_levels(prior_rth_close_t: Optional[float], pre_open_t: Optional[float]) -> List[Level]:
    """The two edges of the overnight gap, plus its midpoint.

    The gap is only defined once both ends exist; a missing prior close returns
    nothing rather than a level anchored to a guess.
    """
    if prior_rth_close_t is None or pre_open_t is None:
        return []
    return [Level("gap_prior_close", prior_rth_close_t, "gap", "pdc"),
            Level("gap_open_ref", pre_open_t, "gap", "pre"),
            Level("gap_mid", (prior_rth_close_t + pre_open_t) / 2, "gap", "mid")]


def vwap_bands(anchor_vwap_t: Optional[float], sigma_t: Optional[float],
               name: str, ks: Sequence[float] = (1.0, 1.5, 2.0, 2.5, 3.0)) -> List[Level]:
    """Volume-weighted deviation bands around an anchored VWAP."""
    if anchor_vwap_t is None or sigma_t is None or not np.isfinite(sigma_t) or sigma_t <= 0:
        return []
    out = [Level(f"{name}_vwap", anchor_vwap_t, "vwap_band", name)]
    for k in ks:
        out += [Level(f"{name}_vwap_p{k:g}", anchor_vwap_t + k * sigma_t, "vwap_band", name),
                Level(f"{name}_vwap_m{k:g}", anchor_vwap_t - k * sigma_t, "vwap_band", name)]
    return out


def node_levels(prof: pd.DataFrame, name: str, **kw) -> List[Level]:
    """HVN / LVN structure of a profile window."""
    return [Level(f"{name}_{n.kind}", n.price_t, "volume_node", name, weight=n.prominence)
            for n in find_nodes(prof, **kw)]


def swing_levels(bars: pd.DataFrame, fractal_n: int = 300,
                 max_levels: int = 12) -> List[Level]:
    """Swing highs/lows over a multi-session bar frame.

    A swing high is the maximum of a ``2*fractal_n+1``-second neighbourhood.
    Strict inequality on one side breaks ties, so a flat run of equal highs
    yields a single swing instead of one per second — the bug that turns an
    equal-highs cluster into a dozen duplicate levels.
    """
    if bars.empty or len(bars) < 2 * fractal_n + 1:
        return []
    hi = bars["high_t"].to_numpy().astype("float64")
    lo = bars["low_t"].to_numpy().astype("float64")
    rh = pd.Series(hi).rolling(2 * fractal_n + 1, center=True).max().to_numpy()
    rl = pd.Series(lo).rolling(2 * fractal_n + 1, center=True).min().to_numpy()
    out: List[Level] = []
    for arr, roll, kind in ((hi, rh, "swing_high"), (lo, rl, "swing_low")):
        # Walk newest-first and keep the most recent swing of each distinct
        # price: recency is the tie-break that matters, and it also caps an
        # equal-highs cluster at one level instead of one per second.
        seen: List[float] = []
        picked: List[Level] = []
        for i in np.flatnonzero(arr == roll)[::-1]:
            v = float(arr[i])
            if all(abs(v - s) > 4 for s in seen):
                seen.append(v)
                picked.append(Level(kind, v, "swing", f"n{fractal_n}@{int(i)}"))
            if len(picked) >= max_levels:
                break
        out.extend(picked)
    return out


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def dedupe_levels(levels: Sequence[Level], min_sep_t: int = 2) -> pd.DataFrame:
    """Collapse coincident levels into one row that remembers every family.

    Confluence is a hypothesis to *test*, not a bonus to hand out, so the
    output records ``n_families`` and the contributing kinds without scoring
    them. The confluence experiment reads those columns; nothing else should
    infer strength from them.
    """
    if not levels:
        return pd.DataFrame(columns=["price_t", "kinds", "families", "n_kinds",
                                     "n_families", "weight"])
    df = pd.DataFrame([l.as_row() for l in levels]).sort_values("price_t")
    cluster, groups = [], []
    for _, r in df.iterrows():
        if cluster and abs(r["price_t"] - cluster[-1]["price_t"]) <= min_sep_t:
            cluster.append(r)
        else:
            if cluster:
                groups.append(cluster)
            cluster = [r]
    if cluster:
        groups.append(cluster)
    rows = []
    for g in groups:
        gd = pd.DataFrame(g)
        rows.append({
            "price_t": float(np.average(gd["price_t"], weights=gd["weight"].clip(lower=1e-9))),
            "kinds": ",".join(sorted(set(gd["kind"]))),
            "families": ",".join(sorted(set(gd["family"]))),
            "n_kinds": int(gd["kind"].nunique()),
            "n_families": int(gd["family"].nunique()),
            "weight": float(gd["weight"].max()),
        })
    return pd.DataFrame(rows).sort_values("price_t").reset_index(drop=True)
