"""How wide should a zone be?

A level is a price; a zone is an interval, and its width is a modelling choice
with real consequences. Too narrow and genuine reactions are recorded as
misses; too wide and every zone gets touched every day and the touch signal
dissolves into noise. "Five points, because that looks about right" is exactly
the arbitrary constant this project is meant to avoid, so width is a family of
estimators to be compared, with :class:`FixedTicks` present only as the control
the others must beat.

The measurement that decides it is the **empirical penetration distribution**:
for a given level type, how far past the reference price does the market
actually go on a visit? :func:`fit_penetration_width` reads that off training
sessions per level family, so each family gets the width its own history
justifies, and :func:`width_selection_report` is what the width experiment
reports — hit rate, reaction quality and false-touch rate as functions of
width, rather than a single chosen number with no error bars.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


class WidthModel(ABC):
    """Maps a session's context to a zone half-width in ticks."""

    name: str = "base"

    @abstractmethod
    def half_width_t(self, ctx: pd.Series) -> float:
        ...

    def zone_edges(self, price_t: float, ctx: pd.Series) -> tuple[int, int]:
        w = max(1.0, self.half_width_t(ctx))
        return int(np.floor(price_t - w)), int(np.ceil(price_t + w))


@dataclass
class FixedTicks(WidthModel):
    """Control: the same width every day, ignoring volatility entirely."""
    ticks: float = 6.0
    name: str = "fixed"

    def half_width_t(self, ctx: pd.Series) -> float:
        return float(self.ticks)


@dataclass
class AtrFraction(WidthModel):
    """Half-width as a fraction of prior-session ATR — the obvious scaling."""
    k: float = 0.05
    floor_t: float = 2.0
    cap_t: float = 40.0
    name: str = "atr"

    def half_width_t(self, ctx: pd.Series) -> float:
        a = float(ctx.get("atr_t", np.nan))
        if not np.isfinite(a):
            return self.floor_t
        return float(np.clip(self.k * a, self.floor_t, self.cap_t))


@dataclass
class OvernightRangeFraction(WidthModel):
    """Scaled by the night just traded rather than the last fortnight.

    Faster-adapting than ATR: an overnight that has already gone 300 points
    says more about this morning than the trailing average does.
    """
    k: float = 0.06
    floor_t: float = 2.0
    cap_t: float = 40.0
    name: str = "on_range"

    def half_width_t(self, ctx: pd.Series) -> float:
        r = float(ctx.get("on_range_t", np.nan))
        if not np.isfinite(r):
            return self.floor_t
        return float(np.clip(self.k * r, self.floor_t, self.cap_t))


@dataclass
class EmpiricalPenetration(WidthModel):
    """Per-family width read off historical overshoot.

    ``table`` maps a level family to the chosen quantile of its penetration
    distribution, fitted on training sessions only. This is the estimator that
    actually answers the question, because it defines the zone as "the region
    the market treats as the same price" rather than as a round number of
    points.
    """
    table: Dict[str, float]
    default_t: float = 6.0
    scale_by_atr: bool = True
    name: str = "penetration"

    def half_width_t(self, ctx: pd.Series) -> float:
        base = float(self.table.get(str(ctx.get("kind", "")), self.default_t))
        if not self.scale_by_atr:
            return base
        a, ref = float(ctx.get("atr_t", np.nan)), float(ctx.get("atr_ref_t", np.nan))
        if np.isfinite(a) and np.isfinite(ref) and ref > 0:
            return base * a / ref
        return base


def fit_penetration_width(touches: pd.DataFrame, quantile: float = 0.6,
                          kind_col: str = "kind",
                          pen_col: str = "max_penetration_t",
                          min_n: int = 100) -> Dict[str, float]:
    """Half-width per family as a quantile of observed penetration.

    Fit on **training sessions only** — the penetration distribution is an
    outcome, and fitting it on data a model is later scored against is a
    textbook leak. Families with fewer than ``min_n`` observations are omitted
    so the caller falls back to a default rather than trusting a width
    estimated from a handful of touches.
    """
    out: Dict[str, float] = {}
    for kind, g in touches.groupby(kind_col):
        v = pd.to_numeric(g[pen_col], errors="coerce").dropna()
        if len(v) >= min_n:
            out[str(kind)] = float(np.quantile(v, quantile))
    return out


def width_selection_report(labelled: pd.DataFrame, width_col: str = "half_width_t",
                           touched_col: str = "touched",
                           clean_col: str = "clean_reject",
                           bins: Sequence[float] = (2, 4, 6, 8, 12, 16, 24, 40)
                           ) -> pd.DataFrame:
    """Hit rate and reaction quality as functions of width.

    The point of reporting the curve rather than an argmax is that the right
    width is the one where reaction quality is still high *and* the curve is
    flat around it. A sharp optimum is a fitted artifact; a plateau is a
    property of the market.
    """
    d = labelled.copy()
    d["_b"] = pd.cut(d[width_col], list(bins))
    rows = []
    for b, g in d.groupby("_b", observed=True):
        t = g[touched_col].mean() if touched_col in g else np.nan
        hit = g[g[touched_col] == 1] if touched_col in g else g
        rows.append({
            "half_width_t": str(b), "n": len(g), "n_sessions": g["session"].nunique(),
            "touch_rate": float(t),
            "clean_rate": float(hit[clean_col].mean()) if clean_col in hit and len(hit) else np.nan,
            "n_touched": int(len(hit)),
        })
    return pd.DataFrame(rows)
