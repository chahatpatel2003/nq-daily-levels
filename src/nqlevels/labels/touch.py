"""L4a: did price reach the zone, and when?

A *zone* is a closed price interval ``[lo, hi]``. A *touch* is any second whose
bar range intersects it — ``bar.low <= hi and bar.high >= lo``. Using the bar
range rather than the close matters: a wick through a level is a real
interaction that a resting order would have been filled by, and requiring a
close inside would systematically under-count exactly the fast rejections this
project is looking for.

Three cases are kept apart because they are different questions:

``open_inside``
    The 06:30 print is already in the zone. There is no "arrival", so touch
    probability is meaningless and these rows are excluded from touch models
    (they remain valid for reaction models, flagged).
``above`` / ``below``
    The zone sits above or below the pre-open reference price. Distance to
    travel is the dominant driver of touch probability, so every touch
    statistic in this project is reported *against a distance-matched
    baseline*; a zone that is touched 70 % of the time because it is 5 points
    away has demonstrated nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Zone:
    """A candidate price zone, fully determined before the decision cut."""
    lo_t: int                 # inclusive lower edge, price ticks
    hi_t: int                 # inclusive upper edge, price ticks
    kind: str                 # level family, e.g. 'pdh', 'on_low', 'axis'
    ref_t: Optional[int] = None   # the un-widened level, if the family has one

    def __post_init__(self) -> None:
        if self.hi_t < self.lo_t:
            raise ValueError(f"zone edges inverted: {self.lo_t} > {self.hi_t}")

    @property
    def mid_t(self) -> float:
        return (self.lo_t + self.hi_t) / 2.0

    @property
    def width_t(self) -> int:
        return self.hi_t - self.lo_t

    def near_edge_t(self, side: str) -> int:
        """The edge price meets first when approaching from ``side``."""
        return self.lo_t if side == "above" else self.hi_t

    def far_edge_t(self, side: str) -> int:
        return self.hi_t if side == "above" else self.lo_t


def zone_side(zone: Zone, ref_t: float) -> str:
    """Where the zone sits relative to the pre-open reference price."""
    if zone.lo_t <= ref_t <= zone.hi_t:
        return "inside"
    return "above" if zone.lo_t > ref_t else "below"


def touch_mask(bars: pd.DataFrame, zone: Zone) -> np.ndarray:
    return ((bars["low_t"].to_numpy() <= zone.hi_t)
            & (bars["high_t"].to_numpy() >= zone.lo_t))


@dataclass
class TouchOutcome:
    touched: bool
    first_touch_idx: Optional[int]
    first_touch_ts: Optional[pd.Timestamp]
    minutes_to_touch: Optional[float]
    side: str
    distance_t: float                  # ticks from reference to the near edge
    open_inside: bool
    n_touch_seconds: int               # total seconds intersecting the zone
    touched_by: Dict[int, bool]        # horizon (minutes) -> touched by then
    max_approach_t: Optional[float]    # closest approach if never touched


def evaluate_touch(bars: pd.DataFrame, zone: Zone, ref_t: float,
                   window_start: pd.Timestamp,
                   horizons_min: Sequence[int] = (15, 30, 60, 90, 150)) -> TouchOutcome:
    """First-touch outcome for one zone over one target window.

    ``bars`` must already be restricted to the target window and to the front
    contract, sorted by time. ``ref_t`` is the reference price in ticks — by
    convention the last trade strictly before the decision cut, so distance is
    measured from what a live run would have seen.
    """
    side = zone_side(zone, ref_t)
    open_inside = side == "inside"
    near = zone.near_edge_t(side) if not open_inside else zone.mid_t
    distance_t = abs(near - ref_t)

    if bars.empty:
        return TouchOutcome(False, None, None, None, side, distance_t, open_inside,
                            0, {h: False for h in horizons_min}, None)

    m = touch_mask(bars, zone)
    ts = pd.DatetimeIndex(bars["ts"])
    n_touch = int(m.sum())

    if not m.any():
        gap = np.where(bars["low_t"].to_numpy() > zone.hi_t,
                       bars["low_t"].to_numpy() - zone.hi_t,
                       zone.lo_t - bars["high_t"].to_numpy())
        return TouchOutcome(False, None, None, None, side, distance_t, open_inside,
                            0, {h: False for h in horizons_min}, float(np.min(gap)))

    i = int(np.argmax(m))
    t0 = ts[i]
    mins = (t0 - window_start).total_seconds() / 60.0
    return TouchOutcome(True, i, t0, mins, side, distance_t, open_inside, n_touch,
                        {h: bool(mins <= h) for h in horizons_min}, 0.0)


def first_touch_order(bars: pd.DataFrame, zones: Sequence[Zone]) -> List[int]:
    """Indices of ``zones`` ordered by first touch; untouched zones dropped.

    Needed for "reached before the competing zone" labels: with two zones on
    opposite sides of the open, which one price visits first is often the whole
    trade, and it is not recoverable from the two touch probabilities alone.
    """
    hits = []
    for k, z in enumerate(zones):
        m = touch_mask(bars, z)
        if m.any():
            hits.append((int(np.argmax(m)), k))
    return [k for _, k in sorted(hits)]
