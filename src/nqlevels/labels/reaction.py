"""L4b: given that price reached the zone, what happened next?

The brief is explicit that "price came back 12 points" and "price wobbled
around the level for forty minutes and then drifted 12 points" must not score
the same. So reaction is not one number. It is measured on three independent
axes, and a model is free to learn that they matter in different proportions:

**Magnitude** — excursions from the touch reference in both directions.
``rev`` is the rejection direction (down for a zone approached from below, up
for one approached from above); ``cont`` is continuation. Reported as MFE/MAE
over several horizons plus a first-passage outcome, because a level that
usually gives 8 points and a level that occasionally gives 40 are different
instruments even at the same win rate.

**Cleanliness** — how directly the move happened. ``directional_efficiency``
is net displacement over path length: 1.0 is a straight line, values near 0 are
chop. Combined with seconds spent inside the zone, full crossings and
re-entries, this is the tap-and-go / chop discriminator.

**Persistence** — did the zone survive. ``penetration`` is how deep price got
past the near edge; ``acceptance`` is sustained trade beyond the far edge;
``reclaim`` is a return after penetration.

Everything is anchored at the **first-touch second** and measured forward. The
touch reference is the zone's near edge, not the touch price, because that edge
is where a mechanical order would rest and is known before the open — using the
realised touch price would quietly leak a tick of hindsight into the anchor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .touch import Zone

# Reaction sizes are volatility-scaled in the feature layer, but a fixed grid is
# also reported so results stay legible in the units a trader thinks in.
POINT_GRID = (5, 10, 15, 20, 30)
HORIZON_MIN = (1, 3, 5, 15, 30, 60)


def _pos(bars: pd.DataFrame, zone: Zone) -> np.ndarray:
    """-1 below the zone, 0 inside, +1 above, from each second's close."""
    c = bars["close_t"].to_numpy()
    return np.where(c > zone.hi_t, 1, np.where(c < zone.lo_t, -1, 0)).astype(np.int8)


def crossing_counts(pos: np.ndarray) -> Tuple[int, int]:
    """(full crossings, re-entries) from a position sequence.

    A *full crossing* is a move from strictly one side of the zone to strictly
    the other, ignoring however long it lingered inside — that is the event
    that stops out a fade. A *re-entry* is a return into the zone from either
    side. Zero-length runs are compressed first so a single jittering second
    cannot manufacture a dozen events.
    """
    if pos.size == 0:
        return 0, 0
    runs = pos[np.insert(np.diff(pos) != 0, 0, True)]
    outside = runs[runs != 0]
    crossings = int((np.diff(outside) != 0).sum()) if outside.size > 1 else 0
    reentries = int(((runs[1:] == 0) & (runs[:-1] != 0)).sum())
    return crossings, reentries


@dataclass
class ReactionOutcome:
    # anchor
    side: str
    ref_t: int                       # near edge = measurement anchor, ticks
    n_seconds: int                   # seconds observed after first touch

    # magnitude (ticks, from ref, positive in each named direction)
    mfe_rev: Dict[int, float] = field(default_factory=dict)   # horizon_min -> ticks
    mfe_cont: Dict[int, float] = field(default_factory=dict)
    time_to_mfe_rev_s: Optional[float] = None
    first_passage: Dict[str, Optional[int]] = field(default_factory=dict)

    # cleanliness, all measured over the first `clean_window_s` seconds
    clean_window_s: int = 0
    secs_inside: int = 0
    secs_inside_share: float = 0.0
    n_crossings: int = 0
    n_reentries: int = 0
    directional_efficiency: float = 0.0
    efficiency_step_s: int = 0
    rev_velocity_tps: float = 0.0    # ticks per second over the first minute

    # persistence
    max_penetration_t: float = 0.0   # past the near edge, ticks (>=0)
    traversed: bool = False          # cleared the far edge at all
    accepted: bool = False           # sustained beyond the far edge
    secs_beyond_far: int = 0
    reclaimed: Optional[bool] = None
    secs_to_reclaim: Optional[float] = None

    def as_row(self) -> Dict:
        d = asdict(self)
        for k in ("mfe_rev", "mfe_cont"):
            for h, v in d.pop(k).items():
                d[f"{k}_{h}m"] = v
        for k, v in d.pop("first_passage").items():
            d[f"fp_{k}"] = v
        return d


def evaluate_reaction(bars: pd.DataFrame, zone: Zone, side: str,
                      first_touch_idx: int,
                      tick_size: float = 0.25,
                      accept_seconds: int = 300,
                      clean_window_s: int = 900,
                      efficiency_sample_s: int = 30,
                      point_grid: Sequence[int] = POINT_GRID,
                      horizons_min: Sequence[int] = HORIZON_MIN) -> ReactionOutcome:
    """Measure what happened after the first touch.

    ``bars`` is the target-window, front-contract 1-second frame; ``side`` is
    the approach side from :func:`~nqlevels.labels.touch.zone_side`. The
    forward window is truncated at the end of the target window, so a touch at
    08:55 is scored on the five minutes that actually existed and
    ``n_seconds`` records it — a short window is a data fact, not a weak
    reaction, and models must be able to tell them apart.

    Excursions are measured **from the zone edges, not from a single price**.
    Movement inside the zone is neither reaction nor failure: the zone is by
    construction the band where price counts as "at the level". So rejection is
    distance back past the near edge, continuation is distance beyond the far
    edge, and both are zero while price is inside. Anchoring adverse movement
    at the near edge instead — the obvious first implementation — scores a
    zone as failed the instant a fast touch prints through it, which would
    have condemned every wide zone before the width study even ran.
    """
    if side == "inside":
        side = "above"                      # measure an open-inside zone as resistance
    fwd = bars.iloc[first_touch_idx:]
    n = len(fwd)
    near, far = zone.near_edge_t(side), zone.far_edge_t(side)
    out = ReactionOutcome(side=side, ref_t=int(near), n_seconds=n)
    if n == 0:
        return out

    hi = fwd["high_t"].to_numpy().astype("float64")
    lo = fwd["low_t"].to_numpy().astype("float64")
    cl = fwd["close_t"].to_numpy().astype("float64")

    # side 'above' = resistance met from below: reject DOWN past lo, break UP past hi.
    # side 'below' = support met from above: the mirror image.
    if side == "above":
        rev_ex, cont_ex = near - lo, hi - far
    else:
        rev_ex, cont_ex = hi - near, far - lo

    run_rev = np.maximum.accumulate(np.maximum(rev_ex, 0.0))
    run_cont = np.maximum.accumulate(np.maximum(cont_ex, 0.0))
    for h in horizons_min:
        k = min(n, h * 60)
        out.mfe_rev[h] = float(run_rev[k - 1])
        out.mfe_cont[h] = float(run_cont[k - 1])
    out.time_to_mfe_rev_s = float(np.argmax(rev_ex)) if run_rev[-1] > 0 else None

    # first passage: does a k-point rejection arrive before an m-point adverse move?
    ticks_per_point = 1.0 / tick_size
    for k in point_grid:
        kt = k * ticks_per_point
        i_rev = np.argmax(rev_ex >= kt) if (rev_ex >= kt).any() else None
        for m in point_grid:
            mt = m * ticks_per_point
            i_adv = np.argmax(cont_ex >= mt) if (cont_ex >= mt).any() else None
            if i_rev is None:
                v = 0 if i_adv is not None else None      # neither -> unresolved
            elif i_adv is None:
                v = 1
            else:
                v = int(i_rev < i_adv)
            out.first_passage[f"rev{k}_adv{m}"] = v

    # ---- cleanliness, over a FIXED window so touches are comparable --------
    # A touch at 06:35 has 145 minutes of forward data and one at 08:55 has 5;
    # measuring "time spent loitering" over whatever remains would make the
    # clock, not the level, the dominant explanatory variable.
    k = min(n, clean_window_s)
    out.clean_window_s = int(k)
    win = fwd.iloc[:k]
    pos = _pos(win, zone)
    out.secs_inside = int((pos == 0).sum())
    out.secs_inside_share = out.secs_inside / k
    out.n_crossings, out.n_reentries = crossing_counts(pos)

    # Path length is sampled every `efficiency_sample_s` seconds. At 1-second
    # resolution the path is dominated by bid-ask jitter, so the ratio collapses
    # toward zero for every level and stops discriminating at all — the demo run
    # showed 0.00-0.03 for both clean impulses and pure chop.
    #
    # The step also has to shrink for short windows: sampling a 40-second
    # window every 30 s leaves two points, and any two points are a straight
    # line, so a late touch would score a perfect 1.0 no matter how it moved.
    # Below MIN_EFF_SAMPLES points the metric is not defined at all.
    MIN_EFF_SAMPLES = 8
    cw = cl[:k]
    if k >= MIN_EFF_SAMPLES:
        step = max(1, min(efficiency_sample_s, k // MIN_EFF_SAMPLES))
        samp = cw[::step]
        path = float(np.abs(np.diff(samp)).sum())
        out.directional_efficiency = (float(abs(samp[-1] - samp[0]) / path)
                                      if path > 0 else 0.0)
        out.efficiency_step_s = int(step)
    else:
        out.directional_efficiency = float("nan")
    k60 = min(n, 60)
    out.rev_velocity_tps = float(run_rev[k60 - 1] / k60)

    # persistence — depth is measured past the NEAR edge (where price entered),
    # so a zone that was merely tapped scores near 0 and one that was blown
    # through scores more than its own width.
    near, far = zone.near_edge_t(side), zone.far_edge_t(side)
    depth = (hi - near) if side == "above" else (near - lo)
    beyond = (hi - far) if side == "above" else (far - lo)
    out.max_penetration_t = float(max(0.0, np.max(depth)))
    out.traversed = bool((beyond > 0).any())
    cl_beyond = (cl > far) if side == "above" else (cl < far)
    out.secs_beyond_far = int(cl_beyond.sum())
    out.accepted = bool(_longest_run(cl_beyond) >= accept_seconds)

    if out.traversed:
        j = int(np.argmax(beyond > 0))
        back = (cl[j:] < zone.lo_t) if side == "above" else (cl[j:] > zone.hi_t)
        out.reclaimed = bool(back.any())
        out.secs_to_reclaim = float(np.argmax(back)) if back.any() else None
    return out


def _longest_run(mask: np.ndarray) -> int:
    if mask.size == 0 or not mask.any():
        return 0
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return int(np.max(idx[1::2] - idx[0::2]))


# --------------------------------------------------------------------------- #
# derived summaries
# --------------------------------------------------------------------------- #


def clean_rejection(r: ReactionOutcome, points: int = 10, adverse: int = 5,
                    max_inside_share: float = 0.5, max_crossings: int = 1) -> Optional[bool]:
    """The project's headline "did it reject cleanly" label.

    Deliberately conjunctive: a reaction counts only if it produced a real move
    (``points`` before ``adverse``) *and* did not do it by loitering
    (``max_inside_share``, ``max_crossings``). Returns ``None`` when the first
    passage never resolved inside the observed window, so unresolved cases are
    dropped rather than silently scored as failures.

    The thresholds are arguments, not constants, because the width study and
    the outcome-surface study will move them; nothing downstream should assume
    these defaults are the answer.
    """
    fp = r.first_passage.get(f"rev{points}_adv{adverse}")
    if fp is None:
        return None
    return bool(fp == 1
                and r.secs_inside_share <= max_inside_share
                and r.n_crossings <= max_crossings)


def chop_score(r: ReactionOutcome) -> float:
    """0 = clean impulse away from the zone, 1 = pure chop.

    A blunt composite for ranking and for eyeballing; models get the raw
    components instead.
    """
    a = min(1.0, r.secs_inside_share / 0.5)
    b = min(1.0, r.n_crossings / 3.0)
    c = 1.0 - min(1.0, r.directional_efficiency / 0.5)
    return float((a + b + c) / 3.0)
