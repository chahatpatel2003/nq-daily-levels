"""Fast, vectorised measurement of what a zone did during the target window.

Everything operates on offsets from the session's 06:25 reference, in ticks, so
a zone is just a pair of integers and no per-session price bookkeeping is
needed.

Two structural shortcuts make a full-corpus width sweep take seconds:

* **Touch is a binary search.** The running maximum of ``hi`` is non-decreasing
  by construction, so the first second a zone above the reference is reached is
  ``searchsorted(run_max_hi, zone_lo)`` — O(log n), not a scan. The mirror holds
  below. This is the same monotonicity that makes touch probability a survival
  function in E1.
* **Horizon excursions are table lookups.** Rolling and suffix extrema of
  ``hi``/``lo`` are precomputed once per session, so the maximum favourable and
  adverse move over any supported horizon is an array index per zone.

Only the cleanliness metrics and first-passage ordering need an actual slice,
and those are bounded to the clean window.

Nothing here decides anything about trades. These are measurements of a price
area: how far past its edge price went, how long it stayed, how many times it
crossed. Reading them as entries or exits is a category error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HORIZONS_S: Tuple[int, ...] = (60, 180, 300, 900, 1800, 3600)
CLEAN_WINDOW_S: int = 900
EFF_SAMPLE_S: int = 30
POINT_GRID: Tuple[int, ...] = (5, 10, 15, 20, 30)
TICKS_PER_POINT = 4


def _rolling_max(a: np.ndarray, w: int) -> np.ndarray:
    """``out[i] = max(a[i : i+w])``, truncated at the end of the array.

    The last ``w-1`` positions have fewer than ``w`` elements left, so they take
    the suffix maximum — which is what "the most price reached in the next 5
    minutes" should mean when only 2 minutes of window remain.
    """
    n = a.size
    if n == 0:
        return a.copy()
    w = int(min(max(1, w), n))
    trailing = pd.Series(a).rolling(w, min_periods=1).max().to_numpy()
    out = np.empty(n, dtype="float64")
    out[: n - w + 1] = trailing[w - 1:]
    if w > 1:
        out[n - w + 1:] = np.maximum.accumulate(a[::-1])[::-1][n - w + 1:]
    return out


def _rolling_min(a: np.ndarray, w: int) -> np.ndarray:
    return -_rolling_max(-a, w)


@dataclass
class SessionPath:
    """One session's 06:30-09:00 PT path, with precomputed extremum tables."""
    session: object
    sec: np.ndarray
    hi: np.ndarray
    lo: np.ndarray
    cl: np.ndarray
    run_max_hi: np.ndarray = field(init=False)
    run_min_lo: np.ndarray = field(init=False)
    suf_max_hi: np.ndarray = field(init=False)
    suf_min_lo: np.ndarray = field(init=False)
    roll_max: Dict[int, np.ndarray] = field(init=False, default_factory=dict)
    roll_min: Dict[int, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.run_max_hi = np.maximum.accumulate(self.hi)
        self.run_min_lo = np.minimum.accumulate(self.lo)
        self.suf_max_hi = np.maximum.accumulate(self.hi[::-1])[::-1]
        self.suf_min_lo = np.minimum.accumulate(self.lo[::-1])[::-1]
        for h in HORIZONS_S:
            w = max(1, int(round(h * len(self.sec) / max(1, self.sec[-1] + 1))))
            self.roll_max[h] = _rolling_max(self.hi, w)
            self.roll_min[h] = _rolling_min(self.lo, w)

    @property
    def n(self) -> int:
        return self.sec.size

    # -- touch ---------------------------------------------------------- #
    def first_touch(self, zlo: int, zhi: int) -> Optional[int]:
        """Index of the first second whose bar reaches the zone, or ``None``.

        A second that jumps clean across the zone counts as a touch: price
        traded through that area, which is exactly the interaction being
        measured.
        """
        if zlo > 0:                                   # zone above the reference
            i = int(np.searchsorted(self.run_max_hi, zlo, side="left"))
        elif zhi < 0:                                 # zone below
            i = int(np.searchsorted(-self.run_min_lo, -zhi, side="left"))
        else:                                         # straddles the reference
            a = int(np.searchsorted(self.run_max_hi, zlo, side="left"))
            b = int(np.searchsorted(-self.run_min_lo, -zhi, side="left"))
            i = min(a, b)
        return i if i < self.n else None

    def side_of(self, zlo: int, zhi: int) -> str:
        return "above" if zlo > 0 else ("below" if zhi < 0 else "inside")


def build_paths(paths: pd.DataFrame) -> Dict[object, SessionPath]:
    """Group the path layer into per-session objects, once."""
    out: Dict[object, SessionPath] = {}
    for s, g in paths.groupby("session", sort=False):
        g = g.sort_values("sec")
        out[s] = SessionPath(s, g["sec"].to_numpy("int32"),
                             g["hi"].to_numpy("float64"),
                             g["lo"].to_numpy("float64"),
                             g["cl"].to_numpy("float64"))
    return out


# --------------------------------------------------------------------------- #
# zone measurement
# --------------------------------------------------------------------------- #


def measure_zone(p: SessionPath, zlo: float, zhi: float,
                 horizons_s: Sequence[int] = HORIZONS_S,
                 clean_window_s: int = CLEAN_WINDOW_S,
                 point_grid: Sequence[int] = POINT_GRID) -> Dict:
    """Every destination and reaction measurement for one zone.

    Excursions are measured **from the zone edges**: rejection is distance back
    past the near edge, continuation is distance beyond the far edge, and both
    are zero while price is inside. Movement inside a zone is neither reaction
    nor failure — the zone is by construction the band where price counts as
    "at the level".
    """
    zlo, zhi = float(zlo), float(zhi)
    side = p.side_of(zlo, zhi)
    out: Dict = {"side": side, "zlo_t": zlo, "zhi_t": zhi,
                 "half_width_t": (zhi - zlo) / 2.0}

    i = p.first_touch(zlo, zhi)
    out["touched"] = i is not None
    if i is None:
        gap = min(abs(zlo - p.run_max_hi[-1]), abs(p.run_min_lo[-1] - zhi))
        out["closest_approach_t"] = float(gap)
        return out

    out["first_touch_sec"] = int(p.sec[i])
    out["first_touch_min"] = float(p.sec[i]) / 60.0
    n_fwd = p.n - i
    out["n_seconds_fwd"] = int(n_fwd)

    near, far = (zlo, zhi) if side != "below" else (zhi, zlo)
    sgn = 1.0 if side != "below" else -1.0        # +1: reject downward past zlo

    # --- horizon excursions, from precomputed tables --------------------
    for h in horizons_s:
        mn = p.roll_min[h][i]
        mx = p.roll_max[h][i]
        if side != "below":
            out[f"rev_{h}s"] = float(max(0.0, near - mn))
            out[f"cont_{h}s"] = float(max(0.0, mx - far))
        else:
            out[f"rev_{h}s"] = float(max(0.0, mx - near))
            out[f"cont_{h}s"] = float(max(0.0, far - mn))
    if side != "below":
        out["rev_full"] = float(max(0.0, near - p.suf_min_lo[i]))
        out["cont_full"] = float(max(0.0, p.suf_max_hi[i] - far))
        out["max_penetration_t"] = float(max(0.0, p.suf_max_hi[i] - near))
    else:
        out["rev_full"] = float(max(0.0, p.suf_max_hi[i] - near))
        out["cont_full"] = float(max(0.0, far - p.suf_min_lo[i]))
        out["max_penetration_t"] = float(max(0.0, near - p.suf_min_lo[i]))

    out["traversed"] = bool(out["cont_full"] > 0)

    # --- first passage: which side resolved first ------------------------
    # Computed as |grid| rejection crossings plus |grid| continuation crossings,
    # then compared pairwise. The obvious nesting recomputes the continuation
    # scan inside the rejection loop and costs |grid|^2 array passes per zone —
    # 25 instead of 10, which over 300k zones is the difference between a
    # one-minute width sweep and a twenty-minute one.
    hi_f, lo_f, cl_f = p.hi[i:], p.lo[i:], p.cl[i:]
    rev_run = (near - np.minimum.accumulate(lo_f)) if side != "below" \
        else (np.maximum.accumulate(hi_f) - near)
    cont_run = (np.maximum.accumulate(hi_f) - far) if side != "below" \
        else (far - np.minimum.accumulate(lo_f))
    # both running series are non-decreasing, so the first crossing is a
    # binary search rather than a scan
    t_rev, t_cont = {}, {}
    for k in point_grid:
        j = int(np.searchsorted(rev_run, k * TICKS_PER_POINT, side="left"))
        t_rev[k] = j if j < rev_run.size else None
        out[f"t_rev{k}_s"] = float(j) if j < rev_run.size else None
    for m in point_grid:
        j = int(np.searchsorted(cont_run, m * TICKS_PER_POINT, side="left"))
        t_cont[m] = j if j < cont_run.size else None
    for k in point_grid:
        for m in point_grid:
            ir, ia = t_rev[k], t_cont[m]
            if ir is None:
                out[f"fp_rev{k}_adv{m}"] = 0 if ia is not None else None
            elif ia is None:
                out[f"fp_rev{k}_adv{m}"] = 1
            else:
                out[f"fp_rev{k}_adv{m}"] = int(ir < ia)

    # --- cleanliness, over a fixed window so touches are comparable -------
    k = min(n_fwd, clean_window_s)
    out["clean_window_s"] = int(k)
    cw, hw, lw = cl_f[:k], hi_f[:k], lo_f[:k]
    pos = np.where(cw > zhi, 1, np.where(cw < zlo, -1, 0)).astype(np.int8)
    out["secs_inside"] = int((pos == 0).sum())
    out["secs_inside_share"] = float(out["secs_inside"] / k)
    runs = pos[np.insert(np.diff(pos) != 0, 0, True)]
    outside = runs[runs != 0]
    out["n_crossings"] = int((np.diff(outside) != 0).sum()) if outside.size > 1 else 0
    out["n_reentries"] = int(((runs[1:] == 0) & (runs[:-1] != 0)).sum())

    if k >= 8:
        step = max(1, min(EFF_SAMPLE_S, k // 8))
        samp = cw[::step]
        path = float(np.abs(np.diff(samp)).sum())
        out["dir_eff"] = float(abs(samp[-1] - samp[0]) / path) if path > 0 else 0.0
    else:
        out["dir_eff"] = np.nan

    # slice-through: cleared the far edge inside the first minute
    j = min(n_fwd, 60)
    if side != "below":
        out["slice_through_60s"] = bool((hi_f[:j] > far).any())
    else:
        out["slice_through_60s"] = bool((lo_f[:j] < far).any())
    # acceptance: a sustained run of closes beyond the far edge
    beyond = (cw > far) if side != "below" else (cw < far)
    out["secs_beyond_far"] = int(beyond.sum())
    out["accepted"] = bool(_longest_run(beyond) >= 300)
    return out


def _longest_run(mask: np.ndarray) -> int:
    if mask.size == 0 or not mask.any():
        return 0
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return int(np.max(idx[1::2] - idx[0::2]))


def measure_touch_sequence(p: SessionPath, zlo: float, zhi: float,
                           max_touches: int = 5, exit_ticks: float = 4.0,
                           horizons_s: Sequence[int] = (300, 1800),
                           point_grid: Sequence[int] = (10,)) -> List[Dict]:
    """One record per *interaction* with the zone, not just the first.

    A new touch is only counted once price has genuinely **left** — the close
    must clear the zone edge by ``exit_ticks`` — so a price oscillating on the
    boundary registers as one long interaction rather than a dozen touches.
    Without that hysteresis the "second touch" population would be dominated by
    the noisiest zones, and any decay found would be an artefact of the
    counting rule.

    Each record carries ``touch_no``, how deep the *previous* interaction
    penetrated, and how long price was away in between — the three things E4
    needs to ask whether a level is spent after being tested.
    """
    zlo, zhi = float(zlo), float(zhi)
    side = p.side_of(zlo, zhi)
    near, far = (zlo, zhi) if side != "below" else (zhi, zlo)
    hi, lo, cl = p.hi, p.lo, p.cl
    inside = (lo <= zhi) & (hi >= zlo)
    if not inside.any():
        return []
    out: List[Dict] = []
    i = int(np.argmax(inside))
    prev_pen, prev_end = np.nan, None
    for k in range(max_touches):
        # find where this interaction ends: close clears an edge by exit_ticks
        after = np.arange(i, p.n)
        gone = (cl[i:] > zhi + exit_ticks) | (cl[i:] < zlo - exit_ticks)
        if not gone.any():
            end = p.n - 1
        else:
            end = i + int(np.argmax(gone))
        seg_hi, seg_lo = hi[i:end + 1], lo[i:end + 1]
        pen = (seg_hi.max() - near) if side != "below" else (near - seg_lo.min())
        rec = {"touch_no": k + 1, "idx": i, "sec": int(p.sec[i]),
               "prev_penetration_t": prev_pen,
               "secs_away": (float(p.sec[i] - prev_end) if prev_end is not None else np.nan),
               "side": side}
        # forward reaction from this touch
        for h in horizons_s:
            j = min(p.n, i + h)
            mn, mx = lo[i:j].min(), hi[i:j].max()
            rec[f"rev_{h}s"] = float(max(0.0, near - mn)) if side != "below" \
                else float(max(0.0, mx - near))
            rec[f"cont_{h}s"] = float(max(0.0, mx - far)) if side != "below" \
                else float(max(0.0, far - mn))
        rev_run = (near - np.minimum.accumulate(lo[i:])) if side != "below" \
            else (np.maximum.accumulate(hi[i:]) - near)
        cont_run = (np.maximum.accumulate(hi[i:]) - far) if side != "below" \
            else (far - np.minimum.accumulate(lo[i:]))
        for kk in point_grid:
            ir = int(np.searchsorted(rev_run, kk * TICKS_PER_POINT, "left"))
            ia = int(np.searchsorted(cont_run, (kk // 2) * TICKS_PER_POINT, "left"))
            r_ok, a_ok = ir < rev_run.size, ia < cont_run.size
            rec[f"fp_rev{kk}"] = (None if not r_ok and not a_ok
                                  else 1 if not a_ok else 0 if not r_ok else int(ir < ia))
        rec["max_penetration_t"] = float(max(0.0, pen))
        out.append(rec)
        prev_pen, prev_end = rec["max_penetration_t"], int(p.sec[end])
        nxt = inside[end + 1:]
        if not nxt.any():
            break
        i = end + 1 + int(np.argmax(nxt))
    return out


def measure_many(paths: Dict[object, SessionPath], zones: pd.DataFrame,
                 **kw) -> pd.DataFrame:
    """Measure a table of zones. ``zones`` needs ``session``, ``zlo_t``, ``zhi_t``."""
    rows = []
    for s, g in zones.groupby("session", sort=False):
        p = paths.get(s)
        if p is None:
            continue
        for _, z in g.iterrows():
            r = measure_zone(p, z["zlo_t"], z["zhi_t"], **kw)
            r["session"] = s
            for c in g.columns:
                if c not in r and c != "session":
                    r[c] = z[c]
            rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# derived quality labels
# --------------------------------------------------------------------------- #


def clean_rejection(d: pd.DataFrame, points: int = 10, adverse: int = 5,
                    max_inside_share: float = 0.5,
                    max_crossings: int = 1) -> pd.Series:
    """The headline "did this behave like a real price area" label.

    Conjunctive on purpose: a genuine move (``points`` before ``adverse``) that
    was **not** achieved by loitering. Unresolved first passages stay ``NA`` so
    they are dropped rather than silently counted as failures.
    """
    fp = d[f"fp_rev{points}_adv{adverse}"]
    ok = (fp == 1) & (d["secs_inside_share"] <= max_inside_share) \
        & (d["n_crossings"] <= max_crossings)
    return ok.where(fp.notna())


def chop_score(d: pd.DataFrame) -> pd.Series:
    a = (d["secs_inside_share"] / 0.5).clip(upper=1.0)
    b = (d["n_crossings"] / 3.0).clip(upper=1.0)
    c = 1.0 - (d["dir_eff"] / 0.5).clip(upper=1.0)
    return (a + b + c) / 3.0
