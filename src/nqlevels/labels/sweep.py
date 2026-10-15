"""Path-aware sweep/reclaim measurement — the reaction-definition audit layer.

The clean-rejection label scores this sequence as a failure:

    touch -> trade through the zone -> sweep the liquidity beyond ->
    reclaim the zone -> meaningful move away

because the initial penetration loses the X-before-Y race. Whether that
sequence is (a) real, (b) distinguishable from "acceptance through the level
followed by an unrelated later reversal", and (c) *predictable by location*,
is the question this module exists to answer. It takes no position: it walks
the 1-second path once per touched zone and returns raw quantities; every
taxonomy threshold is applied downstream, so a definition grid costs nothing
and cannot quietly re-measure the data.

Conventions, chosen to match the trading concepts being tested:

* **Penetration is measured by range** (a stop-run is a wick), **reclaim by
  closes held for ``hold_s`` seconds** (a reclaim is commitment; a single
  jittering second is not).
* Three reclaim levels are always measured: back inside the zone (``en``),
  back past the midpoint (``mid``), back past the near edge (``ne``).
* **Post-reclaim reaction is anchored at the same near edge as a direct
  rejection**, so magnitudes are comparable across the taxonomy.
* Everything is forward from the first touch, in ticks, truncated at the end
  of the target window.

The taxonomy itself (direct / sweep+reclaim / acceptance / chop) lives in
:func:`classify`, parameterised by reaction size R, direct-tolerance d0,
sweep-depth cap D, reclaim-time cap T and reclaim level L.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..eval.labeling import SessionPath

REV_GRID = (5, 10, 15, 20, 30)          # points, matching the existing outcome grid
AFTER_HORIZONS = (300, 900, 1800)       # seconds after reclaim
TICKS_PER_POINT = 4
LEVELS = ("en", "mid", "ne")


def _first_hold_run(b: np.ndarray, hold: int) -> Optional[int]:
    """Start index of the first True-run of length >= hold, else None."""
    if b.size == 0 or not b.any():
        return None
    db = np.diff(np.concatenate(([0], b.astype(np.int8), [0])))
    starts = np.flatnonzero(db == 1)
    ends = np.flatnonzero(db == -1)
    ok = np.flatnonzero(ends - starts >= hold)
    return int(starts[ok[0]]) if ok.size else None


def sweep_metrics(p: SessionPath, zlo: float, zhi: float, i: int,
                  hold_s: int = 60,
                  rev_grid: Sequence[int] = REV_GRID) -> Dict:
    """All sweep/reclaim path quantities for one touched zone.

    ``i`` is the first-touch index (as produced by ``SessionPath.first_touch``).
    Returns ticks and seconds; ``None`` for events that never happened.
    """
    side = p.side_of(zlo, zhi)
    if side == "inside":
        raise ValueError("sweep metrics are undefined for zones containing the reference")
    below = side == "below"                     # approached from above, support-flavoured
    near, far = (zhi, zlo) if below else (zlo, zhi)
    mid = (zlo + zhi) / 2.0

    hi_f, lo_f, cl_f = p.hi[i:], p.lo[i:], p.cl[i:]
    n = hi_f.size
    out: Dict = {"side": side, "n_fwd": int(n)}

    # reaction excursion from the near edge (running), and time-to-R
    rev_run = (np.maximum.accumulate(hi_f) - near) if below \
        else (near - np.minimum.accumulate(lo_f))
    np.clip(rev_run, 0.0, None, out=rev_run)
    for k in rev_grid:
        j = int(np.searchsorted(rev_run, k * TICKS_PER_POINT, side="left"))
        out[f"t_rev{k}"] = j if j < n else None

    # beyond-far penetration depth (running max, by range) and close-dwell
    depth_inst = (far - lo_f) if below else (hi_f - far)
    depth_run = np.maximum.accumulate(np.clip(depth_inst, 0.0, None))
    beyond_cl = (cl_f < far) if below else (cl_f > far)
    dwell_run = np.concatenate(([0], np.cumsum(beyond_cl)))

    for k in rev_grid:
        t = out[f"t_rev{k}"]
        out[f"depth_before_rev{k}"] = float(depth_run[t]) if t is not None else None

    idx_beyond = np.flatnonzero(depth_inst > 0)
    out["first_beyond"] = int(idx_beyond[0]) if idx_beyond.size else None
    out["last_beyond"] = int(idx_beyond[-1]) if idx_beyond.size else None
    out["depth_max_full"] = float(depth_run[-1]) if n else 0.0
    out["dwell_beyond_full"] = int(dwell_run[-1])

    # position runs over closes: crossings / re-entries over the whole window
    pos = np.where(cl_f > zhi, 1, np.where(cl_f < zlo, -1, 0)).astype(np.int8)
    runs = pos[np.insert(np.diff(pos) != 0, 0, True)] if n else pos
    outside = runs[runs != 0]
    out["crossings_full"] = int((np.diff(outside) != 0).sum()) if outside.size > 1 else 0
    out["reentries_full"] = int(((runs[1:] == 0) & (runs[:-1] != 0)).sum()) if runs.size > 1 else 0
    out["inside_share_full"] = float((pos == 0).mean()) if n else np.nan

    # reclaim at each level, only meaningful after a beyond-excursion exists
    fb = out["first_beyond"]
    for tag, lvl in (("en", far), ("mid", mid), ("ne", near)):
        t_rec = None
        if fb is not None:
            b = (cl_f[fb:] >= lvl) if below else (cl_f[fb:] <= lvl)
            r = _first_hold_run(b, hold_s)
            t_rec = fb + r if r is not None else None
        out[f"t_reclaim_{tag}"] = t_rec
        if t_rec is None:
            for c in (f"depth_before_{tag}", f"dwell_before_{tag}",
                      f"t_maxdepth_{tag}", f"re_swept_{tag}"):
                out[c] = None
            for h in AFTER_HORIZONS:
                out[f"rev_after_{tag}_{h}"] = None
            out[f"rev_after_{tag}_full"] = None
            continue
        out[f"depth_before_{tag}"] = float(depth_run[t_rec])
        out[f"dwell_before_{tag}"] = int(dwell_run[t_rec])
        out[f"t_maxdepth_{tag}"] = int(np.argmax(depth_run[:t_rec + 1]))
        out[f"re_swept_{tag}"] = bool(out["last_beyond"] > t_rec)
        a = i + t_rec
        for h in AFTER_HORIZONS:
            m = p.roll_max[h][a] if below else p.roll_min[h][a]
            out[f"rev_after_{tag}_{h}"] = float(max(0.0, (m - near) if below
                                                    else (near - m)))
        m = p.suf_max_hi[a] if below else p.suf_min_lo[a]
        out[f"rev_after_{tag}_full"] = float(max(0.0, (m - near) if below
                                                 else (near - m)))
    return out


# --------------------------------------------------------------------------- #
# the taxonomy, applied to a frame of sweep metrics
# --------------------------------------------------------------------------- #


def classify(d: pd.DataFrame, R: int = 15, d0_t: float = 4.0,
             D_t: float | pd.Series = 24.0, T_s: int = 900,
             level: str = "ne", after_h: int = 1800,
             min_fwd_s: int = 1800) -> pd.DataFrame:
    """Apply one taxonomy definition to a sweep-metrics frame.

    Returns columns: ``resolved`` plus one-hot ``cat_A`` (direct rejection),
    ``cat_B`` (sweep+reclaim), ``cat_C`` (acceptance/failure), ``cat_D``
    (chop / no decisive response), and ``y_any`` = A or B. Unresolved rows
    (window shorter than ``min_fwd_s``) carry NA in every category.

    Definitions:
      A: R-point rejection from the near edge arrives while beyond-far
         penetration has not exceeded d0.
      B: not A; price penetrated beyond d0; reclaimed ``level`` (closes held)
         with max depth <= D and time-from-first-beyond-to-reclaim <= T; and
         a R-point rejection from the near edge occurs within ``after_h``
         seconds of the reclaim.
      C: not A/B and penetration exceeded d0 — the zone was violated and never
         validly recovered into a reaction.
      D: not A/B, penetration never exceeded d0, no R rejection — touched and
         held-ish, but no decisive response.
    """
    Rt = R * TICKS_PER_POINT
    t_rev = d[f"t_rev{R}"]
    dep_at_rev = d[f"depth_before_rev{R}"]
    resolved = d["n_fwd"] >= min_fwd_s

    A = t_rev.notna() & (dep_at_rev.fillna(np.inf) <= d0_t)

    t_rec = d[f"t_reclaim_{level}"]
    dep_rec = d[f"depth_before_{level}"]
    t_fb = d["first_beyond"]
    rev_after = d[f"rev_after_{level}_{after_h}"]
    B = (~A) & t_fb.notna() & (d["depth_max_full"] > d0_t) & t_rec.notna() \
        & (dep_rec <= D_t) \
        & ((t_rec - t_fb) <= T_s) \
        & (rev_after >= Rt)

    beyond = d["depth_max_full"] > d0_t
    C = (~A) & (~B) & beyond
    D_ = (~A) & (~B) & (~beyond)

    out = pd.DataFrame(index=d.index)
    out["resolved"] = resolved
    for name, v in (("cat_A", A), ("cat_B", B), ("cat_C", C), ("cat_D", D_)):
        out[name] = v.where(resolved).astype("float32")
    out["y_any"] = ((A | B).where(resolved)).astype("float32")
    return out
