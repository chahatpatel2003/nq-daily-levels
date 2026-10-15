"""Enriched interaction outcomes for Phase 6 — magnitudes, adverse depth,
turn locations, path-based endpoints. One walker, applied identically to
real zones and shadow anchors, so exact-location contrasts cannot be an
artifact of asymmetric measurement.

Conventions extend labels/sweep.py:

* **Adverse penetration is measured from the NEAR edge and includes in-zone
  travel** — the trader's "how far did it go against the level", not the
  beyond-far-edge depth the taxonomy uses for sweep caps.
* **Rejection excursion** is from the near edge (as everywhere) with
  horizon values (15/30/60 min, full window) and a path-based endpoint: the
  move attributed to the level ends at the first close back inside the band
  AFTER the rejection has reached 15 points — later movement belongs to the
  next interaction, not this one.
* All quantities are in ticks; report layers divide by 4 for points.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..eval.labeling import SessionPath
from .sweep import sweep_metrics

TICKS = 4
R15 = 15 * TICKS
HORIZONS = (900, 1800, 3600)


def first_touch_after(p: SessionPath, zlo: float, zhi: float,
                      t0: int) -> Optional[int]:
    """First second at/after t0 whose bar range intersects the band."""
    if t0 >= p.n:
        return None
    m = (p.lo[t0:] <= zhi) & (p.hi[t0:] >= zlo)
    if not m.any():
        return None
    return t0 + int(np.argmax(m))


def enriched_metrics(p: SessionPath, zlo: float, zhi: float, i: int) -> Dict:
    """sweep_metrics plus magnitude / adverse / turn-location fields."""
    out = sweep_metrics(p, zlo, zhi, i)
    below = out["side"] == "below"
    near = zhi if below else zlo
    width = zhi - zlo
    hi_f, lo_f, cl_f = p.hi[i:], p.lo[i:], p.cl[i:]
    n = hi_f.size

    rev_run = (np.maximum.accumulate(hi_f) - near) if below \
        else (near - np.minimum.accumulate(lo_f))
    np.clip(rev_run, 0.0, None, out=rev_run)
    pen_run = (near - np.minimum.accumulate(lo_f)) if below \
        else (np.maximum.accumulate(hi_f) - near)
    np.clip(pen_run, 0.0, None, out=pen_run)

    for h in HORIZONS:
        out[f"rev_pts_{h}"] = float(rev_run[min(h, n) - 1])
    out["rev_full_t"] = float(rev_run[-1])
    out["pen_max_full_t"] = float(pen_run[-1])
    j15 = int(np.searchsorted(rev_run, R15, side="left"))
    if j15 < n:
        out["adverse_before_rev15_t"] = float(pen_run[j15])
        out["turn_frac_rev15"] = float(pen_run[j15] / width) if width > 0 else np.nan
        out["t_rev15_s"] = int(j15)
        j30 = int(np.searchsorted(rev_run, 2 * R15, side="left"))
        out["adverse_before_rev30_t"] = float(pen_run[j30]) if j30 < n else np.nan
        # endpoint: first close back inside the band after the 15-pt rejection
        inside = (cl_f[j15:] <= zhi) & (cl_f[j15:] >= zlo)
        if inside.any():
            e = j15 + int(np.argmax(inside))
            out["rev_endpoint_t"] = float(rev_run[e])
            out["t_endpoint_s"] = int(e)
        else:
            out["rev_endpoint_t"] = float(rev_run[-1])
            out["t_endpoint_s"] = int(n - 1)
        jmax = int(np.searchsorted(rev_run, rev_run[-1], side="left"))
        out["t_revmax_s"] = int(jmax)
    else:
        for c in ("adverse_before_rev15_t", "turn_frac_rev15",
                  "adverse_before_rev30_t", "rev_endpoint_t"):
            out[c] = np.nan
        out["t_rev15_s"] = None
        out["t_endpoint_s"] = None
        out["t_revmax_s"] = None
    return out
