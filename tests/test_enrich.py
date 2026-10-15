"""Enriched outcome walker: planted paths in, exact magnitudes out."""
from __future__ import annotations

import numpy as np
import pytest

from nqlevels.eval.labeling import SessionPath
from nqlevels.labels.enrich import enriched_metrics, first_touch_after

ZLO, ZHI = -40.0, -28.0        # support band, near edge -28, width 12t (3 pts)


def path_of(cl):
    cl = np.asarray(cl, "float64")
    return SessionPath("s", np.arange(cl.size, dtype="int32"),
                       cl + 1.0, cl - 1.0, cl)


def seg(*parts):
    return np.concatenate(parts)


def ramp(nmin, a, b):
    return np.linspace(a, b, nmin * 60)


def flat(nmin, v):
    return np.full(nmin * 60, float(v))


def test_magnitude_adverse_and_turn():
    # dip 6t into the zone (to -34), rally 120t above near edge, then return
    cl = seg(flat(5, 0.0), ramp(10, 0, -34), flat(2, -34.0),
             ramp(20, -34, 92), flat(5, 92.0), ramp(20, 92, -30),
             flat(60, -30.0))
    p = path_of(cl)
    i = p.first_touch(ZLO, ZHI)
    m = enriched_metrics(p, ZLO, ZHI, i)
    # adverse before the 15-pt (60t) rejection: low -35 vs near -28 -> 7t
    assert m["adverse_before_rev15_t"] == pytest.approx(7.0)
    assert m["turn_frac_rev15"] == pytest.approx(7.0 / 12.0)
    # full rejection: high 93 - (-28) = 121t
    assert m["rev_full_t"] == pytest.approx(121.0)
    # endpoint: the return into the band ends the attributed move at its max
    assert m["rev_endpoint_t"] == pytest.approx(121.0)
    assert m["t_endpoint_s"] is not None


def test_endpoint_cuts_later_move():
    # rally 20 pts, come back INSIDE the band, then a huge later rally that
    # must NOT be attributed to this interaction
    cl = seg(flat(5, 0.0), ramp(5, 0, -30), ramp(10, -30, 52), flat(2, 52.0),
             ramp(10, 52, -30), flat(5, -30.0), ramp(20, -30, 400),
             flat(30, 400.0))
    p = path_of(cl)
    i = p.first_touch(ZLO, ZHI)
    m = enriched_metrics(p, ZLO, ZHI, i)
    assert m["rev_full_t"] == pytest.approx(429.0)          # full window sees 400+1+28
    assert m["rev_endpoint_t"] == pytest.approx(81.0)       # 53 - (-28) at return
    assert m["rev_pts_900"] <= m["rev_full_t"]


def test_no_reaction_gives_nans():
    cl = seg(flat(5, 0.0), ramp(5, 0, -34), flat(60, -34.0))
    p = path_of(cl)
    i = p.first_touch(ZLO, ZHI)
    m = enriched_metrics(p, ZLO, ZHI, i)
    assert np.isnan(m["adverse_before_rev15_t"])
    assert m["t_rev15_s"] is None


def test_first_touch_after():
    cl = seg(flat(5, 0.0), ramp(5, 0, -34), flat(5, -34.0), ramp(5, -34, 20),
             flat(5, 20.0), ramp(5, 20, -34), flat(30, -34.0))
    p = path_of(cl)
    t1 = first_touch_after(p, ZLO, ZHI, 0)
    assert t1 is not None and abs(p.lo[t1] - ZHI) < 2.0
    # after the first visit ends, the next touch is the second dip
    t_mid = 16 * 60
    t2 = first_touch_after(p, ZLO, ZHI, t_mid)
    assert t2 is not None and t2 >= 25 * 60
    assert first_touch_after(p, 500.0, 520.0, 0) is None
