"""Sweep/reclaim classifier: planted paths in, exact categories out.

Each scenario is a hand-built 1-second path around a support zone at
[-40, -28] ticks (reference 0, approached from above), or its resistance
mirror. The taxonomy must separate: direct rejection, classic sweep+reclaim,
deep acceptance with a late unrelated reversal, reclaim-without-reaction,
chop, and unresolved windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nqlevels.eval.labeling import SessionPath
from nqlevels.labels.sweep import classify, sweep_metrics

ZLO, ZHI = -40.0, -28.0          # support zone, near edge -28, far edge -40


def path_of(cl: np.ndarray) -> SessionPath:
    cl = np.asarray(cl, "float64")
    return SessionPath("s", np.arange(cl.size, dtype="int32"),
                       cl + 1.0, cl - 1.0, cl)


def seg(*parts):
    return np.concatenate(parts)


def ramp(n, a, b):
    return np.linspace(a, b, n)


def metrics_for(cl, zlo=ZLO, zhi=ZHI):
    p = path_of(cl)
    i = p.first_touch(zlo, zhi)
    assert i is not None
    m = sweep_metrics(p, zlo, zhi, i)
    return pd.DataFrame([m])


def cats(d, **kw):
    c = classify(d, **kw).iloc[0]
    if not c["resolved"]:
        return "unresolved"
    for k in ("A", "B", "C", "D"):
        if c[f"cat_{k}"] == 1.0:
            return k
    return "?"


def test_direct_rejection():
    cl = seg(np.full(300, 0.0), ramp(300, 0, -34), np.full(60, -34.0),
             ramp(600, -34, 40), np.full(3000, 40.0))
    d = metrics_for(cl)
    assert cats(d, R=15, d0_t=4.0) == "A"
    assert d["depth_before_rev15"].iloc[0] == 0.0
    assert d["first_beyond"].iloc[0] is None or np.isnan(d["first_beyond"].iloc[0])


def test_classic_sweep_reclaim():
    cl = seg(np.full(300, 0.0), ramp(300, 0, -60), np.full(120, -60.0),
             ramp(120, -60, -20), np.full(120, -20.0),
             ramp(600, -20, 40), np.full(2400, 40.0))
    d = metrics_for(cl)
    assert cats(d, R=15, d0_t=4.0, D_t=24.0, T_s=900, level="ne") == "B"
    r = d.iloc[0]
    assert r["depth_max_full"] == pytest.approx(21.0)     # lo = cl - 1
    assert r["depth_before_ne"] == pytest.approx(21.0)
    assert r["t_reclaim_ne"] is not None
    assert r["rev_after_ne_1800"] >= 60.0
    # the same interaction under a stricter depth cap is not a valid sweep
    assert cats(d, R=15, d0_t=4.0, D_t=12.0, T_s=900, level="ne") == "C"


def test_acceptance_with_late_reversal_is_C():
    cl = seg(np.full(300, 0.0), ramp(300, 0, -120), np.full(2400, -120.0),
             ramp(600, -120, 40), np.full(600, 40.0))
    d = metrics_for(cl)
    # deep, long acceptance; the eventual huge rally must NOT credit the zone
    assert cats(d, R=15, d0_t=4.0, D_t=24.0, T_s=900, level="ne") == "C"
    # even a generous cap keeps it C on the time dimension
    assert cats(d, R=15, d0_t=4.0, D_t=200.0, T_s=900, level="ne") == "C"


def test_reclaim_without_reaction_is_C():
    cl = seg(np.full(300, 0.0), ramp(200, 0, -50), np.full(60, -50.0),
             ramp(60, -50, -25), np.full(3000, -25.0))
    d = metrics_for(cl)
    assert cats(d, R=15, d0_t=4.0, D_t=24.0, T_s=900, level="ne") == "C"
    r = d.iloc[0]
    assert r["t_reclaim_ne"] is not None                  # reclaim happened
    assert r["rev_after_ne_1800"] < 60.0                  # reaction did not


def test_chop_is_D():
    base = seg(*[np.concatenate([np.full(60, -20.0), np.full(60, -36.0)])
                 for _ in range(30)])
    cl = seg(np.full(120, 0.0), ramp(60, 0, -20), base, np.full(600, -20.0))
    d = metrics_for(cl)
    assert cats(d, R=15, d0_t=4.0) == "D"
    assert d["reentries_full"].iloc[0] >= 5


def test_resistance_mirror():
    zlo, zhi = 28.0, 40.0
    cl = seg(np.full(300, 0.0), ramp(300, 0, 60), np.full(120, 60.0),
             ramp(120, 60, 20), np.full(120, 20.0),
             ramp(600, 20, -40), np.full(2400, -40.0))
    d = metrics_for(cl, zlo, zhi)
    assert d["side"].iloc[0] == "above"
    assert cats(d, R=15, d0_t=4.0, D_t=24.0, T_s=900, level="ne") == "B"


def test_short_window_unresolved():
    cl = seg(np.full(100, 0.0), ramp(100, 0, -30), np.full(400, -30.0))
    d = metrics_for(cl)
    assert cats(d, R=15) == "unresolved"


def test_fast_v_sweep_counts_as_B():
    # pierce and rocket straight back: no dwell, reclaim on the way up
    cl = seg(np.full(300, 0.0), ramp(200, 0, -60), ramp(200, -60, 40),
             np.full(2400, 40.0))
    d = metrics_for(cl)
    assert cats(d, R=15, d0_t=4.0, D_t=24.0, T_s=900, level="ne") == "B"
    # it must not be A: the sweep preceded the rejection
    assert d["depth_before_rev15"].iloc[0] > 4.0
