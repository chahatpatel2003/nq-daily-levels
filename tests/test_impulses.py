"""Zigzag legs and origin zones: planted swings in, exact legs out."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nqlevels.features.impulses import detect_impulses, zigzag_pivots

ATR = 100.0


def bars_from_close(cl):
    cl = np.asarray(cl, "float64")
    return pd.DataFrame({"high_t": cl + 1.0, "low_t": cl - 1.0, "close_t": cl,
                         "volume": np.full(cl.size, 100.0),
                         "buy_vol": np.full(cl.size, 70.0),
                         "sell_vol": np.full(cl.size, 30.0)})


def test_single_v_shape():
    cl = np.concatenate([np.full(20, 100.0), np.linspace(100, 40, 30),
                         np.linspace(40, 130, 40), np.full(30, 130.0),
                         np.linspace(130, 60, 30), np.full(20, 60.0)])
    legs = detect_impulses(bars_from_close(cl), ATR, thresh_atr=0.35)
    # down leg 100->40 and up leg 40->130; the final 130->60 swing ends the
    # tape without a confirmed low pivot, so it is dropped by design
    assert list(legs["dir"]) == [-1, 1]
    up = legs[legs["dir"] == 1].iloc[0]
    assert up["p0_t"] == pytest.approx(39.0)          # pivot low = close - 1
    assert up["mag_atr"] == pytest.approx((131.0 - 39.0) / ATR, abs=0.02)
    # demand origin zone sits on the pivot, bounded above by the base
    assert up["zlo_t"] == pytest.approx(39.0)
    assert up["zhi_t"] > up["zlo_t"]
    assert up["zhi_t"] - up["zlo_t"] <= max(4.0, 0.06 * ATR) + 1e-9


def test_small_wiggle_no_pivot():
    cl = np.concatenate([np.full(50, 100.0), np.linspace(100, 120, 20),
                         np.linspace(120, 105, 20), np.full(50, 105.0)])
    legs = detect_impulses(bars_from_close(cl), ATR, thresh_atr=0.35)
    assert len(legs) == 0                              # 20-ish < 35 threshold


def test_velocity_and_delta():
    cl = np.concatenate([np.full(30, 100.0), np.linspace(100, 160, 12),
                         np.full(40, 160.0), np.linspace(160, 100, 30),
                         np.full(20, 100.0)])
    legs = detect_impulses(bars_from_close(cl), ATR, thresh_atr=0.35)
    up = legs[legs["dir"] == 1].iloc[0]
    assert up["vel_atr_h"] > 1.0                       # ~0.6 ATR in ~1 h
    assert up["delta_share"] == pytest.approx(0.4)     # (70-30)/100 everywhere


def test_prefix_property():
    rng = np.random.default_rng(9)
    cl = 1000 + np.cumsum(rng.normal(0, 5, 800))
    cl[200:230] += np.linspace(0, 90, 30)
    cl[230:260] -= np.linspace(0, 120, 30)
    full = detect_impulses(bars_from_close(cl), ATR)
    pre = detect_impulses(bars_from_close(cl[:500]), ATR)
    # every confirmed leg wholly inside the prefix appears identically
    f = full[full["i1"] < 450][["i0", "i1", "dir", "p0_t", "p1_t"]]
    p = pre[["i0", "i1", "dir", "p0_t", "p1_t"]]
    merged = f.merge(p, on=list(f.columns), how="left", indicator=True)
    assert (merged["_merge"] == "both").all()


def test_zigzag_alternates():
    rng = np.random.default_rng(4)
    cl = 1000 + np.cumsum(rng.normal(0, 8, 1200))
    piv = zigzag_pivots(cl + 1, cl - 1, 60.0)
    kinds = [k for _, _, k in piv]
    assert all(a != b for a, b in zip(kinds[:-1], kinds[1:]))
    idx = [i for i, _, _ in piv]
    assert idx == sorted(idx)
