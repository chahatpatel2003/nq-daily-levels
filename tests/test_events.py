"""Rejection-event detector: planted shapes in, exact events out.

The detector feeds every reaction-memory feature, so its edge rules are pinned
here: confirmation threshold, truncation at a new extreme, flat-top
de-duplication, the low-side mirror, and the prefix property that makes
same-morning use leak-free.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nqlevels.features.events import detect_events, roll_adjustment

ATR = 100.0


def bars_from_close(cl: np.ndarray) -> pd.DataFrame:
    cl = np.asarray(cl, "float64")
    return pd.DataFrame({"high_t": cl + 1.0, "low_t": cl - 1.0, "close_t": cl,
                         "volume": np.full(cl.size, 100.0),
                         "buy_vol": np.full(cl.size, 60.0),
                         "sell_vol": np.full(cl.size, 40.0)})


def ramp(n, a, b):
    return np.linspace(a, b, n)


def test_high_rejection_detected():
    cl = np.concatenate([np.full(60, 100.0), ramp(20, 100, 140),
                         ramp(20, 140, 90), np.full(80, 90.0)])
    ev = detect_events(bars_from_close(cl), ATR, w_min=15, confirm_min=60,
                       theta_atr=0.25)
    highs = ev[ev["dir"] == 1]
    assert len(highs) == 1
    e = highs.iloc[0]
    assert e["price_t"] == pytest.approx(141.0)         # high = close + 1
    assert e["mag_t"] == pytest.approx(141.0 - 89.0)    # down to low = close - 1
    assert e["mag_atr"] == pytest.approx(52.0 / ATR)


def test_small_pullback_not_confirmed():
    cl = np.concatenate([np.full(60, 100.0), ramp(20, 100, 140),
                         ramp(20, 140, 125), np.full(80, 125.0)])
    ev = detect_events(bars_from_close(cl), ATR, w_min=15, confirm_min=60,
                       theta_atr=0.25)
    assert len(ev[ev["dir"] == 1]) == 0                 # 15+2 ticks < 25 needed


def test_truncated_at_new_high():
    # spike, shallow dip, break to a fresh high, then a real collapse:
    # the first spike must NOT get credit for the collapse after the break
    cl = np.concatenate([np.full(60, 100.0), ramp(10, 100, 140),
                         ramp(10, 140, 130), ramp(10, 130, 160),
                         ramp(30, 160, 80), np.full(60, 80.0)])
    ev = detect_events(bars_from_close(cl), ATR, w_min=8, confirm_min=80,
                       theta_atr=0.25)
    highs = ev[ev["dir"] == 1]
    assert len(highs) == 1
    assert highs.iloc[0]["price_t"] == pytest.approx(161.0)


def test_flat_top_deduplicated():
    top = np.full(6, 140.0)
    cl = np.concatenate([np.full(40, 100.0), ramp(10, 100, 140), top,
                         ramp(20, 140, 90), np.full(60, 90.0)])
    ev = detect_events(bars_from_close(cl), ATR, w_min=10, confirm_min=60,
                       theta_atr=0.25)
    assert len(ev[ev["dir"] == 1]) == 1


def test_low_mirror():
    cl = np.concatenate([np.full(60, 100.0), ramp(20, 100, 60),
                         ramp(20, 60, 110), np.full(80, 110.0)])
    ev = detect_events(bars_from_close(cl), ATR, w_min=15, confirm_min=60,
                       theta_atr=0.25)
    lows = ev[ev["dir"] == -1]
    assert len(lows) == 1
    assert lows.iloc[0]["price_t"] == pytest.approx(59.0)


def test_prefix_property_for_same_day_use():
    """Events confirmed inside a prefix are found identically by running the
    detector on the prefix alone — the same-morning path never needs the full
    tape, so it cannot leak."""
    rng = np.random.default_rng(5)
    cl = 1000 + np.cumsum(rng.normal(0, 4, 900))
    cl[300:320] += np.linspace(0, 60, 20)
    cl[320:360] -= np.linspace(0, 80, 40)
    full = detect_events(bars_from_close(cl), ATR, w_min=20, confirm_min=60,
                         theta_atr=0.25)
    cut = 500
    pre = detect_events(bars_from_close(cl[:cut]), ATR, w_min=20,
                        confirm_min=60, theta_atr=0.25)
    done = full[(full["minute"] + full["t_confirm_min"] < cut - 20)
                & (full["minute"] < cut - 20)]
    got = set(zip(pre["minute"], pre["dir"]))
    assert set(zip(done["minute"], done["dir"])) <= got


def test_roll_adjustment_chain():
    sessions = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04",
                               "2024-01-05"]).date
    front = pd.Series(["NQH4", "NQH4", "NQM4", "NQM4"], index=sessions)
    ts = pd.date_range("2024-01-03 00:00", periods=300, freq="1min", tz="UTC")
    rows = []
    for s in sessions[:3]:
        rows.append(pd.DataFrame({"session": s, "symbol": "NQH4", "ts": ts,
                                  "close_t": 1000.0}))
        rows.append(pd.DataFrame({"session": s, "symbol": "NQM4", "ts": ts,
                                  "close_t": 1080.0}))
    bars = pd.concat(rows, ignore_index=True)
    adj = roll_adjustment(bars, front, min_overlap_minutes=200)
    assert list(adj["epoch"]) == [0, 0, 1, 1]
    # a price of 1000 in the old scale reads 1080 in the new one
    a = adj.set_index("session")["adj_t"]
    assert a[sessions[3]] - a[sessions[0]] == pytest.approx(80.0)
    assert adj["chain"].nunique() == 1
