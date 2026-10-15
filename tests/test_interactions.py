"""Phase-4 zone/interaction engine: planted sessions in, exact taps out.

Pins the behaviours that silently break: pivot confirmation vs extreme time,
tap-second detection, re-arm hysteresis, approach-feature causality
(truncation invariance), and cross-session carry with roll adjustment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nqlevels.features.impulses import zigzag_pivots
from nqlevels.features.interactions import (build_session_data,
                                            capped_state_label,
                                            process_session)
from nqlevels.features.zones import make_zones, zigzag_confirmed

ATR = 1000.0
T0 = 1_700_000_000 - (1_700_000_000 % 60)


def frames_from_close(cl: np.ndarray):
    cl = np.asarray(cl, "float64")
    n = cl.size
    ts = pd.to_datetime(T0 + np.arange(n), unit="s", utc=True)
    b1s = pd.DataFrame({"ts": ts, "close_t": cl, "high_t": cl + 1.0,
                        "low_t": cl - 1.0, "volume": np.full(n, 10.0),
                        "buy_vol": np.full(n, 6.0), "sell_vol": np.full(n, 4.0)})
    esec = pd.DatetimeIndex(b1s["ts"]).as_unit("ns").asi8 // 10**9
    g = b1s.groupby(esec // 60)
    b1m = pd.DataFrame({
        "ts": pd.to_datetime(g["ts"].first().index * 60, unit="s", utc=True),
        "high_t": g["high_t"].max().to_numpy(),
        "low_t": g["low_t"].min().to_numpy(),
        "close_t": g["close_t"].last().to_numpy(),
        "volume": g["volume"].sum().to_numpy(),
        "buy_vol": g["buy_vol"].sum().to_numpy(),
        "sell_vol": g["sell_vol"].sum().to_numpy()}).reset_index(drop=True)
    return b1s, b1m


def seg(*parts):
    return np.concatenate(parts)


def ramp(nmin, a, b):
    return np.linspace(a, b, nmin * 60)


def flat(nmin, v):
    return np.full(nmin * 60, float(v))


def run(cl, registry=None, sess_ord=0, adj=0.0):
    b1s, b1m = frames_from_close(cl)
    sd = build_session_data(b1s, b1m)
    zones = make_zones(b1m, ATR, adj, sess_ord,
                       np.zeros(len(b1m), dtype="int8"), zid0=sess_ord * 10000)
    reg = registry if registry is not None else {}
    rows = process_session(reg, zones, sd, ATR, adj, 0, sess_ord, 0, 0.5,
                           lambda a, b: {"volz_share": np.nan}, (np.nan, np.nan))
    return reg, rows, zones


def test_zigzag_confirmed_matches_and_confirms():
    rng = np.random.default_rng(7)
    cl = 1000 + np.cumsum(rng.normal(0, 8, 1500))
    hi, lo = cl + 1, cl - 1
    a = zigzag_pivots(hi, lo, 150.0)
    b = zigzag_confirmed(hi, lo, 150.0)
    assert [(i, p, k) for i, p, k, _ in b] == a
    for i, p, k, ic in b:
        assert ic > i
        if k == 1:
            assert p - lo[i:ic + 1].min() >= 150.0
            assert p - lo[i:ic].min() < 150.0 or ic == i + 1
        else:
            assert hi[i:ic + 1].max() - p >= 150.0


# a session: flat, drop (forms a high pivot), base, rally back to tap it,
# reject down again, base, rally to tap a second time
CL = seg(flat(30, 1000), ramp(30, 1000, 800), flat(40, 800),
         ramp(30, 800, 1004), ramp(20, 1004, 900), flat(40, 900),
         ramp(30, 900, 1004), ramp(20, 1004, 880), flat(60, 880))


def test_tap_detected_and_features_causal():
    reg, rows, zones = run(CL)
    r = pd.DataFrame(rows)
    # the scale-0 high pivot near 1001 must exist and be tapped twice
    hz = [z for z in zones if z.dir == 1 and abs(z.price_rf - 1001) < 5
          and z.scale == 0]
    assert hz
    zid = hz[0].zid
    mine = r[r["zid"] == zid].sort_values("tap_no")
    assert len(mine) == 2
    assert list(mine["tap_no"]) == [1, 2]
    t1 = int(mine.iloc[0]["tap_sec"])
    # tap second: first second whose range reaches the band (zlo-1 on highs)
    zlo = mine.iloc[0]["zlo_t"]
    assert CL[t1] + 1.0 >= zlo and CL[t1 - 1] + 1.0 < zlo
    # second tap has history features from the first
    assert mine.iloc[1]["mins_since_last"] > 15
    assert mine.iloc[1]["n_prior_A"] + mine.iloc[1]["n_prior_B"] \
        + mine.iloc[1]["n_prior_C"] + mine.iloc[1]["n_pending"] == 1


def test_truncation_invariance_of_features():
    reg1, rows1, _ = run(CL)
    r1 = pd.DataFrame(rows1).sort_values(["tap_sec", "zid"]).reset_index(drop=True)
    first = r1.iloc[0]
    t = int(first["tap_sec"])
    wrecked = CL.copy()
    wrecked[t + 1:] = 555.0                  # garbage strictly after the tap
    # (the tap event itself needs bar t; features may use only <= t-1)
    reg2, rows2, _ = run(wrecked)
    r2 = pd.DataFrame(rows2)
    m2 = r2[(r2["zid"] == first["zid"]) & (r2["tap_sec"] == t)]
    assert len(m2) == 1
    feat_cols = [c for c in r1.columns if not c.startswith("o_")
                 and c not in ("session_ord",)]
    for c in feat_cols:
        a, b = first[c], m2.iloc[0][c]
        if isinstance(a, float) and np.isnan(a):
            assert np.isnan(b), c
        else:
            assert a == pytest.approx(b), c


def test_rearm_blocks_immediate_retap():
    # touch, tiny bounce that stays near the band, touch again quickly:
    # must be ONE tap because price never left by hysteresis + 900 s
    cl = seg(flat(30, 1000), ramp(30, 1000, 800), flat(40, 800),
             ramp(30, 800, 1004), ramp(3, 1004, 990), ramp(3, 990, 1004),
             ramp(20, 1004, 900), flat(60, 900))
    reg, rows, zones = run(cl)
    r = pd.DataFrame(rows)
    hz = [z.zid for z in zones if z.dir == 1 and abs(z.price_rf - 1001) < 5
          and z.scale == 0]
    assert len(r[r["zid"].isin(hz)]) == 1


def test_cross_session_carry_with_roll():
    reg, rows1, zones = run(seg(flat(30, 1000), ramp(30, 1000, 800),
                                flat(120, 800)), sess_ord=0, adj=0.0)
    hz = [z.zid for z in zones if z.dir == 1 and z.scale == 0]
    assert not pd.DataFrame(rows1).get("zid", pd.Series(dtype=int)).isin(hz).any() \
        if rows1 else True
    # next session trades 80t higher in the new contract (adj = 80)
    cl2 = seg(flat(30, 880), ramp(30, 880, 1084), ramp(20, 1084, 950),
              flat(120, 950))
    reg, rows2, _ = run(cl2, registry=reg, sess_ord=1, adj=80.0)
    r2 = pd.DataFrame(rows2)
    mine = r2[r2["zid"].isin(hz)]
    assert len(mine) == 1
    assert mine.iloc[0]["age_sessions"] == 1
    assert abs(mine.iloc[0]["price_t"] - (1001 + 80)) < 5


def test_capped_state_label_cases():
    cl = seg(flat(5, 900), ramp(10, 900, 1004), ramp(20, 1004, 900),
             flat(150, 900))
    b1s, b1m = frames_from_close(cl)
    sd = build_session_data(b1s, b1m)
    t = int(np.argmax(cl + 1 >= 997))
    cat, depth = capped_state_label(sd, 997.0, 1005.0, t, ATR)
    assert cat in ("A", "B")
    cl2 = seg(flat(5, 900), ramp(10, 900, 1100), flat(150, 1100))
    b1s, b1m = frames_from_close(cl2)
    sd2 = build_session_data(b1s, b1m)
    t2 = int(np.argmax(cl2 + 1 >= 997))
    cat2, _ = capped_state_label(sd2, 997.0, 1005.0, t2, ATR)
    assert cat2 == "F"                        # accepted through, never back
