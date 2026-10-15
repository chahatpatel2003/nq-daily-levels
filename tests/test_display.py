"""Display-time scoring must be feature-identical to tap-time scoring.

The live simulation's whole validity rests on one property: the feature
vector `zone_row` computes at an arbitrary second equals what
`process_session` recorded when that second was an actual tap. Both planted
taps (one with no history, one with an in-session prior tap) are compared
field for field. Plus eligibility basics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nqlevels.features.display import (ZoneSnap, build_aux, eligible, snap_of,
                                       zone_row)
from nqlevels.features.interactions import build_session_data, process_session
from nqlevels.features.zones import make_zones
from test_interactions import ATR, CL, frames_from_close


@pytest.fixture(scope="module")
def sim():
    b1s, b1m = frames_from_close(CL)
    sd = build_session_data(b1s, b1m)
    zones = make_zones(b1m, ATR, 0.0, 0, np.zeros(len(b1m), dtype="int8"), 0)
    snaps = {z.zid: snap_of(z) for z in zones}
    auxs = {z.zid: build_aux(z, sd, 0.0, sess_ord=0) for z in zones}
    reg = {}
    st = {}
    vf = lambda a, b: {"volz_share": np.nan}                    # noqa: E731
    rows = process_session(reg, zones, sd, ATR, 0.0, 0, 0, 0, 0.5,
                           vf, (np.nan, np.nan), state_out=st)
    return sd, zones, snaps, auxs, rows, st["upd"], vf


def test_display_row_equals_tap_row(sim):
    sd, zones, snaps, auxs, rows, upd, vf = sim
    zmap = {z.zid: z for z in zones}
    skip = {"session_ord"}
    for row in rows:
        zid = row["zid"]
        r2 = zone_row(zmap[zid], snaps[zid], auxs[zid], upd.get(zid, []),
                      sd, int(row["tap_sec"]), ATR, 0.0, 0, 0, 0.5,
                      vf, (np.nan, np.nan))
        for k, v in row.items():
            if k.startswith("o_") or k in skip:
                continue
            w = r2[k]
            if isinstance(v, float) and (v != v):
                assert w != w, k
            else:
                assert v == pytest.approx(w), (k, v, w)


def test_eligibility_rules(sim):
    sd, zones, snaps, auxs, rows, upd, vf = sim
    z = [x for x in zones if x.dir == 1 and abs(x.price_rf - 1001) < 5][0]
    snap, aux = snaps[z.zid], auxs[z.zid]
    tap1 = int([r for r in rows if r["zid"] == z.zid][0]["tap_sec"])
    # far before the tap, price ~200t below zone: distance 0.2 ATR -> eligible
    t_far = tap1 - 1800
    assert eligible(z, snap, aux, [], sd, t_far, 0, ATR)
    # price inside the band -> not eligible
    t_in = tap1 + 30
    if aux.zlo <= sd.cl[t_in - 1] <= aux.zhi:
        assert not eligible(z, snap, aux, [], sd, t_in, 0, ATR)
    # immediately after a tap (< 900 s) -> not eligible
    t_soon = tap1 + 400
    u = [(float(sd.sec_ts[tap1]), "A", 0.0)]
    assert not eligible(z, snap, aux, u, sd, t_soon, 0, ATR)
    # too far away -> not eligible
    zfar = [x for x in zones if x.dir == -1][0]
    afar = auxs[zfar.zid]
    mid = (afar.zlo + afar.zhi) / 2
    t0 = 100
    d = abs(mid - sd.cl[t0 - 1]) / ATR
    if d > 0.75:
        assert not eligible(zfar, snaps[zfar.zid], afar, [], sd, t0, 0, ATR)
