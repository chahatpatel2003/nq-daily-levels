"""Touch and reaction measurement, on synthetic tapes with known answers.

Every case here was drawn on paper first: if the definitions in
``labels/`` ever drift, these fail rather than quietly re-scoring history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nqlevels.labels.reaction import (chop_score, clean_rejection,
                                      crossing_counts, evaluate_reaction)
from nqlevels.labels.touch import Zone, evaluate_touch, first_touch_order, zone_side

T0 = pd.Timestamp("2024-03-05 14:30", tz="UTC")


def tape(closes, highs=None, lows=None) -> pd.DataFrame:
    c = np.asarray(closes, dtype="int64")
    h = np.asarray(highs if highs is not None else closes, dtype="int64")
    lo = np.asarray(lows if lows is not None else closes, dtype="int64")
    return pd.DataFrame({"ts": T0 + pd.to_timedelta(np.arange(len(c)), "s"),
                         "open_t": c, "high_t": h, "low_t": lo, "close_t": c,
                         "volume": 1, "n_trades": 1, "buy_vol": 1, "sell_vol": 0})


# ------------------------------------------------------------------- zones

def test_zone_edges_and_sides():
    z = Zone(100, 110, "test")
    assert z.mid_t == 105 and z.width_t == 10
    assert z.near_edge_t("above") == 100 and z.far_edge_t("above") == 110
    assert z.near_edge_t("below") == 110 and z.far_edge_t("below") == 100


def test_inverted_zone_rejected():
    with pytest.raises(ValueError):
        Zone(110, 100, "bad")


def test_zone_side():
    z = Zone(100, 110, "t")
    assert zone_side(z, 50) == "above"
    assert zone_side(z, 200) == "below"
    assert zone_side(z, 105) == "inside"


# ------------------------------------------------------------------ touch

def test_touch_uses_the_bar_range_not_the_close():
    """A wick through the zone is a real interaction."""
    b = tape([90, 90, 90], highs=[90, 101, 90], lows=[90, 90, 90])
    o = evaluate_touch(b, Zone(100, 110, "t"), 90, T0)
    assert o.touched and o.first_touch_idx == 1


def test_untouched_reports_closest_approach():
    b = tape([90, 95, 92])
    o = evaluate_touch(b, Zone(100, 110, "t"), 90, T0)
    assert not o.touched and o.max_approach_t == 5


def test_time_to_touch_and_horizon_flags():
    b = tape([90] * 100 + [105])
    o = evaluate_touch(b, Zone(100, 110, "t"), 90, T0, horizons_min=(1, 5))
    assert o.minutes_to_touch == pytest.approx(100 / 60)
    assert o.touched_by[1] is False and o.touched_by[5] is True


def test_open_inside_is_flagged_not_counted_as_arrival():
    b = tape([105, 106])
    o = evaluate_touch(b, Zone(100, 110, "t"), 105, T0)
    assert o.open_inside and o.side == "inside"


def test_first_touch_order_ranks_competing_zones():
    b = tape([100, 90, 80, 120])
    zones = [Zone(118, 122, "up"), Zone(78, 82, "dn"), Zone(500, 510, "far")]
    assert first_touch_order(b, zones) == [1, 0]


# --------------------------------------------------------------- reaction

def test_rejection_measured_from_the_near_edge_outward():
    """Zone 100-110 met from below, price drops to 80 -> 20 ticks of rejection."""
    b = tape([100, 95, 80, 85])
    r = evaluate_reaction(b, Zone(100, 110, "t"), "above", 0)
    assert r.mfe_rev[1] == 20
    assert r.mfe_cont[1] == 0          # never traded above the far edge


def test_movement_inside_the_zone_is_neither_reaction_nor_failure():
    b = tape([100, 105, 109, 104])
    r = evaluate_reaction(b, Zone(100, 110, "t"), "above", 0)
    assert r.mfe_rev[1] == 0 and r.mfe_cont[1] == 0
    assert r.secs_inside == 4 and r.secs_inside_share == 1.0


def test_breakthrough_counts_as_continuation():
    b = tape([100, 108, 115, 130])
    r = evaluate_reaction(b, Zone(100, 110, "t"), "above", 0)
    assert r.mfe_cont[1] == 20 and r.traversed


def test_support_side_is_the_mirror_image():
    b = tape([110, 118, 130])
    r = evaluate_reaction(b, Zone(100, 110, "t"), "below", 0)
    assert r.mfe_rev[1] == 20 and r.mfe_cont[1] == 0


def test_first_passage_prefers_whichever_came_first():
    up_first = tape([100, 90, 60])          # rejects 40 ticks (10 pts) fast
    r = evaluate_reaction(up_first, Zone(100, 110, "t"), "above", 0)
    assert r.first_passage["rev10_adv5"] == 1
    # 140 is 30 ticks past the far edge = 7.5 pts of continuation, which clears
    # adv5 before the 10-pt rejection arrives a second later.
    down_first = tape([100, 140, 60])
    r2 = evaluate_reaction(down_first, Zone(100, 110, "t"), "above", 0)
    assert r2.first_passage["rev10_adv5"] == 0


def test_first_passage_unresolved_is_none_not_false():
    b = tape([100, 102, 104])               # never leaves the zone
    r = evaluate_reaction(b, Zone(100, 110, "t"), "above", 0)
    assert r.first_passage["rev10_adv5"] is None
    assert clean_rejection(r) is None


def test_crossing_counts_ignore_dwell_and_jitter():
    # below -> inside -> above -> inside -> below  == 2 full crossings
    pos = np.array([-1, -1, 0, 0, 1, 1, 0, -1], dtype=np.int8)
    assert crossing_counts(pos) == (2, 2)
    assert crossing_counts(np.array([], dtype=np.int8)) == (0, 0)


def test_chop_and_clean_reaction_separate_the_two_shapes():
    clean = tape(list(range(100, 60, -1)))                   # straight down
    chop = tape([100, 112, 98, 111, 99, 113, 101] * 6)       # in and out repeatedly
    rc = evaluate_reaction(clean, Zone(100, 110, "t"), "above", 0)
    rh = evaluate_reaction(chop, Zone(100, 110, "t"), "above", 0)
    assert rc.directional_efficiency > rh.directional_efficiency
    assert chop_score(rc) < chop_score(rh)
    assert rc.n_crossings < rh.n_crossings


def test_efficiency_is_undefined_rather_than_perfect_on_a_tiny_window():
    """Two samples are always a straight line; that must not read as 'clean'."""
    r = evaluate_reaction(tape([100, 105, 98]), Zone(100, 110, "t"), "above", 0)
    assert np.isnan(r.directional_efficiency)


def test_cleanliness_uses_a_fixed_window_so_touches_are_comparable():
    """A late touch must not look 'cleaner' merely by having less data."""
    long_tape = tape([105] * 3000)
    r = evaluate_reaction(long_tape, Zone(100, 110, "t"), "above", 0, clean_window_s=900)
    assert r.clean_window_s == 900 and r.secs_inside == 900
    short = tape([105] * 100)
    r2 = evaluate_reaction(short, Zone(100, 110, "t"), "above", 0, clean_window_s=900)
    assert r2.clean_window_s == 100 and r2.secs_inside_share == 1.0


def test_penetration_is_depth_past_the_near_edge():
    b = tape([100, 104, 102])
    r = evaluate_reaction(b, Zone(100, 110, "t"), "above", 0)
    assert r.max_penetration_t == 4 and not r.traversed


def test_reclaim_requires_a_prior_breakthrough():
    b = tape([100, 115, 120, 108, 95])       # breaks up, comes back below lo
    r = evaluate_reaction(b, Zone(100, 110, "t"), "above", 0)
    assert r.traversed and r.reclaimed and r.secs_to_reclaim is not None
    never = tape([100, 95, 90])
    r2 = evaluate_reaction(never, Zone(100, 110, "t"), "above", 0)
    assert r2.reclaimed is None


def test_empty_forward_window_is_safe():
    b = tape([100])
    r = evaluate_reaction(b.iloc[0:0], Zone(100, 110, "t"), "above", 0)
    assert r.n_seconds == 0
