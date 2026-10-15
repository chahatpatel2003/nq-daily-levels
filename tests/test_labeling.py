"""The vectorised zone-measurement engine, against hand-checked answers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nqlevels.eval.labeling import (SessionPath, _rolling_max, _rolling_min,
                                    clean_rejection, measure_zone)


def path(hi, lo=None, cl=None) -> SessionPath:
    hi = np.asarray(hi, dtype="float64")
    lo = np.asarray(lo if lo is not None else hi, dtype="float64")
    cl = np.asarray(cl if cl is not None else hi, dtype="float64")
    return SessionPath("s", np.arange(hi.size, dtype="int32"), hi, lo, cl)


# ------------------------------------------------------------- primitives

def test_rolling_max_matches_a_naive_loop():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    for w in (1, 2, 7, 50, 199, 200, 500):
        want = np.array([a[i:i + w].max() for i in range(a.size)])
        assert np.allclose(_rolling_max(a, w), want), f"w={w}"


def test_rolling_min_matches_a_naive_loop():
    rng = np.random.default_rng(1)
    a = rng.normal(size=150)
    for w in (1, 3, 40, 150):
        want = np.array([a[i:i + w].min() for i in range(a.size)])
        assert np.allclose(_rolling_min(a, w), want), f"w={w}"


def test_rolling_max_on_empty_and_single():
    assert _rolling_max(np.array([]), 5).size == 0
    assert _rolling_max(np.array([3.0]), 5) == 3.0


# ------------------------------------------------------------------ touch

def test_touch_above_is_the_first_second_the_high_reaches_the_lower_edge():
    p = path([0, 5, 9, 12, 20])
    assert p.first_touch(10, 14) == 3
    assert p.first_touch(21, 25) is None


def test_touch_below_mirrors():
    p = path(hi=[0, -1, -5, -12], lo=[0, -3, -8, -15])
    assert p.first_touch(-12, -8) == 2
    assert p.first_touch(-40, -30) is None


def test_a_second_that_jumps_clean_across_still_counts():
    """Price traded through the area; that is an interaction."""
    p = path(hi=[0, 30], lo=[0, 25])
    assert p.first_touch(10, 14) == 1


def test_side_classification():
    p = path([0, 1])
    assert p.side_of(10, 20) == "above"
    assert p.side_of(-20, -10) == "below"
    assert p.side_of(-5, 5) == "inside"


# --------------------------------------------------------------- reaction

def test_rejection_and_continuation_are_measured_from_the_edges():
    # zone 10..14 above; price touches 10 then falls to -30
    p = path(hi=[0, 10, 5, -10], lo=[0, 10, 0, -30])
    r = measure_zone(p, 10, 14)
    assert r["touched"] and r["first_touch_sec"] == 1
    assert r["rev_full"] == 40           # 10 - (-30)
    assert r["cont_full"] == 0           # never above 14


def test_movement_inside_the_zone_is_neither_reaction_nor_failure():
    p = path(hi=[0, 11, 13, 12], lo=[0, 10, 11, 11])
    r = measure_zone(p, 10, 14)
    assert r["rev_full"] == 0 and r["cont_full"] == 0
    assert r["secs_inside"] == 3         # from the touch second onward


def test_breakthrough_is_continuation_and_sets_traversed():
    p = path(hi=[0, 10, 20, 40], lo=[0, 10, 15, 35])
    r = measure_zone(p, 10, 14)
    assert r["cont_full"] == 26 and r["traversed"]
    assert r["max_penetration_t"] == 30  # past the near edge


def test_support_side_is_the_mirror_image():
    p = path(hi=[0, -10, 20], lo=[0, -14, 10])
    r = measure_zone(p, -14, -10)
    assert r["side"] == "below"
    assert r["rev_full"] == 30           # 20 - (-10)
    assert r["cont_full"] == 0


def test_horizon_excursions_are_truncated_not_extrapolated():
    p = path(hi=[0, 10, 11], lo=[0, 10, -50])
    r = measure_zone(p, 10, 14)
    assert r["rev_60s"] == 60            # only 2 seconds of window existed
    assert r["n_seconds_fwd"] == 2


def test_first_passage_order():
    fast_reject = path(hi=[0, 10, 0], lo=[0, 10, -30])
    assert measure_zone(fast_reject, 10, 14)["fp_rev10_adv5"] == 1
    fast_break = path(hi=[0, 10, 60], lo=[0, 10, 50])
    assert measure_zone(fast_break, 10, 14)["fp_rev10_adv5"] == 0


def test_unresolved_first_passage_is_none_and_drops_from_the_label():
    p = path(hi=[0, 11, 12], lo=[0, 10, 11])
    r = measure_zone(p, 10, 14)
    assert r["fp_rev10_adv5"] is None
    d = pd.DataFrame([r])
    assert clean_rejection(d).isna().all()


def test_clean_rejection_accepts_a_straight_move_away():
    clean = path(hi=np.r_[0, 10, np.arange(0, -80, -2)],
                 lo=np.r_[0, 10, np.arange(-2, -82, -2)])
    d = pd.DataFrame([measure_zone(clean, 10, 14)])
    assert clean_rejection(d).iloc[0] is True or bool(clean_rejection(d).iloc[0])


def test_clean_rejection_is_na_when_price_never_leaves_the_zone():
    loiter = path(hi=np.r_[0, np.tile([11, 13], 400)],
                  lo=np.r_[0, np.tile([10, 12], 400)])
    d = pd.DataFrame([measure_zone(loiter, 10, 14)])
    assert pd.isna(clean_rejection(d).iloc[0])       # unresolved, not a failure


def test_clean_rejection_rejects_a_move_that_was_reached_by_chopping():
    """Same 10-point rejection, but only after crossing the zone repeatedly."""
    # closes alternate above 14 and below 10 — full crossings — but stay inside
    # +-5 points of the edges, so neither first-passage threshold fires early.
    churn = np.tile([18.0, 6.0], 300)
    hi = np.r_[0, 10, churn, np.arange(0, -80, -2)]
    lo = np.r_[0, 10, churn, np.arange(-2, -82, -2)]
    d = pd.DataFrame([measure_zone(path(hi=hi, lo=lo, cl=np.r_[0, 10, churn,
                                                               np.arange(0, -80, -2)]),
                                   10, 14)])
    r = d.iloc[0]
    assert r["n_crossings"] >= 2, r["n_crossings"]
    assert r["fp_rev10_adv5"] == 1                # the move did eventually happen
    assert not bool(clean_rejection(d).iloc[0])   # but not cleanly


def test_slice_through_flag():
    sliced = path(hi=[0, 10, 40], lo=[0, 10, 30])
    assert measure_zone(sliced, 10, 14)["slice_through_60s"]
    held = path(hi=[0, 10, 8], lo=[0, 10, 2])
    assert not measure_zone(held, 10, 14)["slice_through_60s"]


def test_untouched_reports_closest_approach():
    p = path(hi=[0, 5, 7], lo=[0, -3, -1])
    r = measure_zone(p, 10, 14)
    assert not r["touched"] and r["closest_approach_t"] == 3
