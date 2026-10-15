"""Session-clock tests. These guard the one invariant the project rests on."""
from __future__ import annotations

from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from nqlevels.clock import (CausalityError, assert_before_decision,
                            decision_time_utc, load_calendar, prior_trade_date,
                            rth_window_utc, segment_windows_utc,
                            session_close_utc, session_open_utc,
                            target_window_utc, trade_date, trade_dates,
                            usable_sessions)

CONFIGS = str(Path(__file__).resolve().parents[1] / "configs")
UTC = ZoneInfo("UTC")


@pytest.fixture(scope="module")
def cal():
    return load_calendar(CONFIGS)


def u(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


# --------------------------------------------------------------- trade dates

def test_evening_books_to_next_trade_date(cal):
    assert trade_date(u("2024-03-04 23:30"), cal) == date(2024, 3, 5)   # 17:30 CT Mon
    assert trade_date(u("2024-03-05 21:30"), cal) == date(2024, 3, 5)   # 15:30 CT Tue


def test_sunday_evening_books_to_monday(cal):
    assert trade_date(u("2024-03-03 23:30"), cal) == date(2024, 3, 4)


def test_halt_holiday_books_forward(cal):
    """MLK Monday's whole stretch clears to Tuesday, as CME books it."""
    assert trade_date(u("2024-01-14 23:30"), cal) == date(2024, 1, 16)
    assert trade_date(u("2024-01-15 15:00"), cal) == date(2024, 1, 16)
    assert date(2024, 1, 15) not in trade_dates(date(2024, 1, 12), date(2024, 1, 20), cal)


def test_naive_datetime_rejected(cal):
    with pytest.raises(ValueError):
        trade_date(datetime(2024, 3, 5, 12, 0), cal)


def test_timestamp_rejected_where_trade_date_expected(cal):
    """A datetime silently truncated to its date is the classic session bug."""
    with pytest.raises(TypeError):
        target_window_utc(datetime(2024, 3, 5, 23, 0, tzinfo=UTC), cal)


# ---------------------------------------------------------------------- DST

@pytest.mark.parametrize("d", [date(2024, 3, 8), date(2024, 3, 11),
                               date(2024, 11, 1), date(2024, 11, 4),
                               date(2021, 1, 5), date(2026, 6, 16)])
def test_decision_is_always_0625_pt(cal, d):
    t = decision_time_utc(d, cal).astimezone(cal.ttz)
    assert (t.hour, t.minute) == (6, 25)


@pytest.mark.parametrize("d", [date(2024, 3, 8), date(2024, 3, 11)])
def test_target_window_is_always_0630_0900_pt(cal, d):
    lo, hi = target_window_utc(d, cal)
    assert lo.astimezone(cal.ttz).strftime("%H:%M") == "06:30"
    assert hi.astimezone(cal.ttz).strftime("%H:%M") == "09:00"
    assert (hi - lo).total_seconds() == 2.5 * 3600


def test_utc_offset_actually_shifts_across_dst(cal):
    """Guards against a hard-coded offset that happens to pass in one month."""
    a = decision_time_utc(date(2024, 3, 8), cal)
    b = decision_time_utc(date(2024, 3, 11), cal)
    assert a.strftime("%H:%M") == "14:25" and b.strftime("%H:%M") == "13:25"


# ------------------------------------------------------------------ windows

def test_decision_precedes_target(cal):
    for d in trade_dates(date(2024, 1, 2), date(2024, 3, 31), cal):
        assert decision_time_utc(d, cal) < target_window_utc(d, cal)[0]


def test_segments_never_cross_the_cut(cal):
    for d in trade_dates(date(2024, 1, 2), date(2024, 2, 29), cal):
        cut = decision_time_utc(d, cal)
        for name, (lo, hi) in segment_windows_utc(d, cal).items():
            assert hi <= cut, f"{name} on {d} ends after the cut"
            assert lo >= session_open_utc(d, cal)


def test_post_holiday_globex_starts_at_the_real_open(cal):
    """Tuesday after MLK has been trading since Sunday 17:00 CT."""
    d = date(2024, 1, 16)
    segs = segment_windows_utc(d, cal)
    assert segs["globex"][0] == session_open_utc(d, cal)
    assert segs["globex"][0].astimezone(cal.xtz).strftime("%a %H:%M") == "Sun 17:00"
    assert "extended" in segs


def test_normal_session_has_no_extended_segment(cal):
    assert "extended" not in segment_windows_utc(date(2024, 3, 5), cal)


def test_rth_contains_target(cal):
    d = date(2024, 3, 5)
    r, t = rth_window_utc(d, cal), target_window_utc(d, cal)
    assert r[0] == t[0] and r[1] > t[1]


def test_session_bounds_ordered(cal):
    for d in trade_dates(date(2024, 1, 2), date(2024, 12, 31), cal):
        assert session_open_utc(d, cal) < decision_time_utc(d, cal) < session_close_utc(d, cal)


# ----------------------------------------------------------------- guardrail

def test_assert_before_decision_catches_a_late_stamp(cal):
    d = date(2024, 3, 5)
    cut = decision_time_utc(d, cal)
    assert_before_decision([cut.replace(microsecond=0) - __import__("datetime").timedelta(seconds=1)], d, cal)
    with pytest.raises(CausalityError):
        assert_before_decision([cut], d, cal)


def test_usable_sessions_count(cal):
    s = usable_sessions(date(2021, 1, 4), date(2026, 6, 17), cal)
    assert len(s) == 1371                      # matches the L1 build (+ the seed session)
    assert s[0] == date(2021, 1, 4) and s[-1] == date(2026, 6, 17)


def test_prior_trade_date_skips_weekends_and_holidays(cal):
    assert prior_trade_date(date(2024, 3, 4), cal) == date(2024, 3, 1)
    assert prior_trade_date(date(2024, 1, 16), cal) == date(2024, 1, 12)  # skips MLK
