"""Session clock for NQ daily-levels research.

Two timezones, both resolved through :mod:`zoneinfo` so DST is automatic and no
UTC offset is ever hard-coded:

* **exchange clock** (``America/Chicago``) defines what a *session* is. CME
  Globex opens 17:00 CT and that evening belongs to the NEXT calendar day's
  trade date; the session closes 16:00 CT. Holidays and early-close halts come
  from ``configs/cme_calendar.yaml``, which was measured from recorded Globex
  behaviour by the earlier NQ program and is vendored here unchanged.

* **trader clock** (``America/Los_Angeles``) defines the windows the research is
  judged on: the 06:25 PT decision instant and the 06:30-09:00 PT target
  window. PT and CT share US DST rules, so 06:30 PT is 08:30 CT on every trade
  date, but both are still computed from wall-clock time rather than assumed.

The single most important invariant in the project lives here:
:func:`decision_time_utc` is the causal cut. Anything a candidate zone or a
pre-open feature is allowed to see must be strictly earlier than that instant.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import yaml

UTC = ZoneInfo("UTC")

# --------------------------------------------------------------------------- #
# calendar
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Calendar:
    """CME session calendar plus the trader-facing window definitions."""

    exchange_tz: str = "America/Chicago"
    trader_tz: str = "America/Los_Angeles"
    globex_open_ct: time = time(17, 0)
    globex_close_ct: time = time(16, 0)
    # Trade dates the exchange is closed for. Activity from the prior 17:00 CT
    # open through the holiday halt books to the NEXT trade date.
    holidays: frozenset = frozenset()
    # calendar date -> Globex halt time (CT) for shortened sessions.
    early_closes: Dict[date, time] = None  # type: ignore[assignment]

    decision_pt: time = time(6, 25)
    target_start_pt: time = time(6, 30)
    target_end_pt: time = time(9, 0)
    rth_start_pt: time = time(6, 30)
    rth_end_pt: time = time(13, 0)

    @property
    def xtz(self) -> ZoneInfo:
        return ZoneInfo(self.exchange_tz)

    @property
    def ttz(self) -> ZoneInfo:
        return ZoneInfo(self.trader_tz)


def _as_time(v) -> time:
    if isinstance(v, time):
        return v
    h, m = str(v).split(":")[:2]
    return time(int(h), int(m))


def _as_date(v) -> date:
    return v if isinstance(v, date) and not isinstance(v, datetime) \
        else date.fromisoformat(str(v))


@lru_cache(maxsize=4)
def load_calendar(configs_dir: str) -> Calendar:
    """Build the calendar from ``cme_calendar.yaml`` + ``research.yaml``."""
    cdir = Path(configs_dir)
    cal = (yaml.safe_load((cdir / "cme_calendar.yaml").read_text()) or {})["sessions"]
    res = yaml.safe_load((cdir / "research.yaml").read_text()) or {}
    w = res.get("windows", {})
    ck = res.get("clock", {})
    return Calendar(
        exchange_tz=ck.get("exchange_tz", "America/Chicago"),
        trader_tz=ck.get("trader_tz", "America/Los_Angeles"),
        globex_open_ct=_as_time(cal.get("globex_open_ct", "17:00")),
        globex_close_ct=_as_time(cal.get("globex_close_ct", "16:00")),
        holidays=frozenset(_as_date(d) for d in cal.get("holidays", [])),
        early_closes={_as_date(k): _as_time(v)
                      for k, v in (cal.get("early_closes") or {}).items()},
        decision_pt=_as_time(w.get("decision_pt", "06:25")),
        target_start_pt=_as_time(w.get("target_start_pt", "06:30")),
        target_end_pt=_as_time(w.get("target_end_pt", "09:00")),
        rth_start_pt=_as_time(w.get("rth_start_pt", "06:30")),
        rth_end_pt=_as_time(w.get("rth_end_pt", "13:00")),
    )


# --------------------------------------------------------------------------- #
# trade dates
# --------------------------------------------------------------------------- #


def is_trade_date(d: date, cal: Calendar) -> bool:
    return d.weekday() < 5 and d not in cal.holidays


def _require_trade_date(d: date, cal: Calendar) -> date:
    """Reject datetimes outright.

    ``datetime`` subclasses ``date``, so a timestamp passed here would silently
    lose its time component and resolve to the wrong session for any evening
    stamp — the classic off-by-one-session bug. Callers convert first with
    :func:`trade_date`.
    """
    if isinstance(d, datetime):
        raise TypeError("expected a trade date; call trade_date(ts, cal) first")
    if not is_trade_date(d, cal):
        raise ValueError(f"{d} is not a trade date (weekend or holiday)")
    return d


def trade_date(ts: datetime, cal: Calendar) -> date:
    """CME trade date a UTC instant belongs to.

    At/after 17:00 CT the stamp belongs to the next calendar day; weekends and
    holidays roll forward, so Sunday evening books to Monday and a Monday
    holiday's whole stretch books to Tuesday.
    """
    if ts.tzinfo is None:
        raise ValueError("naive datetime; all timestamps must be tz-aware UTC")
    ct = ts.astimezone(cal.xtz)
    d = ct.date()
    if ct.time() >= cal.globex_open_ct:
        d += timedelta(days=1)
    while not is_trade_date(d, cal):
        d += timedelta(days=1)
    return d


def trade_dates(start: date, end: date, cal: Calendar) -> List[date]:
    out, d = [], start
    while d <= end:
        if is_trade_date(d, cal):
            out.append(d)
        d += timedelta(days=1)
    return out


def prior_trade_date(d: date, cal: Calendar, n: int = 1) -> date:
    for _ in range(n):
        d -= timedelta(days=1)
        while not is_trade_date(d, cal):
            d -= timedelta(days=1)
    return d


def next_trade_date(d: date, cal: Calendar) -> date:
    d += timedelta(days=1)
    while not is_trade_date(d, cal):
        d += timedelta(days=1)
    return d


# --------------------------------------------------------------------------- #
# instants
# --------------------------------------------------------------------------- #


def ct_instant(d: date, t: time, cal: Calendar) -> datetime:
    """UTC instant of a CT wall-clock time on calendar date ``d``."""
    return datetime.combine(d, t, tzinfo=cal.xtz).astimezone(UTC)


def pt_instant(d: date, t: time, cal: Calendar) -> datetime:
    """UTC instant of a PT wall-clock time on calendar date ``d``."""
    return datetime.combine(d, t, tzinfo=cal.ttz).astimezone(UTC)


def session_open_utc(sess: date, cal: Calendar) -> datetime:
    """When this trade date's Globex session opened.

    17:00 CT on the previous calendar day, walking back over Saturdays and over
    *halt* holidays (whose entire stretch books forward to this trade date).
    Full-closure holidays are not skipped: the exchange reopens 17:00 CT on the
    holiday itself.
    """
    _require_trade_date(sess, cal)
    prior = sess - timedelta(days=1)
    while prior.weekday() == 5 or (prior in cal.holidays
                                   and prior in cal.early_closes):
        prior -= timedelta(days=1)
    return ct_instant(prior, cal.globex_open_ct, cal)


def session_close_utc(sess: date, cal: Calendar) -> datetime:
    """Globex close, or the halt time on an early-close trade date."""
    _require_trade_date(sess, cal)
    t = cal.early_closes.get(sess, cal.globex_close_ct)
    return ct_instant(sess, min(t, cal.globex_close_ct), cal)


def decision_time_utc(sess: date, cal: Calendar) -> datetime:
    """The causal cut. Zones for ``sess`` may use data strictly before this."""
    _require_trade_date(sess, cal)
    return pt_instant(sess, cal.decision_pt, cal)


def target_window_utc(sess: date, cal: Calendar) -> Tuple[datetime, datetime]:
    """[06:30, 09:00) PT — the window touch/reaction labels are measured in."""
    _require_trade_date(sess, cal)
    return (pt_instant(sess, cal.target_start_pt, cal),
            pt_instant(sess, cal.target_end_pt, cal))


def rth_window_utc(sess: date, cal: Calendar) -> Tuple[datetime, datetime]:
    _require_trade_date(sess, cal)
    return (pt_instant(sess, cal.rth_start_pt, cal),
            pt_instant(sess, cal.rth_end_pt, cal))


def segment_windows_utc(sess: date, cal: Calendar) -> Dict[str, Tuple[datetime, datetime]]:
    """Pre-open segments for ``sess``, all ending at or before the decision cut.

    ``globex`` is the authoritative one: everything the session has traded so
    far, from the real Globex open to the cut. It starts at
    :func:`session_open_utc` rather than at a fixed wall-clock time, because
    after a halt holiday the session opened the *previous Sunday* — the Tuesday
    following MLK Monday has been trading since Sunday 17:00 CT, including the
    thin Monday-morning holiday session, and that stretch carries real
    overnight highs and lows.

    ``asia`` / ``london`` / ``premarket`` are the ordinary wall-clock thirds of
    the most recent night. ``extended`` holds whatever the globex window has
    *before* those begin, and is empty on a normal session.

    Every window is clipped to the decision cut, so a segment can never leak
    across it.
    """
    _require_trade_date(sess, cal)
    open_utc = session_open_utc(sess, cal)
    cut = decision_time_utc(sess, cal)
    prev_cal_day = sess - timedelta(days=1)
    night_start = pt_instant(prev_cal_day, time(15, 0), cal)

    raw = {
        "globex":    (open_utc, cut),
        "extended":  (open_utc, night_start),
        "asia":      (night_start, pt_instant(sess, time(0, 0), cal)),
        "london":    (pt_instant(sess, time(0, 0), cal),
                      pt_instant(sess, time(5, 0), cal)),
        "premarket": (pt_instant(sess, time(5, 0), cal), cut),
    }
    out = {}
    for name, (lo, hi) in raw.items():
        lo, hi = max(lo, open_utc), min(hi, cut)
        if hi > lo:
            out[name] = (lo, hi)
    return out


def prior_rth_window_utc(sess: date, cal: Calendar, n: int = 1) -> Tuple[datetime, datetime]:
    """Full RTH of the n-th prior trade date (source of PDH/PDL/PDC/prior POC)."""
    return rth_window_utc(prior_trade_date(sess, cal, n), cal)


# --------------------------------------------------------------------------- #
# session integrity
# --------------------------------------------------------------------------- #


def halt_time_utc(sess: date, cal: Calendar) -> Optional[datetime]:
    t = cal.early_closes.get(sess)
    return ct_instant(sess, t, cal) if t is not None else None


def target_window_is_intact(sess: date, cal: Calendar) -> bool:
    """False when an early-close halt cuts into 06:30-09:00 PT.

    Ordinary CME half days halt at 12:00/12:15 CT (10:00/10:15 PT), safely after
    the window. The exception in the recorded data is Good Friday 2026-04-03,
    which traded only a jobs-report mini session ending 08:15 CT — before the
    window opens at 08:30 CT. Such sessions are excluded from labelling.
    """
    halt = halt_time_utc(sess, cal)
    if halt is None:
        return True
    return halt >= target_window_utc(sess, cal)[1]


def usable_sessions(start: date, end: date, cal: Calendar) -> List[date]:
    """Trade dates whose full target window was open for business."""
    return [d for d in trade_dates(start, end, cal)
            if target_window_is_intact(d, cal)]


# --------------------------------------------------------------------------- #
# leakage guard
# --------------------------------------------------------------------------- #


class CausalityError(AssertionError):
    """Raised when data at/after the decision instant reaches a pre-open path."""


def assert_before_decision(stamps: Iterable[datetime], sess: date,
                           cal: Calendar, what: str = "feature input") -> None:
    """Hard check that every stamp is strictly before the decision cut.

    Cheap enough to call on every pre-open build path, and it is: the whole
    project's credibility rests on this one inequality.
    """
    cut = decision_time_utc(sess, cal)
    bad = [s for s in stamps if s >= cut]
    if bad:
        raise CausalityError(
            f"{what} for session {sess} contains {len(bad)} stamp(s) at/after "
            f"the {cal.decision_pt} PT decision cut ({cut.isoformat()}); "
            f"first offender {min(bad).isoformat()}")
