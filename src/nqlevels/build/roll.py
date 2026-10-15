"""Contract rollover for NQ.

NQ trades quarterly (H/M/U/Z). Liquidity migrates to the next month a few days
before expiry, and the two contracts differ in price by the cost of carry minus
expected dividends — for NQ that is tens of index points, far larger than any
zone this project will draw. Getting the roll wrong therefore does not blur a
level, it invents one.

Three decisions, kept separate on purpose:

1. **Which contract is "the market" for session S?**  The one with the most
   volume on session *S-1*. Deciding from S's own volume would let the session
   pick its own contract using data from after the decision cut. The cost of
   that discipline is that the roll lands one session after the true volume
   crossover, which is exactly the lag a live 06:25 run would also have.

2. **Which contracts get a full L1 build?**  Every quarterly outright above a
   small volume share, not just the front. Across a roll the *incoming* month
   already prints all session, so its own prior-session high/low/profile exist
   in its own price scale — no adjustment needed, no fabricated gap.

3. **How is a prior-session level carried across a roll?**  Preference order:
   (a) recompute it from the incoming contract's own prints (exact, preferred);
   (b) if the incoming contract was too thin on that session, shift the
   outgoing contract's level by the roll spread measured *before* the decision
   cut. Sessions where neither is clean are flagged, never silently patched.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

QUARTERLY_MONTH = {"H": 3, "M": 6, "U": 9, "Z": 12}
_CONTRACT_RE = re.compile(r"^NQ([HMUZ])(\d{1,2})$")


def parse_contract(symbol: str, ref_year: int) -> Optional[Tuple[int, int]]:
    """``NQH4`` -> ``(2024, 3)``.

    Databento emits single-digit year codes, which are ambiguous across
    decades; they are resolved to the smallest year at or after
    ``ref_year - 1`` whose last digit matches. Returns ``None`` for spreads and
    non-quarterly symbols, which are never tradeable fronts.
    """
    m = _CONTRACT_RE.match(symbol)
    if not m:
        return None
    month = QUARTERLY_MONTH[m.group(1)]
    digits = m.group(2)
    if len(digits) == 2:
        return 2000 + int(digits), month
    d, y = int(digits), ref_year - 1
    while y % 10 != d:
        y += 1
    return y, month


def is_quarterly_outright(symbol: str) -> bool:
    return _CONTRACT_RE.match(symbol) is not None


def contract_order(symbol: str, ref_year: int) -> Tuple[int, int]:
    c = parse_contract(symbol, ref_year)
    return c if c is not None else (9999, 99)


@dataclass
class RollEvent:
    session: date
    previous: Optional[str]
    front: str
    decided_from_session: date
    prev_volume: int
    new_volume: int

    @property
    def is_bootstrap(self) -> bool:
        """First selection of a run, not a contract change.

        Counted separately so audits report the true number of rolls; a
        bootstrap tells you the run started, a roll tells you the price scale
        moved and prior-session levels need carrying.
        """
        return self.previous is None


class FrontMonthSelector:
    """Picks session S's front month from session S-1's volumes only.

    Usage contract, enforced by :meth:`select` raising if violated: call
    ``select(S)`` *before* ``observe(S, volumes)``. The selector is monotone —
    it never rolls back to an earlier contract, so a single thin day in the new
    front cannot bounce the series between two price scales.
    """

    def __init__(self) -> None:
        self._prior_session: Optional[date] = None
        self._prior_volumes: Dict[str, int] = {}
        self._current: Optional[str] = None
        self._last_selected: Optional[date] = None
        self.roll_log: List[RollEvent] = []

    @property
    def current(self) -> Optional[str]:
        return self._current

    def select(self, session: date) -> Optional[str]:
        if self._last_selected is not None and session <= self._last_selected:
            raise ValueError(
                f"sessions must be selected in strictly increasing order; "
                f"got {session} after {self._last_selected}")
        self._last_selected = session
        if not self._prior_volumes:
            return None                       # bootstrap: nothing observed yet
        ref = session.year
        cur_key = contract_order(self._current, ref) if self._current else None
        best, best_vol = None, -1
        for sym, vol in self._prior_volumes.items():
            if not is_quarterly_outright(sym):
                continue
            if cur_key is not None and contract_order(sym, ref) < cur_key:
                continue                      # monotone: never roll backwards
            if vol > best_vol:
                best, best_vol = sym, int(vol)
        if best is None:
            best = self._current
        if best != self._current:
            self.roll_log.append(RollEvent(
                session=session, previous=self._current, front=best,
                decided_from_session=self._prior_session,      # type: ignore[arg-type]
                prev_volume=int(self._prior_volumes.get(self._current or "", 0)),
                new_volume=best_vol))
        self._current = best
        return best

    def observe(self, session: date, volumes: Dict[str, int]) -> None:
        self._prior_session = session
        self._prior_volumes = dict(volumes)


def symbols_to_materialize(volumes: Dict[str, int], min_share: float) -> List[str]:
    """Quarterly outrights worth a full L1 build for one session.

    Normally the front alone; two or three across a roll. Materialising the
    incoming month is what lets prior-session levels be recomputed in the
    incoming scale instead of shifted.
    """
    q = {s: v for s, v in volumes.items() if is_quarterly_outright(s)}
    total = sum(q.values())
    if total <= 0:
        return []
    return sorted([s for s, v in q.items() if v / total >= min_share],
                  key=lambda s: -q[s])


def roll_spread(trades: pd.DataFrame, outgoing: str, incoming: str,
                min_pairs: int = 200,
                max_lag: pd.Timedelta = pd.Timedelta("1s")) -> Optional[float]:
    """``incoming - outgoing`` price offset from contemporaneous prints.

    Pairs each incoming print with the most recent outgoing print within
    ``max_lag`` and takes the median difference; the median resists the
    one-sided prints that dominate a thin contract. ``trades`` must already be
    restricted to a window ending before the decision cut — this function does
    not know the cut and will not check it.

    Returns ``None`` when there is not enough overlap to measure honestly,
    which the caller must treat as "cannot carry this level", not as zero.
    """
    a = trades[trades["symbol"] == outgoing][["ts_event", "price"]]
    b = trades[trades["symbol"] == incoming][["ts_event", "price"]]
    if len(a) < min_pairs or len(b) < min_pairs:
        return None
    a = a.sort_values("ts_event", kind="stable")
    b = b.sort_values("ts_event", kind="stable")
    m = pd.merge_asof(b, a, on="ts_event", direction="backward",
                      tolerance=max_lag, suffixes=("_in", "_out")).dropna()
    if len(m) < min_pairs:
        return None
    return float(np.median(m["price_in"] - m["price_out"]))


def carry_level(level: float, outgoing: str, incoming: str,
                spread: Optional[float]) -> Tuple[Optional[float], str]:
    """Express an outgoing-contract level in the incoming contract's scale.

    Returns ``(value, method)``; ``method`` is ``'same_contract'`` when no roll
    happened, ``'spread_shift'`` when the offset was applied, and
    ``'unavailable'`` when the spread could not be measured. Callers record the
    method so roll-adjacent sessions can be analysed — or excluded — as a
    group rather than trusted blindly.
    """
    if outgoing == incoming:
        return level, "same_contract"
    if spread is None or not np.isfinite(spread):
        return None, "unavailable"
    return level + spread, "spread_shift"
