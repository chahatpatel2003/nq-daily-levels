"""Leakage checks that run as part of a build, not as an afterthought.

Look-ahead bias does not announce itself; it shows up as a good result. The
checks here are cheap enough to run on every experiment, and any experiment
reported without them should be treated as unverified.

The specific risks this project carries, and the check that catches each:

============================== ==========================================
risk                           check
============================== ==========================================
future bars in a pre-open      :func:`check_decision_cut`
feature
a session's own ATR / range    :func:`check_prefix_invariance`
in its own scaling
outcome columns joined into    :func:`check_feature_columns`
the feature matrix
the same information used to   :func:`check_truncation_invariance`
build a level and to score it
contract-roll contamination    :func:`check_roll_causality`
train/test normalisation       :func:`check_fit_on_train_only`
holdout reuse                  ``eval.splits.holdout_sessions`` (sealed)
============================== ==========================================

:func:`check_truncation_invariance` is the strongest of them and the one worth
running when something looks too good: rebuild a session's candidates from a
tape that has been physically truncated at the decision instant. If any zone
moves, the generator was reading the future, whatever the code appeared to say.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..build.daily import OUTCOME_SUFFIXES
from ..clock import Calendar, decision_time_utc


@dataclass
class AuditResult:
    name: str
    passed: bool
    detail: str = ""
    n_checked: int = 0

    def __str__(self) -> str:
        return (f"[{'PASS' if self.passed else 'FAIL'}] {self.name}"
                + (f" — {self.detail}" if self.detail else ""))


@dataclass
class AuditReport:
    results: List[AuditResult] = field(default_factory=list)

    def add(self, r: AuditResult) -> AuditResult:
        self.results.append(r)
        return r

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def to_dict(self) -> Dict:
        return {"passed": self.passed,
                "n_pass": sum(r.passed for r in self.results),
                "n_total": len(self.results),
                "results": [r.__dict__ for r in self.results]}

    def __str__(self) -> str:
        head = "\n".join(str(r) for r in self.results)
        return f"{head}\n{sum(r.passed for r in self.results)}/{len(self.results)} passed"


# --------------------------------------------------------------------------- #


def check_decision_cut(frames: Dict[str, pd.DataFrame], sess: date, cal: Calendar,
                       ts_col: str = "ts") -> AuditResult:
    """No pre-open frame may contain a stamp at or after the cut."""
    cut = pd.Timestamp(decision_time_utc(sess, cal))
    bad, n = {}, 0
    for name, df in frames.items():
        if df is None or df.empty or ts_col not in df:
            continue
        n += len(df)
        k = int((pd.DatetimeIndex(df[ts_col]) >= cut).sum())
        if k:
            bad[name] = k
    return AuditResult("decision-cut containment", not bad,
                       f"{bad}" if bad else f"cut {cut.isoformat()}", n)


def check_feature_columns(cols: Iterable[str],
                          extra_forbidden: Sequence[str] = ()) -> AuditResult:
    """Refuse outcome-suffixed columns in a feature matrix."""
    cols = list(cols)
    bad = [c for c in cols
           if any(c.endswith(s) or f"{s}_" in c for s in OUTCOME_SUFFIXES)
           or c in set(extra_forbidden)]
    return AuditResult("no outcome columns among features", not bad,
                       f"forbidden: {bad[:10]}" if bad else f"{len(cols)} columns clean",
                       len(cols))


def check_prefix_invariance(transform: Callable[[pd.DataFrame], pd.DataFrame],
                            df: pd.DataFrame, cols: Sequence[str],
                            cut_points: Sequence[int] = (),
                            session_col: str = "session") -> AuditResult:
    """A causal transform's value at row *i* must not depend on rows after *i*.

    Applies ``transform`` to the full frame and to truncated prefixes, and
    compares the overlapping rows. Deterministic, and it needs no assumption
    about which statistic was used — unlike a correlation heuristic, which on a
    trending series (exactly what price is) cannot separate a shifted rolling
    mean from an unshifted one at all.

    This is the check that catches the classic mistake of scaling a session by
    an ATR window that includes the session's own range.
    """
    d = df.sort_values(session_col).reset_index(drop=True)
    full = transform(d).reset_index(drop=True)
    n = len(d)
    if not cut_points:
        cut_points = [k for k in (n // 4, n // 2, (3 * n) // 4) if k > 5]
    bad: Dict[str, int] = {}
    for k in cut_points:
        part = transform(d.iloc[:k].copy()).reset_index(drop=True)
        for c in cols:
            if c not in full or c not in part:
                bad[f"{c}@missing"] = 1
                continue
            a = pd.to_numeric(full[c].iloc[:k], errors="coerce").to_numpy(dtype="float64")
            b = pd.to_numeric(part[c], errors="coerce").to_numpy(dtype="float64")
            m = int(np.sum(~np.isclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-9)))
            if m:
                bad[f"{c}@cut{k}"] = m
    return AuditResult("prefix invariance (no future in a causal column)", not bad,
                       f"differing rows: {bad}" if bad
                       else f"{len(cols)} column(s) stable across "
                            f"{len(cut_points)} truncations", n)


def check_roll_causality(roll_log: Sequence[Dict]) -> AuditResult:
    """Every roll must be decided from a strictly earlier session."""
    bad = []
    for r in roll_log:
        if r.get("is_bootstrap"):
            continue
        s = pd.Timestamp(r["session"]).date()
        d = pd.Timestamp(r["decided_from_session"]).date()
        if d >= s:
            bad.append((str(s), str(d)))
    return AuditResult("roll decided from strictly prior session", not bad,
                       f"violations: {bad}" if bad else
                       f"{len(roll_log)} selections checked", len(roll_log))


def check_truncation_invariance(build_fn: Callable[[pd.DataFrame], pd.DataFrame],
                                full_tape: pd.DataFrame, sess: date, cal: Calendar,
                                ts_col: str = "ts",
                                key_cols: Sequence[str] = ("price_t",)) -> AuditResult:
    """Rebuild candidates from a physically truncated tape and compare.

    The strongest available check, because it does not rely on reading the
    code correctly. ``build_fn`` is handed first the full session and then only
    the rows before the cut; if the two candidate sets differ, something in the
    generator saw past the cut.
    """
    cut = pd.Timestamp(decision_time_utc(sess, cal))
    full = build_fn(full_tape)
    trunc = build_fn(full_tape[pd.DatetimeIndex(full_tape[ts_col]) < cut])
    if full is None or trunc is None:
        return AuditResult("truncation invariance", False, "builder returned None")
    if len(full) != len(trunc):
        return AuditResult("truncation invariance", False,
                           f"{len(full)} candidates from full tape vs {len(trunc)} "
                           f"from truncated", len(full))
    diffs = {}
    for c in key_cols:
        if c in full.columns and c in trunc.columns:
            a = full[c].to_numpy(dtype="float64")
            b = trunc[c].to_numpy(dtype="float64")
            k = int(np.sum(~np.isclose(a, b, equal_nan=True)))
            if k:
                diffs[c] = k
    return AuditResult("truncation invariance", not diffs,
                       f"differing values: {diffs}" if diffs
                       else f"{len(full)} candidates identical", len(full))


def check_fit_on_train_only(fit_sessions: Iterable[date],
                            eval_sessions: Iterable[date]) -> AuditResult:
    """Anything fitted (baseline curves, widths, scalers) must not have seen
    the sessions it is evaluated on."""
    f, e = set(fit_sessions), set(eval_sessions)
    overlap = sorted(f & e)
    return AuditResult("fit set disjoint from eval set", not overlap,
                       f"{len(overlap)} overlapping sessions, first {overlap[0]}"
                       if overlap else f"{len(f)} fit / {len(e)} eval sessions",
                       len(f) + len(e))


def check_label_window(labels: pd.DataFrame, sess_col: str = "session",
                       ts_col: str = "first_touch_ts", cal: Optional[Calendar] = None
                       ) -> AuditResult:
    """Touch stamps must fall inside their own session's target window."""
    if cal is None or labels.empty or ts_col not in labels:
        return AuditResult("labels inside target window", True, "nothing to check")
    from ..clock import target_window_utc
    bad = 0
    for s, g in labels.dropna(subset=[ts_col]).groupby(sess_col):
        lo, hi = target_window_utc(s, cal)
        t = pd.DatetimeIndex(g[ts_col])
        bad += int(((t < lo) | (t >= hi)).sum())
    return AuditResult("labels inside target window", bad == 0,
                       f"{bad} out-of-window touches", len(labels))
