"""Chronological splits, walk-forward folds, and the holdout seal.

Three rules, enforced rather than documented:

1. **Never shuffle.** Sessions are ordered and correlated; a random split
   leaks tomorrow into today and makes every result meaningless.
2. **Embargo across every boundary.** A label measured inside a session can
   only use that session, but features look back many sessions, so a training
   session immediately adjacent to a test session shares information with it.
   Sessions inside the embargo belong to neither side.
3. **The holdout is sealed.** :func:`holdout_sessions` refuses to return
   anything without a pre-registration token — a written statement of exactly
   what is being tested, recorded before the data is read. This is the one
   control that cannot be added retroactively: the previous NQ program
   consumed its entire holdout and then had nothing left to confirm its one
   promising result on, and re-running a tail after seeing it fail is not a
   test.

The default span is 2021-01-05 .. 2026-06-17 (1,371 usable sessions).
MBP-1 stops at 2025-06-30, so anything using microstructure runs on a strictly
shorter span; :func:`assert_mbp1_available` makes that explicit instead of
letting a model quietly train on fewer sessions than its comparison.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


class HoldoutSealed(RuntimeError):
    """Raised on any attempt to read the holdout without pre-registration."""


@dataclass(frozen=True)
class Split:
    name: str
    start: date
    end: date

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end


@dataclass
class SplitPlan:
    train: Split
    validation: Split
    holdout: Split
    early_regime: Split
    mbp1_end: date
    embargo_sessions: int = 5

    @property
    def splits(self) -> Dict[str, Split]:
        return {s.name: s for s in (self.train, self.validation, self.holdout,
                                    self.early_regime)}


def load_split_plan(configs_dir: str, embargo_sessions: int = 5) -> SplitPlan:
    cfg = yaml.safe_load((Path(configs_dir) / "research.yaml").read_text())["splits"]

    def mk(name: str) -> Split:
        a, b = cfg[name]
        return Split(name, date.fromisoformat(a), date.fromisoformat(b))

    return SplitPlan(train=mk("train"), validation=mk("validation"),
                     holdout=mk("holdout"), early_regime=mk("early_regime"),
                     mbp1_end=date.fromisoformat(str(cfg["mbp1_end"])),
                     embargo_sessions=embargo_sessions)


def assign(sessions: Sequence[date], plan: SplitPlan) -> Dict[str, List[date]]:
    """Bucket sessions, dropping the embargo band around each boundary."""
    s = sorted(sessions)
    out: Dict[str, List[date]] = {"train": [], "validation": [], "holdout": [],
                                  "embargo": []}
    raw = {n: [d for d in s if sp.contains(d)]
           for n, sp in plan.splits.items() if n != "early_regime"}
    e = plan.embargo_sessions
    for name, days in raw.items():
        if not days:
            continue
        head = days[:e] if name != "train" else []
        tail = days[-e:] if name != "holdout" else []
        keep = [d for d in days if d not in set(head) | set(tail)]
        out[name] = keep
        out["embargo"].extend(head + tail)
    out["embargo"] = sorted(set(out["embargo"]))
    return out


# --------------------------------------------------------------------------- #
# walk-forward
# --------------------------------------------------------------------------- #


@dataclass
class Fold:
    idx: int
    train: List[date]
    test: List[date]

    def summary(self) -> Dict:
        return {"fold": self.idx, "n_train": len(self.train), "n_test": len(self.test),
                "train": f"{self.train[0]}..{self.train[-1]}" if self.train else "",
                "test": f"{self.test[0]}..{self.test[-1]}" if self.test else ""}


def walk_forward(sessions: Sequence[date], train_sessions: int = 250,
                 test_sessions: int = 63, embargo: int = 5,
                 expanding: bool = False) -> List[Fold]:
    """Rolling (or expanding) chronological folds with an embargo gap.

    Defaults are roughly a year of training and a quarter of testing, which
    keeps enough folds to see whether a result is persistent rather than one
    lucky regime.
    """
    s = sorted(sessions)
    folds: List[Fold] = []
    i, k = 0, 0
    while True:
        tr_end = (train_sessions if expanding else i + train_sessions)
        tr_lo = 0 if expanding else i
        if tr_end + embargo + test_sessions > len(s):
            break
        tr = s[tr_lo:i + train_sessions]
        te = s[i + train_sessions + embargo: i + train_sessions + embargo + test_sessions]
        if not te:
            break
        folds.append(Fold(k, tr, te))
        i += test_sessions
        k += 1
    return folds


# --------------------------------------------------------------------------- #
# the seal
# --------------------------------------------------------------------------- #


def preregister(project_root: Path, name: str, hypothesis: str, rule: str,
                params: Dict, gates: Dict, author_note: str = "") -> str:
    """Freeze a hypothesis before touching the holdout; returns its token.

    The token is a hash of the content, so a registration cannot be edited
    after the fact without invalidating the token that the holdout read
    requires.
    """
    body = {"name": name, "hypothesis": hypothesis, "rule": rule,
            "params": params, "gates": gates, "note": author_note,
            "registered_utc": datetime.now(timezone.utc).isoformat()}
    blob = json.dumps(body, sort_keys=True, default=str)
    token = hashlib.sha256(blob.encode()).hexdigest()[:16]
    body["token"] = token
    d = Path(project_root) / "outputs" / "preregistrations"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}_{token}.json").write_text(json.dumps(body, indent=2, default=str))
    return token


def holdout_sessions(sessions: Sequence[date], plan: SplitPlan,
                     project_root: Path, token: Optional[str] = None) -> List[date]:
    """Holdout sessions — only with a valid pre-registration token."""
    if not token:
        raise HoldoutSealed(
            "the holdout is sealed. Call preregister(...) with the exact rule, "
            "parameters and pass/fail gates first, then pass the returned "
            "token. Reading it to 'have a look' spends it permanently.")
    d = Path(project_root) / "outputs" / "preregistrations"
    if not any(p.name.endswith(f"_{token}.json") for p in d.glob("*.json")):
        raise HoldoutSealed(f"no pre-registration found for token {token!r}")
    return [s for s in sorted(sessions) if plan.holdout.contains(s)]


def assert_mbp1_available(sessions: Iterable[date], plan: SplitPlan) -> None:
    """Refuse to run microstructure work past the MBP-1 coverage end."""
    bad = [d for d in sessions if d > plan.mbp1_end]
    if bad:
        raise ValueError(
            f"{len(bad)} requested sessions are after the MBP-1 coverage end "
            f"({plan.mbp1_end}); first is {min(bad)}. Microstructure models run "
            "on a shorter span than trades-only models — state that explicitly "
            "rather than comparing them on unequal data.")
