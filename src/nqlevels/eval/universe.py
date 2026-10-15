"""Which candidate levels are worth analysing, and what a fair control is.

## The density problem

Measured on train (2026-08-14): pooling all 86 generated level kinds puts
**137 levels per session inside ±1.5 ATR** of the market. On that axis a
uniformly random price sits within 1.28 points of *some* named level at the
median, within 1.5 points 55 % of the time and within 5 points 87 % of the
time. "This price is at a named level" is then true of almost every price and
carries close to no information — and, worse, any placebo drawn by shifting a
level lands on another level, so the control is contaminated and the contrast
is null by construction rather than by evidence.

Thirteen kinds cause it: ``round_25``, ``round_50``, ``round_100``, the HVN/LVN
node families over four profile windows, and the swing families — 118 of the
137. They are legitimate hypotheses individually, but they cannot be pooled
with the sparse families or used as a control population.

## The core universe

Restricting to the 26 classical kinds gives **28.5 levels per session** and a
median nearest-level distance of **8.40 points**, with only 15 % of random
prices within 1.5 points. That is a set where membership is a real statement
about a price, so it is the default analysis universe, and dense families are
studied separately with their own controls.

## Fair controls

:func:`make_controls` draws each placebo at matched distance and side but
**rejects draws that land within ``min_sep`` of any core level in that
session**, retrying before falling back. Without that rejection step a
"control" is simply another level.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# The sparse, classical set: prior session, current night, session extremes,
# the profile's own structural prices, and the big round handles.
CORE_KINDS: Tuple[str, ...] = (
    "pdh", "pdl", "pdc", "pdo", "pdmid", "pd_vwap",
    "pd_poc", "pd_vah", "pd_val", "pd_globex_high", "pd_globex_low",
    "on_high", "on_low", "on_mid", "on_vwap", "on_poc", "on_vah", "on_val",
    "ldn_high", "ldn_low", "asia_high", "asia_low", "pre_high", "pre_low",
    "round_100", "gap_prior_close",
)

# Kinds that saturate the price axis — analysed on their own, never pooled with
# the core and never used to define "is this price at a level".
DENSE_KINDS: Tuple[str, ...] = (
    "round_25", "round_50", "swing_high", "swing_low",
    "on_hvn", "on_lvn", "pd_hvn", "pd_lvn",
    "lb5_hvn", "lb5_lvn", "lb20_hvn", "lb20_lvn",
)


def explode_kinds(c: pd.DataFrame) -> pd.DataFrame:
    """One row per (cluster, kind). Clusters keep their identity in ``kinds``."""
    return c.assign(kind=c["kinds"].str.split(",")).explode("kind")


def select_universe(c: pd.DataFrame, universe: str = "core",
                    max_dist_atr: float = 1.5) -> pd.DataFrame:
    """Filter candidates to an analysis universe.

    ``core``   — the 26 sparse classical kinds
    ``dense``  — the saturating families, for separate study
    ``all``    — everything, for density diagnostics only

    Levels beyond ``max_dist_atr`` are dropped: their label is a foregone
    "no touch" and they would pad every statistic with free negatives.
    """
    x = explode_kinds(c)
    if universe == "core":
        x = x[x["kind"].isin(CORE_KINDS)]
    elif universe == "dense":
        x = x[x["kind"].isin(DENSE_KINDS)]
    elif universe != "all":
        raise ValueError(f"unknown universe {universe!r}")
    return x[x["dist_atr"] <= max_dist_atr].reset_index(drop=True)


def core_levels_by_session(c: pd.DataFrame) -> Dict[object, np.ndarray]:
    """Session -> sorted array of core level offsets, for control rejection."""
    x = explode_kinds(c)
    x = x[x["kind"].isin(CORE_KINDS)]
    return {s: np.sort(g["off_t"].to_numpy("float64"))
            for s, g in x.groupby("session", sort=False)}


def _min_gap(off: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Distance from each offset to the nearest level in the same session."""
    if levels.size == 0:
        return np.full(off.shape, np.inf)
    i = np.searchsorted(levels, off)
    lo = np.abs(off - levels[np.clip(i - 1, 0, levels.size - 1)])
    hi = np.abs(off - levels[np.clip(i, 0, levels.size - 1)])
    return np.minimum(lo, hi)


def make_controls(cands: pd.DataFrame, core_by_session: Dict[object, np.ndarray],
                  kind: str = "shift", shift_atr: float = 0.35,
                  min_sep_t: float = 12.0, tries: int = 12,
                  seed: int = 7, atr_col: str = "atr_t") -> pd.DataFrame:
    """Matched control zones that are *not* themselves near a core level.

    ``shift``  — same side, same session, displaced by ``shift_atr`` ATR
    ``random`` — same side, uniform over the reachable band
    ``mirror`` — the same distance reflected across the reference

    Each draw is retried until it is at least ``min_sep_t`` ticks from every
    core level of that session. ``ctrl_ok`` records whether a clean draw was
    found, so contaminated rows can be excluded rather than silently weakening
    the contrast.
    """
    rng = np.random.default_rng(seed)
    d = cands.copy()
    o0 = d["off_t"].to_numpy("float64")
    atr = d[atr_col].to_numpy("float64")
    sess = d["session"].to_numpy()
    out = np.full(len(d), np.nan)
    ok = np.zeros(len(d), dtype=bool)

    for s in pd.unique(sess):
        m = sess == s
        lv = core_by_session.get(s, np.empty(0))
        o, a = o0[m], atr[m]
        best = np.full(o.shape, np.nan)
        best_gap = np.full(o.shape, -1.0)
        for _ in range(tries):
            if kind == "shift":
                sgn = np.where(rng.random(o.size) < 0.5, -1.0, 1.0)
                cand = o + sgn * shift_atr * a
            elif kind == "random":
                cand = np.sign(np.where(o == 0, 1.0, o)) * rng.uniform(0.05, 1.25, o.size) * a
            elif kind == "mirror":
                cand = -o
            else:
                raise ValueError(f"unknown control kind {kind!r}")
            gap = _min_gap(cand, lv)
            better = gap > best_gap
            best[better] = cand[better]
            best_gap[better] = gap[better]
            if kind == "mirror" or (best_gap >= min_sep_t).all():
                break
        out[m] = best
        ok[m] = best_gap >= min_sep_t

    d["off_t"] = out
    d["ctrl_ok"] = ok
    d["control"] = kind
    d["dist_t"] = d["off_t"].abs()
    d["dist_atr"] = d["dist_t"] / d[atr_col]
    return d


def density_report(c: pd.DataFrame, ctx: pd.DataFrame, universes=("core", "dense", "all"),
                   band_atr: float = 1.0, n_draws: int = 50, seed: int = 0) -> pd.DataFrame:
    """How much of the price axis each universe covers — the vacuity check.

    Any level concept must pass this before its statistics mean anything: if a
    random price is usually "at a level", the concept is a description of the
    axis, not of the market.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in universes:
        x = select_universe(c, u, max_dist_atr=band_atr + 0.5)
        if x.empty:
            continue
        ds, per = [], []
        for s, g in x.groupby("session", sort=False):
            o = g["off_t"].to_numpy("float64") * 0.25
            a = float(g["atr_t"].iloc[0]) * 0.25
            per.append(len(g))
            if o.size == 0 or not np.isfinite(a):
                continue
            u_draw = rng.uniform(-band_atr * a, band_atr * a, n_draws)
            ds.append(np.abs(u_draw[:, None] - o[None, :]).min(axis=1))
        ds = np.concatenate(ds) if ds else np.array([np.nan])
        rows.append({"universe": u, "kinds": int(x["kind"].nunique()),
                     "levels_per_session": float(np.mean(per)),
                     "median_gap_pts": float(np.median(ds)),
                     "within_1.5pt": float((ds <= 1.5).mean()),
                     "within_3pt": float((ds <= 3).mean()),
                     "within_5pt": float((ds <= 5).mean())})
    return pd.DataFrame(rows)
