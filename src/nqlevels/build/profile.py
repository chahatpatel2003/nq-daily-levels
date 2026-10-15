"""Volume-at-price analytics over the L1 profile layer.

All inputs are the tick-keyed frames produced by
:func:`nqlevels.build.tape.window_profile`: one row per traded price, columns
``price_t, volume, n_trades, buy_vol, sell_vol``. Because the L1 profile is
stored in 30-minute buckets, any window — prior RTH, the overnight, a ten-day
composite — is a groupby-sum away, and every function here works on the result
without caring which window it came from.

Two things worth stating because they are usually left implicit:

* This is a **traded-volume** profile, not a TPO/time profile. They disagree
  most in fast one-way moves, which is exactly where level research lives, so
  the difference is a hypothesis to test rather than a detail to gloss over.
* Untraded prices inside the range are **absent rows, not zeros**. Anything
  that scans for low-volume areas has to reindex onto the full tick grid first,
  or it will step straight over the emptiest prices in the profile — the very
  LVNs it was looking for.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def densify(prof: pd.DataFrame) -> pd.DataFrame:
    """Reindex onto every tick between min and max traded price, filling 0.

    Untraded ticks are real information — they are gaps in the auction.
    """
    if prof.empty:
        return prof
    grid = np.arange(int(prof["price_t"].min()), int(prof["price_t"].max()) + 1)
    out = (prof.set_index("price_t").reindex(grid, fill_value=0)
               .rename_axis("price_t").reset_index())
    return out


def poc(prof: pd.DataFrame) -> Optional[int]:
    """Point of control: the most-traded price.

    Ties are broken toward the volume-weighted centre of the profile, which is
    stable under small resampling changes; picking the first or lowest tick is
    not, and an unstable POC produces an unstable level.
    """
    if prof.empty or prof["volume"].max() <= 0:
        return None
    top = prof[prof["volume"] == prof["volume"].max()]["price_t"].to_numpy()
    if len(top) == 1:
        return int(top[0])
    centre = np.average(prof["price_t"], weights=prof["volume"].clip(lower=0))
    return int(top[np.argmin(np.abs(top - centre))])


def value_area(prof: pd.DataFrame, coverage: float = 0.70
               ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """``(val, poc, vah)`` for the smallest contiguous band holding
    ``coverage`` of volume, grown from the POC a tick-pair at a time.

    This is the standard market-profile construction rather than a quantile of
    the volume distribution; the two differ on bimodal days, and the
    market-profile version is the one traders' levels actually come from.
    """
    if prof.empty:
        return None, None, None
    d = densify(prof).reset_index(drop=True)
    p = poc(d)
    if p is None:
        return None, None, None
    vol = d["volume"].to_numpy().astype("int64")
    px = d["price_t"].to_numpy()
    i = int(np.flatnonzero(px == p)[0])
    lo = hi = i
    target = coverage * vol.sum()
    acc = float(vol[i])
    while acc < target and (lo > 0 or hi < len(px) - 1):
        below = vol[lo - 1] if lo > 0 else -1
        above = vol[hi + 1] if hi < len(px) - 1 else -1
        if above >= below:
            hi += 1
            acc += vol[hi]
        else:
            lo -= 1
            acc += vol[lo]
    return int(px[lo]), int(p), int(px[hi])


def profile_vwap(prof: pd.DataFrame) -> Optional[float]:
    if prof.empty or prof["volume"].sum() <= 0:
        return None
    return float(np.average(prof["price_t"], weights=prof["volume"]))


def profile_sigma(prof: pd.DataFrame) -> Optional[float]:
    """Volume-weighted standard deviation in ticks — the natural width unit
    for a profile-anchored zone and the scale for VWAP deviation bands."""
    if prof.empty or prof["volume"].sum() <= 0:
        return None
    m = profile_vwap(prof)
    v = np.average((prof["price_t"] - m) ** 2, weights=prof["volume"])
    return float(np.sqrt(v))


def smooth(prof: pd.DataFrame, window_t: int = 8) -> pd.DataFrame:
    """Centred rolling mean of volume over ``window_t`` ticks.

    Raw tick-level NQ volume is spiky enough that every other tick is a local
    maximum; node detection is meaningless without smoothing, and the window is
    a first-class research parameter, not a hidden constant.
    """
    d = densify(prof).copy()
    d["vol_s"] = (d["volume"].rolling(window_t, center=True, min_periods=1).mean())
    return d


@dataclass(frozen=True)
class Node:
    price_t: int
    kind: str            # 'hvn' | 'lvn'
    volume: float        # smoothed volume at the node
    prominence: float    # how far it stands out from its neighbourhood, 0-1
    lo_t: int            # extent of the node
    hi_t: int


def find_nodes(prof: pd.DataFrame, window_t: int = 8, neighbourhood_t: int = 40,
               min_prominence: float = 0.15) -> List[Node]:
    """High- and low-volume nodes as prominent local extrema.

    ``prominence`` is the smoothed volume's distance from its local
    neighbourhood mean, normalised by that mean, so it is comparable across
    sessions with very different total volume. A node only survives if it
    stands out by ``min_prominence`` — this keeps the candidate list to the few
    structural features of the profile instead of every ripple, which is the
    difference between a research object and a chart full of lines.
    """
    if prof.empty:
        return []
    d = smooth(prof, window_t)
    v = d["vol_s"].to_numpy()
    px = d["price_t"].to_numpy()
    if len(v) < 2 * neighbourhood_t + 3:
        return []
    local = pd.Series(v).rolling(neighbourhood_t, center=True, min_periods=1).mean().to_numpy()
    out: List[Node] = []
    half = neighbourhood_t // 2
    for i in range(half, len(v) - half):
        win = v[i - half:i + half + 1]
        base = local[i]
        if base <= 0:
            continue
        if v[i] == win.max() and v[i] > base:
            prom = (v[i] - base) / base
            if prom >= min_prominence:
                out.append(Node(int(px[i]), "hvn", float(v[i]), float(prom),
                                int(px[i - half]), int(px[i + half])))
        elif v[i] == win.min() and v[i] < base:
            prom = (base - v[i]) / base
            if prom >= min_prominence:
                out.append(Node(int(px[i]), "lvn", float(v[i]), float(prom),
                                int(px[i - half]), int(px[i + half])))
    return _dedupe_nodes(out)


def _dedupe_nodes(nodes: List[Node], min_sep_t: int = 8) -> List[Node]:
    """Keep the most prominent node in each cluster of the same kind."""
    out: List[Node] = []
    for kind in ("hvn", "lvn"):
        group = sorted([n for n in nodes if n.kind == kind], key=lambda n: -n.prominence)
        kept: List[Node] = []
        for n in group:
            if all(abs(n.price_t - k.price_t) >= min_sep_t for k in kept):
                kept.append(n)
        out.extend(kept)
    return sorted(out, key=lambda n: n.price_t)


def composite(profiles: Sequence[pd.DataFrame],
              weights: Optional[Sequence[float]] = None) -> pd.DataFrame:
    """Merge several windows' profiles, optionally decaying older ones.

    Multi-day composites are the standard way to find levels that survived more
    than one session; the weights argument makes "recency-weighted composite" a
    parameter rather than a separate code path.
    """
    frames = [p for p in profiles if p is not None and not p.empty]
    if not frames:
        return pd.DataFrame(columns=["price_t", "volume", "n_trades", "buy_vol", "sell_vol"])
    if weights is None:
        weights = [1.0] * len(frames)
    scaled = []
    for f, w in zip(frames, weights):
        g = f.copy()
        for c in ("volume", "n_trades", "buy_vol", "sell_vol"):
            if c in g:
                g[c] = g[c] * w
        scaled.append(g)
    return (pd.concat(scaled, ignore_index=True)
              .groupby("price_t", as_index=False).sum()
              .sort_values("price_t").reset_index(drop=True))
