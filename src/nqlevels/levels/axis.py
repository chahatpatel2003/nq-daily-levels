"""L3b: algorithmic price-axis discovery — score every price, not a name list.

Instead of asking "is PDH useful?", this asks: *for every price region within
reach of the market at 06:25, what does the data say about it?* Named levels
then become one feature among many, and can lose.

A structural fact drives the whole design, and getting it wrong would produce
a very impressive and completely fake result:

    **Touch probability of a price above the reference is monotone in
    distance.** If price reaches 21,700 it has necessarily passed 21,690. So
    P(touch) for a grid price is exactly the survival function of the session's
    maximum excursion, and *any* level type will show a high touch rate simply
    by sitting close to the open.

So "attraction" is not P(touch). It is **P(touch) in excess of the
distance-matched baseline** — the residual after conditioning on distance and
regime. :func:`baseline_touch_curve` fits that baseline empirically from
training sessions only, and :func:`attraction_residual` is what the models are
actually asked to predict. A level type that cannot beat its own distance curve
has demonstrated nothing, however high its raw hit rate looks.

Reaction quality has no such structure: what price does *at* 21,700 is genuinely
local, does not follow from what it did at 21,690, and is where most of the
usable signal is likely to be. The two problems are modelled separately for
this reason, not merely as a matter of taste.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..build.profile import densify, smooth


@dataclass(frozen=True)
class AxisSpec:
    """Geometry of the candidate price grid."""
    span_points: float = 250.0      # half-width of the grid around the reference
    step_ticks: int = 2             # 0.5 index points
    tick_size: float = 0.25

    def grid(self, ref_t: float) -> np.ndarray:
        span_t = int(self.span_points / self.tick_size)
        lo = int(round((ref_t - span_t) / self.step_ticks)) * self.step_ticks
        hi = int(round((ref_t + span_t) / self.step_ticks)) * self.step_ticks
        return np.arange(lo, hi + self.step_ticks, self.step_ticks, dtype="int64")


# --------------------------------------------------------------------------- #
# per-price features
# --------------------------------------------------------------------------- #


def volume_at_price_features(grid: np.ndarray, prof: pd.DataFrame,
                             prefix: str, smooth_t: int = 8) -> pd.DataFrame:
    """How much trade has happened at, and around, each grid price.

    Emits the level, the neighbourhood-relative level (a price can be busy in
    absolute terms and still be a hole in its own neighbourhood — that is what
    an LVN is), and the local gradient, which is what "volume cliff" means
    mechanically.
    """
    out = pd.DataFrame({"price_t": grid})
    if prof is None or prof.empty:
        for c in ("vol", "vol_rel", "vol_grad", "vol_share"):
            out[f"{prefix}_{c}"] = np.nan
        return out
    d = smooth(prof, smooth_t).set_index("price_t")
    v = d["vol_s"].reindex(grid, fill_value=0.0).to_numpy()
    total = float(d["volume"].sum()) or 1.0
    local = pd.Series(v).rolling(80, center=True, min_periods=1).mean().to_numpy()
    rel = np.divide(v, local, out=np.full_like(v, np.nan), where=local > 0)
    return pd.concat([out, pd.DataFrame({
        f"{prefix}_vol": v,
        f"{prefix}_vol_rel": rel,
        f"{prefix}_vol_grad": np.gradient(v) if len(v) > 1 else np.zeros_like(v),
        f"{prefix}_vol_share": v / total,
    })], axis=1)


def interaction_features(grid: np.ndarray, bars: pd.DataFrame,
                         prefix: str) -> pd.DataFrame:
    """How price has *behaved* at each grid price over a historical window.

    ``touches`` counts distinct visits (re-entries after leaving), not seconds,
    so a price that was sat on for an hour scores 1 visit and a large
    ``secs_at``; the pair separates acceptance from repeated rejection.
    ``rejections`` counts visits that were followed by a move away of at least
    a few ticks without the price being revisited soon after.
    """
    out = pd.DataFrame({"price_t": grid})
    if bars is None or bars.empty:
        for c in ("touches", "secs_at", "last_touch_frac", "rej_up", "rej_dn"):
            out[f"{prefix}_{c}"] = np.nan
        return out
    hi = bars["high_t"].to_numpy()
    lo = bars["low_t"].to_numpy()
    n = len(hi)
    touches = np.zeros(len(grid), dtype="int32")
    secs = np.zeros(len(grid), dtype="int32")
    last = np.full(len(grid), np.nan)
    for i, p in enumerate(grid):
        inside = (lo <= p) & (hi >= p)
        if not inside.any():
            continue
        secs[i] = int(inside.sum())
        # a visit starts on a False->True edge
        touches[i] = int(np.count_nonzero(inside & ~np.concatenate(([False], inside[:-1]))))
        last[i] = float(np.flatnonzero(inside)[-1]) / max(1, n - 1)
    out[f"{prefix}_touches"] = touches
    out[f"{prefix}_secs_at"] = secs
    out[f"{prefix}_last_touch_frac"] = last          # 1.0 = touched most recently
    out[f"{prefix}_visit_len"] = np.where(touches > 0, secs / np.maximum(touches, 1), np.nan)
    return out


def geometry_features(grid: np.ndarray, ref_t: float, atr_t: float,
                      tick_size: float = 0.25) -> pd.DataFrame:
    """Distance, direction and round-number geometry.

    ``dist_atr`` is the feature that matters: raw tick distance is not
    comparable between a 150-point day and a 500-point day, and every touch
    statistic in this project is conditioned on it.
    """
    out = pd.DataFrame({"price_t": grid})
    d = grid - ref_t
    out["dist_t"] = np.abs(d)
    out["dist_signed_t"] = d
    out["above"] = (d > 0).astype("int8")
    out["dist_atr"] = np.abs(d) / max(atr_t, 1e-9)
    px = grid * tick_size
    for step in (25, 50, 100):
        r = np.abs(((px + step / 2) % step) - step / 2)
        out[f"round{step}_dist_pts"] = r
        out[f"is_round{step}"] = (r < 1e-9).astype("int8")
    return out


def named_level_features(grid: np.ndarray, levels: pd.DataFrame,
                         near_t: int = 8) -> pd.DataFrame:
    """Proximity to the classical families, one column per family.

    This is how the named generator and the axis model are compared on equal
    terms: if "PDH" carries information the axis features miss, this column
    will earn weight; if it does not, it will not, and that is the answer.
    """
    out = pd.DataFrame({"price_t": grid})
    if levels is None or levels.empty:
        out["named_n_within"] = 0
        out["named_min_dist_t"] = np.nan
        return out
    lp = levels["price_t"].to_numpy(dtype="float64")
    dist = np.abs(grid[:, None] - lp[None, :])
    out["named_min_dist_t"] = dist.min(axis=1)
    out["named_n_within"] = (dist <= near_t).sum(axis=1)
    if "kinds" in levels.columns:
        fam = levels["kinds"].astype(str).str.split(",")
        allk = sorted({k for ks in fam for k in ks if k})
        for k in allk:
            cols = np.array([k in ks for ks in fam])
            if not cols.any():
                continue
            out[f"d_{k}_t"] = dist[:, cols].min(axis=1)
    return out


def build_axis_frame(ref_t: float, atr_t: float, spec: AxisSpec,
                     profiles: Dict[str, pd.DataFrame],
                     bar_windows: Dict[str, pd.DataFrame],
                     named: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Assemble one session's price-axis feature table.

    Every input must already be windowed to end before the decision cut; this
    function has no clock and will happily encode the future if handed it. The
    caller that owns the cut is :mod:`nqlevels.levels.candidates`.
    """
    grid = spec.grid(ref_t)
    parts = [geometry_features(grid, ref_t, atr_t, spec.tick_size)]
    for name, prof in profiles.items():
        parts.append(volume_at_price_features(grid, prof, name).drop(columns=["price_t"]))
    for name, bars in bar_windows.items():
        parts.append(interaction_features(grid, bars, name).drop(columns=["price_t"]))
    if named is not None:
        parts.append(named_level_features(grid, named).drop(columns=["price_t"]))
    return pd.concat(parts, axis=1)


# --------------------------------------------------------------------------- #
# distance baseline — the thing every attraction claim must beat
# --------------------------------------------------------------------------- #


class InsufficientBaseline(ValueError):
    """Not enough data to fit a distance baseline worth comparing against."""


def baseline_touch_curve(df: pd.DataFrame, dist_col: str = "dist_atr",
                         touched_col: str = "touched", side_col: str = "above",
                         n_bins: int = 40, min_per_bin: int = 200) -> pd.DataFrame:
    """Empirical P(touch | distance, side) from training sessions only.

    Quantile bins keep resolution where the mass is. The returned table is the
    null model: it says how often *any* price at that distance gets touched.

    Raises rather than returning an empty frame when no bin reaches
    ``min_per_bin``. A silently empty curve turns every downstream attraction
    number into ``NaN``, which is easy to mistake for "no effect" instead of
    "nothing was measured".
    """
    d = df[[dist_col, touched_col, side_col]].dropna()
    rows = []
    for side, g in d.groupby(side_col):
        try:
            g = g.assign(_b=pd.qcut(g[dist_col], n_bins, duplicates="drop"))
        except ValueError:
            continue
        for b, gg in g.groupby("_b", observed=True):
            if len(gg) < min_per_bin:
                continue
            rows.append({"side": int(side), "lo": float(b.left), "hi": float(b.right),
                         "p": float(gg[touched_col].mean()), "n": int(len(gg))})
    out = pd.DataFrame(rows)
    if out.empty or out["side"].nunique() < 2:
        raise InsufficientBaseline(
            f"could not fit a distance baseline from {len(d):,} rows "
            f"(n_bins={n_bins}, min_per_bin={min_per_bin}); got "
            f"{len(out)} bin(s) across {out['side'].nunique() if len(out) else 0} "
            "side(s). Use more training sessions, fewer bins, or a smaller "
            "min_per_bin — but do not compare against a baseline this thin.")
    return out


def apply_baseline(df: pd.DataFrame, curve: pd.DataFrame,
                   dist_col: str = "dist_atr", side_col: str = "above") -> np.ndarray:
    """Baseline P(touch) for each row, interpolated within side."""
    if curve.empty:
        raise InsufficientBaseline("empty baseline curve; nothing to compare against")
    out = np.full(len(df), np.nan)
    for side, c in curve.groupby("side"):
        c = c.sort_values("lo")
        mid = ((c["lo"] + c["hi"]) / 2).to_numpy()
        m = (df[side_col].to_numpy() == side)
        if m.any():
            out[m] = np.interp(df.loc[m, dist_col].to_numpy(), mid, c["p"].to_numpy(),
                               left=c["p"].iloc[0], right=c["p"].iloc[-1])
    return out


def attraction_residual(df: pd.DataFrame, curve: pd.DataFrame,
                        touched_col: str = "touched", **kw) -> pd.Series:
    """Observed touch minus its distance-matched expectation.

    Positive means the market went out of its way to reach this price. This —
    not the raw touch rate — is the destination signal the project is after.
    """
    return pd.Series(df[touched_col].to_numpy() - apply_baseline(df, curve, **kw),
                     index=df.index, name="attraction")
