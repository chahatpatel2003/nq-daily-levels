"""Level-quality metrics, placebo controls, and honest inference.

Three problems this module exists to solve, all of which would otherwise
manufacture a convincing but fake result:

**Width inflation.** A wider zone is touched more often and is harder to
"break", so any statistic that rewards width can be maximised by drawing
enormous zones. Two defences, used together: touch is scored against the
baseline evaluated at the zone's **near edge**, so widening moves the edge
closer and raises the null in step; and every comparison is against a
**placebo of identical width and identical distance**, so width cancels.

**Correlated candidates.** Four levels within ten points of each other that all
"worked" on one morning are one observation of one move, not four. Every
interval here is a **session-level block bootstrap**: sessions are resampled
whole, so within-session correlation is preserved rather than assumed away.

**Distance confounding.** Named levels cluster near the market, and near prices
are touched more. Every destination statistic is therefore an *attraction
residual* — observed touch minus the distance-and-side-matched baseline — never
a raw rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TICK = 0.25


# --------------------------------------------------------------------------- #
# placebo controls
# --------------------------------------------------------------------------- #


def make_placebos(cands: pd.DataFrame, kinds: Sequence[str] = ("shift", "mirror", "random"),
                  shift_atr: float = 0.35, seed: int = 7,
                  atr_col: str = "atr_t", off_col: str = "off_t") -> pd.DataFrame:
    """Matched null zones for each candidate.

    ``shift``  — the same distance moved sideways by a fixed fraction of ATR.
                 The tightest control: same session, same side, near-identical
                 distance, but deliberately *not* where the level is.
    ``mirror`` — the same distance on the opposite side of the reference.
                 Controls for the session's directional character.
    ``random`` — a uniform offset drawn within the session's plausible range,
                 same side. The loosest control, closest to "an arbitrary price".

    A level family only means something if it beats these at the same width.
    """
    rng = np.random.default_rng(seed)
    out = []
    for kind in kinds:
        p = cands.copy()
        p["placebo"] = kind
        o = p[off_col].to_numpy("float64")
        atr = p[atr_col].to_numpy("float64")
        if kind == "shift":
            sgn = np.where(rng.random(len(p)) < 0.5, -1.0, 1.0)
            p[off_col] = o + sgn * shift_atr * atr
        elif kind == "mirror":
            p[off_col] = -o
        elif kind == "random":
            u = rng.uniform(0.05, 1.25, size=len(p))
            p[off_col] = np.sign(np.where(o == 0, 1, o)) * u * atr
        else:
            raise ValueError(f"unknown placebo kind {kind!r}")
        out.append(p)
    r = pd.concat(out, ignore_index=True)
    r["dist_t"] = r[off_col].abs()
    r["dist_atr"] = r["dist_t"] / r[atr_col]
    return r


# --------------------------------------------------------------------------- #
# zone geometry from a width model
# --------------------------------------------------------------------------- #


def apply_width(cands: pd.DataFrame, half_width_t: np.ndarray) -> pd.DataFrame:
    """Turn level offsets into zone edges, and record the near-edge distance.

    ``near_dist_t`` is the quantity the touch baseline must be evaluated at:
    a zone is reached when price gets to its *near* edge, not its centre.
    Getting this wrong is precisely how width inflation sneaks into a result.
    """
    d = cands.copy()
    w = np.asarray(half_width_t, dtype="float64")
    d["half_width_t"] = w
    d["zlo_t"] = d["off_t"] - w
    d["zhi_t"] = d["off_t"] + w
    above = d["off_t"] > 0
    d["side"] = np.where(above, "above", "below")
    d["near_dist_t"] = np.where(above, d["zlo_t"], -d["zhi_t"]).clip(min=0.0)
    d["near_dist_atr"] = d["near_dist_t"] / d["atr_t"]
    return d


# --------------------------------------------------------------------------- #
# attraction
# --------------------------------------------------------------------------- #


def measure_candidates(cands: pd.DataFrame, paths, half_width_t, baseline,
                       ctx: pd.DataFrame, point_grid=(5, 10, 15, 20, 30),
                       clean_specs=((10, 5), (15, 10), (20, 10), (30, 15))) -> pd.DataFrame:
    """Apply a width, measure every zone, and attach the attraction residual.

    The one entry point every experiment uses, so width handling, edge-anchored
    reaction measurement and baseline matching cannot drift apart between them.
    """
    from .labeling import clean_rejection, measure_zone

    z = apply_width(cands, half_width_t).reset_index(drop=True)
    rows = []
    for s, g in z.groupby("session", sort=False):
        p = paths.get(s)
        if p is None:
            continue
        zlo, zhi = g["zlo_t"].to_numpy("float64"), g["zhi_t"].to_numpy("float64")
        idx = g.index.to_numpy()
        for j in range(len(g)):
            r = measure_zone(p, zlo[j], zhi[j], point_grid=point_grid)
            r["_r"] = idx[j]
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    m = pd.DataFrame(rows).set_index("_r")
    out = m.join(z[[c for c in z.columns if c not in m.columns]]).reset_index(drop=True)
    # a clean spec is only computable if both its thresholds are in the grid
    grid = set(point_grid)
    specs = [(k, adv) for k, adv in clean_specs if k in grid and adv in grid]
    if not specs:
        raise ValueError(f"no clean spec in {clean_specs} is supported by "
                         f"point_grid {point_grid}")
    for k, adv in specs:
        out[f"clean_{k}_{adv}"] = clean_rejection(out, k, adv)
    out["clean"] = out[f"clean_{specs[0][0]}_{specs[0][1]}"]
    out["rev30m_atr"] = out["rev_1800s"] / out["atr_t"]
    out["pen_atr"] = out["max_penetration_t"] / out["atr_t"]
    return add_attraction(out, baseline, ctx)


def add_attraction(labelled: pd.DataFrame, baseline, ctx: pd.DataFrame,
                   horizon: int = 150) -> pd.DataFrame:
    """Attach the distance-matched touch baseline and the attraction residual.

    The baseline is evaluated per row at that row's own side and near-edge
    distance, using the session's own fitted scale. This is what makes the E1
    down-side miscalibration harmless: it hits a level and its placebo
    identically, so it cancels in every comparison drawn from this column.
    """
    d = labelled.merge(ctx[["session"]].assign(_i=np.arange(len(ctx))),
                       on="session", how="left")
    s_all = baseline.scale(ctx)
    s_flat = s_all if np.ndim(s_all) == 1 else s_all
    idx = d["_i"].to_numpy()
    ok = np.isfinite(idx)
    p = np.full(len(d), np.nan)
    for side in ("above", "below"):
        m = (d["side"] == side).to_numpy() & ok
        if not m.any():
            continue
        rows = idx[m].astype(int)
        s_rows = (s_flat[rows] if np.ndim(s_flat) == 1 else s_flat[rows])
        p[m] = baseline.p_touch(d.loc[m, "near_dist_t"].to_numpy("float64"),
                                s_rows, side, horizon)
    d["p_base"] = p
    d["attraction"] = d["touched"].astype(float) - d["p_base"]
    return d.drop(columns=["_i"])


# --------------------------------------------------------------------------- #
# session-clustered inference
# --------------------------------------------------------------------------- #


def block_bootstrap(d: pd.DataFrame, stat: Callable[[pd.DataFrame], float],
                    n_boot: int = 2000, seed: int = 11,
                    session_col: str = "session") -> Tuple[float, float, float, float]:
    """(point estimate, lo95, hi95, bootstrap sd), resampling whole sessions.

    Resampling rows would treat four overlapping zones on one morning as four
    independent successes and shrink the interval by roughly the square root of
    the average cluster size — which is exactly how a correlated-candidate
    result comes to look significant.
    """
    point = stat(d)
    sessions = d[session_col].to_numpy()
    uniq = pd.unique(sessions)
    if uniq.size < 8:
        return point, np.nan, np.nan, np.nan
    groups = {s: np.flatnonzero(sessions == s) for s in uniq}
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        idx = np.concatenate([groups[s] for s in pick])
        vals[b] = stat(d.iloc[idx])
    vals = vals[np.isfinite(vals)]
    if vals.size < 50:
        return point, np.nan, np.nan, np.nan
    return point, float(np.quantile(vals, .025)), float(np.quantile(vals, .975)), float(vals.std())


# Statistics the fast stratified path supports. Each reduces to "mean of
# `value` over rows where `valid`", which is what makes a bincount
# implementation possible.
STRAT_STATS: Dict[str, Tuple[str, Optional[str]]] = {
    "clean_rate": ("clean", "touched"),        # mean of clean among touched
    "attraction": ("attraction", None),        # mean of attraction over all rows
    "touch_rate": ("touched", None),
    "slice_rate": ("slice_through_60s", "touched"),
    "accept_rate": ("accepted", "touched"),
    "rev30m_atr": ("rev30m_atr", "touched"),
}


def stratified_contrast(d: pd.DataFrame, flag: str, stat: str = "clean_rate",
                        strata: Sequence[str] = ("side", "_dband"),
                        n_boot: int = 400, seed: int = 13, min_cell: int = 40,
                        session_col: str = "session") -> Dict[str, float]:
    """Difference between flagged and unflagged rows, pooled across strata.

    The only sound way to ask "does being at a level matter?" on a price axis:
    at a given distance and side there is exactly one price, so a control must
    differ in distance, and distance drives everything. Comparing *within*
    distance-and-side strata and pooling with cell weights removes that
    confound without needing an impossible matched control.

    Cells are weighted by their flagged count, so the pooled number answers
    "how much better is a level than a non-level at the distances levels
    actually occur at" rather than at distances they never occupy.

    Implemented with ``bincount`` over precomputed integer cell codes rather
    than a pandas ``groupby`` per bootstrap draw: with 600 k rows and a few
    hundred draws the groupby version takes minutes per contrast, which makes a
    width sweep impossible to run at all.
    """
    if stat not in STRAT_STATS:
        raise ValueError(f"unknown stat {stat!r}; known: {sorted(STRAT_STATS)}")
    vcol, mcol = STRAT_STATS[stat]
    if vcol not in d.columns:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "p_gt0": np.nan}

    codes = pd.MultiIndex.from_frame(d[list(strata)].astype(str)).codes
    cell = np.zeros(len(d), dtype="int64")
    for c in codes:
        cell = cell * (int(np.asarray(c).max()) + 2) + (np.asarray(c) + 1)
    cell = pd.factorize(cell)[0]
    n_cell = int(cell.max()) + 1

    fl = (d[flag].to_numpy() == 1).astype("int64")
    key = cell * 2 + fl
    n_key = n_cell * 2

    val = pd.to_numeric(d[vcol], errors="coerce").to_numpy("float64")
    valid = np.isfinite(val)
    if mcol is not None:
        valid &= (d[mcol].to_numpy() == True)                       # noqa: E712
    val = np.where(valid, val, 0.0)
    vmask = valid.astype("float64")

    def pooled(idx: Optional[np.ndarray]) -> float:
        k = key if idx is None else key[idx]
        v = val if idx is None else val[idx]
        m = vmask if idx is None else vmask[idx]
        cnt = np.bincount(k, weights=m, minlength=n_key)
        sm = np.bincount(k, weights=v, minlength=n_key)
        c0, c1 = cnt[0::2], cnt[1::2]
        ok = (c0 >= min_cell) & (c1 >= min_cell)
        if not ok.any():
            return np.nan
        m0 = np.divide(sm[0::2], c0, out=np.zeros_like(c0), where=c0 > 0)
        m1 = np.divide(sm[1::2], c1, out=np.zeros_like(c1), where=c1 > 0)
        w = c1[ok]
        return float(np.sum(w * (m1[ok] - m0[ok])) / w.sum()) if w.sum() > 0 else np.nan

    point = pooled(None)
    sess = d[session_col].to_numpy()
    uniq, inv = np.unique(sess, return_inverse=True)
    if uniq.size < 8:
        return {"diff": point, "lo": np.nan, "hi": np.nan, "p_gt0": np.nan}
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(uniq.size))
    ends = np.searchsorted(inv[order], np.arange(uniq.size), side="right")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        idx = np.concatenate([order[starts[p]:ends[p]] for p in pick])
        vals[b] = pooled(idx)
    vals = vals[np.isfinite(vals)]
    if vals.size < 30:
        return {"diff": point, "lo": np.nan, "hi": np.nan, "p_gt0": np.nan}
    return {"diff": float(point), "lo": float(np.quantile(vals, .025)),
            "hi": float(np.quantile(vals, .975)), "p_gt0": float((vals > 0).mean()),
            "n_flag": int(fl.sum()), "n_ctrl": int((1 - fl).sum())}


def add_distance_bands(d: pd.DataFrame, n: int = 8, col: str = "dist_atr") -> pd.DataFrame:
    """Distance deciles used as strata. Quantile bins keep cells populated."""
    out = d.copy()
    out["_dband"] = pd.qcut(out[col], n, duplicates="drop")
    return out


def paired_contrast(real: pd.DataFrame, placebo: pd.DataFrame,
                    stat: Callable[[pd.DataFrame], float], n_boot: int = 2000,
                    seed: int = 11, session_col: str = "session") -> Dict[str, float]:
    """Difference between a family and its placebo, bootstrapped on shared sessions.

    Sessions are resampled once and applied to *both* arms, so the contrast is
    paired and the common session-level noise — a wild morning where everything
    gets touched — cancels instead of widening the interval.
    """
    common = np.intersect1d(real[session_col].unique(), placebo[session_col].unique())
    if common.size < 8:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "p_gt0": np.nan, "n_sessions": int(common.size)}
    r = real[real[session_col].isin(common)]
    q = placebo[placebo[session_col].isin(common)]
    gr = {s: np.flatnonzero(r[session_col].to_numpy() == s) for s in common}
    gq = {s: np.flatnonzero(q[session_col].to_numpy() == s) for s in common}
    point = stat(r) - stat(q)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(common, size=common.size, replace=True)
        ir = np.concatenate([gr[s] for s in pick])
        iq = np.concatenate([gq[s] for s in pick])
        vals[b] = stat(r.iloc[ir]) - stat(q.iloc[iq])
    vals = vals[np.isfinite(vals)]
    return {"diff": float(point), "lo": float(np.quantile(vals, .025)),
            "hi": float(np.quantile(vals, .975)),
            "p_gt0": float((vals > 0).mean()), "n_sessions": int(common.size)}


# --------------------------------------------------------------------------- #
# statistics used as `stat` above
# --------------------------------------------------------------------------- #


def mean_attraction(d: pd.DataFrame) -> float:
    v = d["attraction"].to_numpy("float64")
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else np.nan


def clean_rate(d: pd.DataFrame, col: str = "clean") -> float:
    v = d.loc[d["touched"] == True, col]        # noqa: E712
    v = v.dropna()
    return float(v.mean()) if len(v) else np.nan


def touch_rate(d: pd.DataFrame) -> float:
    return float(d["touched"].mean()) if len(d) else np.nan


def median_reaction(d: pd.DataFrame, col: str = "rev_1800s") -> float:
    v = d.loc[d["touched"] == True, col].dropna()   # noqa: E712
    return float(v.median()) if len(v) else np.nan


def reaction_ratio(d: pd.DataFrame) -> float:
    """Median rejection divided by median penetration, on touched zones.

    A pure shape statistic: how much a zone gives back relative to how far it
    is pushed through. Insensitive to the overall volatility of the day.
    """
    t = d[d["touched"] == True]                  # noqa: E712
    if not len(t):
        return np.nan
    pen = t["max_penetration_t"].median()
    rev = t["rev_1800s"].median()
    return float(rev / pen) if pen and np.isfinite(pen) and pen > 0 else np.nan


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def by_group(d: pd.DataFrame, group: str, stats: Dict[str, Callable],
             min_n: int = 60, min_sessions: int = 30) -> pd.DataFrame:
    """Per-group statistics with sample sizes, sorted by the first statistic."""
    rows = []
    for key, g in d.groupby(group, observed=True):
        if len(g) < min_n or g["session"].nunique() < min_sessions:
            continue
        r = {group: key, "n": len(g), "sessions": int(g["session"].nunique()),
             "n_touched": int((g["touched"] == True).sum())}          # noqa: E712
        for name, fn in stats.items():
            r[name] = fn(g)
        rows.append(r)
    out = pd.DataFrame(rows)
    if len(out) and stats:
        out = out.sort_values(list(stats)[0], ascending=False)
    return out.reset_index(drop=True)
