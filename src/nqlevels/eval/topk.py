"""Top-K daily selection evaluation — the Phase-3 yardstick.

The product is a *selection*: a few zones per morning, chosen at 06:25. So the
evaluation must be a selection too. Ranking rows in a pooled table answers a
different question, and Phase 2 showed how badly pooled evaluation can lie
(P2-108: any ranker correlated with distance inherits the extreme-price
tautology).

Three rules are enforced here rather than left to each experiment:

1. **Selection sees nothing post-cut.** The score column is produced upstream
   from pre-open features; this module only sorts by it. It never conditions
   on ``touched`` or any outcome when picking.
2. **The headline control is matched by construction.** For every selected
   zone, control zones are drawn from the *same morning, same side, same
   distance band*. Distance and session character then cancel identically,
   which is the only defence against the tautology that survived Phase 2.
3. **Inference is paired and session-clustered.** Model and control arms are
   resampled on the same session draw, so a wild morning widens neither side
   of the contrast alone.

Metrics come in two shapes. *Zone-level*: touch rate, P(clean | touch), the
joint P(touched AND clean), reaction magnitude, penetration, slice-through.
*Morning-level*: P(at least one selected zone reached), P(at least one
reacted), P(two or more reacted). The joint rate is the primary contrast — it
needs no conditioning on touch, so it cannot inherit selection-into-touch bias.
For P(clean | touch) the matched control keeps the touched subsets comparable,
but the joint rate is still reported first.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REQUIRED = ("session", "off_t", "dist_atr", "touched")


def _check(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    for c in REQUIRED:
        if c not in df.columns:
            raise ValueError(f"top-k frame needs column {c!r}")
    if score_col not in df.columns:
        raise ValueError(f"score column {score_col!r} not in frame")
    d = df
    if "above" not in d.columns:
        d = d.copy()
        d["above"] = (d["off_t"] > 0).astype("int8")
    return d


def add_bands(df: pd.DataFrame, n: int = 10, col: str = "dist_atr") -> pd.DataFrame:
    """Global distance-band labels used for matching. Quantile bins over the
    whole frame keep every band populated on most mornings."""
    d = df.copy()
    d["_dband"] = pd.qcut(d[col], n, duplicates="drop").cat.codes.astype("int16")
    return d


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #


def select_topk(g: pd.DataFrame, score_col: str, k: int,
                min_sep_t: float = 24.0,
                dist_window: Optional[Tuple[float, float]] = None,
                per_side_cap: Optional[int] = None) -> pd.DataFrame:
    """Greedy top-K for one morning: best score first, skipping anything within
    ``min_sep_t`` ticks of an already-selected zone.

    ``min_sep_t`` is the de-duplication the product needs anyway (two zones
    six ticks apart are one zone) and the statistics need too (correlated
    successes). ``dist_window`` restricts eligibility in ATR units — a
    selection policy knob, applied before ranking so it is a property of the
    candidate pool, not of the outcome. ``per_side_cap`` optionally limits how
    many picks may share one side, without forcing symmetry.
    """
    e = g[np.isfinite(g[score_col])]
    if dist_window is not None:
        lo, hi = dist_window
        e = e[(e["dist_atr"] >= lo) & (e["dist_atr"] <= hi)]
    if e.empty:
        return e
    e = e.sort_values(score_col, ascending=False, kind="mergesort")
    taken: List[int] = []
    offs: List[float] = []
    side_n = {0: 0, 1: 0}
    for idx, off, ab in zip(e.index, e["off_t"].to_numpy("float64"),
                            e["above"].to_numpy()):
        if any(abs(off - o) < min_sep_t for o in offs):
            continue
        if per_side_cap is not None and side_n[int(ab)] >= per_side_cap:
            continue
        taken.append(idx)
        offs.append(off)
        side_n[int(ab)] += 1
        if len(taken) >= k:
            break
    return e.loc[taken]


def matched_controls(pool: pd.DataFrame, picks: pd.DataFrame, rng,
                     n_draws: int = 8, min_sep_t: float = 24.0) -> pd.DataFrame:
    """Same-morning, same-side, same-distance-band controls for each pick.

    Controls are excluded from the neighbourhood of *every* pick (not only the
    one they match), otherwise "the control" can be the selected zone's own
    shoulder. Each pick gets up to ``n_draws`` independent draws, tagged
    ``_draw``, so control noise averages out without pretending one draw is
    the truth. Picks with an empty eligible cell yield no control rows — the
    contrast reports how many were matchable.
    """
    if "_dband" not in pool.columns or "_dband" not in picks.columns:
        raise ValueError("run add_bands() on the frame before matching")
    offs = picks["off_t"].to_numpy("float64")
    out = []
    for _, r in picks.iterrows():
        cell = pool[(pool["_dband"] == r["_dband"]) & (pool["above"] == r["above"])]
        if len(offs):
            far = np.abs(cell["off_t"].to_numpy("float64")[:, None] - offs[None, :])
            cell = cell[(far >= min_sep_t).all(axis=1)]
        if cell.empty:
            continue
        take = cell.sample(min(n_draws, len(cell)),
                           random_state=int(rng.integers(1 << 31)))
        take = take.copy()
        take["_draw"] = np.arange(len(take))
        take["_pick_off"] = r["off_t"]
        out.append(take)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=list(pool.columns) + ["_draw"])


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def _joint(d: pd.DataFrame, outcome: str) -> float:
    """P(touched AND outcome), unresolved outcomes counted as not-reacted.
    The one rate that needs no conditioning on touch."""
    if not len(d):
        return np.nan
    y = (d["touched"].to_numpy() == True) & (pd.to_numeric(d[outcome], errors="coerce").fillna(0.0).to_numpy() == 1)  # noqa: E712
    return float(y.mean())


def _cond(d: pd.DataFrame, outcome: str) -> float:
    """P(outcome | touched), unresolved dropped."""
    t = d[(d["touched"] == True)]                                   # noqa: E712
    v = pd.to_numeric(t[outcome], errors="coerce").dropna()
    return float(v.mean()) if len(v) else np.nan


def _touch(d: pd.DataFrame) -> float:
    return float(d["touched"].mean()) if len(d) else np.nan


def _day_rates(d: pd.DataFrame, outcome: str) -> Dict[str, float]:
    """Morning-level hit rates over whatever sessions appear in ``d``.
    For control frames with several draws per pick, rates are computed per
    draw and averaged, so the control is 'one alternative selection', not a
    union of eight."""
    if not len(d):
        return {"p_ge1_reached": np.nan, "p_ge1_react": np.nan, "p_ge2_react": np.nan,
                "mean_reached": np.nan}
    d = d.copy()
    d["_hit"] = ((d["touched"] == True) &                            # noqa: E712
                 (pd.to_numeric(d[outcome], errors="coerce").fillna(0.0) == 1)).astype(int)
    d["_tch"] = (d["touched"] == True).astype(int)                   # noqa: E712
    keys = ["session", "_draw"] if "_draw" in d.columns else ["session"]
    g = d.groupby(keys, observed=True)[["_hit", "_tch"]].sum()
    return {"p_ge1_reached": float((g["_tch"] >= 1).mean()),
            "p_ge1_react": float((g["_hit"] >= 1).mean()),
            "p_ge2_react": float((g["_hit"] >= 2).mean()),
            "mean_reached": float(g["_tch"].mean())}


def arm_summary(d: pd.DataFrame, outcome: str, atr_col: str = "atr_t") -> Dict[str, float]:
    """Zone- and morning-level description of one arm."""
    t = d[(d["touched"] == True)]                                    # noqa: E712
    s = {"n_zones": int(len(d)), "n_sessions": int(d["session"].nunique()) if len(d) else 0,
         "touch": _touch(d), "joint": _joint(d, outcome), "cond": _cond(d, outcome),
         "med_dist_atr": float(d["dist_atr"].median()) if len(d) else np.nan}
    s["rev30m_atr_med"] = float(t["rev30m_atr"].median()) if "rev30m_atr" in d.columns and len(t) else np.nan
    s["pen_atr_med"] = float(t["pen_atr"].median()) if "pen_atr" in d.columns and len(t) else np.nan
    s["slice60"] = float(t["slice_through_60s"].mean()) if "slice_through_60s" in d.columns and len(t) else np.nan
    s.update(_day_rates(d, outcome))
    return s


def paired_ci(model: pd.DataFrame, control: pd.DataFrame,
              stat: Callable[[pd.DataFrame], float], n_boot: int = 600,
              seed: int = 17) -> Dict[str, float]:
    """model − control with a paired session-clustered bootstrap.

    Sessions are drawn once per iteration and applied to both arms, so shared
    morning-level noise cancels instead of widening the interval.
    """
    common = np.intersect1d(model["session"].unique(), control["session"].unique())
    if common.size < 8:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "p_gt0": np.nan,
                "n_sessions": int(common.size)}
    m = model[model["session"].isin(common)].reset_index(drop=True)
    c = control[control["session"].isin(common)].reset_index(drop=True)
    gm = {s: np.flatnonzero(m["session"].to_numpy() == s) for s in common}
    gc = {s: np.flatnonzero(c["session"].to_numpy() == s) for s in common}
    point = stat(m) - stat(c)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(common, size=common.size, replace=True)
        vals[b] = (stat(m.iloc[np.concatenate([gm[s] for s in pick])])
                   - stat(c.iloc[np.concatenate([gc[s] for s in pick])]))
    vals = vals[np.isfinite(vals)]
    if vals.size < 50:
        return {"diff": float(point), "lo": np.nan, "hi": np.nan, "p_gt0": np.nan,
                "n_sessions": int(common.size)}
    return {"diff": float(point), "lo": float(np.quantile(vals, .025)),
            "hi": float(np.quantile(vals, .975)), "p_gt0": float((vals > 0).mean()),
            "n_sessions": int(common.size)}


# --------------------------------------------------------------------------- #
# the full evaluation
# --------------------------------------------------------------------------- #


def evaluate(df: pd.DataFrame, score_col: str, ks: Sequence[int] = (1, 2, 3, 4),
             outcome: str = "clean", min_sep_t: float = 24.0,
             dist_window: Optional[Tuple[float, float]] = None,
             per_side_cap: Optional[int] = None, n_draws: int = 8,
             n_boot: int = 600, n_bands: int = 10, seed: int = 17,
             by_year: bool = True) -> Dict:
    """Run the whole top-K evaluation for one scored frame.

    Returns, per K: the model arm, the matched-control arm, next-ranked /
    random / nearest reference arms, and paired CIs on the joint rate and on
    P(outcome | touch). ``df`` must contain one row per (session, candidate)
    with pre-open score and measured outcomes.
    """
    d = _check(df, score_col)
    d = add_bands(d, n=n_bands)
    rng = np.random.default_rng(seed)
    by_sess = {s: g for s, g in d.groupby("session", sort=False)}
    out: Dict = {"score_col": score_col, "outcome": outcome, "ks": list(ks),
                 "min_sep_t": min_sep_t, "dist_window": dist_window}

    for k in ks:
        picks, nxt, rand, near, ctrl = [], [], [], [], []
        for s, g in by_sess.items():
            p = select_topk(g, score_col, k, min_sep_t, dist_window, per_side_cap)
            if p.empty:
                continue
            picks.append(p)
            # next-ranked: the following K under the same policy, minus the picks
            p2 = select_topk(g.drop(index=p.index), score_col, k, min_sep_t,
                             dist_window, per_side_cap)
            nxt.append(p2)
            # random same-morning under the same eligibility
            e = g[np.isfinite(g[score_col])]
            if dist_window is not None:
                e = e[(e["dist_atr"] >= dist_window[0]) & (e["dist_atr"] <= dist_window[1])]
            if len(e):
                rand.append(e.sample(min(k, len(e)),
                                     random_state=int(rng.integers(1 << 31))))
            near.append(g.nsmallest(k, "dist_atr"))
            ctrl.append(matched_controls(g, p, rng, n_draws=n_draws,
                                         min_sep_t=min_sep_t))
        M = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
        arms = {"model": M,
                "matched": pd.concat(ctrl, ignore_index=True) if ctrl else pd.DataFrame(),
                "next": pd.concat(nxt, ignore_index=True) if nxt else pd.DataFrame(),
                "random": pd.concat(rand, ignore_index=True) if rand else pd.DataFrame(),
                "nearest": pd.concat(near, ignore_index=True) if near else pd.DataFrame()}
        r: Dict = {name: arm_summary(a, outcome) for name, a in arms.items() if len(a)}
        C = arms["matched"]
        if len(M) and len(C):
            r["match_rate"] = float(C.groupby("session")["_pick_off"].nunique().sum()
                                    / max(1, len(M)))
            r["ci_joint"] = paired_ci(M, C, lambda x: _joint(x, outcome), n_boot, seed)
            r["ci_cond"] = paired_ci(M, C, lambda x: _cond(x, outcome), n_boot, seed)
            r["ci_touch"] = paired_ci(M, C, _touch, n_boot, seed)
            if by_year:
                yr = {}
                My = M.copy()
                My["_y"] = pd.to_datetime(My["session"].astype(str)).dt.year
                Cy = C.copy()
                Cy["_y"] = pd.to_datetime(Cy["session"].astype(str)).dt.year
                for y in sorted(My["_y"].unique()):
                    my, cy = My[My["_y"] == y], Cy[Cy["_y"] == y]
                    yr[int(y)] = {"n": int(len(my)),
                                  "joint_model": _joint(my, outcome),
                                  "joint_ctrl": _joint(cy, outcome),
                                  "cond_model": _cond(my, outcome),
                                  "cond_ctrl": _cond(cy, outcome)}
                r["by_year"] = yr
        out[f"K{k}"] = r
    return out


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Calibration table for a probabilistic score: predicted vs realised."""
    y = np.asarray(y, "float64")
    p = np.asarray(p, "float64")
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    b = pd.qcut(pd.Series(p), n_bins, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "b": b}).groupby("b", observed=True)
    return pd.DataFrame({"n": g.size(), "p_mean": g["p"].mean(), "y_rate": g["y"].mean()}).reset_index(drop=True)


def format_report(res: Dict) -> str:
    """Terminal-friendly rendering of an evaluate() result."""
    L = [f"score={res['score_col']} outcome={res['outcome']} "
         f"min_sep={res['min_sep_t']}t dist_window={res['dist_window']}"]
    for k in res["ks"]:
        r = res.get(f"K{k}")
        if not r:
            continue
        L.append(f"\nK={k}")
        hdr = (f"  {'arm':>8} {'zones':>7} {'sess':>5} {'touch':>6} {'joint':>7} "
               f"{'cond':>7} {'P>=1rct':>8} {'P>=2rct':>8} {'rev30m':>7} {'dist':>6}")
        L.append(hdr)
        for name in ("model", "matched", "next", "random", "nearest"):
            a = r.get(name)
            if not a:
                continue
            L.append(f"  {name:>8} {a['n_zones']:7,} {a['n_sessions']:5,} "
                     f"{a['touch']:6.3f} {a['joint']:7.4f} {a['cond']:7.4f} "
                     f"{a['p_ge1_react']:8.4f} {a['p_ge2_react']:8.4f} "
                     f"{a['rev30m_atr_med']:7.3f} {a['med_dist_atr']:6.3f}")
        for key, label in (("ci_joint", "joint"), ("ci_cond", "cond |touch"),
                           ("ci_touch", "touch")):
            c = r.get(key)
            if c:
                L.append(f"  model-matched {label:>11}: {c['diff']:+.4f} "
                         f"[{c['lo']:+.4f}, {c['hi']:+.4f}]  p>0 {c['p_gt0']:.3f}")
        if "by_year" in r:
            for y, v in r["by_year"].items():
                L.append(f"    {y}: joint {v['joint_model']:.4f} vs {v['joint_ctrl']:.4f} | "
                         f"cond {v['cond_model']:.4f} vs {v['cond_ctrl']:.4f} (n={v['n']})")
    return "\n".join(L)
