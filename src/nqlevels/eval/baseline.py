"""E1: the distance/excursion baseline — the null model everything is scored against.

## Why this shape

Touch probability is not a per-price question. Reaching a price above the
reference means passing every price below it, so for any distance ``d``

    P(touch d above by minute h)  ==  P(U_h >= d)

where ``U_h`` is the session's running maximum up-excursion from the 06:25
reference. The baseline is therefore a **conditional distribution model of
session excursions**, which is the same object as a per-price classifier but
exact, monotone by construction, and about three orders of magnitude cheaper to
fit and to evaluate.

## The model

A scale-and-shape decomposition:

1. **Scale.** ``R = U + D`` is the target window's total range around the
   reference and is always positive, which makes it the natural thing to
   regress. ``log R`` is fitted on pre-open context by ridge regression, giving
   a per-session scale ``s``.
2. **Shape.** ``z = excursion / s`` is then fitted as an empirical survival
   function, separately per side and per horizon. Up and down are kept apart
   because a gap makes them genuinely asymmetric.

Prediction is then ``P(touch d) = S_side,h(d / s)``.

Excursions may be **negative** — if price gaps away at 06:30 and never returns,
it never trades on one side of the reference at all. That is real and the
empirical survival function handles it without special-casing; taking logs of
the excursions themselves would not.

## What it is for

Two things, both essential:

* Every destination claim in the project is scored against it. A level with a
  70 % touch rate at a distance where the baseline says 71 % has demonstrated
  nothing.
* It is the honest answer to "how likely is price to travel this far today",
  which is itself part of the eventual morning display.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HORIZONS: Tuple[int, ...] = (15, 30, 60, 90, 150)
SIDES: Tuple[str, ...] = ("above", "below")
_EX_COL = {"above": "up", "below": "dn"}

# --------------------------------------------------------------------------- #
# The frozen E1 specification (selected on train, reported on validation).
#
# Four features. The eleven-feature set scored identically (validation Brier
# skill 0.3931 vs 0.3933) with worse calibration, and the answer is flat in
# ridge alpha over 0.3-10 and in embargo over 1-30 days — a plateau, not a peak.
# Realised volatility beats ATR as a scale predictor: overnight RV correlates
# 0.62 with log range versus 0.47 for ATR(14).
#
# Symmetric, deliberately. A directional share model was fitted and rejected:
# its coefficients are all within +-0.024 of an intercept of 0.497, i.e. pre-open
# context barely predicts *which side* of the reference the range falls on, and
# it improved validation skill by only 0.002 while worsening ECE.
# --------------------------------------------------------------------------- #
E1_FEATURES: Tuple[str, ...] = ("log_rv_on", "log_prior_rv", "log_on_range", "log_atr14")
E1_ALPHA: float = 1.0
E1_REFIT: str = "monthly expanding window, 7-day embargo"

# Known, unfixable-from-context limitation: in 2024-25 the *down* tail fattened
# relative to all prior history (normalised p90 1.04-1.17 in 2022-23 vs
# 1.45-1.49 in 2024-25), so far downside touches are under-predicted by roughly
# 3 pp. It is a genuine out-of-sample regime shift, not a modelling error.
# MITIGATION, and it is mandatory: every level comparison must be made against
# the baseline evaluated at the SAME SIDE and a matched distance band, so the
# miscalibration hits the level and its control equally and cancels.


# --------------------------------------------------------------------------- #
# empirical survival
# --------------------------------------------------------------------------- #


@dataclass
class Survival:
    """Monotone empirical survival function of a normalised excursion."""
    z: np.ndarray                 # ascending grid
    s: np.ndarray                 # P(Z >= z), non-increasing
    n: int

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype="float64")
        out = np.interp(z, self.z, self.s, left=1.0, right=self.s[-1])
        return np.clip(out, 0.0, 1.0)


def fit_survival(z: np.ndarray, grid: Optional[np.ndarray] = None,
                 z_max: float = 4.0, n_grid: int = 400) -> Survival:
    """P(Z >= z) on a fixed grid, from the empirical distribution.

    The tail is floored at ``1/(2n)`` rather than zero: an event unobserved in
    871 sessions is rare, not impossible, and a hard zero would make any level
    beyond the observed maximum look infinitely attractive.
    """
    z = np.asarray(z, dtype="float64")
    z = z[np.isfinite(z)]
    n = z.size
    if n == 0:
        raise ValueError("no finite excursions to fit a survival function")
    g = np.linspace(-0.5, z_max, n_grid) if grid is None else np.asarray(grid)
    zs = np.sort(z)
    # P(Z >= g) = 1 - F(g^-) = (n - searchsorted(zs, g, 'left')) / n
    s = (n - np.searchsorted(zs, g, side="left")) / n
    s = np.maximum(s, 1.0 / (2 * n))
    s = np.minimum.accumulate(s)
    return Survival(z=g, s=s, n=n)


# --------------------------------------------------------------------------- #
# scale model
# --------------------------------------------------------------------------- #


DEFAULT_SCALE_FEATURES: Tuple[str, ...] = (
    "log_atr14", "log_atr5", "log_on_range", "log_pre_range",
    "log_rv_on", "log_prior_rv", "log_last60_range",
    "abs_gap_atr", "on_vol_rel_c", "loc_in_on_c", "macro_volx_c",
)


def build_scale_features(ctx: pd.DataFrame) -> pd.DataFrame:
    """Context -> the regressor matrix for ``log R``.

    Everything volatility-like enters in logs, because the range is
    multiplicative in volatility; everything else enters winsorised and
    centred. All inputs are pre-cut columns.
    """
    def lg(x, floor=1.0):
        v = pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64")
        return np.log(np.clip(v, floor, None))

    def wz(x, lo=0.01, hi=0.99):
        v = pd.to_numeric(x, errors="coerce").to_numpy(dtype="float64")
        a, b = np.nanquantile(v, lo), np.nanquantile(v, hi)
        return np.clip(v, a, b)

    f = pd.DataFrame(index=ctx.index)
    f["log_atr14"] = lg(ctx["atr14_t"])
    f["log_atr5"] = lg(ctx["atr5_t"])
    f["log_on_range"] = lg(ctx["globex_range_pre"])
    f["log_pre_range"] = lg(ctx["premarket_range_pre"])
    f["log_rv_on"] = lg(ctx["globex_rv_pre"], 0.1)
    f["log_prior_rv"] = lg(ctx["prior_rth_rv_prior"], 0.1)
    f["log_last60_range"] = lg(ctx["last60m_range_pre"])
    f["abs_gap_atr"] = wz(np.abs(pd.to_numeric(ctx["gap_atr_prior"], errors="coerce")))
    f["on_vol_rel_c"] = wz(ctx["on_vol_rel"])
    f["loc_in_on_c"] = wz(ctx["loc_in_on_pre"], 0.02, 0.98)
    f["macro_volx_c"] = np.log1p(wz(ctx["macro_volx_pre"]))
    return f


@dataclass
class ScaleModel:
    """Ridge regression of ``log R`` on pre-open context."""
    features: List[str]
    coef: np.ndarray
    intercept: float
    mu: np.ndarray
    sd: np.ndarray
    resid_sd: float

    def predict_log(self, f: pd.DataFrame) -> np.ndarray:
        x = f[self.features].to_numpy(dtype="float64")
        x = np.where(np.isfinite(x), x, self.mu)          # impute to train mean
        xs = (x - self.mu) / self.sd
        return self.intercept + xs @ self.coef

    def predict(self, f: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(f))


def fit_scale_model(f: pd.DataFrame, r: np.ndarray, features: Sequence[str],
                    alpha: float = 1.0) -> ScaleModel:
    """Fit ``log R ~ features`` with standardised ridge.

    Ridge rather than OLS because the volatility regressors are strongly
    collinear by construction — ATR, overnight range and realised vol all
    measure the same thing at different lags — and the goal is a stable scale,
    not an interpretation of individual coefficients.
    """
    feats = list(features)
    x = f[feats].to_numpy(dtype="float64")
    y = np.log(np.clip(np.asarray(r, dtype="float64"), 1.0, None))
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[ok], y[ok]
    if len(y) < 50:
        raise ValueError(f"only {len(y)} usable training sessions for the scale model")
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    xs = (x - mu) / sd
    n, p = xs.shape
    a = xs.T @ xs + alpha * np.eye(p)
    b = xs.T @ (y - y.mean())
    coef = np.linalg.solve(a, b)
    intercept = float(y.mean())
    resid = y - (intercept + xs @ coef)
    return ScaleModel(feats, coef, intercept, mu, sd, float(resid.std()))


# --------------------------------------------------------------------------- #
# the baseline
# --------------------------------------------------------------------------- #


DIRECTION_FEATURES: Tuple[str, ...] = (
    "gap_atr_c", "on_ret_atr", "pre_ret_atr", "last60_ret_atr",
    "loc_in_on_c2", "prior_ret_atr", "on_delta_rel",
)


def build_direction_features(ctx: pd.DataFrame) -> pd.DataFrame:
    """Context -> regressors for *which side* of the reference the range falls on.

    All volatility-normalised, because "gapped 40 points up" means something
    different on a 150-point day and a 500-point day.
    """
    atr = pd.to_numeric(ctx["atr14_t"], errors="coerce").to_numpy("float64")
    atr = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)

    def wz(v, lo=0.01, hi=0.99):
        v = np.asarray(v, dtype="float64")
        a, b = np.nanquantile(v, lo), np.nanquantile(v, hi)
        return np.clip(v, a, b)

    def col(name):
        return pd.to_numeric(ctx[name], errors="coerce").to_numpy("float64")

    f = pd.DataFrame(index=ctx.index)
    f["gap_atr_c"] = wz(col("gap_t_prior") / atr)
    f["on_ret_atr"] = wz(col("globex_ret_pre") / atr)
    f["pre_ret_atr"] = wz(col("premarket_ret_pre") / atr)
    f["last60_ret_atr"] = wz(col("last60m_ret_pre") / atr)
    f["loc_in_on_c2"] = wz(col("loc_in_on_pre"), 0.02, 0.98)
    f["prior_ret_atr"] = wz(col("prior_rth_ret_prior") / atr)
    on_vol = col("globex_volume_pre")
    f["on_delta_rel"] = wz(np.divide(col("globex_delta_pre"), on_vol,
                                     out=np.full_like(on_vol, np.nan), where=on_vol > 0))
    return f


@dataclass
class ShareModel:
    """Ridge regression of the up-share ``U / (U + D)`` on directional context.

    Splits the predicted range asymmetrically instead of assuming the reference
    sits in the middle of it. Without this the model under-predicts whichever
    side happens to have the fatter tail in the evaluation period, which would
    make every level on that side look spuriously attractive — a bias that
    would contaminate the whole level-family comparison downstream.
    """
    features: List[str]
    coef: np.ndarray
    intercept: float
    mu: np.ndarray
    sd: np.ndarray
    floor: float = 0.15

    def predict(self, f: pd.DataFrame) -> np.ndarray:
        x = f[self.features].to_numpy("float64")
        x = np.where(np.isfinite(x), x, self.mu)
        p = self.intercept + ((x - self.mu) / self.sd) @ self.coef
        return np.clip(p, self.floor, 1.0 - self.floor)


def fit_share_model(f: pd.DataFrame, up: np.ndarray, dn: np.ndarray,
                    features: Sequence[str] = DIRECTION_FEATURES,
                    alpha: float = 5.0, floor: float = 0.15) -> ShareModel:
    feats = list(features)
    x = f[feats].to_numpy("float64")
    r = np.asarray(up, "float64") + np.asarray(dn, "float64")
    y = np.where(r > 0, np.asarray(up, "float64") / np.where(r > 0, r, 1.0), np.nan)
    y = np.clip(y, -0.5, 1.5)
    ok = np.isfinite(y) & np.isfinite(x).all(axis=1)
    x, y = x[ok], y[ok]
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    xs = (x - mu) / sd
    coef = np.linalg.solve(xs.T @ xs + alpha * np.eye(xs.shape[1]), xs.T @ (y - y.mean()))
    return ShareModel(feats, coef, float(y.mean()), mu, sd, floor)


@dataclass
class ExcursionBaseline:
    scale_model: ScaleModel
    surv: Dict[Tuple[str, int], Survival] = field(default_factory=dict)
    surv_by_bin: Dict[Tuple[str, int, str], Survival] = field(default_factory=dict)
    bin_col: Optional[str] = None
    n_sessions: int = 0
    share_model: Optional[ShareModel] = None

    # -- scale ----------------------------------------------------------- #
    def scale(self, ctx: pd.DataFrame) -> np.ndarray:
        """Per-side scale. Returns one array when there is no share model, or
        a 2-column array ``[up, down]`` when the range is split directionally."""
        s = self.scale_model.predict(build_scale_features(ctx))
        if self.share_model is None:
            return s
        share = self.share_model.predict(build_direction_features(ctx))
        return np.column_stack([s * share, s * (1.0 - share)])

    @staticmethod
    def _side_scale(s: np.ndarray, side: str) -> np.ndarray:
        if np.ndim(s) == 1:
            return s
        return s[:, 0] if side == "above" else s[:, 1]

    # -- probability ------------------------------------------------------ #
    def p_touch(self, d_t, s: np.ndarray, side: str, horizon: int = 150,
                bins: Optional[Sequence[str]] = None) -> np.ndarray:
        """P(price travels ``d_t`` ticks to ``side`` by ``horizon`` minutes).

        ``s`` is the per-row scale from :meth:`scale`; passing it in rather than
        recomputing keeps the per-candidate cost to one interpolation.
        """
        d = np.asarray(d_t, dtype="float64")
        z = d / self._side_scale(np.asarray(s, dtype="float64"), side)
        if bins is not None and self.surv_by_bin:
            out = np.full(z.shape, np.nan)
            b = np.asarray(bins, dtype=object)
            for key in set(b[pd.notna(b)]):
                m = b == key
                fn = self.surv_by_bin.get((side, horizon, str(key)))
                out[m] = (fn or self.surv[(side, horizon)])(z[m])
            miss = ~np.isfinite(out)
            if miss.any():
                out[miss] = self.surv[(side, horizon)](z[miss])
            return out
        return self.surv[(side, horizon)](z)


def fit_baseline(ctx: pd.DataFrame, features: Sequence[str] = DEFAULT_SCALE_FEATURES,
                 horizons: Sequence[int] = HORIZONS, alpha: float = 1.0,
                 bin_col: Optional[str] = None, min_bin: int = 120,
                 directional: bool = False, share_alpha: float = 5.0
                 ) -> ExcursionBaseline:
    """Fit the whole baseline on training sessions.

    ``bin_col`` optionally lets the *shape* vary by a coarse regime label as
    well as the scale; bins with fewer than ``min_bin`` sessions fall back to
    the pooled survival rather than being fitted from thin air.
    """
    h_max = max(horizons)
    up, dn = ctx[f"up{h_max}_out"].to_numpy("float64"), ctx[f"dn{h_max}_out"].to_numpy("float64")
    r = up + dn
    f = build_scale_features(ctx)
    sm = fit_scale_model(f, r, features, alpha=alpha)
    b = ExcursionBaseline(scale_model=sm, bin_col=bin_col, n_sessions=int(len(ctx)))
    if directional:
        b.share_model = fit_share_model(build_direction_features(ctx), up, dn,
                                        alpha=share_alpha)
    s = b.scale(ctx)

    for side in SIDES:
        ss = ExcursionBaseline._side_scale(s, side)
        for h in horizons:
            z = ctx[f"{_EX_COL[side]}{h}_out"].to_numpy("float64") / ss
            b.surv[(side, h)] = fit_survival(z)
            if bin_col is not None and bin_col in ctx:
                for key, idx in ctx.groupby(bin_col, observed=True).groups.items():
                    zz = z[ctx.index.get_indexer(idx)]
                    zz = zz[np.isfinite(zz)]
                    if zz.size >= min_bin:
                        b.surv_by_bin[(side, h, str(key))] = fit_survival(zz)
    return b


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


def touch_panel(ctx: pd.DataFrame, baseline: ExcursionBaseline,
                distances_atr: Sequence[float] = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0),
                horizon: int = 150, scale_col: str = "atr14_t",
                bin_col: Optional[str] = None) -> pd.DataFrame:
    """One row per (session, side, distance): predicted vs realised touch.

    Distances are expressed in ATR so the panel spans the same *relative*
    geometry every day rather than piling observations at whatever absolute
    distance happened to be reachable.
    """
    s = baseline.scale(ctx)
    atr = ctx[scale_col].to_numpy("float64")
    sc_flat = s if np.ndim(s) == 1 else s.mean(axis=1)
    bins = ctx[bin_col].astype(str).to_numpy() if bin_col and bin_col in ctx else None
    rows = []
    for side in SIDES:
        ex = ctx[f"{_EX_COL[side]}{horizon}_out"].to_numpy("float64")
        for k in distances_atr:
            d = k * atr
            p = baseline.p_touch(d, s, side, horizon, bins=bins)
            rows.append(pd.DataFrame({
                "session": ctx["session"].to_numpy(), "side": side,
                "dist_atr": k, "d_t": d, "scale": sc_flat,
                "p_pred": p, "touched": (ex >= d).astype("int8"),
            }))
    out = pd.concat(rows, ignore_index=True)
    return out[np.isfinite(out["d_t"]) & np.isfinite(out["p_pred"])]


def calibration(panel: pd.DataFrame, n_bins: int = 12) -> pd.DataFrame:
    """Reliability table: predicted probability versus realised frequency."""
    d = panel.dropna(subset=["p_pred", "touched"]).copy()
    d["_b"] = pd.qcut(d["p_pred"], n_bins, duplicates="drop")
    g = d.groupby("_b", observed=True).agg(
        n=("touched", "size"), sessions=("session", "nunique"),
        p_pred=("p_pred", "mean"), p_obs=("touched", "mean"))
    g["gap"] = g["p_obs"] - g["p_pred"]
    return g.reset_index(drop=True)


def scores(panel: pd.DataFrame) -> Dict[str, float]:
    """Brier, log loss, ECE and the base-rate reference."""
    p = np.clip(panel["p_pred"].to_numpy("float64"), 1e-6, 1 - 1e-6)
    y = panel["touched"].to_numpy("float64")
    cal = calibration(panel)
    w = cal["n"] / cal["n"].sum()
    base = y.mean()
    brier = float(np.mean((p - y) ** 2))
    return {
        "n": int(len(y)), "sessions": int(panel["session"].nunique()),
        "base_rate": float(base),
        "brier": brier,
        "brier_base": float(np.mean((base - y) ** 2)),
        "brier_skill": float(1 - brier / np.mean((base - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "ece": float((w * cal["gap"].abs()).sum()),
        "max_gap": float(cal["gap"].abs().max()),
    }
