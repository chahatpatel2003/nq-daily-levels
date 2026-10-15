#!/usr/bin/env python
"""Build the price-axis grid: every price around the market, described and measured.

This is the central dataset of Phase 2, and it exists because the obvious
comparison is impossible. At a given distance and side there is exactly **one**
price, so "a control level at the same distance" cannot be drawn — any control
must sit at a different distance, and distance drives everything. The only
sound way to ask "does being at a level matter?" is to lay down a uniform grid
over the reachable price axis, mark which grid prices are at a level, and
compare **within distance-and-side strata**.

That framing also subsumes the algorithmic-discovery question: once every price
carries a feature vector and an outcome, "which prices matter" is a modelling
problem over this table, and named families are just indicator columns in it
that have to earn their weight like anything else.

Grid: every 4 ticks (1 index point) out to ±1.5 ATR from the 06:25 reference —
about 500 prices per session, 700 k rows over train+validation.

Every feature is computed from pre-cut information only. Outcome columns are
suffixed and never fed back in.
"""
from __future__ import annotations

import argparse
import sys
import time as _time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.build.profile import densify, smooth                             # noqa: E402
from nqlevels.build.tape import window_profile                                 # noqa: E402
from nqlevels.clock import (decision_time_utc, load_calendar,                  # noqa: E402
                            prior_rth_window_utc, prior_trade_date,
                            segment_windows_utc, session_open_utc, trade_dates)
from nqlevels.eval.universe import CORE_KINDS, DENSE_KINDS, explode_kinds      # noqa: E402
from nqlevels.io.raw import assert_not_raw                                     # noqa: E402
from nqlevels.io.store import layer_dir, read_layer                            # noqa: E402

CONFIGS = str(ROOT / "configs")
TICK = 0.25


def _profile_feats(grid: np.ndarray, prof: pd.DataFrame, ref: float,
                   prefix: str, smooth_t: int = 8) -> dict:
    """Volume at, and around, each grid price."""
    out = {}
    if prof is None or prof.empty:
        for c in ("vol", "volrel", "grad"):
            out[f"{prefix}_{c}"] = np.full(grid.size, np.nan)
        return out
    d = smooth(prof, smooth_t)
    px = d["price_t"].to_numpy("float64") - ref
    v = d["vol_s"].to_numpy("float64")
    order = np.argsort(px)
    px, v = px[order], v[order]
    vv = np.interp(grid, px, v, left=0.0, right=0.0)
    local = pd.Series(vv).rolling(60, center=True, min_periods=1).mean().to_numpy()
    tot = float(v.sum()) or 1.0
    out[f"{prefix}_vol"] = vv / tot
    out[f"{prefix}_volrel"] = np.divide(vv, local, out=np.full_like(vv, np.nan),
                                        where=local > 0)
    out[f"{prefix}_grad"] = np.gradient(vv) if vv.size > 1 else np.zeros_like(vv)
    return out


def _interaction_feats(grid: np.ndarray, bars: pd.DataFrame, ref: float,
                       prefix: str) -> dict:
    """How often, how long and how recently price has been at each grid price."""
    out = {}
    if bars is None or bars.empty:
        for c in ("touches", "mins_at", "recency", "visitlen"):
            out[f"{prefix}_{c}"] = np.full(grid.size, np.nan)
        return out
    hi = bars["high_t"].to_numpy("float64") - ref
    lo = bars["low_t"].to_numpy("float64") - ref
    n = hi.size
    inside = (lo[None, :] <= grid[:, None]) & (hi[None, :] >= grid[:, None])
    mins = inside.sum(axis=1)
    starts = inside & ~np.concatenate([np.zeros((grid.size, 1), bool), inside[:, :-1]], axis=1)
    touches = starts.sum(axis=1)
    last = np.where(inside.any(axis=1), inside.shape[1] - 1 - np.argmax(inside[:, ::-1], axis=1), np.nan)
    out[f"{prefix}_touches"] = touches.astype("float32")
    out[f"{prefix}_mins_at"] = mins.astype("float32")
    out[f"{prefix}_recency"] = (last / max(1, n - 1)).astype("float32")
    out[f"{prefix}_visitlen"] = np.where(touches > 0, mins / np.maximum(touches, 1), np.nan)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-02-15")
    ap.add_argument("--end", default="2026-06-17")
    ap.add_argument("--step-t", type=int, default=4, help="grid step in ticks")
    ap.add_argument("--span-atr", type=float, default=1.5)
    ap.add_argument("--near-t", type=float, default=6.0,
                    help="ticks: how close counts as 'at' a level")
    a = ap.parse_args()

    cal = load_calendar(CONFIGS)
    d = layer_dir("daily", CONFIGS)
    print("loading ...")
    ctx = pd.read_parquet(d / "context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    bars = pd.read_parquet(d / "bars_1m.parquet")
    bars["ts"] = pd.DatetimeIndex(bars["ts"]).tz_convert("UTC")
    prof = read_layer("profiles", CONFIGS)
    prof["bucket"] = pd.DatetimeIndex(prof["bucket"]).tz_convert("UTC")
    cand = pd.read_parquet(d / "candidates_named.parquet")
    cx = explode_kinds(cand)
    core_by = {s: g for s, g in cx[cx["kind"].isin(CORE_KINDS)].groupby("session", sort=False)}
    dense_by = {s: g for s, g in cx[cx["kind"].isin(DENSE_KINDS)].groupby("session", sort=False)}
    bars_by = {s: g for s, g in bars[bars["is_front"]].groupby("session", sort=False)}
    prof_by = {s: g for s, g in prof[prof["is_front"]].groupby("session", sort=False)}
    ctxi = ctx.set_index("session")
    print(f"  ctx {len(ctx)} | bars_1m {len(bars):,} | profiles {len(prof):,} | cand {len(cand):,}")

    sessions = trade_dates(date.fromisoformat(a.start), date.fromisoformat(a.end), cal)
    out, t0, skipped = [], _time.time(), 0

    for i, s in enumerate(sessions):
        if s not in ctxi.index or s not in bars_by:
            skipped += 1
            continue
        row = ctxi.loc[s]
        ref, atr = float(row["ref_t"]), float(row["atr14_t"])
        if not (np.isfinite(ref) and np.isfinite(atr) and atr > 0):
            skipped += 1
            continue
        span = int(a.span_atr * atr)
        grid = np.arange(-(span // a.step_t) * a.step_t,
                         (span // a.step_t) * a.step_t + a.step_t, a.step_t, dtype="float64")

        cut = pd.Timestamp(decision_time_utc(s, cal))
        fb = bars_by[s]
        pre = fb[fb["ts"] < cut]
        pd_ = prior_trade_date(s, cal)
        r_lo, r_hi = prior_rth_window_utc(s, cal)
        pb = bars_by.get(pd_, pd.DataFrame())
        prior_rth = pb[(pb["ts"] >= r_lo) & (pb["ts"] < r_hi)] if len(pb) else pd.DataFrame()

        f = {"session": s, "off_t": grid}
        f["dist_t"] = np.abs(grid)
        f["dist_atr"] = np.abs(grid) / atr
        f["above"] = (grid > 0).astype("int8")

        # round-number geometry
        px = (grid + ref) * TICK
        for step in (25, 50, 100):
            f[f"round{step}_d"] = np.abs(((px + step / 2) % step) - step / 2)

        # profiles: overnight (pre-cut), prior RTH, 5- and 20-session composites
        p_s = prof_by.get(s)
        if p_s is not None and len(p_s):
            on = window_profile(p_s, pd.Timestamp(session_open_utc(s, cal)).floor("30min"),
                                cut.floor("30min"))
        else:
            on = pd.DataFrame()
        pp = prof_by.get(pd_)
        pdrth = window_profile(pp, pd.Timestamp(r_lo), pd.Timestamp(r_hi)) if pp is not None and len(pp) else pd.DataFrame()
        lb = [pd_]
        dd = pd_
        for _ in range(19):
            dd = prior_trade_date(dd, cal)
            lb.append(dd)
        lbp = [prof_by[x] for x in lb[:20] if x in prof_by]
        comp = (pd.concat(lbp, ignore_index=True)
                  .groupby("price_t", as_index=False)[["volume", "n_trades", "buy_vol", "sell_vol"]]
                  .sum() if lbp else pd.DataFrame())
        f.update(_profile_feats(grid, on, ref, "on"))
        f.update(_profile_feats(grid, pdrth, ref, "pd"))
        f.update(_profile_feats(grid, comp, ref, "lb20"))

        # interaction history
        f.update(_interaction_feats(grid, pre, ref, "on"))
        f.update(_interaction_feats(grid, prior_rth, ref, "pd"))
        lbb = [bars_by[x] for x in lb[:20] if x in bars_by]
        f.update(_interaction_feats(grid, pd.concat(lbb, ignore_index=True) if lbb else pd.DataFrame(),
                                    ref, "lb20"))

        # distance to the nearest core / dense level, and to each core kind
        for tag, src in (("core", core_by), ("dense", dense_by)):
            g = src.get(s)
            if g is None or g.empty:
                f[f"d_{tag}_t"] = np.full(grid.size, np.nan)
                continue
            lv = np.sort(g["off_t"].to_numpy("float64"))
            idx = np.searchsorted(lv, grid)
            a1 = np.abs(grid - lv[np.clip(idx - 1, 0, lv.size - 1)])
            a2 = np.abs(grid - lv[np.clip(idx, 0, lv.size - 1)])
            f[f"d_{tag}_t"] = np.minimum(a1, a2)
        gc = core_by.get(s)
        if gc is not None and len(gc):
            for kind, gg in gc.groupby("kind"):
                lv = np.sort(gg["off_t"].to_numpy("float64"))
                idx = np.searchsorted(lv, grid)
                a1 = np.abs(grid - lv[np.clip(idx - 1, 0, lv.size - 1)])
                a2 = np.abs(grid - lv[np.clip(idx, 0, lv.size - 1)])
                f[f"dk_{kind}"] = np.minimum(a1, a2)

        out.append(pd.DataFrame(f))
        if i % 200 == 0:
            print(f"  {s}: {sum(len(x) for x in out):,} rows, {_time.time()-t0:.0f}s", flush=True)

    ax = pd.concat(out, ignore_index=True)
    for c in ax.columns:
        if ax[c].dtype == "float64" and c not in ("off_t",):
            ax[c] = ax[c].astype("float32")
    ax["near_core"] = (ax["d_core_t"] <= a.near_t).astype("int8")
    ax["near_dense"] = (ax["d_dense_t"] <= a.near_t).astype("int8")
    dst = assert_not_raw(d / "axis.parquet")
    ax.to_parquet(dst, index=False, compression="zstd")
    print(f"\naxis: {len(ax):,} rows over {ax['session'].nunique()} sessions "
          f"({len(ax)/ax['session'].nunique():.0f}/session), {ax.shape[1]} cols, "
          f"skipped {skipped}, {dst.stat().st_size/1e6:.0f} MB, "
          f"{(_time.time()-t0)/60:.1f} min")
    print(f"near_core share {ax['near_core'].mean():.3f} | "
          f"near_dense share {ax['near_dense'].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
