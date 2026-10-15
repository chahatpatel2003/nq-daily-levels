#!/usr/bin/env python
"""P4-306 — do the discovered distinctions identify strong zones in advance?

Selection test among reached zones, out of time: per session, rank that
session's deduplicated resolved interactions by the walk-forward OOS score
(T1 = knowable while the zone sits untapped; T2 = including the approach,
knowable seconds before the tap), select the top 1..4, and compare their
success rate against same-session controls:

  random   — the session's average tap (all deduped resolved taps)
  matched  — random draws matched on (segment × approach side), 8 draws
  next     — the following K in the ranking
  bottom   — the lowest-ranked K

Paired session-clustered bootstrap on top-K minus matched; per-year tables.
Plus the availability answers: how many valid zones typically sit near price,
and how far the strongest ones are.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.io.store import layer_dir                                        # noqa: E402

CONFIGS = str(ROOT / "configs")


def paired_ci(mrows, crows, n_boot=500, seed=11):
    common = np.intersect1d(mrows["session"].unique(), crows["session"].unique())
    if common.size < 8:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan}
    m = mrows[mrows["session"].isin(common)].reset_index(drop=True)
    c = crows[crows["session"].isin(common)].reset_index(drop=True)
    gm = {s: np.flatnonzero(m["session"].to_numpy() == s) for s in common}
    gc = {s: np.flatnonzero(c["session"].to_numpy() == s) for s in common}
    point = float(m["y"].mean() - c["y"].mean())
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(common, common.size, replace=True)
        vals[b] = (m.iloc[np.concatenate([gm[s] for s in pick])]["y"].mean()
                   - c.iloc[np.concatenate([gc[s] for s in pick])]["y"].mean())
    return {"diff": point, "lo": float(np.quantile(vals, .025)),
            "hi": float(np.quantile(vals, .975)),
            "p_gt0": float((vals > 0).mean())}


def run_tier(s, tier, ks=(1, 2, 3, 4), n_draws=8, boot=500):
    col = f"p_{tier}"
    d = s[s[col].notna() & s["y"].notna()].copy()
    print(f"\n=== ranking by {tier} (OOS), {len(d):,} taps, "
          f"{d['session'].nunique()} sessions, base y {d['y'].mean():.4f} ===")
    out = {}
    rng = np.random.default_rng(7)
    for K in ks:
        picks, nxt, bot, mat = [], [], [], []
        for sess, g in d.groupby("session"):
            g = g.sort_values(col, ascending=False)
            if len(g) < K + 2:
                continue
            p = g.head(K)
            picks.append(p)
            nxt.append(g.iloc[K:2 * K])
            bot.append(g.tail(K))
            for _, r in p.iterrows():
                pool = g[(g["seg"] == r["seg"])
                         & (g["app_side_above"] == r["app_side_above"])]
                pool = pool[pool.index != r.name]
                if len(pool):
                    q = pool.sample(min(n_draws, len(pool)),
                                    random_state=int(rng.integers(1 << 31)))
                    mat.append(q)
        P = pd.concat(picks, ignore_index=True)
        N = pd.concat(nxt, ignore_index=True) if nxt else pd.DataFrame()
        B = pd.concat(bot, ignore_index=True)
        M = pd.concat(mat, ignore_index=True) if mat else pd.DataFrame()
        ci = paired_ci(P, M, boot) if len(M) else {}
        row = {"top": float(P["y"].mean()), "n": int(len(P)),
               "sessions": int(P["session"].nunique()),
               "matched": float(M["y"].mean()) if len(M) else np.nan,
               "next": float(N["y"].mean()) if len(N) else np.nan,
               "bottom": float(B["y"].mean()),
               "ci": ci}
        out[f"K{K}"] = row
        print(f"  K={K}: top {row['top']:.4f} | matched {row['matched']:.4f} "
              f"| next {row['next']:.4f} | bottom {row['bottom']:.4f} | "
              f"top-matched {ci.get('diff', np.nan):+.4f} "
              f"[{ci.get('lo', np.nan):+.4f},{ci.get('hi', np.nan):+.4f}]")
        if K == 2:
            yr = {}
            P2, M2 = P.copy(), M.copy()
            P2["yr"] = pd.to_datetime(P2["session"].astype(str)).dt.year
            M2["yr"] = pd.to_datetime(M2["session"].astype(str)).dt.year
            for y in sorted(P2["yr"].unique()):
                a_ = P2[P2["yr"] == y]["y"].mean()
                b_ = M2[M2["yr"] == y]["y"].mean()
                yr[int(y)] = {"top": float(a_), "matched": float(b_)}
            row["by_year"] = yr
            print("    by year (K=2): " + " ".join(
                f"{y}:{v['top']:.3f}/{v['matched']:.3f}" for y, v in yr.items()))
    return out


def availability(d0) -> dict:
    z = pd.read_parquet(d0 / "zones.parquet",
                        columns=["session", "zid", "price_rf", "form_ord",
                                 "form_ts_val", "scale", "hw_t"])
    it = pd.read_parquet(d0 / "interactions.parquet",
                         columns=["zid", "tap_ts"])
    ctx = pd.read_parquet(d0 / "context.parquet",
                          columns=["session", "ref_t", "atr14_t", "ref_ts"])
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    adj = pd.read_parquet(d0 / "roll_adjust.parquet").set_index("session")
    ord_of = {s: i for i, s in enumerate(sorted(ctx["session"]))}
    taps_by_zone = it.groupby("zid")["tap_ts"].apply(sorted).to_dict()
    rows = []
    ctx = ctx.set_index("session")
    zs = z.sort_values("form_ord")
    f_ord = zs["form_ord"].to_numpy()
    for s in sorted(ctx.index)[40::10]:               # every 10th session
        if s not in adj.index:
            continue
        o = ord_of[s]
        ref = float(ctx.loc[s, "ref_t"])
        atr = float(ctx.loc[s, "atr14_t"])
        ref_ts = pd.Timestamp(ctx.loc[s, "ref_ts"]).timestamp()
        a0, a1 = np.searchsorted(f_ord, o - 30), np.searchsorted(f_ord, o + 1)
        W = zs.iloc[a0:a1]
        W = W[W["form_ts_val"] < ref_ts]
        px = W["price_rf"].to_numpy() + float(adj.loc[s, "adj_t"])
        ntaps = np.array([len([t for t in taps_by_zone.get(zid, [])
                               if t < ref_ts]) for zid in W["zid"]])
        alive = ntaps < 6
        dist = np.abs(px[alive] - ref) / atr
        rows.append({"n_05": int((dist <= 0.5).sum()),
                     "n_10": int((dist <= 1.0).sum()),
                     "d_nearest4": float(np.sort(dist)[:4].mean()) if dist.size >= 4 else np.nan})
    av = pd.DataFrame(rows)
    return {"median_active_within_0.5atr": float(av["n_05"].median()),
            "median_active_within_1.0atr": float(av["n_10"].median()),
            "median_dist_nearest4_atr": float(av["d_nearest4"].median()),
            "sampled_sessions": len(av)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    s = pd.read_parquet(d0 / "p4_scores.parquet")
    res = {}
    for tier in ("T1", "T2"):
        res[tier] = run_tier(s, tier, boot=a.boot)
    print("\navailability near price (every 10th session, at the 06:25 ref):")
    av = availability(d0)
    res["availability"] = av
    for k, v in av.items():
        print(f"  {k}: {v}")
    out = ROOT / "outputs" / "p4_topk.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
