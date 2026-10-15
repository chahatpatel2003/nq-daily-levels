#!/usr/bin/env python
"""P6 spatial contrast — does X beat X±δ at the same moment? (Question 1)

Design (registered, RESEARCH_PLAN §17.1): every real deduped interaction has
8 shadow anchors at ±2.5/5/10/20 pts, identical width, common decision time
t0 = tap − 15 min, each scored from its own first touch under identical
rules. Primary estimate: EVENT-PAIRED — real outcome minus the mean of its
touched ± δ shadow pair (symmetric offsets neutralise distance to first
order) — with session-clustered bootstrap. Views: primary/strict outcomes,
reaction magnitude (endpoint-based, points), adverse-before-reaction,
excluding shadows that overlap another real zone, per year, per side, and
the offset decay curve that IS the empirical zone width.
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
sys.path.insert(0, str(ROOT / "scripts"))

from nqlevels.io.store import layer_dir                                        # noqa: E402
from nqlevels.labels.sweep import classify                                     # noqa: E402

CONFIGS = str(ROOT / "configs")
KEY = ["session", "zid", "tap_sec"]


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # the builder's touched flag was NaN for touched shadows (fixed since);
    # a touched anchor is exactly one with walked metrics
    if "touched" in d.columns:
        d["touched"] = d["touched"].fillna(0).astype(bool) | d["n_fwd"].notna()
    c = classify(d, R=15, d0_t=4.0, D_t=d["atr_t"] * 0.10, T_s=1800, level="ne")
    d["y"] = c["y_any"]
    cs = classify(d, R=15, d0_t=4.0, D_t=24.0, T_s=900, level="ne")
    d["y_strict"] = cs["y_any"]
    d["rev_ep_pts"] = d["rev_endpoint_t"] / 4.0
    d["adv15_pts"] = d["adverse_before_rev15_t"] / 4.0
    return d


def paired(re: pd.DataFrame, sh: pd.DataFrame, col: str, off_pts: float,
           n_boot=400, seed=5, sub=None):
    """real minus mean(±δ shadows), among events where real and both/either
    shadow are touched+resolved; session-clustered bootstrap."""
    s = sh[(sh["offset_t"].abs() == off_pts * 4) & sh["elig"]
           & (sh["touched"] == True) & (sh["n_fwd"] >= 1800)]          # noqa: E712
    if sub is not None:
        s = s[sub(s)]
    sm = s.groupby(KEY)[col].mean().rename("shadow")
    r = re.set_index(KEY)[[col, "session_c"]].rename(columns={col: "real"})
    j = r.join(sm, how="inner").dropna()
    if len(j) < 500:
        return None
    j = j.reset_index()
    d = j["real"] - j["shadow"]
    sess = j["session"].to_numpy()
    uniq = pd.unique(sess)
    idx = {u: np.flatnonzero(sess == u) for u in uniq}
    rng = np.random.default_rng(seed)
    v = np.empty(n_boot)
    dv = d.to_numpy()
    for b in range(n_boot):
        pick = rng.choice(uniq, uniq.size, replace=True)
        v[b] = np.nanmean(np.concatenate([dv[idx[u]] for u in pick]))
    return {"n_events": int(len(j)), "diff": float(np.nanmean(dv)),
            "lo": float(np.quantile(v, .025)), "hi": float(np.quantile(v, .975)),
            "real": float(j["real"].mean()), "shadow": float(j["shadow"].mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    re = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    sh = add_labels(pd.read_parquet(d0 / "p6_shadow.parquet"))
    re = re[re["has_t0"] & ~re["inside_t0"] & (re["n_fwd"] >= 1800)
            & re["y"].notna()].copy()
    re["session_c"] = re["session"]
    res: dict = {}
    print(f"events (real, eligible-at-t0, resolved): {len(re):,} | "
          f"shadow rows: {len(sh):,}")
    print(f"real y {float(re['y'].mean()):.4f} | shadow touched share by |off|: "
          + " ".join(f"{o}: {float(((sh['offset_t'].abs()==o*4)&sh['touched']).sum()/max(1,(sh['offset_t'].abs()==o*4).sum())):.2f}"
                     for o in (2.5, 5, 10, 20)))

    print("\n=== PRIMARY: event-paired real-minus-shadow, by |offset| ===")
    for col, label in (("y", "P(success|touch)"), ("y_strict", "strict"),
                       ("rev_ep_pts", "reaction pts (endpoint)"),
                       ("adv15_pts", "adverse-before-15pt pts")):
        print(f"  {label}:")
        res[col] = {}
        for off in (2.5, 5.0, 10.0, 20.0):
            r = paired(re, sh, col, off, a.boot)
            if r is None:
                print(f"    ±{off:>4}: underpowered")
                continue
            res[col][str(off)] = r
            print(f"    ±{off:>4}: real {r['real']:8.3f} vs shadow "
                  f"{r['shadow']:8.3f} | diff {r['diff']:+8.4f} "
                  f"[{r['lo']:+.4f},{r['hi']:+.4f}] (n={r['n_events']:,})")

    print("\n=== excluding shadows that overlap another real zone ===")
    for off in (5.0, 10.0, 20.0):
        r = paired(re, sh, "y", off, a.boot,
                   sub=lambda s: ~s["overlaps_zone"])
        if r:
            print(f"    ±{off:>4}: diff {r['diff']:+.4f} "
                  f"[{r['lo']:+.4f},{r['hi']:+.4f}] (n={r['n_events']:,})")
            res.setdefault("y_noovl", {})[str(off)] = r

    print("\n=== per year (y, ±10 pts) ===")
    res["by_year"] = {}
    for y_, gy in re.groupby("year"):
        shy = sh[sh["year"] == y_]
        r = paired(gy, shy, "y", 10.0, 200)
        if r:
            res["by_year"][int(y_)] = r
            print(f"  {y_}: {r['diff']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] "
                  f"(n={r['n_events']:,})")

    print("\n=== per side (y, ±10 pts) ===")
    for sd_, gs in re.groupby("side_above"):
        shs = sh[sh["side_above"] == sd_]
        r = paired(gs, shs, "y", 10.0, 200)
        if r:
            print(f"  side_above={sd_}: {r['diff']:+.4f} "
                  f"[{r['lo']:+.4f},{r['hi']:+.4f}] (n={r['n_events']:,})")

    print("\n=== decay curve: anchor performance by signed offset ===")
    st = sh[sh["elig"] & (sh["touched"] == True) & (sh["n_fwd"] >= 1800)   # noqa: E712
            & sh["y"].notna()]
    curve = {"0.0": {"y": float(re["y"].mean()),
                     "rev": float(re["rev_ep_pts"].mean()),
                     "n": int(len(re))}}
    for off, g in st.groupby(st["offset_t"] / 4.0):
        curve[str(off)] = {"y": float(g["y"].mean()),
                           "rev": float(g["rev_ep_pts"].mean()),
                           "n": int(len(g))}
    res["curve"] = curve
    for k in sorted(curve, key=float):
        v = curve[k]
        print(f"  {float(k):+6.1f} pts: y {v['y']:.4f} | rev {v['rev']:6.1f} "
              f"pts (n={v['n']:,})")

    print("\n=== pairwise: P(real better | discordant), y, by |offset| ===")
    for off in (2.5, 5.0, 10.0, 20.0):
        s = sh[(sh["offset_t"].abs() == off * 4) & sh["elig"]
               & (sh["touched"] == True) & (sh["n_fwd"] >= 1800)]      # noqa: E712
        m = s.groupby(KEY)["y"].mean().rename("sy")
        j = re.set_index(KEY)[["y"]].join(m, how="inner").dropna()
        disc = j[j["y"] != j["sy"]]
        if len(disc) > 500:
            acc = float((disc["y"] > disc["sy"]).mean())
            print(f"    ±{off:>4}: {acc:.4f} (n={len(disc):,})")
            res.setdefault("pairwise", {})[str(off)] = acc

    out = ROOT / "outputs" / "p6_spatial.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
