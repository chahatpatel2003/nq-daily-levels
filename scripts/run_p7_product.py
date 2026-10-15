#!/usr/bin/env python
"""P7 Parts 3-6 — the resistance-pivot product numbers.

1. Magnitude/adverse/mode tables for the RESISTANCE population only.
2. Width decision: strict lift at ±5/±10 within volatility terciles.
3. Lifetime/invalidation summaries for resistance pivots.
4. The live product, offline from the stored P5 pool: resistance-only
   display (top-4 by the walk-forward state score at each refresh),
   episodes, touch/outcome metrics per year, and the exact-anchor advantage
   verified on DISPLAYED-and-tapped zones via their Phase-6 shadow rows.
"""
from __future__ import annotations

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
from run_p5_report import attach_touch, build_episodes, tap_table              # noqa: E402
from run_p6_magnitude import fmt, stats_pts                                    # noqa: E402
from run_p6_spatial import add_labels, paired                                  # noqa: E402
from run_p5_live import rank_pool                                              # noqa: E402

CONFIGS = str(ROOT / "configs")
KEY = ["session", "zid", "tap_sec"]


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    res: dict = {}
    re = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    it = pd.read_parquet(d0 / "interactions.parquet",
                         columns=KEY + ["atr_pctile", "tod_pt_min"]).drop_duplicates(KEY)
    re = re.merge(it, on=KEY, how="left", validate="1:1")
    R = re[(re["side_above"] == 0) & (re["n_fwd"] >= 1800)
           & re["y"].notna()].copy()
    S = R[R["y_strict"] == 1]
    print(f"resistance taps: {len(R):,} | strict {float(R['y_strict'].mean()):.4f} "
          f"| sweep-incl {float(R['y'].mean()):.4f}")

    print("\n=== magnitudes, resistance successes (strict), NQ points ===")
    c = classify(R, R=15, d0_t=4.0, D_t=R["atr_t"] * 0.10, T_s=1800, level="ne")
    R["mode"] = np.where(c["cat_B"] == 1, "sweep",
                         np.where(R["turn_frac_rev15"] <= 0.5, "front", "deep"))
    Sy = R[R["y"] == 1]
    print(f"  reaction (endpoint): {fmt(stats_pts(Sy['rev_ep_pts']))}")
    print(f"  adverse before 15pt: {fmt(stats_pts(Sy['adv15_pts']))}")
    res["reaction"] = stats_pts(Sy["rev_ep_pts"])
    res["adverse"] = stats_pts(Sy["adv15_pts"])
    for md, g in Sy.groupby("mode"):
        print(f"    {md:>6} (share {len(g)/len(Sy):.2f}): rev "
              f"{float(g['rev_ep_pts'].median()):5.1f} med | adv "
              f"{float(g['adv15_pts'].median()):5.1f} med")
        res[f"mode_{md}"] = {"share": len(g) / len(Sy),
                             "rev_med": float(g["rev_ep_pts"].median()),
                             "adv_med": float(g["adv15_pts"].median())}
    B = Sy[Sy["mode"] == "sweep"]
    print(f"  sweeps: depth med {float(B['depth_before_ne'].median())/4:.1f} pts | "
          f"reclaim med {float(((B['t_reclaim_ne']-B['first_beyond'])/60).median()):.1f} min | "
          f"after med {float((B['rev_after_ne_1800']/4).median()):.1f} pts")
    fails = R[R["y"] == 0]
    print(f"  failures penetrate: med {float(fails['pen_max_full_t'].median())/4:.0f} pts")

    print("\n=== width: strict exact-anchor lift at ±5 / ±10 by vol tercile ===")
    sh = add_labels(pd.read_parquet(d0 / "p6_shadow.parquet"))
    Rt = R[R["has_t0"] & ~R["inside_t0"]].copy()
    Rt["session_c"] = Rt["session"]
    q1, q2 = Rt["atr_pctile"].quantile([1 / 3, 2 / 3])
    for lab, mask in (("lowvol", Rt["atr_pctile"] <= q1),
                      ("midvol", (Rt["atr_pctile"] > q1) & (Rt["atr_pctile"] <= q2)),
                      ("hivol", Rt["atr_pctile"] > q2)):
        for off in (5.0, 10.0):
            r = paired(Rt[mask], sh[sh["side_above"] == 0], "y_strict", off, 150)
            if r:
                print(f"  {lab:>7} ±{off:>4}: {r['diff']:+.4f} "
                      f"[{r['lo']:+.4f},{r['hi']:+.4f}]")
                res.setdefault("width", {})[f"{lab}_{off}"] = r

    # ---- live resistance-only display, offline from the P5 pool ------------
    print("\n=== live resistance display (offline re-rank of the P5 pool) ===")
    pool = pd.read_parquet(d0 / "p5_pool.parquet")
    pool = pool[pool["app_side_above"] == 0].copy()
    pool["rank"] = 0
    for (s, t), g in pool.groupby(["session", "t_sec"], sort=False):
        pool.loc[g.index, "rank"] = rank_pool(g, float(g["atr_t"].iloc[0]))
    itaps = tap_table()
    taps_by = {}
    for (sess, zid), g in itaps.groupby(["session", "zid"], sort=False):
        g = g.sort_values("tap_sec")
        taps_by[(sess, zid)] = (g["tap_sec"].to_numpy("int64"),
                                g["y"].to_numpy(), g["y_strict"].to_numpy(),
                                g["catA"].to_numpy(), g["catB"].to_numpy())
    ep = attach_touch(build_episodes(pool), taps_by)
    ep["year"] = pd.to_datetime(ep["session"].astype(str)).dt.year
    per = ep.groupby("session").size()
    t = ep[ep["touched_ss"] & ep["y"].notna()]
    print(f"episodes {len(ep):,} over {ep['session'].nunique()} sessions "
          f"({per.median():.0f}/day) | touched-ss {float(ep['touched_ss'].mean()):.3f} "
          f"| touched-while-disp {float(ep['touched_disp'].mean()):.3f}")
    print(f"y|touch {float(t['y'].mean()):.4f} | strict {float(t['y_strict'].mean()):.4f} "
          f"| med dist {float(ep['dist0'].median()):.3f} ATR")
    res["display"] = {"eps_per_day": float(per.median()),
                      "touch_ss": float(ep["touched_ss"].mean()),
                      "touch_disp": float(ep["touched_disp"].mean()),
                      "y": float(t["y"].mean()),
                      "y_strict": float(t["y_strict"].mean()),
                      "dist_med": float(ep["dist0"].median())}
    print("per year (strict | sweep-incl | touch):")
    for y_, gy in ep.groupby("year"):
        ty = gy[gy["touched_ss"] & gy["y"].notna()]
        print(f"  {y_}: {float(ty['y_strict'].mean()):.3f} | "
              f"{float(ty['y'].mean()):.3f} | {float(gy['touched_ss'].mean()):.3f} "
              f"(n={len(gy):,})")
        res.setdefault("display_yr", {})[int(y_)] = {
            "strict": float(ty["y_strict"].mean()), "y": float(ty["y"].mean()),
            "touch": float(gy["touched_ss"].mean())}
    dd = ep[ep["touched_disp"]]
    print(f"time display->touch: med {float(((dd['tap_sec']-dd['t0'])/60).median()):.0f} min | "
          f"successes/day (strict, while displayed): "
          f"{float(ep.assign(s=(ep['touched_disp'] & (ep['y_strict'].fillna(0)==1))).groupby('session')['s'].sum().mean()):.1f}")

    # exact-anchor advantage among DISPLAYED-and-tapped zones
    disp_taps = dd.copy()
    disp_taps["tap_sec"] = disp_taps["tap_sec"].astype("int64")
    j = disp_taps.merge(re[KEY + ["has_t0", "inside_t0", "n_fwd"]], on=["session", "zid"],
                        how="inner", suffixes=("", "_r"))
    j = j[(j["tap_sec_r"] == j["tap_sec"]) if "tap_sec_r" in j.columns else
          j.index >= 0]
    gg = re[re["has_t0"] & ~re["inside_t0"] & (re["n_fwd"] >= 1800)
            & re["y_strict"].notna()].copy()
    gg["session_c"] = gg["session"]
    keys = set(map(tuple, disp_taps[["session", "zid"]].to_numpy()))
    gg = gg[[tuple(x) in keys for x in gg[["session", "zid"]].to_numpy()]]
    r = paired(gg, sh, "y_strict", 10.0, 200)
    if r:
        print(f"exact-anchor lift among displayed-and-tapped zones: "
              f"{r['diff']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}] (n={r['n_events']:,})")
        res["display_anchor_lift"] = r

    out = ROOT / "outputs" / "p7_product.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
