#!/usr/bin/env python
"""E2 — how wide should a zone be?

Run on the **core universe** (26 sparse classical kinds, ~28 levels/session).
Pooling all 86 generated kinds was tried first and is not a valid design: it
puts 137 levels per session inside ±1.5 ATR, which covers most of the price
axis, so a shifted control lands on another level and the contrast is null by
construction. See ``eval/universe.py``.

The trap this is designed around: any statistic that rewards width can be
maximised by drawing enormous zones. Three defences run throughout.

* Touch is scored against the E1 baseline evaluated at the zone's **near edge**,
  so widening moves the edge closer to the market and raises the null in step.
* Every number is reported next to a **control of identical width at matched
  distance**, drawn so that it is at least 3 points from any core level.
* The realised median width in points is printed on every row, so inflation is
  visible rather than buried in a score.

Selection order: contrast against the control; homogeneity of that contrast
across volatility regimes (this chooses the *scaling family*); stability across
neighbouring parameters; then usability.

Train only.
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

from nqlevels.eval.baseline import E1_FEATURES, fit_baseline                   # noqa: E402
from nqlevels.eval.labeling import build_paths                                 # noqa: E402
from nqlevels.eval.quality import (clean_rate, mean_attraction,                # noqa: E402
                                   measure_candidates, paired_contrast,
                                   reaction_ratio, touch_rate)
from nqlevels.eval.splits import assign, load_split_plan                        # noqa: E402
from nqlevels.eval.universe import (core_levels_by_session, density_report,     # noqa: E402
                                    make_controls, select_universe)
from nqlevels.io.store import layer_dir                                         # noqa: E402

CONFIGS = str(ROOT / "configs")
TICK = 0.25
pd.set_option("display.width", 220)


def hdr(s: str) -> None:
    print(f"\n{'='*96}\n{s}\n{'='*96}")


def width_families(c: pd.DataFrame):
    atr = c["atr_t"].to_numpy("float64")
    onr = c["on_range_t"].to_numpy("float64")
    rv = c["rv_on_t"].to_numpy("float64")
    f = {}
    for t in (2, 4, 6, 8, 12, 16, 24, 32):
        f[f"fixed_{t}t"] = np.full(len(c), float(t))
    for k in (0.010, 0.015, 0.020, 0.030, 0.045, 0.065, 0.090):
        f[f"atr_{k:.3f}"] = np.clip(k * atr, 1.0, 200.0)
    for k in (0.015, 0.025, 0.040, 0.060, 0.090):
        f[f"onrange_{k:.3f}"] = np.clip(k * onr, 1.0, 200.0)
    for k in (0.05, 0.10, 0.18, 0.30):
        f[f"rv_{k:.2f}"] = np.clip(k * rv, 1.0, 200.0)
    return f


def main() -> int:
    t0 = _time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--universe", default="core")
    a = ap.parse_args()

    d = layer_dir("daily", CONFIGS)
    ctx = pd.read_parquet(d / "context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    bk = assign(list(ctx["session"]), load_split_plan(CONFIGS))
    tr_days = set(bk["train"])

    raw = pd.read_parquet(d / "candidates_named.parquet")
    raw = raw[raw["session"].isin(tr_days)]
    cols = ["session", "atr14_t", "globex_range_pre", "globex_rv_pre", "atr_pctile"]
    raw = raw.merge(ctx[cols], on="session", how="left").rename(
        columns={"atr14_t": "atr_t", "globex_range_pre": "on_range_t",
                 "globex_rv_pre": "rv_on_t"})
    raw = raw[np.isfinite(raw["atr_t"]) & np.isfinite(raw["on_range_t"])
              & np.isfinite(raw["rv_on_t"])]

    hdr("0. Density check — is 'at a level' even a meaningful statement?")
    dr = density_report(raw, ctx)
    print(dr.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  A universe whose random-price gap is ~1 point is describing the price")
    print("  axis, not the market. Only 'core' survives this; the sweep uses it.")

    core_by = core_levels_by_session(raw)
    c = select_universe(raw, a.universe, max_dist_atr=1.5)
    c = c.drop_duplicates(subset=["session", "off_t"]).reset_index(drop=True)
    print(f"\nuniverse={a.universe}: {len(c):,} zones over {c['session'].nunique()} "
          f"sessions ({len(c)/c['session'].nunique():.1f}/session)")

    trctx = ctx[ctx["session"].isin(tr_days)].reset_index(drop=True)
    baseline = fit_baseline(trctx, features=list(E1_FEATURES))
    paths = build_paths(pd.read_parquet(d / "paths.parquet"))
    ctrl = make_controls(c, core_by, kind="shift", shift_atr=0.35, min_sep_t=12.0)
    print(f"controls: {100*ctrl['ctrl_ok'].mean():.1f}% drawn >=3 pts from any core level")
    ctrl = ctrl[ctrl["ctrl_ok"]].reset_index(drop=True)

    fams, fams_c = width_families(c), width_families(ctrl)

    hdr("1. Width sweep — core levels vs matched, level-free controls (TRAIN)")
    print(f"{'family':>14} {'w(pts)':>7} {'touch':>7} {'ctrl_t':>7} {'attract':>8} "
          f"{'c_attr':>8} {'clean':>7} {'ctrl_c':>7} {'dclean':>8} {'95% CI':>18} "
          f"{'p>0':>6} {'revRat':>7}")
    rows = []
    for name, w in fams.items():
        real = measure_candidates(c, paths, w, baseline, trctx, point_grid=(5, 10, 15, 20))
        con_f = measure_candidates(ctrl, paths, fams_c[name], baseline, trctx,
                                   point_grid=(5, 10, 15, 20))
        k = paired_contrast(real, con_f, clean_rate, n_boot=a.boot)
        ka = paired_contrast(real, con_f, mean_attraction, n_boot=a.boot)
        r = {"family": name, "w_pts": float(np.median(w) * TICK),
             "touch": touch_rate(real), "ctrl_touch": touch_rate(con_f),
             "attraction": mean_attraction(real), "ctrl_attr": mean_attraction(con_f),
             "d_att": ka["diff"], "d_att_p": 1 - ka["p_gt0"],
             "clean": clean_rate(real), "ctrl_clean": clean_rate(con_f),
             "d_clean": k["diff"], "lo": k["lo"], "hi": k["hi"], "p_gt0": k["p_gt0"],
             "rev_ratio": reaction_ratio(real), "n": len(real)}
        print(f"{name:>14} {r['w_pts']:7.2f} {r['touch']:7.3f} {r['ctrl_touch']:7.3f} "
              f"{r['attraction']:+8.4f} {r['ctrl_attr']:+8.4f} {r['clean']:7.3f} "
              f"{r['ctrl_clean']:7.3f} {r['d_clean']:+8.4f} "
              f"[{r['lo']:+.4f},{r['hi']:+.4f}] {r['p_gt0']:6.3f} "
              f"{r['rev_ratio']:7.3f}", flush=True)
        r["_frame"] = real
        rows.append(r)
    sweep = pd.DataFrame([{k2: v for k2, v in r.items() if k2 != "_frame"} for r in rows])

    hdr("2. Homogeneity across volatility regimes — chooses the SCALING family")
    print(f"{'family':>14} {'low':>7} {'mid':>7} {'high':>7} {'spread':>8} {'dclean':>8}")
    homo = []
    for r in rows:
        f = r["_frame"]
        f = f[np.isfinite(f["atr_pctile"])]
        if f.empty:
            continue
        terc = pd.qcut(f["atr_pctile"], 3, labels=["low", "mid", "high"])
        g = f.groupby(terc, observed=True).apply(clean_rate)
        if g.dropna().size < 3:
            continue
        sp = float(g.max() - g.min())
        homo.append({"family": r["family"], "low": float(g["low"]), "mid": float(g["mid"]),
                     "high": float(g["high"]), "spread": sp, "d_clean": r["d_clean"],
                     "w_pts": r["w_pts"]})
        print(f"{r['family']:>14} {g['low']:7.3f} {g['mid']:7.3f} {g['high']:7.3f} "
              f"{sp:8.3f} {r['d_clean']:+8.4f}")
    homo = pd.DataFrame(homo)

    hdr("3. Best per scaling type")
    sweep["kind"] = sweep["family"].str.split("_").str[0]
    for k, g in sweep.groupby("kind"):
        b = g.loc[g["d_clean"].idxmax()]
        h = homo[homo["family"] == b["family"]]
        sp = float(h["spread"].iloc[0]) if len(h) else np.nan
        print(f"  {k:>8}: {b['family']:>14} w={b['w_pts']:5.2f}pts  "
              f"dclean {b['d_clean']:+.4f} [{b['lo']:+.4f},{b['hi']:+.4f}] "
              f"p>0 {b['p_gt0']:.3f}  regime spread {sp:.3f}")

    out = {"density": dr.to_dict("records"), "sweep": sweep.to_dict("records"),
           "homogeneity": homo.to_dict("records"), "universe": a.universe,
           "n_sessions": int(c["session"].nunique()), "n_zones": int(len(c))}
    (ROOT / "outputs" / "e2_width.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote outputs/e2_width.json  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
