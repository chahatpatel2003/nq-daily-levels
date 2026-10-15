#!/usr/bin/env python
"""P3-205 — do 05:30 PT release structures mark strong zones on release mornings?

The one Phase-1 macro fact this builds on: the main US release slot is 55
minutes before the decision cut, so the release and the market's reaction to
it are fully observable pre-open with no feed and no leakage (P1-006).

Registered design (before outcomes are read):
  * release morning = macro_volx_pre >= 4 (30 % of sessions; Thu+Fri 64 %,
    Mon 3 % — the calendar fingerprint, measured from the tape alone);
  * structures from the release window [cut-55m, cut-25m) = 05:30-06:00 PT:
    spike high, spike low, midpoint, VWAP;
  * each structure marks grid prices within 6 ticks (same convention as
    near_core);
  * hypotheses: H1 structure prices beat matched non-structure prices on the
    same release mornings (clean, at side x distance strata); H2 the spike
    extremes are the strong kinds; H3 effects are energy-flavoured
    (attraction, rev30m, slice) even if cleanliness is flat.
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.clock import decision_time_utc, load_calendar                    # noqa: E402
from nqlevels.eval.quality import add_distance_bands, stratified_contrast      # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402

CONFIGS = str(ROOT / "configs")
NEAR_T = 6.0
VOLX_MIN = 4.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", action="store_true")
    ap.add_argument("--max-dist-atr", type=float, default=1.2)
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    t0 = _time.time()
    split = "validation" if a.validation else "train"

    cal = load_calendar(CONFIGS)
    d0 = layer_dir("daily", CONFIGS)
    ctx = pd.read_parquet(d0 / "context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    rel = ctx[ctx["macro_volx_pre"] >= VOLX_MIN]
    rel_sessions = set(rel["session"])
    ctxi = ctx.set_index("session")

    m = pd.read_parquet(d0 / "axis_measured.parquet")
    m = m[(m["split"] == split) & (m["dist_atr"] <= a.max_dist_atr)
          & (m["side"] != "inside") & m["p_base"].notna()]
    m = m[m["session"].isin(rel_sessions)].reset_index(drop=True)
    m = add_distance_bands(m, n=5)
    sessions = sorted(m["session"].unique())
    print(f"{split}: {len(m):,} grid rows on {len(sessions)} release mornings")

    bars = pd.read_parquet(d0 / "bars_1m.parquet")
    bars["ts"] = pd.DatetimeIndex(bars["ts"]).tz_convert("UTC")
    fb_by = {s: g for s, g in bars[bars["is_front"]].groupby("session", sort=False)
             if s in set(sessions)}

    # ---- release-window structures per morning -----------------------------
    rows = []
    for s in sessions:
        g = fb_by.get(s)
        if g is None:
            continue
        cut = pd.Timestamp(decision_time_utc(s, cal))
        w = g[(g["ts"] >= cut - timedelta(minutes=55))
              & (g["ts"] < cut - timedelta(minutes=25))]
        if len(w) < 15:
            continue
        ref = float(ctxi.loc[s, "ref_t"])
        hi, lo = float(w["high_t"].max()), float(w["low_t"].min())
        vwap = float((w["close_t"] * w["volume"]).sum() / max(1.0, w["volume"].sum()))
        for kind, px in (("rel_high", hi), ("rel_low", lo),
                         ("rel_mid", (hi + lo) / 2.0), ("rel_vwap", vwap)):
            rows.append({"session": s, "kind": kind, "px_off": px - ref,
                         "rel_range_t": hi - lo})
    st = pd.DataFrame(rows)
    print(f"structures: {len(st):,} over {st['session'].nunique()} mornings | "
          f"median spike range {st['rel_range_t'].median():.0f}t")

    # mark grid rows near each structure kind
    for kind, g in st.groupby("kind"):
        gg = g.set_index("session")["px_off"]
        off = m["session"].map(gg)
        m[f"at_{kind}"] = ((m["off_t"] - off).abs() <= NEAR_T).fillna(False).astype("int8")
    m["at_rel_ext"] = ((m["at_rel_high"] == 1) | (m["at_rel_low"] == 1)).astype("int8")
    m["at_rel_any"] = ((m[[f"at_rel_{k}" for k in ("high", "low", "mid", "vwap")]]
                        .sum(axis=1)) > 0).astype("int8")

    res = {"split": split, "volx_min": VOLX_MIN, "n_sessions": len(sessions),
           "contrasts": {}}
    print(f"\nstratified contrasts on release mornings (side x dband, 5 bands)")
    print(f"  {'flag':>12} {'stat':>11} {'diff':>8} {'lo':>8} {'hi':>8} {'p>0':>6} {'n_flag':>7}")
    for k in ("at_rel_high", "at_rel_low", "at_rel_ext", "at_rel_mid",
              "at_rel_vwap", "at_rel_any"):
        for stat in ("clean_rate", "attraction", "rev30m_atr", "slice_rate",
                     "touch_rate"):
            r = stratified_contrast(m, k, stat=stat, strata=("side", "_dband"),
                                    n_boot=a.boot, min_cell=20)
            res["contrasts"][f"{k}.{stat}"] = r
            if stat == "clean_rate" or abs(r.get("diff") or 0) > 0.006:
                print(f"  {k:>12} {stat:>11} {r['diff']:8.4f} {r['lo']:8.4f} "
                      f"{r['hi']:8.4f} {r['p_gt0']:6.3f} {r.get('n_flag', 0):7,}")

    # conditional: does a BIG spike make its extremes stronger?
    big = m["session"].map(st.groupby("session")["rel_range_t"].first()) \
        >= st["rel_range_t"].median()
    sub = m[m["at_rel_ext"] == 1].copy()
    sub["_f"] = big.loc[sub.index].fillna(False).astype("int8")
    r = stratified_contrast(sub, "_f", stat="clean_rate", strata=("side", "_dband"),
                            n_boot=a.boot, min_cell=15)
    res["contrasts"]["ext.big_vs_small"] = r
    print(f"  {'ext|bigspike':>12} {'clean_rate':>11} {r['diff']:8.4f} {r['lo']:8.4f} "
          f"{r['hi']:8.4f} {r['p_gt0']:6.3f} {r.get('n_flag', 0):7,}")

    out = ROOT / "outputs" / f"p3_macro_{split}.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
