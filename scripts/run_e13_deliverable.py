#!/usr/bin/env python
"""E13 — what does the data actually support as a morning product?

Everything that would let us *select* strong zones has failed: named families,
volume-at-price, interaction history, order-book history, and learned rankers
all collapse to zero once distance is held fixed. So the honest question is no
longer "which zones are strong" but "what can be said truthfully each morning".

Three parts:

1. **Time structure.** Is reaction quality a function of *when* in the window
   price arrives, even if it is not a function of *where*? Cheap to check and a
   different axis from everything tested so far.
2. **The E1 destination product.** Calibrated touch probability is the one
   thing that survived. This measures what it would look like in use: pick the
   price levels at fixed probabilities and check the realised hit rates.
3. **Sample output.** A worked morning, so the claim being made is concrete.

Validation for reporting. Holdout sealed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.eval.baseline import E1_FEATURES, HORIZONS, fit_baseline        # noqa: E402
from nqlevels.eval.quality import add_distance_bands                          # noqa: E402
from nqlevels.eval.splits import assign, load_split_plan                       # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402

CONFIGS = str(ROOT / "configs")
TICK = 0.25
pd.set_option("display.width", 220)


def hdr(s):
    print(f"\n{'='*94}\n{s}\n{'='*94}")


def main() -> int:
    d = layer_dir("daily", CONFIGS)
    ctx = pd.read_parquet(d / "context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    bk = assign(list(ctx["session"]), load_split_plan(CONFIGS))
    tr_days, va_days = set(bk["train"]), set(bk["validation"])

    m = pd.read_parquet(d / "axis_measured.parquet")
    m = m[(m["dist_atr"] <= 1.2) & (m["side"] != "inside")]
    va = m[m["split"] == "validation"]

    # ------------------------------------------------------------------ 1
    hdr("1. Is reaction quality a function of WHEN price arrives?")
    t = va[(va["touched"] == True) & va["clean"].notna()].copy()      # noqa: E712
    t["arr"] = pd.cut(t["first_touch_min"], [0, 15, 30, 60, 90, 120, 150],
                      labels=["0-15", "15-30", "30-60", "60-90", "90-120", "120-150"])
    g = t.groupby("arr", observed=True).agg(
        n=("clean", "size"), clean=("clean", "mean"),
        rev30m_atr=("rev30m_atr", "median"), pen_atr=("pen_atr", "median"),
        slice60=("slice_through_60s", "mean"), inside=("secs_inside_share", "mean"))
    print(g.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n  NOTE: later arrivals have less forward window, which mechanically")
    print("  lowers `clean`. The columns that are NOT window-limited (penetration,")
    print("  slice-through, time inside) are the interpretable ones.")

    # ------------------------------------------------------------------ 2
    hdr("2. The E1 destination product — calibrated touch probability")
    trctx = ctx[ctx["session"].isin(tr_days)].reset_index(drop=True)
    vactx = ctx[ctx["session"].isin(va_days)].reset_index(drop=True)
    b = fit_baseline(trctx, features=list(E1_FEATURES))
    s = b.scale(vactx)
    print("  For each target probability, solve for the distance and check the")
    print("  realised hit rate on validation (368 sessions):\n")
    print(f"  {'target p':>9} {'horizon':>8} {'side':>6} {'med dist (pts)':>15} "
          f"{'realised':>9} {'n':>6}")
    rows = []
    zgrid = np.linspace(0, 4, 2001)
    for h in (30, 60, 150):
        for side in ("above", "below"):
            surv = b.surv[(side, h)]
            sv = surv(zgrid)
            ex = vactx[f"{'up' if side == 'above' else 'dn'}{h}_out"].to_numpy("float64")
            for p_target in (0.8, 0.65, 0.5, 0.35, 0.2):
                z = float(np.interp(-p_target, -sv, zgrid))
                dist = z * s
                hit = float(np.mean(ex >= dist))
                rows.append({"p": p_target, "h": h, "side": side,
                             "dist_pts": float(np.median(dist) * TICK),
                             "realised": hit, "n": int(len(dist))})
                print(f"  {p_target:9.2f} {h:8d} {side:>6} {np.median(dist)*TICK:15.1f} "
                      f"{hit:9.3f} {len(dist):6d}")
    R = pd.DataFrame(rows)
    err = (R["realised"] - R["p"]).abs()
    print(f"\n  mean |realised - target| = {err.mean():.4f}, max {err.max():.4f}")
    print("  (the down side runs hot by ~3-5 pp — the documented 2024-25 regime shift)")

    # ------------------------------------------------------------------ 3
    hdr("3. Worked example — what a morning would honestly say")
    sess = sorted(va_days)[-1]
    row = vactx[vactx["session"] == sess]
    if len(row):
        r = row.iloc[0]
        sc = b.scale(row)
        ref_px = float(r["ref_t"]) * TICK
        print(f"  {sess}  reference {ref_px:,.2f} (last trade before 06:25 PT)")
        print(f"  overnight range {float(r['globex_range_pre'])*TICK:.0f} pts | "
              f"ATR(14) {float(r['atr14_t'])*TICK:.0f} pts | "
              f"gap {float(r['gap_t_prior'])*TICK:+.0f} pts\n")
        print(f"  {'':>10} {'price':>12} {'dist':>7} {'P(touch by 09:00)':>18}")
        for side in ("above", "below"):
            for p_target in (0.75, 0.5, 0.25):
                z = float(np.interp(-p_target, -b.surv[(side, 150)](zgrid), zgrid))
                dd = z * float(sc[0]) * TICK
                px = ref_px + dd if side == "above" else ref_px - dd
                print(f"  {side:>10} {px:12,.2f} {dd:7.0f} {p_target:18.2f}")
        print("\n  That is the honest product: a distance-and-volatility curve.")
        print("  Nothing in Phase 2 supports labelling any of these as a 'strong level'.")

    (ROOT / "outputs" / "e13_deliverable.json").write_text(
        json.dumps({"calibration": R.to_dict("records"),
                    "arrival_time": g.reset_index().to_dict("records")},
                   indent=2, default=str))
    print("\nwrote outputs/e13_deliverable.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
