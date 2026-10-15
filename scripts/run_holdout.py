#!/usr/bin/env python
"""P3-209 — THE one-shot 2026 sealed-holdout test of the frozen sweep-aware
engine. Pre-registered as holdout_sweep_engine_2026 (token 16ce7f7051547785).

Order of operations, enforced in code:
  0. REGRESSION CHECK (no holdout data): the walk-forward P(clean|touch)
     reimplementation must reproduce the stored validation scores from
     scored_axis.parquet. Abort on mismatch — the engine run on holdout must
     be bit-compatible with the engine that was validated.
  1. Unseal via eval.splits.holdout_sessions with the pre-registration token.
  2. Measure the holdout grid with the exact build_axis_measured path
     (half-width 6t, train-fitted E1 baseline, same clean specs).
  3. Sweep-walk holdout touched zones (labels/sweep, unchanged).
  4. Score: p_base x monthly-expanding walk-forward p_clean (history =
     everything strictly before each 2026 month).
  5. Frozen taxonomy at the three registered cells.
  6. Registered evaluation (topk.evaluate, seed 23, n_draws 8, n_boot 500)
     and the PASS/FAIL gate, verbatim from RESEARCH_PLAN §13.5.
  7. Practical, calibration, artifact-check and stability blocks.

Nothing is tuned, nothing is refit beyond the pre-registered monthly
expanding procedure, and nothing may be rerun with different parameters:
2026 is spent when this completes.
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

from nqlevels.eval import topk                                                 # noqa: E402
from nqlevels.eval.baseline import E1_FEATURES, fit_baseline                   # noqa: E402
from nqlevels.eval.labeling import build_paths                                 # noqa: E402
from nqlevels.eval.quality import measure_candidates                           # noqa: E402
from nqlevels.eval.splits import assign, holdout_sessions, load_split_plan     # noqa: E402
from nqlevels.io.raw import assert_not_raw                                     # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402
from nqlevels.labels.sweep import classify, sweep_metrics                      # noqa: E402
from run_p3_product import CTX_FEATS, design                                   # noqa: E402
from build_axis_measured import CTX_COLS                                       # noqa: E402

CONFIGS = str(ROOT / "configs")
TOKEN = "16ce7f7051547785"
TICKS = 4
CELLS = {"strict": ("24t", 900), "mid": ("5%atr", 900), "primary": ("10%atr", 1800)}
POLICY = dict(min_sep_t=24.0, dist_window=(0.15, 1.0))
SEED, N_DRAWS, N_BOOT = 23, 8, 500


def wf_pclean(pool: pd.DataFrame, split_name: str) -> np.ndarray:
    """Monthly-expanding walk-forward P(clean|touch) — mirrors
    run_p3_product.walkforward_pclean's out-of-sample branch exactly."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    t = pool[(pool["touched"] == True) & pool["clean"].notna()]      # noqa: E712
    Xt = t[CTX_FEATS].to_numpy("float64")
    yt = t["clean"].to_numpy("float64")
    mon_t = pd.to_datetime(t["session"].astype(str)).dt.to_period("M")
    mon = pd.to_datetime(pool["session"].astype(str)).dt.to_period("M")
    p = np.full(len(pool), np.nan)
    for mth in sorted(mon[pool["split"] == split_name].unique()):
        hist = (mon_t < mth).to_numpy()
        if hist.sum() < 5000:
            continue
        sc = StandardScaler().fit(Xt[hist])
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(sc.transform(Xt[hist]), yt[hist])
        sel = ((mon == mth) & (pool["split"] == split_name)).to_numpy()
        p[sel] = lr.predict_proba(
            sc.transform(pool.loc[sel, CTX_FEATS].to_numpy("float64")))[:, 1]
    return p


def frame_filter(d: pd.DataFrame) -> pd.DataFrame:
    return d[(d["side"] != "inside") & d["p_base"].notna()
             & (d["dist_atr"] <= 1.5)].reset_index(drop=True)


def ece(y, p, bins=10):
    t = topk.reliability(np.asarray(y, "float64"), np.asarray(p, "float64"), bins)
    w = t["n"] / t["n"].sum()
    return float((w * (t["p_mean"] - t["y_rate"]).abs()).sum()), t


def brier(y, p):
    y = np.asarray(y, "float64")
    p = np.asarray(p, "float64")
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    b = float(np.mean((p - y) ** 2))
    ref = float(np.mean((y.mean() - y) ** 2))
    return b, (1.0 - b / ref if ref > 0 else np.nan), int(y.size)


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)

    # ---- 0. regression check: reproduce the validated engine ---------------
    print("[0] regression check: reproducing stored validation scores ...")
    am = pd.read_parquet(d0 / "axis_measured.parquet")
    tv = frame_filter(am)
    tv = design(tv)
    p_val = wf_pclean(tv, "validation")
    stored = pd.read_parquet(d0 / "scored_axis.parquet",
                             columns=["session", "off_t", "p_clean_A", "score_A"])
    chk = tv[["session", "off_t", "p_base", "split"]].copy()
    chk["p_new"] = p_val
    chk = chk[chk["split"] == "validation"].merge(stored, on=["session", "off_t"],
                                                 validate="1:1")
    diff = float((chk["p_new"] - chk["p_clean_A"]).abs().max())
    print(f"    validation rows compared: {len(chk):,} | max |p_clean diff| = {diff:.2e}")
    if not (diff < 1e-4):
        raise SystemExit("REGRESSION CHECK FAILED — engine reimplementation "
                         "does not match the validated engine. Holdout NOT opened.")

    # ---- 1. unseal ---------------------------------------------------------
    ctx = pd.read_parquet(d0 / "context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    plan = load_split_plan(CONFIGS)
    bk = assign(list(ctx["session"]), plan)
    unsealed = holdout_sessions(list(ctx["session"]), plan, ROOT, token=TOKEN)
    hold = [s for s in bk["holdout"]]
    assert set(hold) <= set(unsealed)
    print(f"[1] HOLDOUT UNSEALED with token {TOKEN}: {len(hold)} post-embargo "
          f"sessions {hold[0]} .. {hold[-1]}")

    # ---- 2. measure the holdout grid (frozen build path) -------------------
    print("[2] measuring holdout grid zones (half-width 6t, frozen E1 baseline) ...")
    ax = pd.read_parquet(d0 / "axis.parquet")
    ax = ax[ax["session"].isin(hold)]
    have = [c for c in CTX_COLS if c in ctx.columns]
    ax = ax.merge(ctx[have], on="session", how="left").rename(columns={"atr14_t": "atr_t"})
    ax = ax[np.isfinite(ax["atr_t"])].reset_index(drop=True)
    trctx = ctx[ctx["session"].isin(bk["train"])].reset_index(drop=True)
    baseline = fit_baseline(trctx, features=list(E1_FEATURES))
    paths = build_paths(pd.read_parquet(d0 / "paths.parquet"))
    hctx = ctx[ctx["session"].isin(hold)].reset_index(drop=True)
    w = np.full(len(ax), 6.0)
    mh = measure_candidates(ax, paths, w, baseline, hctx,
                            point_grid=(5, 10, 15, 20, 30),
                            clean_specs=((10, 5), (15, 10), (20, 10), (30, 15)))
    mh["split"] = "holdout"
    dst = assert_not_raw(d0 / "axis_measured_holdout.parquet")
    mh.to_parquet(dst, index=False, compression="zstd")
    print(f"    measured {len(mh):,} zones over {mh['session'].nunique()} sessions")

    # ---- 3. sweep-walk holdout touched zones -------------------------------
    print("[3] sweep-walking touched holdout zones ...")
    tz = mh[(mh["touched"] == True) & (mh["side"] != "inside")]      # noqa: E712
    rows = []
    for s, g in tz.groupby("session", sort=True):
        p = paths.get(s)
        if p is None:
            continue
        for zlo, zhi, off in zip(g["zlo_t"].to_numpy("float64"),
                                 g["zhi_t"].to_numpy("float64"),
                                 g["off_t"].to_numpy("float64")):
            i = p.first_touch(zlo, zhi)
            if i is None:
                continue
            r = sweep_metrics(p, zlo, zhi, i)
            r["off_t"] = off
            r["session"] = s
            rows.append(r)
    sw = pd.DataFrame(rows)
    for c in sw.columns:
        if c not in ("session", "side"):
            sw[c] = pd.to_numeric(sw[c], errors="coerce").astype("float32")
    dst = assert_not_raw(d0 / "sweep_labels_holdout.parquet")
    sw.to_parquet(dst, index=False, compression="zstd")
    print(f"    walked {len(sw):,} touched zones")

    # ---- 4. score with the frozen engine -----------------------------------
    print("[4] scoring: p_base x walk-forward p_clean (expanding monthly) ...")
    hf = frame_filter(mh)
    hf = design(hf)
    pool = pd.concat([tv, hf], ignore_index=True, sort=False)
    p_all = wf_pclean(pool, "holdout")
    hf = pool[pool["split"] == "holdout"].copy()
    hf["p_clean_A"] = p_all[(pool["split"] == "holdout").to_numpy()]
    hf["score_A"] = hf["p_base"] * hf["p_clean_A"]
    hf["above"] = (hf["off_t"] > 0).astype("int8")
    hf = hf.merge(sw.drop(columns=["side"]), on=["session", "off_t"], how="left",
                  validate="1:1")
    for name, (D, T) in CELLS.items():
        Dv = (hf["atr_t"] * (0.05 if D == "5%atr" else 0.10)) if "atr" in D else 24.0
        c = classify(hf, R=15, d0_t=4.0, D_t=Dv, T_s=T, level="ne")
        for k in ("y_any", "cat_A", "cat_B", "cat_C"):
            hf[f"{k}_{name}"] = c[k]

    # ---- 5/6. registered evaluation + gate ---------------------------------
    print("[5] registered evaluation ...\n")
    res = {"token": TOKEN, "sessions": len(hf["session"].unique()),
           "policy": POLICY, "cells": {}}
    ev = {}
    for name in CELLS:
        ev[name] = topk.evaluate(hf, "score_A", ks=(1, 2, 3, 4),
                                 outcome=f"y_any_{name}",
                                 min_sep_t=POLICY["min_sep_t"],
                                 dist_window=POLICY["dist_window"],
                                 n_draws=N_DRAWS, n_boot=N_BOOT, seed=SEED,
                                 by_year=False)
        res["cells"][name] = ev[name]

    print("=" * 92)
    print("REGISTERED RESULT — 2026 HOLDOUT (one shot)")
    print("=" * 92)
    for name in CELLS:
        print(f"\n--- cell {name} ---")
        print(topk.format_report(ev[name]))

    c2 = ev["strict"]["K2"]["ci_cond"]
    c4 = ev["strict"]["K4"]["ci_cond"]
    m2 = ev["mid"]["K2"]["ci_cond"]
    gate = {"strict_K2_cond": c2, "strict_K4_cond": c4, "mid_K2_cond": m2,
            "cond_positive_K2": bool(c2["diff"] > 0),
            "cond_positive_K4": bool(c4["diff"] > 0),
            "ci_lo_gt0_at_least_one": bool(c2["lo"] > 0 or c4["lo"] > 0),
            "mid_K2_same_sign": bool(m2["diff"] > 0)}
    gate["PASS"] = (gate["cond_positive_K2"] and gate["cond_positive_K4"]
                    and gate["ci_lo_gt0_at_least_one"] and gate["mid_K2_same_sign"])
    res["gate"] = gate
    print("\n" + "=" * 92)
    print(f"GATE (verbatim §13.5): strict cond K=2 {c2['diff']:+.4f} "
          f"[{c2['lo']:+.4f},{c2['hi']:+.4f}] | K=4 {c4['diff']:+.4f} "
          f"[{c4['lo']:+.4f},{c4['hi']:+.4f}] | mid K=2 {m2['diff']:+.4f}")
    print(f"VERDICT: {'PASS' if gate['PASS'] else 'FAIL'}")
    print("=" * 92)

    # ---- 7. practical / calibration / artifact / stability -----------------
    print("\n[7] practical blocks (per cell, K=2/3/4)")
    prac = {}
    for K in (2, 3, 4):
        picks = []
        for s, g in hf.groupby("session"):
            picks.append(topk.select_topk(g, "score_A", K, POLICY["min_sep_t"],
                                          POLICY["dist_window"]))
        P = pd.concat(picks, ignore_index=True)
        for name in CELLS:
            T = P[(P["touched"] == True) & P[f"y_any_{name}"].notna()]   # noqa: E712
            B = T[T[f"cat_B_{name}"] == 1]
            a = ev[name][f"K{K}"]
            row = {"mornings": int(P["session"].nunique()),
                   "zones_per_morning": float(P.groupby("session").size().mean()),
                   "total_zones": int(len(P)),
                   "pct_reached": float(P["touched"].mean()),
                   "direct|touch": float(T[f"cat_A_{name}"].mean()),
                   "sweep|touch": float(T[f"cat_B_{name}"].mean()),
                   "combined|touch": float(T[f"y_any_{name}"].mean()),
                   "fail|touch": float(T[f"cat_C_{name}"].mean()),
                   "matched_combined": a["matched"]["cond"],
                   "lift_pp": a["ci_cond"]["diff"],
                   "ci": [a["ci_cond"]["lo"], a["ci_cond"]["hi"]],
                   "p_ge1_react": a["model"]["p_ge1_react"],
                   "p_ge2_react": a["model"]["p_ge2_react"],
                   "med_sweep_depth_pts_B": float((B["depth_before_ne"] / TICKS).median()) if len(B) else np.nan,
                   "med_reclaim_min_B": float(((B["t_reclaim_ne"] - B["first_beyond"]) / 60).median()) if len(B) else np.nan,
                   "med_after_reclaim_pts_B": float((B["rev_after_ne_1800"] / TICKS).median()) if len(B) else np.nan}
            prac[f"K{K}_{name}"] = row
            print(f"  K={K} {name:>7}: reached {row['pct_reached']:.3f} | "
                  f"A {row['direct|touch']:.3f} B {row['sweep|touch']:.3f} "
                  f"any {row['combined|touch']:.3f} fail {row['fail|touch']:.3f} | "
                  f"matched {row['matched_combined']:.3f} lift {row['lift_pp']:+.4f} "
                  f"[{row['ci'][0]:+.3f},{row['ci'][1]:+.3f}] | "
                  f"P>=1 {row['p_ge1_react']:.3f} P>=2 {row['p_ge2_react']:.3f}")
    res["practical"] = prac

    print("\n[7b] calibration (frozen annotations, measured only)")
    t = hf[(hf["touched"] == True) & hf["clean"].notna()]            # noqa: E712
    e1, tab1 = ece(t["clean"], t["p_clean_A"])
    b1, s1, n1 = brier(t["clean"], t["p_clean_A"])
    yj = ((hf["touched"] == True)
          & (pd.to_numeric(hf["clean"], errors="coerce").fillna(0) == 1)).astype(float)  # noqa: E712
    e2, _ = ece(yj, hf["score_A"])
    b2, s2, n2 = brier(yj, hf["score_A"])
    e3, _ = ece(hf["touched"].astype(float), hf["p_base"])
    b3, s3, n3 = brier(hf["touched"].astype(float), hf["p_base"])
    cal = {"cond_ece": e1, "cond_brier": b1, "cond_brier_skill_vs_const": s1,
           "cond_n": n1, "cond_table": tab1.to_dict("records"),
           "joint_ece": e2, "joint_brier": b2, "joint_n": n2,
           "ptouch_ece": e3, "ptouch_brier": b3,
           "ptouch_brier_skill_vs_const": s3, "ptouch_n": n3,
           "pred_vs_real_cond": [float(t["p_clean_A"].mean()), float(t["clean"].mean())]}
    res["calibration"] = cal
    print(f"  P(clean|touch): ECE {e1:.4f} | Brier {b1:.4f} (skill vs const "
          f"{s1:+.3f}) | pred {cal['pred_vs_real_cond'][0]:.4f} vs real "
          f"{cal['pred_vs_real_cond'][1]:.4f} (n={n1:,})")
    print(f"  joint P(touch&clean): ECE {e2:.4f} | Brier {b2:.4f}")
    print(f"  P(touch): ECE {e3:.4f} | Brier {b3:.4f} (skill vs const {s3:+.3f})")
    print(tab1.round(4).to_string(index=False))

    print("\n[7c] artifact checks (diagnostic, K=4 strict)")
    hb = topk.add_bands(hf, n=10)
    rng = np.random.default_rng(SEED)
    mp, cp = [], []
    for s, g in hb.groupby("session"):
        p = topk.select_topk(g, "score_A", 4, POLICY["min_sep_t"], POLICY["dist_window"])
        if p.empty:
            continue
        mp.append(p)
        cp.append(topk.matched_controls(g, p, rng, n_draws=N_DRAWS,
                                        min_sep_t=POLICY["min_sep_t"]))
    M, C = pd.concat(mp, ignore_index=True), pd.concat(cp, ignore_index=True)

    def cond_y(df, tmax=None):
        x = df[(df["touched"] == True) & df["y_any_strict"].notna()]  # noqa: E712
        if tmax is not None:
            x = x[x["first_touch_min"] <= tmax]
        return float(x["y_any_strict"].mean()), len(x)

    art = {}
    for tmax, lab in ((None, "all"), (60, "<=07:30"), (30, "<=07:00")):
        a_, na = cond_y(M, tmax)
        b_, nb = cond_y(C, tmax)
        art[f"window_{lab}"] = {"model": a_, "matched": b_, "diff": a_ - b_,
                                "n_model": na, "n_matched": nb}
        print(f"  window {lab:>8}: model {a_:.4f} (n={na}) vs matched {b_:.4f} "
              f"(n={nb})  diff {a_-b_:+.4f}")
    tm = M[M["touched"] == True]                                      # noqa: E712
    tc = C[C["touched"] == True]                                      # noqa: E712
    art["touch_time_med_min"] = {"model": float(tm["first_touch_min"].median()),
                                 "matched": float(tc["first_touch_min"].median())}
    art["med_dist"] = {"model": float(M["dist_atr"].median()),
                       "matched": float(C["dist_atr"].median())}
    art["touch_ci_K4"] = ev["strict"]["K4"]["ci_touch"]
    print(f"  med first-touch: model {art['touch_time_med_min']['model']:.1f} vs "
          f"matched {art['touch_time_med_min']['matched']:.1f} min | med dist "
          f"{art['med_dist']['model']:.3f} vs {art['med_dist']['matched']:.3f} ATR")
    tt = ev["strict"]["K4"]["ci_touch"]
    print(f"  touch contrast K=4: {tt['diff']:+.4f} [{tt['lo']:+.4f},{tt['hi']:+.4f}]")

    # monthly composition (stability inside 2026)
    print("\n[7d] monthly (2026) strict-cell cond, model vs matched, K=4")
    Mm = M.copy()
    Mm["_m"] = pd.to_datetime(Mm["session"].astype(str)).dt.month
    Cm = C.copy()
    Cm["_m"] = pd.to_datetime(Cm["session"].astype(str)).dt.month
    mon_rows = {}
    for mth in sorted(Mm["_m"].unique()):
        a_, na = cond_y(Mm[Mm["_m"] == mth])
        b_, nb = cond_y(Cm[Cm["_m"] == mth])
        mon_rows[int(mth)] = {"model": a_, "matched": b_, "n": na}
        print(f"    2026-{mth:02d}: {a_:.3f} vs {b_:.3f} (n={na})")
    art["monthly"] = mon_rows
    res["artifacts"] = art

    out = ROOT / "outputs" / "holdout_result.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    print("\n2026 HOLDOUT OPENED — MAY NEVER BE USED AS A SEALED HOLDOUT AGAIN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
