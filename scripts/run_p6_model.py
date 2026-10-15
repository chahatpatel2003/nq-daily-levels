#!/usr/bin/env python
"""P6 Model A vs Model B — does exact-location spatial evidence add anything
beyond state + side + distance + zone history? (Question 10/11)

Spatial features are LOCAL DIFFERENCES computed strictly from prior-session
layers, roll-adjusted, at each real tap's price X:

  ev_n6 / ev_prom      decayed rejection-event density at X vs ±5-20 pt ring
  piv_n6 / piv_prom    pivot(zone-centre) density at X vs ring
  turn_n6 / turn_prom  prior successful TURN-price concentration vs ring
  vol_grad/curv/prom   trailing-5-session profile shape at X
  tapm_prom            trailing time-at-price prominence
  nest                 active centres within 1.5 pts

Model A = the Phase-4/5 T1 feature set. Model B = A + spatial. Walk-forward
(train < year, test year) with the frozen HGB hyper-parameters; permutation
of the spatial block on test data gives the incremental attribution.
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

from nqlevels.io.store import layer_dir, read_layer                            # noqa: E402
from run_p4_model import T1_FEATS                                              # noqa: E402
from run_p4_strongweak import load_frame                                       # noqa: E402
from run_p6_spatial import add_labels                                          # noqa: E402

CONFIGS = str(ROOT / "configs")
KEY = ["session", "zid", "tap_sec"]
LOOK = 30
HL = 10.0
NEAR_T = 6.0
RING_T = (20, 40, 60, 80)
SPATIAL = ["ev_n6", "ev_prom", "piv_n6", "piv_prom", "turn_n6", "turn_prom",
           "vol_grad", "vol_curv", "vol_prom", "tapm_prom", "nest"]


def density(sorted_px, wts, X, half):
    a = np.searchsorted(sorted_px, X - half)
    b = np.searchsorted(sorted_px, X + half)
    return float(wts[a:b].sum()) if wts is not None else float(b - a)


def prom(sorted_px, wts, X, half=NEAR_T):
    n0 = density(sorted_px, wts, X, half)
    ring = np.mean([0.5 * (density(sorted_px, wts, X - r, half)
                           + density(sorted_px, wts, X + r, half))
                    for r in RING_T])
    return n0, float(n0 - ring)


def build_spatial() -> pd.DataFrame:
    d0 = layer_dir("daily", CONFIGS)
    re = pd.read_parquet(d0 / "p6_real.parquet", columns=KEY + ["side"])
    it = pd.read_parquet(d0 / "interactions.parquet",
                         columns=KEY + ["price_t", "zhi_t", "zlo_t"])
    re = re.merge(it, on=KEY, how="left", validate="1:1")
    enr = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    enr = enr[KEY + ["y", "adverse_before_rev15_t", "side"]].rename(
        columns={"side": "side_e"})
    ctx = pd.read_parquet(d0 / "context.parquet", columns=["session", "atr14_t"])
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    adj = pd.read_parquet(d0 / "roll_adjust.parquet").set_index("session")
    ord_of = {s: i for i, s in enumerate(sorted(ctx["session"]))}

    ev = pd.read_parquet(d0 / "events_full.parquet",
                         columns=["session", "price_t", "mag_atr"])
    ev["ord"] = ev["session"].map(ord_of)
    ev["adj"] = ev["session"].map(adj["adj_t"])
    ev = ev.dropna(subset=["ord"]).sort_values("ord")
    zn = pd.read_parquet(d0 / "zones.parquet",
                         columns=["session", "price_rf", "form_ord"]).sort_values("form_ord")
    # prior successful turn prices, roll-free
    tt = re.merge(enr, on=KEY, how="left")
    tt = tt[(tt["y"] == 1) & tt["adverse_before_rev15_t"].notna()].copy()
    sgn = np.where(tt["side_e"] == "below", -1.0, 1.0)
    near = np.where(tt["side_e"] == "below", tt["zhi_t"], tt["zlo_t"])
    tt["turn_px"] = near + sgn * tt["adverse_before_rev15_t"]
    tt["ord"] = tt["session"].map(ord_of)
    tt["adj"] = tt["session"].map(adj["adj_t"])
    tt["turn_rf"] = tt["turn_px"] - tt["adj"]
    tt = tt.sort_values("ord")

    prof = read_layer("profiles", CONFIGS)
    prof = prof[prof["is_front"]]
    prof = prof.groupby(["session", "price_t"], as_index=False)["volume"].sum()
    prof_by = {s: g for s, g in prof.groupby("session", sort=False)}
    del prof
    b1m = pd.read_parquet(d0 / "bars_1m.parquet",
                          columns=["session", "high_t", "low_t", "is_front"])
    b1m = b1m[b1m["is_front"]]
    b1m_by = {s: g for s, g in b1m.groupby("session", sort=False)}
    del b1m

    rows = []
    prof_dq, bars_dq = [], []
    ev_ord = ev["ord"].to_numpy()
    zn_ord = zn["form_ord"].to_numpy()
    tt_ord = tt["ord"].to_numpy()
    re_by = {s: g for s, g in re.groupby("session", sort=False)}
    for s in sorted(ctx["session"]):
        if s not in adj.index:
            continue
        o = ord_of[s]
        adj_s = float(adj.loc[s, "adj_t"])
        g = re_by.get(s)
        if g is not None and len(g):
            def win(tab, ocol, lo=o - LOOK, hi=o):
                a = np.searchsorted(ocol, lo)
                b = np.searchsorted(ocol, hi)
                return tab.iloc[a:b]
            E = win(ev, ev_ord)
            e_px = np.sort((E["price_t"] - E["adj"]).to_numpy("float64") + adj_s) \
                if len(E) else np.array([])
            e_w = None
            Z = win(zn, zn_ord)
            z_px = np.sort(Z["price_rf"].to_numpy("float64") + adj_s) if len(Z) else np.array([])
            T = win(tt, tt_ord)
            t_px = np.sort(T["turn_rf"].to_numpy("float64") + adj_s) if len(T) else np.array([])
            if prof_dq:
                px = np.concatenate([p for p, _ in prof_dq])
                vv = np.concatenate([v for _, v in prof_dq])
                so = np.argsort(px)
                px, vv = px[so] + adj_s, vv[so]
                cum = np.concatenate(([0.0], np.cumsum(vv)))
                def vol_at(X, half=10.0):
                    a = np.searchsorted(px, X - half)
                    b = np.searchsorted(px, X + half)
                    return float(cum[b] - cum[a])
            else:
                vol_at = None
            bar_h = np.concatenate([b["high_t"].to_numpy("float64") for b in bars_dq]) + 0 \
                if bars_dq else None
            bar_l = np.concatenate([b["low_t"].to_numpy("float64") for b in bars_dq]) \
                if bars_dq else None

            for r in g.itertuples():
                X = float(r.price_t)
                f = dict(session=s, zid=int(r.zid), tap_sec=int(r.tap_sec))
                f["ev_n6"], f["ev_prom"] = prom(e_px, e_w, X) if e_px.size else (0, 0)
                f["piv_n6"], f["piv_prom"] = prom(z_px, None, X) if z_px.size else (0, 0)
                f["turn_n6"], f["turn_prom"] = prom(t_px, None, X) if t_px.size else (0, 0)
                if vol_at is not None:
                    v0 = vol_at(X)
                    vp = vol_at(X + 40)
                    vm = vol_at(X - 40)
                    ring = np.mean([vol_at(X - 80), vol_at(X + 80),
                                    vol_at(X - 120), vol_at(X + 120)])
                    f["vol_grad"] = (vp - vm) / (abs(v0) + 1)
                    f["vol_curv"] = (vp + vm - 2 * v0) / (abs(v0) + 1)
                    f["vol_prom"] = v0 / (ring + 1)
                else:
                    f["vol_grad"] = f["vol_curv"] = f["vol_prom"] = np.nan
                if bar_h is not None:
                    def mins_at(Xq):
                        return float(((bar_l <= Xq) & (bar_h >= Xq)).sum())
                    m0 = mins_at(X)
                    ring = np.mean([mins_at(X - 80), mins_at(X + 80)])
                    f["tapm_prom"] = m0 / (ring + 1)
                else:
                    f["tapm_prom"] = np.nan
                f["nest"] = f["piv_n6"]
                rows.append(f)
        p = prof_by.get(s)
        if p is not None and len(p):
            prof_dq.append(((p["price_t"].to_numpy("float64") - adj_s),
                            p["volume"].to_numpy("float64")))
            if len(prof_dq) > 5:
                prof_dq.pop(0)
        bb = b1m_by.get(s)
        if bb is not None:
            bars_dq.append(pd.DataFrame(
                {"high_t": bb["high_t"].to_numpy("float64") - adj_s,
                 "low_t": bb["low_t"].to_numpy("float64") - adj_s}))
            if len(bars_dq) > 5:
                bars_dq.pop(0)
    return pd.DataFrame(rows)


def auc(y, s):
    y = np.asarray(y, "float64")
    s = np.asarray(s, "float64")
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size < 200 or y.min() == y.max():
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    sp_path = d0 / "p6_spatial_feats.parquet"
    if sp_path.exists():
        sp = pd.read_parquet(sp_path)
    else:
        print("building spatial features ...")
        sp = build_spatial()
        sp.to_parquet(sp_path, index=False, compression="zstd")
    print(f"spatial features: {len(sp):,} rows")

    f = load_frame()
    f = f[f["y"].notna()]
    f = f.merge(sp, on=KEY, how="inner", validate="1:1")
    enr = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    f = f.merge(enr[KEY + ["rev_ep_pts"]], on=KEY, how="left")
    print(f"modeling frame: {len(f):,} taps, base y {float(f['y'].mean()):.4f}")

    from sklearn.ensemble import HistGradientBoostingClassifier
    res = {"by_year": {}}
    print(f"\n{'year':>6} {'base':>6} {'A_auc':>8} {'B_auc':>8} {'B-A':>8} "
          f"{'spatial_only':>13} {'rhoA':>6} {'rhoB':>6}")
    for Y in (2022, 2023, 2024, 2025, 2026):
        tr = f[f["year"] < Y]
        te = f[f["year"] == Y]
        if len(tr) < 8000 or len(te) < 2000:
            continue
        outs = {}
        for name, feats in (("A", T1_FEATS), ("B", T1_FEATS + SPATIAL),
                            ("S", SPATIAL)):
            g = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=4,
                l2_regularization=1.0, random_state=0)
            g.fit(tr[feats].to_numpy("float64"), tr["y"].to_numpy("float64"))
            outs[name] = g.predict_proba(te[feats].to_numpy("float64"))[:, 1]
        aA, aB, aS = (auc(te["y"], outs[k]) for k in ("A", "B", "S"))
        S = te[te["y"] == 1]
        rhoA = float(pd.Series(outs["A"][te["y"].to_numpy() == 1]).corr(
            S["rev_ep_pts"].reset_index(drop=True), method="spearman"))
        rhoB = float(pd.Series(outs["B"][te["y"].to_numpy() == 1]).corr(
            S["rev_ep_pts"].reset_index(drop=True), method="spearman"))
        res["by_year"][int(Y)] = {"base": float(te["y"].mean()), "A": aA,
                                  "B": aB, "S": aS, "rhoA": rhoA, "rhoB": rhoB,
                                  "n": len(te)}
        print(f"{Y:>6} {te['y'].mean():6.3f} {aA:8.4f} {aB:8.4f} "
              f"{aB-aA:+8.4f} {aS:13.4f} {rhoA:6.3f} {rhoB:6.3f}")

    # univariate spatial screen within (seg x side), quick
    from nqlevels.eval.quality import STRAT_STATS, stratified_contrast
    STRAT_STATS["y"] = ("y", None)
    print("\nspatial univariate (top-vs-bottom tercile on y, seg x side strata):")
    f["above"] = f["app_side_above"]
    res["screen"] = {}
    for c in SPATIAL:
        v = f[c]
        ok = v.notna()
        if ok.sum() < 20000:
            continue
        q1, q2 = v[ok].quantile([1 / 3, 2 / 3])
        if q1 == q2:
            sub = f[ok & ((v <= q1) | (v > q1))].copy()
            sub["_f"] = (sub[c] > q1).astype("int8")
        else:
            sub = f[ok & ((v <= q1) | (v >= q2))].copy()
            sub["_f"] = (sub[c] >= q2).astype("int8")
        r = stratified_contrast(sub, "_f", stat="y", strata=("seg", "above"),
                                n_boot=200, min_cell=30)
        res["screen"][c] = r
        print(f"  {c:>10}: {r['diff']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]")

    out = ROOT / "outputs" / "p6_model.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
