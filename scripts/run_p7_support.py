#!/usr/bin/env python
"""P7 Part 2 — where do support reactions actually anchor?

No forced symmetry. Three attacks:
  1. The signed-offset outcome curve for SUPPORT events from the shadow
     corpus, with peak tests: anchor(δ) vs its own ±10 neighbours — the same
     construction that validated the resistance pivot (and whose placebos
     were negative). Interior peak ⇒ candidate anchor band.
  2. Backward mapping: for successful support reactions, the signed offset
     of the ACTUAL turn price from the pivot low, by mode; plus distances to
     mechanical references (nearest other active zone below, prior
     turn-price concentration, volatility-scaled sweep extension).
  3. Any candidate found on 2021-2023 is confirmed on 2024-2026 only.
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
from run_p6_spatial import add_labels                                          # noqa: E402

CONFIGS = str(ROOT / "configs")
KEY = ["session", "zid", "tap_sec"]


def peak_test(center: pd.Series, nb1: pd.Series, nb2: pd.Series,
              sessions: pd.Series, n_boot=300, seed=7):
    j = pd.concat([center.rename("c"), nb1.rename("a"), nb2.rename("b")],
                  axis=1, join="inner").dropna()
    if len(j) < 2000:
        return None
    d = j["c"] - 0.5 * (j["a"] + j["b"])
    sess = sessions.reindex(j.index).to_numpy()
    uniq = pd.unique(sess)
    idx = {u: np.flatnonzero(sess == u) for u in uniq}
    rng = np.random.default_rng(seed)
    v = np.empty(n_boot)
    da = d.to_numpy()
    for k in range(n_boot):
        p = rng.choice(uniq, uniq.size, replace=True)
        v[k] = np.nanmean(np.concatenate([da[idx[u]] for u in p]))
    return {"diff": float(np.nanmean(da)), "lo": float(np.quantile(v, .025)),
            "hi": float(np.quantile(v, .975)), "n": int(len(j))}


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    re = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    sh = add_labels(pd.read_parquet(d0 / "p6_shadow.parquet"))
    it = pd.read_parquet(d0 / "interactions.parquet",
                         columns=KEY + ["price_t", "zhi_t", "zlo_t"]).drop_duplicates(KEY)
    re = re.merge(it, on=KEY, how="left", validate="1:1")
    re = re[re["has_t0"] & ~re["inside_t0"] & (re["n_fwd"] >= 1800)
            & re["y"].notna()].copy()
    sup = re[re["side_above"] == 1].copy()          # approached from above
    shs = sh[sh["side_above"] == 1]
    sess_of = sup.assign(_s=sup["session"]).set_index(KEY)["_s"]
    res: dict = {}
    print(f"support events: {len(sup):,}")

    # ---- 1. signed-offset curves + peak tests -----------------------------
    print("\nsigned-offset outcome (support; negative = BELOW the pivot low)")
    vals = {}
    for col in ("y", "y_strict"):
        vals[col] = {0.0: sup.set_index(KEY)[col]}
        for off, g in shs[shs["elig"] & shs["touched"]
                          & (shs["n_fwd"] >= 1800)].groupby(shs["offset_t"] / 4.0):
            vals[col][off] = g.groupby(KEY)[col].mean()
    print(f"{'offset':>8} {'y':>8} {'y_strict':>9} {'n':>8}")
    for off in sorted(vals["y"], key=float):
        a = vals["y"][off]
        b = vals["y_strict"][off]
        print(f"{off:>+8.1f} {a.mean():8.4f} {b.mean():9.4f} {len(a):8,}")
        res.setdefault("curve", {})[str(off)] = {
            "y": float(a.mean()), "y_strict": float(b.mean()), "n": int(len(a))}

    print("\npeak tests (anchor vs its ±10 neighbours), support side:")
    for col in ("y", "y_strict"):
        for c, n1, n2, label in ((0.0, -10.0, 10.0, "pivot low"),
                                 (-10.0, -20.0, 0.0, "10 pts BELOW"),
                                 (10.0, 0.0, 20.0, "10 pts ABOVE"),
                                 (-5.0, -10.0, 0.0, "5 pts below*"),
                                 (5.0, 0.0, 10.0, "5 pts above*")):
            r = peak_test(vals[col][c], vals[col][n1], vals[col][n2], sess_of)
            if r:
                res.setdefault(f"peak_{col}", {})[label] = r
                print(f"  {col:>8} {label:>14}: {r['diff']:+.4f} "
                      f"[{r['lo']:+.4f},{r['hi']:+.4f}] (n={r['n']:,})")

    # ---- 2. backward mapping of actual turns ------------------------------
    S = sup[sup["y"] == 1].copy()
    S["turn_px"] = S["zhi_t"] - S["adverse_before_rev15_t"]
    S["turn_off_pts"] = (S["turn_px"] - S["price_t"]) / 4.0
    q = S["turn_off_pts"]
    print(f"\nACTUAL turn offset from pivot low (successful support, pts; "
          f"negative = below):")
    print(f"  mean {q.mean():+.1f} | p10 {q.quantile(.1):+.1f} p25 "
          f"{q.quantile(.25):+.1f} med {q.median():+.1f} p75 "
          f"{q.quantile(.75):+.1f} p90 {q.quantile(.9):+.1f} (n={len(S):,})")
    res["turn_off"] = {p: float(q.quantile(p / 100)) for p in (10, 25, 50, 75, 90)}
    from nqlevels.labels.sweep import classify
    c = classify(S, R=15, d0_t=4.0, D_t=S["atr_t"] * 0.10, T_s=1800, level="ne")
    S["mode"] = np.where(c["cat_B"] == 1, "sweep",
                         np.where(S["turn_frac_rev15"] <= 0.5, "front", "deep"))
    for md, g in S.groupby("mode"):
        print(f"    {md:>6}: med {g['turn_off_pts'].median():+.1f} "
              f"p25 {g['turn_off_pts'].quantile(.25):+.1f} "
              f"p75 {g['turn_off_pts'].quantile(.75):+.1f} (n={len(g):,}, "
              f"share {len(g)/len(S):.2f})")
    print(f"  turn in ATR units: med {(S['turn_off_pts']*4/S['atr_t']).median():+.4f}")

    # references: nearest OTHER active zone centre below the pivot
    zn = pd.read_parquet(d0 / "zones.parquet",
                         columns=["session", "zid", "price_rf", "form_ord"])
    adj = pd.read_parquet(d0 / "roll_adjust.parquet").set_index("session")["adj_t"]
    ctx = pd.read_parquet(d0 / "context.parquet", columns=["session"])
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    ord_of = {s: i for i, s in enumerate(sorted(ctx["session"]))}
    zn = zn.sort_values("form_ord")
    zn_ord = zn["form_ord"].to_numpy()
    dists = {"other_zone_below": [], "volscaled_ext": []}
    for s, g in S.groupby("session"):
        o = ord_of[s]
        a0, a1 = np.searchsorted(zn_ord, o - 30), np.searchsorted(zn_ord, o + 1)
        Z = zn.iloc[a0:a1]
        centers = np.sort(Z["price_rf"].to_numpy("float64") + float(adj.loc[s]))
        for r in g.itertuples():
            below = centers[(centers < r.price_t - 2) & (centers > r.price_t - 400)]
            if below.size:
                dists["other_zone_below"].append(
                    abs(float(r.turn_px) - below[np.argmin(np.abs(below - r.turn_px))]) / 4.0)
            dists["volscaled_ext"].append(
                abs(float(r.turn_px) - (r.price_t - 0.033 * r.atr_t)) / 4.0)
    print("\n|turn - reference| (pts): share within 5 pts / median")
    piv_d = (S["turn_off_pts"].abs())
    print(f"  {'pivot low itself':>20}: {float((piv_d<=5).mean()):.3f} / "
          f"{float(piv_d.median()):.1f}")
    for k, v in dists.items():
        v = pd.Series(v)
        print(f"  {k:>20}: {float((v<=5).mean()):.3f} / {float(v.median()):.1f} "
              f"(n={len(v):,})")
        res[f"ref_{k}"] = {"within5": float((v <= 5).mean()),
                          "med": float(v.median())}

    # ---- 3. era split of any candidate ------------------------------------
    print("\npeak tests split by era (candidate confirmation):")
    yr = pd.to_datetime(sess_of.astype(str)).dt.year
    for label, c, n1, n2 in (("pivot low", 0.0, -10.0, 10.0),
                             ("10 below", -10.0, -20.0, 0.0),
                             ("10 above", 10.0, 0.0, 20.0)):
        for era, mask in (("2021-23", yr <= 2023), ("2024-26", yr >= 2024)):
            ss = sess_of[mask.to_numpy()]
            r = peak_test(vals["y_strict"][c].reindex(ss.index).dropna(),
                          vals["y_strict"][n1], vals["y_strict"][n2], ss)
            if r:
                print(f"  {label:>10} {era}: strict {r['diff']:+.4f} "
                      f"[{r['lo']:+.4f},{r['hi']:+.4f}] (n={r['n']:,})")
                res.setdefault("era", {})[f"{label}_{era}"] = r

    out = ROOT / "outputs" / "p7_support.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
