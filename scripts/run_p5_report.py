#!/usr/bin/env python
"""P5 report — everything the live simulation implies, in one pass.

Units: a display EPISODE = a zone's continuous run on display (rank 1-4)
within one session. Touch lookup joins the interaction ledger on integer
within-session seconds. Two touch notions, both reported:
  touched-while-displayed  — tap during the episode (display metric)
  touched-same-session     — next tap after episode start, same session
                             (the uniform forward window used for arm
                             comparisons, applied identically to controls)

Controls are drawn at the episode's first refresh from the SAME scored pool,
rank 0 only, using information available at that moment: side+distance
matched, next-ranked, random eligible, nearest.
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
from run_p4_strongweak import SWEEP_COLS                                       # noqa: E402

CONFIGS = str(ROOT / "configs")
GAP_S = 1200          # refresh-run break
GRACE_S = 900         # tap shortly after last displayed refresh still counts


def tap_table():
    d0 = layer_dir("daily", CONFIGS)
    it = pd.read_parquet(d0 / "interactions.parquet",
                         columns=["session", "zid", "tap_sec", "tap_ts",
                                  "atr_t", "app_side_above", "seg", "dir",
                                  "scale"] + [f"o_{c}" for c in SWEEP_COLS])
    sw = it[[f"o_{c}" for c in SWEEP_COLS]].rename(
        columns={f"o_{c}": c for c in SWEEP_COLS})
    c = classify(sw, R=15, d0_t=4.0, D_t=it["atr_t"] * 0.10, T_s=1800, level="ne")
    it["y"] = c["y_any"]
    it["catA"] = c["cat_A"]
    it["catB"] = c["cat_B"]
    cs = classify(sw, R=15, d0_t=4.0, D_t=24.0, T_s=900, level="ne")
    it["y_strict"] = cs["y_any"]
    return it


def build_episodes(pool: pd.DataFrame) -> pd.DataFrame:
    d = pool[pool["rank"] >= 1].sort_values(["zid", "session", "t_sec"])
    rows = []
    for (zid, sess), g in d.groupby(["zid", "session"], sort=False):
        t = g["t_sec"].to_numpy()
        brk = np.concatenate(([0], np.flatnonzero(np.diff(t) > GAP_S) + 1,
                              [len(t)]))
        for a, b in zip(brk[:-1], brk[1:]):
            gg = g.iloc[a:b]
            r0 = gg.iloc[0]
            rows.append({"zid": zid, "session": sess,
                         "t0": int(r0["t_sec"]), "t1": int(gg["t_sec"].iloc[-1]),
                         "n_ref": int(b - a), "rank0": int(r0["rank"]),
                         "rank_min": int(gg["rank"].min()),
                         "dist0": float(r0["dist_atr"]),
                         "side0": int(r0["app_side_above"]),
                         "score0": float(r0["score_T1"]),
                         "seg0": int(r0["seg"]), "atr_t": float(r0["atr_t"])})
    return pd.DataFrame(rows)


def attach_touch(ep: pd.DataFrame, taps_by) -> pd.DataFrame:
    tt, tw, y, ys, ca, cb, tsec = [], [], [], [], [], [], []
    for r in ep.itertuples():
        arr = taps_by.get((r.session, r.zid))
        if arr is None:
            tt.append(False); tw.append(False)
            y.append(np.nan); ys.append(np.nan); ca.append(np.nan)
            cb.append(np.nan); tsec.append(np.nan)
            continue
        secs, yv, ysv, cav, cbv = arr
        j = int(np.searchsorted(secs, r.t0, side="right"))
        if j >= secs.size:
            tt.append(False); tw.append(False)
            y.append(np.nan); ys.append(np.nan); ca.append(np.nan)
            cb.append(np.nan); tsec.append(np.nan)
            continue
        tt.append(True)
        tw.append(bool(secs[j] <= r.t1 + GRACE_S))
        y.append(yv[j]); ys.append(ysv[j]); ca.append(cav[j]); cb.append(cbv[j])
        tsec.append(float(secs[j]))
    ep = ep.copy()
    ep["touched_ss"] = tt
    ep["touched_disp"] = tw
    ep["y"] = y
    ep["y_strict"] = ys
    ep["catA"] = ca
    ep["catB"] = cb
    ep["tap_sec"] = tsec
    return ep


def sess_boot(a: pd.DataFrame, b: pd.DataFrame, col, n=400, seed=3):
    common = np.intersect1d(a["session"].unique(), b["session"].unique())
    if common.size < 8:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan}
    A = a[a["session"].isin(common)].reset_index(drop=True)
    B = b[b["session"].isin(common)].reset_index(drop=True)
    ga = {s: np.flatnonzero(A["session"].to_numpy() == s) for s in common}
    gb = {s: np.flatnonzero(B["session"].to_numpy() == s) for s in common}
    point = float(np.nanmean(A[col]) - np.nanmean(B[col]))
    rng = np.random.default_rng(seed)
    v = np.empty(n)
    for i in range(n):
        p = rng.choice(common, common.size, replace=True)
        v[i] = (np.nanmean(A.iloc[np.concatenate([ga[s] for s in p])][col])
                - np.nanmean(B.iloc[np.concatenate([gb[s] for s in p])][col]))
    v = v[np.isfinite(v)]
    return {"diff": point, "lo": float(np.quantile(v, .025)),
            "hi": float(np.quantile(v, .975))}


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    pool = pd.read_parquet(d0 / "p5_pool.parquet")
    it = tap_table()
    res: dict = {}
    print(f"pool: {len(pool):,} scored rows, {pool['session'].nunique()} "
          f"sessions | displayed {int((pool['rank'] > 0).sum()):,}")

    taps_by = {}
    for (sess, zid), g in it.groupby(["session", "zid"], sort=False):
        g = g.sort_values("tap_sec")
        taps_by[(sess, zid)] = (g["tap_sec"].to_numpy("int64"),
                                g["y"].to_numpy("float64"),
                                g["y_strict"].to_numpy("float64"),
                                g["catA"].to_numpy("float64"),
                                g["catB"].to_numpy("float64"))

    ep = attach_touch(build_episodes(pool), taps_by)
    ep["year"] = pd.to_datetime(ep["session"].astype(str)).dt.year
    print(f"episodes: {len(ep):,} | touched same-session "
          f"{float(ep['touched_ss'].mean()):.3f} | touched while displayed "
          f"{float(ep['touched_disp'].mean()):.3f}")

    # ---- controls at episode start -----------------------------------------
    pool_ix = {k: g for k, g in pool.groupby(["session", "t_sec"], sort=False)}
    rng = np.random.default_rng(11)
    ctl_rows = {"matched": [], "next": [], "random": [], "nearest": []}
    for r in ep.itertuples():
        g = pool_ix.get((r.session, r.t0))
        if g is None:
            continue
        c = g[g["rank"] == 0]
        if not len(c):
            continue
        m = c[(c["app_side_above"] == r.side0)
              & ((c["dist_atr"] - r.dist0).abs() <= 0.10)]
        base = {"session": r.session, "t0": r.t0, "ep_zid": r.zid}
        if len(m):
            pick = m.sample(min(4, len(m)),
                            random_state=int(rng.integers(1 << 31)))
            for _, q in pick.iterrows():
                ctl_rows["matched"].append({**base, "zid": int(q["zid"])})
        nx = c.nlargest(min(4, len(c)), "score_T1")
        for _, q in nx.iterrows():
            ctl_rows["next"].append({**base, "zid": int(q["zid"])})
        rd = c.sample(min(4, len(c)), random_state=int(rng.integers(1 << 31)))
        for _, q in rd.iterrows():
            ctl_rows["random"].append({**base, "zid": int(q["zid"])})
        nr = c.nsmallest(min(4, len(c)), "dist_atr")
        for _, q in nr.iterrows():
            ctl_rows["nearest"].append({**base, "zid": int(q["zid"])})

    ctels = {}
    for name, lst in ctl_rows.items():
        cf = pd.DataFrame(lst)
        if len(cf):
            cf["t1"] = cf["t0"]              # controls have no display period
            cf = attach_touch(cf, taps_by)
        ctels[name] = cf

    # ---- headline tables ----------------------------------------------------
    def block(d, label):
        t = d[d["touched_ss"] & d["y"].notna()]
        return {"n": len(d), "p_touch": float(d["touched_ss"].mean()),
                "y": float(t["y"].mean()) if len(t) else np.nan,
                "y_strict": float(t["y_strict"].mean()) if len(t) else np.nan,
                "catA": float(t["catA"].mean()) if len(t) else np.nan,
                "catB": float(t["catB"].mean()) if len(t) else np.nan,
                "joint": float((d["touched_ss"]
                                & (d["y"].fillna(0) == 1)).mean())}

    print("\n=== arms (touched = next tap same session after display start) ===")
    arms = {"displayed": ep}
    for k in (1, 2, 3, 4):
        arms[f"top{k}"] = ep[ep["rank0"] <= k]
    arms.update({name: c for name, c in ctels.items() if len(c)})
    res["arms"] = {}
    for name, d in arms.items():
        b = block(d, name)
        res["arms"][name] = b
        print(f"  {name:>9}: n {b['n']:7,} | touch {b['p_touch']:.3f} | "
              f"y {b['y']:.4f} | strict {b['y_strict']:.4f} | A {b['catA']:.3f} "
              f"B {b['catB']:.3f} | joint {b['joint']:.4f}")

    tt = ep[ep["touched_ss"] & ep["y"].notna()].copy()
    mm = ctels["matched"]
    mm = mm[mm["touched_ss"] & mm["y"].notna()].copy()
    ci = sess_boot(tt, mm, "y", a.boot)
    cis = sess_boot(tt, mm, "y_strict", a.boot)
    print(f"\n  displayed-vs-matched  y: {ci['diff']:+.4f} "
          f"[{ci['lo']:+.4f},{ci['hi']:+.4f}] | strict: {cis['diff']:+.4f} "
          f"[{cis['lo']:+.4f},{cis['hi']:+.4f}]")
    res["ci_y"] = ci
    res["ci_strict"] = cis

    print("\n=== per-year (displayed vs matched, y | touched; joint) ===")
    res["by_year"] = {}
    for y_, gy in ep.groupby("year"):
        my = ctels["matched"]
        my = my[pd.to_datetime(my["session"].astype(str)).dt.year == y_]
        ty = gy[gy["touched_ss"] & gy["y"].notna()]
        tm = my[my["touched_ss"] & my["y"].notna()]
        row = {"n_ep": len(gy), "touch": float(gy["touched_ss"].mean()),
               "y": float(ty["y"].mean()), "y_m": float(tm["y"].mean()),
               "strict": float(ty["y_strict"].mean()),
               "strict_m": float(tm["y_strict"].mean())}
        res["by_year"][int(y_)] = row
        print(f"  {y_}: eps {row['n_ep']:6,} touch {row['touch']:.3f} | "
              f"y {row['y']:.3f} vs {row['y_m']:.3f} "
              f"({row['y']-row['y_m']:+.3f}) | strict {row['strict']:.3f} vs "
              f"{row['strict_m']:.3f}")

    # ---- product usefulness -------------------------------------------------
    print("\n=== product usefulness (touched while displayed) ===")
    ep["succ_disp"] = ep["touched_disp"] & (ep["y"].fillna(0) == 1)
    per = ep.groupby("session").agg(n_ep=("zid", "size"),
                                    n_touch=("touched_disp", "sum"),
                                    n_succ=("succ_disp", "sum"))
    u = {"episodes_per_day": float(per["n_ep"].mean()),
         "touched_disp_per_day": float(per["n_touch"].mean()),
         "successes_per_day": float(per["n_succ"].mean()),
         "p_day_ge1_success": float((per["n_succ"] >= 1).mean()),
         "p_day_zero_touch": float((per["n_touch"] == 0).mean()),
         "med_dist0": float(ep["dist0"].median()),
         "med_time_to_touch_min": float(
             ((ep.loc[ep["touched_disp"], "tap_sec"]
               - ep.loc[ep["touched_disp"], "t0"]) / 60).median()),
         "med_episode_min": float(((ep["t1"] - ep["t0"]) / 60 + 15).median()),
         "share_touched_while_disp": float(ep["touched_disp"].mean()),
         "share_outranked": np.nan}
    res["usefulness"] = u
    for k, v in u.items():
        print(f"  {k:>26}: {v if isinstance(v, float) else v:.3f}")

    # stability / churn
    d1 = pool[pool["rank"] == 1].sort_values(["session", "t_sec"])
    same = (d1["zid"] == d1.groupby("session")["zid"].shift()).mean()
    print(f"  rank-1 persistence (same zid at next refresh): {float(same):.3f}")
    res["rank1_persist"] = float(same)

    # ---- advance notice -----------------------------------------------------
    print("\n=== advance notice (score at lead vs tap outcome) ===")
    ld = pd.read_parquet(d0 / "p5_leads.parquet")
    key = ["session", "zid", "tap_sec"]
    ld = ld.rename(columns={"tap_sec_of": "tap_sec"}).merge(
        it[key + ["y"]], on=key, how="inner")
    res["leads"] = {}
    for L, g in ld.groupby("lead_s"):
        q = g["score_T1"].quantile(2 / 3)
        top = g[g["score_T1"] >= q]
        row = {"n": len(g), "auc_T1": auc(g["y"], g["score_T1"]),
               "auc_T2": auc(g["y"], g["score_T2"]),
               "y_top_tercile": float(top["y"].mean()),
               "y_all": float(g["y"].mean()),
               "med_dist": float(g["dist_atr_lead"].median())}
        res["leads"][int(L)] = row
        print(f"  lead {int(L)//60:>3}m: AUC T1 {row['auc_T1']:.4f} T2 "
              f"{row['auc_T2']:.4f} | top-tercile y {row['y_top_tercile']:.4f} "
              f"vs all {row['y_all']:.4f} | med dist {row['med_dist']:.3f}")
    w = ld.pivot_table(index=key, columns="lead_s", values="score_T1")
    if 1800 in w.columns and 120 in w.columns:
        cc = w[[1800, 120]].dropna().corr().iloc[0, 1]
        print(f"  score stability corr(lead30m, lead2m): {cc:.3f}")
        res["lead_corr"] = float(cc)

    # ---- side asymmetry adversarial ----------------------------------------
    print("\n=== side asymmetry: adversarial checks ===")
    rc = pd.read_parquet(d0 / "p5_races.parquet")
    p_dn = float((rc["winner"] == "dn").mean())
    p_up = float((rc["winner"] == "up").mean())
    print(f"  random-moment ±15pt race: down-first {p_dn:.3f}, up-first "
          f"{p_up:.3f}, unresolved {1-p_dn-p_up:.3f}")
    rc["year"] = pd.to_datetime(rc["session"].astype(str)).dt.year
    print("  down-first by year: " + " ".join(
        f"{y}:{(g['winner']=='dn').mean():.3f}" for y, g in rc.groupby("year")))
    res["race_dn"] = p_dn
    t = it[it["y"].notna()]
    comp = t.groupby("app_side_above").agg(y=("y", "mean"),
                                           A=("catA", "mean"),
                                           B=("catB", "mean"), n=("y", "size"))
    print("  tap success by side (0=resistance-side, 1=support-side):")
    print(comp.round(3).to_string())
    res["side_table"] = comp.reset_index().to_dict("records")
    dirs = it.groupby(["app_side_above", "dir"]).size().unstack()
    print("  zone formation dir by approach side:")
    print(dirs.to_string())

    out = ROOT / "outputs" / "p5_report.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
