#!/usr/bin/env python
"""P14 walk-forward fire-time models + abstention frontier.

Protocol (fixed before results): expanding yearly walk-forward, test year
Y trained on years < Y (2021 train-only). Label: target-30 before
stop-12.5 from the ck2 entry (development label; S/T re-selected later).
Models: L2 logistic (imputer+scaler fit on train only) and shallow HGB.
Abstention thresholds = TRAIN-score quantiles applied to the test year.
Evaluation: one-open-trade portfolio, trade + day-level DD + underwater.
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

FIRE = "/Volumes/T9/nq_levels/daily/p14_fire.parquet"

BASE_PRE = ["app_speed_5m", "app_accel", "app_volx_5m", "app_delta_5m",
            "app_toward_30m", "app_range_5m", "app_retraces_15m",
            "rv60_atr", "atr_pctile", "range_used_atr", "vwap_dist_atr",
            "tod_pt_min", "dep_vel", "mins_since_last", "age_min",
            "hw_atr", "tap_no", "seg", "zprior_n", "zprior_succ", "n_fwd"]
FIRE_F = ["fire_s", "t_m2", "t_m4", "t_m6", "t_m8", "v_inst", "v_avg",
          "mono", "eff_ratio", "n_flips", "ret_above", "t_inzone",
          "pen_max", "pen_hw", "hit_mid", "hit_far", "beyond_far",
          "s_above_far", "max_run_far", "dlt_fire", "vol_fire",
          "sell_share", "intx_fire", "sell_imp", "buy_fail",
          "buy_pre_prog", "app_net_5m", "rng_used", "ret_1h", "ret_3h",
          "tod_hr", "dow"]
HEALTH = ["recent5", "recent10", "recent20", "recent50"]
FAMS = {"pre": BASE_PRE, "fire": FIRE_F, "health": HEALTH}
RET_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)


def portfolio(d):
    d = d.sort_values(["session", "t_open", "zid", "tap_sec"],
                      kind="stable")     # deterministic tie-break (P16 A1)
    keep, cs, cc = [], None, -1.0
    for r in d.itertuples():
        ok = r.session != cs or r.t_open >= cc
        keep.append(ok)
        if ok:
            cs, cc = r.session, r.t_close
    return d[np.array(keep)]


def metrics(d, sessions, label="", show=True):
    r = d["R"].to_numpy()
    if len(r) < 50:
        return {}
    w, l = r[r > 0], r[r <= 0]
    eq = np.cumsum(r)
    dd_t = float((np.maximum.accumulate(eq) - eq).max())
    day = d.groupby("session")["R"].sum().reindex(sessions).fillna(0.0)
    eqd = np.cumsum(day.to_numpy())
    dd_d = float((np.maximum.accumulate(eqd) - eqd).max())
    hw = np.maximum.accumulate(eqd)
    uw_runs, cur = [], 0
    for i in range(len(eqd)):
        if eqd[i] < hw[i] - 1e-9:
            cur += 1
        else:
            if cur:
                uw_runs.append(cur)
            cur = 0
    if cur:
        uw_runs.append(cur)
    mon = d.assign(m=d["session"].str[:7]).groupby("m")["R"].sum()
    all_m = sorted({s[:7] for s in sessions})
    mon = mon.reindex(all_m).fillna(0.0)
    stk, run = 0, 0
    for v in r:
        run = run + 1 if v <= 0 else 0
        stk = max(stk, run)
    out = dict(n=len(r), tpd=len(r) / len(sessions),
               wr=float((r > 0).mean()), expR=float(r.mean()),
               totR=float(r.sum()), pf=float(w.sum() / -l.sum()),
               dd_t=dd_t, dd_d=dd_d,
               uw_max=int(max(uw_runs)) if uw_runs else 0,
               uw20=sum(1 for u in uw_runs if u >= 20),
               worst_m=float(mon.min()), neg_m=int((mon < 0).sum()),
               streak=stk)
    if show:
        print(f"  {label:>22}: n {out['n']:5,} tpd {out['tpd']:4.2f} "
              f"WR {out['wr']:.3f} expR {out['expR']:+.3f} "
              f"totR {out['totR']:6.0f} PF {out['pf']:4.2f} "
              f"DDt {out['dd_t']:5.1f} DDd {out['dd_d']:5.1f} "
              f"uwMax {out['uw_max']:3d} uw20 {out['uw20']} "
              f"worstM {out['worst_m']:+5.1f} negM {out['neg_m']:2d} "
              f"stk {out['streak']}")
    return out


def make_xy(df, feats, k):
    up = df[f"ck{k}_up12.5"].fillna(-1)
    dn = df[f"ck{k}_dn30.0"].fillna(-1)
    ok = df[f"ck{k}_epx"].notna()
    y = ((dn >= 0) & ((up < 0) | (dn < up))).astype(int)
    return df[feats], y, ok


def fit_score(df, feats, k, model="hgb"):
    """Expanding-yearly walk-forward scores + train-quantile thresholds."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y, ok = make_xy(df, feats, k)
    score = np.full(len(df), np.nan)
    thr_train: dict = {}
    aucs = {}
    for Y in (2022, 2023, 2024, 2025, 2026):
        tr = df["year"] < Y
        te = (df["year"] == Y) & ok
        if tr.sum() < 3000 or te.sum() < 300:
            continue
        if model == "hgb":
            g = HistGradientBoostingClassifier(
                max_depth=3, max_iter=200, learning_rate=0.05,
                min_samples_leaf=200, random_state=0)
            g.fit(X[tr], y[tr])
            s_te = g.predict_proba(X[te])[:, 1]
            s_tr = g.predict_proba(X[tr])[:, 1]
        else:
            g = make_pipeline(SimpleImputer(strategy="median"),
                              StandardScaler(),
                              LogisticRegression(C=0.1, max_iter=2000))
            g.fit(X[tr], y[tr])
            s_te = g.predict_proba(X[te])[:, 1]
            s_tr = g.predict_proba(X[tr])[:, 1]
        score[te.to_numpy()] = s_te
        thr_train[Y] = {q: float(np.quantile(s_tr, 1 - q))
                        for q in RET_GRID}
        from sklearn.metrics import roc_auc_score
        if y[te].nunique() > 1:
            aucs[Y] = float(roc_auc_score(y[te], s_te))
    return score, thr_train, aucs


def eval_retention(df, score, thr_train, k, sessions, label,
                   ret_grid=RET_GRID, show=True):
    """Apply train-quantile thresholds per test year; portfolio metrics."""
    out = {}
    for q in ret_grid:
        keep = np.zeros(len(df), bool)
        for Y, thr in thr_train.items():
            m = (df["year"] == Y).to_numpy() & np.isfinite(score)
            keep |= m & (score >= thr[q])
        d = df[keep].copy()
        if len(d) < 100:
            continue
        up = d[f"ck{k}_up12.5"].fillna(-1).to_numpy()
        dn = d[f"ck{k}_dn30.0"].fillna(-1).to_numpy()
        win = (dn >= 0) & ((up < 0) | (dn < up))
        loss = (up >= 0) & ~win
        end_pnl = (d[f"ck{k}_epx"] - d[f"ck{k}_endpx"]).to_numpy() / 4.0
        d["pnl"] = np.where(win, 30.0, np.where(loss, -12.5, end_pnl)) - 1.0
        d["R"] = d["pnl"] / 12.5
        d["t_open"] = d[f"ck{k}_eidx"]
        d["t_close"] = d["t_open"] + np.where(
            win, dn, np.where(loss, up, 10 ** 9))
        d = portfolio(d)
        out[q] = metrics(d, sessions, f"{label} top{int(q*100)}%",
                         show=show)
        out[q]["_df"] = d
    return out


def main() -> int:
    t0 = _time.time()
    df = pd.read_parquet(FIRE)
    df["session"] = df["session"].astype(str)
    df["year"] = pd.to_datetime(df["session"]).dt.year
    sessions = sorted(df.loc[df["year"] >= 2022, "session"].unique())
    df = df.sort_values(["session", "tap_sec"]).reset_index(drop=True)
    ev = df[df["year"] >= 2022]
    print(f"fires: {len(df):,} (2021 train-only {int((df['year']==2021).sum()):,})")

    feats_all = [f for f in BASE_PRE + FIRE_F + HEALTH if f in df.columns]
    print(f"features: {len(feats_all)}")
    res = {}

    print("\n=== ck2 (decision fire+2s, entry fire+3s) — model comparison ===")
    for model in ("logit", "hgb"):
        sc, thr, aucs = fit_score(df, feats_all, 2, model)
        print(f"  {model} AUC by year: " +
              " ".join(f"{y}:{a:.3f}" for y, a in aucs.items()))
        res[f"auc_{model}"] = aucs
        ev2 = df[df["year"] >= 2022]
        r = eval_retention(ev2, sc[df["year"] >= 2022], thr, 2, sessions,
                           model, ret_grid=(0.3, 0.5, 0.7, 1.0))
        res[f"ck2_{model}"] = {str(q): {kk: vv for kk, vv in m.items()
                                        if kk != "_df"}
                               for q, m in r.items()}

    out = ROOT / "outputs" / "p14_model.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {out.relative_to(ROOT)} [{(_time.time()-t0)/60:.1f} min]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
