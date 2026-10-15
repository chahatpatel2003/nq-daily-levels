#!/usr/bin/env python
"""P14 finalists: ck0 entries for mechanical rules, threshold plateaus,
model-within-rule hybrid, staged S/T reselection (select 2021-23,
evaluate 2024-26), MC, year/session breakdowns."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_p14_model import (FIRE, BASE_PRE, FIRE_F, HEALTH, fit_score,          # noqa: E402
                           metrics, portfolio)

S_GRID = (7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 25.0, 30.0)
T_GRID = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0)


def trades_for(d, k, s_=12.5, t_=30.0):
    d = d[d[f"ck{k}_epx"].notna()].copy()
    up = d[f"ck{k}_up{s_}"].fillna(-1).to_numpy()
    dn = d[f"ck{k}_dn{t_}"].fillna(-1).to_numpy()
    win = (dn >= 0) & ((up < 0) | (dn < up))
    loss = (up >= 0) & ~win
    end_pnl = (d[f"ck{k}_epx"] - d[f"ck{k}_endpx"]).to_numpy() / 4.0
    d["pnl"] = np.where(win, t_, np.where(loss, -s_, end_pnl)) - 1.0
    d["R"] = d["pnl"] / s_
    d["t_open"] = d[f"ck{k}_eidx"]
    d["t_close"] = d["t_open"] + np.where(win, dn,
                                          np.where(loss, up, 10 ** 9))
    return portfolio(d)


def mc_dd(day, n_sim=3000, horizon=250, block=5, seed=14):
    rng = np.random.default_rng(seed)
    x = day.to_numpy()
    n = len(x)
    dds = []
    for _ in range(n_sim):
        out = []
        while len(out) < horizon:
            s0 = rng.integers(0, n - block)
            out.extend(x[s0:s0 + block])
        e = np.cumsum(out[:horizon])
        dds.append(float((np.maximum.accumulate(e) - e).max()))
    d = np.array(dds)
    return {p: float(np.percentile(d, p)) for p in (50, 90, 95, 99)}


def main() -> int:
    df = pd.read_parquet(FIRE)
    df["session"] = df["session"].astype(str)
    df["year"] = pd.to_datetime(df["session"]).dt.year
    df = df.sort_values(["session", "tap_sec"]).reset_index(drop=True)
    sess_all = sorted(df.loc[df["year"] >= 2022, "session"].unique())
    tr_era = df["year"] <= 2023
    q50 = float(df.loc[tr_era, "fire_s"].quantile(0.5))
    qm = float(df.loc[tr_era, "mono"].median())
    res = {}

    RULE = ((df["tap_no"] <= 3) & (df["fire_s"] <= q50)
            & (df["mono"] >= qm)).to_numpy()
    R_T1 = (df["tap_no"] == 1).to_numpy()

    print("=== ck0 vs ck2 entry for finalists (12.5/30) ===")
    for nm, mask in (("t3+fast+mono", RULE), ("tap1", R_T1)):
        for k in (0, 2):
            d = trades_for(df[mask & (df["year"] >= 2022)], k)
            res[f"{nm}_ck{k}"] = metrics(d, sess_all, f"{nm} ck{k}")

    print("\n=== threshold plateau (ck0, 12.5/30, 2022-26) ===")
    for fs in (15, 20, 24, 30, 40):
        for mth in (0.0, qm):
            m2 = ((df["tap_no"] <= 3) & (df["fire_s"] <= fs)
                  & (df["mono"] >= mth)).to_numpy()
            d = trades_for(df[m2 & (df["year"] >= 2022)], 0)
            lab = f"fs<={fs}" + ("+mono" if mth > 0 else "")
            res.setdefault("plateau", {})[lab] = metrics(d, sess_all, lab)
    for tmax in (1, 2, 3, 4):
        m2 = ((df["tap_no"] <= tmax) & (df["fire_s"] <= q50)
              & (df["mono"] >= qm)).to_numpy()
        d = trades_for(df[m2 & (df["year"] >= 2022)], 0)
        res["plateau"][f"tap<={tmax}"] = metrics(d, sess_all,
                                                 f"tap<={tmax}+fast+mono")

    print("\n=== hybrid: hgb ranking within rule (train-quantile 70%) ===")
    feats = [f for f in BASE_PRE + FIRE_F + HEALTH if f in df.columns]
    sc, thr, aucs = fit_score(df, feats, 0, "hgb")
    keep = np.zeros(len(df), bool)
    for Y, t_ in thr.items():
        keep |= ((df["year"] == Y).to_numpy() & np.isfinite(sc)
                 & (sc >= t_[0.7]))
    d = trades_for(df[RULE & keep & (df["year"] >= 2022)], 0)
    res["hybrid"] = metrics(d, sess_all, "rule+model70")

    print("\n=== staged S/T on t3+fast+mono (ck0): select 21-23 ===")
    sel = df[RULE & (df["year"] <= 2023)]
    best = []
    for s_ in S_GRID:
        for t_ in T_GRID:
            d = trades_for(sel, 0, s_, t_)
            r = d["R"].to_numpy()
            if len(r) < 300:
                continue
            eq = np.cumsum(r)
            dd = float((np.maximum.accumulate(eq) - eq).max())
            best.append((float(r.mean()), s_, t_, float(r.sum()), dd,
                         float((r > 0).mean())))
    best.sort(reverse=True)
    print("  top-8 by selection expR (S, T, expR, totR, DD, WR):")
    for e, s_, t_, tot, dd, wr in best[:8]:
        print(f"    S{s_:5.1f} T{t_:5.1f}: {e:+.3f}R tot {tot:5.0f} "
              f"DD {dd:5.1f} WR {wr:.3f}")
    res["st_sel"] = [(s_, t_, e) for e, s_, t_, *_ in best[:8]]

    print("\n=== frozen OOS 24-26 for chosen S/T + neighbours ===")
    sess_oos = [s for s in sess_all if s[:4] >= "2024"]
    for s_, t_ in {(best[0][1], best[0][2]), (12.5, 30.0), (10.0, 30.0),
                   (15.0, 30.0), (12.5, 25.0), (12.5, 40.0)}:
        d = trades_for(df[RULE & (df["year"] >= 2024)], 0, s_, t_)
        res.setdefault("st_oos", {})[f"S{s_}T{t_}"] = metrics(
            d, sess_oos, f"OOS S{s_}/T{t_}")

    print("\n=== finalist deep profile: t3+fast+mono ck0 12.5/30 ===")
    d = trades_for(df[RULE & (df["year"] >= 2022)], 0)
    day = d.groupby("session")["R"].sum().reindex(sess_all).fillna(0.0)
    res["mc"] = mc_dd(day)
    print(f"  MC 250d maxDD: p50 {res['mc'][50]:.1f} p90 {res['mc'][90]:.1f}"
          f" p95 {res['mc'][95]:.1f} p99 {res['mc'][99]:.1f}")
    for y, g in d.groupby(d["session"].str[:4]):
        r = g["R"].to_numpy()
        sess_y = [s for s in sess_all if s[:4] == y]
        print(f"  {y}: n {len(r):4} tpd {len(r)/len(sess_y):.2f} "
              f"WR {(r>0).mean():.3f} expR {r.mean():+.3f} "
              f"PF {r[r>0].sum()/-r[r<=0].sum():.2f}")
        res.setdefault("yr", {})[y] = dict(n=len(r), expR=float(r.mean()))
    segn = {0: "asia", 1: "london", 2: "premkt", 3: "ny-am", 4: "ny-pm"}
    for sgi, g in d.groupby("seg"):
        r = g["R"].to_numpy()
        if len(r) < 30:
            continue
        print(f"  {segn.get(int(sgi), sgi):>7}: n {len(r):4} "
              f"WR {(r>0).mean():.3f} expR {r.mean():+.3f} "
              f"PF {r[r>0].sum()/-r[r<=0].sum():.2f}")
        res.setdefault("seg", {})[segn.get(int(sgi), str(sgi))] = \
            dict(n=len(r), expR=float(r.mean()))
    out = ROOT / "outputs" / "p14_final.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
