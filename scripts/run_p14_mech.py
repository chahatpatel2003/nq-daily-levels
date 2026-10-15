#!/usr/bin/env python
"""P14: checkpoint-model grid + simple mechanical filter family.
Mechanical thresholds are 2021-23 quantiles (selection era); 2024-26 is
the frozen forward evaluation. No threshold touches 24-26."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_p14_model import (FIRE, BASE_PRE, FIRE_F, HEALTH, fit_score,          # noqa: E402
                           eval_retention, metrics, portfolio)

CK_FEATS = {0: [], 2: ["ck2_rev", "ck2_cl", "ck2_dlt"],
            5: ["ck5_rev", "ck5_cl", "ck5_dlt"],
            10: ["ck10_rev", "ck10_cl", "ck10_dlt"],
            15: ["ck15_rev", "ck15_cl", "ck15_dlt"]}


def trades_for(d, k):
    d = d[d[f"ck{k}_epx"].notna()].copy()
    up = d[f"ck{k}_up12.5"].fillna(-1).to_numpy()
    dn = d[f"ck{k}_dn30.0"].fillna(-1).to_numpy()
    win = (dn >= 0) & ((up < 0) | (dn < up))
    loss = (up >= 0) & ~win
    end_pnl = (d[f"ck{k}_epx"] - d[f"ck{k}_endpx"]).to_numpy() / 4.0
    d["pnl"] = np.where(win, 30.0, np.where(loss, -12.5, end_pnl)) - 1.0
    d["R"] = d["pnl"] / 12.5
    d["t_open"] = d[f"ck{k}_eidx"]
    d["t_close"] = d["t_open"] + np.where(win, dn,
                                          np.where(loss, up, 10 ** 9))
    return portfolio(d)


def main() -> int:
    df = pd.read_parquet(FIRE)
    df["session"] = df["session"].astype(str)
    df["year"] = pd.to_datetime(df["session"]).dt.year
    df = df.sort_values(["session", "tap_sec"]).reset_index(drop=True)
    sess_all = sorted(df.loc[df["year"] >= 2022, "session"].unique())
    sess_oos = [s for s in sess_all if s[:4] in ("2024", "2025", "2026")]
    sess_sel = [s for s in sess_all if s[:4] in ("2022", "2023")]
    res = {}

    print("=== checkpoint model grid (hgb, top-30%/50% by train quantile) ===")
    for k in (0, 2, 5, 10, 15):
        feats = [f for f in BASE_PRE + FIRE_F + HEALTH + CK_FEATS[k]
                 if f in df.columns]
        sc, thr, aucs = fit_score(df, feats, k, "hgb")
        a = " ".join(f"{y}:{v:.3f}" for y, v in aucs.items())
        print(f"\n  ck{k} AUC: {a}")
        ev = df[df["year"] >= 2022]
        r = eval_retention(ev, sc[df["year"] >= 2022], thr, k, sess_all,
                           f"ck{k}", ret_grid=(0.3, 0.5))
        res[f"ck{k}"] = {str(q): {kk: vv for kk, vv in m.items()
                                  if kk != "_df"} for q, m in r.items()}

    print("\n=== mechanical family (thresholds = 2021-23 quantiles) ===")
    tr = df[df["year"] <= 2023]
    q_fire = {q: float(tr["fire_s"].quantile(q)) for q in (0.25, 0.5, 0.75)}
    q_mono = float(tr["mono"].median())
    q_eff = float(tr["eff_ratio"].median())
    print(f"  thresholds: fire_s q25/50/75 = "
          f"{q_fire[0.25]:.0f}/{q_fire[0.5]:.0f}/{q_fire[0.75]:.0f}s, "
          f"mono med {q_mono:.2f}, eff med {q_eff:.2f}")
    rules = {
        "all fires": np.ones(len(df), bool),
        "tap<=3": (df["tap_no"] <= 3).to_numpy(),
        "tap1": (df["tap_no"] == 1).to_numpy(),
        "fast50": (df["fire_s"] <= q_fire[0.5]).to_numpy(),
        "t3+fast50": ((df["tap_no"] <= 3)
                      & (df["fire_s"] <= q_fire[0.5])).to_numpy(),
        "t3+fast25": ((df["tap_no"] <= 3)
                      & (df["fire_s"] <= q_fire[0.25])).to_numpy(),
        "t3+fast50+mono": ((df["tap_no"] <= 3)
                           & (df["fire_s"] <= q_fire[0.5])
                           & (df["mono"] >= q_mono)).to_numpy(),
        "t3+fast50+eff": ((df["tap_no"] <= 3)
                          & (df["fire_s"] <= q_fire[0.5])
                          & (df["eff_ratio"] >= q_eff)).to_numpy(),
        "t3+slow (anti)": ((df["tap_no"] <= 3)
                           & (df["fire_s"] > q_fire[0.75])).to_numpy(),
    }
    for era, sess in (("SEL 22-23", sess_sel), ("OOS 24-26", sess_oos)):
        print(f"\n  [{era}]")
        yrs = (2022, 2023) if era.startswith("SEL") else (2024, 2025, 2026)
        sub = df[df["year"].isin(yrs)]
        for nm, mask in rules.items():
            d = trades_for(sub[mask[df["year"].isin(yrs).to_numpy()]], 2)
            m = metrics(d, sess, nm)
            if m:
                res.setdefault("mech", {}).setdefault(era, {})[nm] = m
    out = ROOT / "outputs" / "p14_mech.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
