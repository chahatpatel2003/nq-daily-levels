#!/usr/bin/env python
"""P11 Monte Carlo + prop engine + sizing/extraction.

Day-level resampling preserves within-day clustering exactly (the trades of
a day stay together); moving blocks preserve week-scale regime runs.
Reported: 250-day max-DD distributions per method, design-DD percentiles,
violation probabilities on a parameterised trailing-buffer prop account
across risk-per-trade and buffer grids, and expected extraction until
violation/horizon. Also: correlation diagnostics, session-risk and tap
overlays, and the overlap-allowed comparison.
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

from run_p10_eval import rows_for                                              # noqa: E402
from run_p10_stage1 import load_trades, metrics, trade_pnl                     # noqa: E402

RISK = 12.5
N_SIM = 2000
HORIZON = 250


def max_dd(x):
    eq = np.cumsum(x)
    return float((np.maximum.accumulate(eq) - eq).max())


def sim_days(day_R, method, rng, horizon=HORIZON, block=5):
    n = len(day_R)
    if method == "iid_day":
        return day_R[rng.integers(0, n, horizon)]
    if method == "block":
        out = []
        while len(out) < horizon:
            s = rng.integers(0, n - block)
            out.extend(day_R[s:s + block])
        return np.array(out[:horizon])
    raise ValueError(method)


def main() -> int:
    t0 = _time.time()
    pf = pd.read_parquet("/Volumes/T9/nq_levels/daily/p11_stream.parquet")
    t = load_trades()
    sessions = sorted(t[t["year"] >= 2022]["session"].unique())
    day = pf.groupby("session")["R"].sum().reindex(sessions).fillna(0.0)
    day_R = day.to_numpy()
    res: dict = {}
    rng = np.random.default_rng(7)
    print(f"day series: {len(day_R):,} days | mean {day_R.mean():+.2f}R")

    # capped variant (max 8 trades/day) as the execution-realistic stream
    cap8 = pf.sort_values(["session", "t_open"]).groupby("session").head(8)
    day8 = cap8.groupby("session")["R"].sum().reindex(sessions).fillna(0.0).to_numpy()

    # trade-level IID reference
    trR = pf["R"].to_numpy()
    tpd = len(trR) / len(day_R)

    print("\n=== PART 11: 250-day max-DD distributions (R), 2,000 sims ===")
    print(f"{'method':>18} {'p50':>6} {'p75':>6} {'p90':>6} {'p95':>6} "
          f"{'p97.5':>7} {'p99':>6} {'p99.5':>7}")
    for name, gen in (
            ("chronological", None),
            ("iid trades", lambda: trR[rng.integers(0, len(trR),
                                                    int(tpd * HORIZON))]),
            ("iid days", lambda: sim_days(day_R, "iid_day", rng)),
            ("block-5 days", lambda: sim_days(day_R, "block", rng)),
            ("block-5 (cap8)", lambda: sim_days(day8, "block", rng))):
        if gen is None:
            dds = []
            for s in range(0, len(day_R) - HORIZON, 21):
                dds.append(max_dd(day_R[s:s + HORIZON]))
            dds = np.array(dds)
            lbl = f"(rolling n={len(dds)})"
        else:
            dds = np.array([max_dd(gen()) for _ in range(N_SIM)])
            lbl = ""
        q = np.percentile(dds, [50, 75, 90, 95, 97.5, 99, 99.5])
        res.setdefault("mc", {})[name] = {p: float(v) for p, v in
                                          zip([50, 75, 90, 95, 97.5, 99, 99.5], q)}
        print(f"{name:>18} " + " ".join(f"{v:6.1f}" for v in q) + f"  {lbl}")
    dds_blk = np.array([max_dd(sim_days(day_R, "block", rng))
                        for _ in range(N_SIM)])
    print("\nP(250-day max DD >= X): " + " ".join(
        f"{x}R:{float((dds_blk>=x).mean()):.3f}"
        for x in (20, 25, 27, 30, 35, 40, 45, 50, 60)))
    res["exceed"] = {str(x): float((dds_blk >= x).mean())
                     for x in (20, 25, 27, 30, 35, 40, 45, 50, 60)}

    # ---- PART 19: correlation diagnostics ----------------------------------
    print("\n=== PART 19: within-day clustering ===")
    g = pf.groupby("session")["R"]
    var_ratio = float(day.var() / (pf["R"].var() * g.size().reindex(day.index)
                                   .fillna(0).mean()))
    w = pf.assign(prev=pf.groupby("session")["R"].shift())
    pl = w[w["prev"].notna()]
    print(f"day-sum variance ratio vs independent: {var_ratio:.2f}")
    print(f"P(loss | prior same-day loss) {float((pl[pl['prev']<=0]['R']<=0).mean()):.3f} "
          f"vs P(loss) {float((pf['R']<=0).mean()):.3f}")
    res["clustering"] = var_ratio

    # ---- PART 17/18/20: overlays ------------------------------------------
    print("\n=== overlays (full stream metrics) ===")
    base_m = metrics(pf["pnl"].to_numpy(), np.full(len(pf), RISK),
                     pf["pnl"].to_numpy() > 0, pf["pnl"].to_numpy() <= 0,
                     pf["session"].to_numpy())
    def show(name, m_):
        if m_:
            print(f"  {name:>26}: n {m_['n']:6,} expR {m_['exp_R']:+.3f} "
                  f"totR {m_['exp_R']*m_['n']:7.0f} maxDD {m_['maxdd_R']:5.0f} "
                  f"PF {m_['pf']:.2f}")
    show("base (one-open)", base_m)
    half = pf.copy()
    half["w"] = np.where(half["seg"].isin([0, 1]), 0.5, 1.0)
    pnl_h = (half["pnl"] * half["w"]).to_numpy()
    show("Asia/London half-risk", metrics(pnl_h, np.full(len(half), RISK),
                                          pnl_h > 0, pnl_h <= 0,
                                          half["session"].to_numpy()))
    sk = pf[pf["tap_no"] <= 3]
    show("skip tap>=4 overlay", metrics(sk["pnl"].to_numpy(),
                                        np.full(len(sk), RISK),
                                        sk["pnl"].to_numpy() > 0,
                                        sk["pnl"].to_numpy() <= 0,
                                        sk["session"].to_numpy()))
    tap_dd = pf.copy()
    for lab, m_ in (("tap<=3 only DD", None),):
        pass
    allf = rows_for(t[t["year"] >= 2022], "E1d", ("r4",), 0.85)
    pnl_a, S_a, w_a, l_a, ses_a = trade_pnl(allf, "ts1_2", 12.5, 30.0, 1.0)
    show("overlap-allowed (all fires)", metrics(pnl_a, S_a, w_a, l_a, ses_a))

    # ---- PARTS 10/12-16: prop engine --------------------------------------
    print("\n=== PARTS 12-16: parameterised trailing-buffer prop engine ===")
    print("violation = trailing max-drawdown buffer breached intraday-approx "
          "(daily granularity); horizon 250 days;\nexpected extraction = "
          "E[cum P&L until violation-or-horizon]. REAL FIRM RULES REQUIRED "
          "for final numbers.\n")
    print(f"{'buffer$':>8} {'risk$':>6} {'≈R buf':>7} {'P(viol)':>8} "
          f"{'med d2v':>8} {'E[extract]$':>12} {'E[$/day]':>9}")
    day_blocks = day_R
    for B in (3000.0, 4500.0, 6000.0, 9000.0):
        for riskusd in (50, 75, 100, 125, 150, 200, 250, 300, 400, 500):
            bufR = B / riskusd
            viol, extract, d2v = 0, [], []
            for _ in range(600):
                path = sim_days(day_blocks, "block", rng)
                eq = np.cumsum(path) * riskusd
                trail = np.maximum.accumulate(np.concatenate(([0.0], eq)))[1:]
                bust = np.flatnonzero(trail - eq >= B)
                if bust.size:
                    viol += 1
                    extract.append(eq[bust[0]] + 0)   # equity at violation
                    d2v.append(bust[0])
                else:
                    extract.append(eq[-1])
            row = {"P_viol": viol / 600,
                   "med_d2v": float(np.median(d2v)) if d2v else np.nan,
                   "E_extract": float(np.mean(extract)),
                   "E_per_day": float(np.mean(extract)) / HORIZON}
            res.setdefault("prop", {})[f"B{int(B)}_r{riskusd}"] = row
            if riskusd in (75, 100, 150, 200, 300, 500) or B == 4500:
                print(f"{int(B):>8} {riskusd:>6} {bufR:>7.0f} "
                      f"{row['P_viol']:>8.3f} {row['med_d2v']:>8.0f} "
                      f"{row['E_extract']:>12,.0f} {row['E_per_day']:>9,.0f}")

    out = ROOT / "outputs" / "p11_mc.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
