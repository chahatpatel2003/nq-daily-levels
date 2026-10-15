#!/usr/bin/env python
"""P12 final pass: corrected failure-matched comparison (rising-limb
crossing), annual threshold probabilities, payout cadence, philosophy
table, and the four final configurations."""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from p12_engine import STD, load_streams, simulate                            # noqa: E402
from run_p12_sweeps import IDX, REP, ST, q                                    # noqa: E402

RULES = dict(STD, cap=10000.0)
RES: dict = {}
SESSIONS, _ = load_streams()


def pol(risk, cmult=18.0):
    return dict(T1=0.0, C=float(cmult) * risk)


def run_annual(nm, risk, cmult=18.0, n=1500):
    return simulate(ST[nm], IDX[:n, :250], risk, RULES, pol(risk, cmult),
                    replace=REP)


def p_die250(nm, risk, cmult=18.0, n=1500):
    r = simulate(ST[nm], IDX[:n, :250], risk, RULES, pol(risk, cmult))
    return float(r["dead"].mean())


def stage_E2():
    print("=== E2 — failure-matched streams (rising-limb crossing) ===")
    grid = (50., 75., 100., 125., 150., 175., 200., 250., 300., 350.,
            400., 500., 600.)
    curves = {}
    for nm in ("raw", "cap8", "tap3", "tap3cap8"):
        ps = [p_die250(nm, r) for r in grid]
        curves[nm] = ps
        print(f"  {nm:>9} P(die<=250d): " +
              " ".join(f"{int(r)}:{p:.2f}" for r, p in zip(grid, ps)))
    for tgt in (0.05, 0.10, 0.20):
        print(f"\n  matched P(die<=250d) = {tgt:.0%} → annual net "
              f"(mean/p10):")
        for nm, ps in curves.items():
            risk_t = None
            for i in range(1, len(grid)):
                if ps[i - 1] < tgt <= ps[i]:
                    w = (tgt - ps[i - 1]) / (ps[i] - ps[i - 1])
                    risk_t = grid[i - 1] + w * (grid[i] - grid[i - 1])
                    break
            if risk_t is None:
                lab = ("below target at ALL risks (max "
                       f"{max(ps):.2f})" if max(ps) < tgt else
                       "always above target")
                print(f"    {nm:>9}: {lab}")
                RES.setdefault("E2", {})[f"{nm}|{tgt}"] = dict(risk=None)
                continue
            r = run_annual(nm, risk_t)
            RES.setdefault("E2", {})[f"{nm}|{tgt}"] = dict(
                risk=risk_t, mean=float(r["net"].mean()),
                p10=float(np.percentile(r["net"], 10)))
            print(f"    {nm:>9}: risk ${risk_t:>3.0f} → "
                  f"${r['net'].mean():>8,.0f} / "
                  f"${np.percentile(r['net'], 10):>8,.0f}")


def thresholds(nm, risk, cmult, lab, n=2000):
    r = run_annual(nm, risk, cmult, n)
    net = r["net"]
    fp = r["first_pay"]
    kd = r["kday"]
    gaps = []
    for i in range(min(400, n)):
        d = kd[i][kd[i] >= 0]
        if len(d) >= 2:
            gaps.extend(np.diff(np.sort(d)))
    th = {t: float((net >= t).mean()) for t in
          (10000, 25000, 50000, 75000, 100000, 150000)}
    out = dict(mean=float(net.mean()), **q(net), neg=float((net < 0).mean()),
               pay_yr=float(r["n_pay"].mean()),
               deaths_yr=float(r["deaths"].mean()),
               p_first=float((fp >= 0).mean()),
               d_first_med=float(np.median(fp[fp >= 0])),
               d_first_p75=float(np.percentile(fp[fp >= 0], 75)),
               d_first_p90=float(np.percentile(fp[fp >= 0], 90)),
               gap_med=float(np.median(gaps)), thresholds=th)
    RES.setdefault("final", {})[lab] = out
    print(f"\n  [{lab}] {nm} ${int(risk)} C={int(cmult)}R:")
    print(f"    net/yr mean ${out['mean']:,.0f} | p10 ${out['p10']:,.0f} "
          f"p25 ${out['p25']:,.0f} p50 ${out['p50']:,.0f} "
          f"p75 ${out['p75']:,.0f} p90 ${out['p90']:,.0f} | "
          f"P(net<0) {out['neg']:.3f}")
    print(f"    payouts {out['pay_yr']:.1f}/yr | deaths "
          f"{out['deaths_yr']:.2f}/yr | first payout med day "
          f"{out['d_first_med']:.0f} (p75 {out['d_first_p75']:.0f}, "
          f"p90 {out['d_first_p90']:.0f}) | med gap "
          f"{out['gap_med']:.0f}d")
    print("    P(>=$): " + " ".join(f"{k//1000}k:{v:.2f}"
                                    for k, v in th.items()))
    return out


def philosophy():
    print("\n=== PART 25 — risk philosophies (cap8 and raw, C=18R) ===")
    rows = [("conservative b/45", 100.0), ("survival b/35", 128.0),
            ("moderate b/30", 150.0), ("hist-DD b/27", 167.0),
            ("aggressive b/20", 225.0), ("max-extract (raw cap)", 400.0)]
    for nm in ("cap8", "raw"):
        print(f"\n  [{nm}]  (MNQ contracts = risk$/25)")
        print(f"{'philosophy':>22} {'risk$':>6} {'MNQ':>5} {'P1st':>6} "
              f"{'PdieB4':>7} {'E pays':>7} {'E cash/acct':>12} "
              f"{'net/yr':>9} {'p10':>9} {'p90':>9}")
        for lab, risk in rows:
            ra = simulate(ST[nm], IDX[:1200, :750], risk, RULES, pol(risk))
            an = run_annual(nm, risk, n=1200)
            RES.setdefault("phil", {})[f"{nm}|{lab}"] = dict(
                risk=risk, p_first=float((ra["n_pay"] >= 1).mean()),
                p_dieb4=float((ra["dead"] & (ra["n_pay"] == 0)).mean()),
                pays=float(ra["n_pay"].mean()),
                cash=float(ra["cash"].mean()),
                net_yr=float(an["net"].mean()),
                p10=float(np.percentile(an["net"], 10)),
                p90=float(np.percentile(an["net"], 90)))
            d = RES["phil"][f"{nm}|{lab}"]
            print(f"{lab:>22} {int(risk):>6} {risk/25:>5.1f} "
                  f"{d['p_first']:>6.2f} {d['p_dieb4']:>7.3f} "
                  f"{d['pays']:>7.1f} {d['cash']:>12,.0f} "
                  f"{d['net_yr']:>9,.0f} {d['p10']:>9,.0f} "
                  f"{d['p90']:>9,.0f}")


def cap_binding(nm, risk, cmult):
    """Median payout size + share at cap for a config (chronological)."""
    idxc = np.arange(len(SESSIONS))[None, :]
    r = simulate(ST[nm], idxc, risk, RULES, pol(risk, cmult))
    return r


def main() -> int:
    t0 = _time.time()
    stage_E2()
    print("\n=== PART 19/20 — headline configurations, N=2000 ===")
    thresholds("cap8", 167.0, 18.0, "BALANCED cap8 $167")
    thresholds("cap8", 250.0, 18.0, "cap8 $250")
    thresholds("cap8", 400.0, 18.0, "cap8 $400 max")
    thresholds("tap3", 300.0, 18.0, "tap3 $300")
    thresholds("tap3", 400.0, 18.0, "tap3 $400")
    thresholds("tap3cap8", 300.0, 18.0, "tap3cap8 $300")
    thresholds("raw", 250.0, 18.0, "raw $250")
    thresholds("raw", 400.0, 18.0, "raw $400")
    thresholds("cap8", 100.0, 18.0, "survival cap8 $100")
    thresholds("tap3", 150.0, 27.0, "tap3 $150 C=27R safe")
    philosophy()
    out = ROOT / "outputs" / "p12_final.json"
    out.write_text(json.dumps(RES, indent=1, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  [{(_time.time()-t0)/60:.1f} min]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
