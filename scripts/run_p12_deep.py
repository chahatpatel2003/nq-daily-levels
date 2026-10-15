#!/usr/bin/env python
"""P12 deep stages: C fraction test, D risk grid per stream (annual +
per-account), E failure-matched streams, F replacement sensitivity,
G dynamic risk, H chronological case studies, sensitivities."""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from p12_engine import (STD, Stream, load_streams, make_idx,                  # noqa: E402
                        simulate)
from run_p12_sweeps import ST, IDX, REP, annual, acct, q                      # noqa: E402

RULES = dict(STD, cap=10000.0)           # standard + promo cap (DLL free)
CMULT = 18.0                             # retained cushion in R (stage-B)
RES: dict = {}
SESSIONS, STREAMS = load_streams()
NS = len(SESSIONS)

RISKS = (50., 75., 100., 125., 150., 167., 175., 185., 200., 225., 250.,
         300., 350., 400., 500., 600.)


def pol_for(risk, cmult=CMULT):
    return dict(T1=0.0, C=float(cmult) * risk)


def stage_C():
    print("\n=== STAGE C — withdrawal fraction (cap8, $167, C=18R) ===")
    print(f"{'frac':>5} {'mean$':>9} {'p50':>9} {'p10':>9} {'pay/yr':>7} "
          f"{'die/yr':>7}")
    for f in (0.25, 0.33, 0.40, 0.50):
        a = annual(ST["cap8"], 167.0, RULES, dict(pol_for(167.0), frac=f))
        RES.setdefault("C", {})[str(f)] = a
        print(f"{f:>5.2f} {a['mean']:>9,.0f} {a['p50']:>9,.0f} "
              f"{a['p10']:>9,.0f} {a['pay_yr']:>7.1f} "
              f"{a['deaths_yr']:>7.2f}")


def stage_D():
    print("\n=== STAGE D — risk grid per stream ===")
    print("annual: net$ (mean/p50/p10/p90) with replacement $150/5d, "
          "C=18R | account: no-replacement 750d")
    for nm in ("raw", "cap8", "cap6", "cap4", "tap3", "tap3cap8"):
        print(f"\n  [{nm}]")
        print(f"{'risk$':>6} {'mean':>8} {'p50':>8} {'p10':>8} {'p90':>8} "
              f"{'py/yr':>6} {'dy/yr':>6} | {'P1st':>5} {'Pdie':>5} "
              f"{'PdieB4':>6} {'medPay':>6} {'medCash':>8} {'P>=10':>5} "
              f"{'S|1':>5} {'S|5':>5} {'medLife':>7}")
        for risk in RISKS:
            a = annual(ST[nm], risk, RULES, pol_for(risk))
            c = acct(ST[nm], risk, RULES, pol_for(risk))
            RES.setdefault("D", {})[f"{nm}|{int(risk)}"] = dict(a=a, c=c)
            print(f"{int(risk):>6} {a['mean']:>8,.0f} {a['p50']:>8,.0f} "
                  f"{a['p10']:>8,.0f} {a['p90']:>8,.0f} "
                  f"{a['pay_yr']:>6.1f} {a['deaths_yr']:>6.2f} | "
                  f"{c['p_first']:>5.2f} {c['p_die']:>5.2f} "
                  f"{c['p_die_before_first']:>6.3f} {c['pay_med']:>6.0f} "
                  f"{c['cash_med']:>8,.0f} {c['p_ge'][10]:>5.2f} "
                  f"{c['surv_after'].get(1, float('nan')):>5.2f} "
                  f"{c['surv_after'].get(5, float('nan')):>5.2f} "
                  f"{c['life_med']:>7.0f}")


def stage_D2():
    print("\n=== D2 — cushion multiplier x risk (cap8) ===")
    print(f"{'risk$':>6} " + "".join(f"{m:>21}" for m in
                                     ("C=0", "C=12R", "C=18R", "C=27R",
                                      "C=35R")))
    for risk in (150.0, 250.0, 400.0):
        cells = []
        for m in (0.0, 12.0, 18.0, 27.0, 35.0):
            a = annual(ST["cap8"], risk, RULES, pol_for(risk, m))
            RES.setdefault("D2", {})[f"{int(risk)}|{int(m)}"] = a
            cells.append(f"{a['mean']/1e3:5.1f}k/{a['p10']/1e3:5.1f}k "
                         f"{a['deaths_yr']:4.2f}")
        print(f"{int(risk):>6} " + " ".join(f"{s:>20}" for s in cells))
    print("\n  max-strip (C=0) per-account, cap8 (the user /27 layout):")
    print(f"{'risk$':>6} {'P1st':>6} {'PdieB4':>7} {'medPay':>7} "
          f"{'meanPay':>8} {'medCash':>9} {'meanCash':>9} "
          f"{'P>=1':>5} {'P>=3':>5} {'P>=5':>5} {'P>=10':>6}")
    for risk in (150., 160., 167., 175., 185., 200.):
        c = acct(ST["cap8"], risk, RULES, dict(T1=0.0, C=0.0))
        RES.setdefault("D2strip", {})[str(int(risk))] = c
        print(f"{int(risk):>6} {c['p_first']:>6.3f} "
              f"{c['p_die_before_first']:>7.3f} {c['pay_med']:>7.0f} "
              f"{c['pay_mean']:>8.1f} {c['cash_med']:>9,.0f} "
              f"{c['cash_mean']:>9,.0f} {c['p_ge'][1]:>5.2f} "
              f"{c['p_ge'][3]:>5.2f} {c['p_ge'][5]:>5.2f} "
              f"{c['p_ge'][10]:>6.2f}")


def stage_E():
    print("\n=== STAGE E — failure-matched stream comparison ===")
    print("annual net at the risk giving matched P(account death within "
          "250d, no replacement, C=18R policy)")
    targets = (0.10, 0.25, 0.50)
    grid = {}
    for nm in ("raw", "cap8", "tap3", "tap3cap8"):
        ps, ann = [], []
        for risk in RISKS:
            d = RES["D"].get(f"{nm}|{int(risk)}")
            if d is None:
                continue
            r = simulate(ST[nm], IDX[:1500, :250], risk, RULES,
                         pol_for(risk))
            ps.append((risk, float(r["dead"].mean()),
                       RES["D"][f"{nm}|{int(risk)}"]["a"]["mean"]))
        grid[nm] = ps
    for tgt in targets:
        print(f"\n  P(die<=250d) = {tgt:.0%}:")
        for nm, ps in grid.items():
            arr = np.array(ps)
            if arr[:, 1].max() < tgt:
                print(f"    {nm:>9}: unreachable (max {arr[:,1].max():.2f})")
                continue
            i = int(np.searchsorted(arr[:, 1], tgt))
            i = min(max(i, 1), len(arr) - 1)
            r0, r1 = arr[i - 1], arr[i]
            w = (tgt - r0[1]) / (r1[1] - r0[1] + 1e-12)
            risk_t = r0[0] + w * (r1[0] - r0[0])
            net_t = r0[2] + w * (r1[2] - r0[2])
            RES.setdefault("E", {})[f"{nm}|{tgt}"] = dict(risk=risk_t,
                                                         net=net_t)
            print(f"    {nm:>9}: risk ${risk_t:.0f} → net ${net_t:,.0f}/yr")


def stage_F():
    print("\n=== STAGE F — replacement cost x delay (cap8, C=18R) ===")
    for risk in (150.0, 250.0, 400.0):
        print(f"\n  risk ${int(risk)} (mean net $k/yr):")
        print("  cost\\delay " + "".join(f"{d:>8}d" for d in
                                         (0, 2, 5, 10, 20)))
        for cost in (0, 100, 250, 500, 750, 1000):
            row = []
            for delay in (0, 2, 5, 10, 20):
                r = simulate(ST["cap8"], IDX[:600, :250], risk, RULES,
                             pol_for(risk),
                             replace=dict(cost=float(cost), delay=delay))
                RES.setdefault("F", {})[f"{int(risk)}|{cost}|{delay}"] = \
                    float(r["net"].mean())
                row.append(r["net"].mean() / 1e3)
            print(f"  {cost:>6}$    " + "".join(f"{v:>9.1f}" for v in row))


def stage_G():
    print("\n=== STAGE G — dynamic risk / house money (cap8, C=$3k) ===")
    print(f"{'policy':>26} {'mean$':>9} {'p50':>9} {'p10':>9} {'die/yr':>7}")
    base_pol = dict(T1=0.0, C=3000.0)
    tests = [("const $167", 167.0, None, None),
             ("const $250", 250.0, None, None),
             ("$167→$250 after pay1", 167.0, 250.0, ("pay", 1)),
             ("$167→$250 after pay3", 167.0, 250.0, ("pay", 3)),
             ("$167→$250 after $5k", 167.0, 250.0, ("cash", 5000.0)),
             ("$167→$400 after $10k", 167.0, 400.0, ("cash", 10000.0)),
             ("$167→$400 after $20k", 167.0, 400.0, ("cash", 20000.0)),
             ("$100→$250 after pay1", 100.0, 250.0, ("pay", 1)),
             ("$250→$150 after pay3", 250.0, 150.0, ("pay", 3)),
             ("const $400", 400.0, None, None),
             ("$167→$600 after $20k", 167.0, 600.0, ("cash", 20000.0))]
    for lab, r1, r2, sw in tests:
        pol = dict(base_pol)
        if r2 is not None:
            pol.update(risk2=r2, switch=sw)
        r = simulate(ST["cap8"], IDX[:1000, :250], r1, RULES, pol,
                     replace=REP)
        net = r["net"]
        RES.setdefault("G", {})[lab] = dict(mean=float(net.mean()), **q(net))
        print(f"{lab:>26} {net.mean():>9,.0f} "
              f"{np.median(net):>9,.0f} {np.percentile(net,10):>9,.0f} "
              f"{r['deaths'].mean():>7.2f}")


def stage_H():
    print("\n=== STAGE H — chronological case studies ===")
    idx_all = np.arange(NS)[None, :]
    i_dd = next(i for i, s in enumerate(SESSIONS) if str(s) == "2023-08-04")
    for nm in ("cap8", "raw"):
        for risk in (150.0, 167.0, 250.0):
            for cm, cl in ((0.0, "C=0"), (18.0, "C=18R")):
                r = simulate(ST[nm], idx_all, risk, RULES,
                             pol_for(risk, cm))
                kd = r["kday"][0]
                kc = r["kcash"][0]
                pays_before = int((kd[(kd >= 0)] < i_dd).sum()) \
                    if (kd >= 0).any() else 0
                # cash banked by DD start: rerun truncated
                r2 = simulate(ST[nm], idx_all[:, :i_dd], risk, RULES,
                              pol_for(risk, cm))
                dd_dead = bool(r["dead"][0]) and i_dd <= r["death_day"][0] \
                    <= i_dd + 45
                RES.setdefault("H", {})[f"{nm}|{int(risk)}|{cl}"] = dict(
                    cash_by_dd=float(r2["cash"][0]),
                    npay_by_dd=int(r2["n_pay"][0]),
                    died_in_dd=dd_dead, death_day=int(r["death_day"][0]),
                    total_cash=float(r["cash"][0]),
                    total_pay=int(r["n_pay"][0]))
                dd_lab = ("DIED in Aug-Sep23" if dd_dead else
                          (f"died day {r['death_day'][0]}"
                           if r["dead"][0] else "survived 22-26"))
                print(f"  {nm:>5} ${int(risk)} {cl:>6}: by 2023-08-04 → "
                      f"{r2['n_pay'][0]:>2} payouts ${r2['cash'][0]:>9,.0f}"
                      f" | {dd_lab} | 22-26 total {r['n_pay'][0]} pays "
                      f"${r['cash'][0]:,.0f}")
    print("\n  adversarial starts (cap8, $167):")
    for start, lab in ((i_dd, "2023-08-04 (DD eve)"),):
        for cm, cl in ((0.0, "C=0"), (18.0, "C=18R")):
            idx = np.arange(start, NS)[None, :]
            r = simulate(ST["cap8"], idx, 167.0, RULES, pol_for(167.0, cm))
            print(f"    start {lab} {cl}: "
                  f"{'died day ' + str(r['death_day'][0]) if r['dead'][0] else 'survived'}"
                  f", payouts {r['n_pay'][0]}, cash ${r['cash'][0]:,.0f}")
            RES.setdefault("Hadv", {})[f"{lab}|{cl}"] = dict(
                dead=bool(r["dead"][0]), cash=float(r["cash"][0]))


def sensitivities():
    print("\n=== sensitivities (cap8, $167, C=18R) ===")
    a0 = annual(ST["cap8"], 167.0, RULES, pol_for(167.0))
    r = simulate(ST["cap8"], IDX[:800, :250], 167.0, RULES, pol_for(167.0),
                 replace=REP, adverse=True)
    idx20 = make_idx(NS, 800, 250, block=20, seed=13)
    r20 = simulate(ST["cap8"], idx20, 167.0, RULES, pol_for(167.0),
                   replace=REP)
    nosplit = dict(RULES, split=1.0)
    rns = simulate(ST["cap8"], IDX[:800, :250], 167.0, nosplit,
                   pol_for(167.0), replace=REP)
    print(f"  base:            mean ${a0['mean']:,.0f} p10 ${a0['p10']:,.0f}"
          f" die/yr {a0['deaths_yr']:.2f}")
    print(f"  full-1R adverse: mean ${r['net'].mean():,.0f} "
          f"p10 ${np.percentile(r['net'],10):,.0f} "
          f"die/yr {r['deaths'].mean():.2f}")
    print(f"  block-20:        mean ${r20['net'].mean():,.0f} "
          f"p10 ${np.percentile(r20['net'],10):,.0f} "
          f"die/yr {r20['deaths'].mean():.2f}")
    print(f"  no split (100%): mean ${rns['net'].mean():,.0f}")
    RES["sens"] = dict(adverse=float(r["net"].mean()),
                       block20=float(r20["net"].mean()),
                       nosplit=float(rns["net"].mean()))


def main() -> int:
    t0 = _time.time()
    stage_C(); stage_D(); stage_D2(); stage_E(); stage_F(); stage_G()
    stage_H(); sensitivities()
    out = ROOT / "outputs" / "p12_deep.json"
    out.write_text(json.dumps(RES, indent=1, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  [{(_time.time()-t0)/60:.1f} min]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
