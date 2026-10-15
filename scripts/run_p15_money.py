#!/usr/bin/env python
"""P15 lifecycle economics for the five legal strategies.
Policy (re-derived on legal streams): immediate first payout, 50 % max,
C = 12R cushion, std $5k cap (winner also under DLL+$10k promo),
replacement $150/5d. All dollars via risk$ = $ per 1R (2 NQ = $500)."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from p12_engine import STD, Stream, make_idx, simulate                         # noqa: E402

REP = dict(cost=150.0, delay=5)
CM = 12.0
MARKS = (1000, 2500, 5000, 10000, 25000, 50000, 100000)
RISKS = (100., 150., 200., 250., 300., 350., 400., 450., 500., 550.,
         600., 700., 800., 1000.)
RES: dict = {}

d = pickle.load(open("/Volumes/T9/nq_levels/daily/p15_streams.pkl", "rb"))
ST = {k: Stream(v) for k, v in d["streams"].items()}
NS = {k: len(v) for k, v in d["sessions"].items()}
IDX = {k: make_idx(NS[k], 1500, 750, seed=15) for k in ST}


def pol(risk):
    return dict(T1=0.0, C=CM * risk)


def annual(nm, risk, n=1200, rules=STD, marks=()):
    return simulate(ST[nm], IDX[nm][:n, :250], risk, rules, pol(risk),
                    replace=REP, cash_marks=marks)


def acct(nm, risk, n=1200):
    return simulate(ST[nm], IDX[nm][:n, :750], risk, STD, pol(risk))


def q(x, ps=(10, 25, 50, 75, 90)):
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def main() -> int:
    print("=== PART 1 — everyone at $500 (2 NQ) ===")
    print(f"{'strat':>10} {'net/yr':>8} {'p50':>8} {'p10':>8} {'p90':>8} "
          f"{'pay/yr':>7} {'avg$':>6} {'d1st':>5} {'die/yr':>7} "
          f"{'PdieB4':>7} {'medLife':>8}")
    for nm in ST:
        a = annual(nm, 500.0)
        c = acct(nm, 500.0)
        fp = a["first_pay"]
        dead = c["dead"]
        npay = c["n_pay"]
        life = c["death_day"][dead]
        RES.setdefault("p1", {})[nm] = dict(
            mean=float(a["net"].mean()), **q(a["net"]),
            pay_yr=float(a["n_pay"].mean()),
            avg_pay=float(a["cash"].sum() / max(1, a["n_pay"].sum())),
            d1st=float(np.median(fp[fp >= 0])),
            die_yr=float(a["deaths"].mean()),
            p_dieb4=float((dead & (npay == 0)).mean()),
            med_life=float(np.median(life)) if dead.any() else 750.0,
            pay_before=q(npay), cash_before_med=float(np.median(c["cash"])),
            cash_before_mean=float(c["cash"].mean()),
            p_ge={k: float((npay >= k).mean()) for k in (1, 2, 3, 5, 10)})
        r = RES["p1"][nm]
        print(f"{nm:>10} {r['mean']:>8,.0f} {r["p50"]:>8,.0f} {r["p10"]:>8,.0f} "
              f"{r["p90"]:>8,.0f} {r['pay_yr']:>7.1f} {r['avg_pay']:>6,.0f} "
              f"{r['d1st']:>5.0f} {r['die_yr']:>7.2f} {r['p_dieb4']:>7.3f} "
              f"{r['med_life']:>8.0f}")

    print("\n=== PART 2 — money speed at $500: median sessions to "
          "cumulative withdrawn ===")
    print(f"{'strat':>10} " + "".join(f"{m//1000:>6}k" for m in MARKS))
    for nm in ST:
        a = annual(nm, 500.0, n=1000, marks=MARKS)
        row = []
        for m in MARKS:
            md = a["mark_day"][m]
            hit = md >= 0
            row.append(f"{np.median(md[hit]):>6.0f}" if hit.mean() > 0.5
                       else f"{'--':>6}")
        RES.setdefault("p2", {})[nm] = {str(m): float(np.median(
            a["mark_day"][m][a["mark_day"][m] >= 0]))
            for m in MARKS if (a["mark_day"][m] >= 0).any()}
        print(f"{nm:>10} " + "".join(row))

    print("\n=== PART 3 — matched account-failure sizing ===")
    curves = {}
    for nm in ST:
        ps = []
        for rk in RISKS:
            r = simulate(ST[nm], IDX[nm][:800, :250], rk, STD, pol(rk))
            ps.append(float(r["dead"].mean()))
        curves[nm] = ps
        print(f"  {nm:>10} P(die<=250d): " +
              " ".join(f"{int(k)}:{p:.2f}" for k, p in zip(RISKS, ps)))
    for tgt in (0.01, 0.025, 0.05, 0.10, 0.20):
        print(f"\n  target {tgt:.1%}:")
        for nm, ps in curves.items():
            rk_t = None
            for i in range(1, len(RISKS)):
                if ps[i - 1] < tgt <= ps[i]:
                    w = (tgt - ps[i - 1]) / (ps[i] - ps[i - 1])
                    rk_t = RISKS[i - 1] + w * (RISKS[i] - RISKS[i - 1])
                    break
            if rk_t is None:
                rk_t = RISKS[-1] if max(ps) < tgt else RISKS[0]
            a = annual(nm, rk_t, n=800)
            RES.setdefault("p3", {})[f"{nm}|{tgt}"] = dict(
                risk=rk_t, mean=float(a["net"].mean()),
                p10=float(np.percentile(a["net"], 10)),
                d1st=float(np.median(a["first_pay"][a["first_pay"] >= 0])))
            v = RES["p3"][f"{nm}|{tgt}"]
            print(f"    {nm:>10}: risk ${v['risk']:>4.0f} "
                  f"({v['risk']/25:.0f} MNQ) → ${v['mean']:>8,.0f}/yr "
                  f"p10 ${v['p10']:>8,.0f} d1st {v['d1st']:.0f}")

    print("\n=== PART 4 — optimal risk per strategy (mean/p50/p10 $/yr) ===")
    for nm in ST:
        best = {}
        print(f"  [{nm}]")
        for rk in RISKS:
            a = annual(nm, rk, n=800)
            row = dict(mean=float(a["net"].mean()),
                       p50=float(np.median(a["net"])),
                       p10=float(np.percentile(a["net"], 10)),
                       die=float(a["deaths"].mean()))
            RES.setdefault("p4", {})[f"{nm}|{int(rk)}"] = row
            if rk in (200., 300., 400., 500., 600., 800., 1000.):
                print(f"    ${int(rk):>4}: mean {row['mean']:>8,.0f} "
                      f"p50 {row['p50']:>8,.0f} p10 {row['p10']:>8,.0f} "
                      f"die/yr {row['die']:.2f}")

    print("\n=== PART 15 — chronological replays at $500 ===")
    for nm in ST:
        idx_all = np.arange(NS[nm])[None, :]
        r = simulate(ST[nm], idx_all, 500.0, STD, pol(500.0),
                     replace=REP)
        sess = d["sessions"][nm]
        # per-year cash: rerun truncated at each year end
        marks = {}
        for y in (2022, 2023, 2024, 2025, 2026):
            iy = [i for i, s in enumerate(sess) if int(s[:4]) <= y]
            if not iy:
                continue
            r2 = simulate(ST[nm], np.arange(iy[-1] + 1)[None, :], 500.0,
                          STD, pol(500.0), replace=REP)
            marks[y] = float(r2["cash"][0])
        yr_cash = {y: marks[y] - marks.get(y - 1, 0.0) for y in marks}
        RES.setdefault("p15", {})[nm] = dict(total=float(r["cash"][0]),
                                             deaths=int(r["deaths"][0]),
                                             by_year=yr_cash)
        print(f"  {nm:>10}: total ${r['cash'][0]:>9,.0f} "
              f"deaths {r['deaths'][0]} | " +
              " ".join(f"{y}:{v/1e3:.0f}k" for y, v in yr_cash.items()))

    out = ROOT / "outputs" / "p15_money.json"
    out.write_text(json.dumps(RES, indent=1, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
