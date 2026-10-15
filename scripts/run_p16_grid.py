#!/usr/bin/env python
"""P16 Stage B: Balanced risk optimization on the canonical frozen
ledger. Grid $250-1500; std $5k cap and DLL+promo $10k scenarios;
payout gaps, fresh-account survival, before-death economics, cushion and
timing sweeps, DLL interference, cap saturation, edge degradation."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from p12_engine import STD, Stream, make_idx, simulate                         # noqa: E402

D = pickle.load(open("/Volumes/T9/nq_levels/daily/p16_streams_canonical.pkl",
                     "rb"))
SEQS = D["streams"]["E_hybrid"]
NS = len(SEQS)
ST = Stream(SEQS)
IDX = make_idx(NS, 2000, 750, seed=15)
REP = dict(cost=150.0, delay=5)
RISKS = (250., 300., 350., 400., 450., 500., 550., 600., 650., 700.,
         750., 800., 850., 900., 1000., 1100., 1200., 1300., 1400., 1500.)
PROMO = dict(STD, cap=10000.0)
RES: dict = {}


def pol(risk, cm=12.0, min_take=500.0):
    return dict(T1=0.0, C=cm * risk, min_take=min_take)


def run(risk, rules=STD, st=ST, cm=12.0, min_take=500.0, n=2000, h=250,
        rep=REP):
    return simulate(st, IDX[:n, :h], risk, rules, pol(risk, cm, min_take),
                    replace=rep)


def gapstats(g, h=250):
    if not len(g):
        return {}
    cov = {}
    for w in (20, 40, 60):
        cov[w] = 1.0 - float(np.clip(g - w, 0, None).sum()) / max(
            1, len(g) * np.mean(g))
    return dict(med=float(np.median(g)), p75=float(np.percentile(g, 75)),
                p90=float(np.percentile(g, 90)),
                p95=float(np.percentile(g, 95)),
                p99=float(np.percentile(g, 99)), mx=int(g.max()), cov=cov)


def main() -> int:
    print("=== B5 main grid (std $5k | DLL+promo $10k), C=12R ===")
    print(f"{'risk':>5} {'MNQ':>4} | {'std mean':>9} {'p50':>8} {'p10':>8} "
          f"{'pay/yr':>6} {'gapMed':>6} {'gap90':>6} {'die':>5} | "
          f"{'promo mean':>10} {'p50':>8} {'p10':>8} {'capLost%':>8}")
    st_dll = {}
    for rk in RISKS:
        a = run(rk)
        st_d = Stream(SEQS, dll_R=3000.0 / rk)
        st_dll[rk] = st_d
        p = run(rk, PROMO, st=st_d)
        g = gapstats(a["gaps"])
        pot = p["gross_paid"] + p["cap_lost"]
        RES.setdefault("grid", {})[int(rk)] = dict(
            std_mean=float(a["net"].mean()),
            std_p50=float(np.median(a["net"])),
            std_p10=float(np.percentile(a["net"], 10)),
            std_p90=float(np.percentile(a["net"], 90)),
            pay_yr=float(a["n_pay"].mean()),
            avg_pay=float(a["cash"].sum() / max(1, a["n_pay"].sum())),
            die=float(a["deaths"].mean()), gaps=g,
            promo_mean=float(p["net"].mean()),
            promo_p50=float(np.median(p["net"])),
            promo_p10=float(np.percentile(p["net"], 10)),
            promo_die=float(p["deaths"].mean()),
            cap_lost_share=float(p["cap_lost"].sum() / max(1, pot.sum())),
            cap_lost_std=float(a["cap_lost"].sum()
                               / max(1, (a["gross_paid"]
                                         + a["cap_lost"]).sum())))
        r = RES["grid"][int(rk)]
        print(f"{int(rk):>5} {rk/25:>4.0f} | {r['std_mean']:>9,.0f} "
              f"{r['std_p50']:>8,.0f} {r['std_p10']:>8,.0f} "
              f"{r['pay_yr']:>6.1f} {g['med']:>6.0f} {g['p90']:>6.0f} "
              f"{r['die']:>5.2f} | {r['promo_mean']:>10,.0f} "
              f"{r['promo_p50']:>8,.0f} {r['promo_p10']:>8,.0f} "
              f"{r['cap_lost_share']:>8.1%}")

    print("\n=== B10/B11 fresh-account survival + before-death ===")
    print(f"{'risk':>5} {'bufR':>5} {'s10':>5} {'s20':>5} {'s40':>5} "
          f"{'pay1':>5} {'pay3':>5} {'pay5':>5} {'pay10':>6} "
          f"{'medPayB4':>8} {'medCashB4':>10} {'medLife':>8}")
    for rk in (250., 400., 500., 650., 750., 900., 1100., 1500.):
        c = simulate(ST, IDX[:2000, :750], rk, STD, pol(rk))
        dd_ = c["death_day"]
        alive = lambda k: float(((dd_ < 0) | (dd_ > k)).mean())
        npay = c["n_pay"]; dead = c["dead"]
        RES.setdefault("surv", {})[int(rk)] = dict(
            bufR=4500 / rk, s10=alive(10), s20=alive(20), s40=alive(40),
            p1=float((npay >= 1).mean()), p3=float((npay >= 3).mean()),
            p5=float((npay >= 5).mean()), p10=float((npay >= 10).mean()),
            p20=float((npay >= 20).mean()), p30=float((npay >= 30).mean()),
            med_pay_b4=float(np.median(npay[dead])) if dead.any() else None,
            med_cash_b4=float(np.median(c["cash"][dead])) if dead.any()
            else None,
            med_life=float(np.median(dd_[dead])) if dead.any() else None)
        v = RES["surv"][int(rk)]
        print(f"{int(rk):>5} {v['bufR']:>5.1f} {v['s10']:>5.2f} "
              f"{v['s20']:>5.2f} {v['s40']:>5.2f} {v['p1']:>5.2f} "
              f"{v['p3']:>5.2f} {v['p5']:>5.2f} {v['p10']:>6.2f} "
              f"{v['med_pay_b4'] if v['med_pay_b4'] is not None else -1:>8.0f} "
              f"{v['med_cash_b4'] if v['med_cash_b4'] is not None else -1:>10,.0f} "
              f"{v['med_life'] if v['med_life'] is not None else -1:>8.0f}")

    print("\n=== B13 cushion sweep (std cap) ===")
    for rk in (500., 750., 1000.):
        row = []
        for cm in (8., 10., 12., 15., 18., 24.):
            a = run(rk, cm=cm, n=1200)
            g = a["gaps"]
            row.append(f"{cm:.0f}R:{a['net'].mean()/1e3:.1f}k/"
                       f"{a['deaths'].mean():.2f}/g{np.median(g):.0f}")
            RES.setdefault("cush", {})[f"{int(rk)}|{int(cm)}"] = dict(
                mean=float(a["net"].mean()), die=float(a["deaths"].mean()))
        print(f"  ${int(rk)}: " + " | ".join(row))

    print("\n=== B14 payout timing (min meaningful payout) ===")
    for rk in (500., 750.):
        row = []
        for mt_lab, mt in (("imm", 500.), ("3R", 3 * rk), ("6R", 6 * rk),
                           ("12R", 12 * rk)):
            a = run(rk, min_take=mt, n=1200)
            row.append(f"{mt_lab}:{a['net'].mean()/1e3:.1f}k"
                       f"/g{np.median(a['gaps']):.0f}")
        print(f"  ${int(rk)}: " + " | ".join(row))

    print("\n=== B16 DLL interference by risk ===")
    for rk in (500., 750., 1000., 1250., 1500.):
        dll_R = 3000.0 / rk
        hit = skip = skipR = 0
        for a in SEQS:
            if not len(a):
                continue
            cum = np.cumsum(a)
            k = np.flatnonzero(cum <= -dll_R)
            if k.size:
                hit += 1
                skip += len(a) - (k[0] + 1)
                skipR += float(a[k[0] + 1:].sum())
        print(f"  ${int(rk):>5} (DLL={dll_R:.1f}R): days hit {hit:>3} "
              f"({hit/NS:.1%}), trades cut {skip:>4}, R forgone "
              f"{skipR:+.1f}")
        RES.setdefault("dll", {})[int(rk)] = dict(hit=hit, skip=skip,
                                                  skipR=skipR)

    print("\n=== B19 edge degradation (recommended-zone risks) ===")
    rng = np.random.default_rng(77)
    flat = np.concatenate([a for a in SEQS if len(a)])
    W = float(flat[flat > 0].mean()); L = float(-flat[flat <= 0].mean())
    wr = float((flat > 0).mean()); exp0 = float(flat.mean())
    for frac_ in (0.8, 0.6, 0.4):
        wr_t = (frac_ * exp0 + L) / (W + L)
        p_flip = (wr - wr_t) / wr
        seqs_d = []
        for a in SEQS:
            b = a.copy()
            wmask = b > 0
            flips = rng.random(len(b)) < p_flip
            b[wmask & flips] = -1.08
            seqs_d.append(b)
        st_d = Stream(seqs_d)
        for rk in (500., 750.):
            a = simulate(st_d, IDX[:1500, :250], rk, STD, pol(rk),
                         replace=REP)
            print(f"  {frac_:.0%} edge ${int(rk)}: mean "
                  f"${a['net'].mean():,.0f} p10 "
                  f"${np.percentile(a['net'],10):,.0f} deaths "
                  f"{a['deaths'].mean():.2f} gapMed "
                  f"{np.median(a['gaps']) if len(a['gaps']) else -1:.0f}")
            RES.setdefault("degr", {})[f"{frac_}|{int(rk)}"] = dict(
                mean=float(a["net"].mean()),
                die=float(a["deaths"].mean()))
    # slippage stress +0.5pt = 0.04R/trade
    seqs_s = [a - 0.04 for a in SEQS]
    st_s = Stream(seqs_s)
    a = simulate(st_s, IDX[:1500, :250], 750.0, STD, pol(750.0),
                 replace=REP)
    print(f"  +0.5pt slip $750: mean ${a['net'].mean():,.0f} "
          f"deaths {a['deaths'].mean():.2f}")

    out = ROOT / "outputs" / "p16_grid.json"
    out.write_text(json.dumps(RES, indent=1, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
