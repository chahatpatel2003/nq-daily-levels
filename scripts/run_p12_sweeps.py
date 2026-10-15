#!/usr/bin/env python
"""P12 staged sweeps. Common random numbers: one global block-5 index
matrix reused across every config so comparisons are paired.

Stages:
  A  payout-path comparison (std / consistency / DLL+promo caps)
  B  policy surface: first-payout trigger T1 x retained cushion C
  C  withdrawal fraction test
  D  deep risk grid per stream (per-account + annual metrics)
  E  failure-matched stream comparison
  F  replacement cost/delay sensitivity
  G  dynamic risk & house-money
  H  chronological case studies (Aug-2023 DD, adversarial start)
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from p12_engine import (CONS, STD, K_TRACK, Stream, load_streams,            # noqa: E402
                        make_idx, simulate)

SESSIONS, STREAMS = load_streams()
NS = len(SESSIONS)
IDX = make_idx(NS, 2000, 750, seed=12)
REP = dict(cost=150.0, delay=5)          # base replacement: $150 / 5 days
RES: dict = {}

ST = {k: Stream(v) for k, v in STREAMS.items()}
ST_DLL = {}                              # built per (stream, dll_R) lazily


def st_dll(name, dll_usd, risk):
    key = (name, round(dll_usd / risk, 3))
    if key not in ST_DLL:
        ST_DLL[key] = Stream(STREAMS[name], dll_R=dll_usd / risk)
    return ST_DLL[key]


def q(x, ps=(10, 25, 50, 75, 90)):
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def annual(st, risk, rules, pol, n=800, dll=None, name=None):
    stx = st_dll(name, dll, risk) if dll else st
    r = simulate(stx, IDX[:n, :250], risk, rules, pol, replace=REP)
    net = r["net"]
    return dict(mean=float(net.mean()), **q(net),
                pay_yr=float(r["n_pay"].mean()),
                deaths_yr=float(r["deaths"].mean()),
                p_first=float((r["first_pay"] >= 0).mean()),
                d_first=float(np.median(r["first_pay"][r["first_pay"] >= 0]))
                if (r["first_pay"] >= 0).any() else np.nan,
                cons_block=float(r["cons_block"].mean()))


def acct(st, risk, rules, pol, n=1500, h=750):
    """Per-account (no replacement) lifecycle stats."""
    r = simulate(st, IDX[:n, :h], risk, rules, pol)
    dead = r["dead"]
    npay = r["n_pay"]
    out = dict(
        p_die=float(dead.mean()),
        p_die_before_first=float((dead & (npay == 0)).mean()),
        p_first=float((npay >= 1).mean()),
        d_first=float(np.median(r["first_pay"][r["first_pay"] >= 0]))
        if (npay >= 1).any() else np.nan,
        pay_mean=float(npay.mean()), pay_med=float(np.median(npay)),
        pay_q=q(npay), cash_mean=float(r["cash"].mean()),
        cash_med=float(np.median(r["cash"])), cash_q=q(r["cash"]),
        life_med=float(np.median(r["death_day"][dead])) if dead.any()
        else np.nan,
        p_ge={k: float((npay >= k).mean()) for k in (1, 2, 3, 5, 10)},
        surv_after={}, cash_at={})
    for k in K_TRACK:
        reach = npay >= k
        if reach.any():
            died_after = dead & reach
            out["surv_after"][k] = float(1 - died_after.mean() / reach.mean())
            out["cash_at"][k] = float(np.nanmedian(r["kcash"][reach, k]))
    if dead.any():
        out["cash_at_death_mean"] = float(np.nanmean(r["cash_death1"][dead]))
        out["cash_at_death_med"] = float(np.nanmedian(r["cash_death1"][dead]))
    return out


def stage_A():
    print("\n" + "=" * 78)
    print("STAGE A — payout path (cap8 stream, immediate/max policy, "
          "annual net $ with replacement $150/5d)")
    print(f"{'risk$':>6} {'path':>14} {'mean':>9} {'p50':>9} {'p10':>9} "
          f"{'p90':>9} {'pay/yr':>7} {'die/yr':>7} {'d1st':>5} {'blockd':>7}")
    paths = [
        ("std $5k", STD, None),
        ("cons $6k", CONS, None),
        ("dll std $5k", STD, 3000.0),
        ("dll std $10k", dict(STD, cap=10000.0), 3000.0),
        ("dll cons $12k", dict(CONS, cap=12000.0), 3000.0),
    ]
    pol = dict(T1=0.0, C=0.0)
    for risk in (100.0, 150.0, 167.0, 200.0, 250.0, 300.0):
        for lab, rules, dll in paths:
            a = annual(ST["cap8"], risk, rules, pol, dll=dll, name="cap8")
            RES.setdefault("A", {})[f"{int(risk)}|{lab}"] = a
            print(f"{int(risk):>6} {lab:>14} {a['mean']:>9,.0f} "
                  f"{a['p50']:>9,.0f} {a['p10']:>9,.0f} {a['p90']:>9,.0f} "
                  f"{a['pay_yr']:>7.1f} {a['deaths_yr']:>7.2f} "
                  f"{a['d_first']:>5.0f} {a['cons_block']:>7.1f}")
    # raw-stream reference at two risks
    print("  raw stream reference:")
    for risk in (150.0, 250.0):
        for lab, rules, dll in (paths[0], paths[3]):
            a = annual(ST["raw"], risk, rules, pol, dll=dll, name="raw")
            RES["A"][f"raw{int(risk)}|{lab}"] = a
            print(f"{int(risk):>6} {'raw ' + lab:>14} {a['mean']:>9,.0f} "
                  f"{a['p50']:>9,.0f} {a['p10']:>9,.0f} {a['p90']:>9,.0f} "
                  f"{a['pay_yr']:>7.1f} {a['deaths_yr']:>7.2f}")


def stage_B(rules, dll, lab):
    print("\n" + "=" * 78)
    print(f"STAGE B — policy surface (cap8, {lab}): first-payout trigger T1 "
          "x retained cushion C\ncell = annual net mean/p50/p10 ($k), "
          "die/yr")
    T1S = (0, 3000, 6000, 9000, 15000)
    CS = (0, 1000, 2000, 3000, 4500, 6000, 9000)
    for risk in (150.0, 167.0, 250.0):
        print(f"\n  risk ${int(risk)}:")
        hdr = "  C\\T1 " + "".join(f"{t:>21,}" for t in T1S)
        print(hdr)
        for c in CS:
            cells = []
            for t1 in T1S:
                a = annual(ST["cap8"], risk, rules,
                           dict(T1=float(t1), C=float(c)), dll=dll,
                           name="cap8")
                RES.setdefault("B", {})[f"{int(risk)}|{t1}|{c}"] = a
                cells.append(f"{a['mean']/1e3:5.0f}/{a['p50']/1e3:4.0f}/"
                             f"{a['p10']/1e3:4.0f} {a['deaths_yr']:3.1f}")
            print(f"  {c:>5,} " + " ".join(f"{s:>20}" for s in cells))


def main() -> int:
    t0 = _time.time()
    stage_A()
    print(f"\n[{(_time.time()-t0)/60:.1f} min]")
    # stage B runs on the stage-A winning path — decided by inspection,
    # default standard-with-DLL-promo caps; override via argv
    lab = sys.argv[1] if len(sys.argv) > 1 else "dll_std10k"
    rules, dll = {"std5k": (STD, None),
                  "dll_std10k": (dict(STD, cap=10000.0), 3000.0),
                  "cons6k": (CONS, None)}[lab]
    stage_B(rules, dll, lab)
    out = ROOT / "outputs" / "p12_AB.json"
    out.write_text(json.dumps(RES, indent=1, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  [{(_time.time()-t0)/60:.1f} min]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
