#!/usr/bin/env python
"""P13 audit battery on the canonical F4+tap<=3 ledger (parts 1-11,
14-cost, 15, 16, 17 + MC). Audit only — no strategy changes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_p10_eval import portfolio                                             # noqa: E402
from run_p10_stage1 import load_trades, trade_pnl                              # noqa: E402

LED = "/Volumes/T9/nq_levels/daily/p13_ledger.parquet"
RES: dict = {}
SEGN = {0: "asia", 1: "london", 2: "premkt", 3: "ny-am", 4: "ny-pm"}


def maxdd(x):
    eq = np.cumsum(x)
    run = np.maximum.accumulate(eq) - eq
    return float(run.max())


def dd_episodes(day_r):
    """(depth, peak_i, trough_i, rec_i or -1) episodes from day sums."""
    eq = np.concatenate(([0.0], np.cumsum(day_r)))
    eps, peak, trough, ti = [], 0, 0.0, 0
    for i in range(1, len(eq)):
        if eq[i] >= eq[peak]:
            if trough > 1e-9:
                eps.append((trough, peak, ti, i))
            peak, trough = i, 0.0
        else:
            d = eq[peak] - eq[i]
            if d > trough:
                trough, ti = d, i
    if trough > 1e-9:
        eps.append((trough, peak, ti, -1))
    return eps


def streaks(r):
    out, run = [], 0
    for v in r:
        if v <= 0:
            run += 1
        elif run:
            out.append(run); run = 0
    if run:
        out.append(run)
    return np.array(out)


def block(led, label):
    r = led["net_R"].to_numpy()
    w, l = r[r > 0], r[r <= 0]
    if not len(r):
        return {}
    d = dict(n=len(r), wr=float((r > 0).mean()),
             expR=float(r.mean()), pf=float(w.sum() / -l.sum()),
             totR=float(r.sum()), dd=maxdd(r),
             mstk=int(streaks(r).max()) if len(streaks(r)) else 0)
    print(f"  {label:>12}: n {d['n']:5,} WR {d['wr']:.3f} "
          f"expR {d['expR']:+.3f} totR {d['totR']:7.1f} PF {d['pf']:.2f} "
          f"DD {d['dd']:5.1f} streak {d['mstk']}")
    return d


def main() -> int:
    led = pd.read_parquet(LED)
    r = led["net_R"].to_numpy()
    w, l = r[r > 0], r[r <= 0]
    sess_all = sorted(pd.read_parquet(
        "/Volumes/T9/nq_levels/daily/p12_streams.pkl".replace(
            "p12_streams.pkl", "p11_stream.parquet"))["session"].astype(str)
        .unique())
    # full session axis 2022+ (incl. zero-trade days)
    t_all = load_trades()
    sessions = sorted(t_all[t_all["year"] >= 2022]["session"]
                      .astype(str).unique())

    print("=== PART 1 — trade-level ===")
    gw = led["gross_pnl_pts"][led["gross_pnl_pts"] > 0]
    d = dict(
        n=len(r), wins=int((r > 0).sum()), losses=int((r <= 0).sum()),
        be=int((np.abs(led["net_pnl_pts"]) < 0.005).sum()),
        wr=float((r > 0).mean()),
        avg_w_R=float(w.mean()), med_w_R=float(np.median(w)),
        avg_l_R=float(l.mean()), med_l_R=float(np.median(l)),
        avg_w_pts=float(w.mean() * 12.5), avg_l_pts=float(l.mean() * 12.5),
        gross_expR=float(led["gross_R"].mean()), net_expR=float(r.mean()),
        exp_pts=float(led["net_pnl_pts"].mean()),
        pf=float(w.sum() / -l.sum()),
        gross_profit_R=float(w.sum()), gross_loss_R=float(l.sum()),
        payoff=float(w.mean() / -l.mean()),
        max_w=float(r.max()), max_l=float(r.min()))
    RES["p1"] = d
    for k, v in d.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")
    oc = led["outcome"].value_counts()
    print(f"  outcomes: {oc.to_dict()}")
    sc = led[led["outcome"] == "session_close"]
    print(f"  session_close mean R {sc['net_R'].mean():+.2f}")

    print("\n=== PART 2 — year by year "
          "(2022-23 SELECTION era | 2024-26 frozen OOS for F4 params; "
          "tap<=3 cutoff chosen on pooled 22-26 — see report) ===")
    yb = {}
    for y, g in led.groupby("year"):
        rr = g["net_R"].to_numpy()
        ww, ll = rr[rr > 0], rr[rr <= 0]
        days = pd.to_datetime(g["session"]).dt.date
        dsum = g.groupby("session")["net_R"].sum()
        mon = g.assign(m=g["session"].str[:7]).groupby("m")["net_R"].sum()
        ns = len([s for s in sessions if s.startswith(str(y))])
        stk = streaks(rr)
        wstk, runw = [], 0
        for v in rr:
            if v > 0:
                runw += 1
            elif runw:
                wstk.append(runw); runw = 0
        if runw:
            wstk.append(runw)
        yb[y] = dict(n=len(rr), tpd=len(rr) / ns, wr=float((rr > 0).mean()),
                     expR=float(rr.mean()), totR=float(rr.sum()),
                     pf=float(ww.sum() / -ll.sum()), dd=maxdd(rr),
                     lstk=int(stk.max()), wstk=int(max(wstk)),
                     worst_d=float(dsum.min()), best_d=float(dsum.max()),
                     pos_m=float((mon > 0).mean()))
        b = yb[y]
        print(f"  {y}: n {b['n']:4} tpd {b['tpd']:4.2f} WR {b['wr']:.3f} "
              f"expR {b['expR']:+.3f} totR {b['totR']:6.1f} PF {b['pf']:4.2f}"
              f" DD {b['dd']:4.1f} lstk {b['lstk']:2} wstk {b['wstk']:2} "
              f"day[{b['worst_d']:+.1f},{b['best_d']:+.1f}] "
              f"posM {b['pos_m']:.2f}")
    RES["p2"] = yb

    print("\n=== PART 3 — frozen-OOS only (2024-2026) ===")
    RES["p3"] = block(led[led["year"] >= 2024], "OOS 24-26")
    oos = led[led["year"] >= 2024]
    mon = oos.assign(m=oos["session"].str[:7]).groupby("m")["net_R"].sum()
    ns_oos = len([s for s in sessions if s[:4] in ("2024", "2025", "2026")])
    print(f"  OOS tpd {len(oos)/ns_oos:.2f} | posM {(mon>0).mean():.3f} "
          f"({int((mon>0).sum())}/{len(mon)})")
    RES["p3"]["tpd"] = len(oos) / ns_oos
    RES["p3"]["posm"] = float((mon > 0).mean())

    print("\n=== PART 4 — tap breakdown ===")
    print("  executed ledger by tap:")
    for k, g in led.groupby("tap_no"):
        RES.setdefault("p4_exec", {})[int(k)] = block(g, f"tap {k}")
    print("  gated FIRE level (no portfolio skip), taps 1..6:")
    t = t_all[t_all["year"] >= 2022]
    fires = t[(t["src"] == "r4") & (t["entry"] == "E1d") & t["filled"]
              & (t["p_both60"] >= 0.85)]
    for k in (1, 2, 3, 4, 5, 6):
        g = fires[fires["tap_no"] == k] if k < 6 else \
            fires[fires["tap_no"] >= 6]
        if not len(g):
            continue
        pnl, S, wn, ls, ses = trade_pnl(g, "ts1_2", 12.5, 30.0, 1.0)
        rr = pnl / 12.5
        RES.setdefault("p4_fire", {})[int(k)] = dict(
            n=len(rr), wr=float((rr > 0).mean()), expR=float(rr.mean()),
            pf=float(rr[rr > 0].sum() / -rr[rr <= 0].sum()))
        dd = RES["p4_fire"][int(k)]
        print(f"    tap {'6+' if k == 6 else k}: n {dd['n']:5,} "
              f"WR {dd['wr']:.3f} expR {dd['expR']:+.3f} PF {dd['pf']:.2f}")
    print("  cumulative executed streams (portfolio):")
    for kmax in (1, 2, 3, 4):
        pf_k = portfolio(t[t["tap_no"] <= kmax], "E1d", ("r4",), 0.85,
                         12.5, 30.0)
        pf_k["net_R"] = pf_k["pnl"] / 12.5
        RES.setdefault("p4_cum", {})[kmax] = block(pf_k, f"tap<={kmax}")

    print("\n=== PART 5 — losing streaks ===")
    stk = streaks(r)
    yrs = len(sessions) / 250
    for k in (3, 4, 5, 6, 7, 8, 10):
        c = int((stk >= k).sum())
        RES.setdefault("p5", {})[k] = c / yrs
        print(f"  >={k:2}: {c:3} times ({c/yrs:5.1f}/yr)")
    mx = int(stk.max())
    run, best_i, cur_start = 0, None, 0
    for i, v in enumerate(r):
        if v <= 0:
            if run == 0:
                cur_start = i
            run += 1
            if run == mx:
                best_i = (cur_start, i)
        else:
            run = 0
    a, b = best_i
    seg = led.iloc[a:b + 1]
    print(f"  MAX streak {mx}: {seg.iloc[0]['session']} "
          f"(entry_ts {seg.iloc[0]['entry_ts']:.0f}) → "
          f"{seg.iloc[-1]['session']} | sessions "
          f"{seg['session'].nunique()} | R {seg['net_R'].sum():.1f}")
    RES["p5"]["max"] = dict(k=mx, s0=str(seg.iloc[0]["session"]),
                            s1=str(seg.iloc[-1]["session"]),
                            nsess=int(seg["session"].nunique()),
                            R=float(seg["net_R"].sum()))

    print("\n=== PART 6 — drawdown profile ===")
    dsum = led.groupby("session")["net_R"].sum().reindex(sessions).fillna(0)
    day_r = dsum.to_numpy()
    print(f"  max DD: trade-gran {maxdd(r):.1f}R | day-gran "
          f"{maxdd(day_r):.1f}R")
    eps = dd_episodes(day_r)
    depths = np.array([e[0] for e in eps])
    big = max(eps, key=lambda e: e[0])
    s_ax = list(dsum.index)
    print(f"  MAX episode {big[0]:.1f}R: peak {s_ax[big[1]-1]} trough "
          f"{s_ax[big[2]-1]} rec "
          f"{s_ax[big[3]-1] if big[3] > 0 else 'OPEN'} | to-trough "
          f"{big[2]-big[1]} sess | to-rec "
          f"{(big[3]-big[1]) if big[3] > 0 else -1} sess")
    for thr in (3, 5, 7.5, 10, 12, 14, 15, 20):
        c = int((depths >= thr).sum())
        RES.setdefault("p6", {})[str(thr)] = c / yrs
        print(f"  >={thr:4}R: {c:3} ({c/yrs:5.1f}/yr)")
    print(f"  episode depth med {np.median(depths):.1f} p75 "
          f"{np.percentile(depths,75):.1f} p90 {np.percentile(depths,90):.1f}"
          f" p95 {np.percentile(depths,95):.1f}")
    RES["p6"]["max_trade"] = maxdd(r)
    RES["p6"]["max_day"] = maxdd(day_r)
    RES["p6"]["big"] = dict(depth=float(big[0]), peak=s_ax[big[1]-1],
                            trough=s_ax[big[2]-1],
                            rec=s_ax[big[3]-1] if big[3] > 0 else "open",
                            to_trough=int(big[2]-big[1]),
                            to_rec=int(big[3]-big[1]) if big[3] > 0 else -1)

    print("\n=== PART 7 — MC block bootstrap (250d, block-5, 4000 sims) ===")
    rng = np.random.default_rng(13)
    n = len(day_r)
    dds = []
    for _ in range(4000):
        out = []
        while len(out) < 250:
            s0 = rng.integers(0, n - 5)
            out.extend(day_r[s0:s0 + 5])
        dds.append(maxdd(np.array(out[:250])))
    dds = np.array(dds)
    qs = {p: float(np.percentile(dds, p)) for p in
          (50, 75, 90, 95, 97.5, 99, 99.5)}
    RES["p7"] = dict(q=qs, exceed={x: float((dds >= x).mean())
                                   for x in (10, 12, 14, 16, 18, 20, 25, 30)})
    print("  " + " ".join(f"p{p}:{v:.1f}" for p, v in qs.items()))
    print("  P(DD>=X): " + " ".join(f"{x}R:{RES['p7']['exceed'][x]:.3f}"
                                    for x in (10, 12, 14, 16, 18, 20, 25, 30)))

    print("\n=== PART 8 — daily ===")
    act = dsum[dsum.index.isin(led["session"])]
    ntr = led.groupby("session").size().reindex(sessions).fillna(0)
    RES["p8"] = dict(
        mean=float(day_r.mean()), med=float(np.median(day_r)),
        q={p: float(np.percentile(day_r, p)) for p in (10, 25, 75, 90)},
        best=float(day_r.max()), worst=float(day_r.min()),
        no_trade=float((ntr == 0).mean()),
        pos=float((day_r > 0).mean()), neg=float((day_r < 0).mean()),
        tpd_mean=float(ntr.mean()), tpd_med=float(ntr.median()),
        hist={str(k): float((ntr == k).mean()) for k in range(8)}
        | {"8+": float((ntr >= 8).mean())})
    p8 = RES["p8"]
    print(f"  R/day mean {p8['mean']:+.2f} med {p8['med']:+.2f} "
          f"p10 {p8['q'][10]:+.1f} p25 {p8['q'][25]:+.1f} "
          f"p75 {p8['q'][75]:+.1f} p90 {p8['q'][90]:+.1f} | "
          f"best {p8['best']:+.1f} worst {p8['worst']:+.1f}")
    print(f"  days: no-trade {p8['no_trade']:.1%} pos {p8['pos']:.1%} "
          f"neg {p8['neg']:.1%} | tpd mean {p8['tpd_mean']:.2f} "
          f"med {p8['tpd_med']:.0f}")
    print("  count hist: " + " ".join(f"{k}:{v:.1%}"
                                      for k, v in p8["hist"].items()))

    print("\n=== PART 9 — monthly ===")
    msum = led.assign(m=led["session"].str[:7]).groupby("m")["net_R"]
    mR = msum.sum()
    all_m = sorted({s[:7] for s in sessions})
    mR = mR.reindex(all_m).fillna(0.0)
    neg = mR[mR < 0]
    runs, cur = [], 0
    for v in mR.to_numpy():
        cur = cur + 1 if v < 0 else 0
        runs.append(cur)
    RES["p9"] = dict(n=len(mR), pos=int((mR > 0).sum()),
                     neg=int((mR < 0).sum()), flat=int((mR == 0).sum()),
                     pct=float((mR > 0).mean()), avg=float(mR.mean()),
                     med=float(mR.median()), best=float(mR.max()),
                     worst=float(mR.min()), maxconsecneg=int(max(runs)))
    p9 = RES["p9"]
    print(f"  months {p9['n']} pos {p9['pos']} neg {p9['neg']} "
          f"({p9['pct']:.1%}) | avg {p9['avg']:+.1f} med {p9['med']:+.1f} "
          f"best {p9['best']:+.1f} worst {p9['worst']:+.1f} | max consec "
          f"neg {p9['maxconsecneg']}")
    print("  negative months:")
    RES["p9"]["neg_detail"] = {}
    for m, v in neg.sort_values().items():
        g = led[led["session"].str.startswith(m)]
        rr = g["net_R"].to_numpy()
        RES["p9"]["neg_detail"][m] = dict(R=float(v), n=len(g),
                                          wr=float((rr > 0).mean()),
                                          dd=maxdd(rr))
        print(f"    {m}: {v:+6.1f}R n {len(g):3} WR {(rr>0).mean():.2f} "
              f"DD {maxdd(rr):4.1f}")

    print("\n=== PART 10 — sessions ===")
    for sgi, g in led.groupby("seg"):
        nm = SEGN.get(int(sgi), str(sgi))
        b = block(g, nm)
        b["share"] = len(g) / len(led)
        RES.setdefault("p10", {})[nm] = b

    print("\n=== PART 11 — time of day (2h PT blocks) ===")
    hr = ((15 + led["entry_sec"] // 3600) % 24).astype(int)
    blk = (hr // 2) * 2
    for b_, g in led.groupby(blk):
        bb = block(g, f"{b_:02d}-{(b_+2)%24:02d} PT")
        RES.setdefault("p11", {})[int(b_)] = bb

    print("\n=== PART 14 (cost) — slippage stress ===")
    for extra in (0.0, 0.5, 1.0, 2.0):
        rr = (led["net_pnl_pts"] - extra) / 12.5
        rw, rl = rr[rr > 0], rr[rr <= 0]
        RES.setdefault("p14cost", {})[str(extra)] = dict(
            wr=float((rr > 0).mean()), expR=float(rr.mean()),
            pf=float(rw.sum() / -rl.sum()))
        c = RES["p14cost"][str(extra)]
        print(f"  +{extra} pts RT: WR {c['wr']:.3f} expR {c['expR']:+.3f} "
              f"PF {c['pf']:.2f}")

    print("\n=== PART 15 — breakeven ===")
    W, L = float(w.mean()), float(-l.mean())
    be_wr = L / (W + L)
    RES["p15"] = dict(W=W, L=L, be_wr=be_wr,
                      margin_pp=(d["wr"] - be_wr) * 100,
                      be_slip_pts=float(r.mean() * 12.5),
                      wr_m5=float((d["wr"] - .05) * (W + L) - L),
                      wr_m10=float((d["wr"] - .10) * (W + L) - L))
    print(f"  avg net W {W:.3f}R, L {L:.3f}R → breakeven WR "
          f"{be_wr:.3f} (actual {d['wr']:.3f}, margin "
          f"{(d['wr']-be_wr)*100:.1f}pp)")
    print(f"  breakeven extra slippage {r.mean()*12.5:.2f} pts RT | "
          f"WR-5pp → expR {RES['p15']['wr_m5']:+.3f} | WR-10pp → "
          f"{RES['p15']['wr_m10']:+.3f}")

    print("\n=== PART 16 — forward-test windows (contiguous, all starts) ===")
    for N in (50, 100, 200, 300, 500):
        wrs, exps, ddz = [], [], []
        for s0 in range(0, len(r) - N, 7):
            seg_ = r[s0:s0 + N]
            wrs.append((seg_ > 0).mean())
            exps.append(seg_.mean())
            ddz.append(maxdd(seg_))
        wrs, exps, ddz = map(np.array, (wrs, exps, ddz))
        RES.setdefault("p16", {})[N] = dict(
            wr=[float(np.percentile(wrs, p)) for p in (5, 50, 95)],
            exp=[float(np.percentile(exps, p)) for p in (5, 50, 95)],
            dd=[float(np.percentile(ddz, p)) for p in (50, 95, 99)],
            p_neg=float((exps <= 0).mean()))
        v = RES["p16"][N]
        print(f"  {N:3} trades (~{N/3.4:.0f} sess): WR [{v['wr'][0]:.2f},"
              f"{v['wr'][2]:.2f}] expR [{v['exp'][0]:+.2f},"
              f"{v['exp'][2]:+.2f}] DD p50 {v['dd'][0]:.1f} p95 "
              f"{v['dd'][1]:.1f} p99 {v['dd'][2]:.1f} | "
              f"P(expR<=0) {v['p_neg']:.2f}")

    out = ROOT / "outputs" / "p13_stats.json"
    out.write_text(json.dumps(RES, indent=1, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
