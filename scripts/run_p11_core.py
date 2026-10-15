#!/usr/bin/env python
"""P11 core — count audit, drawdown anatomy, streaks, daily distribution,
and the wrapper replays, all on the frozen F4 executed stream.

Stream = chronological one-open-trade F4 (r4 ∩ p60 ≥ 0.85, E1d, S 12.5,
T 30, base costs). Windows separated: 22-23 selection era (gate itself
walk-forward), 24-26 frozen OOS, 22-26 pooled for tail statistics.
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

from run_p10_eval import portfolio, rows_for                                   # noqa: E402
from run_p10_stage1 import load_trades                                         # noqa: E402

SEG_N = {0: "asia", 1: "london", 2: "premkt", 3: "ny-am", 4: "ny-pm"}
RISK = 12.5


def dd_episodes(eq: np.ndarray, dates: np.ndarray):
    """Episodes (peak_i, trough_i, rec_i|None, depth) on trade-indexed equity."""
    peaks = np.maximum.accumulate(eq)
    out = []
    i = 0
    n = eq.size
    while i < n:
        if eq[i] < peaks[i]:
            p = i - 1
            while p >= 0 and eq[p] < peaks[i]:
                p -= 1
            peak_v = peaks[i]
            j = i
            trough = i
            while j < n and eq[j] < peak_v:
                if eq[j] < eq[trough]:
                    trough = j
                j += 1
            rec = j if j < n else None
            out.append({"peak_i": max(p, 0), "trough_i": trough, "rec_i": rec,
                        "depth": float(peak_v - eq[trough]),
                        "peak_d": dates[max(p, 0)], "trough_d": dates[trough],
                        "rec_d": dates[rec] if rec is not None else None})
            i = j
        else:
            i += 1
    return out


def replay(day_groups, daily_stop=None, cap=None, max_n=None, consec=None,
           next_session=False):
    """Wrapper replay on the executed stream. Returns kept pnl array plus
    skipped-trade accounting."""
    kept, skip_w, skip_l, days_stopped = [], 0, 0, 0
    for sess, g in day_groups:
        cum, n, run_l = 0.0, 0, 0
        stopped = False
        pause_seg = None
        for r in g.itertuples():
            if pause_seg is not None and r.seg == pause_seg:
                skip_w += int(r.pnl > 0)
                skip_l += int(r.pnl <= 0)
                continue
            elif pause_seg is not None:
                pause_seg = None
                run_l = 0
            if stopped:
                skip_w += int(r.pnl > 0)
                skip_l += int(r.pnl <= 0)
                continue
            kept.append(r.pnl)
            cum += r.pnl / RISK
            n += 1
            run_l = run_l + 1 if r.pnl <= 0 else 0
            trip = ((daily_stop is not None and cum <= daily_stop)
                    or (cap is not None and cum >= cap)
                    or (max_n is not None and n >= max_n)
                    or (consec is not None and run_l >= consec))
            if trip:
                if consec is not None and next_session and run_l >= consec \
                        and not ((daily_stop is not None and cum <= daily_stop)
                                 or (cap is not None and cum >= cap)
                                 or (max_n is not None and n >= max_n)):
                    pause_seg = r.seg
                else:
                    stopped = True
                    days_stopped += 1
    return np.array(kept), skip_w, skip_l, days_stopped


def summarize(pnl_pts, n_days):
    if pnl_pts.size < 30:
        return {}
    R = pnl_pts / RISK
    eq = np.cumsum(R)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    ls, cur = 0, 0
    for x in R:
        cur = cur + 1 if x <= 0 else 0
        ls = max(ls, cur)
    gw = R[R > 0].sum()
    gl = -R[R <= 0].sum()
    return {"n": int(R.size), "tpd": float(R.size / n_days),
            "wr": float((R > 0).mean()), "expR": float(R.mean()),
            "totR": float(R.sum()), "pf": float(gw / gl) if gl > 0 else np.inf,
            "maxdd": dd, "streak": ls}


def main() -> int:
    t0 = _time.time()
    t = load_trades()
    res: dict = {}

    # ---------------- PART 0: count audit -----------------------------------
    print("=" * 88)
    print("PART 0 — COUNT AUDIT (exact denominators)")
    print("=" * 88)
    base = t[(t["entry"] == "E1d")].drop_duplicates(["session", "zid", "tap_sec", "src"])
    inter = t[(t["entry"] == "E1")].drop_duplicates(["session", "zid", "tap_sec"])
    yrs = sorted(t["year"].unique())
    print(f"{'year':>6} {'sessions':>9} {'interactions':>13} {'/day':>6} "
          f"{'r4 fires':>9} {'/day':>6} {'fire%':>6} {'p>=.85':>7} {'/day':>6} "
          f"{'gate%':>6}")
    audit = {}
    r4 = rows_for(t, "E1d", ("r4",), None)
    f4 = rows_for(t, "E1d", ("r4",), 0.85)
    # unique interaction universe per year from the p9 population join
    pop = pd.read_parquet(
        "/Volumes/T9/nq_levels/daily/p9_timeline.parquet",
        columns=["session", "zid", "tap_sec"])
    pop["year"] = pd.to_datetime(pop["session"].astype(str)).dt.year
    for y in yrs:
        ns = t[t["year"] == y]["session"].nunique()
        ni = int((pop["year"] == y).sum())
        nr = int((r4["year"] == y).sum())
        nf = int((f4["year"] == y).sum())
        audit[int(y)] = {"sessions": ns, "interactions": ni, "r4": nr, "f4": nf}
        print(f"{y:>6} {ns:>9} {ni:>13,} {ni/ns:>6.1f} {nr:>9,} {nr/ns:>6.1f} "
              f"{nr/ni:>6.2f} {nf:>7,} {nf/ns:>6.2f} "
              f"{nf/max(1,nr):>6.2f}")
    res["audit"] = audit
    print("\nreconciliation: the earlier '~15 touches/day' was the P5 DISPLAY "
          "subset (top-4 shown zones,\ntouched-while-displayed 16.4/day); the "
          "strategy universe is ALL eligible deduped resistance\ninteractions "
          "(~46/day). R4 fire share is era-dependent (22.5 % pooled 2021-26; "
          "higher in 24-26).\nF4 exists only where walk-forward scores exist "
          "(2022+, 82 % of rows).")

    # ---------------- F4 executed stream ------------------------------------
    pf = portfolio(t, "E1d", ("r4",), 0.85, 12.5, 30.0)
    pf = pf.sort_values(["session", "t_open"]).reset_index(drop=True)
    pf["R"] = pf["pnl"] / RISK
    pf["year"] = pd.to_datetime(pf["session"].astype(str)).dt.year
    n_days_all = t[t["year"] >= 2022]["session"].nunique()
    print(f"\nF4 executed stream: {len(pf):,} trades over "
          f"{pf['session'].nunique():,} active sessions "
          f"({n_days_all:,} total 2022-26 sessions)")
    print("by year: " + " ".join(f"{y}:{len(g):,}" for y, g in pf.groupby("year")))
    print("by seg:  " + " ".join(f"{SEG_N[int(s)]}:{len(g):,}"
                                 for s, g in pf.groupby("seg")))
    res["stream"] = {"n": len(pf),
                     "by_year": {int(y): len(g) for y, g in pf.groupby("year")},
                     "by_seg": {SEG_N[int(s)]: len(g) for s, g in pf.groupby("seg")}}

    # ---------------- PART 1-2: drawdown anatomy ----------------------------
    print("\n" + "=" * 88)
    print("PART 1-2 — DRAWDOWN ANATOMY (executed stream, R units)")
    print("=" * 88)
    eq = np.cumsum(pf["R"].to_numpy())
    dates = pf["session"].to_numpy()
    eps = dd_episodes(eq, dates)
    n_years = (pd.Timestamp("2026-06-17") - pd.Timestamp("2022-01-01")).days / 365.25
    print(f"total DD episodes: {len(eps):,} | span {n_years:.1f} yrs")
    print(f"{'>=R':>6} {'count':>6} {'/year':>6} {'med dur(d)':>11} "
          f"{'p90 dur':>8} {'med rec(d)':>11} {'max rec':>8}")
    sess_sorted = sorted(t[t["year"] >= 2022]["session"].unique())
    s_ix = {s: i for i, s in enumerate(sess_sorted)}
    for thr in (2, 3, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27):
        sel = [e for e in eps if e["depth"] >= thr]
        if not sel:
            print(f"{thr:>6} {0:>6}")
            continue
        dur = [s_ix[e["trough_d"]] - s_ix[e["peak_d"]] for e in sel]
        rec = [s_ix[e["rec_d"]] - s_ix[e["trough_d"]] for e in sel
               if e["rec_d"] is not None]
        print(f"{thr:>6} {len(sel):>6} {len(sel)/n_years:>6.1f} "
              f"{np.median(dur):>11.0f} {np.percentile(dur, 90):>8.0f} "
              f"{(np.median(rec) if rec else np.nan):>11.0f} "
              f"{(max(rec) if rec else np.nan):>8.0f}")
        res.setdefault("dd_thr", {})[str(thr)] = {
            "count": len(sel), "per_year": len(sel) / n_years}
    worst = max(eps, key=lambda e: e["depth"])
    print(f"\nMAX DD: {worst['depth']:.1f}R | peak {worst['peak_d']} -> trough "
          f"{worst['trough_d']} ({s_ix[worst['trough_d']]-s_ix[worst['peak_d']]} "
          f"sessions) -> recovered {worst['rec_d']} "
          f"(+{(s_ix[worst['rec_d']]-s_ix[worst['trough_d']]) if worst['rec_d'] else -1} sessions)")
    res["max_dd"] = {k: str(v) for k, v in worst.items()}
    # longest no-new-high period
    peaks_i = np.flatnonzero(eq == np.maximum.accumulate(eq))
    gaps = np.diff([s_ix[dates[i]] for i in peaks_i])
    print(f"longest period without new equity high: {int(gaps.max())} sessions")
    res["longest_underwater_sessions"] = int(gaps.max())

    # ---------------- PART 3: streaks ---------------------------------------
    print("\n" + "=" * 88)
    print("PART 3 — LOSS STREAKS")
    print("=" * 88)
    Rv = pf["R"].to_numpy()
    runs, start = [], None
    for i, x in enumerate(Rv):
        if x <= 0:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - 1))
                start = None
    if start is not None:
        runs.append((start, len(Rv) - 1))
    lens = np.array([b - a + 1 for a, b in runs])
    for thr in (5, 8, 10, 12, 15):
        print(f"  streaks >= {thr:>2}: {int((lens >= thr).sum()):>4} "
              f"({(lens >= thr).sum()/n_years:.1f}/yr)")
        res.setdefault("streaks", {})[str(thr)] = int((lens >= thr).sum())
    a, b = runs[int(np.argmax(lens))]
    print(f"  LONGEST: {int(lens.max())} losses | {dates[a]} -> {dates[b]} "
          f"({s_ix[dates[b]]-s_ix[dates[a]]+1} sessions) | equity "
          f"{eq[a-1] if a>0 else 0:.1f}R -> {eq[b]:.1f}R | inside max-DD window: "
          f"{worst['peak_d'] <= dates[a] and (worst['rec_d'] is None or dates[b] <= worst['rec_d'])}")
    res["longest_streak"] = {"len": int(lens.max()), "start": str(dates[a]),
                             "end": str(dates[b])}

    # ---------------- PART 4: daily distribution ----------------------------
    print("\n" + "=" * 88)
    print("PART 4 — DAILY P&L (R)")
    print("=" * 88)
    day = pf.groupby("session")["R"].sum()
    day_all = day.reindex(sess_sorted).fillna(0.0)
    q = day.describe(percentiles=[.1, .25, .5, .75, .9])
    print(f"active days {len(day):,}/{len(sess_sorted):,} | mean {day.mean():+.2f} "
          f"med {day.median():+.2f} | p10 {day.quantile(.1):+.2f} p90 "
          f"{day.quantile(.9):+.2f} | best {day.max():+.1f} worst {day.min():+.1f}")
    print(f"positive {float((day>0).mean()):.2f} | negative "
          f"{float((day<0).mean()):.2f} | no-trade days "
          f"{1-len(day)/len(sess_sorted):.2f}")
    for thr in (1, 2, 3, 4, 5, 6):
        print(f"  days <= -{thr}R: {float((day<=-thr).mean()):.3f} | "
              f"days >= +{thr}R: {float((day>=thr).mean()):.3f}")
    res["daily"] = {"mean": float(day.mean()), "worst": float(day.min()),
                    "best": float(day.max()),
                    "p_neg": float((day < 0).mean())}
    ntr = pf.groupby("session").size().reindex(sess_sorted).fillna(0)
    print("trades/day distribution: " + " ".join(
        f"{k}:{float((ntr==k).mean()):.2f}" for k in (0, 1, 2, 3, 4))
        + f" 5+:{float((ntr>=5).mean()):.2f}")
    mon = pf.assign(m=pd.to_datetime(pf["session"].astype(str)).dt.to_period("M")) \
            .groupby("m")["R"].sum()
    print(f"months profitable: {float((mon>0).mean()):.2f} | worst month "
          f"{mon.min():+.1f}R ({mon.idxmin()})")
    res["months"] = {"p_pos": float((mon > 0).mean()),
                     "worst": float(mon.min()), "worst_m": str(mon.idxmin())}

    # ---------------- PARTS 5-9: wrappers -----------------------------------
    print("\n" + "=" * 88)
    print("PARTS 5-9 — WRAPPER REPLAYS (executed stream)")
    print("=" * 88)
    groups = list(pf.groupby("session"))
    nd = len(sess_sorted)
    base_m = summarize(pf["pnl"].to_numpy(), nd)
    print(f"{'wrapper':>28} {'n':>6} {'expR':>7} {'totR':>7} {'maxDD':>6} "
          f"{'streak':>6} {'stops':>6} {'skipW':>6} {'skipL':>6} {'netR_skip':>9}")

    def show(name, ds=None, cap=None, mx=None, cs=None, nx=False):
        k, sw, sl, nstp = replay(groups, ds, cap, mx, cs, nx)
        m = summarize(k, nd)
        net_skip = (sl * RISK - sw * 30) / RISK * -1  # R given up by skipping
        # skipped losers avoided saves +1R each (loss = -S-cost ~ -1.08R);
        # skipped winners cost ~ +2.32R each; report simple net R effect:
        net = m["totR"] - base_m["totR"]
        print(f"{name:>28} {m['n']:>6,} {m['expR']:>+7.3f} {m['totR']:>7.0f} "
              f"{m['maxdd']:>6.0f} {m['streak']:>6} {nstp:>6} {sw:>6} {sl:>6} "
              f"{net:>+9.0f}")
        res.setdefault("wrappers", {})[name] = {**m, "days_stopped": nstp,
                                                "skip_w": sw, "skip_l": sl}

    show("RAW (none)")
    for ds in (-1, -2, -3, -4, -5, -6):
        show(f"daily stop {ds}R", ds=ds)
    for cap in (1, 2, 3, 4, 5, 6, 8):
        show(f"profit cap +{cap}R", cap=cap)
    for ds in (-2, -3, -4, -5):
        for cap in (2, 3, 4, 5):
            show(f"stop {ds}R + cap +{cap}R", ds=ds, cap=cap)
    for mx in (2, 3, 4, 5, 6, 8):
        show(f"max {mx} trades/day", mx=mx)
    for cs in (2, 3, 4, 5):
        show(f"stop after {cs} consec (day)", cs=cs)
        show(f"stop after {cs} consec (sess)", cs=cs, nx=True)

    out = ROOT / "outputs" / "p11_core.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    pf.to_parquet("/Volumes/T9/nq_levels/daily/p11_stream.parquet", index=False)
    print(f"\nwrote {out.relative_to(ROOT)} + p11_stream.parquet "
          f"({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
