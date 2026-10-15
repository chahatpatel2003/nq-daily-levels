#!/usr/bin/env python
"""P25A evaluator: first-passage matrices, MAE-before-target, time to
reaction, categories, tap/scale diagnostics, region penetration.
Consumes outputs/p25a_events.parquet (+ _latertap). All threshold
pairs preregistered in the phase brief; ties = adverse-first."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRS = [(5, 5), (10, 5), (10, 7.5), (10, 10),
         (15, 5), (15, 10), (15, 15),
         (20, 10), (20, 15), (20, 20),
         (30, 10), (30, 15), (30, 20), (30, 25),
         (40, 20), (50, 25), (60, 30)]
NY_PAIRS = [(10, 5), (10, 10), (15, 10), (20, 10), (20, 15),
            (30, 15), (30, 20)]
KEY_PAIRS = [(10, 5), (10, 10), (15, 10), (20, 10)]


def acol(a):
    return "ta" + str(a).replace(".", "_")


def pair_stats(d, F, A):
    tf = d[f"tf{F}"].to_numpy()
    ta = d[acol(A)].to_numpy()
    win = (tf >= 0) & ((ta < 0) | (tf < ta))
    adv1 = (ta >= 0) & ~win
    unres = (tf < 0) & (ta < 0)
    n = len(d)
    return dict(F=F, A=A, n=n, win_pct=float(win.mean()),
                adv_first_pct=float(adv1.mean()),
                unresolved_pct=float(unres.mean()))


def dedup_regions(d, fresh_only):
    """First event per causal component per session (member-set
    disjointness); optionally fresh regions only."""
    keep = []
    for sess, g in d.sort_values(["session", "tap_ts"]) \
            .groupby("session", sort=False):
        used = set()
        for r in g.itertuples():
            if fresh_only and r.reg_prior_tapped > 0:
                continue
            mem = set(r.reg_mem.split("|"))
            if mem & used:
                continue
            used |= mem
            keep.append(r.Index)
    return d.loc[keep]


def qrow(x, ps=(10, 25, 50, 75, 90)):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {f"p{p}": np.nan for p in ps}
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def main() -> int:
    df = pd.read_parquet(ROOT / "outputs" / "p25a_events.parquet")
    dfl = pd.read_parquet(ROOT / "outputs" /
                          "p25a_events_latertap.parquet")
    reg = dedup_regions(df, fresh_only=True)
    reg_sess = dedup_regions(df, fresh_only=False)
    print(f"events: child {len(df):,} | region-fresh {len(reg):,} | "
          f"region-session {len(reg_sess):,}")

    # ---- Parts 8/9/14: matrices --------------------------------------
    rows = []
    for name, d in (("CHILD", df), ("REGION_FRESH", reg),
                    ("REGION_SESSION", reg_sess)):
        for F, A in PAIRS:
            rows.append(dict(pair_stats(d, F, A), population=name))
    mat = pd.DataFrame(rows)
    mat.to_csv(ROOT / "outputs" / "p25a_clean_reaction_matrix.csv",
               index=False)
    for name in ("CHILD", "REGION_FRESH"):
        m = mat[mat["population"] == name]
        print(f"\n=== {name} first-passage (win% / adv-first% / "
              f"unresolved%) n={m['n'].iloc[0]:,} ===")
        for r in m.itertuples():
            print(f"  +{r.F:>2} before -{r.A:<4}: {r.win_pct:6.1%}  "
                  f"{r.adv_first_pct:6.1%}  {r.unresolved_pct:6.1%}")

    cv = []
    for F, A in PAIRS:
        c = pair_stats(df, F, A)
        r1 = pair_stats(reg, F, A)
        r2 = pair_stats(reg_sess, F, A)
        cv.append(dict(F=F, A=A, child_win=c["win_pct"],
                       region_fresh_win=r1["win_pct"],
                       region_session_win=r2["win_pct"],
                       n_child=c["n"], n_region_fresh=r1["n"],
                       n_region_session=r2["n"]))
    pd.DataFrame(cv).to_csv(ROOT / "outputs" /
                            "p25a_child_vs_region.csv", index=False)

    # ---- Part 13: NY-AM ----------------------------------------------
    ny = reg[reg["is_ny"] == 1]
    ny_child = df[df["is_ny"] == 1]
    nyr = []
    for F, A in NY_PAIRS:
        nyr.append(dict(pair_stats(ny, F, A), population="REGION_FRESH"))
        nyr.append(dict(pair_stats(ny_child, F, A), population="CHILD"))
    pd.DataFrame(nyr).to_csv(ROOT / "outputs" /
                             "p25a_clean_reaction_nyam.csv", index=False)
    ns_all = df["session"].nunique()
    ns_25 = df[df["session"] >= "2025-01-01"]["session"].nunique()
    ny25 = ny[ny["session"] >= "2025-01-01"]
    print(f"\n=== NY-AM REGION_FRESH (n={len(ny):,}; "
          f"{len(ny)/ns_all:.2f}/session all, "
          f"{len(ny25)/ns_25:.2f}/session 2025+) ===")
    for F, A in NY_PAIRS:
        s = pair_stats(ny, F, A)
        print(f"  +{F:>2} before -{A:<4}: {s['win_pct']:6.1%} "
              f"(adv {s['adv_first_pct']:.1%} unres "
              f"{s['unresolved_pct']:.1%})")

    # ---- Part 10: MAE before target ----------------------------------
    mrows = []
    for name, d in (("CHILD", df), ("REGION_FRESH", reg)):
        for F in (10, 15, 20, 30):
            m = d[d[f"tf{F}"] >= 0][f"mae{F}"]
            mrows.append(dict(population=name, target=F,
                              n_reached=int(len(m)),
                              reach_pct=float((d[f"tf{F}"] >= 0)
                                              .mean()), **qrow(m)))
    mdf = pd.DataFrame(mrows)
    mdf.to_csv(ROOT / "outputs" / "p25a_mae_before_target.csv",
               index=False)
    print("\n=== MAE before favorable target (pts) ===")
    print(mdf.round(2).to_string(index=False))

    # ---- Part 11: time to reaction -----------------------------------
    trows = {}
    print("\n=== time to first attainment (s; window 3600s; "
          "reached only) ===")
    for name, d in (("CHILD", df), ("REGION_FRESH", reg)):
        for F in (5, 10, 15, 20, 30):
            t = d[d[f"tf{F}"] >= 0][f"tf{F}"]
            q = np.percentile(t, [25, 50, 75, 90]) if len(t) else [np.nan] * 4
            trows[f"{name}_t{F}"] = dict(
                med=float(q[1]), p25=float(q[0]), p75=float(q[2]),
                p90=float(q[3]),
                unreached=float((d[f"tf{F}"] < 0).mean()))
            print(f"  {name:>12} +{F:>2}: med {q[1]:6.0f} p25 {q[0]:5.0f}"
                  f" p75 {q[2]:6.0f} p90 {q[3]:6.0f} | not reached "
                  f"{(d[f'tf{F}'] < 0).mean():.1%}")

    # ---- Part 12: categories (precedence CLEAN > ACCEPTABLE > FAILED
    #      > MESSY > OTHER; FAILED = -20 first-passage before +10) ------
    cats = {}
    for name, d in (("CHILD", df), ("REGION_FRESH", reg)):
        tf10, tf15 = d["tf10"].to_numpy(), d["tf15"].to_numpy()
        ta5 = d["ta5"].to_numpy(); ta10 = d["ta10"].to_numpy()
        ta20 = d["ta20"].to_numpy()
        clean = (tf10 >= 0) & ((ta5 < 0) | (tf10 < ta5))
        acc = ~clean & (tf15 >= 0) & ((ta10 < 0) | (tf15 < ta10))
        fail = ~clean & ~acc & (ta20 >= 0) & ((tf10 < 0) | (ta20 < tf10))
        messy = (~clean & ~acc & ~fail
                 & (((tf10 >= 0) & (d["mae10"] > 10))
                    | ((tf10 < 0) & (tf15 >= 0) & (d["mae15"] > 10))))
        other = ~clean & ~acc & ~fail & ~messy
        cats[name] = dict(CLEAN=float(clean.mean()),
                          ACCEPTABLE=float(acc.mean()),
                          FAILED=float(fail.mean()),
                          MESSY=float(messy.mean()),
                          OTHER=float(other.mean()))
        print(f"\ncategories {name}: " + "  ".join(
            f"{k} {v:.1%}" for k, v in cats[name].items()))

    # ---- Part 15: taps -----------------------------------------------
    print("\n=== taps (key pairs win%) ===")
    tapt = {}
    groups = [("tap1", df)]
    if len(dfl):
        groups += [("tap2", dfl[dfl["tap_no"] == 2]),
                   ("tap3", dfl[dfl["tap_no"] == 3]),
                   ("tap4+", dfl[dfl["tap_no"] >= 4])]
    for lab, d in groups:
        tapt[lab] = {f"{F}b{A}": pair_stats(d, F, A)["win_pct"]
                     for F, A in KEY_PAIRS}
        tapt[lab]["n"] = len(d)
        print(f"  {lab:>5} (n {len(d):>7,}): " + "  ".join(
            f"+{F}/-{A} {tapt[lab][f'{F}b{A}']:.1%}"
            for F, A in KEY_PAIRS))

    # ---- Part 16: scales ---------------------------------------------
    print("\n=== scales (key pairs win%) ===")
    sct = {}
    for lab, d in [(f"scale{s}", df[df["scale"] == s]) for s in
                   (0, 1, 2)] + \
                  [(f"reg_smax{s}", reg[reg["reg_smax"] == s]) for s in
                   (0, 1, 2)]:
        sct[lab] = {f"{F}b{A}": pair_stats(d, F, A)["win_pct"]
                    for F, A in KEY_PAIRS}
        sct[lab]["n"] = len(d)
        print(f"  {lab:>9} (n {len(d):>7,}): " + "  ".join(
            f"+{F}/-{A} {sct[lab][f'{F}b{A}']:.1%}"
            for F, A in KEY_PAIRS))

    # ---- Parts 17/18: entry location + penetration -------------------
    print("\n=== region penetration before target (REGION_FRESH) ===")
    pent = {}
    for F in (10, 20, 30):
        d = reg[reg[f"tf{F}"] >= 0]
        p = d[f"pen{F}"]
        pw = (d[f"pen{F}"] / d["reg_w"].clip(lower=0.25)).replace(
            [np.inf, -np.inf], np.nan)
        loc = d[f"loc{F}"].value_counts(normalize=True).to_dict()
        pent[F] = dict(med=float(p.median()),
                       p75=float(p.quantile(.75)),
                       p90=float(p.quantile(.90)),
                       med_pct_width=float(pw.median()),
                       core_share=float(d[f"core{F}"].mean()),
                       loc=loc)
        print(f"  before +{F}: med {p.median():5.1f}pt p75 "
              f"{p.quantile(.75):5.1f} p90 {p.quantile(.90):5.1f} | "
              f"med %width {pw.median():5.1%} | reached CORE-2 "
              f"{d[f'core{F}'].mean():.1%} | loc " +
              " ".join(f"{k}:{v:.0%}" for k, v in sorted(loc.items())))

    # ---- yearly stability of key pairs -------------------------------
    print("\n=== yearly win% (REGION_FRESH) ===")
    yearly = {}
    for F, A in KEY_PAIRS:
        yr = {}
        for y, g in reg.groupby(reg["session"].str[:4]):
            yr[y] = pair_stats(g, F, A)["win_pct"]
        yearly[f"{F}b{A}"] = yr
        print(f"  +{F}/-{A}: " + " ".join(f"{y} {v:.1%}"
                                          for y, v in sorted(yr.items())))

    # ---- eventual-reach reconciliation (the old '95%' metric) --------
    ev10 = float((df["tf10"] >= 0).mean())
    ev10r = float((reg["tf10"] >= 0).mean())
    print(f"\neventual +10 within 60m: child {ev10:.1%} | region "
          f"{ev10r:.1%}  (ordering-blind — the old-style metric)")

    summary = dict(
        n_child=int(len(df)), n_region_fresh=int(len(reg)),
        n_region_session=int(len(reg_sess)),
        ny_region_fresh=int(len(ny)),
        ny_region_fresh_per_session_all=float(len(ny) / ns_all),
        ny_region_fresh_per_session_2025=float(len(ny25) / ns_25),
        matrix={f"{r.F}b{r.A}_{r.population}": r.win_pct
                for r in mat.itertuples()},
        nyam={f"{F}b{A}": pair_stats(ny, F, A)["win_pct"]
              for F, A in NY_PAIRS},
        mae={f"{r.population}_t{r.target}":
             dict(med=r.p50, p75=r.p75, p90=r.p90, reach=r.reach_pct)
             for r in mdf.itertuples()},
        time_to=trows, categories=cats, taps=tapt, scales=sct,
        penetration={str(k): v for k, v in pent.items()},
        yearly=yearly, eventual_reach10=dict(child=ev10, region=ev10r),
        aref_eq_ref_share=float((df["aref_diff"] == 0).mean()),
        window_s=3600, tie_convention="adverse_first")
    (ROOT / "outputs" / "p25a_summary.json").write_text(
        json.dumps(summary, indent=1, default=float))
    print("\nsaved p25a_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
