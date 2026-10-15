#!/usr/bin/env python
"""P13 holiday / shortened-session and macro-event audits (descriptive
only — no filters adopted). Shortened sessions detected from actual 1m
bar counts; labelled with the CME equity-index holiday calendar dates
(constructed: fixed + floating US holidays 2022-2026). FOMC decision
days exact from the published schedule; NFP = first Friday rule."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.io.store import layer_dir                                        # noqa: E402

LED = "/Volumes/T9/nq_levels/daily/p13_ledger.parquet"

FOMC = """2022-01-26 2022-03-16 2022-05-04 2022-06-15 2022-07-27 2022-09-21
2022-11-02 2022-12-14 2023-02-01 2023-03-22 2023-05-03 2023-06-14
2023-07-26 2023-09-20 2023-11-01 2023-12-13 2024-01-31 2024-03-20
2024-05-01 2024-06-12 2024-07-31 2024-09-18 2024-11-07 2024-12-18
2025-01-29 2025-03-19 2025-05-07 2025-06-18 2025-07-30 2025-09-17
2025-10-29 2025-12-10 2026-01-28 2026-03-18 2026-04-29 2026-06-17""".split()


def us_holidays(years):
    """CME equity-index full/partial holiday dates."""
    import datetime as dt
    out = {}

    def nth_weekday(y, m, wd, n):
        d = dt.date(y, m, 1)
        d += dt.timedelta((wd - d.weekday()) % 7)
        return d + dt.timedelta(7 * (n - 1))

    def last_weekday(y, m, wd):
        d = dt.date(y, m + 1, 1) - dt.timedelta(1) if m < 12 else \
            dt.date(y, 12, 31)
        while d.weekday() != wd:
            d -= dt.timedelta(1)
        return d

    def obs(d):
        if d.weekday() == 5:
            return d - dt.timedelta(1)
        if d.weekday() == 6:
            return d + dt.timedelta(1)
        return d

    EASTER = {2022: dt.date(2022, 4, 17), 2023: dt.date(2023, 4, 9),
              2024: dt.date(2024, 3, 31), 2025: dt.date(2025, 4, 20),
              2026: dt.date(2026, 4, 5)}
    for y in years:
        out[dt.date(y, 1, 1) if dt.date(y, 1, 1).weekday() < 5
            else obs(dt.date(y, 1, 1))] = "NewYear"
        out[nth_weekday(y, 1, 0, 3)] = "MLK"
        out[nth_weekday(y, 2, 0, 3)] = "Presidents"
        out[EASTER[y] - dt.timedelta(2)] = "GoodFriday"
        out[last_weekday(y, 5, 0)] = "Memorial"
        out[obs(dt.date(y, 6, 19))] = "Juneteenth"
        out[obs(dt.date(y, 7, 4))] = "July4"
        out[dt.date(y, 7, 3)] = "July3-early"
        out[nth_weekday(y, 9, 0, 1)] = "Labor"
        out[nth_weekday(y, 11, 3, 4)] = "Thanksgiving"
        out[nth_weekday(y, 11, 3, 4) + dt.timedelta(1)] = "ThxFriday-early"
        out[dt.date(y, 12, 24)] = "XmasEve-early"
        out[obs(dt.date(y, 12, 25))] = "Christmas"
    return {str(k): v for k, v in out.items()}


def blockstat(g, label):
    r = g["net_R"].to_numpy()
    if not len(r):
        print(f"  {label:>22}: no trades")
        return {}
    w, l = r[r > 0], r[r <= 0]
    d = dict(n=len(r), wr=float((r > 0).mean()), expR=float(r.mean()),
             totR=float(r.sum()),
             pf=float(w.sum() / -l.sum()) if len(l) and l.sum() < 0
             else np.inf)
    print(f"  {label:>22}: n {d['n']:5,} WR {d['wr']:.3f} "
          f"expR {d['expR']:+.3f} totR {d['totR']:+7.1f} PF {d['pf']:.2f}")
    return d


def main() -> int:
    led = pd.read_parquet(LED)
    d0 = layer_dir("daily", str(ROOT / "configs"))
    b1m = pd.read_parquet(d0 / "bars_1m.parquet", columns=["session",
                                                           "is_front"])
    cnt = b1m[b1m["is_front"]].groupby("session").size()
    cnt.index = cnt.index.astype(str)
    med = cnt.median()
    short = cnt[cnt < 0.8 * med]
    years = range(2022, 2027)
    hol = us_holidays(years)
    res = {}

    print("=== PART 12 — holiday / shortened sessions ===")
    print(f"  median 1m bars/session {med:.0f}; shortened (<80%): "
          f"{len(short)} sessions")
    lab = {}
    for s in short.index:
        lab[s] = hol.get(s, "short-unlabelled")
    hol_sessions = set(lab) | {s for s in hol if s in set(led["session"])
                               or s in cnt.index}
    is_hol = led["session"].isin(hol_sessions)
    res["hol"] = blockstat(led[is_hol], "holiday/short days")
    res["norm"] = blockstat(led[~is_hol], "ordinary days")
    hs = led[is_hol]
    if len(hs):
        dsum = hs.groupby("session")["net_R"].sum().sort_values()
        print(f"  worst holiday {dsum.index[0]} "
              f"({lab.get(dsum.index[0], hol.get(dsum.index[0], '?'))}) "
              f"{dsum.iloc[0]:+.1f}R | best {dsum.index[-1]} "
              f"({lab.get(dsum.index[-1], hol.get(dsum.index[-1], '?'))}) "
              f"{dsum.iloc[-1]:+.1f}R")
        res["hol_days"] = {s: float(v) for s, v in dsum.items()}
    # adjacency to majors: max streak (2024-12-04..17) and max DD
    # (2026-04-30..05-12) vs holiday dates; churn 2026-05-04
    for probe, lab2 in (("2024-12-04", "max-streak start"),
                        ("2026-05-04", "max-DD trough/churn day"),
                        ("2023-12-01", "neg month 2023-12"),
                        ("2024-03-01", "neg month 2024-03")):
        near = [h for h in hol if abs(
            (pd.Timestamp(h) - pd.Timestamp(probe)).days) <= 5]
        print(f"  {lab2} {probe}: holidays within 5d: {near or 'none'}")
    ex = led[~is_hol]
    res["excl_effect"] = dict(
        totR_kept=float(ex["net_R"].sum()), totR_all=float(led["net_R"].sum()),
        n_dropped=int(is_hol.sum()))
    print(f"  excluding them: totR {res['excl_effect']['totR_all']:.0f} → "
          f"{res['excl_effect']['totR_kept']:.0f} "
          f"({is_hol.sum()} trades dropped)")

    print("\n=== PART 13 — macro events (descriptive) ===")
    fomc = set(FOMC)
    sess = pd.to_datetime(led["session"])
    is_fomc = led["session"].isin(fomc)
    nfp_dates = set()
    for y in years:
        for m in range(1, 13):
            d = pd.Timestamp(year=y, month=m, day=1)
            while d.weekday() != 4:
                d += pd.Timedelta(days=1)
            nfp_dates.add(str(d.date()))
    is_nfp = led["session"].isin(nfp_dates)
    res["fomc"] = blockstat(led[is_fomc], "FOMC decision days")
    res["nfp"] = blockstat(led[is_nfp], "NFP days (1st Fri)")
    res["plain"] = blockstat(led[~is_fomc & ~is_nfp], "non-event days")
    # FOMC afternoon (11:00 PT decision) specifically
    hr = ((15 + led["entry_sec"] // 3600) % 24).astype(int)
    pm = is_fomc & (hr >= 11)
    res["fomc_pm"] = blockstat(led[pm], "FOMC 11:00+ PT")
    out = ROOT / "outputs" / "p13_events.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
