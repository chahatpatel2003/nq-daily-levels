#!/usr/bin/env python
"""P25 parity gate: production engine vs frozen research layers.

For sampled (end_session, at_sec) snapshots compare the LIVE zone set
(alive, above/below anywhere — full active set) between:
  PROD: ResistanceEngine window replay (frozen call-graph, fresh run)
  REF : stored research zones/interactions layers, same causal filter
Match key: (form_session, scale, form_min, price_rf, hw_t).
Compared: existence (both directions), geometry (0 ticks), confirmation
timestamp (exact), tap count <= t (exact), broken flag (same rule).
Plus RESTART parity (window 50 vs 42) and STREAMING parity
(trunc_last_at mid-session vs full run filtered <= t)."""
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

from nqlevels.prod.engine import (ResistanceEngine, epoch_s,            # noqa
                                  causal_break_zids)

DEV = Path("/Volumes/T9/nq_levels/daily")
SEAL = Path("/Volumes/T9/nq_levels/sealed_holdout_1/derived/daily")

END_SESSIONS = ["2021-09-15", "2022-03-14", "2022-11-10", "2023-06-21",
                "2024-02-08", "2024-09-19", "2025-04-08", "2025-07-16",
                "2026-03-05", "2026-06-16", "2026-07-16", "2026-08-12"]
AT_SECS = [10 * 3600, 16 * 3600, 20 * 3600]
KEY = ["session", "scale", "form_min", "price_rf", "hw_t"]


def load_ref():
    zs, it = [], []
    for base, lo, hi in ((DEV, None, "2026-06-17"),
                         (SEAL, "2026-06-18", None)):
        z = pd.read_parquet(base / "zones.parquet")
        z["session"] = z["session"].astype(str)
        t = pd.read_parquet(base / "interactions.parquet",
                            columns=["session", "zid", "tap_ts",
                                     "app_side_above", "zlo_t", "zhi_t",
                                     "o_depth_max_full"])
        t["session"] = t["session"].astype(str)
        if lo:
            z = z[z["session"] >= lo]
            t = t[t["session"] >= lo]
        if hi:
            z = z[z["session"] <= hi]
            t = t[t["session"] <= hi]
        if base == SEAL:
            z = z.copy(); z["zid"] += 10_000_000
            t = t.copy(); t["zid"] += 10_000_000
        zs.append(z); it.append(t)
    return (pd.concat(zs, ignore_index=True).drop_duplicates("zid"),
            pd.concat(it, ignore_index=True))


def ref_state(zref, tref, ord_of, session, t_abs, ep1s, hi1s):
    so = ord_of[session]
    z = zref[(zref["form_ts_val"] <= t_abs)].copy()
    z["ford"] = z["session"].map(ord_of)
    z = z[(so - z["ford"] <= 30) & (z["ford"] <= so)]
    tt = tref[tref["tap_ts"] <= t_abs]
    ntaps = tt.groupby("zid").size()
    z["n_taps"] = z["zid"].map(ntaps).fillna(0).astype(int)
    res = tt[tt["app_side_above"] == 0]
    # identical causal break rule as production snapshot()
    z["broken"] = z["zid"].isin(
        causal_break_zids(res, session, t_abs, ep1s, hi1s))
    return z[(~z["broken"]) & (z["n_taps"] < 6)]


def compare(prod_live, ref_live):
    p = prod_live.copy(); r = ref_live.copy()
    for d in (p, r):
        d["k"] = list(zip(d["session"], d["scale"].astype(int),
                          d["form_min"].astype(int),
                          d["price_rf"].round(6), d["hw_t"].round(6)))
    pk, rk = set(p["k"]), set(r["k"])
    only_p, only_r = pk - rk, rk - pk
    both = pk & rk
    pm = p.set_index("k").loc[list(both)]
    rm = r.set_index("k").loc[list(both)]
    tap_mm = int((pm["n_taps"].sort_index()
                  != rm["n_taps"].sort_index()).sum())
    conf_mm = int((pm["form_ts_val"].sort_index()
                   != rm["form_ts_val"].sort_index()).sum())
    return dict(n_prod=len(p), n_ref=len(r), matched=len(both),
                only_prod=len(only_p), only_ref=len(only_r),
                tap_mismatch=tap_mm, conf_mismatch=conf_mm)


def main() -> int:
    t0 = _time.time()
    zref, tref = load_ref()
    rep = dict(snapshots=[], restart=[], streaming=[])
    ctx_sessions = None
    tot = dict(matched=0, only_prod=0, only_ref=0, tap_mismatch=0,
               conf_mismatch=0, zones=0)
    for end in END_SESSIONS:
        eng = ResistanceEngine()
        eng.run_window(end)
        if ctx_sessions is None:
            ctx_sessions = sorted(eng._ord_of, key=eng._ord_of.get)
        b1m = eng._bars(end, "bars_1m")
        ep0 = epoch_s(b1m["ts"])[0]
        b1s_e = eng._bars(end, "bars_1s")
        ep1s = epoch_s(b1s_e["ts"])
        hi1s = b1s_e["high_t"].to_numpy("float64")
        for at in AT_SECS:
            t_abs = int(ep0 + at)
            snap = eng.snapshot(end, at)
            prod_live = snap["all_zones"]
            prod_live = prod_live[(~prod_live["broken"])
                                  & (prod_live["n_taps"] < 6)]
            ref_live = ref_state(zref, tref, eng._ord_of, end, t_abs,
                                 ep1s, hi1s)
            c = compare(prod_live, ref_live)
            c.update(end=end, at=at)
            rep["snapshots"].append(c)
            for k in ("matched", "only_prod", "only_ref", "tap_mismatch",
                      "conf_mismatch"):
                tot[k] += c[k]
            tot["zones"] += c["n_ref"]
            flag = "" if (c["only_prod"] == c["only_ref"]
                          == c["tap_mismatch"] == c["conf_mismatch"] == 0) \
                else "  <-- MISMATCH"
            print(f"  {end} t+{at//3600}h: prod {c['n_prod']:3d} ref "
                  f"{c['n_ref']:3d} matched {c['matched']:3d} | "
                  f"+p {c['only_prod']} +r {c['only_ref']} tap≠ "
                  f"{c['tap_mismatch']} conf≠ {c['conf_mismatch']}{flag}")
        # restart parity: shorter window, same end
        eng2 = ResistanceEngine(window=42)
        eng2.run_window(end)
        s1 = eng.snapshot(end, 16 * 3600)["all_zones"]
        s2 = eng2.snapshot(end, 16 * 3600)["all_zones"]
        s1 = s1[(~s1["broken"]) & (s1["n_taps"] < 6)]
        s2 = s2[(~s2["broken"]) & (s2["n_taps"] < 6)]
        c = compare(s1, s2)
        rep["restart"].append(dict(c, end=end))
        ok = (c["only_prod"] == c["only_ref"] == c["tap_mismatch"] == 0)
        print(f"  restart(50 vs 42) {end}: matched {c['matched']} "
              f"diffs {c['only_prod']}/{c['only_ref']}/"
              f"{c['tap_mismatch']} {'OK' if ok else '<-- FAIL'}")
    # streaming parity: truncated final session vs full filtered
    for end, at in (("2025-04-08", 14 * 3600), ("2026-07-16", 16 * 3600),
                    ("2026-06-16", 18 * 3600)):
        e_full = ResistanceEngine(); e_full.run_window(end)
        e_tr = ResistanceEngine(); e_tr.run_window(end, trunc_last_at=at)
        a = e_full.snapshot(end, at)["all_zones"]
        b1m = e_tr._bars(end, "bars_1m")
        b = e_tr.snapshot(end, at)["all_zones"]
        a = a[(~a["broken"]) & (a["n_taps"] < 6)]
        b = b[(~b["broken"]) & (b["n_taps"] < 6)]
        c = compare(a, b)
        rep["streaming"].append(dict(c, end=end, at=at))
        ok = (c["only_prod"] == c["only_ref"] == c["tap_mismatch"] == 0)
        print(f"  streaming {end} t+{at//3600}h: matched {c['matched']} "
              f"diffs {c['only_prod']}/{c['only_ref']}/"
              f"{c['tap_mismatch']} {'OK' if ok else '<-- FAIL'}")
    rep["totals"] = tot
    rep["elapsed_min"] = (_time.time() - t0) / 60
    Path(ROOT / "outputs" / "p25_production_parity_report.json") \
        .write_text(json.dumps(rep, indent=1, default=str))
    clean = (tot["only_prod"] == tot["only_ref"] == tot["tap_mismatch"]
             == tot["conf_mismatch"] == 0)
    print(f"\nTOTALS: zones compared {tot['zones']:,} matched "
          f"{tot['matched']:,} | existence diffs {tot['only_prod']}"
          f"/{tot['only_ref']} | tap≠ {tot['tap_mismatch']} | conf≠ "
          f"{tot['conf_mismatch']} | PARITY "
          f"{'PASS' if clean else 'FAIL'} "
          f"[{rep['elapsed_min']:.1f} min]")
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
