#!/usr/bin/env python
"""P25A Parts 6-18: clean first-passage measurement of resistance
encounters. PREREGISTERED conventions (no optimization anywhere):

EVENTS
  A CHILD  : resistance-side (app_side_above==0) tap_no==1 rows — the
             zone's first causal contact ever. No outcome-dependent
             filter (instant trade-throughs stay in the population).
  B REGION : subset of A where the contacted child's overlap component
             (computed at event time over CAUSALLY-live zones: formed
             <= t, age <= 30 sessions, <6 prior taps, causal break
             time > t) contains NO previously-tapped member — a fresh
             region's first-ever encounter. One event per component
             per session (first by tap_ts).
  Also kept: REGION_SESSION = per-session region dedup of A (first
  event per session whose component shares no member with an earlier
  kept event that session) — workload-flavored robustness line.
  Later taps (2/3/4+) kept separately, descriptive only, with the
  causal state filter b_cau > tap_ts.

REFERENCE PRICE
  ref = contacted child's near edge zlo_t — by path continuity the
  first traded price inside the zone. alt_ref = contact-bar open
  clipped into [zlo, zhi] (gap entries); divergence share reported.
  Favorable (short) = ticks BELOW ref, adverse = ticks ABOVE ref.

FIRST PASSAGE
  1-second bars from the tap second onward, window = 3600 s (existing
  p6/P24 standard), truncated at session end (flag kept). t_F / t_A =
  first bar whose running favorable/adverse excursion reaches the
  threshold. Same-bar tie -> ADVERSE FIRST (frozen stop-first
  convention). No end-of-window maxima anywhere.

REGION GEOMETRY AT EVENT
  RL/RU = component min/max, width, max member scale, CORE-2 bands via
  frozen P22 sweep, penetration measured vs RL. No future member is
  ever added backward in time (components use zones formed <= t only).

Output: outputs/p25a_events.parquet (one row per A event with region
tags), later-tap rows in outputs/p25a_events_latertap.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_p22_regions import regions_of, depth_profile, core_bands  # noqa

DEV = "/Volumes/T9/nq_levels"
SEAL = "/Volumes/T9/nq_levels/sealed_holdout_1/derived"
WIN_S = 3600
FAVS = (5, 10, 15, 20, 30, 40, 50, 60)          # points
ADVS = (5, 7.5, 10, 15, 20, 25, 30)             # points


def epoch_s(ts_like) -> np.ndarray:
    return pd.DatetimeIndex(ts_like).as_unit("ns").asi8 // 10 ** 9


def load():
    its, zss = [], []
    for base, lo, hi, off in ((DEV, "2021-01-01", "2026-06-17", 0),
                              (SEAL, "2026-06-18", "2026-08-14",
                               10_000_000)):
        it = pd.read_parquet(
            f"{base}/daily/interactions.parquet",
            columns=["session", "zid", "tap_ts", "tap_no", "scale",
                     "app_side_above", "zlo_t", "zhi_t"])
        it["session"] = it["session"].astype(str)
        it = it[(it["session"] >= lo) & (it["session"] <= hi)]
        it["zid"] += off
        its.append(it)
        zs = pd.read_parquet(
            f"{base}/daily/zones.parquet",
            columns=["session", "zid", "price_rf", "hw_t", "scale",
                     "form_ts_val"])
        zs["session"] = zs["session"].astype(str)
        zs = zs[(zs["session"] >= lo) & (zs["session"] <= hi)]
        zs["zid"] += off
        zss.append(zs)
    return (pd.concat(its, ignore_index=True),
            pd.concat(zss, ignore_index=True))


def main() -> int:
    it, zs = load()
    cx = pd.read_parquet(ROOT / "outputs" / "p25a_break_crossings.parquet")
    b_cau = cx.set_index("zid")["cross_ts"].to_dict()
    adjs = []
    for base in (DEV, SEAL):
        a = pd.read_parquet(f"{base}/daily/roll_adjust.parquet")
        a["session"] = a["session"].astype(str)
        adjs.append(a[["session", "adj_t"]])
    adj = pd.concat(adjs).drop_duplicates("session") \
        .set_index("session")["adj_t"].to_dict()

    all_sess = sorted(set(zs["session"]))
    ford_map = {s: i for i, s in enumerate(all_sess)}
    z = zs.drop_duplicates("zid").set_index("zid")
    z["ford"] = z["session"].map(ford_map)
    # roll-free geometry for cross-session comparability
    z["lo_rf"] = z["price_rf"] - z["hw_t"]
    z["hi_rf"] = z["price_rf"] + z["hw_t"]
    tap_lists = it.sort_values("tap_ts").groupby("zid")["tap_ts"] \
        .apply(lambda x: np.array(x)).to_dict()

    res = it[it["app_side_above"] == 0].copy()
    ev1 = res[res["tap_no"] == 1].copy()
    evL = res[(res["tap_no"] >= 2)].copy()
    evL["b_cau"] = evL["zid"].map(b_cau).fillna(np.inf)
    evL = evL[evL["b_cau"] > evL["tap_ts"]]          # causal state filter
    print(f"child first-touch events: {len(ev1):,} | later-tap live "
          f"events: {len(evL):,}")

    rows, rowsL = [], []
    cache_mon, cache = None, None
    for sess in sorted(set(ev1["session"]) | set(evL["session"])):
        mon = sess[:7]
        root = DEV if sess <= "2026-06-17" else SEAL
        key = (root, mon)
        if key != cache_mon:
            b = pd.read_parquet(f"{root}/bars_1s/{mon}.parquet",
                                columns=["session", "ts", "open_t",
                                         "high_t", "low_t", "is_front"])
            b = b[b["is_front"]]
            b["session"] = b["session"].astype(str)
            cache_mon, cache = key, b
        sb = cache[cache["session"] == sess].sort_values("ts")
        if not len(sb):
            continue
        ep = epoch_s(sb["ts"])
        hi = sb["high_t"].to_numpy("float64")
        lo = sb["low_t"].to_numpy("float64")
        op = sb["open_t"].to_numpy("float64")
        so = ford_map[sess]
        adj_s = float(adj.get(sess, 0.0))
        cand = z[(z["ford"] <= so) & (z["ford"] >= so - 30)]
        czid = cand.index.to_numpy()
        cL = cand["lo_rf"].to_numpy() + adj_s      # session scale, ticks
        cH = cand["hi_rf"].to_numpy() + adj_s
        cF = cand["form_ts_val"].to_numpy()
        cSC = cand["scale"].to_numpy()
        cB = np.array([b_cau.get(x, np.inf) for x in czid])
        ctaps = [tap_lists.get(x, np.empty(0)) for x in czid]

        def comp_at(zid_c, t):
            """Causal overlap component of contacted zid at time t."""
            ntap = np.array([int((ta < t).sum()) for ta in ctaps])
            live = (cF <= t) & (cB > t) & (ntap < 6)
            ix = np.flatnonzero(live)
            if zid_c not in czid[ix]:
                # contacted zone must be live at its own first touch
                ix = np.append(ix, np.flatnonzero(czid == zid_c))
            L, H, zd = cL[ix], cH[ix], czid[ix]
            o = np.argsort(L)
            mem, RL, RU = [], None, None
            cur = None
            for k in o:
                if cur is None or L[k] > cur:
                    if mem and zid_c in mem:
                        break
                    mem, RL, cur = [zd[k]], L[k], H[k]
                else:
                    mem.append(zd[k])
                    cur = max(cur, H[k])
            RU = cur
            memi = np.flatnonzero(np.isin(czid, mem))
            prior = int(sum((ctaps[m] < t).sum() > 0 for m in memi))
            edges, depth = depth_profile(cL[memi], cH[memi])
            c2 = core_bands(edges, depth, 2)
            c2lo = np.nan
            if len(depth):
                m2 = depth >= 2
                if m2.any():
                    c2lo = float(edges[:-1][m2].min())
            return (mem, float(RL), float(RU), int(cSC[memi].max()),
                    prior, float(c2[1]), c2lo)

        for tapno, src, sink in ((1, ev1, rows), (0, evL, rowsL)):
            g = src[src["session"] == sess]
            for r in g.itertuples():
                j0 = int(np.searchsorted(ep, r.tap_ts, "left"))
                if j0 >= len(ep):
                    continue
                jend = int(np.searchsorted(ep, r.tap_ts + WIN_S,
                                           "right"))
                seg_hi = hi[j0:jend]
                seg_lo = lo[j0:jend]
                seg_ep = ep[j0:jend]
                ref = r.zlo_t
                aref = float(np.clip(op[j0], r.zlo_t, r.zhi_t))
                adv = np.maximum.accumulate(seg_hi) - ref
                fav = ref - np.minimum.accumulate(seg_lo)
                n = len(adv)
                rec = dict(session=sess, zid=int(r.zid),
                           tap_ts=float(r.tap_ts), scale=int(r.scale),
                           tap_no=int(r.tap_no), ref=float(ref),
                           aref_diff=float(aref - ref), n_fwd=n,
                           truncated=int(jend == len(ep)
                                         and ep[-1] < r.tap_ts
                                         + WIN_S - 1))
                mpt = ((r.tap_ts - 8 * 3600) % 86400) // 60
                rec["is_ny"] = int(390 <= mpt < 540)
                # threshold times stored as SECONDS from the tap (bars
                # may skip empty seconds; ordering by seconds == by bar)
                tf_idx = {}
                for F in FAVS:
                    tF = int(np.searchsorted(fav, F * 4, "left"))
                    tf_idx[F] = tF if tF < n else -1
                    rec[f"tf{F}"] = int(seg_ep[tF] - r.tap_ts) \
                        if tF < n else -1
                for Aa in ADVS:
                    tA = int(np.searchsorted(adv, Aa * 4, "left"))
                    key_a = str(Aa).replace(".", "_")
                    rec[f"ta{key_a}"] = int(seg_ep[tA] - r.tap_ts) \
                        if tA < n else -1
                for F in (10, 15, 20, 30):
                    tFi = tf_idx[F]
                    rec[f"mae{F}"] = float(adv[tFi] / 4) if tFi >= 0 \
                        else np.nan
                rec["mae_win"] = float(adv[-1] / 4)
                rec["mfe_win"] = float(fav[-1] / 4)
                if tapno == 1:
                    mem, RL, RU, smax, prior, c2l, c2lo = \
                        comp_at(int(r.zid), r.tap_ts)
                    rec.update(reg_lo=RL, reg_hi=RU,
                               reg_w=(RU - RL) / 4, reg_n=len(mem),
                               reg_smax=smax, reg_prior_tapped=prior,
                               reg_c2_longest=c2l / 4,
                               reg_mem="|".join(map(str, mem)))
                    run_hi = np.maximum.accumulate(seg_hi)
                    for F in (10, 20, 30):
                        tFi = tf_idx[F]
                        if tFi >= 0:
                            ext = float(run_hi[tFi])
                            rec[f"pen{F}"] = (ext - RL) / 4
                            rec[f"loc{F}"] = (
                                "child" if ext <= r.zhi_t else
                                "region" if ext <= RU else "beyond")
                            rec[f"core{F}"] = int(
                                np.isfinite(c2lo) and ext >= c2lo)
                        else:
                            rec[f"pen{F}"] = np.nan
                            rec[f"loc{F}"] = ""
                            rec[f"core{F}"] = 0
                sink.append(rec)

    df = pd.DataFrame(rows)
    dfL = pd.DataFrame(rowsL)
    df.to_parquet(ROOT / "outputs" / "p25a_events.parquet", index=False)
    dfL.to_parquet(ROOT / "outputs" / "p25a_events_latertap.parquet",
                   index=False)
    print(f"saved: {len(df):,} first-touch events "
          f"({df['is_ny'].sum():,} NY-AM), {len(dfL):,} later-tap")
    print(f"aref==ref share: {(df['aref_diff'] == 0).mean():.3f} | "
          f"fresh regions (prior_tapped==0): "
          f"{(df['reg_prior_tapped'] == 0).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
