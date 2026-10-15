#!/usr/bin/env python
"""P5 live-display simulation driver.

Chronological replay of 2021-2026 with the Phase-4 registry engine, plus at
every 15-minute refresh: score ALL eligible zones with the walk-forward model
(trained on tap rows of strictly earlier years), rank, and record the full
scored pool with display ranks — no conditioning on any future event.

Also emits: lead-time rescored features for every tap (advance notice), and
random-moment ±15-pt race outcomes (the side-asymmetry noise baseline).

Outputs on T9: p5_pool.parquet (every scored eligible zone at every refresh,
rank 1..4 = displayed), p5_leads.parquet, p5_races.parquet.
"""
from __future__ import annotations

import glob
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nqlevels.features.display import (build_aux, derive, eligible, snap_of,   # noqa: E402
                                       upd_before, zone_row)
from nqlevels.features.interactions import build_session_data, process_session  # noqa: E402
from nqlevels.features.zones import epoch_s, make_zones                         # noqa: E402
from nqlevels.io.raw import assert_not_raw                                      # noqa: E402
from nqlevels.io.store import layer_dir, read_layer                             # noqa: E402
from build_interactions import make_vol_feats, seg_codes                        # noqa: E402
from run_p4_model import T1_FEATS, T2_FEATS                                     # noqa: E402
from run_p4_strongweak import load_frame                                        # noqa: E402

CONFIGS = str(ROOT / "configs")
B1S_DIR = "/Volumes/T9/nq_levels/bars_1s"
REFRESH_MIN = 15
K = 4
SEP_ATR = 0.05
LEADS = (1800, 900, 300, 120)
POOL_COLS = ["session", "t_sec", "ts", "zid", "rank", "score_T1", "score_T2",
             "dist_atr", "app_side_above", "price_t", "tap_no", "age_sessions",
             "age_min", "scale", "flipped", "seg", "prior_succ_share",
             "dep_vel", "rv60_atr", "atr_pctile", "mins_since_last", "atr_t"]


def train_models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    f = load_frame()
    f = f[f["y"].notna()]
    models = {}
    for Y in (2022, 2023, 2024, 2025, 2026):
        tr = f[f["year"] < Y]
        if len(tr) < 8000:
            continue
        pair = []
        for feats in (T1_FEATS, T2_FEATS):
            g = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=4,
                l2_regularization=1.0, random_state=0)
            g.fit(tr[feats].to_numpy("float64"), tr["y"].to_numpy("float64"))
            pair.append(g)
        models[Y] = pair
        print(f"  model {Y}: trained on {len(tr):,} taps (< {Y})", flush=True)
    return models


def rank_pool(df: pd.DataFrame, atr: float) -> np.ndarray:
    """Greedy top-K by score_T1 with >= SEP_ATR separation, per refresh."""
    rank = np.zeros(len(df), dtype="int8")
    order = np.argsort(-df["score_T1"].to_numpy())
    px = df["price_t"].to_numpy()
    taken = []
    for i in order:
        if len(taken) >= K:
            break
        if any(abs(px[i] - p) < SEP_ATR * atr for p in taken):
            continue
        taken.append(px[i])
        rank[i] = len(taken)
    return rank


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    print("training walk-forward models ...")
    models = train_models()

    ctx = pd.read_parquet(d0 / "context.parquet",
                          columns=["session", "atr14_t", "atr_pctile", "dow"])
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    ctxi = ctx.set_index("session")
    adj = pd.read_parquet(d0 / "roll_adjust.parquet").set_index("session")
    b1m_all = pd.read_parquet(d0 / "bars_1m.parquet")
    b1m_all["ts"] = pd.DatetimeIndex(b1m_all["ts"]).tz_convert("UTC")
    b1m_by = {s: g.sort_values("ts").reset_index(drop=True)
              for s, g in b1m_all[b1m_all["is_front"]].groupby("session", sort=False)}
    del b1m_all
    prof_all = read_layer("profiles", CONFIGS)
    prof_all = prof_all[prof_all["is_front"]]
    prof_all = prof_all.groupby(["session", "price_t"], as_index=False)["volume"].sum()
    prof_by = {s: g for s, g in prof_all.groupby("session", sort=False)}
    del prof_all
    ord_of = {s: i for i, s in enumerate(sorted(ctxi.index))}

    registry: dict = {}
    pool_frames, lead_frames, races = [], [], []
    prof_deque, ext_deque = [], []
    zid_counter = 0
    cur_chain = None
    rng = np.random.default_rng(5)
    n_done = 0
    for fpath in sorted(glob.glob(f"{B1S_DIR}/[!._]*.parquet")):
        b1s_m = pd.read_parquet(fpath)
        b1s_m = b1s_m[b1s_m["is_front"]]
        by_sess = {s: g for s, g in b1s_m.groupby("session", sort=False)}
        for s in sorted(by_sess):
            if s not in ctxi.index or s not in adj.index or s not in b1m_by:
                continue
            atr = float(ctxi.loc[s, "atr14_t"])
            if not np.isfinite(atr) or atr <= 0:
                continue
            adj_s = float(adj.loc[s, "adj_t"])
            chain = int(adj.loc[s, "chain"])
            if cur_chain is not None and chain != cur_chain:
                registry.clear()
            cur_chain = chain
            sess_ord = ord_of[s]
            b1s = by_sess[s].sort_values("ts").reset_index(drop=True)
            b1m = b1m_by[s]
            if len(b1s) < 3600 or len(b1m) < 120:
                continue
            sd = build_session_data(b1s, b1m)
            m_pt = ((epoch_s(b1m["ts"]) - 8 * 3600) % 86400) // 60
            segm = seg_codes(m_pt)
            new = make_zones(b1m, atr, adj_s, sess_ord, segm, zid_counter)
            zid_counter += len(new)
            hi20 = max(x[0] for x in ext_deque) if ext_deque else np.nan
            lo20 = min(x[1] for x in ext_deque) if ext_deque else np.nan
            vf = make_vol_feats(prof_deque, adj_s, atr)
            dow = int(ctxi.loc[s, "dow"])
            apc = float(ctxi.loc[s, "atr_pctile"])

            zlist = list(registry.values()) + new
            snaps = {z.zid: snap_of(z) for z in zlist}
            st: dict = {}
            rows = process_session(registry, new, sd, atr, adj_s, chain,
                                   sess_ord, dow, apc, vf, (hi20, lo20),
                                   state_out=st)
            upd = st.get("upd", {})

            year = s.year
            if year in models:
                auxs = {z.zid: build_aux(z, sd, adj_s, sess_ord) for z in zlist}
                gT1, gT2 = models[year]
                # ---- refresh pass ------------------------------------------
                n_min = sd.m_cl.size
                pool_rows = []
                for m in range(REFRESH_MIN, n_min, REFRESH_MIN):
                    t = int(np.searchsorted(sd.min_of_sec, m))
                    if t >= sd.cl.size - 1:
                        break
                    for z in zlist:
                        u = upd.get(z.zid, [])
                        if not eligible(z, snaps[z.zid], auxs[z.zid], u, sd, t,
                                        sess_ord, atr):
                            continue
                        r = zone_row(z, snaps[z.zid], auxs[z.zid], u, sd, t,
                                     atr, adj_s, sess_ord, dow, apc, vf,
                                     (hi20, lo20))
                        r["t_sec"] = t
                        pool_rows.append(r)
                if pool_rows:
                    pf = derive(pd.DataFrame(pool_rows))
                    pf["score_T1"] = gT1.predict_proba(
                        pf[T1_FEATS].to_numpy("float64"))[:, 1]
                    pf["score_T2"] = gT2.predict_proba(
                        pf[T2_FEATS].to_numpy("float64"))[:, 1]
                    pf["dist_atr"] = (pf["price_t"]
                                      - sd.cl[pf["t_sec"].to_numpy() - 1]).abs() / atr
                    pf["rank"] = 0
                    for _, g in pf.groupby("t_sec"):
                        pf.loc[g.index, "rank"] = rank_pool(g, atr)
                    pf["session"] = s
                    pf["ts"] = pf["tap_ts"]
                    keep = pf[POOL_COLS].copy()
                    for c in keep.columns:
                        # ts stays float64: epoch seconds lose ~2-minute
                        # precision in float32, which would corrupt joins
                        if keep[c].dtype == "float64" and c != "ts":
                            keep[c] = keep[c].astype("float32")
                    pool_frames.append(keep)

                # ---- lead-time rescoring for every tap ---------------------
                lead_rows = []
                for row in rows:
                    zid = row["zid"]
                    z = next((x for x in zlist if x.zid == zid), None)
                    if z is None:
                        continue
                    for L in LEADS:
                        tl = int(row["tap_sec"]) - L
                        if tl < 120:
                            continue
                        u = upd_before(upd.get(zid, []), float(sd.sec_ts[tl]))
                        r = zone_row(z, snaps[zid], auxs[zid], u, sd, tl,
                                     atr, adj_s, sess_ord, dow, apc, vf,
                                     (hi20, lo20))
                        r["lead_s"] = L
                        r["tap_sec_of"] = int(row["tap_sec"])
                        r["dist_atr_lead"] = abs(r["price_t"]
                                                 - sd.cl[tl - 1]) / atr
                        lead_rows.append(r)
                if lead_rows:
                    lf = derive(pd.DataFrame(lead_rows))
                    lf["score_T1"] = gT1.predict_proba(
                        lf[T1_FEATS].to_numpy("float64"))[:, 1]
                    lf["score_T2"] = gT2.predict_proba(
                        lf[T2_FEATS].to_numpy("float64"))[:, 1]
                    lf["session"] = s
                    lead_frames.append(lf[["session", "zid", "tap_sec_of",
                                           "lead_s", "score_T1", "score_T2",
                                           "dist_atr_lead",
                                           "app_side_above"]].copy())

                # ---- random-moment race baseline ---------------------------
                n = sd.cl.size
                for t in rng.integers(3600, max(3601, n - 7200), 20):
                    p0 = sd.cl[t]
                    up = np.flatnonzero(sd.hi[t:t + 7200] >= p0 + 60.0)
                    dn = np.flatnonzero(sd.lo[t:t + 7200] <= p0 - 60.0)
                    if up.size and dn.size:
                        w = "dn" if dn[0] < up[0] else "up"
                    elif dn.size:
                        w = "dn"
                    elif up.size:
                        w = "up"
                    else:
                        w = "none"
                    races.append({"session": s, "t": int(t), "winner": w,
                                  "seg": int(sd.seg_of_sec[t])})

            # trailing structures
            p = prof_by.get(s)
            if p is not None and len(p):
                prof_deque.append(((p["price_t"].to_numpy("float64") - adj_s),
                                   p["volume"].to_numpy("float64")))
                if len(prof_deque) > 5:
                    prof_deque.pop(0)
            ext_deque.append((float(b1m["high_t"].max()) - adj_s,
                              float(b1m["low_t"].min()) - adj_s))
            if len(ext_deque) > 20:
                ext_deque.pop(0)
            n_done += 1
            if n_done % 100 == 0:
                np_ = sum(len(x) for x in pool_frames)
                print(f"  {s}: {np_:,} pool rows, "
                      f"{sum(len(x) for x in lead_frames):,} lead rows, "
                      f"{_time.time()-t0:.0f}s", flush=True)
        del b1s_m, by_sess

    pool = pd.concat(pool_frames, ignore_index=True)
    pool.to_parquet(assert_not_raw(d0 / "p5_pool.parquet"), index=False,
                    compression="zstd")
    leads = pd.concat(lead_frames, ignore_index=True)
    leads.to_parquet(assert_not_raw(d0 / "p5_leads.parquet"), index=False,
                     compression="zstd")
    rc = pd.DataFrame(races)
    rc.to_parquet(assert_not_raw(d0 / "p5_races.parquet"), index=False)
    print(f"\npool {len(pool):,} rows ({pool['session'].nunique()} sessions) | "
          f"leads {len(leads):,} | races {len(rc):,} | "
          f"{(_time.time()-t0)/60:.1f} min")
    print(f"displayed rows (rank>=1): {int((pool['rank'] > 0).sum()):,} | "
          f"eligible/refresh median: "
          f"{pool.groupby(['session', 't_sec']).size().median():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
