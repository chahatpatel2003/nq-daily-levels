#!/usr/bin/env python
"""P23: mechanical entry backtests around frozen resistance child zones.
Families (registered, fixed): A midpoint-limit; B engulfing (BODY/RANGE
x in-zone/post-touch x 30s/1m/2m/3m/5m); C rejection close; D sweep+
reclaim (D1/D2); E touch-low break. Zone-anchored exits SL=U+H, TP=L-H.
Entry only at completed-bar closes (confirmations) or limit fills.
Stop-first on same-second ambiguity. Costs 1.0pt RT. One concurrent
trade per zone. Windows: B2/E 300s from latest contact/touch; C/D first
qualifying bar within 900s. All conventions fixed a priori."""
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

from nqlevels.features.interactions import build_session_data                  # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402

SEAL = "/Volumes/T9/nq_levels/sealed_holdout_1/derived/daily"
DEV = "/Volumes/T9/nq_levels/daily"
B1S_DEV = "/Volumes/T9/nq_levels/bars_1s"
B1S_SEAL = "/Volumes/T9/nq_levels/sealed_holdout_1/derived/bars_1s"
OUT = "/Volumes/T9/nq_levels/daily/p23_trades.parquet"
TFS = (30, 60, 120, 180, 300)
COST = 4.0          # ticks RT (1.0 pt)
WIN_CONF = 900      # C/D scan window (s)
WIN_FAST = 300      # B2/E windows (s)


def load_taps():
    frames = []
    for daily, lo, hi in ((DEV, "2021-01-01", "2026-06-17"),
                          (SEAL, "2026-06-18", "2026-08-14")):
        it = pd.read_parquet(f"{daily}/interactions.parquet",
                             columns=["session", "zid", "tap_sec", "tap_ts",
                                      "tap_no", "app_side_above", "zlo_t",
                                      "zhi_t", "seg", "age_min"])
        it["session"] = it["session"].astype(str)
        it = it[(it["session"] >= lo) & (it["session"] <= hi)
                & (it["app_side_above"] == 0)]
        zs = pd.read_parquet(f"{daily}/zones.parquet",
                             columns=["zid", "scale", "form_min"])
        if daily != DEV:
            it["zid"] += 10_000_000
            zs = zs.copy()
            zs["zid"] += 10_000_000
        it = it.merge(zs.drop_duplicates("zid"), on="zid", how="left")
        frames.append(it)
    return pd.concat(frames, ignore_index=True)


def bars_of(cl, hi, lo, tf):
    n = len(cl) // tf
    if n == 0:
        return None
    H = hi[:n * tf].reshape(n, tf).max(1)
    L = lo[:n * tf].reshape(n, tf).min(1)
    C = cl[:n * tf].reshape(n, tf)[:, -1]
    O = cl[:n * tf].reshape(n, tf)[:, 0]
    return O, H, L, C


def passage(hi, lo, e_idx, sl, tp, n):
    """First-passage from e_idx: returns (exit_idx, res) res=+1 tp/-1 sl/
    0 end; stop-first on ties."""
    h = hi[e_idx:]
    l_ = lo[e_idx:]
    up = np.flatnonzero(h >= sl)
    dn = np.flatnonzero(l_ <= tp)
    tu = up[0] if up.size else 10 ** 9
    td = dn[0] if dn.size else 10 ** 9
    if tu == 10 ** 9 and td == 10 ** 9:
        return n - 1, 0
    if tu <= td:
        return e_idx + tu, -1
    return e_idx + td, +1


def mfe_mae(hi, lo, e_idx, x_idx, e_px):
    seg_h = hi[e_idx:x_idx + 1]
    seg_l = lo[e_idx:x_idx + 1]
    return float(e_px - seg_l.min()), float(seg_h.max() - e_px)


def main() -> int:
    t0c = _time.time()
    taps = load_taps()
    by = {s: g for s, g in taps.groupby("session", sort=False)}
    d0 = layer_dir("daily", str(ROOT / "configs"))
    b1m_all = pd.read_parquet(d0 / "bars_1m.parquet")
    b1m_all["ts"] = pd.DatetimeIndex(b1m_all["ts"]).tz_convert("UTC")
    b1m_by = {str(s): g.sort_values("ts").reset_index(drop=True)
              for s, g in b1m_all[b1m_all["is_front"]]
              .groupby("session", sort=False)}
    del b1m_all
    b1m_seal = pd.read_parquet(f"{SEAL}/bars_1m.parquet")
    b1m_seal["ts"] = pd.DatetimeIndex(b1m_seal["ts"]).tz_convert("UTC")
    for s, g in b1m_seal[b1m_seal["is_front"]].groupby("session"):
        ss = str(s)
        if ss >= "2026-06-18":
            b1m_by[ss] = g.sort_values("ts").reset_index(drop=True)
    del b1m_seal

    files = sorted(glob.glob(f"{B1S_DEV}/[!._]*.parquet")) + \
        sorted(glob.glob(f"{B1S_SEAL}/2026-0[678]*.parquet"))
    rows = []
    n_geom_reject = 0
    n_done = 0
    for fpath in files:
        is_seal = "sealed" in fpath
        b1s_m = pd.read_parquet(fpath)
        b1s_m = b1s_m[b1s_m["is_front"]]
        for s, b1s in b1s_m.groupby("session", sort=True):
            ss = str(s)
            if is_seal and ss < "2026-06-18":
                continue
            if not is_seal and ss > "2026-06-17":
                continue
            g = by.get(ss if ss in by else s)
            if g is None or ss not in b1m_by:
                continue
            sd = build_session_data(
                b1s.sort_values("ts").reset_index(drop=True), b1m_by[ss])
            n = sd.cl.size
            cl, hi, lo = sd.cl, sd.hi, sd.lo
            bars = {tf: bars_of(cl, hi, lo, tf) for tf in TFS}
            open_until = {}          # zid -> exit idx (one trade per zone)
            for r in g.sort_values("tap_sec").itertuples():
                t = int(r.tap_sec)
                L, U = float(r.zlo_t), float(r.zhi_t)
                Hh = U - L
                if Hh <= 0 or t >= n - 10:
                    continue
                base = dict(session=ss, zid=int(r.zid), tap_sec=t,
                            tap_no=int(r.tap_no), seg=int(r.seg),
                            scale=int(r.scale) if np.isfinite(r.scale)
                            else -1, L=L, U=U, H=Hh,
                            age_min=float(r.age_min),
                            first=int(r.tap_no == 1))
                SL, TP = U + Hh, L - Hh
                trades_this_tap = []
                # ---- A: midpoint limit ----
                M = (U + L) / 2.0
                w_end = min(n - 1, t + WIN_CONF)
                fill = np.flatnonzero(hi[t:w_end] >= M)
                if fill.size:
                    e_idx = t + int(fill[0])
                    x_idx, res = passage(hi, lo, e_idx, SL, TP, n)
                    pnl = (M - TP if res == 1 else M - SL if res == -1
                           else M - cl[x_idx]) - COST
                    mae, mfe_ = mfe_mae(hi, lo, e_idx, x_idx, M)[1], \
                        mfe_mae(hi, lo, e_idx, x_idx, M)[0]
                    risk = SL - M
                    trades_this_tap.append(dict(base, strat="A_mid",
                        tf=0, e_idx=e_idx, e_px=M, sl=SL, tp=TP,
                        risk=risk, x_idx=x_idx, res=res,
                        netR=pnl / risk, rr=(M - TP) / risk,
                        mfe=mfe_ / risk, mae=mae / risk))
                # ---- bar-based families ----
                for tf in TFS:
                    if bars[tf] is None:
                        continue
                    O, Hb, Lb, Cb = bars[tf]
                    b_t = t // tf
                    b_hi = min(len(Cb) - 1, (t + WIN_CONF) // tf)
                    fired = {}
                    for k in range(max(1, b_t), b_hi + 1):
                        end_s = (k + 1) * tf      # entry time = bar close
                        if end_s >= n - 5:
                            break
                        e_px = float(Cb[k])
                        bear = Cb[k] < O[k]
                        body_eng = (bear and Cb[k - 1] > O[k - 1]
                                    and O[k] >= Cb[k - 1]
                                    and Cb[k] <= O[k - 1])
                        rng_eng = (bear and Hb[k] >= Hb[k - 1]
                                   and Lb[k] <= Lb[k - 1])
                        inzone = Hb[k] >= L and Lb[k] <= U
                        # last zone contact second before bar end
                        seg_hi = hi[t:end_s]
                        lc = np.flatnonzero(seg_hi >= L)
                        last_contact = t + (lc[-1] if lc.size else 0)
                        post5 = (end_s - last_contact) <= WIN_FAST \
                            and lc.size > 0
                        cands = []
                        if body_eng and inzone:
                            cands.append("B_body_inz")
                        if body_eng and post5:
                            cands.append("B_body_p5")
                        if rng_eng and inzone:
                            cands.append("B_rng_inz")
                        if rng_eng and post5:
                            cands.append("B_rng_p5")
                        if Hb[k] >= L and Cb[k] < L:
                            cands.append("C_rej")
                        if Hb[k] > U or (k > b_t and
                                         Hb[b_t:k].max() > U):
                            swept = hi[t:end_s].max() > U
                        else:
                            swept = False
                        if swept and L <= Cb[k] <= U:
                            cands.append("D1_inz")
                        if swept and Cb[k] < L:
                            cands.append("D2_below")
                        for nm in cands:
                            if nm in fired:
                                continue
                            fired[nm] = True
                            risk = SL - e_px
                            rr = (e_px - TP) / risk if risk > 0 else -1
                            if risk <= 0 or e_px <= TP:
                                n_geom_reject += 1
                                continue
                            e_idx = end_s
                            x_idx, res = passage(hi, lo, e_idx, SL, TP, n)
                            pnl = (e_px - TP if res == 1 else
                                   e_px - SL if res == -1 else
                                   e_px - cl[x_idx]) - COST
                            mfe_, mae = mfe_mae(hi, lo, e_idx, x_idx, e_px)
                            trades_this_tap.append(dict(base, strat=nm,
                                tf=tf, e_idx=e_idx, e_px=e_px, sl=SL,
                                tp=TP, risk=risk, x_idx=x_idx, res=res,
                                netR=pnl / risk, rr=rr, mfe=mfe_ / risk,
                                mae=mae / risk))
                    # ---- E: touch-candle low break ----
                    if b_t >= len(Lb):
                        continue
                    tl = float(Lb[b_t])
                    trig_lvl = tl - 1.0
                    e_end = min(n - 5, (b_t + 1) * tf + WIN_FAST)
                    start = (b_t + 1) * tf
                    if start < e_end:
                        brk = np.flatnonzero(lo[start:e_end] <= trig_lvl)
                        if brk.size:
                            e_idx = start + int(brk[0])
                            e_px = trig_lvl
                            risk = SL - e_px
                            if risk > 0 and e_px > TP:
                                x_idx, res = passage(hi, lo, e_idx, SL,
                                                     TP, n)
                                pnl = (e_px - TP if res == 1 else
                                       e_px - SL if res == -1 else
                                       e_px - cl[x_idx]) - COST
                                mfe_, mae = mfe_mae(hi, lo, e_idx, x_idx,
                                                    e_px)
                                trades_this_tap.append(dict(base,
                                    strat="E_lowbrk", tf=tf, e_idx=e_idx,
                                    e_px=e_px, sl=SL, tp=TP, risk=risk,
                                    x_idx=x_idx, res=res,
                                    netR=pnl / risk,
                                    rr=(e_px - TP) / risk,
                                    mfe=mfe_ / risk, mae=mae / risk))
                            else:
                                n_geom_reject += 1
                # one concurrent trade per zone: drop entries while open
                key = int(r.zid)
                for tr in trades_this_tap:
                    ou = open_until.get((key, tr["strat"], tr["tf"]), -1)
                    if tr["e_idx"] < ou:
                        continue
                    open_until[(key, tr["strat"], tr["tf"])] = tr["x_idx"]
                    rows.append(tr)
            n_done += 1
        print(f"  {Path(fpath).name}: sessions {n_done} rows {len(rows):,}"
              f" ({(_time.time()-t0c)/60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    df["year"] = df["session"].str[:4]
    df.to_parquet(OUT)
    print(f"wrote {OUT}: {len(df):,} trades | geometry rejects "
          f"{n_geom_reject:,} | {(_time.time()-t0c)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
