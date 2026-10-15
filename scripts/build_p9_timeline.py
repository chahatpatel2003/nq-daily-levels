#!/usr/bin/env python
"""P9 timeline build: causal confirmation features around every frozen
resistance-zone interaction.

Per interaction, from the dense 1-second path and aggressor flow:
  * checkpoint features at τ ∈ {15,30,60,120,180,300,600,900} s after touch
    (path state, penetration, rejection progress, velocity, time-above,
    closes-below share, buy/sell/delta, intensity vs trailing baseline,
    progress-per-buy-effort);
  * fire times + at-fire adverse/completed stats for every registered rule
    cell (R1-R7, grids fixed in RESEARCH_PLAN §20.1), searched within
    1800 s of touch.

Conventions (resistance, approached from below): near edge = zlo, far =
zhi; rejection = downward from the near edge; penetration = upward from it.
Output: p9_timeline.parquet, one wide row per interaction.
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

from nqlevels.features.interactions import build_session_data                  # noqa: E402
from nqlevels.io.raw import assert_not_raw                                     # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402
from run_p6_spatial import add_labels                                          # noqa: E402

CONFIGS = str(ROOT / "configs")
B1S_DIR = "/Volumes/T9/nq_levels/bars_1s"
KEY = ["session", "zid", "tap_sec"]
CPS = (15, 30, 60, 120, 180, 300, 600, 900)
CAP = 1800
R1_H = (30, 60, 120)
R2 = [(t, p) for t in (60, 180, 300) for p in (2.5, 5.0, 7.5)]
R3 = [(b, h) for b in ("far", "mid", "near") for h in (30, 60)]
R4_V = (5.0, 7.5, 10.0)
R5_T = (60, 180)
R6_T = (120, 300)


def first_run(b: np.ndarray, h: int):
    """Start index of first True-run of length >= h, else None."""
    if not b.any():
        return None
    db = np.diff(np.concatenate(([0], b.view(np.int8), [0])))
    st = np.flatnonzero(db == 1)
    en = np.flatnonzero(db == -1)
    ok = np.flatnonzero(en - st >= h)
    return int(st[ok[0]]) if ok.size else None


def main() -> int:
    t0c = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    re = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    it = pd.read_parquet(d0 / "interactions.parquet",
                         columns=KEY + ["price_t", "zhi_t", "zlo_t"]).drop_duplicates(KEY)
    re = re.merge(it, on=KEY, how="left", validate="1:1")
    pop = re[(re["side_above"] == 0) & (re["n_fwd"] >= 1800)
             & re["y"].notna()].copy()
    pop_by = {s: g for s, g in pop.groupby("session", sort=False)}
    print(f"resistance interactions to walk: {len(pop):,}")

    b1m_all = pd.read_parquet(d0 / "bars_1m.parquet")
    b1m_all["ts"] = pd.DatetimeIndex(b1m_all["ts"]).tz_convert("UTC")
    b1m_by = {s: g.sort_values("ts").reset_index(drop=True)
              for s, g in b1m_all[b1m_all["is_front"]].groupby("session", sort=False)}
    del b1m_all

    rows = []
    n_done = 0
    for fpath in sorted(glob.glob(f"{B1S_DIR}/[!._]*.parquet")):
        b1s_m = pd.read_parquet(fpath)
        b1s_m = b1s_m[b1s_m["is_front"]]
        for s, b1s in b1s_m.groupby("session", sort=True):
            g = pop_by.get(s)
            if g is None or s not in b1m_by:
                continue
            sd = build_session_data(b1s.sort_values("ts").reset_index(drop=True),
                                    b1m_by[s])
            n = sd.cl.size
            for r in g.itertuples():
                t = int(r.tap_sec)
                zlo, zhi = float(r.zlo_t), float(r.zhi_t)
                mid = (zlo + zhi) / 2.0
                e = min(n, t + CAP)
                cl = sd.cl[t:e]
                hi = sd.hi[t:e]
                lo = sd.lo[t:e]
                m = cl.size
                rev = np.clip(zlo - np.minimum.accumulate(lo), 0, None)
                pen = np.clip(np.maximum.accumulate(hi) - zlo, 0, None)
                below = cl < zlo
                above_far = cl > zhi
                cum_below = np.concatenate(([0], np.cumsum(below)))
                cum_above = np.concatenate(([0], np.cumsum(above_far)))
                base_rate = (sd.cum_vol[t] - sd.cum_vol[max(0, t - 10800)]) \
                    / max(1, t - max(0, t - 10800))
                out = {"session": s, "zid": int(r.zid), "tap_sec": t}
                for tau in CPS:
                    j = min(tau, m - 1)
                    v = sd.cum_vol[t + j] - sd.cum_vol[t]
                    d_ = sd.cum_dlt[t + j] - sd.cum_dlt[t]
                    buy = (v + d_) / 2.0
                    p_ = f"cp{tau}"
                    out[f"{p_}_pen"] = float(pen[j]) / 4
                    out[f"{p_}_rev"] = float(rev[j]) / 4
                    out[f"{p_}_now"] = float(cl[j] - zlo) / 4
                    out[f"{p_}_above_s"] = int(cum_above[j + 1])
                    out[f"{p_}_below_sh"] = float(cum_below[j + 1] / (j + 1))
                    out[f"{p_}_vel60"] = float(cl[j] - cl[max(0, j - 60)]) / 4
                    out[f"{p_}_vol"] = float(v)
                    out[f"{p_}_delta_sh"] = float(d_ / v) if v > 0 else 0.0
                    out[f"{p_}_intx"] = float((v / (j + 1)) / base_rate) \
                        if base_rate > 0 else np.nan
                    out[f"{p_}_eff"] = float(pen[j] / 4 / (buy / 1000 + 0.1))
                    out[f"{p_}_d60"] = float((sd.cum_dlt[t + j]
                                              - sd.cum_dlt[t + max(0, j - 60)]))

                def fire_stats(fj):
                    return {"adv": float(pen[min(fj, m - 1)]) / 4,
                            "done": float(rev[min(fj, m - 1)]) / 4}

                def put(name, fj):
                    if fj is None or fj >= m:
                        out[f"{name}_t"] = np.nan
                        return
                    st_ = fire_stats(fj)
                    out[f"{name}_t"] = int(fj)
                    out[f"{name}_adv"] = st_["adv"]
                    out[f"{name}_done"] = st_["done"]

                for h in R1_H:
                    fj = first_run(below, h)
                    put(f"r1h{h}", None if fj is None else fj + h)
                for tau, p in R2:
                    ok = m > tau and pen[tau] <= 4 * p and cl[tau] < zlo
                    put(f"r2t{tau}p{p}", tau if ok else None)
                fb = int(np.argmax(hi > zhi)) if (hi > zhi).any() else None
                for bname, h in R3:
                    if fb is None:
                        put(f"r3{bname}h{h}", None)
                        continue
                    lvl = {"far": zhi, "mid": mid, "near": zlo}[bname]
                    fj = first_run(cl[fb:] < lvl, h)
                    put(f"r3{bname}h{h}", None if fj is None else fb + fj + h)
                for v_ in R4_V:
                    j = int(np.searchsorted(rev, 4 * v_))
                    put(f"r4v{v_}", j if j <= 60 and j < m else None)
                for tau in R5_T:
                    ok = m > tau and pen[tau] <= 20 and \
                        (sd.cum_vol[t + tau] - sd.cum_vol[t]) >= 1.5 * base_rate * tau
                    put(f"r5t{tau}", tau if ok else None)
                for T in R6_T:
                    j = int(np.searchsorted(cum_above[1:], T))
                    put(f"r6T{T}", j if j < m else None)
                # R7: first 1-minute close fully below the near edge
                m1 = int(sd.min_of_sec[t])
                fj7 = None
                for mm in range(m1 + 1, min(m1 + 31, sd.m_cl.size)):
                    if sd.m_cl[mm] < zlo:
                        js = int(np.searchsorted(sd.min_of_sec, mm + 1)) - t
                        if 0 < js < m:
                            fj7 = js
                        break
                put("r7", fj7)
                out["sweep_occurred"] = int(fb is not None) if fb is not None else 0
                out["fb_t"] = fb if fb is not None else np.nan
                rows.append(out)
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {s}: {len(rows):,} rows, {_time.time()-t0c:.0f}s",
                      flush=True)
        del b1s_m

    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    df.to_parquet(assert_not_raw(d0 / "p9_timeline.parquet"), index=False,
                  compression="zstd")
    print(f"\np9_timeline: {len(df):,} rows x {df.shape[1]} cols "
          f"({(_time.time()-t0c)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
