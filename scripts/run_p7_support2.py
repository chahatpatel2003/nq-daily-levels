#!/usr/bin/env python
"""P7 support, second attack: the sweep-termination-cluster anchor.

Hypothesis (the last registered support family): support reactions anchor at
the local concentration of PRIOR successful turn prices below the pivot low
— "where sweeps historically terminate" — rather than at the low itself.

For every support event: candidate anchor A = the maximiser of
prior-30-session successful-turn density (±6t kernel) within
[pivot − 40 pts, pivot − 2 pts], requiring ≥ 2 prior turns; anchor band =
A ± the event's own half-width; A and its ±10-pt shadows walked from the
same t0 with the standard walker. Verdict by peak test (A vs its own
neighbours), frozen on 2021-2023 and confirmed on 2024-2026.
"""
from __future__ import annotations

import glob
import json
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
from nqlevels.labels.enrich import enriched_metrics, first_touch_after         # noqa: E402
from nqlevels.labels.sweep import classify                                     # noqa: E402
from run_p6_spatial import add_labels                                          # noqa: E402
from run_p7_support import peak_test                                           # noqa: E402

CONFIGS = str(ROOT / "configs")
B1S_DIR = "/Volumes/T9/nq_levels/bars_1s"
KEY = ["session", "zid", "tap_sec"]
T0_S = 900


def main() -> int:
    t0c = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    re = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    it = pd.read_parquet(d0 / "interactions.parquet",
                         columns=KEY + ["price_t", "zhi_t", "zlo_t"]).drop_duplicates(KEY)
    re = re.merge(it, on=KEY, how="left", validate="1:1")
    sup = re[(re["side_above"] == 1) & re["has_t0"] & ~re["inside_t0"]
             & (re["n_fwd"] >= 1800)].copy()
    # prior successful turn prices (roll-free), any side, from the enriched set
    ok = re[(re["y"] == 1) & re["adverse_before_rev15_t"].notna()].copy()
    below = ok["side"] == "below"
    near = np.where(below, ok["zhi_t"], ok["zlo_t"])
    sgn = np.where(below, -1.0, 1.0)
    ok["turn_px"] = near + sgn * ok["adverse_before_rev15_t"]
    adj = pd.read_parquet(d0 / "roll_adjust.parquet").set_index("session")["adj_t"]
    ctx = pd.read_parquet(d0 / "context.parquet", columns=["session"])
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    ord_of = {s: i for i, s in enumerate(sorted(ctx["session"]))}
    ok["ord"] = ok["session"].map(ord_of)
    ok["turn_rf"] = ok["turn_px"] - ok["session"].map(adj)
    ok = ok.sort_values("ord")
    ok_ord = ok["ord"].to_numpy()
    ok_rf = ok["turn_rf"].to_numpy("float64")

    b1m_all = pd.read_parquet(d0 / "bars_1m.parquet")
    b1m_all["ts"] = pd.DatetimeIndex(b1m_all["ts"]).tz_convert("UTC")
    b1m_by = {s: g.sort_values("ts").reset_index(drop=True)
              for s, g in b1m_all[b1m_all["is_front"]].groupby("session", sort=False)}
    del b1m_all
    sup_by = {s: g for s, g in sup.groupby("session", sort=False)}

    rows = []
    n_done = 0
    for fpath in sorted(glob.glob(f"{B1S_DIR}/[!._]*.parquet")):
        b1s_m = pd.read_parquet(fpath)
        b1s_m = b1s_m[b1s_m["is_front"]]
        for s, b1s in b1s_m.groupby("session", sort=True):
            g = sup_by.get(s)
            if g is None or s not in b1m_by:
                continue
            o = ord_of[s]
            a0, a1 = np.searchsorted(ok_ord, o - 30), np.searchsorted(ok_ord, o)
            t_rf = np.sort(ok_rf[a0:a1])
            if t_rf.size < 5:
                continue
            adj_s = float(adj.loc[s])
            t_cur = t_rf + adj_s
            sd = build_session_data(b1s.sort_values("ts").reset_index(drop=True),
                                    b1m_by[s])
            p = sd.path
            for r in g.itertuples():
                lo_b, hi_b = r.price_t - 160.0, r.price_t - 8.0
                cand = np.arange(lo_b, hi_b + 4, 4.0)
                dens = np.array([np.searchsorted(t_cur, x + 6, "right")
                                 - np.searchsorted(t_cur, x - 6) for x in cand])
                if dens.max() < 2:
                    continue
                A = float(cand[int(np.argmax(dens))])
                t0 = int(r.tap_sec) - T0_S
                if t0 < 60:
                    continue
                p_t0 = float(sd.cl[t0 - 1])
                hw = (r.zhi_t - r.zlo_t) / 2.0
                out = {"session": s, "zid": int(r.zid),
                       "tap_sec": int(r.tap_sec), "A_off_pts": (A - r.price_t) / 4.0,
                       "dens": int(dens.max())}
                for tag, off in (("c", 0.0), ("nb1", -40.0), ("nb2", 40.0)):
                    zlo, zhi = A + off - hw, A + off + hw
                    if zlo <= p_t0 <= zhi:
                        continue
                    ft = first_touch_after(p, zlo, zhi, t0)
                    if ft is None:
                        continue
                    m = enriched_metrics(p, zlo, zhi, ft)
                    if m["n_fwd"] < 1800:
                        continue
                    cc = classify(pd.DataFrame([m]), R=15, d0_t=4.0,
                                  D_t=r.atr_t * 0.10, T_s=1800, level="ne")
                    cs = classify(pd.DataFrame([m]), R=15, d0_t=4.0,
                                  D_t=24.0, T_s=900, level="ne")
                    out[f"y_{tag}"] = float(cc["y_any"].iloc[0])
                    out[f"ys_{tag}"] = float(cs["y_any"].iloc[0])
                rows.append(out)
            n_done += 1
            if n_done % 200 == 0:
                print(f"  {s}: {len(rows):,} anchor events, "
                      f"{_time.time()-t0c:.0f}s", flush=True)
        del b1s_m

    d = pd.DataFrame(rows)
    d["year"] = pd.to_datetime(d["session"].astype(str)).dt.year
    print(f"\nsweep-cluster anchors: {len(d):,} events | "
          f"A offset med {d['A_off_pts'].median():+.1f} pts")
    res = {"n": len(d)}
    for col, lab in (("y", "primary"), ("ys", "strict")):
        for era, mask in (("2021-23 (freeze)", d["year"] <= 2023),
                          ("2024-26 (confirm)", d["year"] >= 2024)):
            e = d[mask]
            j = e.dropna(subset=[f"{col}_c", f"{col}_nb1", f"{col}_nb2"])
            if len(j) < 1500:
                print(f"  {lab} {era}: underpowered ({len(j):,})")
                continue
            ctr = j.set_index(KEY)[f"{col}_c"]
            n1 = j.set_index(KEY)[f"{col}_nb1"]
            n2 = j.set_index(KEY)[f"{col}_nb2"]
            ses = j.assign(_s=j["session"]).set_index(KEY)["_s"]
            r = peak_test(ctr, n1, n2, ses)
            res[f"{lab}_{era}"] = r
            print(f"  {lab:>8} {era}: {r['diff']:+.4f} "
                  f"[{r['lo']:+.4f},{r['hi']:+.4f}] (n={r['n']:,})")

    out = ROOT / "outputs" / "p7_support2.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0c)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
