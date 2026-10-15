#!/usr/bin/env python
"""P8-341 — the single registered Product-A shot: a volatility-scaled
sweep-termination band.

Anchor = pivot-low centre + k·ATR (k = median ATR-scaled successful turn
depth on 2021-2023 ONLY, frozen before evaluation), band width = the zone's
own; walked with ±10-pt shifted controls from the same t0 by the standard
walker; verdict by peak test, reported separately for the frozen era and
the 2024-2026 confirmation era. One variant, no offset shopping.
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
    sup["year"] = pd.to_datetime(sup["session"].astype(str)).dt.year
    # freeze k on 2021-2023 successes only
    S = sup[(sup["y"] == 1) & (sup["year"] <= 2023)]
    k = float(((S["zhi_t"] - S["adverse_before_rev15_t"] - S["price_t"])
               / S["atr_t"]).median())
    print(f"frozen k (2021-23 median scaled turn depth): {k:+.4f} ATR "
          f"(≈ {k*300/1:.1f}t at ATR 300)")

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
            sd = build_session_data(b1s.sort_values("ts").reset_index(drop=True),
                                    b1m_by[s])
            p = sd.path
            for r in g.itertuples():
                A = float(r.price_t + k * r.atr_t)
                t0 = int(r.tap_sec) - T0_S
                if t0 < 60:
                    continue
                p_t0 = float(sd.cl[t0 - 1])
                hw = (r.zhi_t - r.zlo_t) / 2.0
                out = {"session": s, "zid": int(r.zid),
                       "tap_sec": int(r.tap_sec), "year": int(r.year)}
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
                    cy = classify(pd.DataFrame([m]), R=15, d0_t=4.0,
                                  D_t=r.atr_t * 0.10, T_s=1800, level="ne")
                    cs = classify(pd.DataFrame([m]), R=15, d0_t=4.0,
                                  D_t=24.0, T_s=900, level="ne")
                    out[f"y_{tag}"] = float(cy["y_any"].iloc[0])
                    out[f"ys_{tag}"] = float(cs["y_any"].iloc[0])
                rows.append(out)
            n_done += 1
            if n_done % 250 == 0:
                print(f"  {s}: {len(rows):,} events, {_time.time()-t0c:.0f}s",
                      flush=True)
        del b1s_m

    d = pd.DataFrame(rows)
    res = {"k": k, "n": len(d)}
    print(f"\nvol-scaled band events: {len(d):,}")
    for col, lab in (("y", "primary"), ("ys", "strict")):
        for era, mask in (("2021-23 (frozen)", d["year"] <= 2023),
                          ("2024-26 (confirm)", d["year"] >= 2024)):
            e = d[mask].dropna(subset=[f"{col}_c", f"{col}_nb1", f"{col}_nb2"])
            if len(e) < 1500:
                print(f"  {lab} {era}: underpowered ({len(e):,})")
                continue
            ctr = e.set_index(KEY)[f"{col}_c"]
            n1 = e.set_index(KEY)[f"{col}_nb1"]
            n2 = e.set_index(KEY)[f"{col}_nb2"]
            ses = e.assign(_s=e["session"]).set_index(KEY)["_s"]
            r = peak_test(ctr, n1, n2, ses)
            res[f"{lab}_{era}"] = r
            print(f"  {lab:>8} {era}: {r['diff']:+.4f} "
                  f"[{r['lo']:+.4f},{r['hi']:+.4f}] (n={r['n']:,})")
    out = ROOT / "outputs" / "p8_band.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0c)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
