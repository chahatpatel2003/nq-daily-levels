#!/usr/bin/env python
"""P5 hysteresis variant — the one pre-registered anti-churn rule.

Re-runs the display selection offline from the stored scored pool with a
fixed 10 % incumbent margin: a challenger displaces a displayed zone only if
its score exceeds the incumbent's by 10 %. No parameter search — this exact
margin was registered in RESEARCH_PLAN §16.1 before any result was seen.

Reports stability (rank-1 persistence, episode length/count) and the same
touched/success arms as the primary run, so the stability/quality trade is
explicit.
"""
from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nqlevels.io.store import layer_dir                                        # noqa: E402
from run_p5_report import attach_touch, build_episodes, tap_table              # noqa: E402

CONFIGS = str(ROOT / "configs")
MARGIN = 1.10
K = 4
SEP_ATR = 0.05


def reselect(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool.copy()
    out["rank"] = 0
    rank_col = out.columns.get_loc("rank")
    for sess, gs in out.groupby("session", sort=False):
        incumbents: set = set()
        for t, g in gs.groupby("t_sec", sort=True):
            sc = g["score_T1"].to_numpy("float64").copy()
            boost = g["zid"].isin(incumbents).to_numpy()
            eff = np.where(boost, sc * MARGIN, sc)
            order = np.argsort(-eff)
            px = g["price_t"].to_numpy()
            atr = float(g["atr_t"].iloc[0])
            taken, taken_px = [], []
            for i in order:
                if len(taken) >= K:
                    break
                if any(abs(px[i] - p) < SEP_ATR * atr for p in taken_px):
                    continue
                taken.append(i)
                taken_px.append(px[i])
            idx = g.index[taken]
            for r_, ix in enumerate(idx, 1):
                out.iat[out.index.get_loc(ix), rank_col] = r_
            incumbents = set(out.loc[idx, "zid"])
    return out


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    pool = pd.read_parquet(d0 / "p5_pool.parquet")
    it = tap_table()
    taps_by = {}
    for (sess, zid), g in it.groupby(["session", "zid"], sort=False):
        g = g.sort_values("tap_sec")
        taps_by[(sess, zid)] = (g["tap_sec"].to_numpy("int64"),
                                g["y"].to_numpy("float64"),
                                g["y_strict"].to_numpy("float64"),
                                g["catA"].to_numpy("float64"),
                                g["catB"].to_numpy("float64"))
    for name, p in (("no-hysteresis", pool), ("margin-10%", reselect(pool))):
        d1 = p[p["rank"] == 1].sort_values(["session", "t_sec"])
        persist = float((d1["zid"] == d1.groupby("session")["zid"].shift()).mean())
        ep = attach_touch(build_episodes(p), taps_by)
        t = ep[ep["touched_ss"] & ep["y"].notna()]
        print(f"{name:>14}: episodes {len(ep):,} | med len "
              f"{((ep['t1']-ep['t0'])/60+15).median():.0f} min | rank-1 persist "
              f"{persist:.3f} | touch {ep['touched_ss'].mean():.3f} | "
              f"y|touch {t['y'].mean():.4f}", flush=True)
    print(f"({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
