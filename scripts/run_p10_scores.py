#!/usr/bin/env python
"""Persist the Phase-9 walk-forward calibrated confirmation scores
(p_both60, p_both180) — identical fit to run_p9_model, saved per interaction
for the strategy phase. Never trained on P&L."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nqlevels.io.store import layer_dir                                        # noqa: E402
from run_p6_spatial import add_labels                                          # noqa: E402
from run_p9_model import PRE                                                   # noqa: E402

CONFIGS = str(ROOT / "configs")
KEY = ["session", "zid", "tap_sec"]


def main() -> int:
    d0 = layer_dir("daily", CONFIGS)
    tl = pd.read_parquet(d0 / "p9_timeline.parquet")
    re = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    it = pd.read_parquet(d0 / "interactions.parquet").drop_duplicates(KEY)
    npr = it["n_prior_A"] + it["n_prior_B"] + it["n_prior_C"]
    it["prior_succ_share_f"] = np.where(
        npr > 0, (it["n_prior_A"] + it["n_prior_B"]) / npr, 0.5)
    m = tl.merge(re[KEY + ["y_strict", "seg", "dist_t0_t"]], on=KEY,
                 how="inner", validate="1:1")
    m = m.merge(it[[c for c in PRE if c in it.columns] + KEY],
                on=KEY, how="left", validate="1:1")
    m["year"] = pd.to_datetime(m["session"].astype(str)).dt.year
    m = m[m["y_strict"].notna()].reset_index(drop=True)
    from sklearn.ensemble import HistGradientBoostingClassifier
    pre_feats = [c for c in PRE if c in m.columns]
    for tau in (60, 180):
        post = [f"cp{tau}_{f}" for f in ("pen", "rev", "now", "above_s",
                "below_sh", "vel60", "delta_sh", "intx", "eff", "d60")]
        feats = pre_feats + post
        p = np.full(len(m), np.nan)
        for Y in (2022, 2023, 2024, 2025, 2026):
            tr = m[m["year"] < Y]
            te = m[m["year"] == Y]
            if len(tr) < 8000 or len(te) < 1500:
                continue
            g = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=4,
                l2_regularization=1.0, random_state=0)
            g.fit(tr[feats].to_numpy("float64"), tr["y_strict"].to_numpy())
            p[te.index] = g.predict_proba(te[feats].to_numpy("float64"))[:, 1]
        m[f"p_both{tau}"] = p
    m[KEY + ["p_both60", "p_both180"]].to_parquet(
        d0 / "p10_scores.parquet", index=False, compression="zstd")
    print(f"saved {len(m):,} rows; OOS share "
          f"{float(m['p_both60'].notna().mean()):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
