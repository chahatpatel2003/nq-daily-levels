#!/usr/bin/env python
"""P9 bounded MBP-1 confirmation add-on: does touch-time order-book state
improve the trades-only confirmation at the same checkpoints?

40 sampled sessions 2022-01..2025-06 (coverage end). Per resistance
interaction: book features over [touch, touch+τ] for τ ∈ {60, 180} within
±15 pts of the zone: time-weighted ask vs bid size (imbalance), ask-side
add/cancel counts at-or-above the zone (replenishment vs pulling), traded
volume at ask, spread mean, microprice drift since touch. Evaluation:
session-grouped 5-fold AUC of trades-only vs trades+book on the SAME rows —
sample too small for yearly walk-forward, stated as such; chronology caveat:
nothing here can claim 2026 validity.
"""
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

from nqlevels.io.raw import read_mbp1                                          # noqa: E402
from nqlevels.io.store import layer_dir                                        # noqa: E402
from run_p6_spatial import add_labels                                          # noqa: E402

CONFIGS = str(ROOT / "configs")
KEY = ["session", "zid", "tap_sec"]
N_SESS = 40
TAUS = (60, 180)


def auc(y, s):
    y = np.asarray(y, "float64")
    s = np.asarray(s, "float64")
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size < 100 or y.min() == y.max():
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    n1 = y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def main() -> int:
    t0 = _time.time()
    d0 = layer_dir("daily", CONFIGS)
    tl = pd.read_parquet(d0 / "p9_timeline.parquet")
    re = add_labels(pd.read_parquet(d0 / "p6_real.parquet"))
    it = pd.read_parquet(d0 / "interactions.parquet",
                         columns=KEY + ["price_t", "zhi_t", "zlo_t", "tap_ts"]).drop_duplicates(KEY)
    m = tl.merge(re[KEY + ["y_strict", "side_above"]], on=KEY, how="inner")
    m = m.merge(it, on=KEY, how="left", validate="1:1")
    m = m[m["y_strict"].notna()]
    ctx = pd.read_parquet(d0 / "context.parquet", columns=["session", "symbol"])
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    sym = ctx.set_index("session")["symbol"]
    pool = [s for s in m["session"].unique()
            if pd.Timestamp(str(s)).year >= 2022
            and s <= pd.Timestamp("2025-06-15").date()]
    rng = np.random.default_rng(9)
    sample = sorted(rng.choice(pool, N_SESS, replace=False))
    rows = []
    for k, s in enumerate(sample):
        g = m[m["session"] == s]
        if not len(g):
            continue
        t_lo = pd.Timestamp(float(g["tap_ts"].min()) - 60, unit="s", tz="UTC")
        t_hi = pd.Timestamp(float(g["tap_ts"].max()) + 200, unit="s", tz="UTC")
        try:
            q = read_mbp1(t_lo.to_pydatetime(), t_hi.to_pydatetime(), CONFIGS,
                          symbol=str(sym.loc[s]))
        except Exception as e:                                        # noqa: BLE001
            print(f"  {s}: read failed ({e})")
            continue
        if q.empty:
            continue
        q = q.reset_index()
        if "ts_event" not in q.columns:
            q = q.rename(columns={q.columns[0]: "ts_event"})
        q["ts_event"] = pd.DatetimeIndex(q["ts_event"]).tz_convert("UTC")
        te = q["ts_event"].astype("int64").to_numpy() // 10**9
        bidp = np.rint(q["bid_px_00"].to_numpy("float64") / 0.25)
        askp = np.rint(q["ask_px_00"].to_numpy("float64") / 0.25)
        bids = q["bid_sz_00"].to_numpy("float64")
        asks = q["ask_sz_00"].to_numpy("float64")
        act = q["action"].to_numpy()
        px = np.rint(q["price"].to_numpy("float64") / 0.25)
        sz = q["size"].to_numpy("float64")
        for r in g.itertuples():
            for tau in TAUS:
                a = np.searchsorted(te, r.tap_ts)
                b = np.searchsorted(te, r.tap_ts + tau)
                if b - a < 200:
                    continue
                sl = slice(a, b)
                ok = np.isfinite(bidp[sl]) & np.isfinite(askp[sl])
                if ok.sum() < 100:
                    continue
                bp, ap = bidp[sl][ok], askp[sl][ok]
                bs, as_ = bids[sl][ok], asks[sl][ok]
                imb = float(np.mean((bs - as_) / (bs + as_ + 1e-9)))
                spread = float(np.mean(ap - bp))
                micro0 = (bp[0] * as_[0] + ap[0] * bs[0]) / (bs[0] + as_[0])
                micro1 = (bp[-1] * as_[-1] + ap[-1] * bs[-1]) / (bs[-1] + as_[-1])
                zone_lo = r.zlo_t
                at_zone = (px[sl] >= zone_lo - 20) & (px[sl] <= r.zhi_t + 20)
                adds_a = float(((act[sl] == "A") & at_zone).sum())
                canc_a = float(((act[sl] == "C") & at_zone).sum())
                trd_a = float(sz[sl][(act[sl] == "T") & at_zone].sum())
                rows.append({"session": s, "zid": int(r.zid),
                             "tap_sec": int(r.tap_sec), "tau": tau,
                             "y": float(r.y_strict),
                             "bk_imb": imb, "bk_spread": spread,
                             "bk_micro_d": float(micro1 - micro0),
                             "bk_adds": adds_a, "bk_cancels": canc_a,
                             "bk_repl": adds_a / (canc_a + 1),
                             "bk_trd": trd_a})
        print(f"  [{k+1}/{len(sample)}] {s}: {len(rows):,} rows, "
              f"{_time.time()-t0:.0f}s", flush=True)

    d = pd.DataFrame(rows)
    res = {}
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    for tau in TAUS:
        e = d[d["tau"] == tau].merge(
            tl, on=KEY, how="left")
        post = [f"cp{tau}_{f}" for f in ("pen", "rev", "now", "above_s",
                "below_sh", "vel60", "delta_sh", "intx", "eff", "d60")]
        book = ["bk_imb", "bk_spread", "bk_micro_d", "bk_adds", "bk_cancels",
                "bk_repl", "bk_trd"]
        y = e["y"].to_numpy()
        groups = e["session"].astype(str).to_numpy()
        aucs = {"trades": [], "both": [], "book": []}
        gkf = GroupKFold(n_splits=5)
        for tr_i, te_i in gkf.split(e, y, groups):
            for tag, feats in (("trades", post), ("both", post + book),
                               ("book", book)):
                g_ = HistGradientBoostingClassifier(
                    max_iter=150, learning_rate=0.05, max_depth=3,
                    l2_regularization=1.0, random_state=0)
                g_.fit(e.iloc[tr_i][feats].to_numpy("float64"), y[tr_i])
                p = g_.predict_proba(e.iloc[te_i][feats].to_numpy("float64"))[:, 1]
                aucs[tag].append(auc(y[te_i], p))
        res[f"tau{tau}"] = {k2: float(np.nanmean(v)) for k2, v in aucs.items()}
        res[f"tau{tau}"]["n"] = int(len(e))
        print(f"τ={tau}: n={len(e):,} | trades-only AUC "
              f"{np.nanmean(aucs['trades']):.4f} | +book "
              f"{np.nanmean(aucs['both']):.4f} | book-only "
              f"{np.nanmean(aucs['book']):.4f}")
    out = ROOT / "outputs" / "p9_mbp.json"
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {out.relative_to(ROOT)}  ({(_time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
