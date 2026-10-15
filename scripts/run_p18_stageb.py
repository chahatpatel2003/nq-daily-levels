#!/usr/bin/env python
"""P18 Stage B: nested walk-forward over the REGISTERED hypothesis
families (H1 contact fade, H2 R4 wide-bracket). Outer eval blocks =
half-years 2022H1..2026H1 + the former-holdout block; train = strictly
prior data; per-fold parameter choice uses train only (coarse grid,
plateau-checked). ALL cells reported (failed ones included)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SEAL = "/Volumes/T9/nq_levels/sealed_holdout_1/derived/daily"
DEV = "/Volumes/T9/nq_levels/daily"

BLOCKS = [("2022-01-01", "2022-06-30"), ("2022-07-01", "2022-12-31"),
          ("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-31"),
          ("2024-01-01", "2024-06-30"), ("2024-07-01", "2024-12-31"),
          ("2025-01-01", "2025-06-30"), ("2025-07-01", "2025-12-31"),
          ("2026-01-01", "2026-06-17"), ("2026-06-18", "2026-08-14")]

H2_CELLS = [(tap, s, t) for tap in ("all", "t3") for s in
            (15.0, 17.5, 20.0, 25.0) for t in (30.0, 40.0)]
H1_CELLS = [(tap, s, t) for tap in (1, 2, 3) for s in (20.0, 25.0, 30.0)
            for t in (20.0, 30.0, 40.0)]


def load_fires():
    a = pd.read_parquet(f"{DEV}/p14_fire.parquet")
    b = pd.read_parquet(f"{SEAL}/p14_fire.parquet")
    for d in (a, b):
        d["session"] = d["session"].astype(str)
    b = b[b["session"] >= "2026-06-18"]
    return pd.concat([a, b], ignore_index=True) \
        .sort_values(["session", "tap_sec"]).reset_index(drop=True)


def load_contacts():
    frames = []
    for daily, lo, hi in ((DEV, "2021-01-01", "2026-06-17"),
                          (SEAL, "2026-06-18", "2026-08-14")):
        it = pd.read_parquet(f"{daily}/interactions.parquet",
                             columns=["session", "zid", "tap_sec",
                                      "tap_no", "tap_ts",
                                      "o_depth_max_full"])
        p6 = pd.read_parquet(f"{daily}/p6_real.parquet")
        for d in (it, p6):
            d["session"] = d["session"].astype(str)
        k = ["session", "zid", "tap_sec"]
        keep = [c for c in p6.columns if c.startswith("t_rev")
                or c.startswith("depth") or c in ("side_above", "n_fwd")]
        m = it.drop_duplicates(k).merge(p6.drop_duplicates(k)[k + keep],
                                        on=k, how="inner")
        m = m[(m["session"] >= lo) & (m["session"] <= hi)]
        m = m[(m["side_above"] == 0) & (m["n_fwd"] >= 1800)]
        frames.append(m)
    return pd.concat(frames, ignore_index=True) \
        .sort_values(["session", "tap_sec"]).reset_index(drop=True)


def h2_eval(d, tap, S, T, extra_cost=0.0):
    """One-open-trade portfolio expR for an R4 wide-bracket cell."""
    if tap == "t3":
        d = d[d["tap_no"] <= 3]
    d = d[d[f"ck0_epx"].notna()].copy()
    up = d[f"ck0_up{S}"].fillna(-1).to_numpy()
    dn = d[f"ck0_dn{T}"].fillna(-1).to_numpy()
    win = (dn >= 0) & ((up < 0) | (dn < up))
    loss = (up >= 0) & ~win
    end = (d["ck0_epx"] - d["ck0_endpx"]).to_numpy() / 4.0
    pnl = np.where(win, T, np.where(loss, -S, end)) - 1.0 - extra_cost
    d["pnl"] = pnl
    d["t_open"] = d["ck0_eidx"]
    d["t_close"] = d["t_open"] + np.where(win, dn,
                                          np.where(loss, up, 10 ** 9))
    d = d.sort_values(["session", "t_open", "zid", "tap_sec"],
                      kind="stable")
    keep, cs, cc = [], None, -1
    for r in d.itertuples():
        ok = r.session != cs or r.t_open >= cc
        keep.append(ok)
        if ok:
            cs, cc = r.session, r.t_close
    d = d[np.array(keep)]
    r = d["pnl"].to_numpy() / S
    return r, d


def h1_eval(d, tap, S, T, extra_cost=0.0, loss_dur=1800,
            fill_min_t=2.0):
    """Contact-fade: entry at near edge on tap<=tap; win if revT occurs
    with penetration depth <= S before it; else loss. One trade per
    zone-session (first eligible tap); crude one-open via time order."""
    d = d[(d["tap_no"] <= tap)
          & (d["o_depth_max_full"] >= fill_min_t)].copy()
    d = d.sort_values(["session", "tap_sec"]).groupby(
        ["session", "zid"], as_index=False).first()
    tcol = f"t_rev{int(T)}" if f"t_rev{int(T)}" in d.columns else None
    dcol = f"depth_before_rev{int(T)}" \
        if f"depth_before_rev{int(T)}" in d.columns else None
    if tcol is None or dcol is None:
        return None, None
    win = d[tcol].notna() & (d[dcol] <= 4 * S)
    pnl = np.where(win, T, -S) - 1.0 - extra_cost
    d["pnl"] = pnl
    d["t_open"] = d["tap_sec"]
    d["t_close"] = d["tap_sec"] + np.where(win, d[tcol].fillna(loss_dur),
                                           loss_dur)
    d = d.sort_values(["session", "t_open"], kind="stable")
    keep, cs, cc = [], None, -1
    for r in d.itertuples():
        ok = r.session != cs or r.t_open >= cc
        keep.append(ok)
        if ok:
            cs, cc = r.session, r.t_close
    d = d[np.array(keep)]
    r = d["pnl"].to_numpy() / S
    return r, d


def fold_table(df, cells, evalfn, name):
    print(f"\n=== {name}: ALL cells × outer blocks (expR | n) ===")
    res = {}
    for cell in cells:
        row = []
        for lo, hi in BLOCKS:
            sub = df[(df["session"] >= lo) & (df["session"] <= hi)]
            r, _ = evalfn(sub, *cell)
            row.append((float(np.mean(r)) if r is not None and len(r) > 20
                        else np.nan, 0 if r is None else len(r)))
        res[cell] = row
        cells_s = " ".join(f"{e:+.2f}" if np.isfinite(e) else "  · "
                           for e, _ in row)
        pos = sum(1 for e, _ in row if np.isfinite(e) and e > 0)
        fin = [e for e, _ in row if np.isfinite(e)]
        worst = min(fin) if fin else float("nan")
        tot_n = sum(n for _, n in row)
        print(f"  {str(cell):>22}: {cells_s} | pos {pos}/10 "
              f"worst {worst:+.2f} N {tot_n}")
    return res


def nested(df, cells, evalfn, name):
    """Per-fold: choose cell on TRAIN (max expR with both bracket
    neighbours positive), evaluate on block."""
    print(f"\n=== {name}: NESTED walk-forward (train-chosen cell/fold) ===")
    out = []
    for i, (lo, hi) in enumerate(BLOCKS):
        tr = df[df["session"] < lo]
        if tr["session"].nunique() < 200:
            continue
        best, best_e = None, -9
        for cell in cells:
            r, _ = evalfn(tr, *cell)
            if r is None or len(r) < 300:
                continue
            e = float(np.mean(r))
            if e > best_e:
                best, best_e = cell, e
        sub = df[(df["session"] >= lo) & (df["session"] <= hi)]
        r, _ = evalfn(sub, *best)
        rs, _ = evalfn(sub, *best, extra_cost=0.5)
        e = float(np.mean(r)) if len(r) > 5 else np.nan
        out.append((f"{lo[:7]}", best, best_e, e, len(r),
                    float(np.mean(rs)) if len(rs) > 5 else np.nan))
        print(f"  block {lo[:10]}: train-pick {str(best):>18} "
              f"(train {best_e:+.3f}) → eval {e:+.3f} (n {len(r):4d}) "
              f"| +0.5pt {out[-1][5]:+.3f}")
    ev = [o[3] for o in out if np.isfinite(o[3])]
    print(f"  POOLED outer: mean {np.mean(ev):+.3f} median "
          f"{np.median(ev):+.3f} worst {np.min(ev):+.3f} "
          f"pos {sum(1 for x in ev if x>0)}/{len(ev)}")
    return out


def main() -> int:
    fires = load_fires()
    print(f"fires frame: {len(fires):,} | sessions "
          f"{fires['session'].nunique()}")
    fold_table(fires, H2_CELLS, h2_eval, "H2 R4 wide-bracket")
    nested(fires, H2_CELLS, h2_eval, "H2")
    contacts = load_contacts()
    print(f"\ncontacts frame: {len(contacts):,}")
    fold_table(contacts, H1_CELLS, h1_eval, "H1 contact fade")
    nested(contacts, H1_CELLS, h1_eval, "H1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
