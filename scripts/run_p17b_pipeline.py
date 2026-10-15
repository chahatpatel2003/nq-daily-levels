#!/usr/bin/env python
"""P17B sealed pipeline: convert the holdout DBN batch to the corpus
schema, then run the FROZEN builders (patched only to sealed roots) over
a 2026 warmup+holdout window. Warmup months recompute from dev raw and
must reproduce dev outputs (pipeline-equivalence proof) before any
unseen session is evaluated. No strategy parameter is touched."""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

HD = Path("/Volumes/T9/NQ_HOLDOUT/GLBX-20260815-69WTYRKTDB")
SEAL = Path("/Volumes/T9/nq_levels/sealed_holdout_1")
CFG_H = ROOT / "configs_holdout"
DEV_RAW = Path("/Volumes/T9/nq data/raw/trades")
DEV_B1S = Path("/Volumes/T9/nq_levels/bars_1s")
DEV_DAILY = Path("/Volumes/T9/nq_levels/daily")


def step1_convert():
    out = SEAL / "raw/trades/2026_H2.parquet"
    if out.exists():
        print("convert: exists, skip")
        return
    from databento_dbn import DBNDecoder
    import zstandard as zstd
    frames = []
    for p in sorted(glob.glob(str(HD / "glbx-mdp3-*.trades.dbn.zst"))):
        raw = zstd.ZstdDecompressor().decompress(
            open(p, "rb").read(), max_output_size=1 << 31)
        dec = DBNDecoder()
        dec.write(raw)
        recs = dec.decode()
        meta, rows = recs[0], recs[1:]
        id2sym = {}
        for sym, ivs in meta.mappings.items():
            for iv in ivs:
                id2sym[int(iv["symbol"])] = sym
        n = len(rows)
        df = pd.DataFrame({
            "ts_event": pd.to_datetime([r.ts_event for r in rows],
                                       utc=True),
            "rtype": np.uint8(0),
            "publisher_id": np.uint16(1),
            "instrument_id": np.array([r.instrument_id for r in rows],
                                      np.uint32),
            "action": "T",
            "side": [str(r.side)[-1] if str(r.side)[-1] in "ABN"
                     else {"ASK": "A", "BID": "B", "NONE": "N"}
                     [str(r.side).split(".")[-1]] for r in rows],
            "depth": np.uint8(0),
            "price": np.array([r.price for r in rows], np.int64) / 1e9,
            "size": np.array([r.size for r in rows], np.uint32),
            "flags": np.array([int(r.flags) for r in rows], np.uint8),
            "ts_in_delta": np.array([r.ts_in_delta for r in rows],
                                    np.int32),
            "sequence": np.array([r.sequence for r in rows], np.uint32),
            "symbol": [id2sym.get(r.instrument_id, "?") for r in rows],
            "ts_recv": pd.to_datetime([r.ts_recv for r in rows],
                                      utc=True),
        })
        frames.append(df)
        print(f"  {Path(p).name}: {n:,} recs", flush=True)
    all_ = pd.concat(frames, ignore_index=True).set_index("ts_recv")
    all_ = all_.sort_index(kind="stable")
    outright = ~all_["symbol"].str.contains("-")
    q = all_.loc[outright, "price"].to_numpy() * 4
    assert np.abs(q - np.round(q)).max() < 1e-6, "tick-grid violation"
    assert (all_["symbol"] == "?").sum() == 0, "unmapped instrument ids"
    out.parent.mkdir(parents=True, exist_ok=True)
    all_.to_parquet(out)
    print(f"convert: {len(all_):,} records → {out}")


def step2_configs():
    if CFG_H.exists():
        shutil.rmtree(CFG_H)
    shutil.copytree(ROOT / "configs", CFG_H)
    py = (CFG_H / "paths.yaml").read_text()
    py = py.replace('path: "/Volumes/T9/nq data/raw/trades"',
                    f'path: "{SEAL}/raw/trades"')
    py = py.replace('root: "/Volumes/T9/nq_levels"',
                    f'root: "{SEAL}/derived"')
    (CFG_H / "paths.yaml").write_text(py)
    for d in ("raw/trades", "derived/bars_1s", "derived/profiles",
              "derived/daily"):
        (SEAL / d).mkdir(parents=True, exist_ok=True)
    for f in sorted(DEV_RAW.glob("*.parquet")):
        dst = SEAL / "raw/trades" / f.name
        if not dst.exists():
            dst.symlink_to(f)
    for mk in ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05"):
        dst = SEAL / "derived/bars_1s" / f"{mk}.parquet"
        if not dst.exists():
            dst.symlink_to(DEV_B1S / f"{mk}.parquet")
    print("configs_holdout + sealed dirs + symlinks ready")


def _run(mod_name, argv, extra_patch=None):
    import importlib
    mod = importlib.import_module(mod_name)
    importlib.reload(mod)
    mod.CONFIGS = str(CFG_H)
    if extra_patch:
        extra_patch(mod)
    old = sys.argv
    sys.argv = [mod_name] + argv
    try:
        rc = mod.main()
    finally:
        sys.argv = old
    assert rc in (0, None), f"{mod_name} failed rc={rc}"


def step3_l1():
    if (SEAL / "derived/bars_1s/2026-08.parquet").exists():
        print("l1: exists, skip")
        return
    _run("build_l1", ["--start", "2026-06-01", "--end", "2026-08-14"])


def step4_check_june():
    dev = pd.read_parquet(DEV_B1S / "2026-06.parquet")
    new = pd.read_parquet(SEAL / "derived/bars_1s/2026-06.parquet")
    dev = dev[dev["session"].astype(str) <= "2026-06-16"]
    newc = new[new["session"].astype(str) <= "2026-06-16"]
    same_n = len(dev) == len(newc)
    cols = ["open_t", "high_t", "low_t", "close_t", "volume", "buy_vol"]
    ok = same_n and all(
        np.allclose(dev[c].to_numpy("float64"),
                    newc[c].to_numpy("float64"), equal_nan=True)
        for c in cols)
    print(f"JUNE OVERLAP CHECK (<=06-16): rows {len(dev):,} vs "
          f"{len(newc):,} | OHLCV identical: {ok}")
    assert ok, "sealed rebuild of dev June bars does not match dev"


def step5_bars1m():
    _run("build_bars_1m", [])


def step6_context():
    from nqlevels.build.daily import add_rolling_scales
    from nqlevels.clock import load_calendar, prior_rth_window_utc, \
        trade_dates
    from datetime import date
    dev = pd.read_parquet(DEV_DAILY / "context.parquet")
    dev["session"] = pd.to_datetime(dev["session"]).dt.date
    b1m = pd.read_parquet(SEAL / "derived/daily/bars_1m.parquet")
    b1m["ts"] = pd.DatetimeIndex(b1m["ts"]).tz_convert("UTC")
    fb = b1m[b1m["is_front"]]
    cal = load_calendar(str(CFG_H))
    hold_sessions = sorted(s for s in fb["session"].unique()
                           if str(s) >= "2026-06-18")
    # per-session RTH range (ticks) incl. 06-17 (missing -> NaN)
    rng = {}
    for s, g in fb.groupby("session"):
        # session s's OWN RTH window (13:30-20:00 UTC in summer, matching
        # prior_rth_window_utc's convention for the following session)
        lo_t = pd.Timestamp(s).tz_localize("UTC") + pd.Timedelta(hours=13,
                                                                 minutes=30)
        hi_t = pd.Timestamp(s).tz_localize("UTC") + pd.Timedelta(hours=20)
        w = g[(g["ts"] >= lo_t) & (g["ts"] < hi_t)]
        if len(w) > 30:
            rng[s] = float((w["high_t"].max() - w["low_t"].min()))
    front = fb.groupby("session")["symbol"].first()
    # cross-check my range extraction against dev context on June overlap
    devi = dev.set_index("session")["prior_rth_range_prior"]
    chk = []
    ds = sorted(x for x in rng if str(x).startswith("2026-06")
                and str(x) <= "2026-06-16")
    for i, s in enumerate(ds[:-1]):
        nxt = ds[i + 1]
        if nxt in devi.index and np.isfinite(devi[nxt]):
            chk.append(abs(rng[s] - float(devi[nxt])))
    print(f"  RTH-range crosscheck vs dev (June): n={len(chk)} "
          f"max |Δ| {max(chk) if chk else -1:.1f} ticks")
    assert chk and max(chk) < 1.0, "RTH range extraction mismatch vs dev"
    rows = []
    prev = {s: r for s, r in
            dev.set_index("session")["prior_rth_range_prior"].items()}
    sess_all = sorted(set(dev["session"]) | set(hold_sessions))
    for i, s in enumerate(sess_all):
        if s in set(dev["session"]):
            continue
        p = sess_all[i - 1]
        pr = rng.get(p, np.nan)
        rows.append(dict(session=s, symbol=front.get(s, "?"),
                         prior_rth_range_prior=pr,
                         dow=pd.Timestamp(s).dayofweek))
    new = pd.DataFrame(rows)
    comb = pd.concat([dev, new], ignore_index=True) \
        .sort_values("session").reset_index(drop=True)
    comb2 = add_rolling_scales(comb)
    comb2["atr14_t"] = comb2["atr_t"]
    # dev-row equality check on the recomputed scale
    m = comb2.merge(dev[["session", "atr14_t"]], on="session",
                    suffixes=("", "_dev"), how="inner")
    m = m[m["atr14_t_dev"].notna() & m["atr14_t"].notna()]
    dmax = float((m["atr14_t"] - m["atr14_t_dev"]).abs().max())
    print(f"context ATR recompute vs dev: max |Δ| {dmax:.6f} ticks "
          f"({len(m):,} sessions)")
    assert dmax < 1e-6, "ATR recompute mismatch"
    keep = comb2[["session", "symbol", "atr14_t", "atr_pctile", "dow",
                  "prior_rth_range_prior"]]
    keep.to_parquet(SEAL / "derived/daily/context.parquet", index=False)
    print(f"sealed context: {len(keep):,} rows "
          f"({len(new)} holdout sessions appended)")


def step7_roll_adjust():
    import importlib
    ev = importlib.import_module("build_events")
    b1m = pd.read_parquet(SEAL / "derived/daily/bars_1m.parquet")
    b1m["ts"] = pd.DatetimeIndex(b1m["ts"]).tz_convert("UTC")
    ctx = pd.read_parquet(SEAL / "derived/daily/context.parquet")
    ctx["session"] = pd.to_datetime(ctx["session"]).dt.date
    sess_2026 = ctx[ctx["session"] >= pd.Timestamp("2026-01-01").date()]
    front = sess_2026.set_index("session")["symbol"]
    fn = getattr(ev, "roll_adjustment", None)
    if fn is None:
        from nqlevels.features.events import roll_adjustment as fn
    adj = fn(b1m, front)
    adj.to_parquet(SEAL / "derived/daily/roll_adjust.parquet",
                   index=False)
    print(f"sealed roll_adjust: {len(adj):,} rows | epochs "
          f"{adj['epoch'].nunique() if 'epoch' in adj.columns else '?'}")


def step8_interactions():
    def patch(mod):
        mod.B1S_DIR = str(SEAL / "derived/bars_1s")
    _run("build_interactions", [], patch)


def step9_p6():
    import importlib
    p4 = importlib.import_module("run_p4_strongweak")
    p4.CONFIGS = str(CFG_H)

    def patch(mod):
        mod.B1S_DIR = str(SEAL / "derived/bars_1s")
    _run("build_p6_measure", [], patch)


def step10_p14fire():
    import importlib
    import p14_common as pc
    importlib.reload(pc)
    pc.DAILY = SEAL / "derived/daily"
    pc.B1S_DIR = str(SEAL / "derived/bars_1s")
    sealed_ld = lambda layer, cfg: SEAL / "derived" / layer   # noqa: E731
    pc.layer_dir = sealed_ld
    stub = SEAL / "derived/daily/p9_timeline.parquet"
    if not stub.exists():
        pd.DataFrame({"session": pd.Series([], dtype=object),
                      "zid": pd.Series([], dtype="int64"),
                      "tap_sec": pd.Series([], dtype="int64"),
                      "r4v10.0_t": pd.Series([], dtype="float64")}
                     ).to_parquet(stub)
    bf = importlib.import_module("build_p14_fire")
    importlib.reload(bf)
    bf.CONFIGS = str(CFG_H)
    bf.OUT = SEAL / "derived/daily/p14_fire.parquet"
    if hasattr(bf, "layer_dir"):
        bf.layer_dir = sealed_ld
    bf.B1S_DIR = str(SEAL / "derived/bars_1s") \
        if hasattr(bf, "B1S_DIR") else None
    import p14_common as pc2
    pc2.layer_dir = sealed_ld
    old = sys.argv
    sys.argv = ["build_p14_fire"]
    try:
        rc = bf.main()
        assert rc in (0, None)
    finally:
        sys.argv = old


def main() -> int:
    t0 = _time.time()
    step1_convert()
    step2_configs()
    step3_l1()
    step4_check_june()
    step5_bars1m()
    step6_context()
    step7_roll_adjust()
    step8_interactions()
    step9_p6()
    step10_p14fire()
    print(f"\nsealed pipeline complete [{(_time.time()-t0)/60:.1f} min]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
