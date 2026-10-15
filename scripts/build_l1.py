#!/usr/bin/env python
"""Build the L1 layer: 1-second bars + 30-minute volume-at-price profiles.

    python scripts/build_l1.py --start 2024-01-01 --end 2024-03-31
    python scripts/build_l1.py --all --resume

Resumable at month granularity. The roll selector is seeded from the trade date
*before* the requested window, so the first session's front month is never
chosen using its own data — the same rule that applies to every later session.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nqlevels.build.roll import FrontMonthSelector                     # noqa: E402
from nqlevels.build.tape import TapeParams, build_session_tape         # noqa: E402
from nqlevels.clock import (load_calendar, prior_trade_date,           # noqa: E402
                            trade_dates)
from nqlevels.io.raw import session_symbol_volumes                     # noqa: E402
from nqlevels.io.store import month_key, partition_exists, write_partition  # noqa: E402

CONFIGS = str(ROOT / "configs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-05")
    ap.add_argument("--end", default="2026-06-17")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip months whose partitions already exist")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cal = load_calendar(CONFIGS)
    import yaml
    rc = yaml.safe_load((ROOT / "configs" / "research.yaml").read_text())
    b = rc["build"]
    p = TapeParams(tick_size=rc["instrument"]["tick_size"],
                   min_size=b["trade_filter"]["min_size"],
                   median_window=b["trade_filter"]["median_window"],
                   max_jump_pct=b["trade_filter"]["max_jump_pct"],
                   min_symbol_volume_share=b["min_symbol_volume_share"])

    start = date.fromisoformat(a.start)
    end = date.fromisoformat(a.end)
    sessions = trade_dates(start, end, cal)
    if not sessions:
        print("no trade sessions in range")
        return 1
    print(f"{len(sessions)} trade dates {sessions[0]} .. {sessions[-1]}")
    if a.dry_run:
        return 0

    # Seed: the session before the window, read for its volumes only. Its bars
    # are discarded — it exists so session[0]'s front month comes from prior data.
    sel = FrontMonthSelector()
    seed = prior_trade_date(sessions[0], cal)
    try:
        sel.observe(seed, session_symbol_volumes(seed, cal, CONFIGS))
        print(f"seeded roll from {seed}: {len(sel._prior_volumes)} symbols")
    except Exception as e:                                    # noqa: BLE001
        print(f"seed session {seed} unavailable ({e}); first session will "
              f"bootstrap without a prior front month")

    log = {"sessions": [], "rolls": [], "skipped": [], "params": vars(p),
           "seed_session": str(seed)}
    buf_bars, buf_prof, cur_month = [], [], None
    t_all = time.time()

    def flush(mk: str) -> None:
        if not buf_bars:
            return
        nb = write_partition(pd.concat(buf_bars, ignore_index=True), "bars_1s", mk, CONFIGS)
        npf = write_partition(pd.concat(buf_prof, ignore_index=True), "profiles", mk, CONFIGS)
        print(f"  -> wrote {mk}: bars {nb.stat().st_size/1e6:.0f} MB, "
              f"profiles {npf.stat().st_size/1e6:.0f} MB", flush=True)
        buf_bars.clear()
        buf_prof.clear()

    for i, sess in enumerate(sessions):
        mk = month_key(sess)
        if mk != cur_month:
            flush(cur_month) if cur_month else None
            cur_month = mk
            if a.resume and partition_exists("bars_1s", mk, CONFIGS):
                print(f"{mk}: exists, skipping (resume) — roll state still advanced")
        t0 = time.time()
        front = sel.select(sess)
        tape = build_session_tape(sess, cal, CONFIGS, p)
        sel.observe(sess, tape.volumes)          # feeds the NEXT session only

        if tape.bars.empty:
            log["skipped"].append({"session": str(sess), "why": tape.stats})
            print(f"{sess}: SKIP {tape.stats}", flush=True)
            continue
        if front is None:
            front = tape.symbols[0]              # bootstrap session only
        tape.bars["is_front"] = tape.bars["symbol"] == front
        tape.profile["is_front"] = tape.profile["symbol"] == front

        if not (a.resume and partition_exists("bars_1s", mk, CONFIGS)):
            buf_bars.append(tape.bars)
            buf_prof.append(tape.profile)
        log["sessions"].append({
            "session": str(sess), "front": front, "symbols": tape.symbols,
            "n_bar_rows": int(len(tape.bars)), "n_prof_rows": int(len(tape.profile)),
            "volumes": tape.volumes, "secs": round(time.time() - t0, 2)})
        if i % 20 == 0:
            print(f"{sess} front={front} syms={tape.symbols} "
                  f"bars={len(tape.bars):,} prof={len(tape.profile):,} "
                  f"{time.time()-t0:.1f}s", flush=True)
    flush(cur_month)

    log["rolls"] = [dict(session=str(r.session), previous=r.previous, front=r.front,
                         is_bootstrap=r.is_bootstrap,
                         decided_from_session=str(r.decided_from_session),
                         prev_volume=r.prev_volume, new_volume=r.new_volume)
                    for r in sel.roll_log]
    out = ROOT / "outputs" / f"l1_build_{start}_{end}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(log, indent=1, default=str))
    print(f"\ndone in {(time.time()-t_all)/60:.1f} min — {len(log['sessions'])} sessions, "
          f"{len(log['rolls'])} rolls, {len(log['skipped'])} skipped\nlog: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
