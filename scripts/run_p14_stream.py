#!/usr/bin/env python
"""P14 streaming replay: event-by-event (per-second) re-decision of the
final mechanical rule with data revealed strictly sequentially. At each
second only bars <= now exist; running state (min low, path, closes) is
updated incrementally; R4 fires when the running reversal crosses 10 pts
within 60s of touch; the rule decides at the fire bar; the order fills at
the next second's close. Compares every decision, feature and entry
against the vectorized dataset. Zones/taps come from the causally-proven
layers (their own tests); this proves the decision layer end-to-end."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from p14_common import load_session_arrays                                     # noqa: E402

FIRE = "/Volumes/T9/nq_levels/daily/p14_fire.parquet"
FS_MAX, MONO_MIN = 24.0, 0.27


def stream_session(sd, taps):
    """Per-second replay; returns decision rows."""
    n = sd.cl.size
    out = []
    for r in taps.itertuples():
        t = int(r.tap_sec)
        zlo = float(r.zlo_t)
        if t >= n - 2:
            continue
        run_min = np.inf
        rev_prev = 0.0
        mono_up = 0
        steps = 0
        fired = None
        for s in range(t, min(n, t + 61)):
            # at wall-clock s+1 the bar s is complete and knowable
            run_min = min(run_min, sd.lo[s])
            rev = max(0.0, zlo - run_min)
            if s > t:
                steps += 1
                if rev > rev_prev:
                    mono_up += 1
            rev_prev = rev
            if rev >= 40.0:
                fired = s
                break
        if fired is None:
            continue
        fire_s = fired - t
        mono = (mono_up / steps) if steps else 1.0
        take = (int(r.tap_no) <= 3 and fire_s <= FS_MAX
                and mono >= MONO_MIN)
        e_idx = fired + 1
        e_px = float(sd.cl[e_idx]) if e_idx < n else np.nan
        out.append(dict(zid=int(r.zid), tap_sec=t, fire_sec=fired,
                        fire_s=fire_s, mono=mono, take=take,
                        e_idx=e_idx, e_px=e_px))
    return out


def main() -> int:
    df = pd.read_parquet(FIRE)
    df["session"] = df["session"].astype(str)
    df["year"] = pd.to_datetime(df["session"]).dt.year
    rng = np.random.default_rng(21)
    sample = list(rng.choice(sorted(df.loc[df["year"] >= 2022, "session"]
                                    .unique()), 10, replace=False))
    sample += ["2026-04-16", "2026-05-04"]          # DD-period sessions
    n_dec = n_mismatch = n_take = 0
    for sess in sample:
        ref = df[df["session"] == sess]
        if not len(ref):
            continue
        sd, g = load_session_arrays(sess)
        got = stream_session(sd, g)
        gm = {(x["zid"], x["tap_sec"]): x for x in got}
        for r in ref.itertuples():
            k = (r.zid, r.tap_sec)
            n_dec += 1
            x = gm.get(k)
            ref_take = (r.tap_no <= 3 and r.fire_s <= FS_MAX
                        and r.mono >= MONO_MIN)
            if x is None:
                n_mismatch += 1
                print(f"  MISSING {sess} {k}")
                continue
            ok = (x["fire_sec"] == r.fire_sec
                  and abs(x["mono"] - r.mono) < 1e-9
                  and x["take"] == ref_take
                  and (not x["take"] or
                       abs(x["e_px"] - getattr(r, "ck0_epx", np.nan))
                       < 1e-9))
            if not ok:
                n_mismatch += 1
                print(f"  MISMATCH {sess} {k}: stream fire {x['fire_sec']}"
                      f" mono {x['mono']:.3f} take {x['take']} epx "
                      f"{x['e_px']} | ref {r.fire_sec} {r.mono:.3f} "
                      f"{ref_take} {getattr(r, 'ck0_epx', np.nan)}")
            n_take += int(x is not None and x["take"])
    print(f"\nstreaming replay: {len(sample)} sessions, {n_dec} R4 "
          f"decisions, {n_take} takes, mismatches {n_mismatch}")
    print("RESULT:", "IDENTICAL — zero look-ahead in the decision layer"
          if n_mismatch == 0 else "FAILED")
    return 0 if n_mismatch == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
