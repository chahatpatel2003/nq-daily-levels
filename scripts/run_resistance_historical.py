#!/usr/bin/env python
"""Historical resistance inspection.

  python scripts/run_resistance_historical.py --date 2026-07-16 \
      [--at-sec 57600] [--max-distance 200] [--untouched-only] [--json out.json]

at-sec = seconds from session open (22:00 UTC prior day); default 16h
(~14:00 UTC / NY morning)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from nqlevels.prod.engine import ResistanceEngine                       # noqa
from nqlevels.prod.snapshot import render, to_text, save                # noqa


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--at-sec", type=int, default=16 * 3600)
    ap.add_argument("--max-distance", type=float, default=None)
    ap.add_argument("--untouched-only", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    eng = ResistanceEngine()
    eng.run_window(a.date)
    js = render(eng, a.date, a.at_sec, a.max_distance, a.untouched_only)
    print(to_text(js))
    if a.json:
        save(js, a.json)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
