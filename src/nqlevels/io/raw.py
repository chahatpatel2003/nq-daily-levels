"""Read-only access to the raw Databento NQ parquet corpus on the T9 SSD.

Corpus facts established by the 2026-08-14 audit (see ``DATA_AUDIT.md``):

* ``trades`` — 11 half-year files, 547.0 M rows, 2021-01-03 -> 2026-06-17.
  Databento trades schema, ``rtype`` 0, ``action`` always ``'T'``.
* ``mbp1`` — 9 half-year files, 7.89 B rows, ends 2025-06-30. Top-of-book plus
  interleaved trades; ``action`` in ``{A, C, M, T}``.

Both carry ``ts_event`` (exchange time) and ``ts_recv`` (capture time) as
tz-aware UTC nanoseconds; ``ts_recv`` is the stored pandas index, so it does
not appear as a column unless requested by name. ``price`` is already a float
in index points and is an exact multiple of the 0.25 tick.

**Aggressor convention (verified, not assumed).** Over a 30-minute RTH sample
of 39,674 MBP-1 trade records, ``side == 'B'`` matched *price >= prevailing
ask* 19,744/19,744 times and ``side == 'A'`` matched *price <= prevailing bid*
19,930/19,930 times, with zero records in between. So ``'B'`` is a **buy
aggressor** (lifted the offer) and ``'A'`` is a **sell aggressor** (hit the
bid); ``'N'`` is unattributed and is counted in volume but in neither leg.

**Symbol mix.** Every file contains the whole NQ complex: quarterly outrights
(``NQH4``), calendar spreads (``NQH4-NQM4``), and thin far-dated months. Spread
symbols contain ``'-'`` and must always be excluded — their prices are
differences, not index levels, and would poison any level or profile.

**MBP-1 seam duplicates.** Half-year files overlap by a handful of records at
the boundaries (probed 2026-08-14: 7-8 rows at three seams, 4 byte-identical
records at the 2023_H2/2024_H1 seam). Reads therefore go through canonical
half-open windows from ``paths.yaml`` rather than a naive union.
"""
from __future__ import annotations

import glob
import os
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import yaml

from ..clock import Calendar, session_close_utc, session_open_utc

# --------------------------------------------------------------------------- #
# read-only enforcement
# --------------------------------------------------------------------------- #

_RAW_PREFIXES = ("/Volumes/T9/nq data", "/Volumes/T9/gc")


class RawWriteAttempt(RuntimeError):
    """Raised on any attempt to resolve a writable path under a raw root."""


def assert_not_raw(path) -> Path:
    """Guard every write in the project. Raw data is source material."""
    p = Path(path).resolve()
    for prefix in _RAW_PREFIXES:
        if str(p).startswith(str(Path(prefix).resolve())):
            raise RawWriteAttempt(
                f"refusing to write under the read-only raw root: {p}")
    return p


# --------------------------------------------------------------------------- #
# config + datasets
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=4)
def load_paths(configs_dir: str) -> dict:
    return yaml.safe_load((Path(configs_dir) / "paths.yaml").read_text())


def _files(spec: dict) -> List[Path]:
    out = [Path(p) for p in sorted(glob.glob(os.path.join(spec["path"], spec["pattern"])))]
    # macOS AppleDouble sidecars on exFAT volumes are not parquet.
    return [p for p in out if not p.name.startswith("._")]


@lru_cache(maxsize=8)
def _dataset(kind: str, configs_dir: str):
    spec = load_paths(configs_dir)["raw"][kind]
    return ds.dataset([str(p) for p in _files(spec)])


def trades_files(configs_dir: str) -> List[Path]:
    return _files(load_paths(configs_dir)["raw"]["nq_trades"])


def mbp1_files(configs_dir: str) -> List[Path]:
    return _files(load_paths(configs_dir)["raw"]["nq_mbp1"])


# --------------------------------------------------------------------------- #
# symbols
# --------------------------------------------------------------------------- #

SPREAD_MARKER = "-"
TRADE_COLS = ("ts_event", "price", "size", "side", "symbol")
MBP1_QUOTE_COLS = ("ts_event", "action", "side", "price", "size",
                   "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00",
                   "bid_ct_00", "ask_ct_00", "symbol")


def is_spread(symbol: str) -> bool:
    return SPREAD_MARKER in symbol


def drop_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """Remove calendar-spread prints. Never optional."""
    if df.empty:
        return df
    return df[~df["symbol"].str.contains(SPREAD_MARKER, regex=False)]


# --------------------------------------------------------------------------- #
# reads
# --------------------------------------------------------------------------- #


def _to_utc(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty and "ts_event" in df:
        df["ts_event"] = pd.DatetimeIndex(df["ts_event"]).tz_convert("UTC")
    return df


def read_trades(lo: datetime, hi: datetime, configs_dir: str,
                symbols: Optional[Sequence[str]] = None,
                columns: Sequence[str] = TRADE_COLS,
                exclude_spreads: bool = True) -> pd.DataFrame:
    """Trades with ``lo <= ts_event < hi``.

    Row-group statistics prune the scan to the relevant files, so a single
    session costs about half a second across the whole 13 GB corpus.
    """
    f = (pc.field("ts_event") >= pd.Timestamp(lo)) & (pc.field("ts_event") < pd.Timestamp(hi))
    if symbols is not None:
        f = f & pc.field("symbol").isin(list(symbols))
    df = _dataset("nq_trades", configs_dir).to_table(
        columns=list(columns), filter=f).to_pandas().reset_index(drop=True)
    df = _to_utc(df)
    if exclude_spreads and "symbol" in df:
        df = drop_spreads(df).reset_index(drop=True)
    return df


def mbp1_window_for(file_stem: str, configs_dir: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    w = load_paths(configs_dir)["mbp1_canonical_windows"][file_stem]
    return pd.Timestamp(w[0], tz="UTC"), pd.Timestamp(w[1], tz="UTC")


def read_mbp1(lo: datetime, hi: datetime, configs_dir: str,
              symbol: Optional[str] = None,
              columns: Sequence[str] = MBP1_QUOTE_COLS,
              actions: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """MBP-1 records with ``lo <= ts_event < hi``, seam-deduplicated.

    Each file is read only inside its canonical window, which makes the file
    set a strict partition of the timeline; that is cheaper and more auditable
    than unioning and then dropping duplicate keys over billions of rows.

    MBP-1 stops at 2025-06-30 — callers that silently get an empty frame for a
    later date would be comparing models on unequal data, so that case raises.
    """
    paths = load_paths(configs_dir)
    end = pd.Timestamp(paths["raw"]["nq_mbp1"]["coverage_end"], tz="UTC") + pd.Timedelta(days=1)
    if pd.Timestamp(lo) >= end:
        raise ValueError(
            f"MBP-1 coverage ends {paths['raw']['nq_mbp1']['coverage_end']}; "
            f"requested window starts {pd.Timestamp(lo).date()}. Microstructure "
            "research cannot span this date — state the shorter span explicitly.")
    frames = []
    for p in mbp1_files(configs_dir):
        w_lo, w_hi = mbp1_window_for(p.stem, configs_dir)
        a, b = max(pd.Timestamp(lo), w_lo), min(pd.Timestamp(hi), w_hi)
        if b <= a:
            continue
        f = (pc.field("ts_event") >= a) & (pc.field("ts_event") < b)
        if symbol is not None:
            f = f & (pc.field("symbol") == symbol)
        if actions is not None:
            f = f & pc.field("action").isin(list(actions))
        frames.append(ds.dataset(str(p)).to_table(columns=list(columns), filter=f).to_pandas())
    if not frames:
        return pd.DataFrame(columns=list(columns))
    df = pd.concat(frames, ignore_index=True)
    return _to_utc(df).sort_values("ts_event", kind="stable").reset_index(drop=True)


def read_session_trades(sess: date, cal: Calendar, configs_dir: str,
                        symbols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Every outright NQ print of one trade date, [session open, close)."""
    return read_trades(session_open_utc(sess, cal), session_close_utc(sess, cal),
                       configs_dir, symbols=symbols)


def session_symbol_volumes(sess: date, cal: Calendar, configs_dir: str) -> Dict[str, int]:
    """Outright-only volume by symbol for one trade date.

    This is the *only* input to the roll decision, and it is always consumed by
    the NEXT session — see :mod:`nqlevels.build.roll`.
    """
    df = read_session_trades(sess, cal, configs_dir)
    if df.empty:
        return {}
    return df.groupby("symbol")["size"].sum().astype("int64").to_dict()
