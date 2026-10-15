"""L1: raw prints -> per-session 1-second bars and volume-at-price profiles.

This is the layer every later experiment reads, so it is built once and made
general rather than tailored to the first hypothesis:

* **1-second bars** (not 1-minute). Distinguishing "tapped the zone and left"
  from "hung around the zone and eventually drifted" is the central measurement
  problem of this project, and a 1-minute bar cannot see it. A second is fine
  enough for 5-30 point reactions and still only ~50 k rows per session.

* **Volume-at-price in 30-minute buckets**, keyed by price *tick*, not by a
  session-level profile. Any window's profile — prior RTH, overnight, London,
  a 10-day composite, a rolling 3-session profile — is then a groupby-sum over
  buckets, with no second pass over 547 M prints. Every PT segment boundary the
  research uses (15:00, 00:00, 05:00, 06:30, 09:00, 13:00) lands on a 30-minute
  mark, so no bucket ever straddles a boundary.

* **Per symbol**, for every quarterly outright above a small volume share, so a
  roll session carries both price scales.

Prices are stored as integer ticks (``price / 0.25``). Exact integers beat
floats for equality, bucketing and profile keys, and NQ's ~84 000-tick level
sits comfortably inside int32.

Causality: this module only ever reads inside ``[session_open, session_close)``
of the session it is asked for, and its bad-print filter compares each print to
a rolling median of *strictly prior* prints. It does not know about the
decision cut — windowing to the cut happens in the feature layer, which is
where the guarantee is asserted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..clock import Calendar, session_close_utc, session_open_utc
from ..io.raw import read_session_trades
from .roll import is_quarterly_outright, symbols_to_materialize

TICK = 0.25
BUCKET = "30min"


@dataclass(frozen=True)
class TapeParams:
    tick_size: float = TICK
    min_size: int = 1
    median_window: int = 50
    max_jump_pct: float = 0.005
    min_symbol_volume_share: float = 0.005


# --------------------------------------------------------------------------- #
# cleaning
# --------------------------------------------------------------------------- #


def clean_prints(df: pd.DataFrame, p: TapeParams) -> Tuple[pd.DataFrame, Dict]:
    """Drop non-positive/undersized prints and causal price outliers.

    The reference median is shifted by one row, so a print is judged only
    against prints that preceded it. An uncaught bad print is worse here than
    in most pipelines: a single stray tick becomes a session high, and a
    session high becomes a level the model is asked to predict.
    """
    n0 = len(df)
    if n0 == 0:
        return df, {"n_raw": 0, "n_dropped_size_px": 0, "n_dropped_jump": 0, "n_clean": 0}
    ok = (df["price"] > 0) & (df["size"] >= p.min_size)
    df = df[ok]
    med = df["price"].rolling(p.median_window, min_periods=5).median().shift(1)
    dev = (df["price"] / med - 1.0).abs()
    keep = dev.isna() | (dev <= p.max_jump_pct)
    out = df[keep]
    return out, {"n_raw": n0, "n_dropped_size_px": int(n0 - len(df)),
                 "n_dropped_jump": int((~keep).sum()), "n_clean": int(len(out))}


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def _signed(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Buy/sell legs under the verified convention: 'B' lifts, 'A' hits.

    'N' volume is unattributed; it is inside ``volume`` but in neither leg, so
    ``buy_vol + sell_vol <= volume`` by construction and delta stays honest.
    """
    sz = df["size"].astype("int64")
    return sz.where(df["side"] == "B", 0), sz.where(df["side"] == "A", 0)


def bars_1s(df: pd.DataFrame, p: TapeParams) -> pd.DataFrame:
    """1-second OHLCV + order-flow for prints of ONE symbol."""
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("ts_event", kind="stable")
    sec = df["ts_event"].dt.floor("1s")
    buy, sell = _signed(df)
    ticks = np.rint(df["price"].to_numpy() / p.tick_size).astype("int64")
    g = df.assign(_sec=sec, _t=ticks, _b=buy, _s=sell,
                  _n=df["price"].to_numpy() * df["size"].to_numpy()).groupby("_sec", sort=True)
    out = pd.DataFrame({
        "open_t": g["_t"].first().astype("int32"),
        "high_t": g["_t"].max().astype("int32"),
        "low_t": g["_t"].min().astype("int32"),
        "close_t": g["_t"].last().astype("int32"),
        "volume": g["size"].sum().astype("int32"),
        "n_trades": g["_t"].size().astype("int32"),
        "buy_vol": g["_b"].sum().astype("int32"),
        "sell_vol": g["_s"].sum().astype("int32"),
        "notional": g["_n"].sum(),
    })
    out.index.name = "ts"
    return out


def profile_30m(df: pd.DataFrame, p: TapeParams) -> pd.DataFrame:
    """Volume-at-price for ONE symbol, keyed (bucket, price tick).

    Note this is *traded* volume at price, i.e. a true transaction profile
    rather than the TPO/time profile of classic market profile. Both are worth
    testing; the traded profile is what raw prints give exactly.
    """
    if df.empty:
        return pd.DataFrame()
    buy, sell = _signed(df)
    ticks = np.rint(df["price"].to_numpy() / p.tick_size).astype("int64")
    b = df["ts_event"].dt.floor(BUCKET)
    g = (df.assign(_b=b, _t=ticks, _bv=buy, _sv=sell)
           .groupby(["_b", "_t"], sort=True))
    out = pd.DataFrame({
        "volume": g["size"].sum().astype("int32"),
        "n_trades": g["size"].size().astype("int32"),
        "buy_vol": g["_bv"].sum().astype("int32"),
        "sell_vol": g["_sv"].sum().astype("int32"),
    }).reset_index().rename(columns={"_b": "bucket", "_t": "price_t"})
    out["price_t"] = out["price_t"].astype("int32")
    return out


# --------------------------------------------------------------------------- #
# per-session build
# --------------------------------------------------------------------------- #


@dataclass
class SessionTape:
    session: date
    bars: pd.DataFrame           # [session, symbol, ts, ohlc ticks, flow]
    profile: pd.DataFrame        # [session, symbol, bucket, price_t, volumes]
    symbols: List[str]           # materialised, most-traded first
    volumes: Dict[str, int]      # outright volumes -> feeds NEXT session's roll
    stats: Dict


def build_session_tape(sess: date, cal: Calendar, configs_dir: str,
                       p: TapeParams) -> SessionTape:
    """Build L1 for one trade date across every materialised outright."""
    raw = read_session_trades(sess, cal, configs_dir)
    if raw.empty:
        return SessionTape(sess, pd.DataFrame(), pd.DataFrame(), [], {},
                           {"empty": True})
    raw = raw[raw["symbol"].map(is_quarterly_outright)]
    if raw.empty:
        return SessionTape(sess, pd.DataFrame(), pd.DataFrame(), [], {},
                           {"no_outrights": True})

    volumes = raw.groupby("symbol")["size"].sum().astype("int64").to_dict()
    syms = symbols_to_materialize(volumes, p.min_symbol_volume_share)

    bar_frames, prof_frames, filt = [], [], {}
    for sym in syms:
        one = raw[raw["symbol"] == sym]
        one, st = clean_prints(one, p)
        filt[sym] = st
        if one.empty:
            continue
        b = bars_1s(one, p)
        b.insert(0, "symbol", sym)
        bar_frames.append(b.reset_index())
        pr = profile_30m(one, p)
        pr.insert(0, "symbol", sym)
        prof_frames.append(pr)

    bars = pd.concat(bar_frames, ignore_index=True) if bar_frames else pd.DataFrame()
    prof = pd.concat(prof_frames, ignore_index=True) if prof_frames else pd.DataFrame()
    for f in (bars, prof):
        if not f.empty:
            f.insert(0, "session", pd.to_datetime(sess).date())
    return SessionTape(sess, bars, prof, syms, {k: int(v) for k, v in volumes.items()},
                       {"trade_filter": filt,
                        "n_raw_outright": int(len(raw)),
                        "session_open": str(session_open_utc(sess, cal)),
                        "session_close": str(session_close_utc(sess, cal))})


# --------------------------------------------------------------------------- #
# reading L1 back
# --------------------------------------------------------------------------- #


def ticks_to_price(t, tick_size: float = TICK):
    return np.asarray(t, dtype="float64") * tick_size


def price_to_ticks(px, tick_size: float = TICK):
    return np.rint(np.asarray(px, dtype="float64") / tick_size).astype("int64")


def window_bars(bars: pd.DataFrame, lo, hi) -> pd.DataFrame:
    """Half-open [lo, hi) slice. Every window in this project is half-open, so
    a 09:00 stamp belongs to the next window and never double-counts."""
    ts = bars["ts"]
    return bars[(ts >= lo) & (ts < hi)]


def window_profile(profile: pd.DataFrame, lo, hi) -> pd.DataFrame:
    """Sum 30-minute buckets whose START lies in [lo, hi).

    Callers must pass boundaries on 30-minute marks — every PT segment used by
    the research does. A misaligned boundary would silently drop or absorb a
    whole bucket, so it raises instead.
    """
    for edge in (lo, hi):
        e = pd.Timestamp(edge)
        if (e.minute % 30) or e.second or e.nanosecond:
            raise ValueError(f"profile window edge {e} is not on a 30-minute mark")
    b = profile["bucket"]
    w = profile[(b >= lo) & (b < hi)]
    if w.empty:
        return pd.DataFrame(columns=["price_t", "volume", "n_trades", "buy_vol", "sell_vol"])
    return (w.groupby("price_t", as_index=False)[["volume", "n_trades", "buy_vol", "sell_vol"]]
             .sum().sort_values("price_t").reset_index(drop=True))
