"""Production resistance engine.

PROVENANCE / FIDELITY CONTRACT
------------------------------
This module contains NO new market logic. It executes the frozen research
call-graph over a rolling window of sessions:

  bars_1s + bars_1m + profiles  (layers, or live aggregation later)
  -> nqlevels.features.zones.make_zones            (frozen, imported)
  -> nqlevels.features.interactions.process_session (frozen, imported)
  with the exact per-session sequence of scripts/build_interactions.py
  (chain reset, seg codes, vol/ext deques, zid counter, deque updates
  AFTER processing) mirrored line-for-line.

ATR (atr14_t/atr_pctile) and the roll chain are read from the frozen
context/roll_adjust layers in historical mode; a live deployment
maintains them with the same frozen recipes (add_rolling_scales /
FrontMonthSelector + roll_adjustment) — both independently verified to
0.000000 ticks in P17B/P18b.

Snapshots are pure causal filters over the window result (zones formed
<= t, taps <= t, breaks <= t). Restart safety follows from determinism:
state is never incrementally mutated on disk — a restart replays the
window and lands on the identical state (tested).

Behavioral changes to the frozen methodology: ZERO.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from ..features.interactions import (build_session_data, process_session)
from ..features.zones import make_zones
from build_interactions import make_vol_feats, seg_codes   # frozen, imported

ROOT = Path(__file__).resolve().parents[3]
DEV = Path("/Volumes/T9/nq_levels")
SEAL = Path("/Volumes/T9/nq_levels/sealed_holdout_1/derived")
WINDOW = 50          # 30-session lifecycle + ATR/extrema lookbacks + margin
WARMUP = 35          # snapshots only served for sessions past this depth


def epoch_s(ts_like) -> np.ndarray:
    return pd.DatetimeIndex(ts_like).as_unit("ns").asi8 // 10 ** 9


def causal_break_zids(res_taps: pd.DataFrame, session: str, t_abs: int,
                      ep1s: np.ndarray, hi1s: np.ndarray) -> set:
    """Zids broken *as knowable at t_abs*, from resistance-side tap rows.

    The frozen break rule is o_depth_max_full >= width + 20t, but that
    observable is measured tap -> end of session. For taps in PAST
    sessions the whole window predates t_abs, so the rule is causal as
    stored. For taps in the snapshot session the depth may only be
    reached after t_abs: resolve those by the first 1-second bar at or
    after the tap whose high reaches zhi + width + 20t — the moment the
    break became knowable — and require it <= t_abs. Timestamps follow
    the frozen bar-start stamping convention used for taps/formation."""
    cand = res_taps[res_taps["o_depth_max_full"]
                    >= (res_taps["zhi_t"] - res_taps["zlo_t"]) + 20]
    if not len(cand):
        return set()
    out = set(cand.loc[cand["session"] != session, "zid"])
    for r in cand[cand["session"] == session].itertuples():
        if r.zid in out:
            continue
        j0 = np.searchsorted(ep1s, r.tap_ts, "left")
        thr_px = r.zhi_t + (r.zhi_t - r.zlo_t) + 20
        hit = np.flatnonzero(hi1s[j0:] >= thr_px)
        if hit.size and ep1s[j0 + hit[0]] <= t_abs:
            out.add(r.zid)
    return out


class ResistanceEngine:
    """Rolling-window executor of the frozen zone/tap methodology."""

    def __init__(self, window: int = WINDOW):
        self.window = window
        self._layer_cache: Dict[str, pd.DataFrame] = {}
        self.zones: List[dict] = []
        self.taps: List[dict] = []
        self.sessions: List[str] = []
        self._sd_cache: Dict[str, object] = {}

    # ---------------- data access (historical layers) -----------------
    def _ctx(self) -> pd.DataFrame:
        if "ctx" not in self._layer_cache:
            frames = []
            for base, lo, hi in ((DEV / "daily", None, "2026-06-17"),
                                 (SEAL / "daily", "2026-06-18", None)):
                c = pd.read_parquet(base / "context.parquet")
                c["session"] = c["session"].astype(str)
                if lo:
                    c = c[c["session"] >= lo]
                if hi:
                    c = c[c["session"] <= hi]
                frames.append(c)
            self._layer_cache["ctx"] = pd.concat(frames, ignore_index=True) \
                .drop_duplicates("session").sort_values("session")
        return self._layer_cache["ctx"]

    def _adj(self) -> pd.DataFrame:
        if "adj" not in self._layer_cache:
            frames = []
            for base, lo, hi in ((DEV / "daily", None, "2026-06-17"),
                                 (SEAL / "daily", "2026-06-18", None)):
                a = pd.read_parquet(base / "roll_adjust.parquet")
                a["session"] = a["session"].astype(str)
                if lo:
                    a = a[a["session"] >= lo]
                if hi:
                    a = a[a["session"] <= hi]
                frames.append(a)
            self._layer_cache["adj"] = pd.concat(frames, ignore_index=True) \
                .drop_duplicates("session")
        return self._layer_cache["adj"]

    def _bars(self, sess: str, layer: str) -> Optional[pd.DataFrame]:
        mon = sess[:7]
        root = DEV if sess <= "2026-06-17" else SEAL
        sub = {"bars_1s": "bars_1s", "bars_1m": None,
               "profiles": "profiles"}[layer]
        if layer == "bars_1m":
            key = f"b1m_{root}"
            if key not in self._layer_cache:
                b = pd.read_parquet((root / "daily" if root == DEV
                                     else root / "daily") / "bars_1m.parquet")
                b["ts"] = pd.DatetimeIndex(b["ts"]).tz_convert("UTC")
                b = b[b["is_front"]]
                b["session"] = b["session"].astype(str)
                self._layer_cache[key] = b
            b = self._layer_cache[key]
            g = b[b["session"] == sess]
            return g.sort_values("ts").reset_index(drop=True) if len(g) \
                else None
        p = (root / sub / f"{mon}.parquet")
        key = f"{layer}_{p}"
        if key not in self._layer_cache:
            if not p.exists():
                return None
            d = pd.read_parquet(p)
            d = d[d["is_front"]] if "is_front" in d.columns else d
            d["session"] = d["session"].astype(str)
            self._layer_cache[key] = d
        d = self._layer_cache[key]
        g = d[d["session"] == sess]
        if "ts" in g.columns:
            return g.sort_values("ts").reset_index(drop=True)
        return g.reset_index(drop=True)

    # ---------------- the frozen per-session loop ----------------------
    def run_window(self, end_session: str,
                   trunc_last_at: Optional[int] = None) -> None:
        ctx = self._ctx()
        adj = self._adj().set_index("session")
        all_sessions = [s for s in ctx["session"] if s <= end_session]
        window = all_sessions[-self.window:]
        ctxi = ctx.set_index("session")
        ord_of = {s: i for i, s in enumerate(all_sessions)}
        registry: Dict[int, object] = {}
        zid_counter = 0
        cur_chain = None
        prof_deque, ext_deque = [], []
        zones_all, rows_all = [], []
        for s in window:
            if s not in adj.index or s not in ctxi.index:
                continue
            atr = float(ctxi.loc[s, "atr14_t"])
            if not np.isfinite(atr) or atr <= 0:
                continue
            b1s = self._bars(s, "bars_1s")
            b1m = self._bars(s, "bars_1m")
            if b1s is None or b1m is None or len(b1s) < 3600 \
                    or len(b1m) < 120:
                continue
            if trunc_last_at is not None and s == window[-1]:
                # absolute cut anchored to the 1m session open (same base
                # the snapshot uses), so boundary semantics match exactly
                ep0m = epoch_s(b1m["ts"])[0]
                t_cut = ep0m + trunc_last_at
                b1s = b1s[epoch_s(b1s["ts"]) <= t_cut]
                b1m = b1m[epoch_s(b1m["ts"]) + 60 <= t_cut + 60]
                b1m = b1m[epoch_s(b1m["ts"]) <= t_cut]
                if len(b1s) < 3600 or len(b1m) < 120:
                    break
            sd = build_session_data(b1s.reset_index(drop=True),
                                    b1m.reset_index(drop=True))
            adj_s = float(adj.loc[s, "adj_t"])
            chain = int(adj.loc[s, "chain"])
            if cur_chain is not None and chain != cur_chain:
                registry.clear()
            cur_chain = chain
            sess_ord = ord_of[s]
            m_pt = ((epoch_s(b1m["ts"]) - 8 * 3600) % 86400) // 60
            segm = seg_codes(m_pt)
            new = make_zones(b1m.reset_index(drop=True), atr, adj_s,
                             sess_ord, segm, zid_counter)
            zid_counter += len(new)
            hi20 = max(x[0] for x in ext_deque) if ext_deque else np.nan
            lo20 = min(x[1] for x in ext_deque) if ext_deque else np.nan
            vf = make_vol_feats(prof_deque, adj_s, atr)
            rows = process_session(registry, new, sd, atr, adj_s, chain,
                                   sess_ord, int(ctxi.loc[s, "dow"]),
                                   float(ctxi.loc[s, "atr_pctile"]), vf,
                                   (hi20, lo20))
            for r in rows:
                r["session"] = s
            rows_all.extend(rows)
            for z in new:
                zones_all.append({"session": s, **{k: v for k, v in
                                  asdict(z).items() if k != "retired"}})
            p = self._bars(s, "profiles")
            if p is not None and len(p):
                prof_deque.append(((p["price_t"].to_numpy("float64")
                                    - adj_s),
                                   p["volume"].to_numpy("float64")))
                if len(prof_deque) > 5:
                    prof_deque.pop(0)
            ext_deque.append((float(b1m["high_t"].max()) - adj_s,
                              float(b1m["low_t"].min()) - adj_s))
            if len(ext_deque) > 20:
                ext_deque.pop(0)
        self.zones = zones_all
        self.taps = rows_all
        self.sessions = window
        self._ord_of = ord_of

    # ---------------- causal snapshot ---------------------------------
    def snapshot(self, session: str, at_sec: int) -> dict:
        """Active resistance state at (session, second-of-session)."""
        zdf = pd.DataFrame(self.zones)
        tdf = pd.DataFrame(self.taps)
        b1m = self._bars(session, "bars_1m")
        ep = epoch_s(b1m["ts"])
        t_abs = ep[0] + at_sec
        i = np.searchsorted(ep, t_abs, "right") - 1
        adj_s = float(self._adj().set_index("session")
                      .loc[session, "adj_t"])
        px = float(b1m["close_t"].iloc[max(0, i)]) - adj_s
        so = self._ord_of[session]
        z = zdf[(zdf["form_ts_val"] <= t_abs)
                & (so - zdf["session"].map(self._ord_of) <= 30)].copy()
        t_ok = tdf[tdf["tap_ts"] <= t_abs]
        ntaps = t_ok.groupby("zid").size()
        z["n_taps"] = z["zid"].map(ntaps).fillna(0).astype(int)
        res_taps = t_ok[t_ok["app_side_above"] == 0]
        b1s = self._bars(session, "bars_1s")
        broken = causal_break_zids(res_taps, session, t_abs,
                                   epoch_s(b1s["ts"]),
                                   b1s["high_t"].to_numpy("float64"))
        z["broken"] = z["zid"].isin(broken)
        z["lo"] = z["price_rf"] - z["hw_t"]
        z["hi"] = z["price_rf"] + z["hw_t"]
        z["age_sessions"] = so - z["session"].map(self._ord_of)
        live = z[(~z["broken"]) & (z["n_taps"] < 6)]
        above = live[live["lo"] > px].sort_values("lo")
        return dict(session=session, at_sec=at_sec, t_abs=int(t_abs),
                    price=px, zones=above, all_zones=z, taps=t_ok)
