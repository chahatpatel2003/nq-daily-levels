"""Permanent chronology / leakage guards (Phase 14).

Data-level checks against the real stored layers plus a source scan of the
live-path modules. A candidate strategy cannot be READY unless these pass.
The decisive philosophy: delete the future and require nothing to move
(truncation invariance) — reading code is not proof.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DAILY = Path("/Volumes/T9/nq_levels/daily")

LIVE_PATH_MODULES = [
    ROOT / "src/nqlevels/features/interactions.py",
    ROOT / "src/nqlevels/features/zones.py",
    ROOT / "src/nqlevels/labels/sweep.py",
    ROOT / "src/nqlevels/labels/enrich.py",
    ROOT / "scripts/build_interactions.py",
    ROOT / "scripts/build_p9_timeline.py",
    ROOT / "scripts/run_p10_scores.py",
    ROOT / "scripts/build_p14_fire.py",
]

needs_data = pytest.mark.skipif(not DAILY.exists(),
                                reason="T9 data volume not mounted")


def test_no_leak_patterns_in_live_path():
    """No centered windows, negative shifts, backward fills or nearest
    joins in any module on the live decision path."""
    bad = re.compile(r"center\s*=\s*True|shift\(\s*-\d|\.bfill\(|"
                     r"backfill|direction\s*=\s*['\"]nearest")
    hits = []
    for p in LIVE_PATH_MODULES:
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if bad.search(line) and "noqa: leak-ok" not in line:
                hits.append(f"{p.name}:{i}: {line.strip()}")
    assert not hits, "leak patterns in live path:\n" + "\n".join(hits)


@needs_data
def test_zone_confirmed_before_touch():
    """Every interaction's tap must occur at/after the zone's formation-
    confirmation timestamp (zigzag pivots repaint until confirmed)."""
    it = pd.read_parquet(DAILY / "interactions.parquet",
                         columns=["session", "zid", "tap_sec", "tap_ts"])
    zs = pd.read_parquet(DAILY / "zones.parquet",
                         columns=["zid", "form_ts_val"]).drop_duplicates("zid")
    m = it.merge(zs, on="zid", how="left")
    ok = m["form_ts_val"].notna()
    viol = (m.loc[ok, "tap_ts"] < m.loc[ok, "form_ts_val"]).sum()
    assert viol == 0, f"{viol} taps precede zone confirmation"


@needs_data
def test_tap_no_monotone_within_zone():
    it = pd.read_parquet(DAILY / "interactions.parquet",
                         columns=["zid", "tap_ts", "tap_no"])
    g = it.sort_values("tap_ts").groupby("zid")["tap_no"]
    assert (g.diff().dropna() >= 0).all(), "tap_no decreases within a zone"


def _exact_prior_viol(df, ncol):
    """Rows whose stored prior-count EXCEEDS the number of same-zone taps
    matured (>=2h old) at that moment. Exact recomputation — a diff-based
    check false-positives when an OLDER tap crosses maturity."""
    df = df.sort_values(["zid", "tap_ts"])
    viol = 0
    for _, g in df.groupby("zid", sort=False):
        ts = g["tap_ts"].to_numpy()
        n = g[ncol].to_numpy()
        for i in range(len(g)):
            legal = int(np.searchsorted(ts, ts[i] - 7200, side="right"))
            if n[i] > legal:
                viol += 1
    return viol


@needs_data
@pytest.mark.xfail(strict=True, reason="AUDIT FINDING #2 (P14-386, "
                   "corrected): legacy interactions layer counts an "
                   "immature prior outcome on 1,554 taps (0.455%). Layer "
                   "marked dirty; P14 fire dataset is exact-clean — see "
                   "test_p14_prior_maturity.")
def test_prior_outcome_maturity_2h_legacy():
    it = pd.read_parquet(DAILY / "interactions.parquet",
                         columns=["zid", "tap_ts", "n_prior_A", "n_prior_B",
                                  "n_prior_C"])
    it["tot"] = it[["n_prior_A", "n_prior_B", "n_prior_C"]].sum(axis=1)
    assert _exact_prior_viol(it, "tot") == 0


@needs_data
def test_p14_prior_maturity():
    p = DAILY / "p14_fire.parquet"
    if not p.exists():
        pytest.skip("p14_fire not built yet")
    d = pd.read_parquet(p, columns=["zid", "tap_ts", "zprior_n"])
    assert _exact_prior_viol(d, "zprior_n") == 0


@needs_data
def test_walkforward_year_chronology():
    """Scores exist only for years with strictly-prior training data:
    2021 must be unscored; sessions never straddle a calendar year, so no
    label embargo is required at year boundaries."""
    sc = pd.read_parquet(DAILY / "p10_scores.parquet",
                         columns=["session", "p_both60"])
    yr = pd.to_datetime(sc["session"].astype(str)).dt.year
    assert sc.loc[yr <= 2021, "p_both60"].notna().sum() == 0
    assert sc.loc[yr >= 2022, "p_both60"].notna().mean() > 0.5


@needs_data
def test_roll_one_front_per_session():
    ra = pd.read_parquet(DAILY / "roll_adjust.parquet")
    assert ra["session"].is_unique, "multiple front rows for a session"


@needs_data
def test_fire_features_truncation_invariance():
    """Rebuild fire-time rows for sample sessions from arrays physically
    truncated at each decision second; require identical values to the
    full-tape build. Covers bar-close availability, rolling windows,
    resampling and the R4 clock in one shot."""
    from p14_common import fire_rows_for_session, load_session_arrays  # noqa
    exf = pd.read_parquet(DAILY / "p14_fire.parquet")
    for sess in np.random.default_rng(0).choice(
            exf["session"].unique(), 2, replace=False):
        full = exf[exf["session"] == sess]
        sd, g = load_session_arrays(sess)
        for r in full.head(3).itertuples():
            rows = fire_rows_for_session(sd, g, truncate_at=int(r.dec_sec))
            got = [x for x in rows if x["tap_sec"] == r.tap_sec
                   and x["zid"] == r.zid]
            assert got, f"{sess} {r.zid} fire row vanished under truncation"
            for k, v in got[0].items():
                if isinstance(v, float) and k in full.columns:
                    ref = float(getattr(r, k))
                    assert (np.isnan(v) and np.isnan(ref)) or \
                        abs(v - ref) < 1e-9, f"{k}: {v} != {ref} ({sess})"
