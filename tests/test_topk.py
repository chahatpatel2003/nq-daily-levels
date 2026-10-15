"""The top-k framework must recover a planted lift and return zero on a null.

Everything Phase 3 reports flows through eval/topk.py, so the estimator itself
is tested against synthetic mornings where the truth is known: touch decays
with distance (the real monotone structure), a few 'good' prices per session
carry a planted reaction lift, and everything else is exchangeable.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nqlevels.eval import topk


def synth(n_sessions=140, n_grid=80, lift=0.25, base=0.20, n_good=6, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    d0 = dt.date(2024, 1, 2)
    for s in range(n_sessions):
        sess = d0 + dt.timedelta(days=int(s * 1.4))
        offs = (np.arange(n_grid) - n_grid // 2) * 5.0
        offs = offs[offs != 0]
        atr = 200.0
        dist = np.abs(offs) / atr
        good = np.zeros(offs.size, bool)
        good[rng.choice(offs.size, n_good, replace=False)] = True
        p_touch = np.exp(-2.2 * dist)
        touched = rng.random(offs.size) < p_touch
        p_clean = np.where(good, base + lift, base)
        clean = np.where(rng.random(offs.size) < p_clean, 1.0, 0.0)
        clean = np.where(touched, clean, np.nan)
        # some touched rows unresolved, as in the real label
        unres = touched & (rng.random(offs.size) < 0.10)
        clean = np.where(unres, np.nan, clean)
        rows.append(pd.DataFrame({
            "session": sess, "off_t": offs, "dist_atr": dist,
            "above": (offs > 0).astype("int8"), "touched": touched,
            "clean": clean,
            "rev30m_atr": np.where(touched, rng.gamma(2.0, 0.1, offs.size), np.nan),
            "score_oracle": good.astype(float) + rng.normal(0, 0.01, offs.size),
            "score_random": rng.random(offs.size),
        }))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def frame():
    return synth()


def test_select_topk_respects_separation_and_window(frame):
    g = frame[frame["session"] == frame["session"].iloc[0]]
    p = topk.select_topk(g, "score_random", 6, min_sep_t=25.0)
    offs = np.sort(p["off_t"].to_numpy())
    assert (np.diff(offs) >= 25.0).all()
    p = topk.select_topk(g, "score_random", 6, min_sep_t=5.0, dist_window=(0.2, 0.6))
    assert p["dist_atr"].between(0.2, 0.6).all()
    p = topk.select_topk(g, "score_random", 6, min_sep_t=5.0, per_side_cap=2)
    assert p.groupby("above").size().max() <= 2


def test_matched_controls_share_geometry(frame):
    d = topk.add_bands(frame, n=8)
    g = d[d["session"] == d["session"].iloc[0]]
    rng = np.random.default_rng(0)
    picks = topk.select_topk(g, "score_oracle", 3, min_sep_t=25.0)
    # exclusion radius must be smaller than a band's width for a cell to
    # survive — in the real data bands span ~180 ticks against a 24-tick radius
    ctrl = topk.matched_controls(g, picks, rng, n_draws=6, min_sep_t=10.0)
    assert len(ctrl)
    # every control sits in a band+side some pick occupies, and away from all picks
    pk = set(zip(picks["_dband"], picks["above"]))
    assert all((b, a) in pk for b, a in zip(ctrl["_dband"], ctrl["above"]))
    far = np.abs(ctrl["off_t"].to_numpy()[:, None]
                 - picks["off_t"].to_numpy()[None, :])
    assert (far.min(axis=1) >= 10.0).all()


def test_joint_counts_unresolved_as_no_reaction():
    d = pd.DataFrame({"touched": [True, True, False], "clean": [1.0, np.nan, np.nan]})
    assert topk._joint(d, "clean") == pytest.approx(1 / 3)
    assert topk._cond(d, "clean") == pytest.approx(1.0)


def test_oracle_recovers_planted_lift(frame):
    res = topk.evaluate(frame, "score_oracle", ks=(3,), n_boot=200,
                        n_draws=6, min_sep_t=15.0, seed=5, by_year=False)
    r = res["K3"]
    ci = r["ci_cond"]
    # planted +0.25; matched controls carry ~7% accidental good prices
    assert ci["diff"] > 0.12
    assert ci["lo"] > 0.0
    assert r["ci_joint"]["diff"] > 0.0
    # touch must be matched by construction — no distance smuggling
    assert abs(r["ci_touch"]["diff"]) < 0.08


def test_random_score_reports_nothing(frame):
    res = topk.evaluate(frame, "score_random", ks=(3,), n_boot=200,
                        n_draws=6, min_sep_t=15.0, seed=6, by_year=False)
    ci = res["K3"]["ci_cond"]
    assert ci["lo"] < 0.0 < ci["hi"]
    cj = res["K3"]["ci_joint"]
    assert cj["lo"] < 0.0 < cj["hi"]


def test_day_rates_shape(frame):
    res = topk.evaluate(frame, "score_oracle", ks=(2,), n_boot=100,
                        n_draws=4, min_sep_t=15.0, seed=7, by_year=False)
    a = res["K2"]["model"]
    for k in ("p_ge1_reached", "p_ge1_react", "p_ge2_react"):
        assert 0.0 <= a[k] <= 1.0
    assert a["p_ge2_react"] <= a["p_ge1_react"] <= a["p_ge1_reached"]


def test_reliability_table():
    rng = np.random.default_rng(0)
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(float)
    t = topk.reliability(y, p, n_bins=5)
    assert len(t) == 5
    assert np.all(np.abs(t["p_mean"] - t["y_rate"]) < 0.06)
