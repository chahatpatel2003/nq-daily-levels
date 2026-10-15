"""Roll, profile, split and leakage-audit machinery."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nqlevels.audit.leakage import (check_decision_cut, check_feature_columns,
                                    check_fit_on_train_only, check_prefix_invariance,
                                    check_roll_causality)
from nqlevels.build.profile import (composite, densify, find_nodes, poc,
                                    profile_sigma, profile_vwap, value_area)
from nqlevels.build.roll import (FrontMonthSelector, carry_level,
                                 is_quarterly_outright, parse_contract,
                                 roll_spread, symbols_to_materialize)
from nqlevels.clock import load_calendar
from nqlevels.eval.splits import (HoldoutSealed, holdout_sessions,
                                  load_split_plan, preregister, walk_forward)

CONFIGS = str(Path(__file__).resolve().parents[1] / "configs")
ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- roll

def test_parse_contract_resolves_single_digit_years():
    assert parse_contract("NQH4", 2024) == (2024, 3)
    assert parse_contract("NQZ1", 2021) == (2021, 12)
    assert parse_contract("NQM24", 2024) == (2024, 6)


def test_spreads_and_junk_are_not_outrights():
    assert not is_quarterly_outright("NQH4-NQM4")
    assert not is_quarterly_outright("NQF4")          # January is not quarterly
    assert is_quarterly_outright("NQU5")


def test_selector_uses_only_prior_session_volume():
    s = FrontMonthSelector()
    assert s.select(date(2024, 3, 1)) is None          # nothing observed yet
    s.observe(date(2024, 3, 1), {"NQH4": 100, "NQM4": 5})
    assert s.select(date(2024, 3, 4)) == "NQH4"
    # M4 overtakes today, but today's front was already fixed by yesterday
    s.observe(date(2024, 3, 4), {"NQH4": 10, "NQM4": 900})
    assert s.select(date(2024, 3, 5)) == "NQM4"


def test_selector_never_rolls_backwards():
    s = FrontMonthSelector()
    s.observe(date(2024, 3, 1), {"NQM4": 100})
    s.select(date(2024, 3, 4))
    s.observe(date(2024, 3, 4), {"NQH4": 9999, "NQM4": 1})   # a thin M4 day
    assert s.select(date(2024, 3, 5)) == "NQM4"


def test_selector_requires_increasing_sessions():
    s = FrontMonthSelector()
    s.observe(date(2024, 3, 1), {"NQH4": 1})
    s.select(date(2024, 3, 5))
    with pytest.raises(ValueError):
        s.select(date(2024, 3, 4))


def test_bootstrap_is_not_a_roll():
    s = FrontMonthSelector()
    s.observe(date(2024, 3, 1), {"NQH4": 100})
    s.select(date(2024, 3, 4))
    assert s.roll_log[0].is_bootstrap


def test_symbols_to_materialize_covers_a_roll():
    v = {"NQH4": 500_000, "NQM4": 400_000, "NQU4": 100, "NQH4-NQM4": 9_000}
    assert symbols_to_materialize(v, 0.005) == ["NQH4", "NQM4"]


def test_roll_spread_and_carry():
    t = pd.Timestamp("2024-03-11 12:00", tz="UTC")
    ts = t + pd.to_timedelta(np.arange(500), "s")
    df = pd.concat([
        pd.DataFrame({"ts_event": ts, "symbol": "NQH4", "price": 18000.0}),
        pd.DataFrame({"ts_event": ts, "symbol": "NQM4", "price": 18120.0})])
    sp = roll_spread(df, "NQH4", "NQM4")
    assert sp == pytest.approx(120.0)
    assert carry_level(18050.0, "NQH4", "NQM4", sp) == (18170.0, "spread_shift")
    assert carry_level(18050.0, "NQH4", "NQH4", None) == (18050.0, "same_contract")
    assert carry_level(18050.0, "NQH4", "NQM4", None) == (None, "unavailable")


def test_roll_spread_returns_none_when_overlap_is_too_thin():
    t = pd.Timestamp("2024-03-11 12:00", tz="UTC")
    df = pd.DataFrame({"ts_event": [t, t], "symbol": ["NQH4", "NQM4"],
                       "price": [18000.0, 18120.0]})
    assert roll_spread(df, "NQH4", "NQM4") is None


# ------------------------------------------------------------------ profile

def prof(pairs):
    return pd.DataFrame({"price_t": [p for p, _ in pairs],
                         "volume": [v for _, v in pairs],
                         "n_trades": 1, "buy_vol": 0, "sell_vol": 0})


def test_densify_fills_untraded_prices_with_zero():
    d = densify(prof([(100, 5), (103, 7)]))
    assert list(d["price_t"]) == [100, 101, 102, 103]
    assert list(d["volume"]) == [5, 0, 0, 7]


def test_poc_and_value_area():
    p = prof([(100, 1), (101, 5), (102, 20), (103, 5), (104, 1)])
    assert poc(p) == 102
    val, pc, vah = value_area(p, 0.70)
    assert pc == 102 and val <= 102 <= vah and val >= 101 and vah <= 103


def test_poc_tie_breaks_toward_the_profile_centre():
    p = prof([(100, 10), (101, 1), (102, 10)])
    assert poc(p) in (100, 102)                 # deterministic, centre-nearest
    p2 = prof([(100, 10), (101, 1), (102, 10), (103, 8)])
    assert poc(p2) == 102


def test_vwap_and_sigma():
    p = prof([(100, 1), (200, 1)])
    assert profile_vwap(p) == pytest.approx(150.0)
    assert profile_sigma(p) == pytest.approx(50.0)


def test_find_nodes_locates_a_planted_hvn_and_lvn():
    base = [(p, 100) for p in range(1000, 1400)]
    d = dict(base)
    for p in range(1140, 1160):
        d[p] = 900                       # HVN
    for p in range(1240, 1260):
        d[p] = 5                         # LVN
    nodes = find_nodes(prof(sorted(d.items())), window_t=8, neighbourhood_t=40)
    hv = [n for n in nodes if n.kind == "hvn"]
    lv = [n for n in nodes if n.kind == "lvn"]
    assert any(1130 <= n.price_t <= 1170 for n in hv)
    assert any(1230 <= n.price_t <= 1270 for n in lv)


def test_composite_weights_older_profiles_down():
    a, b = prof([(100, 10)]), prof([(100, 10)])
    c = composite([a, b], [1.0, 0.5])
    assert float(c.loc[c.price_t == 100, "volume"].iloc[0]) == pytest.approx(15.0)


# ------------------------------------------------------------------- splits

def test_walk_forward_is_chronological_and_embargoed():
    s = [date(2021, 1, 4) + timedelta(days=i) for i in range(600)]
    folds = walk_forward(s, train_sessions=250, test_sessions=63, embargo=5)
    assert folds
    for f in folds:
        assert max(f.train) < min(f.test)
        gap = (min(f.test) - max(f.train)).days
        assert gap >= 5
    starts = [min(f.test) for f in folds]
    assert starts == sorted(starts)


def test_expanding_folds_grow():
    s = [date(2021, 1, 4) + timedelta(days=i) for i in range(600)]
    f = walk_forward(s, 250, 63, 5, expanding=True)
    assert len(f[-1].train) >= len(f[0].train)


def test_holdout_is_sealed_without_a_token():
    plan = load_split_plan(CONFIGS)
    s = [date(2026, 2, 2), date(2026, 2, 3)]
    with pytest.raises(HoldoutSealed):
        holdout_sessions(s, plan, ROOT)
    with pytest.raises(HoldoutSealed):
        holdout_sessions(s, plan, ROOT, token="not-a-real-token")


def test_preregistration_unlocks_exactly_its_own_token(tmp_path):
    plan = load_split_plan(CONFIGS)
    tok = preregister(tmp_path, "unit_test_hypothesis", "h", "rule", {}, {})
    s = [date(2026, 2, 2), date(2025, 1, 2)]
    got = holdout_sessions(s, plan, tmp_path, token=tok)
    assert got == [date(2026, 2, 2)]           # 2025 is validation, not holdout


def test_split_ranges_do_not_overlap():
    p = load_split_plan(CONFIGS)
    assert p.train.end < p.validation.start < p.validation.end < p.holdout.start


# ------------------------------------------------------------------- audits

def test_decision_cut_audit_catches_a_late_row():
    cal = load_calendar(CONFIGS)
    s = date(2024, 3, 5)
    ok = pd.DataFrame({"ts": pd.to_datetime(["2024-03-05 14:00"], utc=True)})
    bad = pd.DataFrame({"ts": pd.to_datetime(["2024-03-05 15:00"], utc=True)})
    assert check_decision_cut({"pre": ok}, s, cal).passed
    assert not check_decision_cut({"pre": bad}, s, cal).passed


def test_feature_column_audit_blocks_outcome_columns():
    assert check_feature_columns(["atr_t", "gap_t_prior"]).passed
    assert not check_feature_columns(["atr_t", "target_high_out"]).passed


def test_roll_causality_audit():
    good = [{"session": "2024-03-12", "decided_from_session": "2024-03-11",
             "is_bootstrap": False}]
    bad = [{"session": "2024-03-12", "decided_from_session": "2024-03-12",
            "is_bootstrap": False}]
    assert check_roll_causality(good).passed
    assert not check_roll_causality(bad).passed


def test_prefix_invariance_detects_an_unshifted_rolling_window():
    """A shifted ATR window is prefix-invariant; an unshifted one is not."""
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"session": range(400),
                      "rng": np.abs(rng.normal(size=400).cumsum() + 100)})

    def causal(x):
        x = x.copy()
        x["atr"] = x["rng"].rolling(14).mean().shift(1)
        return x

    def leaky(x):
        x = x.copy()
        # centred window: row i is computed partly from rows after i
        x["atr"] = x["rng"].rolling(14, center=True).mean()
        return x

    assert check_prefix_invariance(causal, d, ["atr"]).passed
    assert not check_prefix_invariance(leaky, d, ["atr"]).passed


def test_fit_eval_disjointness():
    a = [date(2024, 1, 2), date(2024, 1, 3)]
    b = [date(2024, 2, 2)]
    assert check_fit_on_train_only(a, b).passed
    assert not check_fit_on_train_only(a, a).passed
