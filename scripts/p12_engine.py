#!/usr/bin/env python
"""P12 funded-account lifecycle engine, vectorised over simulation paths.

Rule model (Topstep 150K XFA snapshot 2026-08, all parameterised):
  start $0; initial MLL -mll0; MLL value = min(0, EOD-balance-highwater -
  mll0), monotone, locks at $0; permanently $0 after any payout. Intraday
  equity <= MLL kills the account (checked at trade granularity via the
  day's min prefix cum; `adverse` variant subtracts a full stop during
  each open trade). Balance at death is forfeited. Standard path: >=
  `std_days` winning days (day net >= winday$) per cycle, and (after the
  first payout) positive cycle net. Consistency path: >= cons_days trading
  days, cycle net > 0, largest winning day <= cons_frac * cycle net.
  Payout: <= frac * balance, <= cap, processed at EOD of the eligible day
  (simplification), resets the cycle. Split: 100% of the first
  `split_thresh` gross payout dollars per account, `split` thereafter.
  Optional DLL: the trading day is cut at the trade that crosses -dll$.

Policy: T1 = balance required before requesting payout #1; C = cushion
that must remain after any payout; min_take = smallest payout worth
requesting; frac = fraction cap (<= rule frac). Dynamic risk via risk2 +
switch ("pay", k) or ("cash", $).

Replacement mode: at death pay `cost`, sit out `delay` trading days,
restart a fresh account (split counter resets — per-account threshold).
"""
from __future__ import annotations

import pickle

import numpy as np

DAILY = "/Volumes/T9/nq_levels/daily"
ADVERSE_R = 1.08          # realised full-stop loss incl. base cost


def load_streams():
    with open(f"{DAILY}/p12_streams.pkl", "rb") as f:
        d = pickle.load(f)
    return d["sessions"], d["streams"]


class Stream:
    """Per-session day aggregates for one execution stream (R units)."""

    def __init__(self, seqs, dll_R=None):
        m = len(seqs)
        self.tot = np.zeros(m)
        self.minpre = np.zeros(m)
        self.minpre_adv = np.zeros(m)
        self.n = np.zeros(m, dtype=np.int64)
        for i, a in enumerate(seqs):
            if len(a) == 0:
                continue
            c = np.cumsum(a)
            if dll_R is not None:
                k = np.flatnonzero(c <= -dll_R)
                if k.size:
                    a = a[:k[0] + 1]
                    c = c[:k[0] + 1]
            self.tot[i] = c[-1]
            self.minpre[i] = min(0.0, float(c.min()))
            pad = np.concatenate(([0.0], c[:-1]))
            self.minpre_adv[i] = min(self.minpre[i],
                                     float((pad - ADVERSE_R).min()))
            self.n[i] = len(a)


STD = dict(mll0=4500.0, winday=150.0, std_days=5, cap=5000.0, frac=0.5,
           min_req=125.0, split_thresh=10000.0, split=0.90,
           consistency=False)
CONS = dict(STD, consistency=True, cons_days=3, cons_frac=0.40, cap=6000.0)
K_TRACK = (1, 2, 3, 5, 10)


def make_idx(n_sessions, n_paths, horizon, block=5, seed=12):
    rng = np.random.default_rng(seed)
    nblk = horizon // block + 1
    starts = rng.integers(0, n_sessions - block, size=(n_paths, nblk))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :])
    return idx.reshape(n_paths, -1)[:, :horizon]


def simulate(st: Stream, idx, risk, rules, pol=None, replace=None,
             adverse=False, cash_marks=()):
    pol = pol or {}
    N, H = idx.shape
    mark_day = {m: np.full(N, -1, np.int64) for m in cash_marks}
    booked = np.zeros(N)               # trading P&L actually credited
    forfeit = np.zeros(N)              # balance abandoned at deaths
    cap_lost = np.zeros(N)             # payout $ blocked by the cap
    gross_paid = np.zeros(N)           # payouts before profit split
    gaps: list = []                    # sessions between consecutive payouts
    last_pay = np.full(N, -1, np.int64)
    mll0 = rules["mll0"]
    tot_R = st.tot[idx]
    minp_R = (st.minpre_adv if adverse else st.minpre)[idx]
    ntr = st.n[idx]

    T1 = pol.get("T1", 0.0)
    C = pol.get("C", 0.0)
    frac = min(pol.get("frac", rules["frac"]), rules["frac"])
    min_take = max(pol.get("min_take", 500.0), rules["min_req"])
    risk2 = pol.get("risk2")
    switch = pol.get("switch")

    bal = np.zeros(N); hwm = np.zeros(N)
    locked = np.zeros(N, bool)
    windays = np.zeros(N, np.int64); cyc_net = np.zeros(N)
    cyc_td = np.zeros(N, np.int64); cyc_mw = np.zeros(N)
    n_pay = np.zeros(N, np.int64); cum_gross = np.zeros(N)
    cash = np.zeros(N); costs = np.zeros(N)
    dead = np.zeros(N, bool); idle = np.zeros(N, np.int64)
    deaths = np.zeros(N, np.int64)
    first_pay = np.full(N, -1, np.int64)
    death_day = np.full(N, -1, np.int64)
    npay_death1 = np.full(N, -1, np.int64)
    cash_death1 = np.full(N, np.nan)
    K = max(K_TRACK)
    kday = np.full((N, K + 1), -1, np.int64)
    kcash = np.full((N, K + 1), np.nan)
    cons_block = np.zeros(N, np.int64)
    rk = np.full(N, float(risk))

    for td in range(H):
        act = ~dead & (idle == 0)
        idle[idle > 0] -= 1
        if not act.any():
            if replace is None:
                break
            continue
        mll = np.where(locked, 0.0, np.minimum(0.0, hwm - mll0))
        tot = tot_R[:, td] * rk
        minp = minp_R[:, td] * rk

        die = act & (bal + minp <= mll)
        if die.any():
            fd = die & (deaths == 0)
            death_day[fd] = td
            npay_death1[fd] = n_pay[fd]
            cash_death1[fd] = cash[fd]
            deaths[die] += 1
            forfeit[die] += bal[die]
            if replace is None:
                dead[die] = True
            else:
                costs[die] += replace["cost"]
                idle[die] = replace["delay"]
                bal[die] = 0.0; hwm[die] = 0.0
                locked[die] = False; windays[die] = 0
                cyc_net[die] = 0.0; cyc_td[die] = 0
                cyc_mw[die] = 0.0; cum_gross[die] = 0.0
                if risk2 is not None and pol.get("reset_risk", True):
                    rk[die] = float(risk)
            act &= ~die

        bal[act] += tot[act]
        booked[act] += tot[act]
        hwm[act] = np.maximum(hwm[act], bal[act])
        traded = act & (ntr[:, td] > 0)
        windays[act & (tot >= rules["winday"])] += 1
        cyc_net[act] += tot[act]
        cyc_td[traded] += 1
        up = act & (tot > 0)
        cyc_mw[up] = np.maximum(cyc_mw[up], tot[up])

        if rules.get("consistency"):
            base_el = act & (cyc_td >= rules["cons_days"]) & (cyc_net > 0)
            el = base_el & (cyc_mw <= rules["cons_frac"] * cyc_net)
            cons_block += (base_el & ~el)
        else:
            el = (act & (windays >= rules["std_days"])
                  & ((n_pay == 0) | (cyc_net > 0)))
        trig = el & ((n_pay > 0) | (bal >= T1))
        uncapped = np.minimum(frac * bal, bal - C)
        amt = np.minimum(uncapped, rules["cap"])
        ok = trig & (amt >= min_take)
        if ok.any():
            cap_lost[ok] += np.clip(uncapped[ok] - amt[ok], 0, None)
            gp = td - last_pay[ok]
            gaps.append(gp[last_pay[ok] >= 0])
            last_pay[ok] = td
            a = amt[ok]
            within = np.clip(rules["split_thresh"] - cum_gross[ok], 0.0,
                             a)
            cash[ok] += within + (a - within) * rules["split"]
            gross_paid[ok] += a
            cum_gross[ok] += a
            bal[ok] -= a
            locked[ok] = True
            windays[ok] = 0; cyc_net[ok] = 0.0
            cyc_td[ok] = 0; cyc_mw[ok] = 0.0
            n_pay[ok] += 1
            fp = ok & (first_pay < 0)
            first_pay[fp] = td
            for k in K_TRACK:
                hit = ok & (n_pay == k)
                kday[hit, k] = td
                kcash[hit, k] = cash[hit]
            for mk in cash_marks:
                hit = ok & (cash >= mk) & (mark_day[mk] < 0)
                mark_day[mk][hit] = td
            if risk2 is not None and switch is not None:
                kind, thr = switch
                cond = ok & ((n_pay >= thr) if kind == "pay"
                             else (cash >= thr))
                rk[cond] = float(risk2)

    return dict(bal=bal, cash=cash, costs=costs, net=cash - costs,
                n_pay=n_pay, deaths=deaths, dead=dead,
                first_pay=first_pay, death_day=death_day,
                npay_death1=npay_death1, cash_death1=cash_death1,
                kday=kday, kcash=kcash, cons_block=cons_block,
                mark_day=mark_day, booked=booked, forfeit=forfeit,
                cap_lost=cap_lost, gross_paid=gross_paid,
                gaps=np.concatenate(gaps) if gaps else np.empty(0),
                last_pay=last_pay)


def _selftest():
    class S:                                     # 3 synthetic sessions
        pass
    seqs = [np.array([2.4]), np.array([-1.08, -1.08]),
            np.array([2.4, 2.4])]
    st = Stream(seqs)
    assert np.isclose(st.tot[1], -2.16) and np.isclose(st.minpre[1], -2.16)
    # 5 winning days then payout, risk $1000
    idx = np.array([[0, 0, 0, 0, 0, 2]])
    r = simulate(st, idx, 1000.0, STD, dict(T1=0.0, C=0.0))
    assert r["n_pay"][0] == 1 and np.isclose(r["cash"][0], 5000.0), r
    assert np.isclose(r["bal"][0], 12000 - 5000 + 4800)
    assert r["first_pay"][0] == 4
    # death on third consecutive -2.16R day at $1000 (breach mid-day 3)
    idx = np.array([[1, 1, 1, 1]])
    r = simulate(st, idx, 1000.0, STD)
    assert r["dead"][0] and r["death_day"][0] == 2, r
    # split: cap raises past 10k threshold
    idx = np.array([[0] * 5 + [2] * 20])
    rules = dict(STD, cap=12000.0)
    r = simulate(st, idx, 1000.0, rules, dict(min_take=125.0))
    assert r["cash"][0] < r["n_pay"][0] * 12000 or r["n_pay"][0] == 0
    # consistency: one huge day blocks (mw > 40% of net)
    st2 = Stream([np.array([12.0]), np.array([0.6])])
    idx = np.array([[0, 1, 1, 1]])
    r = simulate(st2, idx, 1000.0, CONS)
    assert r["n_pay"][0] == 0                     # 12000 > .4*(12000+1800)
    idx = np.array([[0] + [1] * 35])
    r = simulate(st2, idx, 1000.0, CONS)
    assert r["n_pay"][0] >= 1                     # 30+ small days needed
    print("engine selftest OK")


if __name__ == "__main__":
    _selftest()
