#!/usr/bin/env python
"""P16 REFERENCE simulator — deliberately independent, slow, readable.
One path, explicit day loop, full event log. Used to cross-check the
vectorised engine (identical day sequences must give identical cash,
payouts, deaths) and to print human-readable lifecycle traces."""
from __future__ import annotations

import numpy as np


def ref_run(seqs, order, risk, rules, C, T1=0.0, min_take=500.0,
            replace=None, log=None):
    """seqs: per-session arrays of trade R. order: session indices."""
    mll0 = rules["mll0"]; cap = rules["cap"]; frac = rules["frac"]
    winday = rules["winday"]; need = rules["std_days"]
    sp_th = rules["split_thresh"]; sp = rules["split"]
    bal = hwm = 0.0; locked = False
    windays = 0; cyc_net = 0.0
    n_pay = 0; cum_gross = 0.0
    cash = costs = booked = 0.0
    deaths = 0; idle = 0
    ev = log if log is not None else None
    pays = []
    for td, si in enumerate(order):
        if idle > 0:
            idle -= 1
            continue
        a = seqs[si]
        mll = 0.0 if locked else min(0.0, hwm - mll0)
        tot = float(a.sum()) * risk
        c = np.cumsum(a) * risk if len(a) else np.array([])
        minp = min(0.0, float(c.min())) if len(c) else 0.0
        if bal + minp <= mll:                       # death intra-day
            deaths += 1
            if ev is not None:
                ev.append((td, "DEATH", dict(bal=round(bal, 2),
                                             mll=round(mll, 2),
                                             minp=round(minp, 2))))
            if replace is None:
                return dict(cash=cash, n_pay=n_pay, deaths=deaths,
                            bal=bal, dead=True, day=td, pays=pays,
                            booked=booked, costs=costs)
            costs += replace["cost"]; idle = replace["delay"]
            bal = hwm = 0.0; locked = False
            windays = 0; cyc_net = 0.0; cum_gross = 0.0
            continue
        bal += tot; booked += tot
        hwm = max(hwm, bal)
        if tot >= winday:
            windays += 1
        cyc_net += tot
        el = windays >= need and (n_pay == 0 or cyc_net > 0)
        if el and (n_pay > 0 or bal >= T1):
            amt = min(frac * bal, cap, bal - C)
            if amt >= max(min_take, rules["min_req"]):
                within = min(max(sp_th - cum_gross, 0.0), amt)
                got = within + (amt - within) * sp
                cash += got; cum_gross += amt
                bal -= amt; locked = True
                windays = 0; cyc_net = 0.0; n_pay += 1
                pays.append(td)
                if ev is not None:
                    ev.append((td, "PAYOUT", dict(
                        n=n_pay, gross=round(amt, 2), net=round(got, 2),
                        bal_after=round(bal, 2))))
        if ev is not None and abs(tot) > 1e-9:
            ev.append((td, "DAY", dict(pnl=round(tot, 2),
                                       bal=round(bal, 2),
                                       windays=windays)))
    return dict(cash=cash, n_pay=n_pay, deaths=deaths, bal=bal,
                dead=False, day=len(order), pays=pays, booked=booked,
                costs=costs)
