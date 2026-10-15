# Data Audit — NQ Daily Levels

Audit performed 2026-08-14 against `/Volumes/T9`. Every number below was
measured from disk, not taken from a handoff note. Where a prior handoff
disagreed with the disk, the disk wins and the difference is called out.

---

## 1. What exists

### 1.1 Raw NQ trades — `/Volumes/T9/nq data/raw/trades/`

11 half-year parquet files, **547,020,258 rows**, **13 GB**.

| file | rows | GB | ts_event min (UTC) | ts_event max (UTC) |
|---|---:|---:|---|---|
| 2021_H1 | 43,112,586 | 1.11 | 2021-01-03 23:00:00 | 2021-06-30 23:59:55 |
| 2021_H2 | 42,552,097 | 1.10 | 2021-07-01 00:00:00 | 2021-12-31 21:59:59 |
| 2022_H1 | 60,018,704 | 1.51 | 2022-01-02 23:00:00 | 2022-06-30 23:59:59 |
| 2022_H2 | 55,802,672 | 1.43 | 2022-07-01 00:00:00 | 2022-12-30 21:59:59 |
| 2023_H1 | 50,285,745 | 1.31 | 2023-01-02 23:00:00 | 2023-06-30 20:59:59 |
| 2023_H2 | 50,488,364 | 1.30 | 2023-07-02 22:00:00 | 2023-12-29 21:59:58 |
| 2024_H1 | 51,075,993 | 1.30 | 2024-01-01 23:00:00 | 2024-06-30 23:59:59 |
| 2024_H2 | 48,648,606 | 1.23 | 2024-07-01 00:00:00 | 2024-12-31 21:59:59 |
| 2025_H1 | 52,731,822 | 1.32 | 2025-01-01 23:00:00 | 2025-06-30 23:59:59 |
| 2025_H2 | 45,221,781 | 1.14 | 2025-07-01 00:00:00 | 2025-12-31 21:59:59 |
| 2026_H1 | 47,081,888 | 1.17 | 2026-01-01 23:00:00 | 2026-06-17 03:03:02 |

Files do **not** overlap in time. A trade date spanning a half-year seam has
rows in two files, which is correct; no de-duplication is needed for trades.

Schema (Databento trades, `rtype` 0):

```
ts_event   timestamp[ns, tz=UTC]   exchange event time
rtype      uint8      always 0
publisher_id uint16   always 1
instrument_id uint32
action     large_string   always 'T'
side       large_string   'A' | 'B' | 'N'
depth      uint8      always 0
price      double     index points, exact multiple of 0.25
size       uint32
flags      uint8      always 0
ts_in_delta int32
sequence   uint32
symbol     large_string
ts_recv    timestamp[ns, tz=UTC]   -- stored as the pandas INDEX, not a column
```

### 1.2 Raw NQ MBP-1 — `/Volumes/T9/nq data/raw/mbp1/`

9 half-year parquet files, **7,893,313,081 rows**, **207 GB**.
**Coverage ends 2025-06-30.** There is no MBP-1 for 2025-H2 or 2026-H1.

| file | rows | GB | ts_event min (UTC) | ts_event max (UTC) |
|---|---:|---:|---|---|
| 2021_H1 | 657,942,643 | 18.6 | 2020-12-27 20:00:24 | 2021-06-30 23:59:59 |
| 2021_H2 | 705,575,236 | 19.8 | 2021-06-30 16:50:56 | 2021-12-31 22:00:00 |
| 2022_H1 | 964,817,016 | 27.1 | 2022-01-02 20:00:06 | 2022-06-30 23:59:59 |
| 2022_H2 | 841,208,433 | 23.6 | 2022-06-26 19:00:10 | 2022-12-30 22:00:00 |
| 2023_H1 | 778,472,503 | 21.9 | 2023-01-01 20:00:04 | 2023-06-30 21:00:00 |
| 2023_H2 | 874,549,719 | 24.3 | 2023-07-02 12:00:03 | 2023-12-31 13:00:04 |
| 2024_H1 | 875,658,962 | 24.4 | 2023-12-31 13:00:04 | 2024-06-30 23:59:59 |
| 2024_H2 | 958,974,497 | 27.1 | 2024-06-30 12:00:04 | 2024-12-31 22:00:00 |
| 2025_H1 | 1,236,114,072 | 35.3 | 2024-12-29 13:00:05 | 2025-06-30 23:59:59 |

Schema adds top-of-book to the trades schema:

```
action     'A' add | 'C' cancel | 'M' modify | 'T' trade   (no 'F' or 'R')
bid_px_00 / ask_px_00     double
bid_sz_00 / ask_sz_00     uint32
bid_ct_00 / ask_ct_00     uint32   (order counts at top of book)
flags      130 (= F_LAST|F_TOB) on book updates, 0 on trades
```

**Seam overlaps — measured, not assumed.** Row-group statistics show four
neighbouring files whose ranges touch. Probing each overlap region:

| seam | rows in A | rows in B | byte-identical records |
|---|---:|---:|---:|
| 2021_H1 / 2021_H2 | 951,608 | 7 | 1 |
| 2022_H1 / 2022_H2 | 24,233,206 | 8 | 6 |
| 2023_H2 / 2024_H1 | 7 | 19 | 4 |
| 2024_H2 / 2025_H1 | 27,921 | 0 | 0 |

The duplication is real but tiny — a handful of stray boundary records. Reads
therefore go through **canonical half-open windows** declared in
`configs/paths.yaml`, which make the file set a strict partition of the
timeline. This is cheaper and far more auditable than de-duplicating billions
of rows. Note `2021_H1` legitimately begins 2020-12-27, before its nominal
window, so its canonical window starts 2020-12-01 to retain that data.

### 1.3 Other data on the drive

* `/Volumes/T9/gc/` — 11 files, GC 1-minute continuous bars 2016–2025,
  3.51 M rows, 54 MB. `gc_2024_jan.parquet` duplicates part of
  `gc_2024.parquet`. Not used by this project; GC is a different instrument
  and a different problem.
* `/Volumes/T9/DATA/` — Samsung Magician installer, a NAS-100 downloader
  scaffold, MT5 gold EAs, and a Dukascopy XAUUSD 1-minute CSV/parquet. None
  relevant to NQ futures level research.

---

## 2. Symbols and contracts

Every file contains the whole NQ complex. A representative UTC day
(2024-03-05, 485,770 prints) breaks down as:

| symbol | volume | prints |
|---|---:|---:|
| NQH4 | 785,432 | 479,610 |
| NQM4 | 6,903 | 5,160 |
| NQH4-NQM4 | 2,940 | 880 |
| NQU4 | 68 | 49 |
| NQH4-NQU4 | 43 | 43 |
| NQM4-NQU4 | 30 | 28 |

* **Quarterly outrights only.** `NQ[HMUZ]<year>`. Databento uses single-digit
  year codes, resolved against the session year.
* **Calendar spreads contain `-` and are always excluded.** Their "price" is a
  difference of a few points, not an index level; leaving them in would inject
  fake levels around zero and corrupt every volume profile.
* NQ rolls quarterly. Over 2021-01-05 .. 2026-06-17 the build recorded
  **22 rolls**, exactly 4 per year, all in the expected mid-March / June /
  September / December window.

**Rollover handling.** Three separate decisions, documented in
`src/nqlevels/build/roll.py`:

1. Front month for session *S* = highest-volume quarterly outright on session
   *S-1*. Never *S*'s own volume — that is not known at 06:25. The cost is that
   the roll lands one session after the true volume crossover, which is
   precisely the lag a live run would have. Verified on 2024-03: NQM4 first
   out-traded NQH4 on 2024-03-11, and the front switched on 2024-03-12.
2. L1 is built for **every** outright above 0.5 % of session volume, so both
   price scales exist across a roll (typically 1 symbol, 2 near a roll).
3. Prior-session levels are recomputed from the **incoming contract's own
   prints** when it traded enough that session; otherwise they are shifted by
   a roll spread measured before the decision cut; otherwise dropped. The
   method used is recorded per session in a `carry` column, so roll-adjacent
   sessions can be analysed or excluded as a group.

---

## 3. Aggressor side — verified, not assumed

Databento's `side` needed empirical confirmation, because getting it backwards
would invert every delta and order-flow feature in the project while leaving
everything looking plausible.

Test: 30 minutes of MBP-1 for NQH4 (2024-03-05 15:30–16:00 UTC, the RTH open),
520,545 records including 39,674 trades. Each trade price was compared with the
prevailing top of book.

| `side` | price ≥ prevailing ask | price ≤ prevailing bid | strictly inside |
|---|---:|---:|---:|
| `B` | 19,744 | 0 | 0 |
| `A` | 0 | 19,930 | 0 |

**100 % separation, zero exceptions.** Therefore:

* `side == 'B'` → **buy aggressor** (lifted the offer)
* `side == 'A'` → **sell aggressor** (hit the bid)
* `side == 'N'` → unattributed; counted in `volume`, in neither leg, so
  `buy_vol + sell_vol <= volume` by construction.

Also established from the same sample: the book fields on a trade record show
the **pre-trade** top of book, so they are usable directly as the state the
aggressor traded against.

---

## 4. Timestamps, timezone, sessions

* Both `ts_event` and `ts_recv` are tz-aware UTC nanoseconds. **`ts_event`
  (exchange time) is used everywhere**; `ts_recv` is capture time and would
  introduce a variable feed delay into level construction.
* `ts_recv` is the stored pandas index and is not returned as a column unless
  asked for by name — a real trap when reading these files.
* Session definition is the CME trade date: Globex opens 17:00 CT and that
  evening belongs to the **next** calendar day; close 16:00 CT.
* Trader-facing windows are in **America/Los_Angeles**. PT and CT share US DST
  rules so 06:30 PT is 08:30 CT on every trade date, but both are resolved
  through `zoneinfo` and no UTC offset is hard-coded. Verified: the 06:25 PT
  decision instant is 14:25 UTC in PST and 13:25 UTC in PDT.
* Holiday/early-close calendar (2016–2026) is vendored from the earlier
  `NQ (TOPSTEP)` program, where it was measured from recorded Globex behaviour
  rather than copied from a schedule. 103 holidays, 95 early closes.
* **Halt holidays** (MLK, Presidents' Day, Memorial, Juneteenth, July 4, Labor,
  Thanksgiving) are not trade dates: CME books their thin morning session to
  the *next* trade date. So the Tuesday after MLK has been trading since Sunday
  17:00 CT, and its overnight window starts there — handled explicitly by the
  `globex` / `extended` segments.

---

## 5. Coverage and gaps

A full pass over all 547 M trade prints, bucketed by CME trade date, found:

* **1,411 calendar trade dates with prints, 2021-01-04 → 2026-06-17.**
* **12 missing weekdays — every single one a known full-closure holiday:**
  2021-12-24, 2022-04-15, 2022-12-26, 2023-01-02, 2023-12-25, 2024-01-01,
  2024-03-29, 2024-12-25, 2025-01-01, 2025-04-18, 2025-12-25, 2026-01-01.
* **Zero unexplained gaps.** No corrupt files, no unreadable partitions.

The 1,411 raw dates minus the ~40 halt holidays that book forward gives
**1,371 usable sessions** under the project's session calendar, of which 1,370
were built (the first is consumed as the roll seed).

**Known asymmetry to state in every comparison:** trades cover 2021-01 → 2026-06;
MBP-1 stops 2025-06-30. Trades-only models get 1,370 sessions; microstructure
models get ~1,127. `eval/splits.assert_mbp1_available` raises rather than let a
run silently compare models on unequal data.

---

## 6. Throughput

Measured on the external SSD (exFAT over USB):

* Full-file trades scan, 5 columns: **~2.1 M rows/s** → a complete pass over
  the 547 M-row trade corpus takes about **4 minutes**.
* One session filtered out of the whole trades corpus: **~0.5 s**.
* 30 minutes of single-symbol MBP-1: **6 s** for 520 k rows.

This is the fact that shaped the architecture. Trades are cheap enough to
re-scan; MBP-1 at 7.89 B rows would take roughly **an hour per full pass** and
must be reduced once into compact per-session aggregates.

**Disk budget.** The Mac has **32 GB free**; T9 has **661 GB free**. All
derived data therefore lives on T9, deliberately *outside* `nq data/` so the
raw tree stays pure.

---

## 7. Derived layer built in Phase 1

`/Volumes/T9/nq_levels/` — 1.3 GB total.

| layer | content | rows | size |
|---|---|---:|---:|
| `bars_1s/` | 1-second OHLCV + buy/sell volume + trade count, per session per symbol, monthly partitions | 71,715,726 | 1.2 GB |
| `profiles/` | volume-at-price in 30-minute buckets, keyed by price tick | 11,643,722 | 67 MB |

1,370 sessions, 2021-01-05 → 2026-06-17, 0 skipped, built in **10.1 minutes**.
Mean 52,347 populated seconds per session (63 % of the 23-hour session).
Front-contract rows: 67,929,008.

Yearly front-contract price ranges, continuous and plausible throughout:

| year | low | high | sessions |
|---|---:|---:|---:|
| 2021 | 12,208.75 | 16,766.25 | 251 |
| 2022 | 10,486.75 | 16,564.00 | 251 |
| 2023 | 10,751.50 | 17,165.00 | 250 |
| 2024 | 16,336.00 | 22,387.75 | 252 |
| 2025 | 16,467.75 | 26,398.50 | 251 |
| 2026 H1 | 22,965.25 | 30,974.00 | 115 |

### Validation

`scripts/sanity_l1.py` — **22/22 checks pass** on every period tested
(2021-01, 2021-03, 2022-06, 2022-11, 2023-12, 2024-03, 2024-11, 2025-06,
2026-03, 2026-06), covering price scale, OHLC ordering, session bounds,
trade-date attribution, DST-sensitive window slicing, decision-cut containment,
and profile↔bar volume identity.

The strongest of these is **cross-validation against an independent build**:
aggregating the new 1-second bars to 1 minute and comparing with
`nq_bars_full.parquet` from the earlier `NQ (TOPSTEP)` program gives
**max OHLC difference 0.000000 and 100.00 % identical volume** across every
matched minute, with no minute present in one build and absent from the other.
Two independently written pipelines agreeing exactly is much stronger evidence
than either passing its own checks.

`scripts/audit_leakage.py` — **7/7 audits pass**, including truncation
invariance: 1,603 candidate zones across 8 sessions were byte-identical whether
generated from the complete session tape or from a tape physically cut at
06:25 PT.

---

## 8. Macro-event data — feasibility checked

| source | status |
|---|---|
| BLS API v2 | **works, no key required** — returns CPI/PPI/NFP series directly |
| FRED / ALFRED | reachable; needs a free API key (returns HTTP 400 without one). ALFRED vintages are the correct source for point-in-time actuals |
| federalreserve.gov FOMC calendar | reachable (HTTP 200) |
| Forex Factory | **blocked, HTTP 403** — confirms it is not a usable programmatic source |

A **tape-endogenous** alternative was tested and works well. Measuring
one-minute volume as a multiple of the preceding 15-minute baseline across 502
sessions (2024–2025):

| PT slot | median | p95 | max |
|---|---:|---:|---:|
| 05:30 (08:30 ET — main US macro slot) | 2.18× | 23.5× | 49.2× |
| 06:30 (cash open) | 15.3× | 29.1× | 49.7× |
| 07:00 (10:00 ET — ISM/JOLTS slot) | 1.26× | 2.76× | 5.26× |
| 05:00 / 06:00 / 08:00 (controls) | 1.2–1.9× | 2.4–4.1× | 8–32× |

The 05:30 PT spike distribution reproduces the US release calendar without any
external data: the top-20 % spike sessions are 36.6 % Thursday (weekly jobless
claims), 35.6 % Friday, and **1.0 % Monday** — Mondays have almost no 08:30 ET
releases. Median one-minute range on spike days is **79.8 points versus 9.5**
otherwise.

Two consequences:

1. **The main macro release lands at 05:30 PT, 55 minutes before the decision
   cut.** Both the release and the market's reaction to it are fully observable
   pre-cut, so an endogenous "surprise magnitude" feature carries no leakage
   risk and needs no external feed.
2. An external calendar is still worth adding for what the tape cannot say:
   *which* event it was, and whether one is scheduled **inside** the 06:30–09:00
   window (10:00 ET = 07:00 PT). That is a genuine forward-looking feature and
   the only part that requires a schedule.

---

## 9. Corrections to the prior handoff

| handoff said | disk says |
|---|---|
| Trades 2021_H1 … 2026_H1 | correct — 11 files |
| MBP-1 2021_H1 … 2025_H1 | correct — 9 files, ends 2025-06-30 |
| MBP-1 ≈ 222 GB, 7.9 B rows | 207 GB, 7.893 B rows |
| `/Volumes/T9/nq data/` has a curated layer | **no longer exists** — only `raw/` remains |
| project root `~/Desktop/topstep-lab` | renamed to `~/Desktop/NQ (TOPSTEP)` |

No other discrepancies found.

---

## 10. Open data risks

1. **MBP-1 ends 2025-06-30.** Any microstructure confirmation research is
   limited to ~1,127 sessions and cannot be validated on the 2026 holdout.
2. **The 2026 holdout is 110 sessions** after embargo — enough for a decisive
   single test of one or two pre-registered rules, not for a sweep.
3. **2026-H2 CME holidays in the vendored calendar are pattern-derived**, not
   read off an official notice (cmegroup.com blocks non-browser fetches).
   Marked `VERIFY` in `configs/cme_calendar.yaml`. Only affects dates after the
   current data ends, so it cannot corrupt any current result.
4. **Top-of-book size on NQ is very thin** — mean 3.15 contracts on the bid,
   3.15 on the ask in the RTH sample. MBP-1 book *size* is likely to carry less
   information than book *events* (adds, cancels, replenishment rate); the
   microstructure phase should not assume displayed size means much.
5. **exFAT on the external drive** has no checksums. The corpus was fully read
   once during this audit with no errors, which is reasonable assurance, but
   silent bit-rot would not be detected automatically.
