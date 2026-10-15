# Experiment Log

Append-only. Every entry — kept or rejected — gets recorded so the same ground
is not re-covered. A rejected concept is a fact about that concept, never about
daily levels in general.

Template:

```
## <ID> — <title>                                    [KEPT | REJECTED | OPEN]
date · hypothesis · exact rule/feature · parameters · data period ·
sample size (rows AND distinct sessions) · baseline · train · validation ·
OOS · parameter sensitivity · verdict · reason
```

---

## P1-000 — Phase 1 foundation                                        [KEPT]

**2026-08-14.** Not a hypothesis test — the audit and build everything else
depends on.

**Built.** Project at `~/Desktop/nq-levels`; session clock (PT trader windows
over the CME CT calendar, DST via `zoneinfo`); read-only raw access with
seam-deduplicated MBP-1 windows; causal front-month roll; L1 = 1-second bars +
30-minute volume-at-price profiles; L2 daily context; named + price-axis
candidate generators; zone-width model family; touch and reaction label
machinery; chronological splits with a code-sealed holdout; leakage audits.
69 tests.

**Data.** Trades 547,020,258 rows over 2021-01-04 → 2026-06-17; MBP-1
7,893,313,081 rows ending 2025-06-30. 1,370 sessions built in 10.1 min,
0 skipped. 12 missing weekdays, all known full-closure holidays — no
unexplained gaps. 22 rolls, all decided from strictly prior sessions.

**Validation.** L1 sanity gate 22/22 on ten sampled periods spanning
2021–2026, including exact agreement with the independently built
`nq_bars_full.parquet` (max OHLC difference 0.000000, 100 % identical volume).
Leakage audit 7/7, including truncation invariance — 1,603 candidate zones
identical whether built from the full session tape or one physically cut at
06:25 PT.

**Verdict.** KEPT. Foundation is trustworthy enough to build research on.

---

## P1-001 — Databento aggressor convention                            [KEPT]

**Hypothesis.** `side` encodes the aggressor, but the direction needed
confirming — inverting it would flip every order-flow feature while leaving
everything looking plausible.

**Method.** 30 min of MBP-1 for NQH4 (2024-03-05 15:30–16:00 UTC), 39,674 trade
records; each trade price compared with the prevailing top of book.

**Result.** `side == 'B'` matched *price ≥ prevailing ask* 19,744/19,744;
`side == 'A'` matched *price ≤ prevailing bid* 19,930/19,930; zero records
strictly inside. **100 % separation.**

**Verdict.** KEPT. `'B'` = buy aggressor, `'A'` = sell aggressor, `'N'`
unattributed. Encoded in `io/raw.py` and `build/tape.py`. Also established: the
book fields on a trade record show the pre-trade top of book.

---

## P1-002 — MBP-1 half-year seam duplication                          [KEPT]

**Hypothesis.** Row-group statistics showed four neighbouring MBP-1 files whose
time ranges touch; a naive union might double-count.

**Result.** Overlaps are real but tiny: 1, 6, 4 and 0 byte-identical records at
the four seams. `2021_H1` also legitimately starts 2020-12-27, before its
nominal window.

**Verdict.** KEPT as a design constraint. Reads go through canonical half-open
windows in `configs/paths.yaml`, making the file set a strict partition —
cheaper and more auditable than de-duplicating 7.89 B rows. Trades files do not
overlap and need no such treatment.

---

## P1-003 — Touch probability is monotone in distance                 [KEPT]

**Hypothesis (structural, not statistical).** Since reaching a price requires
passing every price in between, `P(touch)` above the open is the survival
function of the session's maximum excursion — so raw touch rate mostly measures
proximity, not level quality.

**Evidence.** Machinery demo, 4 sessions, 661 candidate zones (far too small to
be a result, but the gradient is unmistakable):

| distance | touch rate | n |
|---|---:|---:|
| 0 – 0.25 ATR | 92.0 % | 138 |
| 0.25 – 0.5 ATR | 62.4 % | 101 |
| 0.5 – 1.0 ATR | 35.8 % | 187 |
| 1.0 – 2.0 ATR | 2.2 % | 179 |
| > 2.0 ATR | 0.0 % | 55 |

**Verdict.** KEPT, and it determines the evaluation design: every destination
claim is scored as **attraction** — touch minus a distance- and regime-matched
baseline fitted on training sessions only
(`levels.axis.baseline_touch_curve`). A raw touch-probability number is not
reportable on its own.

---

## P1-004 — Reaction must be measured from zone edges                 [KEPT]

**Hypothesis.** Anchoring both excursions at the touch reference is the obvious
implementation and is wrong.

**Evidence.** With adverse movement anchored at the near edge, a fast touch
that prints through the zone registers ≥5 points of "adverse" move at the touch
second itself, so the zone is scored as failed before it has done anything. On
the demo window this produced a 12.4 % clean-rejection rate that was an artifact
of zone width, not of level behaviour.

**Fix.** Rejection = distance back past the **near** edge; continuation =
distance past the **far** edge; both zero while inside. Movement inside a zone
is neither reaction nor failure — by construction the zone is the band where
price counts as "at the level". Same window, same zones: 53.8 %.

**Verdict.** KEPT. Neither number is a result; the point is that the definition
change moved it by a factor of four, so the definition is load-bearing and is
now pinned by unit tests.

---

## P1-005 — Directional efficiency at 1-second resolution            [REJECTED]

**Hypothesis.** |net displacement| / path length over 1-second closes separates
tap-and-go from chop.

**Result.** It does not. Over minutes of 1-second bars the path is dominated by
bid-ask jitter, so the ratio collapses toward zero for everything: the demo
showed 0.00–0.03 for clean impulses *and* for pure chop. A unit test then
exposed the opposite failure — sampling a 40-second window every 30 s leaves
two points, and any two points are a straight line, scoring a perfect 1.0.

**Fix.** Sample the path at ~30 s, shrinking the step so at least 8 samples
exist; `NaN` below that. Cleanliness metrics also moved to a **fixed** 900-second
window so a 06:35 touch and an 08:55 touch are comparable. Post-fix the metric
separates the two shapes (demo range 0.04–0.44).

**Verdict.** REJECTED as originally specified, KEPT in the sampled form.

---

## P1-006 — Macro releases are detectable from the tape alone         [KEPT]

**Hypothesis.** A scheduled high-impact US release can be identified from NQ
prints without any external calendar.

**Method.** 502 sessions (2024-01 → 2025-12). One-minute volume at fixed PT
times, as a multiple of the preceding 15-minute per-minute baseline.

**Result.**

| PT slot | median | p95 | max |
|---|---:|---:|---:|
| 05:30 (08:30 ET) | 2.18× | 23.5× | 49.2× |
| 06:30 (cash open) | 15.3× | 29.1× | 49.7× |
| 07:00 (10:00 ET) | 1.26× | 2.76× | 5.26× |
| 05:00 / 06:00 / 08:00 (controls) | 1.2–1.9× | 2.4–4.1× | 8–32× |

The 05:30 spike distribution reproduces the US release calendar unaided —
top-20 % spike sessions are 36.6 % Thursday (weekly jobless claims), 35.6 %
Friday, **1.0 % Monday**. Median one-minute range on spike days is
**79.8 points vs 9.5** otherwise.

**Verdict.** KEPT. The main macro slot is **55 minutes before the decision
cut**, so release *and* market reaction are observable pre-cut with no leakage
risk and no feed. An external calendar is still needed for event *identity* and
for releases scheduled **inside** the 06:30–09:00 window (07:00 PT).

---

## P1-007 — Macro data source feasibility                             [KEPT]

**2026-08-14 reachability check.** BLS API v2 — **works, no key**. FRED/ALFRED
— reachable, needs a free key; ALFRED vintages are the correct source for
point-in-time actuals. federalreserve.gov FOMC calendar — HTTP 200.
**Forex Factory — HTTP 403, blocked**, confirming the brief's suspicion.

**Verdict.** KEPT as the plan of record. Schedule is always a legal pre-open
feature; actual/surprise only after the release stamp; vintage data only, never
revised values.

---

---
---

# PHASE 2

Scope correction applied 2026-08-14: this is a **levels** project. No entries,
stops, targets, sizing, P&L or strategy testing. Reaction metrics are
measurements of a price area, not trades.

---

## P2-100 — L2/L3 infrastructure                                       [KEPT]

Built on top of Phase 1's L1: `context.parquet` (1,370 sessions × 99 pre-open
columns), `excursions.parquet` (533 k session-minutes: running max up/down
excursion per RTH minute), `paths.parquet` (12.2 M rows: 1-second high/low/close
for 06:30–09:00 PT, as offsets from the 06:25 reference),
`bars_1m.parquet` (2.18 M rows), `candidates_named.parquet` (305 626 level
clusters, 223/session, 86 kinds), `axis.parquet` (1.08 M grid prices,
802/session, 59 causal features).

Two decisions worth keeping:

* **Level construction runs on 1-minute bars, not 1-second.** Highs, lows,
  volumes and VWAPs aggregate exactly, so the prices are identical and the
  build is ~50× cheaper. Only *reaction* measurement needs the fine path.
* **Touch needs no path at all.** Reaching a price means passing every price
  below it, so the running-maximum excursion answers touch and time-to-touch at
  any width by binary search. Measurement runs at ~32 000 zones/s, which is
  what makes a full-corpus width sweep a one-minute job.

Bug found and fixed: `io/store` globbed macOS AppleDouble sidecars (`._*.parquet`)
on the exFAT drive and handed them to pyarrow. Now filtered at every read.

---

## P2-101 — E1 distance/excursion baseline                             [KEPT]

**Hypothesis.** Touch probability at distance *d* is the survival function of
the session's running excursion, so the null model is a conditional
distribution of excursions rather than a per-price classifier.

**Model.** `log R` (target-window range around the reference) on pre-open
context by ridge; `z = excursion / scale` fitted as an empirical survival
function per side and per horizon.

**Feature ranking (train, correlation with log R):** overnight realised vol
**0.622**, prior-day realised vol 0.562, overnight range 0.540, ATR(5) 0.526,
ATR(14) 0.474, premarket range 0.463. **Realised volatility beats ATR.**

**Validation (369 sessions, 5 904 session-distance observations):**

| variant | Brier | skill | log loss | ECE |
|---|---:|---:|---:|---:|
| constant scale | 0.1412 | +0.336 | 0.4420 | 0.0624 |
| ATR(14) only | 0.1348 | +0.366 | 0.4152 | 0.0264 |
| overnight range only | 0.1323 | +0.378 | 0.4069 | 0.0313 |
| ATR + overnight range | 0.1311 | +0.384 | 0.4020 | 0.0180 |
| 11-feature full | 0.1292 | +0.392 | 0.3957 | 0.0212 |
| **4-feature (frozen)** | **0.1291** | **+0.393** | **0.3935** | **0.0144** |

Frozen: `log_rv_on, log_prior_rv, log_on_range, log_atr14`, ridge α=1,
monthly expanding refit. The 11-feature set scored identically with worse
calibration.

**Stability.** Full-history walk-forward Brier skill by year: 2022 +0.419,
2023 +0.436, 2024 +0.392, 2025 +0.400; all-years ECE 0.0094. Normalised
excursion medians 0.39–0.47 (up) and 0.42–0.54 (down) across five years — the
distribution is approximately stationary.

**Sensitivity.** Flat in ridge α over 0.3–10 and in embargo over 1–30 days
(skill changes < 0.001). A plateau, not a peak.

**Rejected refinements.** (a) Regime-conditional *shape* (gap bucket, overnight
range bucket): no improvement, 0.1293 vs 0.1292. (b) Walk-forward refit: +0.0007
skill — the model is stable enough that a frozen fit is fine, though WF is used
anyway for realism and slightly better ECE.

**Known limitation, documented not fixed.** Far *downside* touches are
under-predicted by ~3 pp in 2024-25 (normalised down-tail p90 1.04–1.17 in
2022-23 vs 1.45–1.49 in 2024-25). A directional share model was fitted and
**rejected**: its coefficients are all within ±0.024 of an intercept of 0.497,
i.e. pre-open context barely predicts which side of the reference the range
falls on. This is a genuine out-of-sample regime shift. **Mitigation, mandatory:**
every level comparison is made at the same side and matched distance, so the
miscalibration hits level and control equally and cancels.

**Verdict.** KEPT and frozen as the null model for all destination claims.

---

## P2-102 — Named-level density: the vacuity check              [MAJOR FINDING]

**Hypothesis.** Before any level statistic means anything, "this price is at a
level" must be a *selective* statement.

**Result (train, within ±1 ATR of the market):**

| universe | kinds | levels/session | median gap for a random price | within 1.5 pt | within 5 pt |
|---|---:|---:|---:|---:|---:|
| all generated | 86 | 211.6 | **1.29 pts** | 54.9 % | 87.1 % |
| dense families | 12 | 144.5 | 1.82 pts | 43.9 % | 80.1 % |
| **classical core** | **26** | **31.7** | **8.36 pts** | **14.5 %** | **36.7 %** |

**Interpretation.** Pooled over all generated kinds the concept is vacuous — a
uniformly random price is within 1.3 points of *some* named level. Twelve kinds
cause it (`round_25/50`, HVN/LVN over four profile windows, swing highs/lows):
144 of the 211. Worse, it silently destroys the obvious experiment: a control
drawn by shifting a level lands on another level, so the contrast is null **by
construction rather than by evidence**.

**Consequences, both acted on.**
1. The analysis universe is the sparse 26-kind classical core
   (`eval/universe.CORE_KINDS`); dense families are studied separately and are
   never used to define "at a level".
2. The research question is not "is this price a level?" — most prices are —
   but "**which few of the many candidate prices are the strong ones?**" That is
   a ranking problem, and it argues for the algorithmic price-axis approach over
   family membership.

---

## P2-103 — Why a matched-distance control cannot be drawn      [METHOD FINDING]

First attempt at E2/E3 compared each level against a control shifted ±0.35 ATR
at the same side. On the core universe this gave core-level clean rates
*below* the control at every width (e.g. 0.261 vs 0.291 at 4-point half-width,
CI excluding zero).

**That comparison is invalid.** Controls had touch rate 0.291 versus 0.483 for
real levels — i.e. they sit systematically *farther* from the market, and
distance drives both reachability and post-arrival behaviour. At a given
distance and side there is exactly **one** price, so a distance-matched control
is a logical impossibility.

**Fix, and the design used from here on:** a uniform 1-point **price-axis grid**
over ±1.5 ATR (802 prices/session, 1.08 M rows), each price marked as at/not-at
a core level, compared **within distance-and-side strata** with cell weights and
a session-level block bootstrap (`quality.stratified_contrast`). Distance is
held fixed by construction instead of adjusted for afterwards.

The earlier attraction result is unaffected — attraction is already a residual
against the distance-matched baseline — and it stands: core levels +0.0033,
controls +0.0061, i.e. **no destination edge either way**.

---

## P2-104 — E2 zone width                                     [KEPT, with caveat]

**Design.** Uniform 1-point price-axis grid (802 prices/session), each marked
at/not-at a core level, compared **within distance-and-side strata** with
session-clustered bootstrap. Widths swept from 0.5 to 20 points half-width
across fixed-tick, ATR-fraction, overnight-range-fraction and realised-vol
families.

**Result — width is not the binding constraint, because there is nothing for it
to reveal.** Across every width, the level-vs-non-level contrast is inside the
noise floor:

| width | half-width | d_attraction | d_clean | 95 % CI |
|---|---:|---:|---:|---|
| fixed 4t | 1.0 pt | +0.0010 | −0.0015 | [−0.007, +0.004] |
| fixed 8t | 2.0 pt | +0.0012 | +0.0018 | [−0.005, +0.009] |
| fixed 20t | 5.0 pt | +0.0020 | +0.0085 | [−0.001, +0.020] |
| atr 0.010 | 2.4 pt | +0.0011 | −0.0072 | [−0.014, −0.000] |
| atr 0.020 | 4.8 pt | +0.0019 | −0.0097 | [−0.018, +0.001] |
| atr 0.035 | 8.5 pt | +0.0026 | −0.0098 | [−0.021, +0.001] |

**The sign flips between fixed and ATR-scaled widths of the same median size**
(fixed 20t = 5.0 pt gives +0.0085; atr 0.020 = 4.8 pt gives −0.0097). A real
effect does not do that. Regime homogeneity confirms it: d_clean +0.006 / +0.002
/ +0.001 across low/mid/high volatility terciles, every CI straddling zero.

**Noise floor established.** A synthetic control where only *session-level*
heterogeneity exists yields a spurious contrast of ≈ 0.003, so **|d| < 0.005 is
not interpretable**. The estimator recovers a planted +0.030 as +0.0315
[+0.0286, +0.0345] and returns −0.0014 on a null.

**Verdict.** No width choice reveals an edge. Width is therefore chosen on
non-statistical grounds — **6 ticks (1.5 points half-width, 3-point zone)** —
for chart usability, and the measured layer uses it. The choice does not affect
any conclusion, which is itself the finding.

---

## P2-105 — E3 named level families                              [REJECTED]

26 classical core kinds, each with 780-843 sessions and 2 400-17 700 grid
observations, tested against matched-distance non-level prices.

**No family survives FDR < 0.10 on either axis.** Best reaction contrast:
`pd_vwap` d_clean +0.053 [+0.004, +0.101], raw p ≈ 0.017, **FDR 0.166**. Best
destination: `pd_vah` d_att +0.026, FDR 0.468.

Several families are *negative*: `pd_globex_low` d_clean −0.055,
`pre_low` −0.040 [−0.070, −0.011], `round_100` −0.016 [−0.038, +0.006].

The headline families do nothing: `pdh` d_clean −0.010, `pdl` −0.015,
`on_high` −0.005, `on_low` −0.002, `on_poc` +0.023 (FDR 0.26).

**Dense families**: `near_dense` (39.5 % of the axis) d_clean **+0.0007** —
literally nothing.

## P2-106 — E3b higher-timeframe levels           [REJECTED, underpowered]

Prior week/month high-low-close and 5/10/20/60-session extrema were **never
generated** by the first candidate build (`build_named` was called without
week/month bars) — a real omission, fixed by testing them directly against the
measured axis.

Coverage is 0.1-0.4 % of the grid, so n is 341-1 966 and intervals are very
wide. Direction is mildly positive for prior-week high (`pwh` d_clean +0.064
train, +0.070 validation) and prior-week close (+0.039 / +0.075), but no CI
excludes zero and FDR across 13 kinds would remove anything. **Underpowered
rather than refuted** — the one family worth revisiting with more data.

---

## P2-107 — E5 does ANY feature predict zone quality?            [REJECTED]

**ICC of `clean` = 0.020** — 98 % of the variance is *within* session. Zone
quality is a price-level phenomenon, not a "clean-reacting morning" phenomenon.

Pooled validation AUC: distance 0.536, location 0.551, context 0.554,
location+context 0.568, HGB 0.566. Shuffled-label control 0.493 and
session-shuffled context control 0.494 — the pipeline reports nothing when
there is nothing.

**But pooled AUC is contaminated, and the contamination is severe.**

---

## P2-108 — The extreme-price tautology                    [METHOD FINDING]

Ranking prices **among those that were touched** smuggles in the answer. The
farthest price price reached during the window *is* the extreme of the day's
move, so by construction price reversed there.

Measured on validation: selecting "the farthest touched price" each morning
gives a **88.6 % clean-rejection rate** against a 10.5 % base (farthest touched
above: 95.3 %; below: 92.0 %; nearest touched: 4.9 %; random touched: 11.6 %).

This is not a level effect, it is the definition of an extreme, and it is only
knowable after the session. Any ranker correlated with distance inherits it —
which is why "distance only" appeared to have the best within-session AUC
(0.627) in the first pass.

**Correct evaluations, used from here on:**
1. **Within-distance-band AUC** — hold the contaminated axis fixed.
2. **Pre-open selection** — rank ALL grid prices, take top K, *then* look.

---

## P2-109 — E5b honest selection                                 [REJECTED]

**Within-distance-band AUC (validation, weighted):** model score 0.556,
distance 0.506, `on_volrel` 0.497, `near_core` **0.499**.

**Pre-open top-K selection (validation, 368 sessions):**

| K | rule | touch | clean given touch | median dist |
|---|---|---:|---:|---:|
| 3 | model | 0.104 | 0.157 | 1.06 ATR |
| 3 | random | 0.310 | 0.097 | 0.61 ATR |
| 3 | nearest | 0.935 | 0.052 | 0.01 ATR |
| 3 | core levels | 0.488 | 0.104 | 0.33 ATR |

The model's apparent advantage is bought entirely by picking far zones (touched
10 % of the time) — the tautology again.

**Distance-matched by construction** (control drawn at the same distance band,
same side, same morning): K=2 **+0.013**, K=3 **+0.014**, K=4 **−0.006**, on
78-150 touched zones. **The edge collapses to zero and flips sign.**

---

## P2-110 — E4 first touch versus repeat touch          [KEPT — real, but not a level effect]

Touch sequences measured with hysteresis (price must clear an edge by a tick
before a new touch counts), 59 807 records over 15 472 core zones.

**Pooled** (confounded): touch 1 clean 0.427, touch 2 0.379, touch 3 0.360,
touch 4 0.355; penetration deepens 11.7 → 19.0 ticks.

**Within-zone paired** (each zone its own control, removing the selection that
choppy zones get more touches): touch 1 **0.4216** vs touch 2 **0.3786**,
**diff +0.0430, session-clustered 95 % CI [+0.0371, +0.0490], p>0 = 1.000**.
Replicates on both splits: train +0.0349, validation +0.0608.

**But it is universal, not a level property.** Sampling non-level grid prices
and repeating the paired test:

| price type | touch 1 | touch 2 | decay |
|---|---:|---:|---:|
| at a core level | 0.4268 | 0.3778 | **+0.049** |
| not a level | 0.4555 | 0.3903 | **+0.065** |

Non-level prices show a *larger* decay and a *higher* first-touch rate. So the
finding is real and robust — **the first time price reaches any given area
today, the reaction is meaningfully better than subsequent times** — but it says
nothing about which prices to select.

Also visible: a strong selection gradient. First-touch clean rate is 0.662 for
zones touched once, 0.595 twice, 0.490 three times, 0.416 four or more. Choppy
areas react worse *even on their first touch*.

---

## P2-111 — Regime conditioning                                   [REJECTED]

Does `near_core` work on any particular kind of morning? Gap direction
(3 buckets), overnight range vs ATR (3), volatility percentile (3), macro shock
from the tape (3). **Twelve regimes, every d_clean in [−0.007, +0.007], every
CI straddling zero.** No regime rescues named levels.

---

## P2-112 — Target sensitivity                                    [REJECTED]

Ten reaction definitions, to check the verdict is not an artefact of one
arbitrary threshold: `clean` at 10/5, 15/10, 20/10 and 30/15 points; rejection
exceeding penetration; rejection > 0.25 and > 0.40 ATR; not slicing through
within 60 s; not accepting beyond the far edge; fast rejection within 5 minutes.

**Train**: all diffs in [−0.015, +0.003]. **Validation**: all in [−0.021,
+0.012]. The only intervals excluding zero are *negative* — `y_rev25atr`
−0.021 [−0.040, −0.001] and `y_rev40atr` −0.020 [−0.034, −0.008].

Under no reasonable definition of reaction do named levels beat matched
non-level prices, and under several they are worse.

---

## P2-113 — E11 pre-open MBP-1 order-book history                 [REJECTED]

Strictly the pre-open question the brief allows: does resting-liquidity history
left at a price during the night mark it? 100 sampled sessions (2021-02 ..
2025-06, within MBP-1 coverage), London + premarket window, front contract,
~1 M book events per session. Per-price features: time-weighted resting size,
seconds at the top of book, add and cancel counts, replenishment, traded volume,
and an absorption ratio.

**Within-distance-band AUC (validation):**

| feature | AUC touch | AUC clean |
|---|---:|---:|
| bk_secs | 0.562 | 0.485 |
| bk_twsize | 0.568 | 0.477 |
| bk_adds | 0.565 | 0.482 |
| bk_cancels | 0.564 | 0.483 |
| bk_tradevol | 0.565 | 0.482 |
| bk_replenish | 0.563 | 0.483 |
| bk_absorb | 0.571 | **0.453** |
| *[ref]* on_volrel | — | 0.496 |
| *[ref]* dist_atr | — | 0.533 |

**Reaction: nothing, and if anything the wrong sign** — every feature is below
0.500. The apparent *touch* signal (0.56) is residual distance structure inside
the bands, not location information: the reference row shows plain `dist_atr`
still scores 0.533 within the same bands, so the bands do not fully absorb
distance.

Only 21 % of grid prices had any top-of-book time at all, consistent with the
Phase-1 observation that NQ's top of book averages ~3 contracts.

**Verdict.** REJECTED, as the brief instructs when MBP-1 adds little. Do not
build order-book features into the system. The 207 GB dataset earns nothing
here, and *traded* volume at price — a strictly richer signal — already
scored 0.496.

---

## P2-114 — E13 what the data does support                            [KEPT]

**Arrival time is confounded, not informative.** Penetration falls monotonically
with arrival minute (0.336 → 0.064 ATR) and slice-through falls (0.851 →
0.726), but a later arrival has less remaining window by construction. Nothing
clean to extract.

**The E1 destination curve is a genuine, well-calibrated deliverable.** Solving
the fitted survival function for the distance at a target probability and
checking the realised rate on validation, across 3 horizons × 2 sides × 5
probabilities (30 combinations, 369 sessions each):

**mean |realised − target| = 0.0251, max 0.0737.** The down side runs 3-5 pp
hot, exactly the documented 2024-25 regime shift.

Worked example (2025-12-23, reference 25,631.25, overnight range 130 pts,
ATR 307 pts): P(touch by 09:00) = 0.75 at ±30-34 pts, 0.50 at ±64-70 pts,
0.25 at ±114-120 pts.

That is a distance-and-volatility product. It answers "how far is price likely
to travel today" accurately. It does **not** identify strong levels, and
nothing in Phase 2 supports labelling any price as one.

---

# PHASE 2 VERDICT

**The central hypothesis is rejected.** Across 38 named level kinds, a 1.08 M
price-axis grid, 10 reaction definitions, 12 regimes, 9 zone widths, learned
rankers, and pre-open order-book history, **no method identifies which NQ
prices will behave as strong zones**, once distance and side are held fixed.

Everything that appeared to work was traced to a specific confound:
* raw touch rate → distance (P2-103, and E1 by construction)
* apparent reaction edge of far zones → the extreme-price tautology (P2-108)
* apparent selection lift → conditioning on `touched` (P2-109)
* apparent first-touch decay as a level property → universal to all prices (P2-110)

**Survives:**
1. **E1 destination baseline** — calibrated P(touch | distance, volatility),
   validation Brier skill +0.39, ECE 0.014, stable 2021-2025, mean calibration
   error 0.025 in product form.
2. **First-touch decay** — +4.3 pp within-zone [+0.037, +0.049], replicated on
   validation, but a property of *any* price, not of levels.

**Holdout: never read.** 110 sessions remain sealed. Nothing qualified for a
confirmation test, so nothing was spent — which is the correct outcome.

---

**Do not** read the holdout. Sealed in code; 110 sessions is one shot.

---
---

# PHASE 3 — conditional reaction ranking

Opened 2026-08-14. The question changes from "is family X universally better
than matched non-level prices?" (answered: no) to the product question: "rank
each morning's candidates and select the top 2–4 — do the *selected* zones
beat realistic same-morning alternatives?" Design in RESEARCH_PLAN §12.

Rules of engagement, unchanged: explore on train; one logged look at
validation per family; the three Phase-2 confounds controlled in every design;
holdout stays sealed.

---

## P3-200 — Top-K daily selection framework                            [KEPT]

**2026-08-14.** `eval/topk.py` + 7 tests. Per-morning top-K selection with
min-separation de-dup and optional distance window; controls: distance-matched
same-morning (headline), random same-morning, next-ranked, nearest-K; metrics:
touch, P(clean|touch), joint P(touch∧clean), P(≥1)/P(≥2) reacted per morning,
magnitude/penetration/slice; paired session-clustered bootstrap CIs; per-year
splits; calibration table. Estimator validated on synthetic mornings: recovers
a planted +0.25 conditional lift as +0.12..+0.25 with CI > 0 while holding the
touch-rate contrast at zero (no distance smuggling); returns CIs straddling
zero on a null score. All Phase-3 selection claims report through this.

---

## P3-201 — Rejection-event layer                                      [KEPT]

**2026-08-14.** `features/events.py` + `build_events.py` (+7 tests). Event =
local extreme of a ±30-min window whose excursion away reaches ≥ 0.25 ATR
within 120 min, truncated the moment the extreme is exceeded (a later collapse
is not a reaction to a taken-out level). Detector is a pure function, so
same-morning events are recomputed from pre-cut bars — truncation-invariant by
construction. 13,080 events / 1,234 train+validation sessions (median 10 per
session, 48.3 % high-side); magnitude distribution stable 2021-2025 (median
0.39 ATR every year). Cross-roll mapping via per-roll spreads measured from
bars-overlap medians: 22/22 rolls measured, 0 broken chains
(`roll_adjust.parquet`).

---

## P3-202 — Reaction memory                          [REJECTED — sign reversed]

**2026-08-14, train only (843 sessions, 470 k grid rows).** Hypothesis: prices
with strong recent rejection history react better when touched. Features per
grid price: decayed event count/magnitude/velocity at 0.02 and 0.04 ATR bands,
direction-matched and opposed, 60-session lookback, half-life 20; plus
same-morning overnight events from pre-cut bars.

**Result: the opposite, mildly but robustly.** Within side×distance strata:
`mem_strong` (top decile) clean −0.0152 [−0.0249, −0.0039], rev30m −0.021 ATR,
slice-through +0.0175 [+0.003, +0.033]. Survives additionally holding
lb20 visit-intensity fixed (−0.0156). Within-band AUCs all ≈ 0.49. Shape: flat
in event count (n=1 → −0.005, n≥4 → −0.006); driven by recency (last ≤5
sessions −0.0094 [−0.018, −0.001]; >20 sessions +0.003 n.s.). Same-morning
event areas (`mem0_any`, 5.7 % of grid): clean −0.0145 but rev30m +0.026,
slice +0.040, attraction +0.021..+0.036 — **higher-energy, not
higher-quality** areas.

**Verdict.** REJECTED as stated; no validation look spent. Two usable facts:
(1) *recently-active reaction areas are consumed, not strengthened* — a
freshness/penalty signal for any later ranker; (2) overnight-event areas
attract touches and movement without cleanliness — an energy marker. Both may
only be deployed through the top-k framework, never as standalone claims.

---

## P3-203 — Displacement origins (supply/demand)     [REJECTED for cleanliness;
                                                      energy structure KEPT]

**2026-08-14, train only.** `features/impulses.py` (+5 tests) — zigzag legs at
0.35 ATR reversal, transit-timed (departure = last touch of origin extreme,
arrival = first touch of end), origin zone = pivot + pre-departure base capped
at 0.06 ATR; `build_origins.py` tracks every origin to each later morning:
age, mitigation depth from later session extremes + today's overnight
(carry-adjusted), freshness, intact/broken side, same-morning origins from
pre-cut bars. 6,780 legs (median 4/session); zone-coverage marking spans
287k train grid rows (fresh subset 21,880).

**Cleanliness: nothing, again.** At zone centres only (n=1.5k) fresh origins
showed clean +0.060 — with full zone coverage (n=21.9k) it collapses to
+0.0045 [−0.016, +0.024]. Small-sample mirage, the framework caught it.
Within-origin conditionals all null: displacement magnitude −0.007, velocity
−0.004, volume +0.007, intact +0.003, age +0.000. The S/D strength playbook
does not rank NQ origins.

**What is real (CIs excluding zero) is an energy structure, not a quality one:**
* fresh never-revisited origins are LOW-energy: slice −0.044 [−0.072, −0.019],
  rev30m −0.031 [−0.048, −0.014], attraction −0.010 — price under-reaches and
  under-moves around unvisited displacement scars;
* same-morning fresh origins are HIGH-energy: rev30m +0.057 [+0.024, +0.093],
  slice +0.041, attraction +0.040 — same signature as P3-202's overnight
  event areas; movement both ways, no ordering edge (clean −0.007 n.s.);
* consumed/broken origins: mildly negative clean (−0.006), slice +0.019 —
  consistent with the P3-202 consumed-structure penalty.

**Converging conclusion across P2 + P3-202/203: location features predict
movement ENERGY around an area (attraction, excursion, slice-through), and do
not predict directional rejection QUALITY at matched distance.** Same-morning
activity (events, origins) is the strongest energy marker found so far.

---

## P3-204 — Conditional named levels                                   [OPEN]

**2026-08-14, train.** "When is a level good", within-family, at matched side
and distance. Registered conditions: C1 untested overnight; C2 formed by a
direction-matched rejection event yesterday; C3 overnight extreme lightly vs
heavily tested; C4 100-round confluence.

**C2 (−0.012 pooled), C3 (underpowered), C4 (−0.021): nothing.** Confluence
and formed-by-rejection do not help — consistent with the P3-202 consumed-
structure penalty.

**C1 splits by SIDE, and the support half is the first family in the project
to clear the train bar.** Untested-overnight prior-day SUPPORT: `pdl` clean
+0.129 [+0.038, +0.220], `pd_globex_low` +0.099, `pd_val` +0.099 [+0.008,
+0.194]; resistance mirrors: `pdh` +0.007, `pd_vah` +0.001, `pd_globex_high`
−0.084. Robust to near_t 4/6/8t (+0.12..+0.135) and to clean 15/10, 20/10,
30/15 (+0.11..+0.13).

**Pooled registered family** (at min(pdl, pd_globex_low, pd_val) ≤ 6t and
on_touches == 0, train): **clean +0.0856 [+0.0357, +0.1384]** (untested 0.162
vs tested 0.092 raw), slice −0.058, touch −0.119 (fresh support is reached
less, reacts better when reached). Per-year raw: 2021 +6.4pp, 2022 +11.4pp,
2023 +5.6pp, 2024-H1 −1.1pp (n=84/188 — small and the known regime risk:
validation is entirely 2024-25).

**Registered validation gate (written before looking).** Same population and
strata on validation: PASS iff clean_rate diff ≥ +0.03 AND CI lo > 0.
Secondary, non-gating: slice ≤ 0; clean_15_10 / clean_20_10 same sign;
per-year direction. One look, resolved below.

**VALIDATION RESULT: FAIL — and cleanly, not marginally.** 335 sessions,
2,402 rows (1,497 untested, 390 touched-and-resolved). clean_rate +0.0324
[−0.0734, +0.0825] — gate requires CI lo > 0. Raw clean|touch **0.0718
untested vs 0.0730 tested**; 2024: 0.073 vs 0.072; 2025: 0.071 vs 0.073.
clean_15_10 +0.004, clean_20_10 −0.003, slice −0.002. The sample was large
enough to detect the train-sized effect (SE ≈ 1.8 pp); the effect is absent,
not hidden.

**Verdict: REJECTED for deployment.** Untested prior-day support was a real
2021-2023 phenomenon (train CI excluded zero robustly) that does not exist in
2024-2025. Also of note: the *base* clean rate at these levels fell from
0.09-0.16 (2021-23) to ~0.073 (2024-25) — the current regime produces fewer
10-before-5 rejections at support generally. This is the strongest evidence
yet that Phase-2/3 conditional effects are regime-bound.

---

## P3-205 — Macro-release structures                 [REJECTED for cleanliness;
                                                      attraction KEPT]

**2026-08-14, train (275 release mornings).** Release morning = tape-detected
macro_volx_pre ≥ 4 (30 % of sessions, Thu+Fri 64 %, Mon 3 %); structures =
05:30-06:00 PT spike high/low/mid/VWAP (median spike range 282t ≈ 70 pts on
release mornings in-scope); grid marking within 6t, side × distance strata on
the same mornings.

**Attraction is real: every structure kind pulls price.** Within-strata touch
contrast +0.054..+0.076, all CIs > 0 (E1-residual attraction milder,
+0.012..+0.018). **Cleanliness is not:** extremes +0.009 n.s.; interior
structures negative (mid −0.034, VWAP −0.025 with slice +0.072 — the spike
interior is a transit/chop region); big-spike extremes *worse* (−0.049).
No validation look spent.

**Third independent confirmation of the energy-not-quality pattern** (after
P3-202 overnight events and P3-203 same-morning origins).

---

## P3-206 — Higher-timeframe extremes                                  [OPEN]

**2026-08-14, train.** Prior completed week and month H/L/C built from
carry-adjusted session extremes (`run_p3_priorweek.py`). Per kind (all
underpowered alone, exactly as P2-106 warned): pwh +0.047, pwl +0.037, pmh
+0.067, pml +0.066, closes ≈ 0. **Pooled HTF extremes** (pwh ∪ pwl ∪ pmh ∪
pml within 6t; the P2-106 hypothesis at adequate power): 3,538 rows, 756
sessions, 1,122 touched-resolved —

  clean_rate **+0.0388 [+0.0021, +0.0703]** p>0 0.992 (raw 0.165 vs 0.127)
  rev30m +0.0171 p>0 0.956 · attraction −0.016 (under-reached, fresh-flavoured)
  per-year raw diff: 2021 +5.1 pp · 2022 −0.2 · 2023 +5.2 · **2024-H1 +6.3**

Unlike P3-204's untested-support, the effect is present in the 2024-H1 tail,
and P2-106 (logged) already saw pwh +0.070 on validation — so part of the
family has one prior validation sighting, which this look partially re-uses;
stated for honesty.

**Registered validation gate (written before looking): PASS iff pooled
clean_rate diff ≥ +0.02 AND CI lo > 0.** Secondary, non-gating: rev30m ≥ 0,
per-year direction, per-kind signs. One look, resolved below.

**VALIDATION RESULT: FAIL.** 307 sessions, 1,484 rows, 598 touched-resolved.
clean_rate +0.0119 [−0.0301, +0.0581]; raw 0.1137 vs 0.1046; 2024 −1.5 pp,
2025 +1.9 pp. rev30m −0.005. Attraction flipped sign to +0.049 [+0.017,
+0.080] (train −0.016) — regime noise. Per kind raw: pwh 0.172 (n=261), pwl
0.082, pmh 0.065, pml 0.000 (n=43).

**Verdict: REJECTED as a family.** One honest remainder: `pwh` alone has now
been positive in every cut it was ever measured in (P2-106 train +0.064 /
validation +0.070; P3-206 train +0.047; this look raw +6.7 pp, n=261) but was
never pre-registered alone at adequate power, and validation is spent for
this family. **pwh is the one holdout-eligible cleanliness candidate** — a
single pre-registered shot on the sealed 2026 data could settle it, with the
multiplicity caveat stated (it is the best survivor of many looks, so the
prior is modest).

---

## P3-207 — Assembled product engine, end-to-end                       [KEPT]

**2026-08-15, walk-forward on validation.** Score = E1 P(touch | near-edge
distance, side, vol) × P(clean | touch, distance, side, day context), the
latter a logistic refit monthly-expanding (no location features in variant A).
Policy: distance window 0.15-1.0 ATR, 6-point min separation, top-K.
`run_p3_product.py`; scored frame saved to `daily/scored_axis.parquet`.

**Calibration (validation, walk-forward):** P(clean|touch) ECE **0.0104**
(mild +1 pp over-prediction as the expanding window digests the regime
change); joint P(touch∧clean) ECE **0.0027**. The annotation numbers a
display would print are honest.

**Selection vs distance-matched same-morning controls (368 mornings):**

| K | joint model | joint ctrl | Δjoint [CI] | Δcond|touch [CI] |
|---|---|---|---|---|
| 1 | 0.0788 | 0.0683 | +0.011 [−0.019,+0.044] | +0.028 [−0.013,+0.072] |
| 2 | 0.0883 | 0.0734 | +0.015 [−0.006,+0.039] | +0.022 [−0.007,+0.054] |
| 3 | 0.0842 | 0.0788 | +0.005 [−0.012,+0.026] | +0.008 [−0.017,+0.037] |
| 4 | 0.0904 | 0.0717 | **+0.018 [+0.003,+0.037]** | **+0.023 [+0.002,+0.050]** |

Per-year: positive in 7 of 8 year×K cells. Definition-stable: same selection
scored on the pure fp_rev15_adv10 race gives +0.022 joint / +0.020 cond.
**Variant B (adding memory/origin/energy/core location features to the same
model): equal or weaker at every K** (K=4 joint +0.015 [−0.003, +0.031]) —
location features add nothing even when a model may use them. **Policy probe:
forced window 0.30-1.20 ATR → edge vanishes** (−0.004 joint, −0.008 cond):
all skill is near-band placement + day context; there is no ranking skill
among far, traditional-level-like prices.

**Practical numbers (K=4, validation, per morning):** 4 zones; 2.77 reached;
P(≥1 reached) 0.92. Given a touch: clean(10-before-5 + non-loiter) 13.0 %;
pure races — 10-before-5 **46.9 %**, 15-before-10 **48.1 %**, 20-before-10
40.8 %, 30-before-15 37.2 %. P(≥1 of 4 gives 15-before-10) **65 %**, P(≥2)
43 %. Median 30-min rejection 41.5 pts; median eventual penetration 83.5 pts;
slice-through-60s 78 %; median time-to-10pt-rejection ~1 min. Honest shape:
with no location signal, optimal picks degenerate to near-symmetric ±0.15-0.2
ATR bands — E1's expected-range edges presented as zones.

**Verdict.** KEPT as the frozen-methodology candidate (RESEARCH_PLAN §13).
The engine's value = calibrated relevance (zones that get reached), calibrated
per-zone reaction annotation, and a small (+2 pp) matched-control selection
edge — not location identity.

---

## P3-208 — Reaction-definition audit: sweep + reclaim            [KEPT — and
                                                    the strongest result found]

**2026-08-15.** The clean-rejection label scores touch → penetrate → sweep →
reclaim → move as a failure. This audit built a path-aware taxonomy
(`labels/sweep.py`, +8 tests; `sweep_labels.parquet`, 231 k touched zones
walked once, definitions applied downstream) and re-asked every question.

**Definitions (frozen).** From first touch, R = 15 pts, boundary tolerance
d0 = 4t, reclaim = closes past a level held 60 s, penetration by range
(sweeps are wicks), reclaim by close (reclaims are commitment), post-reclaim
reaction anchored at the SAME near edge as a direct rejection; resolved =
≥ 1800 s of window.
  **A direct**: R-pt rejection while penetration-beyond-far ≤ d0.
  **B sweep+reclaim**: not A; penetrated > d0; reclaimed the NEAR edge with
  max depth ≤ D and first-beyond→reclaim ≤ T; R-pt rejection within 30 min
  of reclaim. **C acceptance/failure**: violated, never validly recovered.
  **D chop-without-penetration**: empty at this zone width (96-97 % of
  touches penetrate; churn shows up as re-sweeps instead, rate 0.74-0.87).
  Cells carried everywhere (depth has NO plateau — B triples from 16 % to
  40 % as D goes 6 pts → 0.10 ATR): strict D=24t/T=900 s, mid
  D=0.05·ATR/T=900 s, primary D=0.10·ATR/T=1800 s (the letter of the
  pre-written stage-1 criteria).

**Stage 1 — the behaviour is real but near-universal.** 96-97 % of touched
grid zones trade beyond the far edge; 79-99 % reclaim; 82-99 % of reclaims
produce 15 pts within 30 min. Gradients by depth (react-if-reclaimed 0.995 →
0.889 from 1 % to >15 % ATR) and by reclaim time (0.93 → 0.82) exist but are
shallow — sweep-then-bounce is close to a property of NQ noise, so only
matched-location contrasts can mean anything. Frequencies are year-stable
2021-2025 (A 21.7 % in BOTH splits; B within 1.5 pp across eras at every
cell). Temporary-sweep vs acceptance separation = the D/T caps; empirical
curves show soft boundaries, hence the three-cell discipline.

**Stage 2 — location identity STILL predicts nothing; every significant
family effect is NEGATIVE (train, side × distance strata).** core levels
y_any −1.1..−2.4 pp (CI < 0 at mid/primary); untested resistance −6..−8 pp
(all cells); untested support null even in its own 2021-23 era; pwh
−2..−3.5 pp n.s. (its clean-based candidacy does NOT extend — downgraded);
HTF extremes cat_B −5.3 pp (they sweep-and-FAIL more); mem_strong −1.7..−2.6;
mem0_any −2..−2.6; origins fresh/same-day/consumed and release extremes null.
Crediting the reclaim path makes "special" locations look worse — they are
where energy and churn concentrate. No family qualified for a validation
look; none was spent.

**Stage 3 — the frozen P3-207 engine's selection edge TRIPLES under the
sweep-aware outcome (validation, matched same-morning controls, nothing
refit):** strict cell — K=2 joint +0.0572 [+0.0214, +0.0935], cond +0.0880
[+0.0401, +0.1381]; K=3 +0.0304/+0.0466 [+0.0059, +0.0949]; K=4 **+0.0443
[+0.0154, +0.0756] / +0.0562 [+0.0175, +0.0977]**; P(≥2 of 4 react) 0.353 vs
0.254. 2024 and 2025 both positive at every K; mid/primary cells agree at
K=2 (+0.069 / +0.052 cond, CIs > 0), fade n.s. at K=3-4 — the edge
concentrates in the TIDY sequence. K=1 discounted (within-band geometry
mismatch, touch −11 pp, tautology-inflated). Train weaker and noisier
(+0.017..+0.023 cond; 2021 +6.7, 2022 +2.4, 2023 +1.6..+3.7, 2024-H1
−5.0..−0.9 pp on small n) — note train scores are the in-sample static fit,
validation the walk-forward, so validation is the deployment-realistic
number.

**Checks that could have killed it, run and passed:** (a) window mechanics —
restricting both arms to touches with ≥ 90/≥ 120 min remaining leaves
+0.044; model touches later (10.4 vs 8.8 min median), not earlier; model
wins within every touch-time bucket; (b) tautology direction — at K=4 the
model sits NEARER than matched (0.170 vs 0.235 ATR) so the extreme-price
gradient favours the CONTROL; (c) mechanism localised — within (session,
band, side) the score varies only by distance, so the edge is fine
*placement*, not level identity, consistent with every other result.

**Practical (frozen engine, K=4, validation):** touch 69 %; among reached:
direct 23.7 %, sweep+reclaim 40.0 %, combined 63.6 % (primary) / 52.8 %
(mid) / 40.8 % (strict), outright failure 36.4 %; median sweep depth 7.5 pts,
median reclaim 1.1 min, median post-reclaim travel 56.5 pts, re-sweep 87 %.

**Verdict.** KEPT. The sweep-aware outcome does not resurrect location
identity (stage 2), but it reveals that the engine's calibrated placement is
worth 2-3× more than the clean label showed. Frozen holdout proposal updated
in RESEARCH_PLAN §13.5. Holdout untouched.

---

## P3-209 — THE 2026 HOLDOUT TEST                                     [FAIL]

# **2026 HOLDOUT OPENED — MAY NEVER BE USED AS A SEALED HOLDOUT AGAIN**

**2026-08-15, authorized by the project owner. One shot, executed once, no post-holdout
tuning of any kind.** Pre-registration `holdout_sweep_engine_2026`, token
**16ce7f7051547785** (outputs/preregistrations/), written before any 2026 row
was read. Project is not a git repository; SHA-256 prefixes of the frozen
code/data recorded in the run log: topk.py eb7c60cb…, sweep.py d86a46a4…,
labeling.py 45703cd8…, baseline.py 403f375b…, run_p3_product.py de519d1c…,
scored_axis.parquet ff5f7e5a…, sweep_labels.parquet 38bc97a7….

**Pre-open verification.** 115 tests passing; no preregistrations directory
existed (no prior token ever issued); every research layer ended 2025-12-23;
infrastructure layers (bars, paths, grid features, context) span 2026 but no
experiment artifact contained 2026-derived statistics. Prompt vs registered
gate: no conflict. **Regression check before unsealing: the engine
reimplementation reproduced the stored validation scores with max |diff| =
0.00e+00 over 347,182 rows** — the engine run on 2026 is the engine that was
validated.

**Data.** 110 post-embargo holdout sessions 2026-01-09 → 2026-06-17; 109
measurable (2026-06-17 has no RTH window in the raw data). 120,229 grid zones
measured with the frozen path (half-width 6t, train-fitted E1); 33,960
touched zones sweep-walked. Frozen engine scored with the registered
monthly-expanding walk-forward. Evaluation: eval/topk, seed 23, n_draws 8,
n_boot 500, matched same-morning controls.

**REGISTERED GATE (verbatim §13.5) — RESULT: FAIL.**

| gate condition | required | observed |
|---|---|---|
| strict cond K=2 | > 0 | **−0.0218** [−0.0976, +0.0496] |
| strict cond K=4 | > 0 | +0.0022 [−0.0575, +0.0642] |
| CI lo > 0 at ≥1 of K=2/K=4 | yes | no |
| mid cond K=2 same sign | > 0 | +0.0007 (moot) |

Failed on the first condition. All-cell lifts, K=2/3/4: −2.2 to +2.9 pp,
every CI straddling zero. Monthly strict-cell cond (K=4): model below
matched in 4 of 6 months (Jan −2.7 pp, Feb −6.1, Mar +0.9, Apr −4.2, May
−2.1, Jun +9.8). Sample honesty: CI half-widths ±6-7 pp — the test could
have confirmed a validation-sized (+5-9 pp) effect; the point estimates are
zero/negative, so the edge is absent, not merely unproven.

**Artifact checks (diagnostic, unchanged model):** window-equalised diffs
−0.001..−0.010 (null is genuine, not timing); model med dist 0.171 vs
matched 0.243 ATR (same geometry as validation); model touched EARLIER (5.4
vs 8.4 min — the 2026 tape reached everything fast); touch contrast K=4
+0.023 [−0.007, +0.057].

**What HELD on unseen 2026 (third consecutive era):**
* **E1 destination model: P(touch) Brier skill +0.420** (validation-era
  +0.39-0.41), ECE 0.021 — fully intact.
* **P(clean|touch) annotation: ECE 0.0142**, joint ECE 0.0026, same mild
  +1.3 pp over-prediction as validation (pred 0.106 vs real 0.093).
* **Taxonomy frequencies:** direct ≈ 21 %, combined (primary) ≈ 62-63 % of
  reached zones — stable across 2021-2026. Sweep shape scaled with the
  regime: median B-sweep 11.75 pts (val 7.5), reclaim 0.9 min, post-reclaim
  travel 88 pts (val 56.5).
* Touch rate among picks 79 % (val 69 %) — 2026 ranges ran hot vs forecast,
  consistent with the known regime.

**Interpretation.** The P3-208 selection edge (+4.7..+8.8 pp on 2024-H2/2025)
was itself era-bound — the third such pattern (2021-23 untested-support died
by 2024-25; 2024-25 placement edge died by 2026). What survives every era is
the calibrated *probability* layer, never a *selection* edge over matched
same-morning alternatives.

**Verdict: FAIL on the registered gate. 2026 is spent as historical evidence.
No tuning, no rerun, no post-hoc search of 2026 was or may be performed.**

---
---

# PHASE 4 — strong vs weak zones, all sessions (NQ only)

Opened 2026-08-15. New unit of analysis: (zone, interaction) — real zone
objects with lifecycles, tapped in any session, target
P(successful reaction | touched, pre-touch information) under the frozen
path-aware taxonomy. Registered design and era-robustness bar in
RESEARCH_PLAN §15.1. 2026 is ordinary walk-forward data (holdout spent).
Levels only; no strategy artifacts.

---

## P4-300/302 — Zone + interaction corpus                              [KEPT]

**2026-08-15.** `features/zones.py` (confirmed zigzag pivots at 0.15/0.35/
0.75 ATR; availability = confirmation time), `features/interactions.py`
(dense 1-second engine: taps with re-arm max(8t, half-width) + 900 s, causal
approach/state features, 2-hour maturity rule so prior-tap labels never leak,
frozen sweep outcomes), `build_interactions.py` (chronological corpus pass,
roll-free registry, trailing 5-session profile and 20-session extremes).
6 new tests (121 total): tap-second exactness, truncation invariance of
features, re-arm, cross-session carry with roll adjustment, capped state
labels, pivot-confirmation correctness. Found and fixed: timestamp-resolution
trap (pandas `unit="s"` datetimes), within-session last-tap state, re-arm for
untapped sessions.

**Corpus: 61,354 zones, 341,797 interactions over 1,365 sessions (2021-01 →
2026-06), 99.0 % resolved; deduplicated (same session/side, ≤ 600 s,
≤ 0.05 ATR → oldest zone) to 126,222 observations.** NQ re-taps active areas
relentlessly: ~89 % of zones reach the 6-tap cap; 195 taps/session median
before dedup. Median 223 active zones at any time.

---

## P4-303 — Strong vs weak: the univariate screen           [KEPT — 13 robust]

**2026-08-15.** Base y (A ∪ B | touch, primary cell) = 0.4884 (direct 0.287,
sweep+reclaim 0.201); by year 0.438-0.559; by segment Asia 0.410 → premarket
0.549. 37 continuous features, top-vs-bottom tercile on y within
(segment × side) strata, session-clustered CIs, per-year signs. **13 clear
the registered bar** (|d| ≥ 2 pp, CI excl. 0, ≥ 5/6 years):

  context/tape state: atr_pctile **+10.2 pp**, rv60 +7.0, app_range_30m +5.9,
  app_range_5m +5.7, range_used +5.3 — reactions happen when the tape moves.
  approach: app_volx_5m +3.9, app_delta_5m +3.4, app_speed_5m +2.3,
  app_accel +2.0 — energy into the tap helps.
  zone-specific: **prior_succ_share +2.9** (the project's first positive
  same-zone memory — zones whose earlier taps succeeded succeed again),
  mins_since_last −2.9 (recently-tapped zones do BETTER — folklore inverted
  again), dep_vel +2.2 (formation departure velocity), base_atr +2.2.

Near-bar (6/6 years, < 2 pp): dep_mag +1.7, age −1.4..−1.7, mins_inside
−1.5, dep_delta −1.4. **Dead: tap_no (0/6 stratified), volume-profile
structure (volz/cliff ≈ 0, again), HTF scale (flat 0.488/0.497/0.490),
d20-extreme proximity (4/6 years), width, dep_volx.** Categorical: flipped
−4.9 pp (role invalidation is real and persistent), release-slot +5.4 raw,
fresh-first-tap +3.1 raw.

**Lifetime (Q12/13):** mild early decay (< 2 h 0.525 → 8 h-1 session 0.466)
then FLAT to 30 sessions — no cliff, no natural expiration inside the
tracked horizon (6-tap cap caveat). Objective invalidation = acceptance
through (flip): −5 pp at every subsequent tap number.

---

## P4-304 — Walk-forward models: what actually predicts               [KEPT]

**2026-08-15.** Tiers T0 context / T1 +zone-static / T2 +approach, expanding
year folds (test 2022..2026), HGB. **OOS AUC 0.72-0.76 every year — but one
feature dominates: approach side, permutation ΔAUC +0.189 vs ≤ 0.008 for
everything else.** Resistance-side taps (approached from below) succeed
**0.670**, support-side **0.307** — a 2.2× structural asymmetry, stable in
all six years (0.60-0.75 vs 0.27-0.36): the index's down-fast/up-slow
volatility asymmetry, OPPOSITE to S/R folklore. Increments: T1−T0 +0.005
AUC, T2−T1 +0.001 — zone identity and approach add little on top of
side + vol state at the pooled level. Pooled OOS: T2 AUC 0.7365, ECE 0.025.

---

## P4-306 — Top-k selection among reached zones      [KEPT — era-stable skill]

**2026-08-15.** Per session, rank that session's (deduped, resolved) taps by
walk-forward OOS score; controls: seg×side-matched same-session draws (side
and session-level regime cancel), next-K, bottom-K, session average.

**Primary outcome (T1 ranking): top-1 0.800, top-2 0.790, top-3 0.773,
top-4 0.768 vs matched 0.701-0.707. Top-2 − matched = +0.090
[+0.071, +0.108], positive in every OOS year** (2022 +6.1, 2023 +11.0,
2024 +9.1, 2025 +8.6, 2026 +10.3 pp). Bottom-ranked taps: 0.087-0.116.
**Strict outcome (model never trained on it): top-2 0.719 vs 0.600, +0.120
[+0.099, +0.138], all years positive** — definition-stable. T2 ≈ T1
(approach adds no selection skill). Availability: median 24 active zones
within 0.5 ATR, 51 within 1.0 ATR; nearest-4 median distance 0.08 ATR.

**The first era-stable selection result of the project** — with its honest
decomposition: most of the skill is side + volatility/timing state;
zone-specific properties (prior success, recency, formation velocity, base)
contribute the matched-control margin. 5 OOS years, 1,119 sessions,
103,446 scored taps.

---
---

# PHASE 5 — the true live display simulation

Opened 2026-08-16. Phase 4 ranked among taps (conditioned on touch); Phase 5
replays the actual product: at every 15-minute refresh, score ALL eligible
zones with the walk-forward model, display top-4, and track the displayed
set forward with no future-touch conditioning. Registered design in
RESEARCH_PLAN §16.1 (refresh cadence, eligibility incl. armed-state,
radius 0.02-0.75 ATR, K=4 with 0.05-ATR separation, episode definitions,
same-moment controls, advance-notice leads, side-asymmetry adversarial
checks, one pre-fixed 10 % hysteresis variant). Display-path features are
proven field-identical to tap-path features (tests/test_display.py).
2021 = burn-in; 2022-2026 simulated with strictly-prior-year models.

---

## P5-310 — The live simulation                                        [KEPT]

**2026-08-16.** `features/display.py` (+2 tests, 123 total), `run_p5_live.py`.
4,350,337 scored (refresh × eligible-zone) rows over 1,119 sessions
2022-2026; 414,379 displayed rows; median 38 eligible zones per refresh;
97,512 display EPISODES; 1,099,756 lead-time rescorings; 22,380
random-moment races. A float32-epoch-timestamp precision bug (~128 s error)
was caught before the run; all joins use integer within-session seconds.

---

## P5-311 — Live results                    [CENTRAL NEGATIVE + product core]

**1. The Phase-4 +9 pp zone-choice edge does NOT survive the live frame.**
Displayed vs same-moment side+distance-matched eligible controls, y | touch:
**−0.006 [−0.012, +0.001]**; per year +0.008 / −0.005 / −0.012 / −0.014 /
−0.006. Next-ranked ≈ displayed too (0.631 vs 0.634). The Phase-4 edge was
*moment/timing selection* (its controls shared session+side but not the
moment); at a fixed refresh instant every eligible zone shares the tape
state, side and distance are visible to any control, and the remaining
zone-identity choice adds ≈ 0 — the project's five-phase conclusion,
re-confirmed in the product frame.

**2. What the display IS worth (absolute, all walk-forward):** vs naive
same-moment choices the scored display is far better — random eligible zone
y 0.502, nearest 0.472, displayed 0.634 (+13-16 pp; strict 0.533 vs
0.322/0.321). Practical: 87 episodes/day (top-4, 15-min refresh); 53 % of
episodes' zones touched same-session, 18.8 % touched while displayed
(median episode 30 min); **10.4 successful displayed reactions/day; 99.3 %
of days ≥ 1**; median display distance 0.40 ATR (IQR 0.23-0.57); touched
episodes reached after median 16 min (p25 5, p75 54). Trade-off honesty
(System A vs B): "nearest" zones have joint touch∧success 0.376 vs
displayed 0.333 — proximity buys touches at the cost of quality (0.47 vs
0.63 conditional); the display sits deliberately on the quality side.

**3. Advance notice is real and mostly static:** score AUC at 30/15/5/2 min
before touch = 0.613 / 0.648 / 0.674 / 0.682; top-tercile y 0.630-0.693 vs
0.517 base; score corr(30 m, 2 m) 0.62. Dynamic approach features add
+0.001-0.002 AUC at every lead — approach adds ~nothing over side +
distance + state, again.

**4. Side asymmetry: REAL, passes every adversarial check.** Random-moment
±15-pt races are symmetric (down-first 0.467, up-first 0.474, per-year
0.42-0.50) — so the tap asymmetry (resistance-side y 0.647 vs support-side
0.363) is approach-conditional, not drift or noise skew. Decomposition:
resistance-side wins by DIRECT rejection (A 0.496, B 0.151); support-side
almost never rejects directly (A 0.030) and wins only by sweep+reclaim
(B 0.333). Rising approaches stall and dip; falling approaches plow through
then reclaim. Consequence the product must own: the ranked display is
**90.8 % resistance-side zones**.

**5. Stability/lifecycle:** rank-1 persistence 0.698/refresh; 73.6 % of
untouched episodes end by out-ranking, 26.4 % by eligibility loss. The
registered 10 % hysteresis variant: episodes 97.5 k → 41.1 k, median display
60 min, persistence 0.766, quality −1.1 pp (0.623 vs 0.634) — a good product
trade. Lifecycle keeps: 6-tap cap, flip-invalidation, soft age.

**Verdict.** The live simulation KILLS the "we can pick the special zone"
claim at same-moment controls, and VALIDATES a humbler product: a
state-driven display (side × distance × tape state, walk-forward scored,
calibrated, with 15-30 min of genuine advance notice) that delivers ~10
successful reactions/day at 63 % conditional success vs ~50/47 % naive
baselines, stable across 2022-2026. Whether that product is worth building
is the project owner's call; no display/UI work started.

---
---

# PHASE 6 — the exact-price question, magnitudes, and timing

Opened 2026-08-16. Four questions: does X carry pre-touch information that
X±δ lacks (the hard local control); reaction magnitude in NQ points; adverse
movement before reaction in points; when in the day levels work best.
Registered design in RESEARCH_PLAN §17.1: shadow anchors at ±2.5/5/10/20 pts
with identical width and a common t0 = tap − 15 min, every anchor scored
from its own first touch by one shared walker (`labels/enrich.py`, +4 tests,
127 total); event-paired symmetric-offset contrasts; enriched magnitudes
(near-edge adverse incl. in-zone travel, endpoint-based reaction);
robustness bar |d| ≥ 2 pp, CI excl. 0, ≥ 4/5 walk-forward years.
`events_full.parquet` rebuilt over all 1,363 sessions (14,784 events).

---

## P6-320 — Measurement corpus                                         [KEPT]

**2026-08-16.** `build_p6_measure.py`: 126,222 real anchors + 992,080 shadow
anchors walked identically (11.8 min); shadows 78 % touched, 81 % eligible
at t0; 78 % of shadows overlap another active zone (NQ pivots are dense) —
the no-overlap subset is analysed separately. One builder bug (missing
touched flag on walked shadows) found by the first analysis run and healed
from n_fwd; builder patched.

---

## P6-321 — THE EXACT-PRICE ANSWER                    [KEPT — first positive
                                                       spatial result, 6/6 years]

**Event-paired real-vs-shadow (±10 pts), all sessions 2021-2026:**

| outcome | paired diff | per-year sign |
|---|---|---|
| primary y (sweep-inclusive) | **−0.6 pp** [−1.0, −0.3] | negative 6/6 |
| strict tidy rejection | **+2.2 pp** [+1.9, +2.6] | positive CI>0 6/6 |
| reaction magnitude (endpoint) | **+3.5 pts** [+2.8, +4.3] | positive CI>0 6/6 |
| adverse before reaction | +0.6 pts (real slightly deeper) | — |

**The effect lives on the RESISTANCE side: strict +7.7 pp [+7.2, +8.2]
(per-year +6.2..+10.1, CI>0 all six), magnitude +4.9 pts; vs shadows that
overlap NO other zone: +8.7 pp [+7.6, +9.8]. Support side: −3.2 pp (real
pivots are anti-informative for tidiness — price sweeps them; P5's
side-mode finding predicted exactly this: support works only via
sweep+reclaim).** Pairwise P(real better | discordant), strict@resistance:
0.584 (±5) / 0.607 (±10). Uniform across TOD blocks (Asia +9.7, London
+6.9, premarket+NY-AM +6.6, evening +9.0 pp).

**Falsification passed:** placebo pseudo-pivots at +10/−10 with their own
symmetric neighbours show **negative** diffs (−1.5 pp, −0.7 pp) — the
paired construction manufactures nothing; the real pivot is genuinely the
anchor. The pooled signed-offset curve (monotone 0.34 → 0.68 in y) is the
LOCAL extreme-price tautology (touch-selection); only paired symmetric
estimates are interpretable, as registered.

**Interpretation.** The exact price does NOT react more often under the
sweep-inclusive definition (slightly less, −0.6 pp); it reacts more TIDILY
and travels FARTHER — the pivot anchors the actual turn. Effective width:
the strict lift rises to ±10 pts and flattens (+2.2 → +2.3 at ±20);
band ± ~5-10 pts is the empirical special region. Anchors are known a
median **9.9 h** before their tap (p25 2.3 h, p75 55.7 h; 50 % ≥ 1 session).

**Model reconciliation (P6-322):** spatial features (event/pivot/turn-price
density prominence, volume gradient/curvature, time-at-price prominence,
nesting) add NOTHING to tap-level ranking — B−A AUC −0.002..+0.001 (primary)
and −0.0004/−0.0007 (strict); spatial-only AUC 0.505-0.522; densities
mildly negative univariately (consumed-structure again). Because every tap
row is already pivot-anchored: the exact-location value is an **anchoring
property** (pivot vs displaced anchor), not a ranking feature among pivots.
Live-display implication: P5's selection conclusion stands unchanged; what
Phase 6 adds is that anchoring displayed zones at true pivots (rather than
arbitrary nearby bands) is worth +7.7 pp strict / +4.9 pts on the
resistance side — no simulator re-run needed since Model B ≡ Model A.

---

## P6-323 — Magnitudes and adverse depth, NQ points                    [KEPT]

**Successful interactions (61,647 of 126,222 resolved; 48.8 %).** Reaction,
endpoint-based (move ends when price closes back inside the band after
≥ 15 pts): **mean 63.6, p10 16.9, p25 20.7, median 31.8, p75 63.7, p90
141.6 pts** (30-min horizon: med 35.0; 60-min: 47.0; full-window 103.3 —
the endpoint definition is the honest attribution). **Adverse before the
15-pt reaction: mean 14.6, p10 1.4, p25 4.2, median 11.6, p75 21.9, p90
32.0 pts**, sharply mode-dependent: direct-front 3.0 med / deep-zone 11.2 /
sweep+reclaim 23.9. Reaction/adverse ratio median **3.7×** (p25 1.5, p75
11.8). corr(adverse, reaction) ≈ +0.01 overall, but **within sweeps deeper
is better**: post-reclaim reaction by sweep-depth quartile 37.6 → 40.3 →
43.7 → 53.1 pts (corr +0.18). Sweep shape: median depth 8.1 pts, median
reclaim 6.2 min, median post-reclaim 43.4 pts. Success mix: direct-front
38 %, deep-zone 21 %, sweep+reclaim 41 %. Turn locations: 23 % front
quarter, 15 % second quarter, 18 % back half, 22 % just beyond (1-1.5×),
22 % deeper sweep — bimodal edge-vs-beyond. Failures penetrate median
115.7 pts. 15-pt reaction arrives median 6 min after touch; max reaction
median 2.1 h later.

---

## P6-324 — Time of day                                                [KEPT]

One broad stable plateau: **05:00-11:30 PT** (premarket → NY morning →
early lunch), y 0.51-0.56, every 30-min bucket direction-stable ≥ 4/5
years; stable weak block 14:00-22:00 PT (0.35-0.41) plus 00:30-02:30.
Sessions: premarket 0.549 > NY-AM 0.537 > NY-PM 0.508 > London 0.453 >
Asia 0.410, same ordering every year. Reaction magnitude is nearly flat by
hour (med 30-33 pts); what varies: frequency, adverse depth (overnight
7-9 pts vs RTH 12-13), and speed (15-pt reaction med 2.5-6 min RTH vs
15-22 min overnight). Exact-location (strict@resistance) lift is uniform
across blocks — the anchoring effect is a round-the-clock property.

---

## P6-325 — Bounded MBP-1 spatial screen                          [REJECTED]

**2026-08-16.** 30 sampled sessions 2022-01..2025-06 (coverage end), 2,787
deduped taps, trailing-60-min book events within ±15 pts of the tap;
per-tick time-weighted resting size / adds / cancels / traded / replenish /
absorb aggregated as band-vs-ring prominence. **On strict@resistance — the
outcome where exact-location information demonstrably lives — AUCs are
0.502-0.516, i.e. nothing.** Pooled AUCs 0.52-0.54 reflect side/state
correlation the existing features already carry. Coarse screen by design
(detects ≥ ~8 pp), but the tape-based anchoring effect (+7.7 pp) would have
been visible. **Order-book history adds no local spatial value; P2-113's
rejection extends to the tap-local spatial form. Do not build MBP-1 into
the system.**

---
---

# PHASE 7 — harden resistance, solve support, freeze the product

Opened 2026-08-16. Registered design in RESEARCH_PLAN §18.1.

---

## P7-330 — Confirmation causality                                     [KEPT]

All 61,354 zones matched to their pivot extreme by re-running the confirmed
zigzag (`build_p7_confirm.py`, `zones_confirm.parquet`). **Confirmation
delay: median 7 min from the extreme (p25 2, p75 28, p90 104).** Lead time
confirmation → tap on the resistance population: median 9.6 h (p25 2.2,
p75 55.6, p90 169.7) — the 9.9 h Phase-6 figure stands under the strict
live-confirmation timestamp. Battery slice by confirmation delay: lift
+8.7 / +6.8 / +7.1 pp for ≤5 m / 6-30 m / >30 m — no hindsight dependence.

---

## P7-331 — The adversarial battery                     [PASSED — 40/40 cells]

Paired strict@resistance shadow contrast (±10 pts) inside every registered
slice: vol-percentile terciles (+7.2..+8.1), trendiness (+7.3..+8.5), day
volume (+6.8..+8.8), distance at t0 (+5.4 near .. +9.7 mid), band width
(+7.1..+8.9), nesting (+7.5..+8.1), confirmation delay (+6.8..+8.7), age
< 1 h .. > 2 sessions (+5.7..+8.4 — no expiry), tap number first/2-3/4-6
(+8.4/+8.7/+7.1 — repeat taps keep it), pivot scale (s0 +7.8; s1+ +6.3,
n=598), flip state (intact +7.9 / flipped +7.0), roll-adjacent (+7.3 vs
+7.8), release slots (+6.2 in / +7.8 out), and outcome definitions (strict
+7.7 / mid-cell +5.6 / sweep-inclusive **+3.1** — on the resistance side
even the generous outcome is positive; magnitude +4.9 pts). **Not one sign
flip; every cell CI > 0.** The resistance-pivot anchoring edge is the most
robust effect the project has produced.

---

## P7-332 — Pivot-definition robustness                                [KEPT]

Fractal pivots (k=2) on 5m/15m/30m/1H vs active zigzag centres, 400 sampled
sessions: **81/87/91/95 % of fractal resistance pivots lie within 1.5 pts of
an active zigzag zone; 95-98 % within 6 pts.** Reverse: 74 % of zigzag highs
have a same-session 5m fractal within 3 pts (higher TFs are sparse subsets).
Conventional definitions name the same prices; the effect is the LOCATION,
not the formula. Zigzag construction stands; no alternative-universe rebuild
warranted (registered criterion met).

---

## P7-333 — Support: no exact anchor exists            [ANSWERED — negative,
                                                        with the mechanism]

(1) Offset-curve peak tests on the support side: **the pivot low is a local
VALLEY** (y −5.4 pp, strict −3.1 pp vs its own ±10 neighbours; stable in
both eras −3.3/−2.9); 10-below +0.3..+0.5 pp (noise, far under the bar);
10-above negative; ±5 nothing. No interior peak anywhere in ±20 pts.
(2) Backward mapping: successful support is **92 % sweep-mode**, turning at
median **−15.5 pts below the low** (p25 −22.2, p75 −10.5; ≈ 0.06 ATR),
diffuse (IQR ~12 pts) — the turn tracks the excursion, not a prior price.
The "94.6 % of turns within 5 pts of another zone below (median 0.5 pts)"
is a DENSITY ARTIFACT (active zones are a few points apart everywhere); the
density-controlled peak tests are the honest evidence and they are null.
(3) Sweep-termination-cluster anchors (data-adaptive, prior-turn
concentration below the pivot): P7-334.

---

## P7-334 — Sweep-termination-cluster support anchor              [REJECTED]

The last registered support family: anchor = the prior-30-session
successful-turn concentration peak within [pivot−40, pivot−2] pts (≥ 2
prior turns), walked with its ±10 shadows from the same t0 by the standard
walker. 41,072 anchor events, median anchor 24 pts below the low. Peak
tests: primary −0.5 pp (2021-23) / +0.3 pp (2024-26); strict +0.1 / +0.1 —
every CI spans zero in both the freeze and confirm eras. **Support remains
UNSOLVED as an exact-location product**: successful support reactions are
diffuse reclaim events (92 % sweep-mode, turns tracking the excursion), not
anchored prices. Recorded as the honest asymmetric verdict the phase brief
explicitly allows.

---
---

# PHASE 8 — support only; resistance FROZEN

Opened 2026-08-17. Registered design RESEARCH_PLAN §19.1. The resistance
engine (§18.2) untouched throughout.

---

## P8-340 — Sweep-depth predictability                 [Product A closed by
                                                        the registered rule]

Support taps 63,034 (19,338 successes). Turn depth below the zone centre:
mean −17.3, med −15.2, IQR 11.8 pts; ATR-scaled spread barely tighter
(relative spread 0.73 vs 0.77 — 5 %). Walk-forward conditional quantile
models (HGB, causal state/approach/structure features): median pinball-loss
reduction **6-10 %** per year and predicted [q25, q75] interval width ≈ the
unconditional IQR (8-11 vs 10.4-11.2 pts, coverage 0.46-0.48). Dominant
feature: atr_pctile, then time-since-last-tap. **Far below the registered
20 % bar → the pre-sweep termination band is unpredictable enough to be
closed by inference** (the P6 offset grid had already shown no fixed-point
peak from −2.5 to −20). General penetration (all taps) med 107 pts —
unbounded on failures.

---

## P8-342 — Reclaim-validated old zones                          [REJECTED]

Support retests of zones whose prior tap produced a successful sweep+reclaim
(n_prior_B ≥ 1, 11,506 events): y lift vs shadows **−4.9 pp** — no better
than all-support (−4.3); prior_B ≥ 2 −5.2; any prior success −5.1; recency
of the validating event irrelevant; virgin zones −1.7; both eras negative.
**Nothing in an old low's history rehabilitates it as an exact anchor.**

---

## P8-343 — Sweep-termination pivots                  [REJECTED — first
                                                       retest exactly neutral]

dir = −1 zones tagged at confirmation as sweep-born (an older support zone
≤ 40 pts above swept within 12 h, sweep extreme within 6 pts of the new
low): 17.5 % of lows (5,395). Their support taps: all-taps −4.6 pp; **FIRST
retest: −0.3 pp y / +0.1 pp strict, CIs spanning zero** (base y elevated at
0.352 vs 0.30) — the only non-negative support cell in three phases of
searching, but a null, not an edge; tap 2+ reverts to −5.9. Era split noise.
**The event that creates the low does not make its price special.**

---

## P8-341 — Vol-scaled termination band                          [REJECTED]

The single registered Product-A shot: band centre = pivot low + k·ATR with
k = −0.0577 frozen from 2021-2023 successes, zone-width band, walked with
±10-pt shifted controls (48,857 events). Peak test: frozen era +1.9 pp
[+1.2, +2.5] primary / +1.1 strict (already under the 2 pp bar);
**2024-2026 confirmation: +0.7 pp [+0.0, +1.3] primary, +0.5 [−0.1, +1.0]
strict — era decay, FAIL.** No offset shopping performed.

---

## P8-344 — PHASE 8 VERDICT: OUTCOME D                    [support unsolved;
                                                          resistance stands alone]

Every registered support route exhausted: pre-sweep termination bands
(unpredictable depth: 6-10 % quantile-loss reduction vs the 20 % bar; fixed
offsets flat in P6; vol-scaled band decays to +0.7 pp), reclaim-validated
old zones (−4.9 pp), sweep-born pivots (first retest exactly 0.0, rest
negative), sweep-termination clusters (P7-334, null both eras).
**No exact bullish reaction zone is identifiable in advance in NQ, before
the first sweep or after it.**

**What support IS (characterised, 63,034 taps):** success 0.307 primary /
0.125 strict; when successful, reaction med 31.4 pts (same magnitude as
resistance) but through adverse med **23.2** pts (resistance: 6.5) —
reaction/adverse ratio med 1.48 (resistance ≈ 4.9); 89.7 % sweep-mode;
sweeps 8.3 pts beyond the old zone, reclaim med 6.1 min, 44.1 pts after;
best segments premarket 0.385 / NY-AM 0.371, Asia 0.203. Frequency exists
(41 deduped support taps/day, 14.2 successes/day) — the failure is purely
locational: the bullish turn is an excursion/reclaim EVENT whose location
is diffuse (IQR ~12 pts, centre drifting with volatility), not a remembered
price. The asymmetry with resistance is structural: rising approaches stall
at remembered extremes; falling approaches overshoot and reclaim.

**The resistance engine (§18.2) is unchanged. The final indicator is
legitimately asymmetric: exact resistance levels + state-based support
context (P5 machinery), no manufactured support lines.**

---
---

# PHASE 9 — confirmation around the frozen resistance zones

Opened 2026-08-17. Registered design RESEARCH_PLAN §20.1. Levels untouched;
no P&L/stop/target artifacts.

---

## P9-350 — Confirmation timeline corpus                               [KEPT]

`build_p9_timeline.py`: 63,188 resistance interactions walked; checkpoint
features at τ ∈ {15..900 s} (path, penetration, rejection progress,
velocity, time-above, buy/sell/delta, intensity, progress-per-buy-effort)
plus fire times and at-fire adverse/completed for all 25 registered rule
cells, searched ≤ 1800 s post-touch. `p9_timeline.parquet`.

---

## P9-351 — Mechanical rules                          [KEPT — velocity-break
                                                       wins; reclaim is a trap]

Base: strict 0.589, sweep-incl 0.670. Headlines (full grid in
outputs/p9_rules.json; parameter plateaus present in every family):

* **R4v10 — 10 pts down from the near edge within 60 s of touch: precision
  0.857 strict / 0.901 sweep-incl, fires on 22.5 % of touches, FP 7.8 %,
  median latency 20 s, adverse-at-fire 2.9 pts, median remaining 22.5 pts
  (66 % of the eventual move), worst YEAR 0.837** — six years inside 4 pp
  (0.837-0.873), segments 0.831-0.878. R4v7.5: 0.808 / 74 % remaining.
* **R2 stall family** (penetration ≤ p, back below the edge at τ):
  R2t300p5 0.822 (fires 31 %, remaining 17.0 pts, worst year 0.762);
  R2t60p5 0.740 with 79 % remaining — earlier, cheaper.
* **R3 reclaim-hold rules are a TRAP: precision 0.40-0.51, FP 30-70 %, and
  median remaining 0.0 pts** — by the time a held reclaim confirms, the
  endpoint-attributed move is finished. Sweep-mode reactions (19 % of
  successes) are effectively unconfirmable in time.
* **R6 acceptance filter**: ≥ 120 s of closes above the far edge → failure
  probability 0.687 (base 0.411) — a real disqualifier.
* R7 (1-minute close below zone): 0.666 — no better than trivial holds;
  R5 absorption: 0.769 but recall 0.17; R1 holds: high coverage, low lift.

---

## P9-352 — Checkpoint models                                          [KEPT]

Walk-forward OOS AUC on strict by time-since-touch: **pre-touch only 0.582**
(approach adds little — Q4 answered); τ=15 s 0.701; 60 s 0.787; 180 s
0.878; 300 s 0.914; 900 s 0.956 — every year within a tight band;
calibration ECE 0.018. Threshold curve (τ=60 s): thr 0.70 → 31 % of touches
qualify at realized **0.878** strict (0.908 sweep-incl), adverse-so-far
2.8 pts, median remaining 18.0 pts; thr 0.80 → 20 % at 0.945 / 14.0
remaining. τ=180 s: thr 0.70 → 39.5 % at 0.922 / 10.7 remaining. **The
model's operating points barely beat the simple R4v10 rule (+2 pp precision
for −4.5 pts remaining) — per the registered simplicity preference, the
mechanical rule is the recommendation; the calibrated probability is the
optional dial.**

---

## P9-353 — The frozen confirmation framework                          [KEPT]

**Rule: confirm when EITHER (a) price has moved 10 pts down from the zone's
near edge within 60 s of touch [R4v10], OR (b) at 5 minutes the maximum
penetration is ≤ 5 pts and price is back below the near edge, provided
2 minutes of closes above the far edge have NOT occurred first [R2t300p5
gated by R6T120].** Performance (63,188 interactions, 2021-2026):

  fires on 44.8 % of touches · precision **0.809 strict / 0.841
  sweep-incl** (base 0.589/0.670) · recall 61.5 % of successes · FP 20.8 %
  · latency med 60 s (p25 20, p75 300) · adverse before confirmation med
  **2.3 pts** (p90 7.5) · completed before confirmation med 11.1 pts ·
  **remaining med 20.5 / mean 53.2 pts (66 % of the move)** · by year
  0.758→0.868 (monotone-improving, all six positive) · by segment: NY-AM
  0.874, premarket 0.850, NY-PM 0.839, London 0.766, Asia 0.728 · mode
  coverage: front 82 %, deep 29 %, sweep 17 %.

---

## P9-354 — Bounded MBP-1 confirmation add-on                    [REJECTED]

40 sampled sessions 2022-01..2025-06, 2,106 interactions, touch-time book
features (imbalance, spread, microprice drift, zone-local adds/cancels/
replenishment/traded) vs the trades-only checkpoint features on the SAME
rows (session-grouped 5-fold; sample too small for yearly walk-forward,
stated as such). **τ=60 s: trades 0.7605 vs +book 0.7585; τ=180 s: 0.8574
vs 0.8574 — zero increment.** Book-only AUC 0.72/0.80 shows the book sees
the same event through a redundant lens. Third consecutive rejection of
MBP-1; the trades-only framework additionally generalizes into 2026 where
MBP-1 cannot. Do not build the order book into the system.

---
---

# PHASE 10 — the strategy conversion test

Opened 2026-08-18. Registered design RESEARCH_PLAN §21.1. Frozen levels and
confirmations untouched; first phase permitted entries/stops/targets/P&L.

---

## P10-360 — Trade simulation corpus                                   [KEPT]

`build_p10_trades.py` + `run_p10_scores.py` (persisted walk-forward
calibrated scores, 82 % OOS coverage). Every confirmation event (R4v10;
stall gated by the disqualifier) × 12 entry variants × first-passage times
at 20 stop levels and 44 target levels on the 1-second path; same-second
ambiguity stop-first; limit fills require 1-tick trade-through within
900 s; costs (RT, points): market 0.5/1.0/2.0, limit −0.25.
`p10_trades.parquet`.

---

## P10-361 — Stage 1 (2021-2023 selection)                             [KEPT]

* **Entries: immediate wins.** E1d (2-s delayed market) +0.096R grid-mean,
  E1 +0.078, 2-4-pt pullbacks +0.084-0.087 (fill 89-94 %), deeper pullbacks
  decay to ≤ 0; retracement entries −0.04 to −0.05; near-edge retest −0.035
  with 25 % never filling — pullback-waiting selects adversely.
* **Stops: broad plateau S1 = 10-15 pts** (expR +0.108..+0.112 at T=20);
  extreme+buffer (S3) equivalent; 0.02-ATR too tight.
* **Targets: expectancy rises with T** to 40 pts (+0.180) — the right tail
  pays; high-WR small targets barely clear costs (T=10: +0.029R).
* **The 5-minute stall source is NEGATIVE as a trade everywhere** (−0.03..
  −0.15R) despite its 82 % level-precision: by its 300-s confirmation the
  cheap part of the move is gone and its geometry is poor. The ungated
  two-part framework is thereby dragged to ≈ 0. **R4v10 is the tradeable
  confirmation; the calibrated-model gate rescues/boosts monotonically**
  (framework thr 0.90: +0.235R on 21-23).

---

## P10-362 — Stage 2: finalists frozen on 21-23, executed on 24-26     [KEPT]

| finalist (net base costs) | OOS 24-26 | per-year range |
|---|---|---|
| F1 balanced r4/E1d/S12.5/T30 | +0.126R, WR 0.356, PF 1.18, 14.0 tpd | +0.07..+0.21R, 6/6 positive |
| F2 max-exp T40 | +0.138R, WR 0.292 | +0.07..+0.20 |
| F3 high-WR S30/T15 (frontier-frozen) | **WR 0.722**, +0.051R, PF 1.18, DD 29R, streak 11 | +0.02..+0.08, 6/6 positive |
| **F4 model-gated r4 ∩ p60 ≥ 0.85, S12.5/T30** | **+0.376R, +4.70 pts, WR 0.429, PF 1.61, DD 27R, 9.5 tpd** | **+0.29..+0.44R, 5/5 positive** |
| F5 frequency (r4∪st) ∩ p60 ≥ 0.70, S10/T20 | +0.103R, 19.1 tpd | +0.02..+0.15 |
| stall-alone / ungated framework | −0.065R / +0.004R | REJECTED |

**Battery on the finalists:** costs — F1 stress +0.046R, **F4 stress
+0.296R (PF 1.45)**; delay E1 vs E1d indistinguishable (+0.132/+0.126);
neighbour S×T grid flat (+0.075..+0.138, no spikes); partial 15/40 half-half
lifts F1 to WR 0.521 / +0.142R / PF 1.28 / DD 36R; sessions (F4): Asia
+0.245, London +0.318, premarket +0.402, NY-AM +0.403, NY-PM +0.386 — all
positive; F3's only weak session is Asia (+0.002, breakeven); tap-number
gradient (first +0.322R, 4-6 +0.076R) — largely what the model gate already
encodes. **Portfolio one-open-trade pass:** F4 +0.363R at 7.6 trades/day,
24 % zero-trade days; F1 +0.127R at 10.2 tpd (4 % zero days); F3 WR 0.726,
DD 23R, longest streak 8.

---

## P10-363 — PHASE 10 VERDICT                                          [KEPT]

**The reaction edge converts into a real, robust, multi-year positive
trading strategy.**

---
---

# PHASE 11 — the risk wrapper around frozen F4

Opened 2026-08-18. Registered design RESEARCH_PLAN §22.1. F4 untouched.

---

## P11-370 — Count audit                                               [KEPT]

Exact denominators per year (sessions / interactions / R4 fires / gated):
2022: 251 / 10,901 (43.4/d) / 2,708 (10.8/d, 25 %) / 1,715 (6.8/d);
2023: 250 / 10,400 / 1,541 (15 %) / 937 (3.8/d); 2024: 252 / 11,484 /
2,203 (19 %) / 1,442 (5.7/d); 2025: 250 / 12,449 / 3,847 (31 %) / 2,482
(9.9/d); 2026: 115 / 6,605 / 2,430 (37 %) / 1,634 (14.2/d). Gate passes
61-67 % of fires. **Reconciliation:** the earlier "~15 touches/day"
referenced the P5 top-4 DISPLAY subset (16.4 touched/day); the strategy
universe is ALL eligible deduped resistance interactions (~46/day); fire
share is strongly era-dependent (13-37 %; 22.5 % was the 2021-26 pooled
figure). F4 exists from 2022 (walk-forward score coverage 82 %). Executed
one-open-trade stream: **7,898 trades / 1,041 active of 1,118 sessions**
(2022:1,659 / 2023:864 / 2024:1,385 / 2025:2,393 / 2026:1,597; asia 834,
london 752, premkt 1,548, ny-am 2,593, ny-pm 2,171). OOS terminology fixed:
S/T/entry frozen on 21-23; the p-gate is walk-forward everywhere; genuinely
frozen OOS = 2024-26; tail statistics pooled 22-26 and labelled.

---

## P11-371 — Drawdown anatomy, streaks, daily distribution             [KEPT]

Executed stream, R units. **Max DD 28.8R** (peak 2023-08-04, trough
2023-09-01 after 20 sessions, recovered 2023-10-24 after 36 more; longest
no-new-high span 56 sessions). Episode frequencies: ≥5R 35/yr, ≥10R 7.4/yr,
≥15R 1.6/yr, ≥20R 2 in 4.5 yrs, ≥27R once. **The 27R-class event is a
~p95-of-a-year event, not a freak** (see MC). Streaks: ≥5 51/yr, ≥8 8.5/yr,
≥10 2.0/yr, ≥15 once; **the 18-loss streak occurred INSIDE ONE SESSION
(2026-05-04, −19R day region; NOT part of the max DD)**. Daily: mean
+2.75R, median +1.40R, 68 % positive, worst −17.1R, best **+130R
(2025-04-07, 163 trades — extreme-churn tail days are real but practically
inexecutable; 76 days exceed 20 trades)**; ≤−5R days 2.6 %; months 94 %
profitable, worst month −12.4R (2023-08). Trades/day: 0:7 %, 1-4:43 %,
5+:50 %.

---

## P11-372 — Wrapper replays                    [KEPT — restrictions do not pay]

Every wrapper reduces total return; almost none reduces max DD, because
drawdowns are MULTI-WEEK bleeds (20 sessions to trough) that daily rules
cannot touch, and every skipped trade forfeits +0.36R on average
(within-day sequences are unpredictable: P(loss | prior same-day loss)
0.574 vs 0.575 base). Headlines (totR / maxDD vs RAW 2,865R / 29R):
daily stop −1R: 1,169R / 38R (WORSE DD — blocks same-day recovery); −3R:
2,419 / 33; −6R: 2,784 / 29 (the only near-free stop, −2.8 % return);
profit caps: catastrophic (cap +2R: 1,138R) — the edge lives in right-tail
days; combined grids: all dominated; max-trades caps: cost proportional
(max-8: 1,667R — but this is the practical execution profile); consec-loss
stops: day-scope bad, session-scope −4/−5 nearly free (−197/−123R).
**Verdict: no daily wrapper improves money/DD; optional −6R emergency stop
and a practical trade cap are the only defensible rules.** Overlays:
Asia/London half-risk ≈ nothing; **skip-tap≥4 overlay: expR +0.454, maxDD
17R, PF 1.77 at −41 % total** (the one genuine DD-adjusted improver, kept
as optional overlay); overlap-allowed instead of one-open-trade: +0.191R,
DD 48R — one-open-trade is strongly protective.

---

## P11-373 — Monte Carlo, design DD, prop engine                       [KEPT]

Day-block bootstrap (clustering factor 2.53 preserved), 2,000 × 250-day
paths: **max-DD p50 15.1R, p90 26.4, p95 28.9, p99 34.8, p99.5 37.7;
P(≥27R) = 8.7 %/yr, P(≥35R) = 0.9 %**. The historical 28.8R ≈ p95. **Sizing
by buffer/27 is NOT safe; design DD = 35-40R (p99).** Parameterised
trailing-buffer engine (daily granularity; REAL firm rules still required):
at B = $4,500 — risk $100 (45R buffer): P(viol) ≈ 0 %; $125 (36R): 0.8 %;
$150 (30R): 4.0 %; $200 (22R): 19.8 %; $300 (15R): 53.7 %. Harsh
confiscation model (monthly payout sweeps, violation forfeits unswept):
E[paid]/yr at $4.5k buffer — $125: $74k (P(viol) 0.4 %, p10 $52k); $150:
$88k (3.7 %, p10 $60k); $300: $141k (42 %, p10 $23k). **Expected-extraction
peaks at aggressive risk only by tolerating frequent account death;
risk ≈ buffer/30-36 captures most extraction at ~1-4 % annual violation.** The recommended implementation is **F4**: short on
R4v10 confirmation at a frozen §18.2 resistance zone, gated by the
walk-forward calibrated probability ≥ 0.85, market entry within seconds,
stop 12.5 pts, target 30 pts — +0.376R/trade net OOS (≈ +4.7 NQ pts ≈ $94
per contract at $20/pt), PF 1.61, 27R max drawdown, ~7.6-9.5 trades/day,
positive in every walk-forward year and every session, degrading gracefully
under stress costs and 2-s delays. F3 (S30/T15 ungated) is the validated
high-WR alternative (72 % WR, thin but 6/6-year-positive expectancy, small
drawdowns). Illustrative only — no sizing/account rules researched.

**Mechanism of the asymmetry:** support does not anchor because the falling
approach penetrates and the reaction is a reclaim EVENT, not a price; the
resistance-side rising approach stalls at the remembered extreme.

---

## P7-335 — Resistance pivots CAN be ranked                            [KEPT]

Within the resistance-tap universe (63,188; base strict 0.589), features on
y_strict at seg × distance strata: **prior_succ_share +3.6 pp (6/6 yrs),
mins_since_last +3.8 (6/6 — rested pivots better in THIS universe),
dep_vel +3.3 (6/6), base_atr +3.0 (5/6)**, dep_volx +1.3 (6/6), broke_prior
+1.7 (5/6); weakeners: vol_through −3.6, delay_min −3.2 (6/6), flipped
−3.0 (6/6), age −2.6, tap_no −2.3, dep_delta −2.3, ev_n6 −1.6. Walk-forward
top-2-per-session by pivot score vs SAME-SESSION SEG+DISTANCE-matched
resistance pivots: **2023 +15.2 pp, 2025 +9.2, 2026 +17.2** (top-2 strict
0.67-0.82). The first population in seven phases where pivot-specific
quality ranking works.

---

## P7-336 — The resistance product                                     [KEPT]

**Population (63,188 taps):** strict 0.589, sweep-inclusive 0.670. Successes:
reaction med 32.0 / mean 63.6 pts (p90 140); adverse med **6.5** / mean 9.7
(front 3.0 / deep 10.8 / sweep 23.1); modes 56 % front / 25 % deep / 19 %
sweep; sweeps 7.6 pts deep, reclaim 6.5 min, 41.7 pts after; failures
penetrate med 107 pts. **Width:** lift at ±5 (+3.5..+4.5) and ±10
(+7.2..+8.1) is flat across vol terciles in POINTS — fixed-point geometry
holds; keep band = pivot ± min(base/2, 0.03 ATR) (median ±7.2 pts), special
region extends ~±10. **Live display (offline resistance-only re-rank of the
P5 pool):** 78 episodes/day, 54 % touched same-session (20 % while
displayed), y|touch 0.661 / strict 0.570, per-year strict 0.50-0.63 (all
healthy), med distance 0.38 ATR, med display→touch 16 min, **8.7 strict
successes/day**, and the exact-anchor lift among displayed-and-tapped zones
**+3.0 pp [+2.2, +3.8]** (diluted from +7.7 population-wide because the
display selects near zones, where the battery shows the smallest — still
solidly positive — lift).

---
---

# PHASE 12 — funded-account lifecycle economics (Topstep 150K XFA)

Opened 2026-08-19. Registered design §23.1. Objective flipped: maximise
NET CASH WITHDRAWN, account deaths permitted. F4 untouched.

---

## P12-374 — Lifecycle engine + streams                                 [KEPT]

`p12_engine.py` (vectorised over paths; selftest incl. hand-traced payout,
death, split, consistency-block cases) + `p12_streams.pkl`: raw (7,898 tr,
+0.363R), cap8/6/4 (first-N-of-day: 5,106/4,388/3,366 tr, +0.326/.318/.332R),
tap3 REBUILT through the one-open-trade portfolio on tap<=3 fires (3,835 tr,
+0.453R, PF 1.77 — post-hoc filter would give 3,746), tap3cap8 (3,193,
+0.417R). Raw rebuild byte-matches p11_stream. Rule model per §23.1
snapshot. Sensitivities at the balanced config: full-1R intraday adverse
≈ no change (mean −0.1%); block-20 vs block-5 ≈ no change (p10 −5%);
split-free would add ~9%. Payout processed EOD of eligible day
(simplification), balance forfeited at death, horizon-end balance NOT
credited (conservative for high-cushion configs).

---

## P12-375 — Payout path: STANDARD wins; DLL is free                    [KEPT]

cap8, immediate/max, annual net with $150/5d replacement. Standard pays
24-25×/yr vs Consistency 15.6×/yr: the 40 % largest-day rule leaves the
account in eligible-but-blocked state ~164 days/yr against F4's fat right
tail → std net > cons net at every risk ($150: 48.0k vs 46.6k; $300: 92.4k
vs 80.6k). Consistency's only virtue is fewer deaths (bigger balances).
The $3,000 DLL NEVER binds at risk ≤ $175 (worst prefix day −17.1R ×
$175 = $2,993) and barely at $250-300 → taking the DLL for the promo caps
($10k/$12k) is free money; the $10k cap matters most for fast streams
(raw $250: $132k vs $107k with $5k cap). No consistency-motivated F4
truncation tested further — path dominated before behaviour changes.

---

## P12-376 — Policy surface: T1 flat, cushion knee at ~18R, take 50 %   [KEPT]

First-payout trigger T1 ($0→$15k): net FLAT (the 50 %-of-balance rule
already banks half and leaves half — building buffer first buys nothing).
Retained cushion C: net flat from C=0 to C≈18R·risk, then declines
(idle balance is eventually forfeited); deaths/yr collapse 1.2→0.2→0.0
across C=0→18R→27R ($150 risk). Knee: **C = 18R×risk** (e.g. $3k at $167).
Withdrawal fraction: 50 % > 40 % > 33 % > 25 % ($53.0k vs 49.5k) — always
take the maximum. Payouts land every ~14-16 days (24-25/yr), first payout
median day 10-16 (p90 ~30).

---

## P12-377 — Risk grid + the /27 hypothesis: CONFIRMED, and it under-
shoots — extraction keeps rising far past /27                           [KEPT]

Annual net rises MONOTONICALLY in risk to $600/trade on every stream
(cap8: $100→32k, $167→53k, $250→79k, $400→126k, $600→184k; raw: $150→82k,
$250→134k, $400→195k, $600→229k) because (a) each replacement account's
fresh −$4,500 MLL is subsidised drawdown at $150/5d, and (b) the binding
ceiling is PAYOUT FREQUENCY × CAP (5-winning-day rule ⇒ ~25-27 cycles/yr;
× $10k × ~0.9 split ⇒ ~$240k/yr/account; raw $600 nearly saturates it).
P(die before first payout): ≤1 % to $200, 3-5 % at $250, 12-13 % at $400,
24-26 % at $600. /27 layout (cap8, C=0 max-strip): P(first payout)=1.000
at $150-175, die-before-first ≤0.4 %, payouts before death med 14
(p10 1 / p25 3 / p75 32 / p90 59), cash before death med $29k / mean $47k
(p90 $130k), P(≥1/≥3/≥5/≥10 payouts) = 1.00/.77/.70/.59. With C=18R the
dying minority banks med $64k (cap8 $167) to $135k (raw $250) FIRST.
Survival after payout k (C=18R): flat in k (≈59-64 % cap8, 92-93 % tap3)
— death risk is regime-driven, not payout-count-driven.

---

## P12-378 — Failure-matched streams: tap3 dominates; replacement cost
irrelevant; dynamic risk dominated                                      [KEPT]

At MATCHED P(die≤250d) (rising-limb interpolation): 5 %: tap3 $286→$92.5k
vs raw $68→$37.2k vs cap8 $58→$18.5k; 10 %: tap3 $408→$128.2k vs raw
$119→$64.7k vs cap8 $86→$27.4k — **tap3 extracts 2-5× more per unit of
failure risk** (its 13.4-14.6R DD supports ~3.4× the risk$). tap3 cannot
even reach 20 % failure (max 18 % at $600); raw at 20 % → $203k. At the
SAME RISK $, raw > tap3 (more trades); at the same FAILURE, tap3 > all.
Replacement cost×delay ($0-1,000 × 0-20d): moves annual net ≤3 % even at
$400 risk (deaths 0.2-0.4/yr under C=18R) → aggressive risk never flips.
Dynamic risk (escalate after payouts/cash milestones): every dynamic
policy is dominated by the constant higher risk (early death is cheap;
house-money staging only forfeits early extraction). De-escalation worse.

---

## P12-379 — Chronological case studies                                 [KEPT]

2022-26 replay: every serious config banked 43-44 payouts and $90-197k
BEFORE the Aug-2023 28.8R drawdown began (cap8 $150: $91.6k; raw $250:
$195.6k). With C=0 all died in Aug-Sep-23 (economically fine: banked >>
$150 reset). With C=18R: cap8 $150 and raw $150/$167/$250 SURVIVED the
whole 4.6 years (raw $250: 116 payouts, $634k); cap8 $167/$250 died in
the DD anyway (knife-edge — cap8 DD ≈ 18R·risk ≈ C). Adversarial start
on 2023-08-04 (cap8 $167): C=0 died day 39 with $603; C=18R survived →
64 payouts, $145.6k. Churn day 2026-05-04: raw −17.1R/19t vs cap8 −5.2R,
cap6 −3.1R, cap4 −0.9R; monster day 2025-04-07: raw +130R/163t vs cap8
+8.4R — caps cut both tails; net effect of cap8 ≈ −40 % annual extraction
vs raw at equal risk.

---

## P12-380 — Final configurations                                       [KEPT]

N=2000/config, C=18R, std+$10k promo cap, $150/5d replacement. P(net<0)
= 0.000 in EVERY config (worst p10 strongly positive). Headliners
(mean / p10 / p90 net$/yr | deaths/yr): SURVIVAL tap3 $150 C=27R: 48.5k /
33.8k / 65.4k | 0.00. BALANCED cap8 $167: 53.2k / 40.8k / 65.6k | 0.18.
tap3 $300: 96.9k / 68.9k / 128.8k | 0.06. tap3 $400: 126.3k / 91.3k /
165.0k | 0.10. cap8 $400: 125.6k / 95.7k / 155.5k | 0.34. raw $250:
133.6k / 95.7k / 173.5k | 0.16. raw $400 (16 MNQ, ~7 tpd — automation
territory): 195.4k / 149.1k / 237.8k | 0.23, P(≥$150k)=0.90. Philosophy
table (P25) in outputs/p12_final.json. Caveats: assumes F4 edge persists
live; Topstep behaviour tolerance, payout processing, and rule changes
not modelled; all dollars are rule-snapshot-dependent.

---
---

# PHASE 13 — pre-implementation audit of F4 + tap<=3

Opened 2026-08-20. Registered as final audit before live forward test.
AUDIT ONLY — no parameter changes. Canonical ledger + full battery +
execution-chronology verification.

---

## P13-381 — Canonical ledger                                           [KEPT]

`p13_ledger.parquet`: 3,835 executed trades (portfolio one-open-trade over
tap<=3 R4v10 fires, p_both60>=0.85, E1d fill at close of fire+3s, S 12.5 /
T 30, base cost 1.0 pt RT, roll-adjusted continuous prices). Reconciles
P12 exactly: n 3,835, expR +0.4531, PF 1.766, WR 45.14 %. Outcomes: 2,098
stop / 1,725 target / 12 session-close (+0.13R avg). Row schema: ids,
zone geometry+creation, tap/tap_ts, r4 fire sec, p score, entry/exit
ts+px, stop/target px, gross/net pts+R, seg, costs. An independent 1s-path
recompute reproduces ALL 3,835 rows to 0.0000 pts.

---

## P13-382 — Statistical profile (of the frozen SIMULATED stream)       [KEPT]

Trade level: avg win +2.314R (28.93 pts), avg loss −1.078R (−13.47 pts),
payoff 2.147, no outliers (fixed S/T). Years (ALL positive): 2022 +0.475
(sel), 2023 +0.382 (sel), 2024 +0.342, 2025 +0.541, 2026 +0.446 (frozen-
OOS era for F4 params). OOS 24-26: n 2,593, +0.459R, PF 1.78, DD 17.4R,
posM 96.7 % — tap<=3 cutoff itself chosen on pooled 22-26 (P11), NOT OOS.
Tap decomposition (fire level): 1: +0.367, 2: +0.166, 3: +0.194, 4:
+0.146, 5: +0.127, 6+: +0.100 — smooth plateau, 3 not a knife-edge
(cutoffs ≤2/≤3/≤4 executed: +0.485/+0.453/+0.420). Streaks: max 10
(2024-12-04→17, 5 sess, −10.8R; raw-18 was tap>=4 churn), ≥5: 21.7/yr,
≥8: 3.1/yr. DD: 17.4R trade-gran / 14.6R day-gran (2026-04-30→05-05→
05-12; 3 down, 8 to recover); ≥10R 2.0/yr; episodes med 2.2R p95 10.3R.
MC block-5 250d: p50 12.2 / p90 17.7 / p95 19.7 / p99 24.3 / p99.5 26.5;
P(≥20R)=4.6 %. Daily: mean +1.55R, med +0.16R, best +86.6 (2025-04-07,
93 tr), worst −7.4R; tpd mean 3.43 / median 2; 17.9 % no-trade days.
Monthly: 52/54 positive (96.3 %), worst −6.5R (2024-03), max 1 consec
negative. Sessions all positive: ny-am +0.501 best, asia +0.327 weakest.
Time-of-day: positive in all populated blocks. Cost stress: +0.5/+1/+2
pts RT → +0.413/+0.373/+0.293R (PF 1.67/1.59/1.43); breakeven extra
slippage 5.66 pts RT; breakeven WR 31.8 % vs actual 45.1 %. Forward
windows: 100-trade expR p5 +0.12 (P(neg)≈0.4 %), 50-trade P(neg) 3 %.

---

## P13-383 — Holiday and macro audits                                   [KEPT]

Only 8 trades in 4.6 yrs land on CME holiday/shortened sessions (pipeline
barely fires on thin days): −5.2R total, WR 12.5 %, worst 2025-07-03
−3.2R. No major DD/streak/negative month is holiday-driven. Operational
skip = harmless (+5R/4.6yr) engineering simplification, NOT a statistical
filter. Macro: FOMC days +0.487R (n 262; afternoons +0.489), NFP days
+0.600R (n 251), ordinary +0.439R — event days fine, NO news filter.

---

## P13-384 — LOOK-AHEAD IN THE F4 GATE                    [CRITICAL FINDING]

**The frozen F4 gate is not computable at its own entry time.** p_both60
uses cp60 features (data through touch+60s); R4v10 fires at median 16s
(p75 32s) post-touch; E1d fills at fire+3s → **98.3 % of entries precede
the gate's information window; mean look-ahead 38 s.** Exact 1s-path
recomputation (machinery verified to 0.0000 pts on the frozen variant):
LIVE-LEGAL timing (features at touch+60, fill at close of touch+62/63):
**+0.046R / +0.037R, PF 1.06/1.05, DD ~52-56R** — the edge collapses;
during the mean 40 s wait price falls a further mean 7.85 pts (median
4.0) — the gate was "predicting" a move it could already see. Ungated
R4v10 tap<=3 at fire+3 (fully legal pure price rule): **+0.191R, PF 1.28,
n 5,672, every year positive (+0.10..+0.27), DD 55.7R** — the genuine
live-capturable residual. Gate separation at fire+3 (+0.453 high-p vs
−0.345 low-p) is therefore mostly leaked future price action, not
forecast. Delay stress with the (illegal) gate held: fire+4/+5/+8 →
+0.439/+0.419/+0.353R — latency was never the issue; chronology is.
**Amendments: P10's F4 OOS +0.376R, P11's wrapper numbers and P12's
lifecycle dollars all describe the non-implementable gated stream and are
WITHDRAWN as live expectations.** P9 itself was internally honest
(remaining-reaction measured from the 60s checkpoint); the flaw entered
at P10 when the 60s score was grafted onto fire+3 entries. P10's
"pullbacks adversely selected" is also contaminated (later entries
carried less look-ahead by construction). F3 (stall source, fires at
300s > 60s window) is NOT affected by this mechanism.

---

## P13-385 — Verdict: NOT READY                                         [KEPT]

Blockers: (1) gate/entry chronology (P13-384) — the selected strategy as
frozen cannot exist live; (2) legal renderings are either edgeless
(gated@63s +0.04R) or a different, weaker strategy (ungated +0.19R with
S/T selected under the gated regime and tap<=3 chosen post-hoc). Path
(Phase 14, user decision): build a FIRE-TIME confirmation model — PRE +
checkpoint features evaluated AT the R4 fire second (legal by
construction with the 3s buffer), walk-forward refit, then re-run staged
S/T selection on the legal stream and re-audit. Data-requirements note
for that phase: cp features need per-second volume and signed delta →
live feed must supply trades with aggressor side (or quotes to infer);
zone/tap/PRE state needs persistent cross-session storage; everything
else is OHLCV-derivable.

---
---

# PHASE 14 — leakage audit + legal fire-time strategy

Opened 2026-08-20. Registered §25.1. Objective: full-project chronology
audit (zero violations tolerated), then a completely causal R4v10
confirmation layer attacking the 55.7R regime-bleed baseline.

---

## P14-386 — Project-wide leakage audit                                 [KEPT]

Automated suite `tests/test_chronology.py` (permanent). Verified CLEAN:
front-month roll (prior-session volumes, seeded before window); ATR/ATR-
percentile (shift(1) + trailing rank, explicitly documented); session
vwap (running cumsum through t−1); SessionData ffill (causal); no
bfill/nearest/negative-shift anywhere on the live path; zigzag pivots
carry explicit confirmation indices and NO tap precedes zone
confirmation (data-level test over all 341,797 interactions); tap_no
monotone per zone; walk-forward fits strictly year<Y (2021 never scored;
sessions never straddle years → no embargo needed); no scalers/
calibrators ever fitted (HGB raw; nothing to leak). Centered rolling
windows EXIST in retired research layers only (events/axis/named/
profiles — P1-P3 families, none feeding the live path); e11 MBP-1
shift(-1) was exploratory only. **AUDIT FINDING #2 (corrected by exact
recomputation): legacy interactions layer counts an immature (<2h)
prior same-zone outcome on 1,554 taps (0.455%)** — my first diff-based
estimate (27,526) was inflated by a false-positive mechanism (an OLDER
tap crossing maturity legitimately increments while the latest gap is
<2h); the exact test is now the permanent one; legacy layer marked
dirty via strict xfail; the P14 dataset recomputes matured priors and
audits EXACTLY CLEAN (0 violations). Truncation invariance (delete the
future, require bit-identical features) passes on sampled sessions.

---

## P14-387 — Fire-time dataset                                          [KEPT]

`build_p14_fire.py` → p14_fire.parquet: **14,198 R4v10 fires** (all taps,
2021-26, resistance-role, n_fwd≥1800), 191 cols. Independent R4-second
recompute matches p9_timeline on ALL rows (0 mismatches). Convention:
bar x knowable at x+1; decision at fire-bar end; entry fills at the next
second's close (ck0) or later checkpoints (+1..+15s); features per
family (speed, path efficiency, penetration, acceptance, flow/delta,
absorption, regime, zone quality, matured recent-health) computed from
prefix arrays only; outcome label grids (8 stops × 8 targets first-
passage times) per checkpoint — labels may see the future, features may
not.

---

## P14-388 — Drawdown anatomy: the bleed is a POPULATION               [KEPT]

Baseline (ungated tap≤3 @fire+3): the 55.7R trade-DD = a 27-session
bleed (2026-03-31→05-08, open at data end); 2023-08 episode 44 sessions
down/143 to recover; underwater runs 142/54/33/30 sessions. Bleed-period
trades run **−0.303R vs +0.295R outside**. Descriptive discriminators:
recent50/recent20 health (−0.73σ/−0.60σ — but PURELY retrospective:
quintile expectancy of recent-performance features is FLAT → **recent
strategy health has NO predictive value**; Q18 answered NO); rv60/range-
used (bleeds live in low-vol compression). PREDICTIVE fire-time
features: **fire speed (fast-half +0.24..+0.37 vs slowest quintile
+0.01), monotonicity (top quintile +0.40), path efficiency (+0.28 top),
tap number**. The bleed population ≈ slow, choppy, late-tap fires:
anti-rule (tap≤3 & fire_s>41s) = +0.01R with 442-517-session underwater.

---

## P14-389 — Models vs mechanical rules                                 [KEPT]

Walk-forward (expanding yearly, train-quantile abstention, scaler-free
logistic + shallow HGB): AUC only 0.51-0.57 — top-30% abstention ≈
replicates tap≤3 quality, no transformation. Checkpoint grid: **ck0
(decide at fire bar, enter next second) strictly dominates**; +5/+10/+15s
add no discrimination and cost price (ck15: DD 74R). Simple mechanical
family (thresholds = 2021-23 quantiles: fire_s median 24s, mono median
0.27), frozen-evaluated 2024-26: **tap≤3 & fire_s≤24 & mono≥0.27:
OOS +0.278R, PF 1.43, day-DD 16.3R, uwMax 29** (selection era agrees:
+0.385, DD 22.2 — stable ordering). Plateau BROAD: fs∈{15..40},
mono on/off, tap∈{1..4} all live (+0.24..+0.43R) — no knife edges.
Hybrid (HGB top-70% within rule): +0.376R, PF 1.61, DD 16.1, uwMax 36
— the model earns a modest place on top of the rule.

---

## P14-390 — Staged S/T reselection                                     [KEPT]

On the rule population, ck0 entries, selection 2021-23 ONLY: pure-expR
pick S10/T40 (+0.373); DD-constrained pick **S12.5/T40** (+0.351, DD
22.3). Frozen OOS 24-26: S12.5/T40 → +0.376R PF 1.54 DD 20.3;
S12.5/T30 → +0.300 PF 1.47 DD 21.4. Neighbours all positive (S/T region
stable); S15/T30's shiny OOS (+0.333, DD 11.3) noted as a NEIGHBOUR
OBSERVATION ONLY — not selectable without OOS-peeking, not promoted.
The legacy 12.5/30 remains inside the plateau.

---

## P14-391 — Proofs                                                     [KEPT]

Truncation invariance: PASS (sampled rebuilds bit-identical).
**Streaming replay (per-second sequential reveal, incremental state,
rule decision at fire bar, fill next second): 12 sessions incl. the
2026 DD period, 137 R4 decisions, 30 takes — 0 mismatches vs the
vectorized dataset.** Chronology suite: 7 passed + 1 strict-xfail
(documented legacy dirt). Zero future-information violations in the
candidate path.

---

## P14-392 — Final comparison (2022-26, base cost, ck0 unless noted)    [KEPT]

all-fires ck2 12.5/30: n 11,926 (10.9/d) +0.117R PF 1.17 DDd 56.9 MC95/99
69.7/91.6 uw 239. tap≤3 ck2 (old baseline): n 5,672 (5.2/d) +0.191R PF
1.28 DDd 53.4 MC 42.4/55.4 uw 164, worstM −34.5. **REGISTERED PRIMARY —
tap≤3+fast24+mono, S12.5/T30: n 2,731 (2.5/d) WR 41.0% +0.310R PF 1.49
DDd 21.4 MC 22.8/29.5 uw 76 worstM −8.0 negM 7 stk 13, years +0.23..
+0.37, sessions all positive (ny-am +0.51, asia +0.16 weak-but-positive,
kept).** Operating points on the same plateau: S12.5/T40: +0.386 PF 1.55
DD 20.3 MC 26.6/33.3 (higher expR, longer underwater 96); tap≤2:
+0.353/PF 1.57/DD 15.0/MC 20.2/25.4/uw 47; tap1: +0.427/PF 1.71/DD
10.3/MC 16.5/20.6 at 1.24/d; hybrid+model70: +0.376/PF 1.61/DD 16.1/uw
36/negM 4 at 1.69/d. Verdict registered in §25.2: STRONG legal
improvement — expectancy ×1.6-2.3 over baseline, PF 1.28→1.5-1.7, DD
53→10-21R, MC p99 55→21-33R, underwater 164→36-76 sessions. Dataset-
reuse caveat stands: 2024-26 repeatedly inspected across phases; the
only true untouched test is live forward.

---
---

# PHASE 15 — economic comparison of the five LEGAL strategies

Opened 2026-08-21. Registered §26. Engine/rules from P12 (strategy-
agnostic); ALL streams are clean P14 ledgers; contaminated P10-12 dollar
expectations remain withdrawn.

---

## P15-393 — Ledgers verified; policy re-derived                        [KEPT]

Five ledgers rebuilt from p14_fire and verified vs §25.2 EXACTLY
(A 2,731/+0.310/1.49/DDd 21.4; B 2,716/+0.386/1.55/20.3; C 2,054/
+0.353/1.57/15.0; D 1,357/+0.427/1.71/10.3; E 1,847/+0.376/1.61/16.1 —
E on its own 2023+ axis: 2.20 t/d). Cushion knee RE-DERIVED on legal
streams: **C = 12R** (deaths 1.8→0.6/yr at zero net cost; 18R+ costs).
Saved p15_streams.pkl.

---

## P15-394 — Money results                                              [KEPT]

$500 (2 NQ), std $5k cap, C=12R, replacement $150/5d, N=1200 block-5
paths: **B $66.6k/yr (p10 $49.5k) > E $64.7k > A $63.2k > C $56.9k > D
$47.7k**; first payout med 17-24 sessions; deaths 0.18 (D) - 0.65 (B)
/yr. Money speed (median sessions to cumulative withdrawn): B fastest
everywhere ($10k by 51, $50k by 184). Chronological 2022-26 at $500:
ZERO deaths all five; totals B $315k > A $302k > C $281k > D $239k
(E $229k/4y); every year positive for every strategy; B wins 3/5 years.
Optimal-risk curves flatten past $500-600 (the $5k cap binds; avg payout
already $4.0-4.3k at $500): at $1000 E $75.1k ≈ A $74.8k > B $69.2k.
**DLL($3k)+promo($10k cap): B at $500 → $91.6k (+37 %), E → $82.1k, A →
$77k; at $1000 all ≈ $120k but DLL=3R intraday then interferes with the
strategy — registered recommendation caps risk at $500-600 where DLL
(6R) rarely binds.** Structural: with the 50 % rule + R-denominated
cushion, ACCOUNT-FAILURE PROBABILITY IS ALMOST SCALE-FREE IN RISK$
(deaths = post-payout 12R-cushion breaches + early 9R-initial-buffer
deaths); failure targets ≤5 %/yr are unreachable by sizing alone —
cushion is the lever (A $500: C 12→18→24R gives P(die 250d)
0.374→0.267→0.244 plateau; residual = early deaths). **Part-18 answer:
Ultra's 10R DD does NOT convert to more money** — it supports huge risk
only at strict failure targets nobody needs (deaths cost $150); at 20 %
target D $696 → $53k/yr, still < B at $500. Per-account at $500 (750d):
med 39-48 payouts, med cash $159-209k before any death; P(die before
payout 1) = 10 % (D) - 26 % (B) — the early 9R window is the real
failure mode; accounts that die later banked med $0-8k first (deaths are
overwhelmingly EARLY). Underwater (historical, sessions): B worst (top
96, share 73 %), E best (top 36, no 40+ runs), D lowest share (58 %).
Asia on A: keep = +$0.6k/yr, cut = −0.15 deaths/yr — a wash; registered
rule unchanged. Trades/day med 1, p95 ≤ 9, max 52-92 (tail-day caveat).

---

## P15-395 — Verdict                                                    [KEPT]

**WINNER (registered): B_t40 — tap≤3+fast24+mono, S12.5/T40, at
$500-600/trade (2 NQ / 20-24 MNQ), std path + DLL/promo caps → mean ≈
$91.6k/yr at $500 promo (std-cap $66.6k), p10 (std) $49.5k, first payout
med 17 sessions, deaths ~0.6-1.1/yr ($150 each), chronologically $315k
over 2022-26 with zero deaths.** It wins the same-risk test, the money-
speed test, the chronological test (3/5 years), and the optimized-risk
test at practical sizes. Runner-up E_hybrid: −3-10 % money with HALF the
underwater duration (36 vs 96 max) and deaths (0.34 vs 0.65) — the
bleed-averse alternative, explicitly offered. Same-failure-probability
test winner: D_ultra (only strategy reaching ≤10 %/yr) — but absolute
dollars still lose to B/E; low DD does not monetise under cheap
replacement. $500/2NQ classification: aggressive-but-economically-
rational for A/B (MC p95 ≈ $11-13k vs $4.5k initial MLL; 74-90 % of
accounts survive to payout 1 and then compound), moderate for C/E/D.

---
---

# PHASE 16 — Balanced independent audit + risk optimization

Opened 2026-08-21. Registered §27. Stage A: independent validation
(recompute everything, alternative implementations, accounting
invariants). Stage B only after CLEAN.

---

## P16-396 — Stage A audit                                              [KEPT]

A30: zero references to p_both60/F4/contaminated ledgers in the P14-16
path (grep-proven). A29: p14_fire regenerated from source — 14,198 rows,
byte-identical numerics to stored. **AUDIT FINDING #3: Balanced rebuild
gave 1,840 vs stored 1,847** — two mechanisms: (a) cross-process HGB
refit jitter at the top-70 % train-quantile boundary (same-process
refits bit-identical; fresh-process scores differ microscopically),
(b) same-second entry ties (41 groups / 118 rows) resolved by an
UNSTABLE sort in the one-open-trade pass. Fixes: **scores frozen to
disk (p16_balanced_scores.parquet + thresholds) and deterministic
stable tie-break (session, t_open, zid, tap_sec)** in portfolio().
Canonical rebuild = **1,847 trades, ELEMENTWISE identical to the stored
P15 ledger**; rebuild-twice identical; all five ledgers match P14/P15
stats to 4dp. Aggregate impact of the finding: none (WR/expR/PF/DD
matched to 4dp even at 1,840). A5: decision→entry lag exactly 1s on all
trades, 0 violations. A6/A4: chronology suite 7 pass + 1 documented
xfail; truncation invariance PASS. A7: 0 duplicate keys, 0 overlapping
positions. A8: price-level P&L recompute matches stored to 0.000000 pts;
2NQ: win gross $1,200/net $1,160, loss gross −$500/net −$540, RT $40.
A9/A25/A26: N 1,847, WR 0.4293, expR +0.3760, PF 1.610, DD 19.2R trade
/ 16.1R day, UW max 36 sessions (calendar defn: peak day excluded,
recovery day + no-trade sessions included), median run 3. A10:
accounting identity EXACT — replace mode booked = gross_paid + balance
+ forfeit ($0.000000 max residual, 500 paths); no-replace booked =
gross_paid + balance. A14/A15: 670 deaths / 2,000 path-years = 0.335/yr
(P15's 0.33 confirmed from raw counts). A16: fresh accounts 83.2 %
reach payout #1; 16.9 % die first. A17/A18: independent pure-python
REFERENCE simulator matches the vectorised engine on 20/20 lifecycles
(cash, payouts, deaths, booked — chronological + bootstrap, 2 risks ×
replace on/off); traces hand-verify all three payout branches
(cushion-bound $740, balance-bound $3,260, cap-bound $5,000) and death
arithmetic. A20: chronological $500: $229,312, 55 payouts, 0 deaths
(43.4k/51.0k/76.5k/58.5k by year) — matches P15 to the dollar via the
independent implementation. A21-A24: block-5 day bootstrap documented;
same-seed bit-identical; 5 seeds: mean ±$191, deaths ±0.009; converged
from N=1,000; stationary bootstrap (geometric mean-5) agrees ($64.8k vs
$64.7k). A12: $500 payout gaps med 13 / p90 23 / p95 27 / p99 45 / max
142. A13: worst UW episode (36 sess, 2024-09-11→11-01, 12.6R) contained
2 payouts/$9.0k at $500; longest no-payout stretch inside 30 sessions.
A3 model table: 2023←21-22 (4,177), 2024←21-23 (5,718), 2025←21-24
(7,921), 2026←21-25 (11,768); AUC 0.537-0.548; scores walk-forward;
SELECTION of hybrid/70 % used pooled scored years (no sealed holdout —
standing caveat). **VERDICT: CLEAN — INDEPENDENTLY REPRODUCED.**

---

## P16-397 — Stage B grid                                               [KEPT]

Canonical frozen ledger, C=12R, N=2000, $250-1500. Std $5k cap: rises
to ~$69k at $650-900, steps to ~$75k plateau at $1000-1500 (cap-bound;
deaths 0.75→1.39). Promo $10k+DLL: $82.7k ($500) → $101.3k ($650) →
$106.6k ($750) → $133.2k ($1300 peak) → $117k ($1400+, DLL=2.1R guts
the stream). Payout gaps ~invariant to risk (med 12-14, p90 20-23,
p95 27-29). Fresh-account survival is the binding axis: reach payout #1
= 90 % ($400), 84 % ($500), 73 % ($650), 67 % ($750), 50 % ($1100),
43 % ($1500); accounts dying at ≥$500 die EARLY (median cash-at-death
$0). Cushion sweep: 12R stays the knee (8R: +1.4 % money, 2× deaths).
Timing: immediate ≈ wait-3R/6R (±0.5 %); wait-12R breaks (payouts never
trigger). DLL interference: $500 → 4 days/0.5 % (nil); $750 → 19
days/+31.6R forgone; $1000+ → 53-134 days/95-169R (material). Cap-lost
share (promo): 17.5 % ($500) → 32 % ($650) → 68 % ($1300). Degradation
at $650 promo: 80 % edge → $87.0k (0.90 deaths), 60 % → $67.9k (1.73);
survives 40 % at reduced scale. Marginal churn (promo): 500→650 ≈
+$124k per added death/yr; 650→750 ≈ +$66k; 750→1300 ≈ +$59k — churn
stays formally cheap ($150/death) but fresh-account survival and DLL
interference deteriorate.

---

## P16-398 — Selection                                                  [KEPT]

Objective = money + repeated payouts + fresh-account survival knee.
**RECOMMENDED: $650/trade (26 MNQ or 2 NQ + 6 MNQ), promo/DLL account,
C=12R, immediate 50 % payouts → mean $101k/yr, p50 $99.8k, p10 $65.0k,
p90 $141.2k; 15.6 payouts/yr avg $6.5k; gaps 13/23/28 (med/p90/p95),
max 131; first payout med 19 sessions; deaths 0.48/yr; P(payout#1)
73 %; chronological 2023-26: $377.6k, 55 payouts, 0 deaths; 80 %-edge
$87k, 60 %-edge $68k.** Zone $600-750 flat (B20). Conservative $400;
aggressive-sustainable $750; max-EV $1300 (survival 50 %, DLL-heavy —
not recommended). The 36-session UW episode at $650 contained 2 payouts
($3,955 — chronology-dependent) with a 30-session internal gap: UW ≠
unpaid, but the worst stretch does slow payouts ~1 cycle.

---
---

# PHASE 17A — sealed holdout acquisition plan (NO DATA OPENED)

Opened 2026-08-15. Registered §28. Planning + freezing only.

## P17-399 — Cutoff, requirements, freeze                               [KEPT]

**TRUE RESEARCH CUTOFF = 2026-06-17T03:03:02Z** (latest raw trades
ts_event; the 2026-06-17 session is PARTIAL ~5h; every derived layer —
bars_1m/context/interactions/p14_fire/canonical ledgers — ends at that
session; model training reached 2021-2025, scoring 2026 through it).
Post-cutoff data existing today: 2026-06-18..2026-08-14 = **40 sessions
(~1.9 months) — FEWER THAN 6 MONTHS EXIST**; a 6-month holdout (127
sessions) completes **2026-12-17**. Requirements trace: Balanced needs
ONLY the Databento **trades** schema (1s OHLCV, volume, and signed delta
from the verified aggressor `side` field; zones/taps/R4/features/ATR/
vwap all derive from it; roll = prior-session volumes, causal, seeded).
**MBO not required. MBP-1 not required** (used only in retired e11
exploration; corpus MBP-1 ends 2025-06-30 anyway). Existing corpus:
GLBX.MDP3-style trades parquet, full NQ complex (spreads excluded by
'-'), prices in index points, ts_event/ts_recv UTC ns. Recent volume
8.6M rows/mo ≈ 0.41 GB/mo uncompressed (DBN 48B/rec). Expected Balanced
trades @2.20/session: 3mo≈141, 6mo≈279, 9mo≈418, 12mo≈563. **FREEZE:
HOLDOUT_FREEZE_MANIFEST.json (23 SHA-256 hashes: strategy/zone/R4/
feature/engine/ref-sim code, frozen scores efa8471d…, thresholds,
canonical ledger 5e5edf16…, streams, p14_fire, configs, chronology
tests; versions; no git — recorded). SEALED_HOLDOUT_REGISTRY.json +
empty sealed dirs created OUTSIDE the glob-ingested bars_1s/daily trees
(builders glob those paths — documented hazard). No holdout market
records downloaded, opened, charted, scored, or summarized.** Databento
cost API not runnable here (no SDK/key on machine): exact request
provided to user; size-based estimate only.

## P17-400 — PRE-REGISTERED holdout evaluation (BINDING)                [KEPT]

Hypothesis A (PRIMARY): frozen Balanced net expectancy > 0 on untouched
data at base costs. Test: session-clustered bootstrap 90 % CI of mean
net R. σ_R = 1.680; at N≈280 SE≈0.115R incl. clustering. **CLEAR PASS:
N≥200 AND mean net R ≥ +0.15 AND PF ≥ 1.20 AND CI lower bound > 0**
(a true +0.376 edge fails this <3 % of the time; a zero edge passes
<10 %). **BORDERLINE:** mean in (0, +0.15) or CI straddles 0 or N<200 —
extend with HOLDOUT 2, no verdict. **CLEAR FAIL:** mean ≤ 0 at N≥200,
or PF ≤ 1.0, or (maxDD > 24R [dev MC p99] AND mean < +0.10). Secondary
(reported, non-vetoing): WR expected [38.0, 47.8] %; streak ≤ 14; UW ≤
45 sessions; ≥3/6 months positive; expectancy at +0.5pt slippage > 0.
What 6 months CANNOT establish: distinguish +0.20 from +0.376 (both
PASS). Hypothesis B (SECONDARY, after A): frozen $650/26MNQ prop
implementation replay — reported, never the pass criterion. **RULE:
once HOLDOUT 1 is opened, any Balanced modification demotes it to
development data; only HOLDOUT 2 or later data may then validate.**

---
---

# PHASE 17B — sealed holdout opened: FROZEN BALANCED FAILED

Opened 2026-08-15T21:59:22Z (registry first-open). One evaluation, zero
retuning. Registered plan §28; user authorized early opening of the
partial window as HOLDOUT A.

## P17-401 — Integrity, proofs, result                                  [KEPT]

**Integrity:** batch GLBX-20260815-69WTYRKTDB = the exact registered
query (GLBX.MDP3 / trades / NQ.FUT parent / 2026-06-17T22:00Z →
2026-08-15); 53 files, ALL manifest sha256 verified; freeze manifest 23/23
hashes unchanged before opening. Converted DBN→corpus schema (tick-grid,
aggressor chars, symbol mapping verified). **Pipeline proofs: June
overlap bars OHLCV identical to dev (788,994 rows); context/ATR
recompute 0.000000; RTH-range crosscheck 0.0 ticks; MECHANICAL warmup
ledger (2026-03-02..06-16) 392/392 trades bit-identical to dev.** Sealed
chain fixes were loader-scope only (hardcoded B1S_DIR/layer_dir
bindings, my own RTH-window bug); zero strategy changes. Data note:
2026-06-17 RTH exists in neither corpus (dev cut 03:03Z, purchase starts
22:00Z) → one NaN in the ATR chain, handled by rolling min_periods.

**AUDIT FINDING #4 (freeze-design flaw): the HGB model binary was never
persisted** — P17A froze scores, not the function. Cross-process refits
are NOT bit-reproducible (thread-order FP jitter cascades via early
stopping): today's recipe-refit agrees with the frozen dev-2026
selection only 91.1 % (kept-set Jaccard 0.904). Resolution: refit binary
now persisted (holdout_model_refit.pkl); Balanced evaluated via the
frozen recipe with disclosed fidelity, AND the fully-frozen mechanical
core evaluated as the unambiguous reference. Any future live deployment
must persist model binaries at freeze time.

**RESULT (HOLDOUT A = 2026-06-18..2026-08-14, 40 sessions):**
Balanced: N 140 (3.50/sess vs dev 2.20 — fire rate UP), WR **27.9 %**
(dev 42.9), expR **−0.133R**, PF **0.83**, total −18.6R, maxDD 25.1R
(t)/23.0R (d), UW 32 sess, streak 10, months −7.5/−9.7/−1.4R (all
negative). Clustered 90 % CI [−0.341, +0.061], P(mean≤0)=0.873.
**Historical-window test: 0th percentile for expectancy, PF and WR
against ALL dev 140-trade windows (dev p5 window = +0.159R); DD 100th
percentile.** Payoff mechanics intact (avg win +2.320, avg loss −1.080)
— the failure is pure hit-rate collapse. Sessions: all negative except
Asia (+0.44, n 29) and premkt (+0.13) — diagnostic only, NO subsetting.
Cost stress and outlier exclusion only worsen it. **Mechanical core
(no model, 100 % frozen): N 167, WR 28.7 %, −0.103R, PF 0.87** —
confirms the edge itself failed, not the model layer. Prop replay
(2 NQ/$500, frozen P16 config): 12 winning days/36, **0 payouts, $0
withdrawn, 2 account deaths** (2026-07-02, 2026-07-30).

**Classification per pre-registered framework: FAIL** (PF materially
<1; performance far outside every historical window; both ledgers
agree). N=140 < the registered 200 → the FAIL is called on the
0th-percentile window test + PF, with sample size disclosed. The break
begins exactly at the research boundary — the signature of selection
overfitting on the repeatedly-reused 2023-26 development data, the
standing caveat since §25. What 40 sessions cannot establish: the true
long-run expectancy (CI upper +0.06) or whether this one summer regime
is representative. What they DO establish: the dev-estimated +0.376R
magnitude did not generalize to the first genuinely unseen sample.

**Per the no-retune rule: Balanced was NOT modified. HOLDOUT A is now
spent — for any future modified strategy it is development data. Any
reopening of strategy research is a separate authorized phase; only
genuinely later data (HOLDOUT 2, 2026-12-18+) can validate it.**

---
---

# PHASE 18 — failure autopsy + robust redesign

Opened 2026-08-15. HOLDOUT A reclassified as development/diagnostic per
the standing rule. Future data (>2026-08-14) remains sealed and untouched.

---

## P18-402 — STAGE A AUTOPSY                                            [KEPT]

**Layer 0/1 (zones, contacts, raw reactions): SURVIVED — STRENGTHENED.**
Contacts/session flat (46.5 vs 46.3); reaction rates UP (P(rev10) 0.965
vs 0.948 dev-2025+; med t_rev10 86s vs 150s). **Exact-location control:
real-vs-shadow gap LARGER on holdout (+0.201±0.019 vs +0.146±0.006 —
P(rev10≤300s)).** The resistance phenomenon is real and persisted.

**Layer 2 (R4v10): event intact, continuation premium THINNED.** Fires/
contact ≈ doubled as a secular 2025+ trend (fires/sess 10.4 all-dev →
17.2 dev-2025+ → 19.3 hold; filter pass-shares stable; rule signals
4.05/sess dev-2025+ vs 4.28 hold — the "frequency explosion" vs the
2.20 dev-average was mostly era drift already inside dev). Post-fire
P(30fav before 12.5adv): 0.357 → 0.305 vs bracket breakeven 0.318;
ALL-fires expR@12.5/30: +0.140 (dev-2025+) → −0.042 (hold). Margin was
always razor-thin; it crossed zero.

**Layers 3-5 (tap/fire/mono): INVERTED.** Holdout: tap gradient gone
(tap1 −0.021); fire 0-8s = WORST bucket (−0.143) vs dev best (+0.230);
high-mono = WORST (−0.217) vs dev best. **The frozen rule cell (−0.126)
underperformed its own complement (−0.018)** — filters were adversely
selective out-of-sample. Cell SEs ±0.14R (individually 1-2σ; the JOINT
inversion of three independent filters + rule<complement is the
evidence). Loss taxonomy composition IDENTICAL dev vs hold
(55/22/15/8 vs 55/24/14/7 immediate/reaction/near-miss/slow) — no new
failure mode, uniform hit-rate collapse.

**Layer 6 (HGB): unstable.** Dev yearly increments +0.145/+0.079/
+0.065/+0.066 (shrinking); holdout increment −0.030. Reject ML
complexity going forward.

**Layer 7 (12.5/30): fragile amplifier.** Reactions still reach 30pts
eventually (0.881) and faster (293s), but P(12.5 adverse ever) rose to
0.952 — the tight stop in a faster-reverting tape is tagged by noise.

**A12 session stability: none** (Asia dev eras +0.59/−0.49/+0.05/+0.19;
NY-AM was the STABLE dev pillar and broke worst on hold −0.25). **A13
regime analogues: NEGATIVE** — dev sessions with holdout-like fire
density scored +0.29..+0.40 on the rule; no observable causal state
explains the failure; a regime filter is NOT supported. **A14
microstructure: unremarkable** (1s range 5.6t vs 4.3t, volumes similar;
same contract NQU6 both sides of the boundary).

**A15 selection audit:** final-stage explicit variants on 2023-26 ≈
200+ (P14 plateau 40 + models/checkpoints/retentions ~60 + S/T 64 +
strategy pick of 5 in P15/P16), atop concept selection from thousands
of earlier comparisons (R4v10 itself chosen among ~30 P9 rules; zones
from P4-P7). Cross-variant σ ≈ 0.05-0.08R ⇒ winner's-curse inflation ≈
+0.10-0.20R. Observed dev-vs-reality gap ≈ 0.4R ⇒ selection inflation
explains roughly half; genuine decay of the filter relationships and
the R4 continuation margin explains the rest.

**STAGE A VERDICT: E — MIXED**, precisely: resistance + exact location
SURVIVED (strengthened); R4 survived as an event but its continuation
premium under the tight bracket decayed to ~breakeven (B-component);
the tap/fire/mono/HGB confirmation stack was selection-overfit and
inverted OOS (C-component, largest single contributor); the 12.5/30
bracket's thin margin amplified everything (D-component).

---

## P18-403 — STAGE B HYPOTHESIS REGISTRY (registered BEFORE results)    [KEPT]

Data: DEVELOPMENT CORPUS = everything through 2026-08-14 (holdout A
included as an ordinary block). Nested walk-forward: outer eval blocks =
9 half-years 2022H1..2026H1 + block10 = 2026-06-18..08-14; train =
strictly prior data; any parameter choice uses train only (coarse grids,
plateau required); aggregate outer-fold results only. Controls: parent
rules + shadow-location benchmark; cost stress +0.25/+0.5/+1.0pt.

**H1 — contact-level fade (no R4).** Rationale: layer 0/1 is the
strongest surviving edge; trade the LOCATION not the velocity event.
Entry = first touches (tap small) at near edge; outcome via p6 t_revX /
depth_before_revX. Coarse: tap≤{1,2,3} × S {20,25,30} × T {20,30,40}.
Persistence rationale: location differential vs shadows survived every
era incl. holdout.

**H2 — R4 wide-bracket.** Rationale: fires still reach 30pts (88%) but
tight stops die to noise; wider stop restores path tolerance. All-taps
and tap≤3 only. S {15,17.5,20,25} × T {30,40} from ck0 grids. No
fire/mono/HGB filters (autopsy-dead).

**H3 — pullback/retrace entry after R4** — REGISTERED, NOT EXECUTED
(entry-fill machinery absent from current grids; budget discipline).

**H4 — timed exits** — REGISTERED, EXECUTE ONLY IF H1/H2 fail (needs a
builder pass for timed exit prices).

Hypothesis budget: 2 executed families, ~21 coarse cells total, both
with plateau + worst-fold discipline. Failed cells will be reported.

---

## P18-404 — STAGE B results (ALL cells reported)                       [KEPT]

**H2 (R4 wide-bracket), 16 cells:** entire tap≤3 surface positive in
10/10 blocks (worst folds +0.00..+0.06); all-taps 9-10/10 but thin.
Nested walk-forward: pooled outer +0.193 (median +0.179, worst 0.000,
10/10 positive; +0.5pt stress mean ≈ +0.16). Confirms the autopsy: with
S≥15-17.5 the R4 reaction survives its own noise. Viable but dominated.

**H1 (contact fade), 27 cells (9 untestable — no T40 depth column):**
with the fill-realism rule (limit at near edge, requires ≥2 ticks
trade-through — every counted trade genuinely fills): every evaluable
cell positive in 10/10 blocks; nested pooled +0.343 (worst fold +0.200).
**B9 control — the decisive test: the SAME fade rule on shadow anchors
earns ≈0 (−0.17..+0.16 by block) while real zones earn +0.20..+0.63;
real-minus-shadow edge = +0.26..+0.50 in EVERY block.** The surviving
location layer converts directly into a tradeable rule with no R4, no
speed/shape filters, no model.

**Attempt count (B17): 2 registered families, 43 cells (34 evaluable),
1 fill variant, 2 nested procedures, 1 shadow control. Every cell
reported above; none deleted. H3/H4 registered-not-executed.**

---

## P18-405 — CANDIDATE FREEZE: FADE-1                                   [KEPT]

**Definition (sha256 8b969a07…, outputs/p18_fade1_freeze.json):** frozen
resistance-zone methodology unchanged; FIRST touch of a zone only
(tap_no==1); entry = limit at near edge, fill requires ≥2 ticks
trade-through; stop = edge +20.0 pts; target = edge −30.0 pts; cost
1.0 pt RT; one trade per zone-session; one-open-trade; all sessions.
Selection criterion (pre-stated): max worst-fold on the coarse grid,
plateau required, simplicity preferred. Neighbours all robust (S25/S30,
T20, tap2 cells: worst folds +0.02..+0.12).

**Outer-fold profile:** 10 blocks 2022H1..2026-08: N 4,494 (3.86
trades/session), WR 55.7 %, +0.342R pooled, PF 1.73, fold mean +0.352 /
median +0.321 / worst +0.200, 10/10 positive; +0.5pt stress worst fold
+0.175; block DDs 7-25R (at S=20 denominator); former-holdout block
+0.626 (descriptive only). Median hold ~30 min.

**Standing caveats (mandatory):** (1) HOLDOUT A was development data
here — FADE-1 has ZERO external validation; (2) the H1 outcome model is
grid-approximated (p6 rev/depth machinery; losses forced to −S,
no-reaction = full loss — conservative-leaning, but a production 1s
ledger pass + chronology tests + streaming replay are REQUIRED before
any live consideration); (3) the hypothesis family itself was motivated
by the autopsy, which used holdout data — the nested procedure protects
parameters, not family choice. **Future validation (registered): the
next sealed window (2026-08-15 onward, ideally ≥4 months, target ≥300
fills) evaluated ONE-SHOT under P17-400-style criteria: PASS = mean net
R ≥ +0.10 with clustered 90 % CI > 0 and PF ≥ 1.15; FAIL = mean ≤ 0 or
PF ≤ 1.0; else extend. No parameter may change before that test.**

---

## P18-406 — ADVERSARIAL RAW-DATA AUDIT of HOLDOUT A                    [KEPT]

Mandated stance: assume the June-Aug 2026 batch is bad; try to prove it.
No strategy was run; no profitability used as evidence.

Inventory: **51 market .dbn.zst files** (+3 job files = 54 on disk;
manifest's 53 = 51 dbn + metadata + condition; prior "52/53" language
reconciled) — exactly the 51 expected non-Saturday UTC dates 06-17..
08-14, 0 missing, 0 duplicates; all manifest sha256 verified; **all 51
decode fully (16,207,061 records, 0 failures)**. Request metadata =
registered query exactly. **Independent struct-level DBN parser (no
Databento lib): 0 field mismatches vs official decoder** (180 sampled
records, 3 files). Price encoding: int64×1e-9, 0.25 grid on outrights,
120 sampled raw→decoded verified (off-grid samples = spreads, 0.1 % of
rows, excluded by design). Duplicates: full-record 0.58 %, shared-seq
12.9 % — **identical rates in the dev corpus (0.47-0.85 %/11-16 %)** =
upstream MDP3 multi-fill property, not a defect; 0 backward timestamps.
Contracts: NQM6/NQU6/NQZ6/NQH7 + spreads; front unique per session, all
NQU6 through the holdout; June roll 06-16 per the causal prior-volume
rule (volumes confirm); **0 cross-contract front bars**. Completeness:
front trades/day mean 316.7k ≈ dev Mar-May 323.2k; "low" days = Sundays
+ the by-design 06-17 evening slice; the 33-day >30-min-gap pattern =
exactly the CME maintenance-halt schedule (weekdays minus Fridays).
**Independent 1s-bar reconstruction (own parser + own filter impl from
yaml constants): every joined bar identical — maxΔprice 0.0000, maxΔvol
0 — on holdout sessions AND dev-era controls (2022, 2024)**; ref/proj-
only bars fully explained by file-day vs session windowing (Friday = 0
ref-only confirms). Sessions: 0 window violations; 8 known-time probes
land correctly; no DST transition in window. Old-vs-new raw overlap:
**none exists BY DESIGN** (purchase starts 2026-06-17T22:00Z; old raw
ends 03:03Z) — boundary continuity plausible (372.5 pts over 19 h).
ATR: exact under the frozen convention (**0.000000 ticks**; my audit's
two interim deltas were my own convention/indexing mismatches —
documented; the 06-17-RTH-absent NaN averages 13/14 deterministically
in both). VWAP: 0.00000 at 40 taps. R4 inputs: **0/120 mismatches**
from an independent dense-ffill low-walk. Zones: **60/60** pivots match
an adjusted bar extreme (8 were flipped pivot-LOWS, by design).
External tslab reference covers only ≤2026-06-16 (no holdout overlap;
skipped per rules). Audit artifact: outputs/p18b_audit_sessions.csv.

Hypotheses: H1-H10, H12 REJECTED with evidence; H11 untestable by
construction (no overlapping observations exist), no contrary evidence.

**DATA VERDICT: CLEAN — no material data or pipeline problem found.
The hypothesis that Balanced failed because the new Databento batch was
bad is REJECTED.** The P17B/P18 strategy conclusions stand.

---
---

# PHASE 19 — resistance → support flip (LEVEL research only)

Opened 2026-08-16. Zone engine FROZEN. No confirmation research, no
entries, no SL/TP, no prop math. Data through 2026-08-14 (holdout A =
development); future sealed.

## P19-407 — STAGE A: flip definitions PREREGISTERED (before outcomes) [KEPT]

Break/acceptance family, all evaluated on the SAME frozen zones with
original band [near_edge, far_edge] unchanged; acceptance detected from
frozen per-tap observables of the breaking resistance-side tap (and a
1m-bar pass for A). A zone becomes SUPPORT-ELIGIBLE at the end of its
qualifying break tap; the PRIMARY event = FIRST subsequent approach
from above (app_side_above==1 tap of the same zone).

* FLIP A — close acceptance: first 1-minute close above the far edge.
* FLIP B — distance +5: traded ≥5 pts beyond the far edge
  (depth_max_full ≥ width + 20t).
* FLIP C — distance +10: ≥10 pts beyond (width + 40t).
* FLIP D — time acceptance: ≥60 s spent beyond the far edge
  (dwell_beyond_full ≥ 60).
* FLIP E — ENGINE-REGISTERED: the frozen engine's own cat="F" decisive-
  acceptance flip (close-run beyond far edge per capped_state_label;
  2h-maturity display convention noted — for this research the flip
  TIME is the F-tap end, causal).

Outcomes measured with the frozen sweep machinery (side_above=1 rows of
p6_real: t_rev{5..30} = UPWARD reversal from the zone reference;
depth_before_rev = adverse penetration below). Controls = the
registered P6 shadow anchors keyed per real tap (paired by construction).
Primary metric: P(+10 within 300s/900s) and P(+X before −Y), real vs
shadow, chronological half-year blocks. Success standard per the brief
(location superiority, block stability, plateau, usable frequency).

## P19-408 — RESULTS: flip does NOT create support                      [KEPT]

Analysis correction (documented): my first pass left-joined p6 outcomes
onto all flip rows — p6 measures only deduped representative taps (43 %
coverage) — so unmeasured rows read as "no reaction" and depth units
carried a ×4 error. All results recomputed on measured rows with fixed
units; the corrupted table is retained above for the record.

**Corrected findings (first retests, all five registered definitions):**
(1) Flip conditioning is INERT — flipped-zone first retests react at
the same rate as ALL above-side taps (P(+10) 0.80 vs 0.80; +10<5m 0.16
vs 0.18) and all five acceptance definitions (A/B/C/D/E) give
near-identical results (plateau of nothing). (2) Support-side reactions
remain structurally weaker than resistance-side (same frozen machinery:
+10 within 5m 0.16 vs 0.48; median t_rev10 1,542s vs 266s; +10-before-
−10 0.25 vs 0.73; median MAE ~20pt vs ~0) — P8's "support is diffuse"
replicates at flipped locations. (3) **Exact location FAILS: registered
symmetric-paired shadow contrast on the honest time-boxed metric =
−0.0087 [90 % CI −0.0124, −0.0051]** (vs +0.146..+0.201 for resistance
in P18-402); per-block Δ −0.01..−0.05 in every year incl. the holdout
era (descriptive). The +0.227 "ever-reach" gap is the registered local-
extreme tautology and is not evidence. (4) No tap decay (0.17 flat taps
1→4+), no time-since-break structure, no depth structure that isn't
outcome-contaminated (shallow-touch retests ARE bounces). (5) False
breaks: sweep-aboves reject downward only 37 %; 62 % later break
properly — acceptance filtering changes nothing downstream anyway.
(6) Frequency would have been ample (~30 first retests/session; 71 % of
zones eventually B-accept) — supply was never the problem; validity is.

**VERDICT: RESISTANCE → SUPPORT FLIP IS NOT SUPPORTED.** Broken-
resistance retests behave like generic pullbacks in an advance: high
eventual bounce rates with no promptness, no location specialness, no
definition sensitivity. The zone engine remains a ONE-SIDED
(resistance) instrument; the registered P8 support conclusion stands.
Integrity: 5 acceptance definitions (preregistered), 0 entry/
confirmation/SL-TP/prop research, no future data touched.

---
---

# PHASE 20 — resistance-zone productization study

Opened 2026-08-16. Zone engine = the product; no strategy work.

## P20-409 — freeze, workload, delivery                                 [KEPT]

**Freeze:** outputs/p20_resistance_engine_freeze.json — standalone
implementable spec (zigzag 0.15/0.35/0.75-ATR scales on front 1m;
confirm at reversal completion; hw = clip(base10min/2, 4t, 0.03·ATR);
lifecycle 30 sessions/6 taps/flip; taps on 1s with max(8t,hw)+900s
re-arm; prior-volume roll; ATR14 prior-RTH shifted) + 9 code hashes.

**Workload (2025-01..2026-08-14, 406 sessions):** the raw engine is a
FIREHOSE: 68 unique zones approached/day, 26 first touches/day, 150
taps/day, ~130 engine-episode "distinct" taps/day; 82-90 active zones
above price at any moment; nearest resistance within 10pt 53 % of the
time, within 25pt 81 %; proximity alerts ≈ 70-80/day at ANY threshold
10-75pt (median warning 74-178 min, ~70 % eventually touch). Zone width:
median 14.6pt full (mean 15.9, p25 11.6, p75 19.0) — the "~14pt" memory
confirmed. Tap mix: tap1 18 %, tap2 16 %, tap3 17 %, tap4+ 49 %. Ages at
first touch: 79 % same-session. Segments (MDO/day): ny-am 31.5, ny-pm
31.9, asia 30.0, london 20.6, premkt 15.7; first touches ny-am 6.9/day.
STRUCTURAL narrowings (not outcome-tuned): first touches only = 26/day
(NY-AM 6.9, med 4); first touches of scale≥1 (0.35+ATR pivots) =
4.6/day (NY-AM 1.0/day, zero-days 54 %); scale-2 only = 0.8/day. Two
script bugs caught+fixed during analysis (dev/sealed zid collision;
quick-timeline liveness omission) — final numbers from the corrected
run.

**Delivery:** Pine mapping — zone GEOMETRY (1m zigzag/ATR/width)
approximately-to-exactly reproducible; **tap counter/re-arm (1s
hysteresis) and 30-session persistent lifecycle NOT safely reproducible**
(TV 1s history/limits; ~14 sessions of 1m lookback on Premium; state
lost on reload) and **contract-roll/back-adjustment differs from the
frozen causal chain (basis gaps ~300pt at the June roll)** → pivots
near rolls can move or vanish. Verdict: TradingView exactness = NO
without a parity test; Python/Databento remains SOURCE OF TRUTH.
Parity-test design registered (100 zones; PASS ≥95 % identity, edges
≤1 tick, creation ≤1 bar; expected failures: roll windows, tap-count
drift). Recommended build: **Python zone/alert service** reusing the
frozen engine verbatim (feed → zones → proximity/first-touch alerts,
scope configurable: all first touches ≈26/day or scale≥1 ≈4.6/day);
notification channel = user's choice.

---
---

# PHASE 21 — resistance-zone spacing audit

Opened 2026-08-16. Frozen zones only; no outcomes, no changes.

## P21-410 — the map is HIGHLY REDUNDANT                                [KEPT]

Causal 5-min snapshots 2025-01..2026-08-14 (liveness = formed, ≤30
sessions, <6 prior taps, unbroken). **Edge-to-edge gaps of adjacent
live zones: 79 % OVERLAP; median gap −12.5 pts** (p25 −17.4, p75 −3.1);
only ~11 % of pairs sit >10pt apart. 88.4 % of zones overlap ≥1 other,
**51.4 % entirely contained in another**; mean max stack depth ~10
(extreme 427 on the Apr-2025 crash range). Nearest-five: price→Z1 med
31.8pt, then Z1→Z2..Z4→Z5 gaps −5.9/−10.6/−11.4/−12.2 — the "next few
zones" are one merged mass. Coverage: 31/41/46/43 % of the next
50/100/200/500 pts inside a box; walls med 28.7pt (p90 74; 31 % ≥40pt);
clean gaps within 200pt: med 12.9pt. **Collapse: ~70 live zones → 15.2
overlap-regions (4.6 zones/region) → 9.1 at ≤10pt; within 100 pts: 5.4
raw → 1.8/1.4 regions.** Event collapse: 68 approaches/day → 5.2
overlap-regions (≤10pt 2.9); 26 first touches/day → 6.2 (≤10pt 4.0);
**NY-AM 7.6 raw first touches → 2.9 overlap-regions (≤10pt 2.1)**.
First-touch-eligible inventory: 27 zones → ~9.3 regions (57 % gaps
overlap). Scale structure: shares 82/15/3 %; cross-scale: **100 % of
scale-1/2 zones have a same-session overlapping lower-scale zone**
(42 %/9 % same-pivot ±10min); unfiltered per-scale spacing: scale-0 med
gap −18pt/99 % overlap, scale-1 −13.6/87 %, **scale-2 +6.8pt/40 % — the
only naturally separated tier**. Example snapshots show 6-8 boxes
stacked across single 25-50pt spans. Diagnostic-only clustering used
fixed thresholds {0,5,10,15,20}pt registered a priori; no outcomes
touched. Two implementation notes: per-scale table computed on
unfiltered inventory (levels ~6× live; ratios valid); main snapshot
liveness intra-session exact.

**[P25A NOTE 2026-08-16: near-price numbers in this entry are
SUPERSEDED — the liveness filter backdated break times (audit finding
#7). Corrected causal values in P25A-415: children within 25/50/100pt
+71/+45/+24 %, nearest-region distance med 19.9pt (was 31.9). Totals
(~70 above, 15.2 regions) and the redundancy classification stand.]**

**CLASSIFICATION: HIGHLY REDUNDANT** (not near-continuous: distinct
regions exist — clean 13pt median gaps, identifiable 15-region
structure — but raw display would be 5-10 boxes deep). Phase-20
abundance restated: the engine finds ~5-6 distinct resistance REGIONS
per day being approached, not 68; NY-AM ≈ 2-3 distinct first-touch
regions/day. Recommended next: KEEP ENGINE, ADD VISUAL CLUSTERING/
DISPLAY LAYER (overlap-union regions with contributing-scale badges);
no engine change authorized or performed.

---
---

# PHASE 22 — resistance region display design

Opened 2026-08-16. Presentation layer over the FROZEN engine; geometry
only; no outcomes; engine untouched.

## P22-411 — region-layer diagnostics + design                          [KEPT]

Overlap-connected parent regions (transitive, no gap bridging), causal
snapshots 2025-01..2026-08-14. **Geometry:** 15.2 regions above price
(med 11); width med 28.7pt / p75 43.4 / p90 73.8 (reproduces P21
exactly); zones/region mean 4.6 (30 % singletons, 8 % stacks of 11+,
max 428); **unique pivots/region 4.0 — zones:pivots ratio 1.16** → the
stacking is mostly DISTINCT neighboring pivots congregating, not just
multi-scale copies. **Confluence:** max depth med 3 (p90 7); CORE-2
(≥2-zone coverage) present in 70 % of regions, median longest band
22.0pt inside a 28.7pt parent; CORE-3 in 54 % (19.0pt); CORE-4 36 %.
Cores/region: 0/1/2/3+ = 30/50/13/7 %; **wide parents are chains: >50pt
regions 73 % multi-core, >75pt 85 %** → cores must be displayed.
Scale composition: 51 % scale-0-only, 27 % all-three, 22 % {0,1};
nearest-region max scale s0/s1/s2 = 62/26/11 %. **Nearest region:** med
31.9pt away, 25.2pt wide, CORE-2 longest 10.7pt, 2 zones/2 pivots.
**NY-AM view:** 15.4 regions above (1.8 within 100pt, 0.5 within 25pt);
nearest width med 27.6pt, core 12.4pt — genuinely clean. 69 % of
regions contain ≥1 untouched zone (mean 1.8). **Alert collapse (from
P21 events):** 68 zone-approaches/day → ~5.2 region entries; 26 raw
first-touch alerts/day → ~6.2 first-touch-region events (definitional
note: clustered over different populations); NY-AM 7.6 → ~2.9.
**Part P: 0 derivation violations** — region bounds provably min/max of
child zones; cores from coverage sweep only; no invented prices.
**[P25A NOTE 2026-08-16: nearest-region and near-price stats
SUPERSEDED by P25A-415 (break-state backdating, finding #7): nearest
width med 29.2pt (was 25.2), nearest CORE-2 15.0pt (was 10.7), NY-AM
regions within 25pt 0.7 (was 0.5). Region totals/width/composition
stand. Also: this script double-loaded 2026-H1 bars (dev+sealed),
double-weighting those snapshots — both rules equally; fixed in
run_p25a_impact.py.]**
Examples: outputs/p22_example_regions.csv (incl. a 37-zone stack →
one labeled region). Recommended design (fidelity-based, no outcomes):
**DISPLAY B** — light parent rectangle + emphasized CORE-2/3 bands +
child edges on demand; default label
`R 20270-20323 | 6Z/4piv | Smax1 | 3 untouched | core 20291-20306`;
region-level deduped alerts carrying exact child-zone state; read-only
layer over the frozen engine; lineage = component-membership continuity
(same-member growth ≠ new region; splits inherit by larger member
share). Engine modified: NO. Outcomes used: NO. Future data: NO.

---
---

# PHASE 23 — mechanical entry backtests around frozen zones

Opened 2026-08-17. Five registered families, fixed windows/conventions
(logged in run_p23_build.py docstring), no mining beyond the stated
grid. Data ≤2026-08-14; engine untouched.

## P23-412 — ALL FIVE ENTRY FAMILIES NEGATIVE                           [KEPT]

Ledger p23_trades.parquet: 2,461,359 trades over 1,405 sessions (+
492,741 geometry rejects where the confirmation closed at non-positive
reward — itself evidence of lateness). Conventions: 1s fills; bar
entries at completed-bar closes only; SL=U+H, TP=L−H; stop-first ties;
1.0pt RT cost; one concurrent trade per zone; region-deduped second
ledger (overlap components).

**A (user's 50 % midpoint, 1:1):** raw n 126,589 (90/day): WR 49.3 %
vs breakeven 52.1 % → expR −0.047, PF 0.91. Deduped identical
(−0.046). Best single cell anywhere = tap-1 midpoint −0.002 (zero,
and it is the best of 12 diagnostic cells). NY-AM −0.065.
**B engulfing (BODY/RANGE × in-zone/post-5min × 30s-5m):** ALL 20
variants negative, raw and deduped (−0.015..−0.21). Entry quality
explains it: engulfing closes are below the zone 49-51 % of the time;
RR<0.5 on ~half; median RR 0.49-0.56. Higher TFs raise WR to ~70 %
while RR collapses to ~0.33 — lateness, not confirmation.
**C rejection-close:** WR 73-78 % but median RR 0.30-0.40 (entry below
zone 100 % by construction) → −0.03..−0.04 everywhere.
**D sweep+reclaim:** D1 (close back in zone) keeps RR ~1.4 but WR 40-43 %
→ −0.06..−0.07; D2 (close below) = C-like, −0.02..−0.04.
**E touch-low break:** WR 79-85 %, median RR 0.15-0.27 → −0.03..−0.04.
**Yearly: negative in EVERY year for every key variant** (2025-26
closest to zero, never positive). Dedup changes nothing (no duplicate-
inflation illusion). Tap/scale splits: no positive pocket.

**Mechanism (consistent with P13/P14/P18):** the zone reaction happens
in the first seconds-to-minutes after touch (median rev10 86-275 s);
midpoint fills select adversely (price 0.5H deep is already through
half the zone), and every bar-close confirmation spends the reaction
before entering. The only historically positive construction remains
edge-proximate entry with a wide stop (FADE-1 geometry, P18) — NOT
re-optimized here.

**VERDICT: none of these entries are good enough; nothing from this
phase merits preregistration.** Chronology: 0 violations (completed-bar
entries, causal fills, stop-first, zones pre-confirmed per standing
tests). Ledger saved for audit.

---
---

# PHASE 24 — direct near-edge entry backtest + FADE-1 audit

Opened 2026-08-17. Registered grid (7 stops × 8 targets), two fill
rules, raw+region-dedup, tap variants; 1s first-passage outcomes;
cost 1.0pt. Ledger outputs/p24_direct_entry_trades.parquet (347,116
rows); surfaces outputs/p24_surface_*.csv; summary p24_summary.json.

## P24-413 — direct entry surface: NEGATIVE under realistic fills       [KEPT]

REGION-DEDUP results: **B2 (2-tick trade-through, the realistic rule):
0/56 cells positive for BOTH all-taps and tap1** (key cells −0.023..
−0.029; S20/T30 tap1 = −0.026, folds 3/10 positive, worst −0.082; all
years ≤ +0.00). B1 (perfect touch fill) all-taps 0/56; tap1 17/56
marginally positive ONLY at long targets (T40-60: +0.01..+0.024, PF
1.01-1.03) — economically nil and fill-optimistic. Survival matrix
(tap1): P(−20 before +30) = 0.58; MAE(1h) med 38.2pt, p90 129pt —
direct edge-entries suffer large adverse excursions; the location edge
(fast partial reactions) does not convert to fixed brackets at the
edge. Direct entry beats P23's midpoint (−0.03 vs −0.05) but remains
negative. No post-hoc parameters added; every cell reported.

## P24-414 — AUDIT FINDINGS #5/#6: FADE-1 INVALIDATED                   [KEPT]

Part-U reproduction FAILED → stop-and-audit executed. (#5, minor:
P18's fill filter o_depth_max_full≥2t is future-conditioned with entry
backdated to the touch — but quantified harmless: 98.3 % qualify, fill
delays are seconds, backdated wins 0.6 %.) **(#6, fatal): P18's H1
outcome model (p6 sweep-taxonomy t_rev30/depth_before_rev30) UNDERSTATES
ADVERSE EXCURSION. Event-level reconciliation on P18's exact trade set
(5,248 matched): agreement 87.7 %, ALL 645 disagreements one-directional
(P18-win but the true 1s path hit the 20pt stop first) = 22.4 % of P18
wins were losses; median recorded depth among "wins" = 0 ticks while
the raw path shows P(adverse≥10pt)=83 %.** True-path FADE-1 (exact
conventions): −0.025R sim / −0.045R forced-loss model vs P18's +0.342.
P18's own registered caveat ("grid-approximated, conservative-leaning
— production 1s ledger pass REQUIRED") is hereby exercised: the pass
ran, and **FADE-1 is WITHDRAWN as a candidate. The §29/P18-405 freeze
is void; the planned one-shot test of FADE-1 is cancelled.** The
"conservative-leaning" characterization in P18-405 was wrong — the
approximation was optimistic on the adverse side. NOTE: P18's H2
family (R4 wide-bracket, tap≤3, ck0 entries, S15-25/T30-40; nested
pooled +0.193, worst fold 0.00) was computed on TRUE first-passage
grids and is NOT affected — it remains the only surviving positive
construction, unpromoted and unvalidated.

**Phase-24 verdict: DIRECT ENTRY REJECTED; FADE-1 dead; the honest
frontier of this project's tradable constructions = H2-class wide-
bracket R4 entries (+0.19 development, never externally validated) and
the zone/region display product.** Integrity: engine untouched; future
data sealed; grid fixed a priori; 0 look-ahead in P24 machinery (which
CAUGHT the P18 look-ahead-class defect).

---
---

# PHASE 25 — production resistance engine (PAUSED)

Opened 2026-08-15, paused 2026-08-16 by user instruction before
completion. Built: src/nqlevels/prod/{engine,regions,snapshot}.py
(frozen call-graph imports, zero re-typed logic), historical CLI,
parity harness. Snapshot parity 5,633/5,633 + restart 12/12 PASSED on
the first gate run; STREAMING failed 3/3 → root-caused to AUDIT
FINDING #7 (below) — the streaming/truncated path was the CAUSALLY
CORRECT side. Fix completed in P25A; full gate now PASSES (see
P25A-415). Remaining build work intentionally NOT resumed.

---
---

# PHASE 25A — causal break-state audit + clean-reaction measurement

Opened 2026-08-16. Causal correction + impact audit + first-passage
measurement ONLY (no new strategy search, no optimization, no live
infra). Data ≤ 2026-08-14; engine untouched.

## P25A-415 — AUDIT FINDING #7: broken-state backdating          [KEPT]

**The defect (found by the P25 streaming-parity gate):** every
product-layer "active/broken" computation (P20 inventory, P21/P22
snapshots, prod snapshot()) marked a zone broken from the TAP TIME of
its first breaking tap, but the breaking observable o_depth_max_full
is measured tap→end-of-session — so intraday state used penetration
that had not happened yet. Zones were deleted from the live map
minutes-to-hours before the break was knowable.

**Corrected causal rule:** a zone is broken at the FIRST 1-second bar
(at/after a resistance-side tap) whose high reaches
zhi + (zhi−zlo) + 20t; before that bar it is live. Threshold, tap
stream, geometry, pivots, re-arm: UNCHANGED (verified — the fix is a
presentation/state-layer timing correction; engine outputs are
byte-identical).

**Quantification (product universe, dev+sealed ≥2026-06-18):** 53,520
ever-breaking zones; 99.6 % had backdated break times; gap tap→cross
med 31.9 min, p75 1.9 h, p90 5.3 h (holiday-bridge sessions reach
40 h); 390.2M mislabeled zone-seconds; **64.6 % of 5-min product
snapshots 2025-01..2026-08-14 contained ≥1 mislabeled zone state.**
Verification: 53,520/53,520 stored breaks have a real 1s crossing
(0 anomalies — also cross-validates o_depth_max_full against raw 1s);
independent-path first-crossing check 249/250 exact (the 1 = seam
artifact of the verifier's own unfiltered sealed load, explained);
production parity gate rerun with the causal rule on BOTH sides:
snapshots 5,725/5,725, restart 12/12, streaming 3/3 — **FULL PASS**
(outputs/p25_production_parity_report.json). Sealed-corpus state now
uses full tap history (2026-01+) for break state; events remain
date-partitioned (dev ≤06-17, sealed ≥06-18).

**Impact on P20-22 product numbers (old-rule columns reproduce the
logged values, so attribution is exact; outputs/
p25a_causal_break_impact.json):** region-level display INTACT —
regions above 15.2→15.3, NY-AM regions 15.4→15.4, within-100pt
1.7→1.9, region width 28.7→29.3pt, zones/region 4.6→4.6. NEAR-PRICE
picture materially corrected (the old rule deleted precisely the
recently-penetrated zones near price): children within 25pt 0.8→1.4
(+71 %), 50pt +45 %, 100pt +24 %; regions within 25pt +49 %; nearest
region distance med 31.9→19.9pt (−38 %), nearest width 25.2→29.2pt,
nearest CORE-2 10.7→15.0pt. Event workload: the published P20/21 rates
never applied the broken filter and are REPRODUCED by the causal rule
(FT children 26.16/day vs 26; FT regions 6.14 vs 6.2; NY-AM 6.88 vs
6.9, NY-AM FT regions 2.60); an old-rule-filtered product would have
wrongly suppressed ~80 % of events (e.g. 4.8 vs 26 FT/day).

**Exact-location claims: UNAFFECTED.** P6-321 (+7.7pp strict
resistance-side, 6/6 years), P18-402 (holdout gap +0.201 vs dev
+0.146), P18 B9 shadow control — none consume the broken flag
(verified by code audit: the flag exists only in P19-22/prod/P25
scripts), and the engine tap streams are unchanged, so those results
stand byte-identical. (P18 B9's magnitudes remain separately
superseded by finding #6.) Real resistance still beats matched
shadows: YES, unchanged.

## P25A-416 — CLEAN FIRST-PASSAGE: REACTIVE, NOT CLEAN            [KEPT]

Preregistered measurement (run_p25a_clean_reaction.py docstring):
child events = all 30,818 resistance-side first touches
2021-01..2026-08-14; region events = 4,786 first encounters of
entirely-untouched causal overlap components (member-set dedup);
ref = child near edge (== first traded in-zone price; contact-bar
open differs in 6.1 % gap-entries); 1s first-passage, 60-min window,
ties adverse-first; 0 look-ahead violations (checked).

**The matrix (REGION_FRESH; CHILD within ±2pp everywhere):**
+5/−5 53.0 %, +10/−5 **35.2 %**, +10/−10 52.5 %, +15/−10 41.8 %,
+15/−15 51.4 %, +20/−10 35.0 %, +20/−20 49.4 %, +30/−15 32.4 %,
+30/−20 37.1 %, +40/−20 29.5 %, +60/−30 22.3 %. Every symmetric pair
≈ coin flip; every asymmetric pair within ~±2pp of its cost-free
breakeven A/(F+A) (+10/−5 be 33.3 %; +15/−10 be 40 %; +20/−10 be
33.3 % → actual 35.0 %), i.e. ≈ 0 edge before costs, negative after
1pt RT. Yearly win rates flat 2021-2026 (+10/−10: 49-55 %) — the
behavior is stationary, consistently marginal. NY-AM fresh-region
events 0.80/session: +10/−5 35.3 %, +10/−10 54.5 %, +15/−10 42.9 %,
+20/−15 46.5 % — slightly tidier, same conclusion.

**Why old "reaction" numbers misled: ordering.** Eventual +10 within
60 min = 82-83 % (P18's session-window P(rev10)=0.95-0.97 is the same
family) — but only 35 % deliver +10 before −5. MAE before an eventual
+10: med 6.3pt, p75 14.4, p90 30.8 (before +20: 9.3/21.1/40.5;
before +30: 11.0/25.7/48.7). Median penetration before +10 = 41 % of
region width; before +30 = 63 %. Reactions either turn inside the
first child band (75 %) or blow through the whole region first (24 %).
Taps 2/3/4+ are 2-5pp WORSE than tap 1. Scale 0/1/2 and region
max-scale: flat (no cleaner tier). Categories (region): CLEAN 35.2 %,
ACCEPTABLE 13.5 %, FAILED 27.6 %, MESSY 16.2 %, OTHER 7.6 %.

**Verdict: POOR FOR TIGHT-STOP TRADING** (consistent with P23/P24
trade-level negatives — this is the event-level mechanism: adverse
excursion arrives first roughly as often as chance). Zones mark real
locations (exact-location lift stands) but do NOT produce clean
rejections. Product decision recorded in the P25A report: **KEEP ONLY
MINIMAL HISTORICAL/LOCAL VIEWER** — no alert/live build, no Phase 26.
Artifacts: p25a_{break_crossings,events,events_latertap}.parquet,
p25a_clean_reaction_matrix.csv, p25a_clean_reaction_nyam.csv,
p25a_child_vs_region.csv, p25a_mae_before_target.csv,
p25a_causal_break_impact.json, p25a_summary.json.
