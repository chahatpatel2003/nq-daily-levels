# Research Plan — NQ Daily Levels

**Goal.** Produce 2–4 genuinely strong NQ price zones each morning before the
06:30 PT cash open, for the 06:30–09:00 PT window. Not every common level
plotted at once — a short list of prices where the data says something is
likely to happen.

**Status.** Phase 1 complete (foundation + audit). **Phase 2 complete — the
central hypothesis is rejected.** No method tested identifies which NQ prices
behave as strong zones once distance and side are held fixed. Two things
survive: the E1 destination baseline (calibrated touch probability) and a
universal first-touch decay. Full results in `EXPERIMENT_LOG.md`; the summary
of what was tried and what it cost is in §11 below.

**The holdout was never read.** Nothing qualified for a confirmation test.

---

## 1. The two problems are not one problem

The brief separates them and the data agrees, for a structural reason worth
stating precisely because it determines the whole evaluation design.

### 1.1 Destination — will price get there?

**Touch probability is monotone in distance.** If price reaches 21,700 it has
necessarily passed 21,690. So `P(touch)` for any price above the open is
exactly the survival function of the session's maximum upside excursion, and
*any* level type will show a high hit rate simply by sitting close to the
market. The Phase-1 demo shows this cleanly on four sessions:

| distance from reference | touch rate |
|---|---:|
| 0 – 0.25 ATR | 92 % |
| 0.25 – 0.5 ATR | 62 % |
| 0.5 – 1.0 ATR | 36 % |
| 1.0 – 2.0 ATR | 2 % |
| > 2.0 ATR | 0 % |

A "71 % touch probability" claim is therefore almost content-free on its own.
The quantity to model is **attraction**: touch in excess of the
distance-matched, regime-matched baseline. `levels/axis.py` fits that baseline
on training sessions only and every destination claim is scored against it.

Corollary: the useful destination question is mostly *ordering and timing* —
which of two zones is reached first, how long it takes, whether the far one is
reached at all on this kind of morning — not the marginal probability.

### 1.2 Reaction — what happens when it gets there?

No monotone structure: behaviour at 21,700 does not follow from behaviour at
21,690. This is genuinely local, and it is where the edge most plausibly lives.
It is also the part the earlier NQ program never studied — that program tested
*entry rules* and found the friction wall, not *level quality*.

Consequence: the two are modelled separately, with different targets, different
baselines and different sample definitions. A zone can be an excellent
destination and a terrible entry, and the output should say which.

---

## 2. What "strong" has to mean, quantitatively

The predecessor program's central finding is a hard constraint here: on NQ,
1-minute-visible signals produced a gross edge of \$3–8 per trade against
\$10–20 of friction, across six strategy classes. Its one positive result came
only from **asymmetric exits with wide targets**.

So a level is only *useful* if it supports a move large relative to friction.
A zone that "reacts" 6 points is noise after costs. The reaction surface is
therefore evaluated at 10/15/20/30 points, not at 5, and the headline label is
conjunctive: a real move **and** not achieved by loitering.

This does not narrow the search space — it sets the bar the search has to
clear, which is the lesson the brief asks to be carried forward without
inheriting the old strategy assumptions.

---

## 3. Data architecture

```
L0  raw parquet (read-only)         trades 547 M rows · MBP-1 7.89 B rows
      |
L1  bars_1s/     1-second OHLCV + buy/sell volume        71.7 M rows, 1.2 GB   [BUILT]
    profiles/    volume-at-price, 30-min buckets, by tick 11.6 M rows,  67 MB  [BUILT]
      |
L2  daily/       one row per session: pre-open references,
                 volatility scales, regimes                                     [next]
      |
L3  candidates/  zones known at 06:25 PT — named families + price-axis grid     [next]
      |
L4  labels/      touch outcomes + reaction outcomes                             [next]
      |
L5  micro/       MBP-1 reductions — touch-time confirmation only                [phase 4]
```

Design decisions and why:

* **1-second, not 1-minute.** Tap-and-go versus chop is unmeasurable at
  1-minute resolution, and that distinction is the project's core requirement.
* **Profiles in 30-minute buckets keyed by tick.** Any window's profile — prior
  RTH, overnight, London, a 10-day composite, a rolling 3-session profile — is
  a groupby-sum, with no second pass over 547 M prints. Every PT segment
  boundary the research uses lands on a 30-minute mark, so no bucket straddles
  one.
* **Both contracts materialised across a roll**, so prior-day levels are
  recomputed in the incoming scale rather than shifted by carry.
* **MBP-1 reduced exactly once.** At ~1 hour per full pass it cannot be part of
  an experiment loop. Trades, at ~4 minutes per pass, can be.
* **Everything derived lives on T9** (661 GB free); the internal disk has 32 GB.

---

## 4. Candidate generation — two generators, compared

### 4.1 Named families (the baseline to beat)

Prior session: PDH, PDL, PDC, prior RTH open/mid/VWAP, prior POC/VAH/VAL,
prior Globex high/low. Current night: overnight high/low/mid/VWAP, overnight
POC/VAH/VAL, and Asia / London / premarket extremes **kept separate** because
they form at different hours against different participants. Longer horizon:
prior week and prior month high/low/close, multi-session swings. Derived:
round numbers per step size (25 / 50 / 100 / 250 / 500, separately — so the
research can find that 100s matter and 25s do not), gap edges, VWAP deviation
bands, HVN/LVN from several profile windows.

Coincident levels are collapsed into clusters that **record** `n_kinds` and
`n_families` without scoring them. Confluence is a hypothesis to test, not a
bonus to hand out.

### 4.2 Algorithmic price-axis discovery (the general method)

Score *every* price on a grid (±250 points, 0.5-point step) with features
describing that location: distance and volatility-normalised distance; traded
volume, neighbourhood-relative volume, local gradient and volume share across
several windows; historical touches, seconds-at-price, visit length, recency;
round-number geometry; and distance to each named family. Then model
`P(touch)` — as attraction residual — and `P(clean reaction | touch)`
separately.

Named levels enter this model as *features*. If "PDH" carries information the
axis features miss, that column earns weight; if it does not, it does not, and
that is the answer. This is the formulation the brief asks for and it is
allowed to beat the named generator outright.

---

## 5. Zone width

Width is fitted, not chosen. `levels/width.py` compares: fixed ticks (control),
ATR fraction, overnight-range fraction, realised-vol scaled, **empirical
penetration quantile per family**, and volume-node width.

The deciding measurement is the empirical penetration distribution — for each
family, how far past the reference does price actually go on a visit — fitted
on training sessions only. The report is a *curve* of touch rate and reaction
quality against width, not an argmax: a sharp optimum is a fitted artifact, a
plateau is a property of the market.

---

## 6. Experiment families, in order

Each gets a pre-registered hypothesis, a sample-size floor, a
distance/regime-matched baseline, year-by-year and regime breakdowns, and
parameter-sensitivity. Entries go in `EXPERIMENT_LOG.md` whether they succeed
or fail, so the same dead end is not re-explored.

**E1 — Distance and excursion baseline.** The null model. Empirical
`P(touch | distance, side, regime)` and the session maximum-excursion
distribution. Everything else is scored against this. *Nothing to prove; this
is the yardstick.*

**E2 — Zone width study.** Penetration distributions per family; the
width/quality curve. Fixes the width used by E3 onward.

**E3 — Do named families beat their own distance baseline?** Per family,
attraction residual and clean-reaction rate with confidence intervals, by year
and by regime. Expect most to fail. The point is to find the two or three that
do not, and to have the negative result on record for the rest.

**E4 — First touch versus later touches.** Whether reaction quality decays with
touch number, penetration history and elapsed time. The brief flags this and it
is cheap to test; if first touch is materially better, every later phase
conditions on it.

**E5 — Reaction-quality model.** Given a touch, predict clean rejection.
Interpretable baselines first (logistic on a handful of features), gradient
boosting only if it clearly beats them out of sample, calibrated probabilities
throughout. This is the most promising branch because it has no monotone
structure to fight.

**E6 — Price-axis attraction model.** The general destination model, scored on
attraction residual against E1, and against the named generator head-to-head.

**E7 — Confluence, properly controlled.** Whether overlapping *independent*
structures beat the best single family, after controlling for the fact that
overlapping levels are correlated by construction and that a cluster is
mechanically closer to the market than a lone level.

**E8 — Regime conditioning.** Gap size and direction, overnight range versus
ATR, volatility percentile, open location versus prior value, day of week. Kept
deliberately coarse: fine regime definitions multiply hypotheses fast, and a
finding that survives only inside a tuned regime is usually a finding about the
regime definition.

**E9 — Macro-event conditioning.** See §8.

**E10 — Selection and ranking.** Turning per-zone scores into the 2–4 zone
morning output: how many to emit, the threshold below which the answer is
"nothing today", and whether correlated zones should be de-duplicated. Scored
on the whole morning, not per zone.

**E11 — Microstructure confirmation** (MBP-1, ≤ 2025-06-30, strictly a
touch-time feature). What separates a level that holds from one that fails, at
the moment of the touch. Book *events* — adds, cancels, replenishment rate,
absorption — rather than displayed size: NQ's top of book averages ~3
contracts, so size alone is unlikely to say much.

**E12 — Reclaim.** Only after E5/E11. What a successful reclaim actually looks
like, before distilling it into one mechanical rule. Explicitly *not* starting
from "one candle closes back above".

---

## 7. Validation discipline

* **Chronological only.** Train 2021-01-05 → 2024-06-30 (871 sessions),
  validation 2024-07-01 → 2025-12-31 (369), holdout 2026-01-01 → 2026-06-17
  (110). Five-session embargo at each boundary.
* **The holdout is sealed in code.** `holdout_sessions()` raises without a
  pre-registration token whose hash matches a frozen file. The predecessor
  program consumed its entire holdout and then had nothing left to confirm its
  one promising result on; that must not repeat.
* **Walk-forward on train+validation** for anything selected, so a result has
  to be persistent rather than lucky in one regime.
* **2021–2022 is a robustness set, not a training set** — a bear market and a
  very different volatility regime.
* **Multiple testing is tracked.** The number of hypotheses tested is recorded
  in the log; with a price-axis grid it is easy to test thousands of things
  accidentally. Preference goes to findings that are persistent, simple, stable
  across adjacent parameters, large-sample and economically sensible — in that
  order, ahead of raw effect size.
* **Correlated candidates are not independent samples.** Four zones within 10
  points of each other that all "worked" on 2024-03-06 are one observation of
  one move, not four. Clustering by session and by price proximity is required
  in every confidence interval.

---

## 8. Macro events

Two independent channels, established in Phase 1:

**Endogenous (no external dependency, no leakage risk).** The main US release
slot is 05:30 PT, **55 minutes before the decision cut**. The release and the
market's reaction to it are therefore fully observable pre-cut. Measured
across 502 sessions, one-minute volume at 05:30 PT runs 2.2× baseline at the
median and 23× at the 95th percentile, and the spike distribution reproduces
the release calendar unaided — 36.6 % Thursday, 35.6 % Friday, 1.0 % Monday.
Median one-minute range on spike days is 79.8 points versus 9.5 otherwise. So
"was there a big macro event this morning, and how violently did the market
take it" is a pre-open feature that needs no feed at all.

**Exogenous (schedule only).** What the tape cannot supply: *which* event it
was, and whether one is scheduled **inside** the 06:30–09:00 window — 10:00 ET
= 07:00 PT, which shows a real if smaller signature (median 1.26×, max 5.3×).
That is genuinely forward-looking and needs a calendar.

Sources, from the Phase-1 feasibility check: **BLS API v2 works with no key**;
FRED/ALFRED needs a free key and is the right source for point-in-time
*vintage* actuals (never revised values); federalreserve.gov serves the FOMC
calendar; **Forex Factory returns 403 and is not usable**, as the brief
suspected.

Leakage rule: the *schedule* is known in advance and is always a legal
pre-open feature. The *actual* and the *surprise* are legal only after the
release timestamp — which for 05:30 PT releases means they are legal at 06:25,
and for 07:00 PT releases means they are not. Vintage data only; a revised CPI
print is not what traders saw.

Not a blocker: E9 runs after E5, and the endogenous channel already covers most
of the value.

---

## 9. Known risks

1. **Correlated candidates inflating apparent significance.** The biggest
   statistical risk, and visible already in the Phase-1 demo where one move
   produced four "successful" zones. Mitigated by session-level clustering and
   price-proximity de-duplication in every interval.
2. **Multiple testing across a price grid.** Mitigated by pre-registration,
   the sealed holdout, and preferring plateaus to peaks.
3. **Attraction may not exist.** It is entirely possible that, conditional on
   distance, no level type predicts touch. That is a publishable negative and
   would redirect all effort to reaction quality, which is the better bet
   anyway.
4. **Reaction may exist but be too small.** 8-point clean reactions are real
   and useless. The outcome surface is evaluated at 10–30 points for this
   reason.
5. **MBP-1 stops 2025-06-30.** No microstructure result can be confirmed on the
   2026 holdout. Any such rule needs 2026-H2+ data as it accrues.
6. **Regime shift.** The predecessor's 2026-Q2 tail showed trigger rates
   collapsing versus prior years. 2026 is a different market (NQ ran from
   ~23,000 to ~31,000 in H1); results that only hold pre-2025 should be
   treated as regime-specific.

---

## 10. Deployment — deliberately undecided

Not chosen yet, by design: the research determines the architecture. If the
final signal needs only OHLCV-derivable levels, a Pine indicator is enough. If
it needs traded-volume profiles it needs Python. If it needs book events it
needs a live feed. No paid service, subscription or hosting is assumed by
anything built so far, and none will be recommended without first writing down
the expected value it would add.


---

## 11. Phase 2 outcome (2026-08-14)

### What was tested

| experiment | scope | verdict |
|---|---|---|
| E1 distance/excursion baseline | 1,370 sessions, 6 model variants | **KEPT** — skill +0.39, ECE 0.014 |
| E2 zone width | 9 widths × 4 scaling families, stratified | no width reveals an edge |
| E3 named families | 26 core kinds, FDR-controlled | **REJECTED** — none survive |
| E3b higher timeframe | prior week/month, 5-60 session extrema | rejected, underpowered |
| E4 first vs repeat touch | 59,807 touches, within-zone paired | **KEPT** — but universal |
| E5 feature models | logistic + HGB, 41 features | rejected after distance matching |
| E11 pre-open MBP-1 | 100 sessions, 7 book features | **REJECTED** — AUC 0.45-0.49 |
| regimes | 12 regime buckets | rejected |
| target sensitivity | 10 reaction definitions | rejected |

### The three confounds that had to be removed

Each one produced a large, convincing, false positive first:

1. **Level density.** Pooled over all 86 generated kinds, a random price is
   within 1.3 points of "a level" — so a shifted control is another level and
   the contrast is null by construction. Fixed by restricting to a sparse
   26-kind core (median gap 8.4 points).
2. **No matched control exists.** At a given distance and side there is exactly
   one price. Fixed by a uniform price-axis grid compared *within*
   distance-and-side strata.
3. **The extreme-price tautology.** "The farthest price touched" is the day's
   extreme, so price necessarily reversed there — 88.6 % clean vs a 10.5 % base.
   Any ranker correlated with distance inherits it. Fixed by selecting
   pre-open from all grid prices and only then looking at outcomes.

### What a Phase 3 could still try

Ranked by remaining plausibility, not by appeal:

* **Prior-week high/close.** The one family with a consistently positive
  direction (d_clean +0.064 train, +0.070 validation) that was merely
  underpowered — 0.2-0.3 % axis coverage. Needs a pre-registered test at
  higher power, not another sweep.
* **A different outcome entirely.** Everything here measured *reaction*. Zone
  behaviour might be more predictable as a volatility or dwell-time response
  than as a directional one.
* **Cross-asset or cross-instrument context** (ES, VIX, rates) — untouched,
  and the only genuinely new information source available without new data.
* **Accepting the destination-only product.** E1 is real, calibrated and
  deployable now; it simply is not a levels product.

What should **not** be tried again: named level families in any form, volume
node structure, order-book history, or another learned ranker over the same
feature space. Those are settled.

---

## 12. Phase 3 — conditional reaction ranking (opened 2026-08-14)

### 12.1 The reframing

Phase 2 answered a *universal* question: "is a price at level-family X better,
on average, than a matched non-level price?" — and the answer is no. Phase 3
asks the **product** question, which is weaker and different:

> Every morning, rank the candidate zones available at 06:25 and select the
> top 2–4. Do the *selected* zones react meaningfully more reliably than the
> realistic alternatives that could have been selected that morning?

A family can be flat on average and still contain a rankable conditional
subset. Phase 2 only rules out the average.

**What carries over unchanged:** the decision cut, the sealed holdout, the
three confounds (density, no matched control, extreme-price tautology), the
noise floor |d| < 0.005, edge-anchored reaction measurement, session-clustered
inference, and train-explore / one-look-validation discipline.

**The honest benchmark** is P2-109 §3: rank pre-open, select top-K, compare
against a control drawn *at the same distance band, same side, on the same
morning*. Geometry then cancels by construction. With Phase-2 features that
contrast was +0.013/+0.014/−0.006 (K=2/3/4) — indistinguishable from zero.
Phase 3 succeeds only if new features move it by a stable, repeatable margin.

### 12.2 What is genuinely new (and why it is not a rerun)

Phase 2's per-price history features counted **visits** (touches, minutes-at,
recency, visit length) and **volume**. None of them measured how price
*reacted* — the event of arriving somewhere and leaving fast. The new feature
families are all conditional properties of a specific area:

* **P3-201/202 Reaction memory.** Detect historical rejection events (local
  extremum + confirmed excursion away ≥ θ·ATR) on the full corpus, then give
  each candidate price its decayed history of past events: count, magnitude,
  velocity, consistency, direction mix, recency, invalidation. Includes
  same-morning overnight events, recomputed from pre-cut bars only.
* **P3-203 Displacement origins (supply/demand).** Zones where a confirmed
  impulse launched, scored by origin strength: displacement size/velocity,
  volume, delta, base duration, structure break, freshness/mitigation depth.
* **P3-204 Conditional named levels.** Not "is PDH good" but "is PDH good
  *when untested overnight / fresh / formed by a sharp rejection*" — within-
  family conditional contrasts on the existing measured axis.
* **P3-205 Macro-release structures.** On tape-detected release mornings
  (05:30 PT, 55 min pre-cut), the release spike high/low/mid/VWAP as zones.
* **P3-206 Prior-week levels at pre-registered higher power** — the one
  Phase-2 family with a consistent positive direction, still unresolved.
* **P3-207 Ranker + selection.** Only if individual families survive: a
  calibrated P(clean | touch, distance) combined with the frozen E1
  P(touch | distance) into an expected-utility ranking, evaluated end-to-end
  in the top-k framework.

Cross-roll price mapping uses per-roll spreads measured from the bars_1m
overlap windows (median minute-close difference), cumulated into an epoch
adjustment table. Memory features record whether their window crosses a roll
so carry error is checkable.

### 12.3 The top-k/day evaluation framework (P3-200)

`eval/topk.py`, used by every Phase-3 experiment:

* **Selection**: per session, top-K by score over all candidates, with
  min-separation de-duplication and an optional distance window. Selection
  sees nothing measured after 06:25.
* **Controls**: distance-matched same-morning draws (headline), random
  same-morning, next-ranked (K+1..2K), nearest-K, core-level-K.
* **Metrics**: per zone — touch rate, P(clean | touch) under several
  definitions, joint P(touch ∧ clean), reaction magnitude, penetration,
  slice-through, chop; per morning — P(≥1 selected zone reached),
  P(≥1 reacted), P(≥2 reacted); calibration when the score is a probability;
  all by year and by volatility regime.
* **Inference**: paired session-clustered bootstrap on model-minus-control.

### 12.4 Success criterion

A methodology is promising if, on validation, the selected top zones beat the
distance-matched same-morning control on joint touch∧clean (or on
clean | touch at matched touch rate) by a margin that is stable across years,
survives parameter neighbours, and is large enough to matter after the noise
floor — as a working bar, a repeatable ≥ +5 pp on P(clean | touch) or an
equivalent joint-rate lift, not a hair above zero. Report the practical
numbers: zones/day generated, selected, reached; reaction rates top-1..4;
magnitudes; chop and failure rates; calibration; per-year stability.

---

## 13. Phase 3 outcome (2026-08-15)

### 13.1 What was tested and what happened

| experiment | verdict |
|---|---|
| P3-200 top-k framework + estimator tests | KEPT — every claim reports through it |
| P3-201 rejection-event layer (13,080 events) | KEPT infrastructure |
| P3-202 reaction memory | REJECTED — sign reversed: recently-active areas are *consumed* (clean −1.5 pp, slice +1.8 pp); overnight-event areas are energy, not quality |
| P3-203 displacement origins / freshness | REJECTED for cleanliness (centre-only +6 pp was small-sample noise; full coverage +0.5 pp n.s.); energy structure real |
| P3-204 conditional named levels | untested prior-day support: train +8.6 pp CI>0, **validation FAIL** (0.072 vs 0.073 raw) — a 2021-2023 phenomenon, gone |
| P3-205 macro-release structures | attraction real (+5..8 pp touch, all kinds); cleanliness null-to-negative |
| P3-206 HTF extremes (pw/pm) | pooled train +3.9 pp CI>0, **validation FAIL** (+1.2 pp n.s.); `pwh` alone positive in every cut ever measured — the one holdout-eligible candidate |
| P3-207 assembled engine | KEPT — see §13.2 |

**The converging empirical law of this dataset:** conditional on distance and
side, *location* features (what price is there) predict movement **energy**
around an area — attraction, excursion, slice-through — and do **not** predict
directional rejection **quality**. Every apparent quality effect either failed
honest controls, failed power, or failed 2024-2025. The 2021-2023 regime
genuinely contained conditional-quality effects (untested support +9 pp,
support-side HTF extremes); the 2024-2025 market does not have them.

### 13.2 The frozen methodology proposal (for the eventual holdout test)

**Engine (frozen):**
1. E1 destination model, unchanged (frozen since P2-101).
2. P(clean | touch): logistic on `dist_atr, dist_atr², above, above×dist,
   atr_pctile, on_range_rel_atr, |gap|, log1p(macro_volx), loc_in_pdr`,
   monthly-expanding refit, no location features (P3-207 variant A).
3. Score = P(touch) × P(clean | touch); eligibility 0.15–1.0 ATR; 6-point
   minimum separation; emit top K ≤ 4 (fewer if candidate list is short).
4. Display annotations per zone: P(reach by 09:00), P(clean | reach),
   fp-race rates from trailing data, expected rejection magnitude.

**Validation-measured performance to beat on holdout (2026): joint
P(touch∧clean) ≈ 0.090/zone at K=4 vs matched 0.072 (+1.8 pp, CI > 0), cond
+2.3 pp, calibration ECE ≈ 0.01/0.003, P(≥1 of 4 reaches) 0.92, P(≥1 gives
15-before-10) 0.65.**

**Optional pre-registered add-on, separate gate:** `pwh` cleanliness
(+4..+7 pp in every prior cut, never validated alone). Register exact rule
(±6t, clean_10_5, side×5-band strata) before the holdout read.

### 13.3 What the product honestly is — and is not

With no location-quality signal in the current regime, the optimal 2–4 zones
degenerate to calibrated near-band placements (±0.15–0.2 ATR) with honest
probabilities — a *distance-and-context* product, not a *level-identity*
product. It is well-calibrated, its zones are relevant (92 % of mornings at
least one is reached), and roughly half of reached zones win a 15-before-10
race; but a display implying "this exact price is special" would claim
something the data rejects. Any deployment should present zones with their
probabilities and nothing more mystical.

### 13.4 Not exhausted (for the record)

Release *identity* (which event, via BLS/FRED vintage) refining the null
release-structure result: untested, low prior. Touch-time microstructure
confirmation (E11 proper): out of pre-open scope by design, MBP-1 ends
2025-06. Multi-timeframe *refinement* of a family that itself shows no signal
cannot create signal and was not pursued.

### 13.5 Sweep/reclaim amendment (2026-08-15, P3-208)

The reaction-definition audit changed the frozen proposal in one place only:
**the outcome the selection is judged (and annotated) on.** The engine is
unchanged.

**Frozen taxonomy** (P3-208): R = 15 pts from the near edge; d0 = 4t; sweeps
by range, reclaims by 60 s-held closes past the near edge; post-reclaim
reaction within 30 min, anchored at the near edge; resolved ≥ 1800 s. Cells:
strict (D = 24t, T = 900 s), mid (D = 0.05 ATR, T = 900 s), primary
(D = 0.10 ATR, T = 1800 s). A = direct, B = sweep+reclaim, C = failure;
y_any = A ∪ B.

**What changed empirically:** location identity still predicts nothing
(every significant family contrast under y_any is negative), but the frozen
engine's matched-control selection edge is 2-3× larger under sweep-aware
outcomes than under clean: validation strict-cell cond +4.7..+8.8 pp
(K=2..4, CIs > 0 at K=2 and K=4), joint +3.0..+5.7 pp, P(≥2 of 4)
+9.9 pp, both years positive, window-mechanics and tautology-direction
checks passed. `pwh` is DOWNGRADED as a holdout candidate (its effect does
not extend to sweep-aware outcomes: −2..−3.5 pp n.s.).

**Registered holdout gate for the frozen engine (to be run only on the project owner's
instruction, with a pre-registration token):** on 2026 holdout mornings,
frozen engine (P3-207 variant A, walk-forward), K = 2 and K = 4, strict-cell
y_any vs distance-matched same-morning controls. **PASS iff the cond
contrast is positive at both K with CI lo > 0 for at least one K, and the
mid-cell contrast at K = 2 has the same sign.** Report joint, cond, P(≥1),
P(≥2), calibration, and the A/B/C composition regardless of outcome.

---

## 14. Holdout outcome — 2026 OPENED AND SPENT (2026-08-15)

**2026 HOLDOUT OPENED — MAY NEVER BE USED AS A SEALED HOLDOUT AGAIN.**
Token 16ce7f7051547785, one shot, executed exactly as registered in §13.5
(regression check 0.00e+00 against the validated engine before unsealing).
Full record: EXPERIMENT_LOG P3-209, outputs/holdout_result.json,
outputs/holdout_run.log.

**GATE: FAIL.** Strict-cell cond model−matched: K=2 −0.0218 [−0.0976,
+0.0496], K=4 +0.0022 [−0.0575, +0.0642]. The 2024-25 selection edge did not
persist into 2026 — the third era-bound effect of the project.

**What is validated on all three eras (2021-24 / 2024-25 / 2026):**
the E1 destination model (holdout Brier skill +0.420), the P(clean|touch)
day-context annotation (holdout ECE 0.0142, joint 0.0026), and the sweep
taxonomy's behaviour frequencies (direct ≈ 21 %, combined ≈ 62 % of reached
zones at the primary cell). **What is not validated in any durable sense:
any selection edge over distance-matched same-morning alternatives** —
neither location identity (every era) nor fine placement (died 2026).

**Project status.** The pre-open NY-morning NQ ranking question is answered:
zones can be *annotated* with honest, era-stable probabilities, but no tested
method picks *which* same-distance prices will react better, out of sample,
across regimes. Per the project owner's instruction the broader research direction
(all-session levels, level lifetime, GC, rolling generation times) is NOT
begun; awaiting direction.

---

## 15. Phase 4 — strong vs weak zones, all sessions (opened 2026-08-15)

**The question changes unit.** Not "which grid price at 06:25" but: real zone
objects with lifecycles, tapped at any hour, and the target

> P(successful reaction | the zone is touched, everything knowable before
> the touch)

with success = the frozen path-aware taxonomy (direct rejection OR valid
sweep+reclaim). NQ only. Levels only. 2026 is ordinary data now (the holdout
was spent by P3-209); inference is chronological walk-forward with per-era
reporting.

### 15.1 Registered design (written before any result)

* **Zone universe**: confirmed zigzag pivots on 1-minute bars at three
  reversal scales — 0.15 / 0.35 / 0.75 ATR — which subsume swing highs/lows,
  displacement origins, failed breaks, untouched extremes and session
  extremes. Named concepts (round numbers, VWAP, profile nodes, HTF extrema)
  enter as *features*, not candidate types. Zone band = pivot ± min(base
  range, 0.06 ATR), floor 4t. **Availability = pivot confirmation time**
  (the reversal-threshold crossing), never the pivot extreme time.
* **Lifecycle**: active from confirmation; taps detected on density-filled
  1-second bars (range intersect); re-arm after a tap requires closes out of
  the band by ≥ max(8t, half-width) AND ≥ 900 s; acceptance through the far
  edge (300 s of closes) marks the original role invalidated and flips the
  zone (mirror interactions tracked with `flipped=1`); zones tracked to 30
  sessions of age; cross-roll prices via roll_adjust chains; the *useful*
  lifetime is discovered from the data, not imposed.
* **Outcome per interaction**: frozen P3-208 taxonomy on the forward
  1-second path, within-session, R = 15 pts, d0 = 4t, near-edge reclaim held
  60 s; primary cell D = 0.10 ATR / T = 1800 s, strict cell D = 24t /
  T = 900 s as sensitivity; resolved = ≥ 1800 s of session remaining.
  Primary outcome column: y_any (A ∪ B).
* **Features** (all timestamped strictly before the tap): formation
  (departure magnitude/velocity/volume/delta at confirmation, base
  size/duration, structure break, forming segment), freshness and
  interaction history (tap number, prior penetrations, prior A/B/C at this
  zone, volume traded through since formation), age (minutes, sessions,
  time since last tap, max excursion since formation), HTF alignment
  (nested-scale count, prior 20-session extremes), volume structure
  (trailing-profile volume at and around the zone, node/cliff geometry),
  geometry (width, width/ATR), **approach** (net move, speed, acceleration,
  volume, delta, retraces, compression over the last 5/15/30 min), context
  (segment, time of day, day of week, ATR percentile, realised vol 60 m,
  session range used, distance from session VWAP, release-slot flag), and
  side/role (approach side, role consistency, flipped).
* **Two model tiers**, because the product and the science differ:
  Tier 1 = zone-static + context only (rankable in advance → the top-k
  product test); Tier 2 = Tier 1 + approach (the full at-tap question).
  Approach is reported as the *increment* over Tier 1.
* **De-duplication**: taps within one (session, side) cluster — same
  10-minute window and within 0.05 ATR — are one observation; the oldest
  zone represents the cluster (n and scales recorded). All-interaction view
  kept as sensitivity. Session-clustered bootstrap everywhere.
* **Era-robustness bar (registered)**: a distinction is ROBUST iff the
  pooled y_any contrast is |d| ≥ 2 pp with session-clustered 95 % CI
  excluding zero AND the sign agrees in ≥ 5 of 6 calendar years
  (2021..2026). Weaker findings are reported as suggestive, never headlined.
  Model evaluation: expanding-year walk-forward (train ≤ year k, test year
  k+1), permutation importance on test folds only.
* **Known confounds carried in**: outcome-window truncation vs time of day
  (resolved filter + time-of-day stratification), nested/overlapping zones
  (dedup + clustering), approach features partially measuring short-horizon
  tape state rather than zone quality (named as such in interpretation),
  and the era-boundedness that killed every prior selection edge (the
  per-year bar above is the defence).

### 15.2 Phase 4 outcome (2026-08-15)

Corpus: 61,354 zones / 341,797 interactions / 1,365 sessions, deduplicated
to 126,222 observations (P4-300..302). Findings (P4-303/304/306):

1. **The dominant pre-tap fact is approach side**: resistance-side taps
   succeed 0.670, support-side 0.307, every year — the index's
   down-fast/up-slow asymmetry, opposite to folklore.
2. **Context/tape state is the next layer** (ATR percentile +10.2 pp, recent
   realised vol +7.0, approach range/energy +5.7-5.9, day range used +5.3).
3. **Zone-specific properties are real but third-order**: prior success at
   the same zone +2.9 pp, recency +2.9 (recently-tapped better), formation
   departure velocity +2.2, base size +2.2; flips −4.9; tap count, volume
   profile structure, HTF scale, width: dead.
4. **Lifetime**: mild decay over the first 8 hours, then flat to 30
   sessions; the objective invalidator is acceptance-through (flip).
5. **Selection skill, out of time, era-stable (first time in the project)**:
   ranking each session's taps by walk-forward score — top-2 0.790 vs
   seg×side-matched same-session 0.701 (+9.0 pp [+7.1, +10.8]), positive in
   all five OOS years; strict outcome +12.0 pp; bottom-ranked taps 9-12 %.
   Availability: median 24 active zones within 0.5 ATR of price.

The product implication: an intraday display CAN rank live zones with
era-stable skill, provided it presents what the skill actually is — side ×
tape-state × zone-history — rather than zone identity mystique. Next
decisions (build the live scorer/display, extend, or stop) are the project owner's.

---

## 16. Phase 5 — the true live display simulation (opened 2026-08-16)

Phase 4 ranked among taps (conditioned on touch). Phase 5 simulates the
actual product: at historical time t, with information ≤ t only, which zones
would be displayed — then track the *displayed* set forward with no
conditioning on future touch.

### 16.1 Registered design (written before any result)

* **Refresh**: every 15 minutes on the session clock (event-driven refresh is
  a product optimisation, not needed to evaluate usefulness). No hysteresis
  in the primary run; one pre-fixed variant (challenger must beat incumbent's
  score by 10 % to displace it) measured for churn, never optimised.
* **Eligibility at refresh t**: zone active (age ≤ 30 sessions, < 6 taps,
  roll-chain valid), **armed** (price left its band by hysteresis since the
  last tap and ≥ 900 s elapsed), price not currently inside the band, and
  volatility-normalised distance 0.02–0.75 ATR (primary radius; 0.50 and
  1.00 as sensitivity).
* **Display**: rank eligible zones by the walk-forward base score, greedy
  top-K (K = 4 recorded; top-1..3 are prefixes) with ≥ 0.05 ATR separation
  between displayed centres. Sides never forced to balance. Interpretable
  metadata retained per displayed row.
* **Scores**: base = Phase-4 T1 model (context + zone-static; approach SIDE
  is knowable at display time because an untouched zone's first-touch side
  is determined by its position relative to current price), retrained per
  walk-forward year on tap rows of strictly earlier years with the frozen
  P4 hyper-parameters; dynamic = T2 (approach features evaluated at the
  refresh). 2021 is burn-in; display simulation runs 2022-2026.
* **Episodes**: consecutive refreshes with the zone displayed; ends at its
  next recorded tap (touched-while-displayed, outcome from the frozen
  taxonomy) or at the refresh where it leaves display (replaced / removed /
  invalidated / drifted out of radius — recorded which).
* **Controls, same refresh, information ≤ t only**: distance-band + side
  matched eligible non-displayed zones, next-ranked (K+1..2K), random
  eligible, nearest-K. Their subsequent taps are looked up identically.
* **Advance notice**: for every tap, the score recomputed from data at
  tap − {30, 15, 5, 2} minutes; performance by lead and by distance band at
  scoring time.
* **Side-asymmetry adversarial checks**: unconditional ±15-pt race at random
  non-tap moments (drift/noise-asymmetry baseline), side redefinition at
  −5 min, per-segment and per-year decomposition, A-vs-B decomposition,
  zone-generation composition by side.
* **Pass standard** (user-specified, not one number): top-ranked displayed
  zones beat same-time controls conditional on touch; lift positive across
  years; a meaningful fraction of displayed zones get touched; useful
  advance notice; reasonable turnover; survives strict and primary outcomes;
  no future-touch conditioning.

### 16.2 Phase 5 outcome (2026-08-16)

Simulation: 1,119 sessions 2022-2026, 4.35 M scored (refresh × zone) rows,
97,512 display episodes, walk-forward models, zero future-touch
conditioning (EXPERIMENT_LOG P5-310/311).

**Against the pass standard:**
1. Top-ranked vs same-moment controls | touch: **FAIL vs side+distance-
   matched** (−0.6 pp [−1.2, +0.1]; next-ranked ≈ displayed too). PASS only
   vs naive controls (random +13.2 pp, nearest +16.2 pp conditional).
2. Across years: the matched lift is ≈ 0 in every year (not era-decay —
   structurally zero in the live frame).
3. Touched fraction: PASS — 53 % of episode zones touched same-session,
   18.8 % while displayed.
4. Advance notice: PASS — AUC 0.61 at 30 min out rising to 0.68 at 2 min;
   top-tercile 0.63-0.69 vs 0.52 base; approach dynamics add ~nothing.
5. Turnover: PASS with the registered 10 % hysteresis (median display
   60 min, rank-1 persistence 0.77, −1.1 pp quality).
6. Strict + primary outcomes: same pattern. PASS.
7. Leakage: display features proven identical to tap features; walk-forward
   only. PASS.

**Interpretation.** Phase 4's +9 pp was moment/timing selection; at a fixed
moment, side and distance are free knowledge and zone identity adds ≈ 0 —
the five-phase invariant, now confirmed in the product frame. What remains
is a real but humbler product: a state-driven display (side × distance ×
tape state) delivering **10.4 successful reactions/day at 63 % conditional
success (vs 50 % random / 47 % nearest), 99.3 % of days ≥ 1 success, median
zone at 0.40 ATR shown 16 min before touch, 90.8 % resistance-side** —
era-stable 2022-2026. The side asymmetry passed every adversarial check
(random-moment races symmetric; the asymmetry is approach-conditional:
rising approaches reject directly, falling approaches sweep-and-reclaim).

**Build decision is the project owner's**: the defensible product is the state display
with calibrated annotations; a "this exact price is special" display is not
supported and never was.

---

## 17. Phase 6 — the exact-price question (opened 2026-08-16)

Four questions: (1) does X carry pre-touch information that X±δ lacks;
(2) reaction magnitude in NQ points; (3) adverse movement before reaction in
points; (4) when in the day levels work best. NQ only, all sessions, levels
only, walk-forward 2022-2026.

### 17.1 Registered design (written before any result)

* **Shadow-anchor experiment (the hard local control).** For every deduped
  touched interaction: 8 shadow anchors at X ± {2.5, 5, 10, 20} pts,
  identical width, common decision time t0 = tap − 15 min; every anchor
  (real and shadow) scored from its OWN first touch after t0 under identical
  rules (frozen taxonomy + enriched magnitudes); anchors inside their band
  at t0 ineligible; shadows within 1.5 pts of another active zone centre
  flagged and also analysed excluding those. Primary contrast: event-paired,
  real vs the mean of its ± δ shadow pair (distance-neutral to first order),
  session-clustered bootstrap; pooled side × distance × year strata as the
  check. **Robustness bar: |d| ≥ 2 pp with CI excluding zero and the same
  sign in ≥ 4 of 5 walk-forward years.** Decay-by-offset curve = the
  empirical zone width.
* **Enriched path outcomes (one walker, applied to real and shadow
  identically):** adverse penetration measured from the NEAR edge including
  in-zone travel; rejection excursion from the near edge at 15/30/60-min
  horizons and full window; a path-based endpoint variant — the move
  attributed to the level ends at the first close back inside the band after
  the rejection has reached ≥ 15 pts; adverse-before-15-pt and
  before-30-pt reaction; turn location as a fraction of zone width;
  time-to-15-pt and time-to-max. All reported in NQ points (ticks ÷ 4).
* **Category sub-split (frozen A/B/C preserved):** A_front = direct with
  turn in the front half of the zone; A_deep = direct/partial with turn in
  the back half up to 1 pt beyond; B sweep+reclaim and C failure unchanged.
* **Time-of-day:** 30-minute PT buckets, merged into plateaus only where
  per-year direction agrees (≥ 4/5 years); multiple windows allowed;
  session rollups reported alongside.
* **Spatial feature map (Model B):** local-DIFFERENCE constructions only —
  prior reaction-event density prominence (events layer rebuilt for the full
  corpus), pivot/turning-point density prominence, prior turn-price
  concentration (from pass-1 enriched outcomes, strictly earlier events),
  trailing-profile volume gradient / curvature / peak prominence at X vs
  ± 5-10 pts, time-at-price gradient (acceptance boundary), multi-scale
  nesting. Model A = the Phase-5 T1 baseline (state + side + distance +
  zone history); Model B = A + spatial. Increment reported per walk-forward
  year for probability, magnitude, adverse depth, and shadow-pairwise
  ranking.
* **MBP-1 spatial:** one bounded ~100-session sample test of tap-local book
  features vs neighbours, run only after the tape-based spatial verdict is
  known; P2-113's global rejection stands otherwise.

### 17.2 Phase 6 outcome (2026-08-16)

**Question 1 — answered POSITIVE for the first time in the project, with
precision about what the information is.** The exact pivot price does not
react more OFTEN (sweep-inclusive success: −0.6 pp vs symmetric ±10-pt
shadow anchors, negative all six years); it reacts more TIDILY and travels
FARTHER: strict +2.2 pp pooled, and on the RESISTANCE side **+7.7 pp
[+7.2, +8.2] with CI > 0 in all six years, +4.9 pts larger reactions,
+8.7 pp vs shadows overlapping no other zone**. Support-side pivots are
anti-informative (−3.2 pp). Placebo anchors show negative diffs
(construction clean). Effective special region ≈ band ± 5-10 pts. Anchors
known a median 9.9 h pre-tap. Spatial features add nothing to tap-level
ranking (B−A ≈ 0 both outcomes): the value is ANCHORING, not selection —
P5's live conclusions stand; Phase 6 justifies pivot-anchored display
bands on the resistance side.

**Questions 2/3 — the practical numbers (NQ points):** reaction
(endpoint-based) median 31.8 / mean 63.6 (p25 20.7, p75 63.7, p90 141.6);
adverse-before-reaction median 11.6 / mean 14.6 (p25 4.2, p75 21.9, p90
32.0), by mode 3.0 / 11.2 / 23.9 (direct-front / deep-zone /
sweep+reclaim); ratio median 3.7×; sweeps median 8.1 pts deep, reclaimed
in 6.2 min, 43.4 pts after; deeper sweeps → larger reactions (37.6 → 53.1
pts by quartile). Success mix 38/21/41 direct/deep/sweep.

**Question 4:** one broad stable window — 05:00-11:30 PT (y 0.51-0.56,
year-stable per bucket); weak 14:00-22:00 PT and 00:30-02:30; premarket >
NY-AM > NY-PM > London > Asia every year; magnitudes flat by hour, speed
and adverse vary. Exact-location lift uniform across the clock.

MBP-1 bounded spatial screen: see P6-325.

---

## 18. Phase 7 — harden resistance, solve support, freeze the product
(opened 2026-08-16)

### 18.1 Registered design (written before any result)

* **Part 1 — adversarial battery on the resistance edge.** The paired
  strict@resistance shadow contrast (±10 pts primary) recomputed inside
  every slice: year (done: 6/6), volatility-percentile terciles, session
  blocks (done), age buckets, distance-at-t0 buckets, pivot scale, band
  width (hw terciles), offset grid (done: curve), alternative reaction caps
  (strict/mid/primary cells), roll-adjacent sessions (within 3 of a front
  change) vs not, release-slot taps vs not, high/low-volume days, trend vs
  range days (|day return| / range terciles), overlapping vs isolated
  shadows (done), first vs repeat taps, prior-tap counts, active-zone
  density (nesting terciles), and **confirmation-delay buckets with the
  strict live-confirmation timestamp** (pivot extreme time recovered by
  re-running the confirmed-zigzag per session and matching zones).
  **Survival bar: SAME SIGN in every slice with pooled CI > 0; a slice that
  flips sign with CI excluding zero kills or conditions the claim.**
* **Pivot-definition robustness**: fractal pivots on 5m/15m/30m/1H bars
  (k-bar local extrema, confirmation = k bars later) computed independently;
  overlap = share of fractal resistance pivots within 6t of an active
  zigzag pivot and vice versa; if the locations coincide (majority overlap),
  the effect is about the LOCATION and the zigzag construction stands; a
  low-overlap definition with plausible coverage earns one targeted
  shadow-measurement run.
* **Part 2 — support discovery protocol.** (a) The signed-offset outcome
  curve for support-side events from the existing shadow corpus: if an
  offset band (above or below the pivot low) shows a LOCAL peak that beats
  both the pivot and its own neighbours, that band is the candidate anchor;
  wider offsets (±30/40/60 pts) get one extra measurement run if the peak
  sits at the ±20 edge. (b) Backward mapping: for successful support
  reactions, the signed distance from the actual turn price to: the pivot
  low, the nearest other active zone below, the prior sweep-termination
  cluster, the prior turn-price concentration, and a volatility-scaled
  sweep-extension of the pivot. (c) Any discovered support anchor is
  re-tested with the full paired-shadow hard control, offsets frozen from
  2021-2023 and evaluated 2024-2026 out-of-time. No forced symmetry;
  "support unsolved" is an acceptable verdict.
* **Part 3-5**: magnitude/adverse/width/lifetime/invalidations recomputed
  on the RESISTANCE-PIVOT population specifically (not pooled), using the
  Phase-6 endpoint attribution; width decision from paired shadow decay ×
  volatility interaction; lifetime from age-bucket strict lift; invalidation
  = flip/acceptance rules re-checked on this population.
* **Part 6**: resistance-only display derived OFFLINE from the stored P5
  pool (side filter + re-ranking, hysteresis variant), with the
  exact-anchor advantage verified on DISPLAYED-and-tapped zones by joining
  their Phase-6 shadow rows. No new simulator run is needed unless the
  offline re-ranking changes eligibility (it does not — it filters it).
* **New-claim bar** unchanged: |d| ≥ 2 pp, session-clustered CI excluding
  zero, same sign ≥ 4/5 walk-forward years (support-anchor claims: frozen on
  2021-2023, confirmed 2024-2026).

### 18.2 Phase 7 outcome — THE FROZEN NQ RESISTANCE-LEVEL METHODOLOGY
(2026-08-16)

**Verdicts.** The resistance-pivot anchoring edge passed the full
adversarial battery (40/40 slices positive, P7-331), is definition-robust
(P7-332), causally confirmable (median 7 min after the extreme, P7-330),
rankable within itself (P7-335), and productizes offline (P7-336). Support
has NO exact-location anchor: the pivot low is a local valley, no offset or
data-adaptive cluster anchors it in either era (P7-333/334) — the system is
legitimately asymmetric.

**FROZEN METHODOLOGY — "NQ Resistance Pivot Levels" (proposal for the
final indicator; build awaits the project owner):**
1. **Universe**: confirmed zigzag pivot highs at 0.15/0.35/0.75-ATR scales
   on 1-minute bars (availability = confirmation, median 7 min after the
   extreme); displayed only when price is BELOW the zone (resistance role).
2. **Band**: pivot ± min(base-range/2, 0.03 ATR), floor 1 pt (median
   ± 7.2 pts); the empirically special region extends ~± 10 pts, flat
   across volatility regimes in points-space (no vol scaling).
3. **Lifecycle**: active to 30 sessions and 6 taps; anchoring persists for
   repeat taps and even after flips (lift +7.0 pp), but flip and heavy
   volume-through downgrade rank; re-arm 900 s + hysteresis after taps.
4. **Ranking** (validated walk-forward, +9..+17 pp over seg+distance-matched
   same-session pivots): state score (P5 T1) × pivot-quality features —
   prior success share, time since last tap, departure velocity, base size,
   departure volume, structure break; downgrades: volume-through,
   confirmation delay, flipped, age, tap count, ev-density.
5. **Display**: top-K ≤ 4 nearby (0.02-0.75 ATR) resistance zones, 0.05-ATR
   separation, 10 % incumbent hysteresis; annotations: strict reaction
   probability, expected reaction (med ≈ 32 pts), expected adverse
   (med ≈ 6.5 pts), mode shares, age, prior record.
6. **Support**: displayed, if at all, only as state-conditional context
   (P5 machinery), never as exact levels — no anchor exists.

**Live numbers to expect (offline validation, 2022-2026):** ~78 zone
episodes/day; 54 % touched same-session (20 % while displayed, median 16 min
after display); strict 0.57 / sweep-inclusive 0.66 conditional on touch;
8.7 strict successes/day; exact-anchor lift on displayed zones +3.0 pp
[+2.2, +3.8] (population-wide +7.7 pp); every year 2022-2026 positive.

---

## 19. Phase 8 — support only; the resistance system is FROZEN
(opened 2026-08-17)

### 19.1 Registered design (written before any result)

The resistance engine (§18.2) is untouched. Support attacked on two
products, in order of causal ambition:

* **P8-340 — Is sweep depth predictable at all?** Turn location (successes)
  and max penetration (all support taps) modelled from causal features
  (volatility state, downside approach velocity/acceleration/volume/delta,
  session, range used, prior swing geometry, prior sweep-depth history).
  Report unconditional depth stats and the variance/quantile-loss reduction
  from features. **Decision rule: if features reduce the interquartile
  spread of predicted turn location by < 20 % relative and the ATR-scaled
  spread is no tighter than the points spread, Product A (pre-sweep
  termination band) is closed by inference** — the P6 shadow grid already
  shows no fixed-point offset peak (−2.5..−20), so only scaling or
  conditioning could rescue it.
* **P8-341 — the one Product-A empirical shot regardless**: a
  volatility-scaled termination band (centre = pivot low − k·ATR, k frozen
  as the 2021-2023 median scaled turn depth; width = the zone's own),
  walked with ±10-pt shifted controls at the same t0, confirmed on
  2024-2026 only. One registered variant; no offset shopping.
* **P8-342 — Product B(a): reclaim-validated old zones.** Support-side
  retests of zones whose PRIOR tap already produced a successful
  sweep+reclaim (n_prior_B ≥ 1, maturity-gated as built). Paired shadow
  lift on the existing corpus, per era, sliced by recency of the validating
  event.
* **P8-343 — Product B(b): sweep-termination pivots.** dir = −1 zones
  tagged at CONFIRMATION as sweep-born: an older support zone within 40 pts
  above was swept within the prior 12 h and its sweep extreme lies within
  6 pts of the new pivot low (all knowable at the new pivot's confirmation).
  Their support-side taps' paired shadow lift, per era, vs non-sweep-born
  dir = −1 zones as the family control.
* **P8-344 — characterisation and freeze** of any family passing the bar
  (|d| ≥ 2 pp, session-clustered CI > 0, same sign ≥ 4/5 years 2022-2026):
  magnitudes, adverse, sweep depth vs the NEW anchor, reclaim times, modes,
  time-of-day, frequency/day, advance notice — all with the Phase-6
  endpoint attribution. If nothing passes: Outcome D stated plainly;
  the resistance engine is not altered to compensate.
* Baselines carried everywhere: original pivot low, fixed −15.5 pts,
  vol-scaled offset, random/shifted local bands.

### 19.2 Phase 8 outcome — OUTCOME D (2026-08-17)

Support has no exact-location solution. Every registered route failed
honestly (LOG P8-340..344): sweep depth is essentially unpredictable
(conditional models narrow the interval 6-10 % vs the 20 % bar; ATR scaling
5 % tighter); the frozen vol-scaled band decays from +1.9 pp (2021-23) to
+0.7 pp (2024-26); reclaim-validated old zones −4.9 pp; sweep-born pivots'
first retest exactly neutral, later taps negative. The bullish turn is a
diffuse excursion/reclaim EVENT (89.7 % sweep-mode, adverse med 23.2 pts,
ratio 1.48 vs resistance's 4.9), not a remembered price — the NQ level
structure is fundamentally asymmetric.

**Final product stance (unchanged resistance engine):** exact resistance
pivot levels per §18.2 + state-based support CONTEXT (probability
annotations from the P5 machinery, e.g. "elevated bullish-reclaim state,
expected turn 10-22 pts below the old low"), never drawn as exact support
lines. Support characterisation numbers for the display's honest
annotations are in P8-344.

---

## 20. Phase 9 — confirmation around the frozen resistance zones
(opened 2026-08-17)

### 20.1 Registered design (written before any result)

**Population**: the frozen resistance interactions (side = resistance,
deduped, resolved; 63,188). Levels untouched. Outcomes: y_strict primary,
y sweep-inclusive secondary; magnitude accounting via the Phase-6 endpoint
attribution (rev_ep). Baselines to beat: 58.9 % strict / 67.0 %
sweep-inclusive unconditional.

**Timeline build**: for every interaction, causal features on a checkpoint
grid τ ∈ {15, 30, 60, 120, 180, 300, 600, 900 s} after first touch (path
state, penetration, rejection progress, velocity, time-above, closes-below
share, buy/sell/delta since touch, trade intensity vs trailing baseline,
progress-per-buy-effort) plus the pre-touch approach set at τ = 0. At every
fire/checkpoint: adverse-so-far, reaction-already-completed, reaction
remaining (rev_ep − completed, floored), so reliability and timeliness are
inseparable in every result.

**Registered mechanical rule families (grids fixed in advance):**
  R1 hold-below-near: closes below the near edge for h ∈ {30, 60, 120} s;
  R2 shallow-stall: at τ ∈ {60, 180, 300} s, max penetration ≤ {2.5, 5,
     7.5} pts and price back below the near edge;
  R3 sweep-reclaim: after exceeding the far edge, re-close below
     {far, mid, near} held {30, 60} s;
  R4 velocity break: ≥ {5, 7.5, 10} pts down from the near edge within 60 s;
  R5 absorption: buy volume ≥ 1.5× trailing baseline over [t, t+τ],
     τ ∈ {60, 180}, with penetration ≤ 5 pts (effort without progress);
  R6 acceptance filter (disqualifier): time above the far edge > {120,
     300} s predicts failure;
  R7 bar rule: first 1-minute close fully below the near edge.
Each cell reports: fire share, precision (strict and sweep-incl), recall
over successes, FP rate over failures, latency (med/p25/p75), adverse
before fire, completed before fire, remaining after fire (med/mean/%),
by session block and by year. Parameter plateaus required; cell count
reported (multiple-testing).

**Model**: per-checkpoint HGB (features ≤ τ only), walk-forward by year →
calibrated P(strict | state at τ); the AUC-vs-τ curve is the
information-arrival answer; threshold curve 0.60-0.85 with retained-share,
realized rate, latency and remaining move at each threshold. Ablations:
pre-touch-only vs post-touch-only vs both.

**MBP-1 add-on**: bounded ~40-session sample (≤ 2025-06), touch-time book
features at τ = 60/180 vs the trades-only model at the same τ on the same
sample; honest chronology; rejected if the increment is small/inconsistent.

**Discipline**: chronological walk-forward 2022-2026 (2021 burn-in for
models; rules are parameter-fixed so all years report), session-clustered
CIs on precision lifts, plateaus over grids, no P&L/stop/target artifacts.

### 20.2 Phase 9 outcome — THE FROZEN CONFIRMATION FRAMEWORK (2026-08-17)

**Verdict: confirmation is solved, simply.** Information arrives fast after
touch (pre-touch AUC 0.582 → 0.79 at 60 s → 0.91 at 5 min); the simple
mechanical rules capture essentially all of it (model beats them by ~2 pp
precision at the cost of 4.5 pts of remaining move); reclaim-waiting is a
trap (0 pts median remaining); MBP-1 adds exactly nothing.

**FROZEN RULE (proposal, alongside the §18.2 levels):** on a frozen
resistance-zone touch, CONFIRM when either
  (a) **velocity break** — price moves ≥ 10 pts down from the zone's near
      edge within 60 s of first touch; or
  (b) **shallow stall** — at 5 minutes, maximum penetration ≤ 5 pts and
      price is below the near edge, provided price has NOT spent 120 s of
      closes above the far edge first (acceptance disqualifier).
DISQUALIFY (expect failure, 69 %) once closes above the far edge accumulate
120 s. Sweep-mode reactions are consciously not chased — no in-time
confirmation exists for them.

**Frozen-framework numbers (2021-2026, 63,188 interactions):** fires 44.8 %
of touches; strict precision 0.809 (base 0.589), sweep-inclusive 0.841
(base 0.670); recall 61.5 %; FP 20.8 %; latency med 60 s; adverse before
confirmation med 2.3 pts (p90 7.5); remaining after confirmation med 20.5 /
mean 53.2 pts (66 % of the eventual move); every year 0.758-0.868, every
session block 0.728-0.874. Optional calibrated dial: the τ-checkpoint model
(ECE 0.018) with its threshold curve (e.g. τ=60 s @ 0.80 → 20 % of touches
at 0.945 strict).

**Scope note:** stops/targets/R:R/P&L remain unresearched by design; the
descriptive numbers above (adverse-before-confirmation, remaining-move
distributions) are the inputs a later strategy phase would start from.

---

## 21. Phase 10 — the strategy conversion test (opened 2026-08-18)

### 21.1 Registered design (written before any result)

**Frozen inputs, untouched**: §18.2 levels, §20.2 confirmations (R4v10,
5-min stall, 120-s disqualifier), the Phase-9 calibrated model (re-scored,
never re-trained on P&L).

**Trade simulation conventions:**
* Short-only at frozen resistance confirmations; entry fill = next 1-second
  close after the rule is legally known (+ slippage via costs); limit
  entries fill only when price trades THROUGH the limit by ≥ 1 tick within
  a 15-minute timeout; first-passage between stop (above) and target
  (below) on 1-second bars; **same-second ambiguity always resolves
  stop-first**; horizon = session end (exit at last price) unless a
  time-exit config applies (exit at close of {5,10,15,30,60} min).
* **Costs (round-trip, all-in points)**: optimistic 0.5, base 1.0, stress
  2.0 for market-entry configs; limit-entry configs 0.25 less (entry side
  unslipped). Reported gross and net.
* **Entry families**: E1 immediate; E1d delayed 2 s (robustness); E2
  pullback limits {2,4,6,8,10} pts; E3 retracement of the confirmation leg
  {25,33,50,66} %; E4 near-edge retest limit.
* **Stop families**: S1 fixed {5,7.5,10,12.5,15,20,25,30} pts; S2 far edge
  + {0,5,10,15,20}; S3 pre-confirmation extreme + {1,2,4,6,8}; S4 = ATR-
  scaled ({0.02, 0.04}·ATR) as a two-cell check.
* **Target families**: fixed {10,15,20,25,30,40,50,60} pts; R-multiples
  {0.5,0.75,1,1.25,1.5,2,3} of the S1 stop; T2 partial-half at first
  target, half at second, stop unchanged (no breakeven/trailing — not
  registered, not tested); T3 time exits.
* **Staged selection to prevent grid-mining**: entry family and stop/target
  PLATEAUS chosen on 2021-2023 ONLY; the chosen configurations execute
  unchanged on 2024-2026 (the honest OOS); full-period per-year tables
  reported for the finalists. Isolated parameter spikes rejected; plateau =
  neighbouring cells within a few hundredths of R.
* **Portfolio realism for finalists**: chronological one-open-trade rule
  (confirmations during an open trade are skipped), duplicate-event
  prevention via the frozen dedup, per-session/per-year/per-tap-number
  splits, delay and slippage stress, ±neighbour sensitivity.
* Calibrated-model variants use the walk-forward OOS scores only, at
  pre-listed thresholds {0.65,0.70,0.75,0.80,0.85,0.90}.
* Outcome F (no profitable conversion) is an acceptable verdict.

### 21.2 Phase 10 outcome — THE STRATEGY CONVERTS (2026-08-18)

Full results LOG P10-360..363. Selection on 2021-2023, execution unchanged
on 2024-2026, net of base costs, conservative fills.

**RECOMMENDED (F4, "model-gated fast rejection")**: short when R4v10 fires
at a frozen resistance zone AND the walk-forward calibrated probability
≥ 0.85; market entry within seconds (2-s delay immaterial); stop 12.5 pts;
target 30 pts. OOS 2024-26: **+0.376R / +4.70 pts per trade net, WR 42.9 %,
PF 1.61, max DD 27R, 9.5 signals/day (7.6 with one-open-trade)**; every
year +0.29..+0.44R; every session positive (Asia +0.245 → NY-AM +0.403);
stress costs +0.296R (PF 1.45); flat parameter plateaus.

**Alternates:** F3 high-WR (S=30/T=15, ungated): WR 72.2-72.6 %, +0.05R,
PF 1.18-1.20, DD 23-29R, 6/6 years positive, Asia breakeven — the
psychology-friendly thin-edge point. F1/F2 (ungated S12.5/T30-40):
+0.126-0.138R broad-coverage baselines. Partial 15/40 exit: WR 0.521 /
+0.142R middle point. REJECTED: trading the 5-min stall (negative
everywhere), the ungated two-part framework (≈ 0), pullback/retracement/
retest entries (adverse selection), sweep-mode entries (never designed, per
P9). Sizing, account rules, Topstep constraints, execution automation:
deliberately unresearched — next phase's scope if desired.

---

## 22. Phase 11 — the risk wrapper around frozen F4 (opened 2026-08-18)

### 22.1 Registered design (written before any result)

F4 untouched. Unit = the chronological one-open-trade F4 executed stream
(base costs). Windows kept separate throughout: 2022-2023 (parameter-
selection era; gate itself walk-forward), 2024-2026 (frozen OOS), 2022-2026
pooled (tail statistics, gate-honest). Design decisions prioritise 24-26;
tail estimation uses 22-26 for sample size, stated as such.

* **Audit**: exact per-year counts for every denominator (interactions,
  R4 fires, gated signals, executed trades, per-session, taps included);
  reconciliation of earlier per-day figures (display-subset vs full
  eligible universe).
* **Drawdown anatomy**: R-equity; DD episodes at thresholds {2..27R} with
  counts/frequencies/durations/recoveries; underwater periods; the 18-loss
  streak located exactly; streak frequencies ≥{5,8,10,12,15}; daily P&L
  distribution and tail tables.
* **Wrappers (replayed chronologically on the executed stream; wrappers
  only truncate days, the overlap rule is not recomputed — approximation
  stated)**: daily stops {−1..−6R, none}; profit caps {+1..+8R, none};
  combined grid {−2..−5R} × {none,+2..+5R}; max trades/day {2..8,∞};
  consecutive-loss stops {2..5} with day-scope and next-session-scope;
  session-halved-risk variant; skip-tap≥4 overlay (reported as overlay,
  never a signal change). Every wrapper reports expectancy sacrificed vs
  drawdown saved.
* **Monte Carlo**: methods = chronological, IID trade bootstrap, day-block,
  session-block, 5-day moving-block; 2,000 paths each; per-250-day and
  full-horizon max-DD percentiles {p50..p99.5} and exceedance probabilities
  {20..60R}. Design DD chosen from day-block p95/p99, not the historical
  single maximum.
* **Prop engine (parameterised)**: buffer B$, static/trailing flag, daily
  stop $, profit target $, risk/trade $; violation probability, survival
  curves, expected extraction until violation/horizon; illustrative buffer
  grid {$3k, $4.5k, $6k, $9k} with explicit note that real firm rules must
  be supplied before any final numbers; risk grid {$50..$500}.
* **Frontier**: max-raw, max-DD-adjusted, max-survival, max-expected-
  extraction, balanced — each fully specified.

### 22.2 Phase 11 outcome — THE RISK LAYER (2026-08-18)

Full numbers LOG P11-370..373. Core findings: (1) the 28.8R historical max
DD is ≈ the p95 of any 250-day window (block-MC p99 = 34.8R) — **design DD
= 35-40R, never /27**; (2) drawdowns are multi-week bleeds, so **daily
wrappers do not pay** — every stop/cap/count rule sacrifices more
expectancy than drawdown saved (only a −6R emergency stop is ~free);
(3) one-open-trade is strongly protective (DD 29R vs 48R); (4) within-day
sequences are unpredictable (no consec-loss information); (5) the optional
tap≤3 overlay is the one genuine DD-adjusted improver (+0.454R, 17R DD,
−41 % volume); (6) extreme-churn days (163 trades, +130R) inflate raw
totals — practical execution ≈ the cap-8 profile (MC p95 24.3R).

**RECOMMENDED WRAPPER (extraction-oriented):** risk/trade = usable buffer ÷
35 (p99 design); no daily profit cap; no consec-loss rule; optional −6R
emergency daily stop; one-open-trade; practical trade cap per operator
capacity; optional tap≤3 overlay when lower DD is worth −41 % volume.
Illustrative ($4,500 trailing buffer, real firm rules pending): risk ≈
$125-150/trade → violation ≈ 0.4-3.7 %/yr, E[extraction] ≈ $74-88k/yr,
p10 ≈ $52-60k. Aggressive ($250-300, buffer/15-18) raises E[paid] ~40-60 %
at 32-42 %/yr account loss. All dollar figures illustrative until actual
prop rules are supplied; engine is parameterised and ready.

---

## 23. PHASE 12 — FUNDED-ACCOUNT LIFECYCLE ECONOMICS (registered 2026-08-19)

### 23.1 Registered design

**Objective flip:** maximise TOTAL NET CASH WITHDRAWN over time, account
deaths permitted. Economic result (payouts − replacement costs) is the
optimand, not account longevity. F4 frozen; no signal changes.

**Rule snapshot (Topstep 150K XFA, supplied 2026-08-19; all parameterised):**
start $0; initial MLL −$4,500; MLL trails **end-of-day** balance high-water
(mll = min(0, hwm_eod − 4500)), locks at $0, permanently $0 after any
payout; intraday equity ≤ MLL kills the account (checked at trade
granularity + a full-1R-adverse sensitivity). Standard path: 5 winning
days ≥ $150 net + (after payout 1) positive cycle net → request ≤ 50 % of
balance, cap $5,000 ($10,000 promo). Consistency path: ≥3 trading days,
largest winning day ≤ 40 % of cycle net → ≤ 50 %, cap $6,000 ($12,000
promo). Both reset cycle at payout. Profit split configurable: 100 % of
first $10k of payouts, then 90 % (current published split). Optional $3,000
DLL scenario (day stops at crossing trade) paired with promo caps. Payout
deducted at EOD of request day; min request $125.

**Engine:** vectorised over N simulated paths; day sequences by block-5
session bootstrap from the executed F4 stream (P11 method; block-20
sensitivity); within-day trade order preserved exactly (per-session
cum-R prefix arrays); common random numbers across configs for paired
comparisons; chronological 2022-26 replay as deterministic case study.
Streams: raw, cap-8/6/4 (first-N-trades-of-day, no cherry-picking),
tap≤3 (portfolio re-run on tap≤3 fires — NOT post-hoc filter), tap≤3+cap8.
Risk grid $50-$400 + buffer/{15..45}; $167 = the user /27 hypothesis.
Policy family: (first-payout trigger balance T) × (retained cushion C,
$ or R-based) × (withdrawal fraction ≤ 50 %); dynamic risk (stage-switched)
and house-money escalation tested coarsely. Replacement: renewal cycles
with cost ∈ {0..1000}$ × delay ∈ {0..20}d. Horizons: 250-day years with
inline replacement (annual distributions) + 750-day no-replacement runs
(per-account lifecycle). Report distributions (p10/p25/p50/p75/p90), never
means alone.

**Honesty bars:** the 50 %-of-balance rule already forces retained ≥
withdrawn — note where it binds vs policy cushion; winning-day $150 gate
interacts with risk size (low risk delays eligibility) — report; no
consistency-motivated F4 truncation unless it wins economically; MNQ
feasibility ($25/R granularity) reported alongside exact dollars;
extreme-churn days flagged; all firm-rule outputs labelled as snapshot-
dependent. Outcome "aggressive sizing is NOT economically superior" is
acceptable.

### 23.2 Phase 12 outcome — THE LIFECYCLE LAYER (2026-08-19)

Full numbers LOG P12-374..380, outputs/p12_{AB,deep,final}.json. The user
hypothesis (payouts arrive before the big drawdown) is CONFIRMED and
understated: chronologically, 43-44 payouts and $90-197k were banked
before Aug-2023; in simulation the median max-strip $167 account banks
$29k across 14 payouts before dying. Structural findings: (1) extraction
is ceilinged by payout frequency × cap (~25/yr × $10k×split ≈ $240k/yr),
not by F4's P&L; (2) risk keeps paying to $600/trade because replacement
accounts are subsidised drawdown — deaths are a COST INPUT, not a failure;
(3) Standard path > Consistency (40 % rule blocks ~164 d/yr on F4's fat
tail); (4) the $3k DLL never binds ≤ $175 risk → promo caps are free;
(5) policy: take payout #1 immediately, always take the full 50 %, keep
C = 18R×risk in the account (knee: deaths 1.2→0.2/yr at zero net cost);
(6) at matched failure probability tap≤3 extracts 2-5× raw/cap8 (its
~14R DD supports ~3.4× the risk$); (7) dynamic/house-money risk staging
is dominated by constant higher risk; (8) replacement cost ($0-1k) and
delay (0-20d) move net ≤3 %.

**RECOMMENDED REAL-WORLD POLICY (registered):** tap≤3 stream (~3.4
trades/day, human-executable), standard path + DLL/promo caps, risk $300
(12 MNQ) scaling to $400 (16 MNQ), immediate first payout, always 50 %,
retain 18R×risk (≈$5.4-7.2k), replace on death without hesitation →
$97-126k/yr mean, p10 $69-91k, deaths 0.06-0.10/yr, P(losing year) ≈ 0.
Caveats standing: backtest-grade fills; rule snapshot 2026-08; live
automation still out of scope.

---

## 24. PHASE 13 — F4+TAP<=3 PRE-IMPLEMENTATION AUDIT (registered 2026-08-20)

### 24.1 Registered design
Audit only, zero parameter changes: canonical immutable ledger; full
statistical battery (trade/year/OOS/tap/streak/DD/MC/daily/monthly/
session/ToD); holiday + macro descriptive audits; execution stress (cost
grid, delay grid) recomputed exactly from 1s paths; live-feed
reproducibility check; implementation spec; readiness verdict.

### 24.2 Outcome — CRITICAL: look-ahead in the F4 gate; NOT READY

Ledger `p13_ledger.parquet` (3,835 trades) reconciles P12 exactly and is
the permanent reference FOR THE SIMULATED STREAM. Full profile in LOG
P13-381..383 (all years positive; OOS 24-26 +0.459R/PF 1.78; max DD
17.4R trade-gran; MC p95/p99 19.7/24.3R; 96.3 % positive months; smooth
tap plateau; holidays negligible n=8; macro days fine; breakeven slippage
5.66 pts RT).

**P13-384: the frozen gate (p_both60, cp60 features through touch+60s)
is evaluated 38 s in the future of the median entry (fire+3s ≈ touch+19s;
98.3 % of entries precede the window).** Live-legal rendering (enter at
touch+62/63 once the gate is computable): +0.04R, PF 1.05 — no usable
edge (price falls a further ~7.9 pts during the wait). Fully-legal
ungated R4v10 tap<=3 at fire+3: +0.191R, PF 1.28, all years positive —
real but far below the frozen +0.453R, and its S/T + tap cutoff carry
selection caveats from the gated era. **P10 F4 OOS, P11 wrapper numbers
and P12 lifecycle dollars are withdrawn as live expectations.** F3
(stall source, fires at 300s) is chronologically clean.

**VERDICT: NOT READY.** Phase-14 requirement (new research, needs
authorization): fire-time confirmation model (PRE + checkpoints evaluated
AT the R4 fire second, walk-forward), then staged S/T re-selection on the
legal stream, then repeat this audit. Live-data note: signed delta
(aggressor side) and persistent zone/tap state are required inputs; rest
is OHLCV-derivable.

---

## 25. PHASE 14 — LEAKAGE AUDIT + LEGAL FIRE-TIME STRATEGY (reg. 2026-08-20)

### 25.1 Registered design
Full-project chronology audit with permanent automated tests (zero
violations for any live candidate); fire-time dataset (all-tap R4v10,
prefix-only features, checkpoint entries 0..+15s, S/T label grids);
walk-forward models vs simple mechanical rules; thresholds from 2021-23
quantiles only; staged S/T; DD-primary evaluation (depth, MC tails,
underwater duration, negative months); truncation-invariance + streaming
replay required; dataset-reuse acknowledged (no sealed-holdout claims).

### 25.2 Outcome — STRONG LEGAL IMPROVEMENT; candidate ready for
streaming-verified paper trading

Audit: two leaks total in project history — the fatal P13 gate
chronology (fixed by construction here) and FINDING #2 (legacy immature
prior-outcome counts on 1,554 taps = 0.455%, corrected by exact
recompute; P14 layer exact-clean). Everything else verified causal
(roll, ATR, vwap, zigzag confirmation, tap counting, walk-forward,
no-fit preprocessing). Proofs: truncation invariance PASS; per-second
streaming replay IDENTICAL (0/137 mismatches incl. DD-period sessions).

Discoveries: the 55.7R baseline drawdown is REGIME BLEED (27-44-session
episodes, 142-session underwater) caused by an identifiable POPULATION —
slow, non-monotonic, choppy, late-tap R4 fires (anti-rule ≈ 0R with
400+-session underwater); recent-performance features describe bleeds
retrospectively but predict NOTHING (flat quintiles); waiting past the
fire bar adds no discrimination and costs price (ck0 dominates all
checkpoints); ML adds only a modest layer (AUC ≤0.57) on top of the
mechanical structure.

**REGISTERED PRIMARY (all components selected on 2021-23 or a priori):
short NQ when a tap #1-3 of a frozen resistance zone prints a 10-pt
rejection within 24s of touch with monotone share ≥0.27; decide at the
fire bar, enter market next second; stop 12.5 / target 30; one open
trade; base costs.** 2022-26: n 2,731 (2.5/day), WR 41.0%, +0.310R,
PF 1.49, day-DD 21.4R, MC p95/p99 22.8/29.5R, underwater max 76
sessions, worst month −8.0R, 7 negative months, every year +0.23..+0.37.
Plateau operating points (user's choice, all on smooth surfaces):
T40 (+0.386, PF 1.55, MC p99 33.3); tap≤2 (+0.353, DD 15.0, uw 47);
tap1 (+0.427, PF 1.71, DD 10.3, 1.24/d); hybrid HGB-top70 (+0.376,
PF 1.61, uw 36, negM 4). Sessions all positive; Asia weak (+0.16, one
negative year) — kept, monitored. Cost stress: −0.08R per extra point
RT (breakeven ≈ +3.9 extra pts). Live data needs: 1s OHLCV + per-second
volume/delta (aggressor side), persistent zone/tap state; every feature
live-computable. NEXT (needs authorization): paper-trade the registered
primary; re-audit after 200-300 live signals against P13-style
forward-test envelopes recomputed for this stream.

---

## 26. PHASE 15 — LEGAL-STRATEGY ECONOMICS (registered 2026-08-21)

Design: P12 lifecycle engine (rules parameterised, Standard path,
snapshot 2026-08) on the five §25.2 legal ledgers only; $500/2NQ focus +
risk grids $100-1000; C re-derived (12R knee); matched-failure sizing;
money-speed marks; chronological replays; DLL/promo scenario.

Outcome (LOG P15-393..395): **B_t40 selected — most money, fastest, at
every practical size**; $500 promo ≈ $91.6k/yr mean; chronological
2022-26 $315k, zero deaths; E_hybrid = bleed-averse runner-up (−3-10 %
money, half the underwater/deaths); failure probability is cushion-
governed (scale-free in risk) so Ultra's low DD does not monetise;
recommended size $500-600 (DLL rarely binds); above that the $5k/$10k
caps and DLL interference flatten returns. All numbers snapshot-
dependent; live forward test remains the true unknown.

---

## 27. PHASE 16 — BALANCED AUDIT + RISK OPTIMIZATION (reg. 2026-08-21)

Stage A verdict: **CLEAN — INDEPENDENTLY REPRODUCED** (LOG P16-396;
finding #3 = refit jitter + unstable tie-break, fixed by frozen scores +
stable keyed sort; canonical ledger elementwise-equal to stored P15;
independent reference simulator 20/20; accounting identity exact; deaths
0.335/yr from raw counts; MC seed-stable ±$191, method-robust).
Stage B (P16-397/398): payout frequency ~risk-invariant; survival is
the axis; **recommended $650 (zone $600-750), promo/DLL, C=12R,
immediate 50 % → ~$101k/yr mean, p10 $65k, gaps 13/23, deaths ~0.5/yr,
P(payout#1) 73 %; chron 2023-26 $377.6k/0 deaths; 60 %-edge floor
$68k.** Std-cap fallback if promo unavailable: ~$68k at $650. All
dollars snapshot-dependent; live forward test remains the true unknown.

---

## 28. PHASE 17A — SEALED HOLDOUT PLAN (registered 2026-08-15; BINDING)

Cutoff 2026-06-17T03:03:02Z. Only 40 post-cutoff sessions exist today;
6-month holdout completes 2026-12-17. Purchase spec: GLBX.MDP3, schema
trades ONLY, stype_in=parent, symbols NQ.FUT, start 2026-06-17T22:00Z
(first unseen session open), DBN/parquet; MBO/MBP-1 not required.
HOLDOUT 1 = 2026-06-18..2026-12-17 (127 sess, ≈279 expected trades);
HOLDOUT 2 = subsequent 3-6 months, unpurchasable today. Freeze:
HOLDOUT_FREEZE_MANIFEST.json (23 hashes) + SEALED_HOLDOUT_REGISTRY.json;
sealed dirs outside glob-ingested trees. Pre-registered verdict rules in
LOG P17-400 are BINDING (PASS: N≥200, mean≥+0.15R, PF≥1.20, clustered
90 % CI > 0; FAIL: mean≤0 or PF≤1 at N≥200; else borderline → extend).
No-retune rule: post-open modifications demote HOLDOUT 1 to development
data. NO holdout records were accessed in 17A.

### 28.2 Phase 17B outcome — HOLDOUT FAILED (2026-08-15)

HOLDOUT A (40 sessions, 2026-06-18..08-14) opened per user authorization
of the partial window. Integrity + pipeline equivalence fully proven
(manifest hashes, freeze 23/23, June-overlap bars identical, mechanical
warmup ledger 392/392 bit-equal). FINDING #4: model binary never
persisted; recipe-refit fidelity 91 % — now persisted; mechanical core
carried as unambiguous co-reference. RESULT: Balanced −0.133R/PF 0.83/
WR 27.9 % (N 140); mechanical −0.103R/PF 0.87 (N 167); 0th percentile
vs all dev windows; prop replay 0 payouts/2 deaths. **VERDICT: FAIL —
the development edge did not generalize.** No parameters were touched
after opening. HOLDOUT A is spent as validation for any modified
strategy. Live deployment NOT justified. Research reopening requires a
new authorized phase; HOLDOUT 2 (2026-12-18+) is the only future
validation data.

---

## 29. PHASE 18 — AUTOPSY + ROBUST REDESIGN (2026-08-15)

Stage A (P18-402): verdict E-MIXED — resistance + exact-location edge
SURVIVED and strengthened (+0.201 vs shadows on holdout); R4 event
survived but its continuation premium under 12.5/30 decayed to
~breakeven; tap/fire/mono/HGB confirmation stack INVERTED OOS (selection
overfit; rule cell −0.126 vs complement −0.018); loss composition
unchanged; no regime state explains it (fire-density analogues were
profitable in dev); selection audit: ≈200+ final-stage variants ⇒
winner's-curse ≈ +0.10-0.20R of the 0.4R gap.

Stage B (P18-403/404): 2 registered hypothesis families, 43 cells, all
reported; nested walk-forward with 10 half-year outer blocks (former
holdout = ordinary block 10). H2 (R4 wide-bracket) viable (+0.193
pooled, worst 0.000). **H1 WINS: FADE-1** (P18-405, frozen sha256
8b969a07): first-touch resistance fade at the near edge, ≥2t
trade-through fill, S20/T30 — pooled +0.342R, PF 1.73, WR 55.7 %,
worst fold +0.200, 10/10 blocks, +0.5pt stress robust, and
shadow-control-superior in EVERY block (+0.26..+0.50). ZERO external
validation; production ledger pass + streaming proof + ONE-SHOT sealed
test (≥2026-08-15 data, criteria in P18-405) required before any
deployment step. No payout/risk optimization performed in this phase.

---

## 30. PHASE 19 — RESISTANCE→SUPPORT FLIP (2026-08-16): NOT SUPPORTED

Registered flip family A-E tested on frozen zones, original bands, first
retests, frozen sweep outcomes, registered symmetric-paired shadows.
Result (LOG P19-407/408): flip conditioning inert (retests ≡ all above-
side taps); support-side reactions slow/diffuse exactly as P8 found;
**exact-location contrast −0.009 [−0.012, −0.005] vs resistance's
+0.15/+0.20 — the location edge is one-sided.** No tap/time/definition
structure. Frequency ample (~30/session) but validity absent. The
engine remains resistance-only; manual support coverage must come from
elsewhere (or nowhere). No entries/confirmation/prop research done;
future data sealed.

---

## 31. PHASE 20 — PRODUCTIZATION (2026-08-16)

Engine frozen standalone (p20_resistance_engine_freeze.json, 9 hashes).
Workload: raw engine = firehose (130 episode-taps/day; 82-90 active
zones; nearest ≤10pt 53 % of time; alerts ~70-80/day at any threshold)
→ human product REQUIRES structural scoping: first touches 26/day
(NY-AM 6.9); scale≥1 first touches 4.6/day (NY-AM 1.0). Median width
14.6pt. Delivery: Pine cannot safely reproduce taps/lifecycle/roll →
**Python = source of truth; recommended next build = Python zone/alert
service** (frozen engine verbatim); TV parity test registered if a
chart overlay is later wanted. No strategy/confirmation/prop work done.

---

## 32. PHASE 21 — SPACING AUDIT (2026-08-16): HIGHLY REDUNDANT

79 % of adjacent live zones overlap (median edge gap −12.5pt); half are
fully contained; ~70 live zones = ~15 overlap-regions (1.8 within
100pt); 26 first touches/day = 6.2 regions; NY-AM 7.6 = 2.9 regions;
scale-2 is the only separated tier; 100 % of higher-scale zones
duplicated by lower scales. Verdict: keep engine frozen; product needs
an overlap-union DISPLAY layer before the alert service ships. No
outcomes used; no parameters optimized.

---

## 33. PHASE 22 — REGION DISPLAY DESIGN (2026-08-16)

Overlap-union presentation layer diagnosed and designed (LOG P22-411):
15 regions above price; zones:pivots 1.16 (real congregations, not just
scale copies); CORE-2 in 70 % (22pt band in 29pt parent); wide parents
= chains (73-85 % multi-core >50/75pt); NY-AM nearest region 27.6pt
wide/12.4pt core; alerts collapse 26→~6/day (NY-AM ~3). 0 invented
prices (bounds = child min/max, proven). Design: DISPLAY B (parent +
core emphasis + child detail), region-deduped alerts preserving child
state, read-only layer, membership-lineage IDs. NEXT: build the Python
region/alert layer on the frozen engine.

---

## 34. PHASE 23 — ENTRY BACKTESTS (2026-08-17): ALL NEGATIVE

Five registered families (midpoint-50 %, engulfing ×20, rejection-close,
sweep/reclaim, touch-low break) on all frozen child zones, raw +
region-deduped, 2.46M trades: every variant negative after 1.0pt costs
in every year; high-WR confirmations are lateness machines (median RR
0.15-0.40); midpoint 1:1 = 49.3 % WR vs 52.1 % breakeven. No
preregistration candidate produced. Engine untouched; no future data;
grid fixed a priori. Standing candidate remains FADE-1 (§29) for the
≥2026-08-15 one-shot test.

---

## 35. PHASE 24 — DIRECT ENTRY + FADE-1 AUDIT (2026-08-17)

Direct near-edge fixed-bracket family: REJECTED (B2 realistic fills
0/56 positive cells, tap1 and all-taps; touch-fill tap1 long-target
cells ≤ +0.024 = nil). **FADE-1 INVALIDATED (finding #6): P18's p6-
based outcome model understated adverse excursion — 22.4 % of its wins
were true stop-first losses; true-path FADE-1 ≈ −0.03R. §29 freeze
void; one-shot test cancelled.** P18's H2 (R4 wide-bracket, true-grid,
+0.193 nested, worst fold 0.00) unaffected and remains the only
standing positive construction (development-only). Product path
(zones/regions/alerts) unchanged.
