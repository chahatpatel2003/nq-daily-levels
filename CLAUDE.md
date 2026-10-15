# CLAUDE.md — NQ Daily Levels

Research-first system for producing **2–4 strong NQ zones each morning** for
06:30–09:00 PT, generated at 06:25 PT. Research phase: no edge has been
demonstrated yet, and none should be claimed without the evidence standard in
§6.

## 0. Scope — this is a LEVELS project, not a strategy project

The deliverable is a **display**: "these are the few NQ prices worth watching
today", optionally annotated with touch probability, reaction statistics,
confidence and sample size. Nothing more.

**Out of scope, do not build or research:** entries, buy/sell signals, stops,
targets, risk/reward, position sizing, P&L, prop-firm rules, trade management,
execution, or strategy profitability. Backtesting exists here for exactly one
purpose — to decide whether a zone generated before the session is a genuinely
strong price area. Reaction metrics are *measurements of the zone*, never
trades. Confirmation research comes later, after the level system itself is
proven, and is not part of Phase 2.

**No timeframe restriction.** The 1-second L1 layer is infrastructure, not the
research resolution. Any aggregation is fair game — trade-by-trade, MBP-1
events, 5 s, 30 s, 2 m, 17 m, hourly, daily, weekly, multi-session, volume
clocks, event clocks, volatility-normalised windows, custom rolling windows —
and resolutions may be combined. The only requirements are that the information
was genuinely available before 06:25 PT, can be recomputed every morning, and
improves out-of-sample zone quality.

Read `RESEARCH_PLAN.md` for the research design, `DATA_AUDIT.md` for what is on
disk, `GLOSSARY.md` for exact definitions, `EXPERIMENT_LOG.md` for what has
already been tried.

---

## 1. Orientation

```
~/Desktop/nq-levels/            this project
  configs/                      paths, research contract, CME calendar
  src/nqlevels/
    clock.py                    sessions, DST, the decision cut     <- read first
    io/raw.py                   read-only raw access
    io/store.py                 derived-layer store on T9
    build/{roll,tape,profile,daily}.py
    levels/{named,axis,width,candidates}.py
    labels/{touch,reaction}.py
    eval/splits.py              chronological splits + sealed holdout
    audit/leakage.py
  scripts/                      build_l1, sanity_l1, audit_leakage, demo_pipeline
  tests/                        69 tests, `.venv/bin/python -m pytest`
  outputs/                      reports, audits, pre-registrations

/Volumes/T9/nq data/raw/        RAW, READ-ONLY — never write here
/Volumes/T9/nq_levels/          derived layers (1.3 GB so far)
```

Python 3.12 in `.venv` (uv). Run everything from the project root.

---

## 2. The one rule

**Nothing at or after 06:25 PT may influence a zone for that morning.**

`clock.decision_time_utc(session, cal)` is the cut. `levels/candidates.py`
owns it and asserts it; every other module takes already-windowed frames and
asks no questions. If a leak appears, it came through there.

The strongest check is `audit.leakage.check_truncation_invariance`: rebuild the
session's zones from a tape physically cut at 06:25 and require them identical.
Reading the code and concluding it is causal is not the same thing.
`scripts/audit_leakage.py` runs it on real sessions — currently 7/7 audits
pass, 1,603 zones identical across 8 sessions.

---

## 3. Conventions that will bite you

* **Prices are integer ticks** (`price / 0.25`) everywhere in L1 and above.
  `tape.ticks_to_price` / `price_to_ticks` convert. Points = 4 ticks = \$20.
* **`ts_event`, never `ts_recv`.** `ts_recv` is capture time and is the *stored
  pandas index* of the raw files — it will not appear as a column unless you
  ask for it by name.
* **Aggressor: `side == 'B'` is a BUY, `'A'` is a SELL.** Verified on 39,674
  trades with 100 % separation (`DATA_AUDIT.md` §3). `'N'` is in `volume` but
  in neither leg.
* **Drop calendar spreads always** — any symbol containing `-`. Their price is
  a difference of a few points and would poison every profile.
* **Windows are half-open `[lo, hi)`**, so a 09:00 stamp belongs to the next
  window.
* **Trade date ≠ calendar date.** 17:00 CT rolls to the next day; halt holidays
  book forward. Passing a `datetime` where a trade date is expected raises —
  that silent truncation is the classic off-by-one-session bug.
* **Untraded prices are missing rows in a profile, not zeros.** Call
  `profile.densify()` before scanning for low-volume areas, or you will step
  over the emptiest prices in the profile.
* **Column suffix `_out`, `_rth`, `_target` = outcome, never a feature.**
  `build.daily.preopen_columns` is the allow-list; `check_feature_columns`
  enforces it.

---

## 4. Data facts worth memorising

* Trades 2021-01-04 → 2026-06-17, 547 M rows, 13 GB. Full pass ≈ **4 min**.
* MBP-1 2021 → **2025-06-30 only**, 7.89 B rows, 207 GB. Full pass ≈ **1 h** —
  reduce once, never inside an experiment loop.
* **1,370 sessions built**, 0 skipped, 12 missing weekdays all known holidays.
* 22 contract rolls, all decided from strictly prior sessions.
* Internal disk has ~32 GB free. **Derived data goes on T9.**

---

## 5. Splits

| split | dates | sessions |
|---|---|---:|
| train | 2021-01-05 → 2024-06-30 | 871 |
| validation | 2024-07-01 → 2025-12-31 | 369 |
| **holdout (SEALED)** | 2026-01-01 → 2026-06-17 | 110 |

Chronological, 5-session embargo. `eval.splits.holdout_sessions` **raises**
without a pre-registration token. Do not work around it. The predecessor
program consumed its whole holdout and then had nothing left to confirm its one
promising result on.

MBP-1 ends 2025-06-30, so microstructure models run on a shorter span and must
say so; `assert_mbp1_available` enforces it.

---

## 6. Evidence standard

A finding is reportable only with: sample size **and** number of distinct
sessions; the distance/regime-matched baseline it beats; a confidence interval
clustered by session (correlated nearby zones are not independent samples);
year-by-year and regime breakdowns; parameter sensitivity; and train /
validation results kept separate.

Report negatives with the same care as positives — a rejected concept goes in
`EXPERIMENT_LOG.md` so it is not re-tested. And note the difference: "PDH did
not beat its distance baseline" is a fact about PDH, **not** a fact about daily
levels.

---

## 6b. Phase 2 result — read this before proposing work

**The central hypothesis is rejected.** No named level family, volume-profile
structure, interaction history, order-book history, or learned ranker predicts
which NQ prices behave as strong zones, once distance and side are held fixed.
38 level kinds, a 1.08 M price grid, 10 reaction definitions, 12 regimes and 9
widths all return zero. Details in `EXPERIMENT_LOG.md` (P2-104 .. P2-114).

**Survives:** the E1 touch-probability baseline (calibrated, skill +0.39, mean
product calibration error 0.025) and a universal first-touch decay (+4.3 pp
within-zone) that applies to *any* price, not to levels.

**Three confounds produced convincing false positives** and any new experiment
must control for all three: level density making "at a level" vacuous; the
impossibility of a distance-matched control; and the extreme-price tautology
(the farthest price touched is the day's extreme, so it "rejects" 88.6 % of the
time). See RESEARCH_PLAN §11.

**Noise floor: |d| < 0.005** on stratified contrasts — session-level
heterogeneity alone produces ~0.003.

**Phase 3 (complete, 2026-08-15) reframed the question** to daily top-k
selection vs distance-matched same-morning controls (`eval/topk.py`). New
families tested at power: reaction memory, displacement origins/freshness,
conditional named levels, macro-release structures, HTF extremes. **Law of
the dataset: location predicts movement ENERGY, not rejection QUALITY.** Two
train-positive families (untested prior-day support +8.6 pp; pooled HTF
extremes +3.9 pp) both FAILED registered validation gates — 2021-2023
phenomena, absent in 2024-25. Kept: P3-207 engine (E1 × walk-forward
P(clean|touch), ECE 0.010/0.003, +2 pp matched-control selection edge, all of
it near-band placement + day context). Frozen proposal in RESEARCH_PLAN §13.
Holdout still sealed.

**P3-208 sweep/reclaim audit (2026-08-15)** — path-aware taxonomy (direct /
sweep+reclaim / failure; `labels/sweep.py`, `sweep_labels.parquet`). Location
identity still predicts nothing (all significant family effects negative),
but the frozen engine's matched-control edge is 2-3× larger under sweep-aware
outcomes: validation strict-cell cond +4.7..+8.8 pp (CIs > 0 at K=2/K=4),
P(≥2 of 4) +9.9 pp. `pwh` downgraded (does not extend to y_any). Holdout gate
registered in RESEARCH_PLAN §13.5.

**2026 HOLDOUT OPENED 2026-08-15 — MAY NEVER BE USED AS A SEALED HOLDOUT
AGAIN. Registered gate: FAIL** (P3-209; token 16ce7f7051547785). Strict cond
K=2 −0.022 [−0.098, +0.050], K=4 +0.002. The 2024-25 placement edge was
era-bound, like every selection effect before it. Held on unseen 2026: E1
(Brier skill +0.42), the P(clean|touch) annotation (ECE 0.014), taxonomy
frequencies (direct 21 %, combined 62 %). No durable selection edge over
matched same-morning controls exists in this framework. 2026 is historical
data now — never a holdout.

**Phase 24 (2026-08-17, complete): direct entry REJECTED + FADE-1
INVALIDATED** (LOG P24-413/414, §35; outputs/p24_*). Direct near-edge
fixed brackets: B2 trade-through fills 0/56 positive (tap1 S20/T30
−0.026, worst fold −0.082); touch-fill tap1 long-targets ≤+0.024 = nil;
MAE(1h) med 38pt. **FINDING #6: P18's p6-based outcome model understated
adverse excursion — event reconciliation showed 22.4 % of FADE-1 "wins"
hit the stop first (all 645 disagreements one-directional); true-path
FADE-1 ≈ −0.03R vs reported +0.342. FADE-1 WITHDRAWN; §29 freeze void;
one-shot cancelled.** (#5 fill backdating = minor, 0.6 %.) P18's H2
(R4 wide-bracket, TRUE grids, +0.193 nested, worst fold 0.00) stands as
the only positive construction (development-only, unvalidated). Product
path (region display/alerts) unchanged.

**Phase 23 (2026-08-17, complete): entry backtests — ALL NEGATIVE**
(LOG P23-412, §34; p23_trades.parquet 2.46M trades, 1,405 sessions).
Five registered families on all frozen child zones, raw+deduped:
midpoint-50 % 1:1 = WR 49.3 % vs 52.1 % BE → −0.047R; engulfing (20
variants) −0.015..−0.21 (closes below zone ~50 %, RR<0.5 ~half);
rejection/D2/E: WR 73-85 % but RR 0.15-0.40 → −0.03; D1 RR 1.4/WR 41 %
→ −0.07. Negative EVERY year, every variant; dedup changes nothing;
no tap/scale pocket (best cell = tap1 midpoint −0.002). Mechanism:
reaction spends itself in the first seconds; bar-close confirmations
are lateness machines; midpoint fills adversely selected. Nothing
preregistered from this phase; FADE-1 (§29) remains the standing
candidate. Engine untouched; 0 chronology violations.

**Phase 22 (2026-08-16, complete): region display design** (LOG
P22-411, §33; examples outputs/p22_example_regions.csv). Overlap-union
regions: 15.2 above price, width med 28.7pt; zones/region 4.6 but
**unique pivots 4.0 (ratio 1.16 — real pivot congregations)**; CORE-2
in 70 % (22pt band); wide parents = chains (>50pt: 73 % multi-core);
nearest region 32pt away/25pt wide/11pt core; NY-AM clean (1.8 regions
within 100pt). Alerts collapse 26→~6/day (NY-AM ~3). 0 invented prices
(bounds = child min/max, proven). Design: DISPLAY B (light parent +
emphasized cores + child edges on demand), label `R lo-hi | nZ/npiv |
Smax | n untouched | core`, region-deduped alerts carrying exact child
state, read-only layer, membership lineage. NEXT: build Python
region/alert layer.

**Phase 21 (2026-08-16, complete): spacing audit — HIGHLY REDUNDANT**
(LOG P21-410, §32). 79 % of adjacent live zones OVERLAP (med edge gap
−12.5pt); 51 % fully contained; stack ~10 deep; ~70 live zones = 15.2
overlap-regions (≤10pt: 9.1); within 100pt: 5.4 zones = 1.8 regions;
coverage 41 % of next 100pt; walls med 29pt; clean gaps med 13pt.
EVENT collapse: 68 approaches/day→5.2 regions; 26 first-touches/day→
6.2; NY-AM 7.6→2.9 (≤10pt 2.1). Scale-2 only separated tier (+6.8pt
gap, 40 % ov); 100 % of scale-1/2 zones duplicated by lower scales.
P20 abundance restated: ~5-6 distinct regions approached/day, NY-AM
2-3. NEXT: keep engine frozen + build overlap-union DISPLAY layer
before the alert service.

**Phase 20 (2026-08-16, complete): productization** (LOG P20-409, §31;
freeze outputs/p20_resistance_engine_freeze.json). Raw engine =
firehose (130 episode-taps/day, 82-90 active zones above, nearest ≤10pt
53 % of time, alerts 70-80/day at any threshold) → usable product needs
structural scope: first touches 26/day (NY-AM 6.9/day, med 4); scale≥1
first touches 4.6/day (NY-AM 1.0). Width med 14.6pt. Pine CANNOT safely
reproduce taps (1s re-arm), 30-session state, or the causal roll chain
→ **Python = source of truth; next build = Python zone/alert service**
(frozen engine verbatim; parity-test design registered if TV overlay
wanted).

**Phase 19 (2026-08-16, complete): resistance→support flip — NOT
SUPPORTED** (LOG P19-407/408, §30). Five preregistered acceptance
definitions (1m-close/+5pt/+10pt/60s-dwell/engine-F): all INERT —
flipped first retests ≡ all above-side taps (P+10 0.80 both; +10<5m
0.16 vs resistance 0.48; med 1,542s vs 266s). Registered symmetric-
paired shadow contrast: **−0.009 [−0.012,−0.005]** vs resistance's
+0.15/+0.20 — the exact-location edge is ONE-SIDED. No tap/time/depth
structure (depth signal = outcome contamination). P8 "support diffuse"
replicates at flipped locations. Frequency ample (~30/sess) — validity
absent. Note: first analysis pass had a left-join/NaN denominator bug
(43 % p6 coverage) + ×4 depth units — corrected, documented. Zone
engine = resistance-only instrument.

**Phase 18b (2026-08-15): ADVERSARIAL DATA AUDIT of holdout — CLEAN**
(P18-406: 51 market files/0 missing/0 dupes; independent struct-level
DBN parser 0 mismatches; independent 1s bars identical maxΔ 0.0000 on
holdout AND dev controls; dup rates = dev corpus (MDP3 multi-fill
property); 0 cross-contract bars; roll causal (NQU6 from 06-16); ATR
0.000000 under frozen convention; VWAP/R4/zones exact; no raw old-new
overlap exists BY DESIGN; bad-data hypothesis REJECTED — Balanced's
failure stands as a strategy result).

**Phase 18 (2026-08-15, complete): autopsy + redesign — FADE-1 frozen**
(LOG P18-402..405, §29). Autopsy verdict E-MIXED: resistance/location
edge SURVIVED (+0.201 vs shadows on holdout, stronger than dev); R4
continuation premium decayed to ~breakeven under 12.5/30; tap/fire/mono/
HGB stack INVERTED OOS (rule −0.126 vs complement −0.018 = selection
overfit); loss composition unchanged; no regime explanation (fire-
density analogues profitable in dev); winner's curse ≈ +0.10-0.20R of
the 0.4R gap. Stage B (2 registered families, 43 cells, all reported,
nested 10-block walk-forward): H2 R4-wide-bracket viable (+0.193, worst
0.00); **H1 FADE-1 WINS: first-touch (tap1) fade at near edge, ≥2t
trade-through fill, S20/T30, no R4/filters/model → pooled +0.342R, PF
1.73, WR 55.7 %, worst fold +0.200, 10/10 blocks, +0.5pt robust,
shadow-superior EVERY block (+0.26..+0.50). Frozen sha256 8b969a07.**
ZERO external validation (holdout A was development); needs production
1s ledger + streaming proof + one-shot sealed test on ≥2026-08-15 data
(PASS: mean ≥ +0.10R, CI>0, PF ≥ 1.15 — registered P18-405).

**Phase 17B (2026-08-15, complete): HOLDOUT OPENED — FROZEN BALANCED
FAILED** (LOG P17-401, §28.2; ledgers + refit binary in
sealed_holdout_1/). Integrity+pipeline proven exactly (manifest 53/53,
freeze 23/23, June bars identical, mechanical warmup ledger 392/392
bit-equal). FINDING #4: HGB binary never persisted; cross-process refits
non-reproducible (91 % selection fidelity) — binary now persisted.
RESULT (40 sess, 2026-06-18..08-14): Balanced N 140, WR 27.9 %,
−0.133R, PF 0.83, −18.6R, all months negative, 0th pctile vs ALL dev
140-trade windows (dev p5 = +0.159R); mechanical core confirms (N 167,
−0.103R, PF 0.87) — the EDGE failed, not the model layer; hit-rate
collapse with intact payoff mechanics; fire rate UP (3.5/sess). Prop
replay: 0 payouts, 2 deaths. **VERDICT: FAIL. No retuning performed.
HOLDOUT A spent. Live deployment NOT justified. Any research reopening
= new phase; only HOLDOUT 2 (2026-12-18+) can validate it.**

**Phase 17A (2026-08-15, complete): sealed-holdout plan — NO DATA
OPENED** (LOG P17-399/400, §28 BINDING; HOLDOUT_FREEZE_MANIFEST.json 23
hashes; SEALED_HOLDOUT_REGISTRY.json). Cutoff 2026-06-17T03:03:02Z
(partial session). Only 40 post-cutoff sessions exist; 6-mo holdout
(127 sess, ≈279 trades) completes 2026-12-17. Purchase: GLBX.MDP3
trades ONLY, parent NQ.FUT, from 2026-06-17T22:00Z (~0.41 GB/mo); MBO/
MBP-1 NOT needed. Pre-registered verdict: PASS = N≥200 & mean≥+0.15R &
PF≥1.20 & clustered 90 % CI>0; FAIL = mean≤0 or PF≤1 @N≥200; else
borderline→extend. No-retune rule binding. Cost API needs user key —
exact request handed over; awaiting purchase authorization.

**Phase 16 (2026-08-21, complete): Balanced audit CLEAN + risk optimum**
(LOG P16-396..398, §27; canonical p16_balanced_canonical.parquet +
p16_streams_canonical.pkl + frozen scores). Stage A: independently
reproduced — finding #3 (HGB refit jitter + unstable tie sort; fixed:
frozen scores + stable keyed sort; canonical = stored P15 elementwise);
ref-sim 20/20 vs engine; accounting identity exact; deaths 0.335/yr raw;
price-level P&L 0.000000 pts; UW 36 defn verified; MC seed-stable
±$191, stationary-bootstrap agrees. Worst 36-sess UW held 2 payouts
($9k @ $500). Stage B: gaps ~risk-invariant (med 13, p90 23); survival
= binding axis (pay#1: 84% @$500, 73% @$650, 43% @$1500); std cap
ceiling $75k; promo peak $133k @$1300 but DLL guts ≥$1400 and survival
50%. **RECOMMENDED: $650 (26 MNQ), promo/DLL, C=12R, immediate 50% →
$101k/yr mean, p10 $65k, deaths 0.48, chron $377.6k/0 deaths; 60%-edge
$68k. Zone $600-750.** NOTE: user selected BALANCED over P15's B_t40.

**Phase 15 (2026-08-21, complete): legal-strategy economics — B_t40
WINS** (LOG P15-393..395, §26; p15_streams.pkl). Five §25.2 ledgers
verified exactly; cushion knee re-derived = 12R on legal streams. At
$500/2NQ (std $5k cap): B $66.6k/yr > E $64.7k > A $63.2k > C $56.9k >
D $47.7k; first payout 17-24 sess; chronological 22-26: zero deaths all
five, B $315k total, all years positive. DLL($3k)+promo($10k): B@$500 →
$91.6k (+37 %); at $1000 all ≈$120k but DLL=3R interferes → cap size at
$500-600. STRUCTURAL: failure prob is ~scale-free in risk$ (cushion-
governed: 12→24R gives 0.37→0.24, plateau = early 9R-initial-buffer
deaths); Ultra's 10R DD does NOT monetise (deaths cost $150). Per-acct
at $500: med 39-48 payouts/$159-209k before death; P(die pre-payout-1)
10-26 %. Underwater: B worst (96 sess, 73 % share), E best (36). Asia ≈
wash, kept. **SELECTED: B_t40 $500-600 + DLL/promo; runner-up E_hybrid
for bleed-aversion.** Live forward test still the true unknown.

**Phase 14 (2026-08-20, complete): leakage audit + LEGAL strategy —
STRONG IMPROVEMENT** (LOG P14-386..392, §25; p14_fire.parquet 14,198
fires; tests/test_chronology.py permanent). Audit: project-wide sweep →
only 2 leaks ever: P13 gate chronology + FINDING #2 (legacy immature
prior-outcome counts, 1,554 taps 0.455% — exact recompute; first diff
estimate 27.5k was false-positive-inflated; P14 layer exact-clean).
Truncation invariance PASS; streaming replay IDENTICAL (0/137). The
55.7R baseline DD = regime bleed BY POPULATION: slow/choppy/late-tap
fires (anti-rule ≈0R, 400+-sess underwater); recent-health predicts
NOTHING (flat quintiles); ck0 dominates (waiting = price decay, no
info); ML weak (AUC ≤0.57). **REGISTERED PRIMARY: tap≤3 + fire_s≤24s +
mono≥0.27, decide at fire bar, enter next second, S12.5/T30 → n 2,731
(2.5/d), WR 41%, +0.310R, PF 1.49, DD 21.4R, MC p95/99 22.8/29.5, uw
max 76 sess, worst month −8.0, all years positive.** Plateau: T40
+0.386/PF 1.55; tap≤2 +0.353/DD 15; tap1 +0.427/PF 1.71/DD 10.3;
hybrid +0.376/uw 36. Next: paper-trade (authorization needed).

**Phase 13 (2026-08-20, complete): F4+tap≤3 pre-implementation audit —
CRITICAL LOOK-AHEAD FOUND; VERDICT NOT READY** (LOG P13-381..385, §24;
canonical ledger `p13_ledger.parquet` 3,835 trades, reproduces to 0.0000
pts from 1s paths). **The frozen F4 gate is not computable at its own
entry time**: p_both60 uses cp60 features (touch+60s) but R4v10 fires at
median 16s and E1d fills at fire+3s → 98.3 % of entries precede the gate
window (mean look-ahead 38s). Live-legal (enter touch+62/63): +0.04R,
PF 1.05 — edge collapses (price falls ~7.9 pts during the wait). Ungated
R4v10 tap≤3 fire+3 (fully legal): +0.191R, PF 1.28, all years positive —
the genuine residual. **P10 F4 OOS, P11 wrappers, P12 lifecycle dollars
WITHDRAWN as live expectations** (they describe the non-implementable
stream). F3 (stall, fires 300s) chronologically clean. Audit profile of
the simulated stream: all years +; OOS 24-26 +0.459R; DD 17.4R; MC
p95/p99 19.7/24.3R; 96 % pos months; tap plateau smooth; holidays n=8
negligible; FOMC/NFP days fine; breakeven slippage 5.66 pts. Phase-14
(needs authorization): fire-time confirmation model (features AT the R4
fire second), staged S/T re-selection on the legal stream, re-audit.

**Phase 12 (2026-08-19, complete): funded-account lifecycle economics**
(LOG P12-374..380, §23.1/.2; engine `p12_engine.py`, streams
`p12_streams.pkl`, outputs/p12_*.json; F4 untouched). Objective = max net
cash withdrawn, deaths allowed. Topstep 150K XFA snapshot modelled (EOD-
trailing MLL −$4.5k→locks $0, 5×$150 winning days, ≤50 %/payout, caps
$5k/$10k-promo, split 100 %→90 % after $10k). Findings: extraction
ceiling = payout freq × cap (~25/yr → ~$240k/yr max); risk pays
monotonically to $600 (replacement = subsidised drawdown); Standard >
Consistency (40 % rule blocks ~164 d/yr); $3k DLL never binds ≤$175 →
promo caps free; policy = immediate first payout, always 50 %, retain
18R×risk cushion (deaths 1.2→0.2/yr free); at matched failure tap≤3
extracts 2-5× raw/cap8; dynamic risk dominated; replacement cost/delay
≤3 % effect; chron 2022-26: $90-197k banked pre-Aug-23-DD in every
config; P(net year < 0) ≈ 0 everywhere. RECOMMENDED: tap≤3, $300→$400
risk (12-16 MNQ), C=18R → $97-126k/yr mean, p10 $69-91k. Caveats:
backtest fills, rule snapshot, no live automation yet.

**Phase 11 (2026-08-18, complete): the risk layer** (LOG P11-370..373,
RESEARCH_PLAN §22.2; F4 untouched; stream `p11_stream.parquet`). Audit:
universe ~46 interactions/day; fire share era-varying 13-37 %; F4 executed
7,898 trades 2022-26. Max DD 28.8R ≈ **p95 of a 250-day window** (block-MC
p99 34.8R) → design DD 35-40R, never /27. DD are multi-week bleeds → **daily
wrappers don't pay** (only −6R emergency stop ~free; profit caps
catastrophic — the edge lives in +130R tail days; 163-trade day noted as
practically inexecutable; cap-8 = practical profile, MC p95 24.3R).
One-open-trade protective (29R vs 48R). No within-day sequence info.
Optional tap≤3 overlay: +0.454R/17R DD at −41 % volume. Extraction: risk =
buffer/30-36 ≈ $125-150 on $4.5k → ~1-4 %/yr violation, E ≈ $74-88k/yr
(harsh confiscation model); real firm rules still required. 18-loss streak
= ONE session (2026-05-04). Months 94 % positive; worst month −12.4R.

**Phase 10 (2026-08-18, complete): the strategy CONVERTS** (LOG
P10-360..363, RESEARCH_PLAN §21.2; levels+confirmations untouched;
selection 21-23, executed unchanged 24-26, net costs, conservative fills).
**Recommended F4: short on R4v10 at a frozen zone ∩ calibrated p ≥ 0.85,
market entry, stop 12.5 pts, target 30 pts → OOS +0.376R / +4.70 pts/trade,
WR 0.429, PF 1.61, DD 27R, 9.5 tpd, every year +0.29..+0.44R, every session
positive, stress-cost +0.296R.** High-WR alternate F3 (S30/T15): 72 % WR,
+0.05R, 6/6 years. Rejected: stall-as-trade (negative), ungated framework
(≈0), pullback/retest entries (adverse selection). Immediate entry wins;
delay-insensitive; plateaus flat. Sizing/execution automation not
researched.

**Phase 9 (2026-08-17, complete): confirmation SOLVED and frozen**
(LOG P9-350..354, RESEARCH_PLAN §20.2; levels untouched). Information
arrives post-touch (pre-touch AUC 0.582 → 0.787 @60 s → 0.914 @5 min, ECE
0.018). **Frozen rule: confirm on 10-pt velocity break within 60 s OR
5-min shallow stall (pen ≤ 5 pts, below edge) gated by the 120-s acceptance
disqualifier (which alone predicts 69 % failure).** Framework: fires 44.8 %
of touches, 0.809 strict / 0.841 sweep-incl precision (base 0.589/0.670),
latency med 60 s, adverse-before-conf med 2.3 pts, remaining med 20.5 /
mean 53.2 pts (66 % of move); years 0.758-0.868; segments 0.728-0.874;
R4v10 alone: 0.857, six years within 4 pp. Reclaim-hold confirmation is a
TRAP (0 pts median remaining; sweep-mode unconfirmable). MBP-1: zero
increment (third rejection). Model barely beats the simple rules — simple
rules recommended. Next possible phase: strategy test (stops/targets/P&L)
— NOT started.

**Phase 8 (2026-08-17, complete): support-only phase → OUTCOME D** (LOG
P8-340..344, RESEARCH_PLAN §19.2; resistance engine untouched). Sweep depth
unpredictable (6-10 % interval narrowing vs 20 % bar; ATR scaling ~5 %);
frozen vol-scaled band +1.9 pp (21-23) → +0.7 pp (24-26) FAIL;
reclaim-validated old zones −4.9 pp; sweep-born pivots' first retest exactly
0.0, later taps −5.9. Support characterised: success 0.307/0.125 strict;
reaction med 31.4 pts through adverse med 23.2 (ratio 1.48 vs resistance
4.9); 89.7 % sweep-mode; 14.2 successes/day exist — the failure is purely
locational. **Final product: asymmetric — exact resistance levels (§18.2) +
state-based support context only. No manufactured support lines.**

**Phase 7 (2026-08-16, complete): resistance hardened, support closed,
methodology FROZEN** (LOG P7-330..336, RESEARCH_PLAN §18.2). The
resistance-pivot anchoring edge passed a 40/40-cell adversarial battery
(+5.4..+9.7 pp strict in every slice: vol/trend/volume/distance/width/
nesting/confirm-delay/age/tap-no/scale/flip/roll/release/outcome-defs;
sweep-inclusive +3.1 on this side); definition-robust (81-95 % of 5m-1H
fractal pivots within 1.5 pts of zigzag zones); confirmation median 7 min
after extreme, lead-to-tap median 9.6 h. **Pivots rank within themselves**
(prior-success +3.6, rested +3.8, dep-velocity +3.3, all 6/6 yrs;
walk-forward top-2 +9..+17 pp vs seg+distance-matched). **Support: NO exact
anchor** — pivot low is a local valley (−3.1 strict both eras); turns are
diffuse reclaims med 15.5 pts below (92 % sweep-mode); sweep-cluster
anchors null in freeze AND confirm eras. Product (offline 2022-26):
resistance display 78 eps/day, strict 0.57|touch, 8.7 successes/day,
displayed-anchor lift +3.0 pp. Frozen methodology in §18.2; indicator build
awaits the project owner.

**Phase 6 (2026-08-16, complete): the exact-price question — FIRST positive
spatial result, 6/6-year stable** (`labels/enrich.py`, `build_p6_measure.py`,
127 tests; shadow anchors ±2.5-20 pts, one walker, placebo-verified). The
exact pivot doesn't react more often (−0.6 pp sweep-inclusive) but reacts
more tidily and farther: **resistance side strict +7.7 pp [7.2, 8.2] CI>0
all six years, +4.9 pts magnitude; support side −3.2 pp**; width band±5-10
pts; anchors known median 9.9 h pre-tap; value is ANCHORING (spatial
features add 0 to tap-ranking; P5 selection conclusions stand). Magnitudes:
reaction med 31.8 / mean 63.6 pts; adverse med 11.6 / mean 14.6 pts
(3.0/11.2/23.9 by mode); ratio 3.7×; deeper sweeps → bigger reactions. TOD:
one stable plateau 05:00-11:30 PT; weak 14:00-22:00; magnitude flat by
hour. See RESEARCH_PLAN §17, LOG P6-320..325.

**Phase 5 (2026-08-16, complete): true live display simulation**
(`features/display.py`, `run_p5_live.py`, `p5_pool/leads/races.parquet`;
123 tests). 4.35 M scored refresh×zone rows, 97,512 episodes, 2022-2026
walk-forward. **The Phase-4 +9 pp zone-choice edge does NOT survive
same-moment side+distance-matched controls (−0.6 pp [−1.2, +0.1]; ≈0 every
year; next-ranked ≈ displayed)** — it was moment/timing selection. What
holds: display beats naive choices by +13-16 pp conditional; 10.4 successful
reactions/day, 99.3 % of days ≥1; median zone 0.40 ATR, shown 16 min before
touch; advance-notice AUC 0.61→0.68 (30→2 min); side asymmetry REAL (races
symmetric; resistance-side wins by direct rejection 0.496, support-side only
by sweep+reclaim 0.333); display is 90.8 % resistance-side; 10 % hysteresis:
60-min median display at −1.1 pp. See RESEARCH_PLAN §16.2, LOG P5-310/311.
Product = state-driven display, not special prices. Build decision awaits
the project owner.

**Phase 4 (2026-08-15, complete): all-session (zone, interaction) framework**
— 61k zones / 342k taps / dedup 126k; P(success|touch) with causal pre-tap
features (`features/zones.py`, `features/interactions.py`,
`interactions.parquet`, `p4_scores.parquet`). Dominant fact: approach side
(resistance-taps 0.670 vs support-taps 0.307, every year). Then tape state
(atr_pctile +10 pp, rv60 +7). Zone-specific third-order but real:
prior-success +2.9, recency +2.9, departure velocity +2.2; flips −4.9;
tap-count/profile-structure/HTF-scale dead. **First era-stable selection
skill: per-session top-2 by walk-forward score 0.790 vs seg×side-matched
0.701 (+9.0 pp, CI>0, positive all 5 OOS years; strict +12.0 pp).** See
RESEARCH_PLAN §15, EXPERIMENT_LOG P4-300..306. Awaiting direction on the
live scorer/display build.

---

## 7. Two structural facts that shape everything

1. **Touch probability is monotone in distance.** Reaching a price means
   passing every price in between, so a raw touch rate mostly measures how
   close the level was (92 % inside 0.25 ATR, 2 % beyond 1 ATR). Model
   **attraction** — touch minus the distance-matched baseline — never raw
   touch. `levels.axis.attraction_residual`
2. **Reaction has no such structure** and is where the edge most plausibly
   lives. It is also what the predecessor program never studied: that program
   tested entry rules, not level quality.

Constraint inherited from it: NQ friction is \$10–20 per round trip and its
1-minute signals produced only \$3–8 of gross edge. So a 6-point reaction is
noise. The outcome surface is evaluated at 10/15/20/30 points.

---

## 8. Relationship to the earlier program

`~/Desktop/NQ (TOPSTEP)` is a completed, closed research program with a
validated leakage-proof platform and an honest overall negative on 1-minute
entry rules. From it this project **reuses**: the empirically measured CME
holiday calendar (vendored as `configs/cme_calendar.yaml`), the causal
front-month roll pattern, and `outputs/nq_bars_full.parquet` **as an
independent cross-check only** — the L1 sanity gate reproduces it to
0.000000 max OHLC difference and 100 % identical volume.

It **does not inherit** that program's strategy assumptions, its objective
function, or its conclusions about what cannot work. Different question.

---

## 9. Commands

```bash
.venv/bin/python -m pytest -q                    # 69 tests
.venv/bin/python scripts/build_l1.py --resume    # rebuild L1 (10 min full)
.venv/bin/python scripts/sanity_l1.py --months 2024-03,2025-06
.venv/bin/python scripts/audit_leakage.py --months 2024-03,2024-11
.venv/bin/python scripts/demo_pipeline.py --start 2024-03-05 --end 2024-03-08
```

---

## 10. Scope discipline

* No paid service, subscription or hosting is assumed. Do not add one without
  first writing down the expected value it would add.
* Deployment target is deliberately undecided — the research determines it.
* Do not run large uncontrolled searches before the baseline (E1) and width
  study (E2) exist; without them a search has nothing to be scored against.
