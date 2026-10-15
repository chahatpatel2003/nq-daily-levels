# Formal terminology

Every term below has exactly one meaning in this project, and each is
implemented in one place. Where a definition was chosen over an obvious
alternative, the reason is given — those are the decisions most likely to be
silently reverted later.

Prices are stored as **integer ticks** (`price / 0.25`). "Points" means index
points (4 ticks, $20 per NQ contract).

---

## Time

**Session / trade date**
The CME trade date. Globex opens 17:00 CT and that evening belongs to the
*next* calendar day; the session closes 16:00 CT. Weekends and holidays roll
forward. `clock.trade_date`

**Halt holiday**
A holiday the exchange trades a shortened morning of, and books to the next
trade date (MLK, Presidents', Memorial, Juneteenth, July 4, Labor,
Thanksgiving). Not a trade date. Its activity belongs to the following
session's overnight.

**Decision cut** — `T_decide`, default **06:25 PT**
The instant a live morning run produces its zones. *Nothing at or after this
stamp may influence a candidate zone or any pre-open feature.* The single
invariant the project rests on. `clock.decision_time_utc`

**Target window** — **06:30–09:00 PT**, half-open
The 2.5 hours all touch and reaction labels are measured in.
`clock.target_window_utc`

**Segments** (all end at or before the cut)
`globex` = the whole session so far, from the real Globex open;
`extended` = the part before the ordinary night begins (non-empty only after a
halt holiday); `asia` 15:00–00:00 PT; `london` 00:00–05:00 PT;
`premarket` 05:00 PT–cut. `clock.segment_windows_utc`

---

## Objects

**Level**
A single price with a provenance — `pdh`, `on_low`, `pd_poc`, `round_100`,
a swing, an HVN. Un-widened. `levels.named.Level`

**Candidate zone** (or just **zone**)
A closed price interval `[lo, hi]` in ticks, generated **before the decision
cut**, with a family label. A zone, never a line: price interacts with a
region, and a one-tick level makes every genuine reaction look like a miss.
`labels.touch.Zone`

**Zone width / half-width**
`half_width = (hi - lo) / 2`. A research object, not a constant — see
`levels/width.py`. Families compared: fixed ticks (the control), ATR fraction,
overnight-range fraction, realised-vol scaled, empirical penetration quantile,
volume-node width.

**Near edge / far edge**
Relative to the approach side. For a zone *above* the reference the near edge
is `lo` and the far edge is `hi`; below, the mirror. All reaction measurement
is anchored to these, not to the touch price — the edges are known before the
open, the touch price is not.

**Approach side**
`above` / `below` / `inside`, from the zone's position relative to the
**reference price**. `labels.touch.zone_side`

**Reference price** — `ref_t`
The last trade strictly before the decision cut. Distance is measured from
what a live run would actually have seen.

---

## Touch

**Touch**
A second whose *bar range* intersects the zone: `low <= hi and high >= lo`.
Deliberately the range and not the close — a wick through a level fills a
resting order, and requiring a close inside would systematically discard the
fast rejections this project is looking for. `labels.touch.touch_mask`

**First touch**
The earliest touching second inside the target window. Everything downstream
is measured forward from here.

**Repeat touch / re-entry**
A subsequent entry into the zone after having left it. Counted separately;
first touch and third touch are different events and are never averaged
together.

**Open-inside**
The zone already contains the 06:30 price. There is no arrival, so these rows
are excluded from touch models and flagged in reaction models.

**Destination hit**
Touched within a stated horizon (15/30/60/90/150 minutes).

**Attraction** — the metric that actually matters
Observed touch **minus the distance-matched baseline** probability.
Touch probability is monotone in distance — reaching a price necessarily means
passing every price in between — so a raw touch rate mostly measures how close
the level was. A level type that cannot beat its own distance curve has
demonstrated nothing. `levels.axis.attraction_residual`

---

## Reaction

All measured from the first-touch second forward, truncated at the end of the
target window; `n_seconds` records how much window actually existed.

**Rejection excursion** — `rev`
Distance price travelled back past the **near** edge (down for a zone met from
below). Zero while inside the zone.

**Continuation excursion** — `cont`
Distance past the **far** edge. Zero while inside the zone.

> Movement *inside* a zone is neither reaction nor failure — by construction
> the zone is the band where price counts as "at the level". Anchoring adverse
> movement at the near edge instead scores a zone as failed the instant a fast
> touch prints through it, which would condemn every wide zone before the width
> study has run.

**MFE / MAE**
Maximum favourable / adverse excursion. Reported per horizon
(1/3/5/15/30/60 min) in each direction, as `mfe_rev_*` and `mfe_cont_*`.

**First passage** — `fp_rev{K}_adv{M}`
`1` if a K-point rejection arrived before an M-point continuation, `0` if not,
**`None` if neither resolved** inside the window. Unresolved cases are dropped,
never scored as failures.

**Penetration depth**
Maximum distance past the **near** edge, in ticks. A tap scores near zero; a
blow-through scores more than the zone's own width.

**Traversed**
Price cleared the far edge at all.

**Acceptance**
Price traded *and stayed* beyond the far edge — longest run of closes past it
≥ `accept_seconds` (default 300 s). Distinguishes a break from a poke.

**Reclaim**
After traversing, price closed back past the opposite edge of the zone.
Recorded with `secs_to_reclaim`. `None` when there was no traversal to reclaim
from.

**Chop metrics** — all over a **fixed** window (default 900 s), so a 06:35
touch and an 08:55 touch are comparable:
- `secs_inside`, `secs_inside_share`
- `n_crossings` — full side-to-side transitions, ignoring dwell time
- `n_reentries` — returns into the zone
- `directional_efficiency` — |net displacement| / path length, sampled every
  ~30 s. At 1-second resolution the path is dominated by bid-ask jitter and the
  ratio collapses toward zero for everything; with fewer than 8 samples it is
  `NaN`, because any two points are a straight line and would score a
  misleading 1.0.
- `rev_velocity_tps` — rejection ticks per second over the first minute

**Clean rejection** — the headline label
Conjunctive by design: a real move (`rev K` before `adv M`) **and** not
achieved by loitering (`secs_inside_share` and `n_crossings` under thresholds).
Thresholds are arguments, not constants. `labels.reaction.clean_rejection`

**Chop score**
0 = clean impulse away, 1 = pure chop. A blunt composite for ranking and
eyeballing; models get the raw components.

---

## Structure

**Volume-at-price profile**
Traded volume by price tick over a window, summed from 30-minute buckets. A
*transaction* profile, not a TPO/time profile. Untraded prices are **absent
rows, not zeros** — anything scanning for low-volume areas must `densify()`
first, or it steps straight over the emptiest prices in the profile.

**POC / VAH / VAL**
Most-traded price; edges of the smallest contiguous band holding 70 % of
volume, grown from the POC. Standard market-profile construction, not a
quantile of the volume distribution — they differ on bimodal days.

**HVN / LVN**
Prominent local maxima/minima of the smoothed profile, where *prominence* is
distance from the local neighbourhood mean normalised by it, so it is
comparable across sessions of very different volume.

---

## Contracts

**Front month**
The highest-volume quarterly outright of session *S-1*, used for session *S*.
Monotone — never rolls backwards.

**Roll**
A change of front month. 22 in 2021-01 .. 2026-06. A *bootstrap* (the first
selection of a run) is logged separately and is not a roll.

**Carry**
How a prior-session level is expressed in the current contract's scale:
`same_contract`, `incoming_contract_own_prints` (preferred),
`spread_shift`, or `unavailable`. Recorded per session.

---

## Evaluation

**Train / validation / holdout**
2021-01-05 → 2024-06-30 / 2024-07-01 → 2025-12-31 / 2026-01-01 → 2026-06-17.
Chronological, never shuffled, 5-session embargo at each boundary.
871 / 369 / 110 sessions after embargo.

**Sealed holdout**
Readable only against a pre-registration token — a hash of a frozen statement
of the rule, parameters and pass/fail gates, written before the data is
touched. `eval.splits.holdout_sessions`

**Baseline**
The distance-matched touch curve, and the simple alternatives any method must
beat: round numbers only, PDH/PDL only, equally spaced levels, random
same-distance zones.

**Prefix invariance**
A causal column's value at row *i* is unchanged when every row after *i* is
deleted. `audit.leakage.check_prefix_invariance`

**Truncation invariance**
Candidate zones are identical whether built from the full session tape or from
one physically cut at the decision instant. The strongest leakage check,
because it does not depend on reading the code correctly.
`audit.leakage.check_truncation_invariance`
