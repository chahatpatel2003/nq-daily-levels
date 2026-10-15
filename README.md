# NQ Daily Levels: Can You Know in Advance Which Price Levels Will Hold?

A research system for one practical question: **at 06:25 PT, before the New York open, can we pick 2–4 NQ price zones that the market will react to that morning?**

Over 25 documented research phases it built a leakage-proof zone engine, a live display simulation, a trade-conversion test and a prop-firm lifecycle model. It then spent two sealed holdouts. **The answer was mostly no.** The project records exactly where it is no, and the one narrow place where real structure was found.

| | |
|---|---|
| **Instrument** | NQ futures, 1-second bars and volume profiles built from Databento trades and MBP-1 (2021 → 2026) |
| **Stack** | Python 3.12, pandas, NumPy, SciPy, scikit-learn, PyArrow, pytest |
| **Scale** | ~26k lines of Python · 103 analysis scripts · a 3,000-line experiment log |
| **Tests** | 135 pytest tests: 128 pass, 7 skip without the market data |
| **Outcome** | The central hypothesis was rejected. Both sealed holdouts failed. One robust *location* effect survives at resistance, but it does not convert into a tradable edge |

---

## The rule everything rests on

> Nothing at or after **06:25 PT** may influence a zone for that morning.

`scripts/audit_leakage.py` proves this by rebuilding the zones from a tape **physically truncated** at the cut-off and requiring an identical result. Holdout sessions are sealed in code and registered in [`SEALED_HOLDOUT_REGISTRY.json`](SEALED_HOLDOUT_REGISTRY.json) and [`HOLDOUT_FREEZE_MANIFEST.json`](HOLDOUT_FREEZE_MANIFEST.json). Each opening was pre-registered with a token written before any holdout row was read.

## What was found

| Phase | Question | Result |
|---|---|---|
| 1–2 | Do named levels (38 kinds), a 1.08 M price-axis grid, learned rankers or pre-open order-book history identify strong zones? | **Rejected.** Every apparent effect traced to a specific confound: distance, the extreme-price tautology, conditioning on "touched", or first-touch decay that applies to *any* price |
| 2 | What survives? | A calibrated **P(touch \| distance, volatility)** model with validation Brier skill +0.39 and ECE 0.014, stable 2021–2025. First-touch decay of +4.3 pp, a property of all prices |
| 3 | Conditional reaction ranking, tested once on the **sealed 2026 holdout** | **FAIL.** Strict conditional lift at K=2: −0.022 [−0.098, +0.050]. Before unsealing, the frozen engine reproduced its validation scores exactly (max diff 0 over 347,182 rows) |
| 4–5 | Strong vs weak zones; a true live display simulation | The Phase 4 edge was timing selection. In the live frame, zone choice adds ≈ 0 (−0.006 [−0.012, +0.001]). The display still beats naive choices by 13–16 pp |
| 6–7 | Does the exact price matter? | **First positive spatial result, 6 of 6 years.** Real resistance pivots produce tidier rejections than matched shadow prices: +7.7 pp [+7.2, +8.2]. A 40/40 adversarial battery passed and placebo pivots were negative. Support has no exact anchor |
| 10–16 | Convert the frozen resistance zones into a strategy; risk wrapper; Topstep funded-account economics | A strategy was built. Then the Phase 13 pre-implementation audit found **look-ahead in the F4 gate**, giving the verdict *NOT READY*. A project-wide leakage audit and a legal fire-time redesign followed |
| 17 | The redesigned frozen strategy on a newly purchased **sealed holdout** (40 sessions) | **FAIL.** Expectancy −0.133 R, PF 0.83; the failure was hit-rate collapse |
| 18–25A | Failure autopsy, support flip, productization, entry backtests, clean first-passage measurement | All five entry families negative. Candidate FADE-1 was invalidated by audit. Zones mark real locations but do **not** produce clean rejections: +10/−5 before-stop rates sit at breakeven |

**Final product decision** (from the project's own log): keep only a minimal historical zone viewer. No alerts, no live build. The full reasoning is in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md), and every experiment is marked `KEPT`, `REJECTED` or `OPEN`.

## How the research was run

- **[`RESEARCH_PLAN.md`](RESEARCH_PLAN.md)** separates *destination* (will price get there?) from *reaction* (what happens when it does?), defines "strong" quantitatively, and queues experiment families in order.
- **[`GLOSSARY.md`](GLOSSARY.md)** gives exact definitions for zone, touch, reaction, chop, reclaim and sweep.
- **[`DATA_AUDIT.md`](DATA_AUDIT.md)** measures what was on disk, including the Databento aggressor convention and MBP-1 seam duplication.
- **Pre-registrations** are in `outputs/preregistrations/`. Frozen configs are in `configs_holdout/`. Results are in `outputs/` as JSON and CSV.
- **[`CLAUDE.md`](CLAUDE.md)** is the working context and conventions file used with an AI coding assistant during the project.

## Repository layout

```
├── RESEARCH_PLAN.md  EXPERIMENT_LOG.md  GLOSSARY.md  DATA_AUDIT.md  CLAUDE.md
├── TECHNICAL_README.md            original setup and command notes
├── SEALED_HOLDOUT_REGISTRY.json   HOLDOUT_FREEZE_MANIFEST.json
├── configs/  configs_holdout/     data paths, instruments, frozen holdout configuration
├── src/nqlevels/                  build (bars, profiles), levels, features, labels, eval (top-k), prod, audit, io, clock
├── scripts/                       one script per experiment (build_l1, sanity_l1, audit_leakage, run_p*)
├── outputs/                       experiment results (JSON/CSV), pre-registrations
└── tests/                         pytest suite (leakage, labeling, zones, sessions)
```

## Running it

```bash
git clone https://github.com/chahatpatel2003/nq-daily-levels.git
cd nq-daily-levels
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q
```

**Data is not included.** The raw Databento files are licensed. Derived layers (1-second bars, profiles, scored grids) were written to an external drive and remain private. `configs/paths.yaml` shows the original locations; point it at your own data to rebuild. Tests that need the market data are skipped.

## Third-party material

- **Market data:** Databento NQ trades and MBP-1 (licensed; not included).
- **Topstep:** funded-account economics are modelled from Topstep's public rules; Topstep is a trademark of its owner and this project is not affiliated with it.
- **Open-source libraries** are installed as dependencies under their own licences; none are copied into this repository.

## Notes

- Assembled on 2026-10-14 from the original `nq-levels` working folder. File contents are byte-identical to that snapshot (SHA-256 verified), except 3 notes files where a personal name was replaced with "the project owner". The project was not under version control before this.
- Two `paths.yaml` files and the 2026 pre-registration record are left exactly as they were, because their hashes are pinned in the project's freeze manifests or form the holdout seal. They still show a local data path or a first name.
- Research software only. Nothing here is investment advice.

© Chahat Patel. All rights reserved. This notice covers the original code, documentation and research results in this repository. It does not extend to the third-party material listed above, which belongs to its respective owners. No license to reuse this code is granted; it is published for review.
