# nq-levels

Research system for generating **2–4 strong NQ price zones each morning**,
produced at 06:25 PT for the 06:30–09:00 PT window.

Research phase. No trading edge has been demonstrated yet.

## Documents

| file | what it is |
|---|---|
| `CLAUDE.md` | working context, conventions, the rules that will bite you |
| `DATA_AUDIT.md` | what is on disk, measured 2026-08-14 |
| `GLOSSARY.md` | exact definitions — zone, touch, reaction, chop, reclaim |
| `RESEARCH_PLAN.md` | the research design and experiment queue |
| `EXPERIMENT_LOG.md` | everything tried, kept or rejected |

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q
```

Raw data lives on the external SSD `T9` and is read-only. Derived layers are
written to `/Volumes/T9/nq_levels/` (the internal disk has ~32 GB free).

## Commands

```bash
.venv/bin/python scripts/build_l1.py --resume            # 1s bars + profiles, ~10 min
.venv/bin/python scripts/sanity_l1.py --months 2024-03   # 22-check data gate
.venv/bin/python scripts/audit_leakage.py                # 7-check leakage gate
.venv/bin/python scripts/demo_pipeline.py --start 2024-03-05 --end 2024-03-08
```

## The one rule

Nothing at or after **06:25 PT** may influence a zone for that morning.
`scripts/audit_leakage.py` proves it by rebuilding zones from a tape physically
truncated at the cut and requiring them identical.
