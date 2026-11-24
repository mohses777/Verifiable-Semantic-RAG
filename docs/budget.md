# Budget ledger

Ceiling set by the researcher on 2026-11-24: **USD 2.50 for all remaining paid API work**. Every paid step
needs its own estimate and approval, and runs use a hard `--max-usd` cap (calls are refused once it is reached).

## Spend to date (measured from API-reported usage unless marked)

| Item | Cost |
| --- | --- |
| Q1 original / post-fix rerun / diagnostic run | $0.001538 / $0.00133 / $0.00127 |
| Parameter probe | $0.000009 |
| RAGU graph build (Komarov paper, 48 chunks) | $0.451742 |
| RAGU local-search query (Q1) | $0.00168 |
| Earlier smoke runs 1 and 2 (Komarov paper) | not measured; the recorded $0.0496 each were estimates at superseded rates (at current rates about $0.006 each) |
| **Measured subtotal** | **about $0.46** (before the unmeasured smoke runs) |

These earlier costs are outside the $2.50 ceiling, which covers the remaining work only.

## Planned allocation for the reduced benchmark run (about 10% corpus slice)

| Step | Estimate | Hard cap |
| --- | --- | --- |
| Question-to-corpus mapping embeddings | **done: $0.049974 measured** (estimate $0.06) | $0.10 |
| RAGU graph build on the slice (113 chunks) | **done: $0.754863 measured** (estimate $1.06) | $1.40 |
| This project's build on the slice (132 chunks) | run 1 measured $0.145711 but its artifacts were lost; a rerun costs about the same | $0.30 per run |
| Answers and judging, about 40 questions, 2 systems | about $0.2 to $0.5 (judge cost unmeasured; judge calls go through the capped proxy) | $0.50 |
| **Planned total** | **about $1.4 to $1.7** | **$2.30** |
| Reserve for reruns | | $0.20 |

Estimates extrapolate linearly from one 10-page document and are uncertain by roughly 50%. If a step's real
cost makes the plan exceed $2.50, the slice shrinks (for example 5%) before anything else is dropped.

## Ledger of paid steps under the $2.50 ceiling

| Date | Step | Measured cost | Running total |
| --- | --- | --- | --- |
| 2026-11-24 | Mapping embeddings (1,321 chunks + 2,062 evidence strings) | $0.049974 | $0.049974 |
| 2026-11-24 | RAGU graph build on the slice (113 chunks; cap $1.40; estimate $1.06) | $0.754863 | $0.804837 |
| 2026-11-24 | This project's slice build, run 1 (132 chunks; cap $0.30). Extraction completed, **artifacts lost** at the final save (Windows folder lock; bug since fixed) | $0.145711 | $0.950548 |
| 2026-11-24 | This project's slice build, run 2 (132 chunks; cap $0.30); artifacts saved and verified | $0.143365 | $1.093913 |
| 2026-11-24 | Answering 39 questions, this project (cap $0.15) | $0.052214 | $1.146127 |
| 2026-11-24 | Answering 39 questions, RAGU (cap $0.20) | $0.033302 | $1.179429 |
| 2026-11-24 | Scoring pilot, 2 items (cap $0.02) | $0.002012 | $1.181441 |
| 2026-11-24 | Scoring this project's 39 answers (cap $0.25) | $0.065534 | $1.246975 |
| 2026-11-24 | Scoring RAGU's 39 answers (cap $0.25) | $0.084986 | $1.331961 |
| 2026-11-24 | Re-scoring 1 timed-out question, this project (cap $0.03; estimate was under $0.01 for all 3) | $0.008525 | $1.340486 |
| 2026-11-24 | Re-scoring 2 timed-out questions, RAGU (cap $0.03) | $0.018316 | $1.358802 |

Remaining under the ceiling: about $1.14.
