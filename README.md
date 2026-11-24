# Verifiable Semantic RAG

Master's research project: **Design and Evaluation of a Verifiable and Interpretable Retrieval-Augmented Generation
Framework Using Structured Semantic Knowledge Bases.**

The system turns documents into a provenance-preserving knowledge graph and answers questions with hybrid
(vector + graph) retrieval. Every factual claim in an answer must carry a citation whose quote is checked verbatim
against the retrieved source text; citations that fail are rejected, and an answer with no verifiable citation fails
closed to "insufficient evidence". Each citation traces back to its chunk, page and source document.

## Results: GraphRAG-Bench comparison with RAGU

This system and [RAGU](https://github.com/RaguTeam/RAGU) (v0.1.0) were evaluated on a documented 10% slice of
GraphRAG-Bench (Medical): 39 questions, scored with the official evaluator behind a documented judge proxy, with
`gpt-6-luna` as both answering model and judge for both systems. On the 36 questions of the three analyzable task types,
pooled Answer Correctness was **0.586 (this system) and 0.588 (RAGU)**: a paired difference of -0.002 (95% bootstrap
interval -0.059 to +0.058), statistically indistinguishable, and both clearly nonzero.

| Task type (questions) | Metric | This project | RAGU | Difference, this project minus RAGU (95% interval) |
| --- | --- | --- | --- | --- |
| Fact Retrieval (12) | Answer Correctness | 0.675 | 0.622 | +0.053 (-0.061 to +0.175) |
| Fact Retrieval (12) | ROUGE-L | 0.474 | 0.407 | +0.068 (-0.028 to +0.155) |
| Complex Reasoning (12) | Answer Correctness | 0.521 | 0.509 | +0.013 (-0.056 to +0.076) |
| Complex Reasoning (12) | ROUGE-L | 0.260 | 0.223 | +0.037 (+0.007 to +0.079) |
| Contextual Summarize (12) | Answer Correctness | 0.562 | 0.632 | -0.071 (-0.171 to +0.030) |
| Contextual Summarize (12) | Coverage | 0.597 | 0.761 | -0.164 (-0.353 to -0.002) |
| **Pooled, 3 types (36)** | **Answer Correctness** | **0.586** | **0.588** | **-0.002 (-0.059 to +0.058)** |
| Creative Generation (3, descriptive only) | Answer Correctness | 0.440 | 0.526 | too few questions to compare |

| Other measured properties (same slice, one run each) | This project | RAGU |
| --- | --- | --- |
| Answers with quote-verified citations | yes (131 of 139 proposed citations accepted) | no citations |
| Flags answers whose evidence is incomplete | yes; flagged answers scored 0.41 vs 0.69 unflagged | no |
| Graph build cost / time (measured) | $0.143 / 6.6 min | $0.755 / 26.7 min |
| Mean answer time per question | 9.6 s | 4.6 s |

Only the pooled row is the headline result; per-type rows are small samples with several comparisons, so a single
interval excluding zero may be chance. Scores are not comparable with the public leaderboard (documented evaluator
deviations), and RAGU ran with its untuned defaults.

Full method, every deviation from the official evaluator, costs and limitations:
[docs/benchmark-investigation.md](docs/benchmark-investigation.md) (sections 14 to 18). Aggregate numbers:
[results/benchmark_scores_summary.json](results/benchmark_scores_summary.json). Spend: [docs/budget.md](docs/budget.md)
(about USD 1.36 for all paid runs).

## How it works

1. **Ingestion with provenance** (`src/ingestion`, `src/processing`): TXT, Markdown and PDF; deterministic document and
   chunk IDs, page numbers and character offsets.
2. **Embeddings and vector index** (`src/embeddings`): `text-embedding-3-large` (3,072 dimensions), cosine similarity.
3. **Knowledge graph** (`src/knowledge`): structured entity and relationship extraction with `gpt-6-luna`; every entity
   and relationship must quote its source chunk, and only explicitly supported relationships become graph edges.
4. **Hybrid retrieval** (`src/retrieval`): dense search plus multi-hop graph traversal; evidence chunks are deduplicated
   and ranked.
5. **Evidence-backed answers** (`src/generation`): structured answers whose citations are checked against the retrieved
   chunks (strict, fail-closed), with per-citation diagnostics explaining every rejection.
6. **Persistent artifacts** (`src/storage`): graphs, vectors and chunks saved once with SHA-256 integrity checks and
   reloaded without further API calls.

All model settings, prices and the pinned RAGU version are version-controlled in `config/`; every result records the
models, request settings, git commit and package versions that produced it.

## Repository layout

```text
src/        the system (packages above) plus src/benchmarks (benchmark selection, judge proxy, evaluator adapter)
scripts/    command-line entry points: build, answer, benchmark preparation, evaluation, score summary
tests/      119 unit and integration tests (mocked; no API calls)
config/     models, request settings, prices, judge policy, pinned dependency records
results/    machine-readable records of every live run (no third-party benchmark text)
docs/       investigation, methodology, RAGU notes, budget ledger, design notes
data/       development papers and their processed artifacts (benchmark data is not included; see below)
```

## Setup and tests

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env      # then put your OpenAI key in .env (never commit it)
python -m unittest discover -s tests -v
```

The test suite makes no API calls. Its 2 end-to-end evaluator tests are skipped unless the evaluator environment
described below is installed.

## Quick start on your own documents

```powershell
python scripts/process_papers.py --papers data/papers/komarov_2026_ragu.pdf --estimate-cost         # free estimate
python scripts/process_papers.py --papers <your.pdf> --save-artifacts data/processed/<name> --concurrency 4 --max-usd 0.30
python scripts/answer_question.py --load-artifacts data/processed/<name> --question "..." --estimate-cost
python scripts/answer_question.py --load-artifacts data/processed/<name> --question "..."
```

Every paid command has a free estimate, and builds accept a hard `--max-usd` spend cap. More commands and the
artifact format are in the developer reference at the end of [docs/methodology.md](docs/methodology.md).

## Reproducing the GraphRAG-Bench comparison

Every paid step has a dry run and a hard `--max-usd` cap. Measured costs of the original runs are in
[docs/budget.md](docs/budget.md) (about $1.36 in total). Three Python environments are used, on purpose:

| Environment | Used for | Dependency record |
| --- | --- | --- |
| main (`requirements.txt`) | this project, data preparation, score summary | `requirements.txt` |
| RAGU venv | RAGU v0.1.0 at commit `abd3f2958c186dc624bbd42560ad6e1e7033efaf` (cloned to `third_party/RAGU`) | `config/ragu_env_requirements.txt` |
| evaluator venv | official GraphRAG-Bench evaluator, commit `fdbab5959b18c96532580877ffe27d112bccc0ec` (`Evaluation/` copied to `third_party/GraphRAG-Benchmark/`; do **not** install the unrelated PyPI package `Evaluation`) | `config/bench_eval_env_requirements.txt` |

The benchmark data is **not** in this repository (its licence is unresolved; see section 2 of the investigation).
Download `Datasets/Corpus/medical.json` and `Datasets/Questions/medical_questions.json` from
`GraphRAG-Bench/GraphRAG-Benchmark` at the commit above into `data/benchmarks/graphrag_bench/`; their SHA-256 hashes
are listed in section 10 of the investigation. Then:

```powershell
python scripts/benchmark_prepare.py embed --execute --max-usd 0.10      # map questions to the corpus
python scripts/benchmark_prepare.py select --per-type 12                 # slice + 39 questions (seed 42)
python scripts/process_papers.py --papers data/benchmarks/graphrag_bench/slice/slice_corpus.txt --save-artifacts data/processed/graphrag_bench_slice --output-summary results/pipeline_graphrag_bench_slice_run2.json --concurrency 4 --max-usd 0.30
<ragu-venv>\Scripts\python.exe scripts/run_ragu_build.py --paper data/benchmarks/graphrag_bench/slice/slice_corpus.txt --workdir data/ragu/graphrag_bench_slice --output results/ragu_build_graphrag_bench_slice.json --execute --max-usd 1.40
python scripts/run_benchmark_answers_ours.py --execute --max-usd 0.15
<ragu-venv>\Scripts\python.exe scripts/run_benchmark_answers_ragu.py --execute --max-usd 0.20
<eval-venv>\Scripts\python.exe scripts/run_benchmark_eval.py --items data/benchmarks/graphrag_bench/runs/ours/items.json --label ours --max-usd 0.25
<eval-venv>\Scripts\python.exe scripts/run_benchmark_eval.py --items data/benchmarks/graphrag_bench/runs/ragu/items.json --label ragu --max-usd 0.25
python scripts/summarize_benchmark_scores.py
```

Re-scoring samples the evaluator drops on its 30-second timeout uses `--judge-timeout 180` on a subset file (section 18).
LLM outputs are not deterministic (no seed is set), so a rerun gives similar but not identical numbers. Without
re-running anything, `python scripts/summarize_benchmark_scores.py` recomputes the published summary from the committed,
text-free `results/benchmark_per_sample_scores.json`.

## Documentation

| Document | Contents |
| --- | --- |
| [docs/benchmark-investigation.md](docs/benchmark-investigation.md) | Benchmark choice, licence, data, evaluator, judge policy and deviations, slice selection, all runs and results |
| [docs/methodology.md](docs/methodology.md) | Implementation record, earlier experiments on the RAGU paper, citation diagnostics, developer reference |
| [docs/ragu-integration.md](docs/ragu-integration.md) | RAGU source findings, installation, live runs and compatibility |
| [docs/budget.md](docs/budget.md) | Ledger of every paid step against the USD 2.50 ceiling |
| [docs/claim-level-evidence-design.md](docs/claim-level-evidence-design.md) | Design note: measuring claim-level evidence support |

## Data and licensing

The code is this project's own. RAGU (MIT) and the GraphRAG-Bench evaluator (MIT, XMU-DeepLIT) are third-party and are
fetched into the git-ignored `third_party/` folder at pinned commits; they are not redistributed here. The
GraphRAG-Bench data (derived from NCCN guidelines) has no separate licence statement, so it is not included: download
it as described above. Graph builds and evaluator outputs that contain benchmark text are git-ignored for the same reason.
