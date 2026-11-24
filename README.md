# Verifiable Semantic RAG

Master's research project: **Design and Evaluation of a Verifiable and
Interpretable Retrieval-Augmented Generation Framework Using Structured
Semantic Knowledge Bases.**

## Development status

Stage 2 is complete across six vertical slices, plus our first verified live
OpenAI API experiment on a real research paper:
1. **Document Ingestion & Provenance** (`src.ingestion`, `src.processing`): TXT, Markdown, and PDF reading with deterministic document/chunk IDs, 1-based page numbers, and character offsets.
2. **Dense Embeddings & Vector Index** (`src.embeddings`): Configurable embedding client (`text-embedding-3-large`, 3,072 dimensions) and cosine-similarity `InMemoryVectorIndex`.
3. **Structured Semantic Extraction & Knowledge Graph** (`src.knowledge`): Pydantic extraction schemas (`ExtractedEntity`, `ExtractedRelationship`, `ExtractionResult`), verbatim quote verification, and a provenance-preserving NetworkX `MultiDiGraph`.
4. **End-to-End Document Pipeline & Real Paper Integration** (`src.pipeline.runner`): Automated document-to-graph pipeline, pre-flight token/cost estimator, and integration of the three development research papers in `data/papers/`.
5. **Hybrid Retrieval & Evidence-Backed Answer Generation** (`src.retrieval`, `src.generation`, `src.pipeline.qa`): Combines dense vector search with configurable multi-hop graph traversal, deduplicates and ranks evidence chunks, generates structured answers via `gpt-6-luna`, validates all citations and quotes against retrieved chunks, and fails closed to `insufficient_evidence=True` when claims lack verifiable source support.
6. **Persistent Research Artifacts** (`src.storage`): Portable, reliable local persistence using JSON metadata and NumPy `.npz` vector arrays. Enables embedding and extracting each document once, then loading pre-computed indexes and knowledge graphs for question answering with zero repeated extraction costs.

**Benchmark comparison (completed 2026-11-24).** This project and RAGU were evaluated on a documented 10% slice of
GraphRAG-Bench (Medical): 39 questions, the official evaluator with a documented judge proxy, `gpt-6-luna` as judge.
On the 36 questions of the three analyzable task types, pooled Answer Correctness was **0.586 (this project) and
0.588 (RAGU)**, a paired difference of -0.002 (95% bootstrap interval -0.059 to +0.058): statistically
indistinguishable, and both clearly nonzero. Full method, deviations, costs and limitations:
[docs/benchmark-investigation.md](docs/benchmark-investigation.md) (sections 14 to 18); aggregate numbers:
[results/benchmark_scores_summary.json](results/benchmark_scores_summary.json); spend: [docs/budget.md](docs/budget.md).

## Intended model configuration

- OpenAI generation, extraction, and summarization where applicable:
  `gpt-6-luna`
- OpenAI dense embeddings: `text-embedding-3-large` (3,072 dimensions)
- RAGU comparison: run with the same two models (RAGU v0.1.0, commit `abd3f29`); this is an updated
  configuration, not a reproduction of RAGU's legacy `gpt-4o-mini` examples.

See [the RAGU integration notes](docs/ragu-integration.md) for source-based
findings and the adapter design. See
[the methodology notes](docs/methodology.md) for the provenance model and
experimental record.

## Strict separation of mocked vs. live experiment artifacts

To maintain research integrity, every generated JSON artifact explicitly records
`"is_mocked": true | false`:

| Artifact | Mode (`is_mocked`) | Scope & Purpose | Recorded Results |
| :--- | :---: | :--- | :--- |
| `results/development_papers_summary.json` | `true` (Mocked) | Zero-cost structural pipeline check across 3 PDFs (55 pages, 262 chunks) using synthetic dummy providers. | 0 live API calls, 0 entities/edges (synthetic no-op extractor), 2,572.4 ms local runtime. **Not an experimental result.** |
| `results/live_smoke_test_ragu.json` | `false` (**Live OpenAI API — Run 1**) | Historical initial smoke test on `data/papers/komarov_2026_ragu.pdf` (10 pages, 54 chunks) prior to Slice 6 persistence. | **54/54** 3,072-d embeddings; **785** entity mentions (**528** unique graph entities); **639** relationship claims (**92** explicit verified graph edges); **$0.049618 USD** *pre-run estimate* (token counts and cost were estimated at the superseded $1.25/$5.00 LLM rates, not measured from API usage; about $0.0060 at current rates); **0** API errors; **1,087,967.369 ms** (~18.13 min) runtime. |
| `results/live_ragu_persisted_summary.json` & `data/processed/komarov_2026_ragu/` | `false` (**Live OpenAI API — Run 2, Persisted**) | Independent live reprocessing of `data/papers/komarov_2026_ragu.pdf` with full Slice 6 artifact persistence (`vectors.npz`, `extractions.json`, `chunks.json`, `documents.json`, `manifest.json`). | **54/54** 3,072-d embeddings; **741** entity mentions (**491** unique graph entities); **595** relationship claims (**80** explicit verified graph edges); **~$0.0496 USD** *estimate* (same superseded-rate estimate; measured API spend not recorded, check the OpenAI billing dashboard); **0** API errors; **1,247,197.133 ms** (~20.79 min) runtime. |
| `results/live_qa_ragu_q1.json` | `false` (**Live OpenAI API — QA Experiment Q1**) | First live hybrid retrieval and evidence-backed question-answering experiment over pre-loaded `data/processed/komarov_2026_ragu/` artifacts (0 re-extraction calls). | **10** hybrid chunks + **13** graph relationships retrieved; **3/3** citations verified verbatim on Pages 3 & 4 (`s3:c0` Table 1 caption, `s2:c3`, `s2:c4`); **37** query embedding tokens + **5,140** prompt / **2,038** completion tokens; **$0.001538 USD** cost computed from measured token usage at `config/pricing.json` rates; **33,475.21 ms** (~33.5 s) runtime. |
| `results/live_qa_ragu_q1_rerun_post_seed_fix.json` | `false` (**Live OpenAI API — Q1 rerun after seed fix**) | Same question, artifacts and retrieval settings as Q1, rerun after the word-boundary seed fix (working tree dirty, base commit `1a0ca56`). Original Q1 preserved. | Retrieval identical (**10** chunks, **188** entities, **13** relationships; same **5,140** prompt tokens); **1,622** completion tokens; **$0.00133 USD**; **23,135 ms**; **`insufficient_evidence=true`, 0/0 verified citations** (all model citations failed verification; cause not recorded). Shows run-to-run LLM variance with no seed/temperature set. |
| `results/live_qa_ragu_q1_diagnostic.json` | `false` (**Live OpenAI API — Q1 diagnostic run**) | Third sample of the same question with default parameters and the new `answer_diagnostics` (working tree dirty, base commit `1a0ca56`). | Retrieval counts and prompt tokens identical (**5,140**); **1,503** completion tokens (**790** reasoning); **$0.00127 USD**; **19.3 s**; **2/2 citations accepted**, both from `s3:c0` (table header and rows). The earlier fail-closed failure did **not** recur, so its cause remains undetermined. |
| `results/ragu_build_komarov_2026.json` | `false` (**Live OpenAI API — RAGU graph build**) | RAGU `v0.1.0` (`abd3f29`) built over the same Komarov text with `gpt-6-luna` + `text-embedding-3-large`, RAGU's documented example configuration, $1.00 cap (dirty tree, base `1a0ca56`). Graph in `data/ragu/komarov_2026_ragu/`. | **563** entities, **600** relations, **57** communities, **48** chunks; **258** chat + **6** embedding calls (all chat via `parse`); **1.80M** prompt / **0.53M** completion tokens (measured); **$0.451742 USD**; **17.2 min** (4 concurrent); 0 errors. Not directly comparable with this project's graph or its estimated build cost. |
| `results/ragu_q1_local.json` | `false` (**Live OpenAI API — RAGU local search, Q1**) | RAGU `LocalSearchEngine` (defaults) on the RAGU-built graph, same question as Q1 (cap $0.02, dirty tree, base `1a0ca56`). One sample. | **20** entities, **75** relations, **35** chunks retrieved; **13,695** prompt / **612** completion tokens (measured); **$0.00168 USD**; **8.3 s**; free-text answer with **no citations**. Same headline figures as this project's answers; larger context budget (about 2.7x tokens), so not a controlled comparison. |

Model pricing used for all cost figures lives in `config/pricing.json` (source and date recorded there; both the `gpt-6-luna` and `text-embedding-3-large` rates were verified against OpenAI's pricing pages on 2026-11-24).

## Reproducibility configuration

- `config/experiment.json`: provider, base URL, `gpt-6-luna` request settings (`temperature`, `reasoning_effort`, `max_completion_tokens`, `seed`; `null` = not sent), embedding model and dimensions, and the pinned RAGU version (tag `v0.1.0`, commit `abd3f2958c186dc624bbd42560ad6e1e7033efaf`).
- `config/pricing.json`: verified model prices with source and date.
- Environment variables (`LLM_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`, `OPENAI_BASE_URL`) must agree with the config or the run aborts; models are never substituted silently.
- Every new QA result and artifact manifest embeds a `run_configuration` snapshot (models, request params, pricing, git commit and dirty flag, Python/package versions, chunking/retrieval settings, RAGU pin check) plus measured token usage and cost. Commit before an experiment so `git_dirty` is `false`.

## Citation diagnostics and parameter controls

- QA results include `answer_diagnostics` (raw model output, per-citation accept/reject with reasons, fail-closed flag, request parameters sent) and `retrieved_chunks`. Citation acceptance is strict and fail-closed; diagnostics only explain rejections.
- `temperature`, `seed`, `reasoning_effort`, `max_completion_tokens` can only be enabled in `config/experiment.json` after `scripts/probe_llm_parameters.py --execute` (paid, tiny) has verified API support. All are currently off.
- `python scripts/analyze_quote_fragility.py` runs the zero-cost PDF quote-fragility analysis.

## Persistent Artifact Storage Layout (`src.storage.artifacts`)

Artifacts are persisted in a portable, human-inspectable format without Python pickle:

```text
data/processed/<artifact-name>/
├── manifest.json     # Schema version (1.0.0), model metadata, and SHA-256 integrity checksums
├── documents.json    # SourceDocument records with page sections and checksums
├── chunks.json       # TextChunk records with exact character/page provenance
├── extractions.json  # ChunkExtraction records with entities, relationships, and support levels
└── vectors.npz       # NumPy float32 vector array and chunk IDs (allow_pickle=False)
```

### Integrity & Reproducibility Guarantees
- **Atomic Writes**: Writes first to an isolated staging directory, verifies file hashes, and performs an atomic directory swap to prevent corrupted partial artifacts.
- **SHA-256 Validation**: On load, `load_artifacts` recomputes and verifies the hash of every artifact file against `manifest.json`.
- **Model & Dimension Checks**: Rejects artifacts if the vector dimension or embedding model does not match the active configuration.
- **Zero-Cost Loading**: Loading artifacts reconstructs `InMemoryVectorIndex` and `KnowledgeGraph` in memory without making any embedding or extraction API calls.

## Setup and local checks

Create a virtual environment, install dependencies, and place credentials only
in a local `.env` copied from `.env.example`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Running the Full Test Suite

Run all 118 unit and integration tests (mocked providers, zero API calls, zero cost; the 2 end-to-end evaluator tests
are skipped unless the evaluator environment and code described below are installed):

```powershell
python -m unittest discover -s tests -v
```

### Saving & Loading Artifacts in Document Processing (`scripts/process_papers.py`)

Process documents and save artifacts to disk:

```powershell
python scripts/process_papers.py --papers data/papers/komarov_2026_ragu.pdf --save-artifacts data/processed/komarov_2026_ragu
```

Load existing artifacts from disk (zero API calls):

```powershell
python scripts/process_papers.py --load-artifacts data/processed/komarov_2026_ragu
```

### Answering Questions with Pre-Loaded Artifacts (`scripts/answer_question.py`)

Query pre-loaded artifacts without re-embedding or re-extracting chunks:

```powershell
python scripts/answer_question.py --load-artifacts data/processed/komarov_2026_ragu --question "What is the primary contribution of RAGU?"
```

Estimate QA token usage and cost for pre-loaded artifacts (zero API calls):

```powershell
python scripts/answer_question.py --load-artifacts data/processed/komarov_2026_ragu --question "What is the primary contribution of RAGU?" --estimate-cost
```

Run QA with mocked providers on sample fixture (zero API calls):

```powershell
python scripts/answer_question.py --mock --papers tests/fixtures/sample_article.pdf --question "How does GraphRAG connect passages?"
```

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

