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

Formal GraphRAG-Bench benchmark evaluation has not yet started.

## Intended model configuration

- OpenAI generation, extraction, and summarization where applicable:
  `gpt-6-luna`
- OpenAI dense embeddings: `text-embedding-3-large` (3,072 dimensions)
- RAGU comparison: planned with the same models where its interfaces allow;
  this is an updated configuration, not a reproduction of RAGU's legacy
  `gpt-4o-mini` examples.

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
| `results/live_smoke_test_ragu.json` | `false` (**Live OpenAI API — Run 1**) | Historical initial smoke test on `data/papers/komarov_2026_ragu.pdf` (10 pages, 54 chunks) prior to Slice 6 persistence. | **54/54** 3,072-d embeddings; **785** entity mentions (**528** unique graph entities); **639** relationship claims (**92** explicit verified graph edges); **$0.049618 USD** spend; **0** API errors; **1,087,967.369 ms** (~18.13 min) runtime. |
| `results/live_ragu_persisted_summary.json` & `data/processed/komarov_2026_ragu/` | `false` (**Live OpenAI API — Run 2, Persisted**) | Independent live reprocessing of `data/papers/komarov_2026_ragu.pdf` with full Slice 6 artifact persistence (`vectors.npz`, `extractions.json`, `chunks.json`, `documents.json`, `manifest.json`). | **54/54** 3,072-d embeddings; **741** entity mentions (**491** unique graph entities); **595** relationship claims (**80** explicit verified graph edges); **~$0.0496 USD** estimated spend; **0** API errors; **1,247,197.133 ms** (~20.79 min) runtime. |

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

Run all 32 unit and integration tests (mocked providers, zero API calls, zero cost):

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
