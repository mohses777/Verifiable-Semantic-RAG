# Verifiable Semantic RAG

Master's research project: **Design and Evaluation of a Verifiable and
Interpretable Retrieval-Augmented Generation Framework Using Structured
Semantic Knowledge Bases.**

## Development status

Stage 2 is complete across five vertical slices, plus our first verified live
OpenAI API experiment on a real research paper:
1. **Document Ingestion & Provenance** (`src.ingestion`, `src.processing`): TXT, Markdown, and PDF reading with deterministic document/chunk IDs, 1-based page numbers, and character offsets.
2. **Dense Embeddings & Vector Index** (`src.embeddings`): Configurable embedding client (`text-embedding-3-large`, 3,072 dimensions) and cosine-similarity `InMemoryVectorIndex`.
3. **Structured Semantic Extraction & Knowledge Graph** (`src.knowledge`): Pydantic extraction schemas (`ExtractedEntity`, `ExtractedRelationship`, `ExtractionResult`), verbatim quote verification, and a provenance-preserving NetworkX `MultiDiGraph`.
4. **End-to-End Document Pipeline & Real Paper Integration** (`src.pipeline.runner`): Automated document-to-graph pipeline, pre-flight token/cost estimator, and integration of the three development research papers in `data/papers/`.
5. **Hybrid Retrieval & Evidence-Backed Answer Generation** (`src.retrieval`, `src.generation`, `src.pipeline.qa`): Combines dense vector search with configurable multi-hop graph traversal, deduplicates and ranks evidence chunks, generates structured answers via `gpt-6-luna`, validates all citations and quotes against retrieved chunks, and fails closed to `insufficient_evidence=True` when claims lack verifiable source support.

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
| `results/live_smoke_test_ragu.json` | `false` (**Live OpenAI API**) | Controlled live experiment on `data/papers/komarov_2026_ragu.pdf` (10 pages, 54 chunks) using `text-embedding-3-large` and `gpt-6-luna`. | **54/54** 3,072-d embeddings; **785** entity mentions (**528** unique graph entities); **639** relationship claims (**92** explicit verified graph edges); **$0.049618 USD** spend; **0** API errors; **1,087,967.369 ms** (~18.1 min) runtime. |

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

Run all 27 unit and integration tests (mocked providers, zero API calls, zero cost):

```powershell
python -m unittest discover -s tests -v
```

### Question-Answering CLI (`scripts/answer_question.py`)

 Estimate token usage and cost for indexing and answering a question (zero API calls):

```powershell
python scripts/answer_question.py --estimate-cost --papers data/papers/komarov_2026_ragu.pdf --question "How does RAGU construct its knowledge graph?"
```

Run the end-to-end question-answering pipeline in mocked mode (zero API calls):

```powershell
python scripts/answer_question.py --mock --papers tests/fixtures/sample_article.pdf --question "How does GraphRAG connect passages?"
```

### Document Processing Pipeline CLI (`scripts/process_papers.py`)

Estimate token usage and cost for document processing:

```powershell
python scripts/process_papers.py --estimate-cost --papers data/papers/lewis_2020_rag.pdf data/papers/edge_2024_graphrag.pdf data/papers/komarov_2026_ragu.pdf
```
