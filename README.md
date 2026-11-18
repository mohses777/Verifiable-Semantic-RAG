# Verifiable Semantic RAG

Master's research project: **Design and Evaluation of a Verifiable and
Interpretable Retrieval-Augmented Generation Framework Using Structured
Semantic Knowledge Bases.**

## Development status

Stage 2 is in progress. Four vertical slices are complete:
1. Document ingestion and chunking with durable character/page provenance.
2. Dense embeddings and cosine similarity retrieval with an in-memory vector index.
3. Structured semantic extraction with Pydantic schemas and a NetworkX knowledge graph.
4. Real document integration with the three development research papers and an end-to-end processing pipeline.

Hybrid retrieval and answer generation have not been implemented. No paid API
experiments have started.

## Intended model configuration

- OpenAI generation, extraction, and summarization where applicable:
  `gpt-6-luna`
- OpenAI dense embeddings: `text-embedding-3-large` (3,072 dimensions)
- RAGU comparison: planned with the same models where its interfaces allow;
  this is an updated configuration, not a reproduction of RAGU's legacy
  `gpt-4o-mini` examples.

See [the RAGU integration notes](docs/ragu-integration.md) for source-based
findings and the compatibility test plan. See
[the methodology notes](docs/methodology.md) for the implemented provenance
model. No API calls have been made.

## Implemented architecture & slices

### 1. Ingestion & Provenance (`src.ingestion`, `src.processing`)
`src.ingestion.ingest_file()` reads UTF-8 TXT, Markdown, and PDF sources.
Each document receives a stable content-and-path-derived ID, SHA-256 checksum,
source type, metadata, and page-level sections.
`src.processing.chunk_document()` splits sections into configurable overlapping
chunks while preserving document ID, deterministic chunk ID, page number, section
index, and character offsets.

### 2. Embeddings & Semantic Retrieval (`src.embeddings`)
`src.embeddings` provides a provider-independent `EmbeddingClient`, a local
`InMemoryVectorIndex` (cosine similarity), and `SemanticRetriever`. Chunks retain
their full provenance records throughout embedding and retrieval ranking.

### 3. Structured Extraction & Knowledge Graph (`src.knowledge`)
`src.knowledge` defines Pydantic schemas (`ExtractedEntity`, `ExtractedRelationship`,
`ExtractionResult`). `SemanticExtractor` grounds extractions with exact source quotes
and demotes unquoted claims to `uncertain`. `KnowledgeGraph` uses NetworkX, merges
repeated entities across chunks, and links explicit edges directly to their evidence
`TextChunk` records (document ID, page number, and offsets).

### 4. End-to-End Pipeline & Real Document Integration (`src.pipeline`)
`src.pipeline.DocumentProcessingPipeline` orchestrates the complete flow from
raw documents to indexed vectors and knowledge graphs. It generates machine-readable
`PipelineSummary` records and includes an exact token/cost estimator
(`estimate_processing_cost`).

Three development research papers are integrated under `data/papers/`:
- Lewis et al. (2020) — *Retrieval-Augmented Generation* (19 pages, 91 chunks)
- Edge et al. (2024) — *Microsoft GraphRAG* (26 pages, 117 chunks)
- Komarov et al. (2026) — *RAGU* (10 pages, 54 chunks)

Total development corpus: 55 pages, 262 chunks (~60,300 tokens).

## Setup and local checks

Create a virtual environment, install dependencies, and place credentials only
in a local `.env` copied from `.env.example`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Running Tests

Run all 21 unit and integration tests (zero API calls, zero cost):

```powershell
python -m unittest discover -s tests
```

### Estimating Pipeline API Cost

Estimate exact tokens and API costs for the 3 research papers without making calls:

```powershell
python scripts/process_papers.py --estimate-cost --papers data/papers/lewis_2020_rag.pdf data/papers/edge_2024_graphrag.pdf data/papers/komarov_2026_ragu.pdf
```

### Running Pipeline in Mock Mode

Run the complete pipeline end-to-end on the real papers using mocked providers:

```powershell
python scripts/process_papers.py --mock --papers data/papers/lewis_2020_rag.pdf data/papers/edge_2024_graphrag.pdf data/papers/komarov_2026_ragu.pdf --output-summary results/development_papers_summary.json
```
