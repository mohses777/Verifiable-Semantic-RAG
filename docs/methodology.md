# Methodology and implementation record

## Stage 2, vertical slice 1: source provenance

Implemented on 2026-11-17 without API calls or dependency installation. This
slice establishes the source-evidence contract for all later RAG components.

### Input boundary

`src/ingestion/loader.py` accepts TXT, Markdown, and PDF files. TXT and
Markdown are decoded as UTF-8. PDFs are processed page by page through
`pypdf.PdfReader` when the optional reader is available. The loader normalizes
line endings, removes NUL bytes, and trims outer whitespace. It does not infer,
summarize, or otherwise alter the document's meaning.

Each `SourceDocument` stores a path-and-content-derived document ID, raw-file
SHA-256 checksum, source type, filename metadata, and ordered text sections.
Each PDF text section stores its original 1-based page number. Character
offsets are defined over the normalized extracted text, not raw PDF bytes.

### Chunking boundary

`src/processing/chunking.py` splits each section independently. It accepts a
character `chunk_size` and `overlap`, prefers a space or line boundary, and
falls back to the maximum length for a long token. Chunks never span PDF pages.

Every `TextChunk` includes:

- `chunk_id`: deterministic `<document-id>:s<section>:c<sequence>` identity;
- `document_id`, source path, source type, and input metadata;
- page number and section index;
- normalized-text start/end offsets within the section; and
- start/end offsets in the document's normalized concatenation.

Future vector, graph, and answer records must store `chunk_id`; this provides a
traceable path from an answer citation back to the exact retrieved text.

### Verification

`tests/test_ingestion_and_chunking.py` uses workspace fixtures and a mocked PDF
reader. It verifies stable document identity, TXT metadata, overlapping chunks,
offsets, page provenance, invalid configuration handling, and unsupported file
rejection. These tests validate local transformation behavior only; they do not
validate an external PDF parser or OpenAI services.

The next implementation slice after retrieval is structured entity and
relationship extraction, using the same `TextChunk` evidence identifiers.
Model calls require a separate cost estimate and approval.

## Stage 2, vertical slice 2: embeddings and semantic retrieval

Implemented on 2026-11-18 without API calls. `src/embeddings/client.py` defines
an `EmbeddingProvider` boundary and `EmbeddingClient`. The OpenAI provider uses
the environment-configured `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`,
`OPENAI_BASE_URL`, and `OPENAI_API_KEY`; it supports ordered batch requests and
validates response count, dimensions, NaN, and infinity. The intended final
configuration remains `text-embedding-3-large` at 3,072 dimensions.

`src/embeddings/index.py` defines the replaceable `VectorIndex` contract and
the initial `InMemoryVectorIndex` backend. The local backend uses cosine
similarity and deterministic tie ordering. `src/embeddings/pipeline.py` joins
this to the source model: `SemanticRetriever.index_chunks()` stores each vector
as a `ChunkEmbedding` that contains the original `TextChunk`; `retrieve()`
returns `RetrievalResult` objects that retain the complete source chunk and all
its provenance fields.

`tests/test_embeddings_and_retrieval.py` uses a fake asynchronous provider.
It verifies batch ordering, provider model selection, invalid input/vector
handling, similarity ranking, empty-index behavior, and retrieval provenance.
The tests verify request-to-index behavior only. They do not establish live
OpenAI embedding availability, output dimension, token accounting, latency, or
cost. Those require a separately approved and recorded API smoke test.

## Stage 2, vertical slice 3: semantic extraction and knowledge graph

Implemented on 2026-11-18. `src/knowledge/schemas.py` supplies Pydantic output
schemas for `ExtractedEntity`, `ExtractedRelationship`, and `ExtractionResult`.
Relationships carry a `support_level` of `explicit`, `uncertain`, or `inferred`.
Each entity and relationship is assigned the source chunk ID by the application;
the model does not control this provenance value.

`src/knowledge/extraction.py` defines a mockable structured-output provider and
an OpenAI implementation using `beta.chat.completions.parse` with the configured
`LLM_MODEL`. `SemanticExtractor` gives the model a source-only extraction prompt,
requires quoted evidence, records execution time, and validates quotations
against the source chunk. It discards unsupported entities and demotes a
relationship to `uncertain` when its quotation is missing or either endpoint is
not a supported entity in that chunk.

`src/knowledge/graph.py` provides a NetworkX `MultiDiGraph` facade. Entity nodes
merge by normalized name but retain every extracted mention and source chunk ID.
Only `explicit` relationships with known endpoints become graph edges. Both
`entity_evidence_for()` and `relationships_for()` return the original
`TextChunk`, allowing callers to trace entities and facts to their document,
page, section, and offsets.

`tests/test_knowledge_extraction_and_graph.py` uses mocked LLM results. It
verifies Pydantic-result handling, chunk provenance attachment, duplicate entity
merging, entity and relationship evidence lookup, and exclusion of unsupported
relationships. The tests do not verify live GPT-6 Luna structured-output
behavior, quality of its extraction, provider cost, or model-specific limits.

## Stage 2, vertical slice 4: real document integration and pipeline

Implemented on 2026-11-18. `src/pipeline/runner.py` provides the
`DocumentProcessingPipeline` coordinating document ingestion, chunking, vector
embedding, semantic extraction, and knowledge graph construction.

### Development Corpus
Three seminal research papers were added to `data/papers/` as the development
testbed (distinct from formal benchmark datasets):
1. `lewis_2020_rag.pdf` (19 pages, 91 chunks)
2. `edge_2024_graphrag.pdf` (26 pages, 117 chunks)
3. `komarov_2026_ragu.pdf` (10 pages, 54 chunks)
Total corpus size: 55 pages, 262 chunks (~241,200 characters, ~60,300 tokens).

### Token and Cost Estimation
`estimate_processing_cost()` models expected API charges:
- `text-embedding-3-large` (3,072d): ~$0.0078 USD.
- `gpt-6-luna` extraction: ~$0.2326 USD.
- Total estimated cost for all three papers: ~$0.2404 USD.

### Machine-Readable Output
`PipelineSummary` captures document metadata, SHA-256 checksums, page/chunk
counts, entity/relation counts, and explicitly records whether mocked providers
were used (`is_mocked: bool`), guarding against any confusion between mocked tests
and live experimental runs.

### Verification
`tests/test_pipeline_integration.py` uses a valid 2-page research PDF fixture
(`tests/fixtures/sample_article.pdf`) to test the complete pipeline end-to-end
with mock providers. It verifies that entity and relationship graph nodes link
directly to their originating page numbers (e.g. Page 2 for GraphRAG) and character
offsets, and verifies full JSON serialization.
