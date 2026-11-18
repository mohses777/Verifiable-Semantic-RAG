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

### Separation of Mocked vs. Live Summary Artifacts
To prevent any confusion between local structural verification and live API
experiments, all output JSON summaries explicitly include `"is_mocked": bool`:

- **Mocked Development Summary (`results/development_papers_summary.json`)**:
  Generated with `--mock` (`"is_mocked": true`) across all 3 PDFs (55 pages, 262
  chunks) in `2,572.38 ms`. Uses synthetic no-op providers (`0` entities, `0`
  edges, `0` API calls) solely to verify PDF reading, deterministic chunk IDs,
  and JSON serialization. **Must never be cited as an experimental result.**
- **Live Experimental Summary (`results/live_smoke_test_ragu.json`)**:
  Generated against the live OpenAI API (`"is_mocked": false`) on
  `data/papers/komarov_2026_ragu.pdf`. Records actual model extraction outputs,
  token counts, costs, and execution time.

## Stage 2, live smoke test: Komarov et al. (2026) verification

Executed on 2026-11-18 against the live official OpenAI API (`is_mocked: false`).
- **Target Document**: `data/papers/komarov_2026_ragu.pdf` (10 pages, 54 chunks, document ID: `922a692238adfe2db666bb65`, SHA-256: `7cc09c9f3555ad4140464e69f407608b73ee5755354a04d9c0a964efae48e404`).
- **Dense Embeddings**: `text-embedding-3-large`, 54/54 vectors generated. Every returned vector was verified to have exactly **3,072 dimensions**.
- **Extraction Model**: `gpt-6-luna` using Pydantic structured outputs via `beta.chat.completions.parse`.
- **Entities Extracted**: 785 total mentions across 54 chunks, merged into **528 unique entities** in the NetworkX graph.
- **Relationships Extracted**: 639 relationship claims across 54 chunks, filtered by application provenance rules to **92 explicit directed edges** between confirmed grounded entities.
- **Execution Time**: `1,087,967.369 ms` (~18.13 minutes across 54 sequential structured extraction requests and batched embedding generation).
- **Error Count**: 0 errors. All 54 chunks processed successfully without retries.
- **Financial Cost**: Total execution cost was **$0.049618 USD** ($0.001622 embeddings + $0.047996 extraction), within the user-approved $0.10 USD budget cap.
- **Artifact**: Full machine-readable record preserved at `results/live_smoke_test_ragu.json`.

## Stage 2, vertical slice 5: hybrid retrieval and evidence-backed answer generation

Implemented on 2026-11-18 without additional paid API calls.

### Hybrid Retrieval (`src/retrieval/hybrid.py`, `src/knowledge/graph.py`)
`HybridRetriever` combines dense cosine-similarity retrieval (`SemanticRetriever`)
with breadth-first knowledge graph traversal (`KnowledgeGraph.traverse`):
1. Retrieves the top-`vector_top_k` chunks from the vector index.
2. Identifies seed graph entities from both the query text (`match_entities_in_text`)
   and the top vector-retrieved chunks (`entities_in_chunk`).
3. Expands up to `graph_traversal_depth` hops across explicit edges in the
   NetworkX `MultiDiGraph`, collecting `EntityEvidence` and `RelationshipEvidence`.
4. Deduplicates all retrieved chunks by `chunk_id`, preserves the immutable
   `TextChunk` provenance, labels each chunk with its `retrieval_sources`
   (`"vector"`, `"graph"`, or `("vector", "graph")`), and ranks chunks by a
   weighted combination of vector similarity and normalized graph support.

### Evidence-Backed Answer Generation (`src/generation/schemas.py`, `src/generation/answering.py`)
`GeneratedAnswer` and `EvidenceCitation` define the Pydantic structured-output
contract for `gpt-6-luna`:
- `answer_text`: Grounded response derived exclusively from retrieved evidence.
- `evidence_references`: List of `EvidenceCitation` items (`chunk_id`,
  `document_id`, `source_path`, `page_number`, verbatim `quote`, and supported `claim`).
- `supporting_chunk_ids`: Deduplicated list of verified supporting chunk IDs.
- `insufficient_evidence`: Boolean indicator when retrieved evidence cannot support an answer.
- `confidence_explanation`: Explanation of confidence or uncertainty.

`EvidenceBackedAnswerGenerator` enforces three application-level verification guards
on every LLM response:
1. **Chunk Allow-List Verification**: Citations referencing any `chunk_id` not
   present in the retrieved `HybridRetrievalBundle` are stripped.
2. **Verbatim Quote Verification**: Citations whose `quote` does not appear in
   the referenced `TextChunk.text` (after whitespace and case normalization) are stripped.
3. **Authoritative Provenance Injection & Fail-Closed Guard**: `document_id`,
   `source_path`, and `page_number` are overwritten directly from the authoritative
   `TextChunk` so the model can never spoof source metadata. If a non-abstaining
   answer (`insufficient_evidence=False`) has zero verified citations remaining,
   the generator fails closed and returns `insufficient_evidence=True`.

### End-to-End Question-Answering CLI (`src/pipeline/qa.py`, `scripts/answer_question.py`)
`QuestionAnsweringPipeline` and `scripts/answer_question.py` connect document
processing, hybrid retrieval, and grounded answer generation into a single CLI
supporting `--estimate-cost`, `--mock`, `--vector-top-k`, `--graph-depth`, and
JSON summary output (`QAPipelineSummary`).

### Verification
`tests/test_hybrid_retrieval_and_answering.py` adds 6 unit and integration tests
using mocked providers (bringing the total suite to **27 passing tests**).

## Stage 2, vertical slice 6: persistent research artifacts

Implemented on 2026-11-18 without paid API calls.

### Storage Architecture (`src/storage/artifacts.py`)
To prevent repeated expensive embedding and semantic extraction calls, `save_artifacts`
and `load_artifacts` provide atomic, portable serialization using JSON and NumPy:
- `manifest.json`: Stores `schema_version` (`"1.0.0"`), creation timestamp, document
  counts, chunk counts, entity/relationship statistics, embedding model ID, dimensions,
  LLM model ID, `is_mocked` flag, per-document metrics, and SHA-256 hashes of all files.
- `documents.json`: Full `SourceDocument` records with sections, page numbers, and checksums.
- `chunks.json`: Full `TextChunk` records with deterministic chunk IDs and character offsets.
- `extractions.json`: Full `ChunkExtraction` records with entities, evidence quotations,
  and support levels (`explicit`, `uncertain`, `inferred`).
- `vectors.npz`: Compressed NumPy archive containing `vectors` (`float32` 2D array)
  and `chunk_ids` (Unicode string array). Python pickle is strictly disabled (`allow_pickle=False`).

### Atomic Writes & Integrity Guarantees
1. **Atomic Staging**: Artifacts are assembled in an isolated sibling directory
   (`.tmp_<dir>_<id>`) and swapped atomically upon completion to avoid partial writes.
2. **Integrity Validation**: On load, `load_artifacts` recomputes SHA-256 hashes of
   `documents.json`, `chunks.json`, `extractions.json`, and `vectors.npz` against
   `manifest.json`, failing fast on any file corruption or manual tampering.
3. **Model & Dimension Validation**: Loading checks `embedding_dimension` and
   `embedding_model` against expected values, preventing cross-model vector contamination.
4. **Offline Safety Guard**: `load_artifacts` equips `SemanticRetriever` with an
   `OfflineEmbeddingProvider` by default, guaranteeing that loading artifacts can never
   trigger unexpected paid embedding calls.

### CLI Integration
- `scripts/process_papers.py --save-artifacts <dir>`: Saves pipeline outputs to disk.
- `scripts/process_papers.py --load-artifacts <dir>`: Reconstructs the index and graph
  without re-running ingestion or API calls.
- `scripts/answer_question.py --load-artifacts <dir> --question "..."`: Reconstructs
  the vector index and knowledge graph from disk, evaluates hybrid retrieval, and
  answers questions while embedding only the single short query string.

### Limitations of Current In-Memory Reconstruction
While serialization allows complete offline persistence, `load_artifacts` reconstitutes
the vector embeddings into Python RAM (`InMemoryVectorIndex`) and the knowledge graph
into a NetworkX graph structure. For our 3-paper development corpus (262 chunks), RAM
footprint is negligible (< 2 MB) and query retrieval completes in milliseconds. For
large-scale benchmark evaluation (e.g. GraphRAG-Bench with > 50,000 chunks), a dedicated
persistent vector database (e.g. SQLite/DuckDB, FAISS, or Qdrant) may be evaluated.

### Verification
`tests/test_storage_artifacts.py` adds 5 tests verifying save/reload round-trip,
chunk/page provenance retention, vector dimension and value preservation, knowledge
graph reconstruction, SHA-256 tampering rejection, model/dimension mismatch rejection,
and offline question-answering.
Total test suite now comprises **32 passing unit and integration tests**.

## Stage 2, live persisted run (Run 2): Komarov et al. (2026) artifact generation

Because the initial live smoke test (Run 1, `results/live_smoke_test_ragu.json`) was
executed prior to Slice 6 persistence and only wrote aggregate metrics before its
Python process exited, an independent second live processing run was executed on
2026-11-18 (`created_at: 2026-11-18T14:46:26.795134+00:00`) using `--save-artifacts`:

```powershell
python scripts/process_papers.py --papers data/papers/komarov_2026_ragu.pdf --save-artifacts data/processed/komarov_2026_ragu --output-summary results/live_ragu_persisted_summary.json
```

### Recorded Measurements (Run 2 vs. Historical Run 1)
This run is recorded separately from Run 1 (`results/live_smoke_test_ragu.json`, which
remains untouched):
- **Target Document**: `data/papers/komarov_2026_ragu.pdf` (10 pages, 54 chunks,
  `document_id`: `922a692238adfe2db666bb65`, SHA-256: `7cc09c9f3555ad4140464e69f407608b73ee5755354a04d9c0a964efae48e404`).
- **Dense Embeddings**: `text-embedding-3-large`, 54/54 vectors persisted in
  `data/processed/komarov_2026_ragu/vectors.npz`, all verified at **3,072 dimensions**.
- **Extraction Model**: `gpt-6-luna` via `beta.chat.completions.parse`.
- **Entities Extracted**: **741 total entity mentions** across 54 chunks, merged into
  **491 unique entities** in the reconstructed NetworkX graph (compared with 785 mentions /
  528 unique entities in Run 1 due to natural LLM sampling variance across 54 calls).
- **Relationships Extracted**: **595 relationship claims** across 54 chunks, filtered by
  verbatim quote and entity-endpoint grounding to **80 explicit directed edges** (compared
  with 639 claims / 92 explicit edges in Run 1).
- **Execution Time**: `1,247,197.133 ms` (~20.79 minutes).
- **Estimated Spend**: ~$0.0496 USD ($0.001622 embeddings + ~$0.0480 extraction).
- **Persisted Artifacts & SHA-256 Checksums (`data/processed/komarov_2026_ragu/`)**:
  - `manifest.json` (`schema_version: 1.0.0`, `is_mocked: false`)
  - `documents.json`: `dfce68695260f237a5f2fddc99a8c8afbe69d4c56d9298a180dad72137a3eb84`
  - `chunks.json`: `aceb24f20e329778c8ad843eb33aefc6d8cfb1c7339df2c13a416afa0f6b2761`
  - `extractions.json`: `b50cb3b12f0ed9f7c08b12f108ce894c187232bcc8f5d9c4663c9410f5c61eb5`
  - `vectors.npz`: `1d79719d35e0fd602581d4fa13e2c9afff338a8aa4b314c05b13234f04d43a3d`
- **Verification**: Reloaded via `load_artifacts("data/processed/komarov_2026_ragu")` with
  zero API calls; all 4 file hashes, 54 chunks, 54×3,072 vectors, 491 graph nodes, and 80
  explicit edges with page and quote provenance verified.
