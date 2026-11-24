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
- **Financial Cost (estimate, not measured)**: `results/live_smoke_test_ragu.json` records **$0.049618 USD** ($0.001622 embeddings + $0.047996 extraction). Its token counts are the pre-run estimator's heuristics (`estimate_processing_cost`), copied into the report, and were priced at the superseded $1.25/$5.00 per-million LLM rates. They are not API-reported usage. At the current `gpt-6-luna` rates ($0.10/$0.50, `config/pricing.json`) the same estimate is about $0.0060. Measured spend for this run was not recorded; check the OpenAI billing dashboard if an exact figure is needed.
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
Test suite at this slice comprised **32 passing unit and integration tests**.

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
- **Estimated Spend**: ~$0.0496 USD, the same superseded-rate pre-run estimate as Run 1 (not measured API usage).
- **Persisted Artifacts & SHA-256 Checksums (`data/processed/komarov_2026_ragu/`)**:
  - `manifest.json` (`schema_version: 1.0.0`, `is_mocked: false`)
  - `documents.json`: `dfce68695260f237a5f2fddc99a8c8afbe69d4c56d9298a180dad72137a3eb84`
  - `chunks.json`: `aceb24f20e329778c8ad843eb33aefc6d8cfb1c7339df2c13a416afa0f6b2761`
  - `extractions.json`: `b50cb3b12f0ed9f7c08b12f108ce894c187232bcc8f5d9c4663c9410f5c61eb5`
  - `vectors.npz`: `1d79719d35e0fd602581d4fa13e2c9afff338a8aa4b314c05b13234f04d43a3d`
- **Verification**: Reloaded via `load_artifacts("data/processed/komarov_2026_ragu")` with
  zero API calls; all 4 file hashes, 54 chunks, 54×3,072 vectors, 491 graph nodes, and 80
  explicit edges with page and quote provenance verified.

## Stage 2, Slice 7: first live question-answering experiment

Executed on 2026-11-18 against pre-loaded `data/processed/komarov_2026_ragu/` artifacts
(`is_mocked: false`, saved to `results/live_qa_ragu_q1.json`).

### Experimental Question
> *"According to the RAGU paper, what performance does RAGU achieve on GraphRAG-Bench (Medical), and how does it compare with HippoRAG 2?"*

### Recorded Measurements & Actual API Cost
- **Chunk Re-embedding / Re-extraction Calls**: **`0`** ($0.00).
- **Hybrid Retrieval (`vector_top_k=5`, `graph_traversal_depth=1`, `max_total_chunks=10`)**:
  - **10** deduplicated evidence chunks retrieved.
  - **188** entity mentions and **13** explicit graph relationships retrieved.
- **Query Embedding (`text-embedding-3-large`)**: **37** actual tokens (`$0.000005 USD`).
- **Answer Generation (`gpt-6-luna`)**: **5,140** actual prompt tokens, **2,038** actual
  completion tokens (**7,178** total tokens, `$0.001533 USD`).
- **Total Actual Cost**: **`$0.001538 USD`**, computed from measured token usage at the `config/pricing.json` rates (`gpt-6-luna` $0.10/$0.50 per million tokens, confirmed on the OpenAI pricing page on 2026-11-24).
- **End-to-End QA Latency**: **`33,475.21 ms`** (~33.48 seconds — a **37× speedup** over
  full document extraction).

### Answer Correctness & Citation Verification Audit
The generated answer reported:
- **RAGU (Meno-Lite-0.1) Answer Correctness (AC)** on GraphRAG-Bench (Medical):
  **54.2%** on Fact Retrieval, **53.7%** on Complex Reasoning, **64.1%** on Contextual
  Summarize, and **59.0%** on Creative Generation.
- **Comparison with HippoRAG 2**:
  - Fact Retrieval: `-18.2` percentage points (`54.2%` vs. `72.4%`).
  - Complex Reasoning: `-14.7` percentage points (`53.7%` vs. `68.4%`).
  - Contextual Summarize: `-0.9` percentage points (`64.1%` vs. `65.0%`).
  - Creative Generation AC: `+2.1` percentage points (`59.0%` vs. `56.9%`).
  - Creative Generation Coverage: `57.4%` vs. `34.7%`.
  - Creative Generation Faithfulness: `34.2%` vs. `26.6%`.

All three returned `EvidenceCitation` records passed application-level chunk allow-list
and verbatim quotation verification against the source PDF:
1. `922a692238adfe2db666bb65:s3:c0` (**Page 4**, chars `0..995`): the verified quote is the
   Table 1 caption only (identifying the table as GraphRAG-Bench Medical generation quality).
   The numerical values in the answer are supported by citations 2 and 3, not by table rows.
2. `922a692238adfe2db666bb65:s2:c3` (**Page 3**, chars `2389..3387`): Section 3.2 text
   reporting the exact cross-over deltas (`-18.2 pp`, `-14.7 pp`, `-0.9 pp`, `59.0 vs. 56.9`,
   and Faithfulness `34.2 vs. 26.6`).
3. `922a692238adfe2db666bb65:s2:c4` (**Page 3**, chars `3187..4130`): Section 3.2 continuation
   reporting Creative Generation Coverage (`57.4 vs. 34.7`).

### Methodological Observation for Graph Seed Matching
Offline inspection of `match_entities_in_text()` for Q1 showed that alongside the true
target entities (`RAGU`, `GraphRAG-Bench`, `GraphRAG-Bench (Medical)`, `HippoRAG 2`),
two short 2-letter acronym nodes (`AC` and `RE`) matched as case-folded substrings inside
the query words `"achieve"` and `"compare"`. This was fixed on 2026-11-24: `match_entities_in_text()` now requires word-boundary matches
(regression test added). On the Q1 question the seeds are now `GraphRAG`, `GraphRAG-Bench`,
`GraphRAG-Bench (Medical)`, `HippoRAG`, `HippoRAG 2`, `RAGU`. Note that Q1's recorded results
were produced before this fix, so rerunning it could retrieve different graph evidence.

## Reproducibility configuration (2026-11-24)

`config/experiment.json` is the version-controlled record of model settings
(`gpt-6-luna`: `temperature`, `reasoning_effort`, `max_completion_tokens` and `seed` are all
`null`, i.e. not sent, which is how Run 1, Run 2 and Q1 were produced), the embedding model
and dimension, and the pinned RAGU version. `validate_environment()` aborts if `.env`
values contradict it. `build_run_configuration()` snapshots the effective settings, pricing,
git commit and dirty flag, Python/package versions, chunking, retrieval and the RAGU pin
check into every new QA result (`run_configuration`) and artifact manifest.
`QAPipelineSummary` also records `query_matched_entities` and `seed_entities` (the graph
seeds), and `artifact_build_configuration` (how the loaded artifacts were built).
Extraction and embedding providers now accumulate API-reported usage, so future ingestion
runs record *measured* token usage and cost (`token_usage.*.measured: true`) instead of
estimates. The new manifest fields are additive: the existing Run 2 artifacts load
unchanged (verified by test) and simply have empty `run_configuration`/`token_usage`.
Because the LLM `seed` is unset, outputs are not guaranteed to be bit-reproducible.

### Q1 before/after the seed fix (offline, no API calls)

For the Q1 question the old substring matcher produced seeds
`AC, GraphRAG, GraphRAG-Bench, GraphRAG-Bench (Medical), HippoRAG, HippoRAG 2, RAGU, RE,
poRAG 2`; the fixed matcher produces `GraphRAG, GraphRAG-Bench, GraphRAG-Bench (Medical),
HippoRAG, HippoRAG 2, RAGU`. Three spurious seeds (`AC`, `RE`, and `poRAG 2`, which matched
inside "HippoRAG 2") were removed from the *query-matched* seeds.
`results/live_qa_ragu_q1.json` is preserved unmodified as the historical pre-fix experiment. The built-in `--estimate-cost` figure for it is about $0.0005, but it assumes 300 completion tokens whereas Q1 actually used 2,038 (likely reasoning tokens), so budget roughly $0.0015 to $0.003 for the rerun.

### Q1 post-fix rerun (executed 2026-11-24, `results/live_qa_ragu_q1_rerun_post_seed_fix.json`)

Run on the same artifacts and retrieval settings (`vector_top_k=5`, depth 1, 10 chunks), with
the working tree dirty (`git_dirty: true`, base commit `1a0ca56`), approved cost ~$0.003.

| Measure | Original Q1 | Post-fix rerun |
| --- | --- | --- |
| Retrieved chunks / graph entities / relationships | 10 / 188 / 13 | 10 / 188 / 13 (identical) |
| Answer prompt tokens | 5,140 | 5,140 (identical) |
| Completion tokens | 2,038 | 1,622 |
| Actual cost | $0.001538 | $0.00133 |
| Latency | 33,475 ms | 23,135 ms |
| `insufficient_evidence` | false, 3 verified citations | **true, 0 verified citations** |

Findings (single sample each, so these are observations, not rates):
- **The seed fix did not change Q1's retrieval.** The graph seeds are the query-matched
  entities plus every entity extracted in the top vector chunks; the latter (53 entities,
  including `AC`, `RE` and `poRAG 2`) are legitimate chunk-derived seeds. Only the 6
  query-matched seeds changed, and the evidence sent to the model was identical in size.
- **The different outcome therefore comes from LLM output variance**, not retrieval: same
  input, no `temperature`/`seed` set (both `null`), different answer. In the rerun every
  returned citation failed chunk/quote verification, so the generator failed closed.
- Why the citations failed cannot be determined from the saved record: stripped citations
  and the raw model answer are not persisted. The original run's verbatim quotes spanned
  PDF line breaks (e.g. `through-
out`), so quote-normalization fragility is a hypothesis,
  not a finding.
- **Implication:** answer and citation outcomes are not stable across runs under the
  current settings. Retrieval consistency (identical here) and answer/citation consistency
  (different here) are separate properties. This pair of runs is preliminary evidence of
  variation in the second only; it is **not** evidence that the seed fix improved retrieval.
  The fail-closed outcome is the intended behaviour of a verifiable RAG system, not a
  pipeline failure.

### Q1 diagnostic run (executed 2026-11-24, `results/live_qa_ragu_q1_diagnostic.json`)

Third live sample of the same question, artifacts and retrieval settings, default model
parameters (none sent; `params_sent = {"model": "gpt-6-luna"}`), run with the new diagnostics
on a dirty tree (base commit `1a0ca56`). Approved cost estimate $0.0015 to $0.003.

| Measure | Original Q1 | Post-fix rerun | Diagnostic run |
| --- | --- | --- | --- |
| Retrieved chunks / entities / relationships | 10 / 188 / 13 | 10 / 188 / 13 | 10 / 188 / 13 |
| Answer prompt tokens | 5,140 | 5,140 | 5,140 |
| Completion tokens (reasoning tokens) | 2,038 (n/a) | 1,622 (n/a) | 1,503 (790) |
| Cost | $0.001538 | $0.00133 | $0.00127 |
| Latency | 33.5 s | 23.1 s | 19.3 s |
| Raw / accepted / rejected citations | 3 / 3 / not recorded | not recorded: all failed closed | 2 / 2 / 0 |
| Cited chunks | `s3:c0` (caption), `s2:c3`, `s2:c4` (prose) | none | `s3:c0` twice (table header and rows) |
| `insufficient_evidence` | false | **true** | false |

Observations (three single samples, not rates):
- **The failure did not recur, so its cause is still undetermined.** The diagnostic run produced
  no rejected citation, so there is nothing to diagnose; if a fail-closed outcome occurs again,
  `answer_diagnostics` will record the reason per citation. Three samples cannot estimate a
  failure rate.
- **Retrieval consistency can only be partly verified.** All three runs report identical counts
  and an identical 5,140-token prompt, but chunk IDs were recorded only in this run, so identity
  of the chunk sets across the first two runs is inferred, not demonstrated.
- **Citation sets are not stable even when the answer succeeds.** The two successful runs cite
  different chunks and quotes, and the completion length varied (2,038, 1,622, 1,503 tokens;
  790 of the last were reasoning tokens).
- **Verbatim quoting of messy PDF text can pass strict verification.** The accepted table quote
  includes extraction artefacts (`72.468.4 65.051.7`), copied exactly. The strict rule is
  unchanged.
- **`provenance_overridden: true` on accepted citations** only means the model left
  `document_id`, `source_path` and `page_number` empty or different, and the authoritative
  chunk values replaced them, as designed.
- **Manual claim-level check (one answer, by hand):** every number in the answer appears in the
  accepted table-row quote, and its mapping to task columns is decided by the header quote
  (citation 1). Neither quote alone supports the claim; support is joint. This is an example of why
  claim-level support (`docs/claim-level-evidence-design.md`) needs a claim-to-citation link and
  cannot be read off per-citation validity. This was a single manual spot check, not an evaluation.

## Citation rejection diagnostics (2026-11-24)

`src/generation/diagnostics.py` explains every accepted or rejected citation. The
**acceptance rule is unchanged and strict** (whitespace collapsing + case folding, then
substring match against the cited chunk) and remains fail-closed; looser normalizations are
used only to *diagnose* a rejection and never to accept a citation.

Each QA result now has `answer_diagnostics` with: the full raw model output; per-citation
records (status, reason, the model's quote, normalized quote, best-match ratio, longest
matching text, chunks where the quote does exist, and model-supplied vs authoritative
provenance with an `provenance_overridden` flag); accepted/rejected counts; a
`rejection_summary`; `fail_closed` and its reason; and `request` (exact request parameters
sent, plus response model, finish reason, system fingerprint and reasoning tokens when the
API returns them). QA results also list `retrieved_chunks` (IDs, scores, sources), so
"identical evidence" can be verified from the record instead of inferred from counts.

Rejection reasons: `empty_quote`, `invalid_chunk_id`, `wrong_chunk_id` (quote exists in another
retrieved chunk; not silently re-attributed), and `quote_not_in_cited_chunk` with a diagnosis in
{`unicode_variant`, `hyphenation_linebreak`, `punctuation_difference`, `spacing_difference`}
(category `formatting_only`), `ellipsis_elision`, `partial_overlap`, or `unsupported_text`
(category `content_difference`). 18 tests cover PDF line breaks, whitespace, punctuation,
Unicode, spacing artefacts, elision, unsupported text, invalid/wrong chunk IDs, provenance
injection, partial acceptance, abstention, and fail-closed behaviour.

### Is the Q1 failure explained by PDF text normalization? (offline, no API calls)

The raw model output of the post-fix rerun was not saved (this feature did not yet exist), so
its rejection cause **cannot be determined**. `scripts/analyze_quote_fragility.py`
(`results/quote_fragility_analysis.json`) measures how fragile verbatim quoting is on the
Komarov PDF text (54 chunks): 49/54 chunks contain a hyphenated line break (205 in total), all
54 contain hard newlines, 47/54 contain non-ASCII characters and 32/54 contain Unicode
dashes/minus signs. Two of the three quotes accepted in the original Q1 contained a
hyphen-linebreak (e.g. `through-
out`); if a model had "cleaned" them, strict verification
would have rejected both (diagnosed `hyphenation_linebreak`, `formatting_only`). So
formatting fragility is a plausible, quantified explanation, but it is a hypothesis; the
next live run will record the actual rejection reason. Whether to add deterministic
de-hyphenation to the acceptance rule is a separate research decision and has **not** been made.

## Reproducibility controls (2026-11-24)

`temperature`, `seed`, `reasoning_effort` and `max_completion_tokens` are configurable in
`config/experiment.json` (`llm.*`). A non-null value is only allowed once
`llm.parameter_support.<name>.status` is `verified`; otherwise loading the settings raises
`UnverifiedParameterError`. All are currently `null` (not sent).

**Live probe result (2026-11-24, `results/llm_parameter_probe.json`, $0.000009, dirty tree):**
`temperature=0.0` was **rejected** by the API (400 `unsupported_value`: only the default 1 is supported),
so temperature cannot be used to reduce variance. `seed=12345`, `reasoning_effort="low"` and
`max_completion_tokens=256` were **accepted** on `chat.completions.create`. Limits of this
evidence: acceptance only, not improved consistency; no `system_fingerprint` was returned; only
the value `"low"` was probed for `reasoning_effort` (the value `none` assumed by the RAGU adapter
is unverified); and the structured-output `parse` path was not separately probed. 
`scripts/probe_llm_parameters.py` sends one tiny request per parameter and records accept/reject
and the exact API error; it defaults to a dry run (verified: no calls) and `--execute` makes
paid calls (worst-case estimate about $0.004, to be approved first). A verified `seed` still does not guarantee
identical outputs. Every QA result records the parameters actually sent
(`answer_diagnostics.request.params_sent`, `run_configuration.llm_request_params_sent`) and the
support status (`run_configuration.llm_parameter_support`). Historical results are unchanged.

### Planned thesis evaluation (not yet implemented)

See `docs/claim-level-evidence-design.md` for why answer-level citation validity is not claim-level
evidence support and how the latter could be measured.


Citation reliability should be evaluated over multiple questions and repeated runs: citation
precision, citation coverage, evidence-supported claim rate, abstention (fail-closed) rate, and
answer correctness, rather than a single successful answer.

## RAGU installation check (2026-11-24)

RAGU `v0.1.0` (`abd3f2958c186dc624bbd42560ad6e1e7033efaf`) installs and initializes in an isolated
Python 3.14.4 venv on Windows with the project's models configured; details, the dependency
lock, and what remains unverified are in `docs/ragu-integration.md`. No API calls were made.

## Cost configuration (2026-11-24)

All rates are read from `config/pricing.json` via `src/core/pricing.py`; a model with no
configured price raises an error instead of falling back. `gpt-6-luna` ($0.10 input / $0.50
output per million tokens) was confirmed against the OpenAI pricing page. The
`text-embedding-3-large` rate ($0.13 per million tokens) was likewise confirmed on its OpenAI model page on 2026-11-24.

## Repository housekeeping (2026-11-24)

- `scripts/smoke_test_ragu_paper.py` was removed. It wrote the pre-run cost *estimate* into report fields that read as
  measured cost, which is why `results/live_smoke_test_ragu.json` shows $0.049618 (an estimate at superseded rates; see
  above). It is superseded by `scripts/process_papers.py`, which records measured usage. Its code remains in git history
  (commit `1a0ca56` and earlier); its result file is kept and documented as an estimate.
- The lost run-1 summary of the benchmark-slice build was renamed to
  `results/pipeline_graphrag_bench_slice_run1_artifacts_lost.json`.
- Cost estimators were recalibrated from measured runs (see `src/pipeline/runner.py` and `src/pipeline/qa.py`); the
  earlier constants (100 completion tokens per extracted chunk, 300 per answer) were far below the measured values.

## Developer reference (moved from the README on 2026-11-25)

These sections were moved verbatim from `README.md` when it was restructured for submission; nothing was
changed except heading levels.

### Strict separation of mocked vs. live experiment artifacts

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

### Reproducibility configuration

- `config/experiment.json`: provider, base URL, `gpt-6-luna` request settings (`temperature`, `reasoning_effort`, `max_completion_tokens`, `seed`; `null` = not sent), embedding model and dimensions, and the pinned RAGU version (tag `v0.1.0`, commit `abd3f2958c186dc624bbd42560ad6e1e7033efaf`).
- `config/pricing.json`: verified model prices with source and date.
- Environment variables (`LLM_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`, `OPENAI_BASE_URL`) must agree with the config or the run aborts; models are never substituted silently.
- Every new QA result and artifact manifest embeds a `run_configuration` snapshot (models, request params, pricing, git commit and dirty flag, Python/package versions, chunking/retrieval settings, RAGU pin check) plus measured token usage and cost. Commit before an experiment so `git_dirty` is `false`.

### Citation diagnostics and parameter controls

- QA results include `answer_diagnostics` (raw model output, per-citation accept/reject with reasons, fail-closed flag, request parameters sent) and `retrieved_chunks`. Citation acceptance is strict and fail-closed; diagnostics only explain rejections.
- `temperature`, `seed`, `reasoning_effort`, `max_completion_tokens` can only be enabled in `config/experiment.json` after `scripts/probe_llm_parameters.py --execute` (paid, tiny) has verified API support. All are currently off.
- `python scripts/analyze_quote_fragility.py` runs the zero-cost PDF quote-fragility analysis.

### Persistent Artifact Storage Layout (`src.storage.artifacts`)

Artifacts are persisted in a portable, human-inspectable format without Python pickle:

```text
data/processed/<artifact-name>/
├── manifest.json     # Schema version (1.0.0), model metadata, and SHA-256 integrity checksums
├── documents.json    # SourceDocument records with page sections and checksums
├── chunks.json       # TextChunk records with exact character/page provenance
├── extractions.json  # ChunkExtraction records with entities, relationships, and support levels
└── vectors.npz       # NumPy float32 vector array and chunk IDs (allow_pickle=False)
```

#### Integrity & Reproducibility Guarantees
- **Atomic Writes**: Writes first to an isolated staging directory, verifies file hashes, and performs an atomic directory swap to prevent corrupted partial artifacts.
- **SHA-256 Validation**: On load, `load_artifacts` recomputes and verifies the hash of every artifact file against `manifest.json`.
- **Model & Dimension Checks**: Rejects artifacts if the vector dimension or embedding model does not match the active configuration.
- **Zero-Cost Loading**: Loading artifacts reconstructs `InMemoryVectorIndex` and `KnowledgeGraph` in memory without making any embedding or extraction API calls.

#### Saving & Loading Artifacts in Document Processing (`scripts/process_papers.py`)

Process documents and save artifacts to disk:

```powershell
python scripts/process_papers.py --papers data/papers/komarov_2026_ragu.pdf --save-artifacts data/processed/komarov_2026_ragu
```

Load existing artifacts from disk (zero API calls):

```powershell
python scripts/process_papers.py --load-artifacts data/processed/komarov_2026_ragu
```

#### Answering Questions with Pre-Loaded Artifacts (`scripts/answer_question.py`)

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
