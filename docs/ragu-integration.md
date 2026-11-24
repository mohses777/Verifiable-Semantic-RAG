# RAGU integration investigation

Status (updated 2026-11-24): RAGU v0.1.0 (commit `abd3f29`) is installed in an isolated environment and has been
run live: two graph builds, one Q1 query and 39 benchmark answers, with 0 errors. The earlier sections below are a
dated record of the investigation; statements in them about RAGU not yet being installed or run were true when
written and are superseded by the later sections and by `docs/benchmark-investigation.md`.

## Sources inspected

- [Official RAGU repository](https://github.com/RaguTeam/RAGU): current README
  recommends cloning and installing with `uv pip install -e .`; PyPI package
  is `graph_ragu`. Optional local-model dependencies use `graph_ragu[local]`.
- [RAGU model module documentation](https://github.com/RaguTeam/RAGU/blob/main/ragu/models/README.md)
- [RAGU components documentation](https://github.com/RaguTeam/RAGU/blob/main/docs/en/ragu_components.md)
- Local upstream source under `third_party/RAGU`, cloned from
  `https://github.com/RaguTeam/RAGU.git`.

## Findings

RAGU exposes `LLMOpenAI(client, model_name=...)` and
`EmbedderOpenAI(client, model_name=..., dim=...)`, backed by its
`CachedAsyncOpenAI` client. The LLM layer uses OpenAI chat-completion APIs and
supports plain text and Pydantic structured outputs. Its documented examples
use `gpt-4o-mini`. The embedder documentation explicitly shows
`text-embedding-3-large` with `dim=3072`; so the embedding model and requested
dimension are supported by the interface.

The intended model IDs can be placed in RAGU's configurable `model_name`
parameters. This is evidence of interface-level configurability, not proof of
runtime compatibility. Source search did not find a `reasoning_effort` setting
in RAGU's model implementation. If GPT-6 Luna requires
`reasoning_effort="none"` for Chat Completions function calling, the relevant
request path may need an explicit configuration hook or client-level default.
RAGU's structured output behavior and any function/tool calling path must be
checked against the actual request construction before claiming compatibility.

## Compatibility assessment

| Component | Assessment | Evidence / caveat |
| --- | --- | --- |
| `gpt-6-luna` model ID | Plausibly configurable | `LLMOpenAI` accepts a model name; no runtime request has been made. |
| Chat Completions | Unverified | RAGU uses OpenAI chat completion interfaces; check model API compatibility and request parameters. |
| `reasoning_effort=none` | Not currently exposed in inspected configuration | Inspect `ragu/models/llm.py` and request call sites; a small upstream wrapper/config change may be needed. |
| Structured outputs | Potential compatibility risk | RAGU parses Pydantic responses; inspect whether it uses `parse`/JSON schema or tool calling and compare GPT-6 Luna support. |
| `text-embedding-3-large`, 3072 | Interface supported | RAGU documents this model and `dim=3072`; runtime dimension still needs a no-cost or approved test. |

Thus, model support appears promising at the configurable interface level, but
runtime compatibility is not established.

## Implemented project-side adapter

`src/integrations/ragu_openai.py` provides `GPT6LunaRAGUClient`, a subclass of
RAGU's `CachedAsyncOpenAI`. It wraps only the configured OpenAI client's
`chat.completions.create` resource. For the configured `LLM_MODEL`, it adds
`reasoning_effort="none"` only when RAGU sends a forced function call: the
outgoing request has both a tool definition and a forced `tool_choice`.

Plain chat completions, Pydantic parsing through
`beta.chat.completions.parse`, and `embeddings.create` pass through untouched.
The adapter uses no model identifier in its source; `client_from_environment()`
reads `LLM_MODEL`, `OPENAI_BASE_URL`, and `OPENAI_API_KEY` from the environment.

Before RAGU calculates its response-cache key, the adapter adds the internal
`_project_reasoning_effort="none"` marker for GPT-6 Luna forced-tool calls. It
removes that marker before RAGU validates provider arguments. Since RAGU's cache
serializes kwargs into its key, this prevents cache sharing across differing
reasoning configurations while ensuring that the internal marker is never sent
to OpenAI. Rate limiting, retrying, cache storage, schema construction, and
response parsing remain in RAGU's inherited implementation.

`tests/test_ragu_openai_adapter.py` contains four mocked tests. On 2026-11-17,
`python -m unittest discover -s tests -v` completed with four passing tests:
normal chat, Pydantic parse, forced function calling with
`reasoning_effort="none"`, and embeddings. The tests use mock modules and do
not install dependencies or make network/API requests.

## Pinned version and dependencies (recorded 2026-11-24)

- Checkout: `third_party/RAGU` (git-ignored), tag `v0.1.0`, commit
  `abd3f2958c186dc624bbd42560ad6e1e7033efaf` (2026-09-08), clean working tree. The pin is
  also in `config/experiment.json`, and every run snapshot checks the checkout against it.
- Package: `graph_ragu 0.1.0`, `requires-python >=3.10` (classifiers list 3.10 to 3.12).
  This project currently runs Python 3.14.4, so the install is a real compatibility risk;
  several dependencies (for example `graspologic-native`, `fastembed`, `lingua-language-detector`,
  `qdrant-client`, `scikit-learn`) may lack 3.14 Windows wheels. A separate Python 3.12
  environment for RAGU is the likely path. (Superseded: RAGU installed and ran on Python 3.14.4; see below.)
- Overlapping pins: `openai>=2.32.0,<3` (project has 2.54.0), `pydantic>=2.11.10,<3` (2.13.4),
  `networkx>=3.4.2,<4` (3.7).
- Not installed in the main environment (deliberately): `pip list` shows no `graph_ragu`.

### Installation check result (2026-11-24, no API calls)

Python 3.12 and `uv` are not installed on this machine (`py -0p` lists only 3.14), and
installing an interpreter was not done without approval. Instead the install was tested
on the project's own Python 3.14.4, which is the stricter test of the main risk:

1. `pip install --dry-run -e .` on 3.14 resolved with no conflicts (no wheel was missing).
2. Created an isolated venv outside the OneDrive folder
   (`%LOCALAPPDATA%erifiable-semantic-ragenv-ragu-py314`); the main environment was not modified.
3. `pip install ./third_party/RAGU` succeeded (exit 0; `graph_ragu 0.1.0`, plus 74 pinned
   dependencies recorded in `config/ragu_env_requirements.txt`). The checkout stayed clean at the pinned SHA.
4. Offline init test: imported `ragu`, `LLMOpenAI`, `EmbedderOpenAI`, `CachedAsyncOpenAI`,
   `KnowledgeGraph`, `Index`; built `GPT6LunaRAGUClient`, `LLMOpenAI(model_name="gpt-6-luna")` and
   `EmbedderOpenAI(model_name="text-embedding-3-large", dim=3072)` with a dummy key and an
   unroutable local URL. All succeeded; no network requests were made.
5. The 4 mocked adapter tests in `tests/test_ragu_openai_adapter.py` pass inside the RAGU venv
   (they use mock modules, so they do not exercise RAGU's real request path).

**Not verified:** actual indexing/query behaviour; whether every extraction path reaches the
adapter's `reasoning_effort="none"` hook; whether `gpt-6-luna` accepts RAGU's request shapes;
and tiktoken's first-use tokenizer download (it needs network access; free but not yet tested).
A Python 3.12 environment remains a fallback only if 3.14 fails at runtime.


## Source findings on request paths and outputs (2026-11-24, source reading only; RAGU not run)

Inspected at the pinned commit `abd3f29`:

- **Structured outputs use `parse` by default.** `CachedAsyncOpenAI.chat_completion` defaults
  `as_tool=False` (`ragu/models/openai.py`), which calls `client.beta.chat.completions.parse`
  with `response_format`, the same path as this project. Nothing in the `ragu/` library sets
  `as_tool=True` (only its mock server mentions it). So the forced-tool
  `reasoning_effort="none"` hook in `GPT6LunaRAGUClient` is probably never triggered in a
  default RAGU pipeline. That reduces the earlier "reasoning_effort=none" concern, but only a
  live run can confirm it.
- **No hard-coded sampling parameters.** `temperature`/`top_p` are sent only if the caller passes
  them (via `LLMOpenAI(**kwargs)` or per call). The live probe showed `gpt-6-luna` rejects
  `temperature=0.0` (only the default 1 is supported), so **do not pass `temperature` to RAGU
  for this model**. RAGU's defaults are compatible.
- **Parameter parity limits.** RAGU's client supports `max_completion_tokens`, but its call
  path only recognizes `temperature` and `top_p` as extra per-call arguments; it has no hook for
  `seed` or `reasoning_effort`. If this project later enables `seed`/`reasoning_effort` for its
  own system, RAGU would need an adapter extension to keep generation settings equal. Until
  then both systems should run with provider defaults, as in all current results.
- **Outputs and provenance.** Query engines return `SearchEngineResponse(response, retrieval,
  payload)`: a free-form answer (`str` or a Pydantic model) plus the retrieved context
  (entities, relations, summaries, chunks). Entities and relations carry `source_chunk_id`
  lists, so provenance exists inside retrieval, but the source shows no quote-level citations
  in the generated answer. Quote-verified citations are therefore a property of this project's
  design, and RAGU's answers can be compared on context-support and correctness, not on
  citation precision, unless RAGU is wrapped (not done).
- **Pipeline differences to document for fairness:** RAGU uses its own chunker
  (`SimpleChunker(max_chunk_size=1000)` in its README example), a `TwoStageArtifactsExtractorLLM`
  with optional ICL examples and entity/relation validation, optional LLM summarization and
  community summaries, and four search strategies (local, global, naive, mix) plus a query-plan
  wrapper. Its embedder uses tiktoken truncation and defaults to batch size 500. None of this
  matches this project's 1000-character/200-overlap chunks, single-pass extraction and hybrid
  retrieval, so architecture and configuration effects must be separated in any comparison.

## Build runner, offline smoke test and pre-run estimate (2026-11-24)

`scripts/run_ragu_build.py` (run with the RAGU venv interpreter) builds a RAGU knowledge graph over
one document. Source text comes from this project's PDF ingestion (10 pages, 41,129 characters),
so both systems read the same text. Modes: default dry run (chunk statistics, no API calls),
`--mock-server` (full pipeline against RAGU's local mock server, zero cost), and `--execute
--max-usd N` (live). Live mode wraps the OpenAI client in `src/integrations/ragu_metering.py`,
which records API-reported usage per call, counts calls per endpoint, and refuses every further
call once the spend cap is reached (4 tests; refused calls never reach the API). RAGU's own response
cache is enabled in the working directory (hits cost nothing and are not metered). The run record
stores models, RAGU configuration, the configuration snapshot (including the RAGU pin check),
usage, cost, errors (secrets redacted) and timing.

**RAGU configuration used** (RAGU's documented README example, not tuned): `SimpleChunker`
(1000 characters, overlap 0), `TwoStageArtifactsExtractorLLM` with ICL (2 examples) and entity and
relation validation, LLM summarization, community summaries, isolated nodes removed, language
English, no `temperature`/`top_p`, 4 concurrent requests, retries after 2, 4 and 8 s.

**Offline smoke test results** (mock server; no cost; scratch directory, nothing written to the repo):
- RAGU's chunker produces **48 chunks** from the text (this project: 54), so chunk boundaries differ.
- The pipeline ran end to end through extraction, validation and vectorization. It ended with an
  embedding-dimension error only because the mock returns 1-dimensional vectors (mock limitation).
- **192 chat calls = 4 per chunk** (entity extraction, entity validation, relation extraction,
  relation validation) plus 2 embedding calls, before any summarization (the mock returns no entities,
  so summarization and community-summary calls were not exercised).
- **All 192 chat calls used `beta.chat.completions.parse`; none used `chat.completions.create`**, so the
  forced-tool `reasoning_effort="none"` hook was not triggered for these stages.
- Request size: 1,302,936 characters of chat prompts (about 326k tokens at 4 characters per token) with
  empty entity lists. Real runs embed extracted entities and relations in later prompts, so real prompts
  will be larger.

**Pre-run cost estimate** (assumptions, not measurements; RAGU's real call count is unknown until a
live run): 300 to 600 chat calls (192 known plus summarization and community summaries); 0.5M to 0.8M
prompt tokens; 800 to 2,500 completion tokens per call including reasoning (this project's QA calls
used 1,500 to 2,000); about 0.1M embedding tokens. At the configured rates
(`gpt-6-luna` $0.10 in / $0.50 out, embeddings $0.13 per million):

| Scenario | Estimated cost |
| --- | --- |
| Low | about $0.19 |
| Expected | about $0.43 |
| High | about $0.85 |

Expected wall time with 4 concurrent requests: roughly 25 to 50 minutes. A hard cap of $1.00 is
proposed. If the cap is reached the build stops and the record is marked `budget_exceeded`
(an incomplete graph, not a result).

## Live RAGU build result (executed 2026-11-24, `results/ragu_build_komarov_2026.json`)

Approved cap $1.00. Run on the dirty working tree (base commit `1a0ca56`), RAGU pinned at
`abd3f29` (pin check passed), `gpt-6-luna` and `text-embedding-3-large` (3072), no request
parameters sent, RAGU documented example configuration (above). Output graph in
`data/ragu/komarov_2026_ragu/` (about 85 MB, including the git-ignored LLM cache).

| Measure | Result |
| --- | --- |
| Status / errors | complete / none; budget not exceeded; 0 calls blocked |
| Graph | **563 entities, 600 relations, 57 communities, 48 chunks** (GML file has 563 nodes, 600 edges) |
| API calls (measured) | **258 chat + 6 embedding** |
| Endpoints | all 258 chat calls via `beta.chat.completions.parse`, **none via `chat.completions.create`** |
| Tokens (API-reported) | 1,795,780 prompt; 527,505 completion (173,204 reasoning); 64,704 embedding |
| Cost | **$0.451742** (at `config/pricing.json` rates) |
| Wall time | **1,032.8 s (17.2 min)** with 4 concurrent requests |

Estimate vs measured (the total was close; the parts were not): estimated expected $0.43
(range $0.19 to $0.85), measured $0.4517. Calls: 258 measured against 300 to 600 assumed
(192 extraction and validation calls plus 66 others). Prompt tokens: 1.80M measured against 0.5M to
0.8M assumed (RAGU's later-stage prompts embed extracted entities, as predicted, but by far more than
assumed). Completion tokens: 0.53M measured against 0.24M to 1.5M assumed (about 2,000 per call).
The characters-per-4 rule slightly overestimates (8.79M request characters, 1.80M tokens measured).
Errors cancelled out, so the estimator should not be treated as reliable.

Notes and limits:
- All stages including summarization used `parse`, so the forced-tool `reasoning_effort="none"`
  hook never fired in this build. A query-time path has not been tested.
- **No cost or speed comparison with this project is valid yet.** This project's extraction
  costs in `results/live_smoke_test_ragu.json` and the persisted run are pre-run estimates, not
  measured usage, and RAGU ran 4 requests concurrently while this project extracts sequentially.
- The graph counts are not a quality measure. RAGU merges and summarizes entities, this project keeps
  per-mention records, and the entity definitions differ, so counts must not be compared as if equal.
- The run is a single sample of an LLM process with no seed; a rebuild would give a different graph.
  The record is the evidence for this graph, which should be rebuilt only deliberately (a rebuild with
  the cache present costs nothing and reproduces cached answers, which is not an independent sample).

## First RAGU query: Q1 with local search (executed 2026-11-24, `results/ragu_q1_local.json`)

Same question and source paper as the three Q1 runs of this project. RAGU `LocalSearchEngine` with
default parameters (not tuned), graph from the build above, hard cap $0.02, dirty tree (base `1a0ca56`),
no request parameters sent. One sample.

| Measure | This project (diagnostic Q1) | RAGU local search |
| --- | --- | --- |
| Retrieved evidence | 10 chunks, 13 graph relationships | 20 entities, 75 relations, 35 chunks (50,334 characters of context) |
| Answer prompt tokens | 5,140 | 13,695 |
| Completion tokens (reasoning) | 1,503 (790) | 612 (392) |
| Cost (measured) | $0.00127 | $0.00168 (incl. 37-token query embedding) |
| Latency | 19.3 s | 8.3 s |
| Evidence verification | 2 of 2 quotes accepted by strict check | none: free-text answer, no quotes |
| Chunk IDs recorded | yes, with scores and sources | yes (IDs and text of all retrieved chunks), not tied to claims |
| Answer endpoint | `beta.chat.completions.parse` (structured) | `chat.completions.create` (plain text) |

Observations (spot check by hand, not an evaluation):
- Both answers state the same task-level figures (54.2, 53.7, 64.1, 59.0 against 72.4, 68.4, 65.0, 56.9; Coverage
  57.4 against 34.7; Faithfulness 34.2 against 26.6). All 14 numbers in RAGU's answer appear verbatim in
  the paper text. RAGU also states evidence recall of 0.84 against at most 0.76, which matches the paper's
  abstract.
- RAGU's answer contains no citations. Its retrieved context holds the chunks that contain the key numbers
  (72.4 appears in 4 retrieved chunks), but nothing ties individual claims to them, so the answer cannot
  be verified the way this project's can. That is an architectural difference, not a correctness one.
- **Retrieval budget confound:** RAGU supplied about 2.7 times as many input tokens (13,695 against 5,140)
  and 35 chunks against 10. Any quality or cost comparison would conflate architecture with context size.
  Equalizing the context budget is a design decision for the comparison and has not been made.
- The plain-text answer call used `chat.completions.create` without tools, so the forced-tool
  `reasoning_effort="none"` hook did not fire here either.
- One sample for one question on the paper that the graph was built from; it says nothing about rates or
  about GraphRAG-Bench.


Upstream recommends `uv pip install -e .`; it also documents `pip install
graph_ragu`. Its project metadata and lock file should be checked for Python
version and platform constraints before choosing the reproducible Windows
environment. The optional `[local]` extra is unnecessary for the planned
OpenAI-hosted models. (Superseded 2026-11-24: installed and run successfully on Windows with Python 3.14.4; see
"Installation check result".)

## Configuration target

Keep secrets in environment variables and model settings in local environment
configuration:

```dotenv
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-6-luna
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=3072
```

Use the RAGU constructors with these model values and explicit embedding
dimension. Do not make paid API calls until a cost estimate is reviewed and
approved.

## Compatibility test plan (no calls executed)

1. Record the cloned RAGU commit SHA and Python/package versions.
2. Inspect the exact chat request construction for extraction, summarization,
   answer generation, structured outputs, and any function/tool calling.
3. Confirm whether a configurable `reasoning_effort="none"` can reach every
   applicable Chat Completions request. If not, propose a minimal project-side
   adapter; do not edit upstream RAGU silently.
4. Run local unit tests or mocked-client tests to verify request payload shape
   and response parsing without network access.
5. After separate user approval and cost estimate, run one minimal API smoke
   test for chat and one embedding request; verify returned embedding length is
   exactly 3072.
6. Only then describe runtime compatibility as tested, and record model IDs,
   request settings, usage, cost, and errors.

## Open issues

Resolved by the live runs (2026-11-24):
- RAGU never issued a forced tool call in any recorded run: every chat request was `beta.chat.completions.parse`
  (graph builds) or a plain `chat.completions.create` without tools (answers). The adapter's
  `reasoning_effort="none"` hook therefore **never fired**; the adapter was inert and did not change any request.
- `gpt-6-luna` accepted every RAGU request (0 API errors across the four live runs).
- Python 3.14.4 on Windows works for RAGU's dependencies (`config/ragu_env_requirements.txt`).
- Runtime compatibility: tested.
