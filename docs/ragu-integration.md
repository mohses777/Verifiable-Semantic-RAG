# RAGU integration investigation

Status: source inspection only. No installation, runtime model requests, or
paid API calls have been performed. The upstream `main` branch is a moving
target; record its commit SHA with any later experiments.

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

## Installation and Windows considerations

Upstream recommends `uv pip install -e .`; it also documents `pip install
graph_ragu`. Its project metadata and lock file should be checked for Python
version and platform constraints before choosing the reproducible Windows
environment. The optional `[local]` extra is unnecessary for the planned
OpenAI-hosted models. Dependency installation has not been attempted, so Windows
compatibility remains unverified.

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

- Whether the live GPT-6 Luna API accepts every RAGU request/prompt combination
  after the adapter inserts `reasoning_effort="none"` for forced tools.
- Whether every RAGU extraction path that selects `as_tool=True` reaches the
  adapter in an installed environment; the adapter is designed at the shared
  `CachedAsyncOpenAI` boundary.
- Python and transitive dependency support on the target Windows installation.
- Runtime compatibility remains untested, by design.
