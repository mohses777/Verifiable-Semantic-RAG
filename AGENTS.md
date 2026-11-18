# Verifiable Semantic RAG — Project Instructions

## Research objective

This is a Master's research project titled **Design and Evaluation of a
Verifiable and Interpretable Retrieval-Augmented Generation Framework Using
Structured Semantic Knowledge Bases.**

Implement and evaluate a verifiable semantic Retrieval-Augmented Generation
(RAG) system. The intended comparison is between:

- this project's semantic RAG implementation;
- RAGU, an existing RAG solution; and
- a suitable, documented subset of GraphRAG-Bench, after confirming its
  licensing, dataset format, and evaluation compatibility.

## Implementation responsibilities

- Implement the research system in Python.
- Support document ingestion, chunking, entity and relationship extraction,
  structured knowledge storage, evidence retrieval, and grounded answer
  generation.
- Preserve provenance from answers to evidence, chunks, and source documents.
- Return answers with clear evidence references.
- Keep this implementation separate from third-party implementations.
- Inspect RAGU's official repository and documentation before integrating it.
- Use reproducible scripts and machine-readable experiment outputs.
- Record experiment configurations, model identifiers, benchmark question IDs,
  metrics, errors, token use/cost when available, and execution times.
- Keep the project reproducible on Windows.

## Model configuration — updated RAGU comparison

The primary controlled experiment uses an updated RAGU configuration. It is
not an exact reproduction of RAGU's legacy GPT-4o-mini configuration.

### Required models

- Generation, extraction, and summarization where applicable: `gpt-6-luna`
- Dense embeddings: `text-embedding-3-large`
- Embedding dimension: `3072`

### Requirements

1. Configure this project's RAG implementation to use `gpt-6-luna` for LLM
   operations and `text-embedding-3-large` for embeddings.
2. Inspect RAGU's actual configuration and determine whether these models can
   be configured for corresponding operations. Record compatibility limits.
3. Use the official OpenAI API for the primary controlled experiment unless a
   documented technical limitation prevents it.
4. Never silently substitute a different model or provider.
5. Keep model names, provider, temperature, token limits, and relevant settings
   in version-controlled configuration files.
6. Record exact model identifiers in every experiment result.
7. Keep API keys only in environment variables; never commit credentials or
   `.env` files.
8. Before any paid API call, estimate the expected cost and ask the user for
   approval.
9. If a model is unavailable or an API call fails, report the exact error;
   never substitute another model automatically.
10. Clearly disclose this as an updated RAGU configuration rather than an
    exact reproduction of its legacy GPT-4o-mini setup.

### Experimental fairness

- Use identical benchmark questions and source documents across compared systems.
- Use the same generation model and, where compatible, embedding model.
- Document differences in chunking, retrieval, graph construction, prompts,
  extraction, and evaluation procedures.
- Separate architecture-related effects from model-related effects.
- Do not claim an architecture is better when a configuration difference could
  explain the result.
- `openrouter/free` is not part of the primary controlled comparison.

## Research integrity and safety rules

1. Never invent experimental results.
2. Never say a test passed unless it was actually run and its result verified.
3. Do not fabricate citations, papers, benchmark details, or dataset results.
4. Verify repository documentation before integrating external projects.
5. Prefer small, testable implementation steps.
6. Do not delete or overwrite existing work without explicit approval.
7. Do not expose API keys, or commit `.env` files.
8. Document limitations and differences in experimental conditions.

## Recommended repository layout

```text
verifiable-semantic-rag/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .env                 # local only; ignored by Git
├── .env.example
├── src/
├── data/
├── results/
└── docs/
```

For the primary experiment, use these environment variable names in
`.env.example` (with no real secret values):

```dotenv
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-6-luna
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=3072
```

## Workflow and reporting

Work in stages: project inspection, own RAG implementation, RAGU setup,
benchmark integration, experiments, then research report.

At the end of each stage, report exactly:

- files changed;
- commands executed;
- tests that passed or failed;
- errors and actual experiment results;
- outstanding issues; and
- the next recommended task.

Do not present unverified work as completed.
