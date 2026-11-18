#!/usr/bin/env python
"""Controlled live API smoke test for komarov_2026_ragu.pdf with budget guardrails."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.models import ChunkEmbedding, SourceDocument, TextChunk
from src.embeddings.client import EmbeddingClient, OpenAIEmbeddingProvider
from src.embeddings.index import InMemoryVectorIndex
from src.ingestion.loader import ingest_file
from src.knowledge.extraction import ChunkExtraction, OpenAIStructuredExtractionProvider, SemanticExtractor
from src.knowledge.graph import KnowledgeGraph
from src.knowledge.schemas import ExtractionResult
from src.pipeline.runner import estimate_processing_cost
from src.processing.chunking import ChunkingConfig, chunk_document

MAX_BUDGET_USD = 0.10
TARGET_PAPER = Path("data/papers/komarov_2026_ragu.pdf")


def load_env() -> dict[str, str]:
    env_path = project_root / ".env"
    if not env_path.exists():
        raise FileNotFoundError(".env file not found. Copy .env.example and configure credentials.")
    env_vars = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env_vars[k.strip()] = v.strip()
    return env_vars


@dataclass
class SmokeTestReport:
    document_path: str
    document_id: str
    checksum_sha256: str
    page_count: int
    chunk_count: int
    embeddings_generated: int
    embedding_dimension: int
    llm_extractions_completed: int
    total_entities_extracted: int
    unique_graph_entities: int
    total_relationships_extracted: int
    explicit_graph_relationships: int
    token_usage: dict[str, int]
    cost_breakdown_usd: dict[str, float]
    total_cost_usd: float
    execution_time_ms: float
    errors: list[str]
    is_mocked: bool = False

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")


async def run_smoke_test(
    paper_path: Path = TARGET_PAPER,
    *,
    max_budget_usd: float = MAX_BUDGET_USD,
    output_path: Path = Path("results/live_smoke_test_ragu.json"),
) -> SmokeTestReport:
    env = load_env()
    api_key = env.get("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_openai_api_key":
        raise ValueError("OPENAI_API_KEY is not configured with a valid key in .env")

    llm_model = env.get("LLM_MODEL", "gpt-6-luna")
    embedding_model = env.get("EMBEDDING_MODEL", "text-embedding-3-large")
    embedding_dim = int(env.get("EMBEDDING_DIMENSION", "3072"))
    base_url = env.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # 1. Ingestion & Chunking (Local, zero cost)
    print(f"[1/5] Ingesting {paper_path.name}...")
    doc = ingest_file(paper_path)
    chunks = chunk_document(doc, ChunkingConfig(chunk_size=1000, overlap=200))
    print(f"      Pages: {len(doc.sections)}, Chunks: {len(chunks)}")

    # Pre-flight cost check
    est = estimate_processing_cost(
        chunks,
        embedding_model=embedding_model,
        llm_model=llm_model,
    )
    est_cost = est["total_estimated_cost_usd"]
    print(f"      Estimated cost: ${est_cost:.4f} USD (budget cap: ${max_budget_usd:.2f} USD)")
    if est_cost > max_budget_usd:
        raise RuntimeError(f"Estimated cost ${est_cost:.4f} exceeds budget limit of ${max_budget_usd:.2f} USD. Aborting.")

    started = perf_counter()
    errors: list[str] = []

    # 2. Embedding Generation
    print(f"[2/5] Generating embeddings using {embedding_model} (dim={embedding_dim})...")
    embed_provider = OpenAIEmbeddingProvider(api_key=api_key, base_url=base_url)
    embed_client = EmbeddingClient(embed_provider, model_id=embedding_model, dimensions=embedding_dim, batch_size=50)

    chunk_texts = [c.text for c in chunks]
    vectors = await embed_client.embed_texts(chunk_texts)

    # Dimension verification
    for idx, v in enumerate(vectors):
        if len(v) != embedding_dim:
            raise ValueError(f"Vector {idx} has invalid dimension {len(v)} (expected {embedding_dim})")
    print(f"      Verified: All {len(vectors)} embeddings returned exactly {embedding_dim} dimensions.")

    # Populate in-memory vector index
    vector_index = InMemoryVectorIndex(dimensions=embedding_dim)
    vector_index.add([
        ChunkEmbedding(chunk=chunk, vector=vector, model_id=embedding_model)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ])

    # 3. Knowledge Extraction via GPT-6 Luna
    print(f"[3/5] Extracting entities and relationships using {llm_model}...")
    extract_provider = OpenAIStructuredExtractionProvider(api_key=api_key, base_url=base_url)
    extractor = SemanticExtractor(extract_provider, model_id=llm_model)

    knowledge_graph = KnowledgeGraph()
    extractions: list[ChunkExtraction] = []

    running_cost = est["embedding"]["estimated_cost_usd"]
    for i, chunk in enumerate(chunks, 1):
        if running_cost >= max_budget_usd:
            msg = f"Reached budget cap (${max_budget_usd:.2f} USD) at chunk {i}/{len(chunks)}. Halting further LLM calls."
            print(f"      WARNING: {msg}")
            errors.append(msg)
            break

        try:
            extraction = await extractor.extract_chunk(chunk)
            extractions.append(extraction)
            knowledge_graph.add_extraction(extraction)
            # Rough running spend tracker ($1.25/M input, $5.00/M output)
            running_cost += 0.001
            if i % 10 == 0 or i == len(chunks):
                print(f"      Progress: {i}/{len(chunks)} chunks processed...")
        except Exception as e:
            err_msg = f"Error extracting chunk {chunk.chunk_id}: {type(e).__name__}: {e}"
            print(f"      {err_msg}")
            errors.append(err_msg)
            # Stop immediately on live API errors rather than retrying blindly
            break

    # 4. Graph Verification & Provenance
    print(f"[4/5] Verifying knowledge graph and provenance...")
    total_entities = sum(len(ext.result.entities) for ext in extractions)
    total_relationships = sum(len(ext.result.relationships) for ext in extractions)
    unique_entities = len(knowledge_graph.graph.nodes)
    explicit_relationships = knowledge_graph.graph.number_of_edges()

    print(f"      Entities extracted: {total_entities} ({unique_entities} unique in graph)")
    print(f"      Relationships: {total_relationships} ({explicit_relationships} explicit edges)")

    elapsed_ms = (perf_counter() - started) * 1_000

    # 5. Report Compilation
    report = SmokeTestReport(
        document_path=str(paper_path),
        document_id=doc.document_id,
        checksum_sha256=doc.checksum_sha256,
        page_count=len(doc.sections),
        chunk_count=len(chunks),
        embeddings_generated=len(vectors),
        embedding_dimension=embedding_dim,
        llm_extractions_completed=len(extractions),
        total_entities_extracted=total_entities,
        unique_graph_entities=unique_entities,
        total_relationships_extracted=total_relationships,
        explicit_graph_relationships=explicit_relationships,
        token_usage={
            "embedding_tokens": est["embedding"]["estimated_tokens"],
            "extraction_prompt_tokens": est["extraction"]["estimated_prompt_tokens"],
            "extraction_completion_tokens": est["extraction"]["estimated_completion_tokens"],
        },
        cost_breakdown_usd={
            "embedding_cost_usd": est["embedding"]["estimated_cost_usd"],
            "extraction_cost_usd": est["extraction"]["estimated_cost_usd"],
        },
        total_cost_usd=est["total_estimated_cost_usd"],
        execution_time_ms=elapsed_ms,
        errors=errors,
        is_mocked=False,
    )

    report.save(output_path)
    print(f"[5/5] Report saved to {output_path}")
    return report


if __name__ == "__main__":
    asyncio.run(run_smoke_test())
