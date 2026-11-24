"""End-to-end pipeline connecting ingestion, chunking, embeddings, and graph construction."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from src.core.config import build_run_configuration
from src.core.models import SourceDocument, TextChunk
from src.core.pricing import get_model_pricing
from src.embeddings.client import EmbeddingClient, EmbeddingProvider
from src.embeddings.index import InMemoryVectorIndex, VectorIndex
from src.embeddings.pipeline import SemanticRetriever
from src.ingestion.loader import ingest_file
from src.knowledge.extraction import ChunkExtraction, SemanticExtractor, StructuredExtractionProvider
from src.knowledge.graph import KnowledgeGraph
from src.knowledge.schemas import ExtractionResult
from src.processing.chunking import ChunkingConfig, chunk_document


def load_dotenv_if_present(env_path: str | Path = ".env") -> None:
    """Populate os.environ from a local .env file without overwriting existing env vars."""
    path = Path(env_path)
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


@dataclass(frozen=True)
class DocumentSummary:
    """Per-document processing metrics."""

    document_id: str
    source_path: str
    source_type: str
    checksum_sha256: str
    page_count: int
    chunk_count: int


@dataclass(frozen=True)
class PipelineSummary:
    """Machine-readable summary of an end-to-end processing execution."""

    document_count: int
    page_count: int
    chunk_count: int
    entity_count: int
    unique_entity_count: int
    relationship_count: int
    explicit_relationship_count: int
    documents: list[DocumentSummary]
    execution_time_ms: float
    is_mocked: bool
    model_configuration: dict[str, Any] = field(default_factory=dict)
    token_usage: dict[str, Any] = field(default_factory=dict)
    run_configuration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")


@dataclass(frozen=True)
class PipelineArtifacts:
    """Holds all in-memory structures produced by the pipeline."""

    documents: list[SourceDocument]
    chunks: list[TextChunk]
    extractions: list[ChunkExtraction]
    knowledge_graph: KnowledgeGraph
    vector_index: VectorIndex
    retriever: SemanticRetriever
    summary: PipelineSummary


def measured_processing_usage(
    embedding_provider: Any, extraction_provider: Any, *, embedding_model: str, llm_model: str
) -> dict[str, Any]:
    """Cost of a run computed from API-reported usage; empty for mocked providers.

    Unlike ``estimate_processing_cost`` this only reports tokens the API returned.
    """
    usage: dict[str, Any] = {}
    total = 0.0
    embed = getattr(embedding_provider, "total_usage", None)
    if embed is not None:
        cost = get_model_pricing(embedding_model).input_cost(int(embed.get("total_tokens", 0)))
        usage["embedding"] = {**embed, "measured": True, "cost_usd": round(cost, 6)}
        total += cost
    extract = getattr(extraction_provider, "total_usage", None)
    if extract is not None:
        pricing = get_model_pricing(llm_model)
        cost = pricing.input_cost(int(extract.get("prompt_tokens", 0))) + pricing.output_cost(
            int(extract.get("completion_tokens", 0))
        )
        usage["extraction"] = {**extract, "measured": True, "cost_usd": round(cost, 6)}
        total += cost
    if usage:
        usage["total_measured_cost_usd"] = round(total, 6)
    return usage


def estimate_processing_cost(
    chunks: Sequence[TextChunk],
    *,
    embedding_model: str = "text-embedding-3-large",
    llm_model: str = "gpt-6-luna",
    embedding_rate_per_million: float | None = None,
    llm_prompt_rate_per_million: float | None = None,
    llm_completion_rate_per_million: float | None = None,
) -> dict[str, Any]:
    """Estimate token use and financial cost for processing a set of chunks.

    Rates default to the version-controlled values in ``config/pricing.json`` for
    the given models; an explicit rate argument overrides the config. A model with
    no configured pricing raises ``KeyError`` rather than guessing.
    """
    if embedding_rate_per_million is None:
        embedding_rate_per_million = get_model_pricing(embedding_model).input
    if llm_prompt_rate_per_million is None or llm_completion_rate_per_million is None:
        llm_pricing = get_model_pricing(llm_model)
        if llm_prompt_rate_per_million is None:
            llm_prompt_rate_per_million = llm_pricing.input
        if llm_completion_rate_per_million is None:
            llm_completion_rate_per_million = llm_pricing.output
    total_characters = sum(len(c.text) for c in chunks)
    estimated_input_tokens = total_characters // 4

    embedding_cost = (estimated_input_tokens / 1_000_000) * embedding_rate_per_million

    prompt_tokens_per_call = 80
    total_extraction_prompt_tokens = estimated_input_tokens + (len(chunks) * prompt_tokens_per_call)
    estimated_output_tokens = len(chunks) * 100

    llm_prompt_cost = (total_extraction_prompt_tokens / 1_000_000) * llm_prompt_rate_per_million
    llm_completion_cost = (estimated_output_tokens / 1_000_000) * llm_completion_rate_per_million
    total_cost = embedding_cost + llm_prompt_cost + llm_completion_cost

    return {
        "chunk_count": len(chunks),
        "total_characters": total_characters,
        "estimated_text_tokens": estimated_input_tokens,
        "embedding": {
            "model": embedding_model,
            "estimated_tokens": estimated_input_tokens,
            "estimated_cost_usd": round(embedding_cost, 6),
        },
        "extraction": {
            "model": llm_model,
            "estimated_prompt_tokens": total_extraction_prompt_tokens,
            "estimated_completion_tokens": estimated_output_tokens,
            "estimated_prompt_cost_usd": round(llm_prompt_cost, 6),
            "estimated_completion_cost_usd": round(llm_completion_cost, 6),
            "estimated_cost_usd": round(llm_prompt_cost + llm_completion_cost, 6),
        },
        "total_estimated_cost_usd": round(total_cost, 6),
    }


class DocumentProcessingPipeline:
    """Executes ingestion, chunking, embedding, extraction, and graph creation."""

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        extractor: SemanticExtractor,
        chunking_config: ChunkingConfig | None = None,
        vector_index: VectorIndex | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        is_mocked: bool = False,
    ) -> None:
        self.embedding_client = embedding_client
        self.extractor = extractor
        self.chunking_config = chunking_config or ChunkingConfig()
        self.vector_index = vector_index or InMemoryVectorIndex(dimensions=embedding_client.dimensions)
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.retriever = SemanticRetriever(self.embedding_client, self.vector_index)
        self.is_mocked = is_mocked

    @classmethod
    def from_environment(
        cls,
        *,
        chunking_config: ChunkingConfig | None = None,
        vector_index: VectorIndex | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        extraction_provider: StructuredExtractionProvider | None = None,
        is_mocked: bool = False,
    ) -> "DocumentProcessingPipeline":
        load_dotenv_if_present()
        embedding_client = EmbeddingClient.from_environment(embedding_provider)
        extractor = SemanticExtractor.from_environment(extraction_provider)
        return cls(
            embedding_client=embedding_client,
            extractor=extractor,
            chunking_config=chunking_config,
            vector_index=vector_index,
            knowledge_graph=knowledge_graph,
            is_mocked=is_mocked,
        )

    async def process_document(self, path: str | Path) -> PipelineArtifacts:
        return await self.process_documents([path])

    async def process_documents(self, paths: Sequence[str | Path]) -> PipelineArtifacts:
        if not paths:
            raise ValueError("At least one document path is required")

        started = perf_counter()
        documents: list[SourceDocument] = []
        all_chunks: list[TextChunk] = []
        doc_summaries: list[DocumentSummary] = []

        # 1. Ingest and chunk each document
        for path in paths:
            doc = ingest_file(path)
            documents.append(doc)
            chunks = chunk_document(doc, self.chunking_config)
            all_chunks.extend(chunks)
            doc_summaries.append(
                DocumentSummary(
                    document_id=doc.document_id,
                    source_path=doc.source_path,
                    source_type=doc.source_type,
                    checksum_sha256=doc.checksum_sha256,
                    page_count=len(doc.sections),
                    chunk_count=len(chunks),
                )
            )

        # 2. Embed and index all chunks
        if not self.is_mocked:
            print(f"  [1/2] Generating embeddings for {len(all_chunks)} chunks using {self.embedding_client.model_id}...")
        await self.retriever.index_chunks(all_chunks)

        # 3. Extract knowledge from chunks and build knowledge graph
        if not self.is_mocked:
            print(f"  [2/2] Extracting semantic entities and relationships using {self.extractor.model_id}...")
        all_extractions: list[ChunkExtraction] = []
        for idx, chunk in enumerate(all_chunks, 1):
            extraction = await self.extractor.extract_chunk(chunk)
            all_extractions.append(extraction)
            self.knowledge_graph.add_extraction(extraction)
            if not self.is_mocked and (idx % 10 == 0 or idx == len(all_chunks)):
                print(f"        Progress: {idx}/{len(all_chunks)} chunks processed...")

        elapsed_ms = (perf_counter() - started) * 1_000

        # 4. Generate machine-readable summary
        total_entity_count = sum(len(ext.result.entities) for ext in all_extractions)
        total_rel_count = sum(len(ext.result.relationships) for ext in all_extractions)
        explicit_rel_count = self.knowledge_graph.graph.number_of_edges()
        unique_entity_count = len(self.knowledge_graph.graph.nodes)

        token_usage = measured_processing_usage(
            self.embedding_client.provider,
            self.extractor.provider,
            embedding_model=self.embedding_client.model_id,
            llm_model=self.extractor.model_id,
        )
        run_configuration = build_run_configuration(
            llm_model=self.extractor.model_id,
            embedding_model=self.embedding_client.model_id,
            embedding_dimensions=self.embedding_client.dimensions,
            is_mocked=self.is_mocked,
            chunking=asdict(self.chunking_config),
        )

        summary = PipelineSummary(
            document_count=len(documents),
            page_count=sum(len(d.sections) for d in documents),
            chunk_count=len(all_chunks),
            entity_count=total_entity_count,
            unique_entity_count=unique_entity_count,
            relationship_count=total_rel_count,
            explicit_relationship_count=explicit_rel_count,
            documents=doc_summaries,
            execution_time_ms=elapsed_ms,
            is_mocked=self.is_mocked,
            model_configuration={
                "llm_model": self.extractor.model_id,
                "embedding_model": self.embedding_client.model_id,
                "embedding_dimensions": self.embedding_client.dimensions,
            },
            token_usage=token_usage,
            run_configuration=run_configuration,
        )

        return PipelineArtifacts(
            documents=documents,
            chunks=all_chunks,
            extractions=all_extractions,
            knowledge_graph=self.knowledge_graph,
            vector_index=self.vector_index,
            retriever=self.retriever,
            summary=summary,
        )


def _cli() -> None:
    load_dotenv_if_present()
    parser = argparse.ArgumentParser(description="Verifiable Semantic RAG document processing pipeline")
    parser.add_argument("--papers", nargs="+", help="Paths to PDF/TXT/MD documents to process")
    parser.add_argument("--save-artifacts", help="Directory path to save processed artifacts (JSON + NumPy vectors)")
    parser.add_argument("--load-artifacts", help="Directory path to load pre-processed artifacts from disk")
    parser.add_argument("--output-summary", help="Path to write the output JSON summary", default="results/pipeline_summary.json")
    parser.add_argument("--estimate-cost", action="store_true", help="Estimate tokens and API cost without running extraction")
    parser.add_argument("--mock", action="store_true", help="Run with mock providers (zero API cost, synthetic responses)")
    args = parser.parse_args()

    from src.storage.artifacts import load_artifacts, save_artifacts

    if args.load_artifacts:
        artifact_path = Path(args.load_artifacts).resolve()
        if not artifact_path.is_dir():
            print(f"Error: artifact directory not found: {artifact_path}")
            sys.exit(1)
        print(f"Loading artifacts from: {artifact_path}...")
        artifacts = load_artifacts(artifact_path)
        print(f"Loaded successfully: {artifacts.summary.document_count} docs, {artifacts.summary.page_count} pages, {artifacts.summary.chunk_count} chunks")
        print(f"Graph: {artifacts.summary.unique_entity_count} unique entities, {artifacts.summary.explicit_relationship_count} explicit edges")
        if args.output_summary:
            artifacts.summary.save(args.output_summary)
            print(f"Summary saved to: {args.output_summary}")
        return

    if not args.papers:
        parser.error("Either --papers or --load-artifacts must be provided.")

    paper_paths = [Path(p).resolve() for p in args.papers]
    for p in paper_paths:
        if not p.is_file():
            print(f"Error: file not found: {p}")
            sys.exit(1)

    if args.estimate_cost:
        print(f"Ingesting {len(paper_paths)} documents for cost estimation...")
        all_chunks: list[TextChunk] = []
        for p in paper_paths:
            doc = ingest_file(p)
            chunks = chunk_document(doc)
            all_chunks.extend(chunks)
            print(f"  - {p.name}: {len(doc.sections)} pages, {len(chunks)} chunks")

        est = estimate_processing_cost(
            all_chunks,
            embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large"),
            llm_model=os.environ.get("LLM_MODEL", "gpt-6-luna"),
        )
        print("\n=== Processing Cost Estimate ===")
        print(json.dumps(est, indent=2))
        return

    if args.mock:
        class _MockEmbeddingProvider:
            async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
                return [[1.0] + [0.0] * 3071 for _ in texts]

        class _MockExtractionProvider:
            async def extract(self, *, model: str, messages: list[dict[str, str]], schema: type[ExtractionResult]) -> ExtractionResult:
                return ExtractionResult(entities=[], relationships=[])

        pipeline = DocumentProcessingPipeline(
            embedding_client=EmbeddingClient(_MockEmbeddingProvider(), model_id="text-embedding-3-large", dimensions=3072),
            extractor=SemanticExtractor(_MockExtractionProvider(), model_id="gpt-6-luna"),
            is_mocked=True,
        )
    else:
        pipeline = DocumentProcessingPipeline.from_environment(is_mocked=False)

    print(f"Processing {len(paper_paths)} documents (mock={args.mock})...")
    artifacts = asyncio.run(pipeline.process_documents(paper_paths))
    print(f"\nCompleted in {artifacts.summary.execution_time_ms:.1f}ms")
    print(f"Summary: {artifacts.summary.document_count} docs, {artifacts.summary.page_count} pages, {artifacts.summary.chunk_count} chunks")
    print(f"Graph: {artifacts.summary.unique_entity_count} unique entities, {artifacts.summary.explicit_relationship_count} explicit edges")

    if args.output_summary:
        artifacts.summary.save(args.output_summary)
        print(f"Summary saved to: {args.output_summary}")

    if args.save_artifacts:
        saved_path = save_artifacts(artifacts, args.save_artifacts)
        print(f"Artifacts saved to: {saved_path}")


if __name__ == "__main__":
    _cli()
