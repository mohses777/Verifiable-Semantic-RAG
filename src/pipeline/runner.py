"""End-to-end pipeline connecting ingestion, chunking, embeddings, and graph construction."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from src.core.models import SourceDocument, TextChunk
from src.embeddings.client import EmbeddingClient, EmbeddingProvider
from src.embeddings.index import InMemoryVectorIndex, VectorIndex
from src.embeddings.pipeline import SemanticRetriever
from src.ingestion.loader import ingest_file
from src.knowledge.extraction import ChunkExtraction, SemanticExtractor, StructuredExtractionProvider
from src.knowledge.graph import KnowledgeGraph
from src.knowledge.schemas import ExtractionResult
from src.processing.chunking import ChunkingConfig, chunk_document


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


def estimate_processing_cost(
    chunks: Sequence[TextChunk],
    *,
    embedding_model: str = "text-embedding-3-large",
    llm_model: str = "gpt-6-luna",
    embedding_rate_per_million: float = 0.13,
    llm_prompt_rate_per_million: float = 1.25,
    llm_completion_rate_per_million: float = 5.00,
) -> dict[str, Any]:
    """Estimate token use and financial cost for processing a set of chunks.

    Rates default to standard published OpenAI API pricing guidelines.
    """
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
        await self.retriever.index_chunks(all_chunks)

        # 3. Extract knowledge from chunks and build knowledge graph
        all_extractions: list[ChunkExtraction] = []
        for chunk in all_chunks:
            extraction = await self.extractor.extract_chunk(chunk)
            all_extractions.append(extraction)
            self.knowledge_graph.add_extraction(extraction)

        elapsed_ms = (perf_counter() - started) * 1_000

        # 4. Generate machine-readable summary
        total_entity_count = sum(len(ext.result.entities) for ext in all_extractions)
        total_rel_count = sum(len(ext.result.relationships) for ext in all_extractions)
        explicit_rel_count = self.knowledge_graph.graph.number_of_edges()
        unique_entity_count = len(self.knowledge_graph.graph.nodes)

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
    parser = argparse.ArgumentParser(description="Verifiable Semantic RAG document processing pipeline")
    parser.add_argument("--papers", nargs="+", help="Paths to PDF/TXT/MD documents to process", required=True)
    parser.add_argument("--output-summary", help="Path to write the output JSON summary", default="results/pipeline_summary.json")
    parser.add_argument("--estimate-cost", action="store_true", help="Estimate tokens and API cost without running extraction")
    parser.add_argument("--mock", action="store_true", help="Run with mock providers (zero API cost, synthetic responses)")
    args = parser.parse_args()

    paper_paths = [Path(p).resolve() for p in args.papers]
    for p in paper_paths:
        if not p.is_file():
            print(f"Error: file not found: {p}")
            return

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
                # Return dummy unit vector
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


if __name__ == "__main__":
    _cli()
