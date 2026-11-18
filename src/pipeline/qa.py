"""End-to-end question-answering pipeline combining document processing, hybrid retrieval, and grounded generation."""

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

from src.core.models import TextChunk
from src.embeddings.client import EmbeddingClient
from src.generation.answering import (
    EvidenceBackedAnswerGenerator,
    GroundedAnswerResult,
)
from src.generation.schemas import EvidenceCitation, GeneratedAnswer
from src.ingestion.loader import ingest_file
from src.knowledge.extraction import SemanticExtractor
from src.knowledge.schemas import ExtractedEntity, ExtractedRelationship, ExtractionResult
from src.pipeline.runner import (
    DocumentProcessingPipeline,
    PipelineArtifacts,
    estimate_processing_cost,
    load_dotenv_if_present,
)
from src.processing.chunking import chunk_document
from src.retrieval.hybrid import (
    HybridRetrievalBundle,
    HybridRetrievalConfig,
    HybridRetriever,
)


@dataclass(frozen=True)
class QAPipelineSummary:
    """Machine-readable record of a question-answering run with full provenance."""

    question: str
    answer_text: str
    insufficient_evidence: bool
    confidence_explanation: str
    supporting_chunk_ids: list[str]
    evidence_references: list[dict[str, Any]]
    retrieved_chunk_count: int
    graph_entities_retrieved: int
    graph_relationships_retrieved: int
    retrieval_config: dict[str, Any]
    model_id: str
    execution_time_ms: float
    is_mocked: bool
    document_count: int = 0
    token_usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")


class QuestionAnsweringPipeline:
    """Coordinates hybrid retrieval and evidence-backed answer generation."""

    def __init__(
        self,
        *,
        document_pipeline: DocumentProcessingPipeline,
        answer_generator: EvidenceBackedAnswerGenerator,
        retrieval_config: HybridRetrievalConfig = HybridRetrievalConfig(),
        is_mocked: bool = False,
    ) -> None:
        self.document_pipeline = document_pipeline
        self.answer_generator = answer_generator
        self.retrieval_config = retrieval_config
        self.is_mocked = is_mocked

    @classmethod
    def from_environment(
        cls,
        *,
        retrieval_config: HybridRetrievalConfig = HybridRetrievalConfig(),
        is_mocked: bool = False,
    ) -> "QuestionAnsweringPipeline":
        load_dotenv_if_present()
        doc_pipeline = DocumentProcessingPipeline.from_environment(is_mocked=is_mocked)
        generator = EvidenceBackedAnswerGenerator.from_environment()
        return cls(
            document_pipeline=doc_pipeline,
            answer_generator=generator,
            retrieval_config=retrieval_config,
            is_mocked=is_mocked,
        )

    async def ask(
        self,
        question: str,
        *,
        artifacts: PipelineArtifacts,
        config: HybridRetrievalConfig | None = None,
    ) -> tuple[GroundedAnswerResult, QAPipelineSummary]:
        started = perf_counter()
        active_config = config or self.retrieval_config
        hybrid_retriever = HybridRetriever(
            artifacts.retriever,
            artifacts.knowledge_graph,
            config=active_config,
        )
        retrieval_bundle = await hybrid_retriever.retrieve(question, config=active_config)
        answer_result = await self.answer_generator.answer_question(question, retrieval_bundle)
        elapsed_ms = (perf_counter() - started) * 1_000

        token_usage: dict[str, Any] = {}
        embed_usage = getattr(artifacts.retriever.client.provider, "last_usage", None)
        if embed_usage is not None:
            token_usage["query_embedding"] = dict(embed_usage)
        answer_usage = getattr(self.answer_generator.provider, "last_usage", None)
        if answer_usage is not None:
            token_usage["answer_generation"] = dict(answer_usage)

        summary = QAPipelineSummary(
            question=question,
            answer_text=answer_result.answer.answer_text,
            insufficient_evidence=answer_result.answer.insufficient_evidence,
            confidence_explanation=answer_result.answer.confidence_explanation,
            supporting_chunk_ids=list(answer_result.answer.supporting_chunk_ids),
            evidence_references=[ref.model_dump() for ref in answer_result.answer.evidence_references],
            retrieved_chunk_count=len(retrieval_bundle.evidence_chunks),
            graph_entities_retrieved=len(retrieval_bundle.graph_entities),
            graph_relationships_retrieved=len(retrieval_bundle.graph_relationships),
            retrieval_config=asdict(active_config),
            model_id=answer_result.model_id,
            execution_time_ms=elapsed_ms,
            is_mocked=self.is_mocked or artifacts.summary.is_mocked,
            document_count=artifacts.summary.document_count,
            token_usage=token_usage,
        )
        return answer_result, summary

    async def process_and_ask(
        self,
        paths: Sequence[str | Path],
        question: str,
        *,
        config: HybridRetrievalConfig | None = None,
    ) -> tuple[PipelineArtifacts, GroundedAnswerResult, QAPipelineSummary]:
        artifacts = await self.document_pipeline.process_documents(paths)
        answer_result, qa_summary = await self.ask(question, artifacts=artifacts, config=config)
        return artifacts, answer_result, qa_summary


def _build_mock_qa_pipeline(
    config: HybridRetrievalConfig,
    dimensions: int = 3072,
) -> QuestionAnsweringPipeline:
    """Create a deterministic zero-cost QA pipeline for local testing and CLI verification."""

    class _MockEmbeddingProvider:
        async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
            return [[1.0] + [0.0] * (dimensions - 1) for _ in texts]

    class _MockExtractionProvider:
        async def extract(
            self,
            *,
            model: str,
            messages: list[dict[str, str]],
            schema: type[ExtractionResult],
        ) -> ExtractionResult:
            user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
            source_text = user_content.replace("Source chunk:\n", "").strip()
            first_line = source_text.splitlines()[0].strip() if source_text else ""
            if not first_line:
                return ExtractionResult()
            return ExtractionResult(
                entities=[
                    ExtractedEntity(
                        name="Primary Concept",
                        entity_type="concept",
                        evidence=first_line[: min(40, len(first_line))],
                    )
                ],
                relationships=[],
            )

    class _MockAnswerProvider:
        async def generate(
            self,
            *,
            model: str,
            messages: list[dict[str, str]],
            schema: type[GeneratedAnswer],
        ) -> GeneratedAnswer:
            user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
            chunk_id = ""
            quote = ""
            for line in user_content.splitlines():
                if "chunk_id=" in line and not chunk_id:
                    parts = [p.strip() for p in line.split("|")]
                    for part in parts:
                        if "chunk_id=" in part:
                            chunk_id = part.split("chunk_id=", 1)[1].strip()
                elif line.startswith("Text: ") and not quote:
                    raw_text = line[len("Text: ") :].strip()
                    quote = raw_text[: min(60, len(raw_text))]

            if not chunk_id or not quote:
                return GeneratedAnswer(
                    answer_text="Insufficient evidence in the retrieved sources.",
                    evidence_references=[],
                    supporting_chunk_ids=[],
                    insufficient_evidence=True,
                    confidence_explanation="Mock provider found no quoteable chunk in context.",
                )

            return GeneratedAnswer(
                answer_text=f"Mocked evidence-backed answer supported by chunk {chunk_id}.",
                evidence_references=[
                    EvidenceCitation(
                        chunk_id=chunk_id,
                        quote=quote,
                        claim="Mocked factual claim derived from the first retrieved passage.",
                    )
                ],
                supporting_chunk_ids=[chunk_id],
                insufficient_evidence=False,
                confidence_explanation="Mocked response verified against retrieved chunk text.",
            )

    doc_pipeline = DocumentProcessingPipeline(
        embedding_client=EmbeddingClient(
            _MockEmbeddingProvider(),
            model_id="text-embedding-3-large",
            dimensions=dimensions,
        ),
        extractor=SemanticExtractor(_MockExtractionProvider(), model_id="gpt-6-luna"),
        is_mocked=True,
    )
    generator = EvidenceBackedAnswerGenerator(_MockAnswerProvider(), model_id="gpt-6-luna")
    return QuestionAnsweringPipeline(
        document_pipeline=doc_pipeline,
        answer_generator=generator,
        retrieval_config=config,
        is_mocked=True,
    )


def _qa_cli() -> None:
    load_dotenv_if_present()
    parser = argparse.ArgumentParser(
        description="End-to-end verifiable question-answering CLI with hybrid retrieval"
    )
    parser.add_argument(
        "--papers",
        nargs="+",
        help="Paths to PDF/TXT/MD documents to index and query",
    )
    parser.add_argument(
        "--load-artifacts",
        help="Directory path to load pre-processed artifacts from disk (e.g. data/processed)",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Research question to answer from the supplied documents",
    )
    parser.add_argument("--vector-top-k", type=int, default=5, help="Top-k vector chunks to retrieve")
    parser.add_argument("--graph-depth", type=int, default=1, help="Knowledge graph traversal depth")
    parser.add_argument("--max-chunks", type=int, default=10, help="Maximum combined evidence chunks")
    parser.add_argument("--output", help="Optional path to save JSON QA summary")
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Estimate token usage and API cost without making live API calls",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run end-to-end with mocked providers (zero API cost)",
    )
    args = parser.parse_args()

    from src.storage.artifacts import load_artifacts

    if not args.papers and not args.load_artifacts:
        parser.error("Either --papers or --load-artifacts must be provided.")

    config = HybridRetrievalConfig(
        vector_top_k=args.vector_top_k,
        graph_traversal_depth=args.graph_depth,
        max_total_chunks=args.max_chunks,
    )

    if args.load_artifacts:
        artifact_path = Path(args.load_artifacts).resolve()
        if not artifact_path.is_dir():
            print(f"Error: Artifact directory not found or not a directory: {artifact_path}", file=sys.stderr)
            sys.exit(1)

        try:
            artifacts = load_artifacts(artifact_path)
        except Exception as e:
            print(f"Error loading artifacts: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)

        if args.estimate_cost:
            # When artifacts are pre-loaded, chunk embedding and extraction are $0.00
            qa_prompt_tokens = config.max_total_chunks * 300 + 250
            qa_completion_tokens = 300
            query_embedding_tokens = max(1, len(args.question.split()) * 2)
            embedding_model = os.environ.get(
                "EMBEDDING_MODEL",
                artifacts.summary.model_configuration.get("embedding_model", "text-embedding-3-large"),
            )
            llm_model = os.environ.get(
                "LLM_MODEL",
                artifacts.summary.model_configuration.get("llm_model", "gpt-6-luna"),
            )
            embedding_cost = (query_embedding_tokens / 1_000_000) * 0.13
            qa_cost = (qa_prompt_tokens / 1_000_000) * 1.25 + (qa_completion_tokens / 1_000_000) * 5.00
            cost_summary = {
                "source": "pre_loaded_artifacts",
                "loaded_chunks": len(artifacts.chunks),
                "loaded_documents": len(artifacts.documents),
                "chunk_embedding_api_calls": 0,
                "semantic_extraction_api_calls": 0,
                "query_embedding": {
                    "model": embedding_model,
                    "estimated_tokens": query_embedding_tokens,
                    "estimated_cost_usd": round(embedding_cost, 6),
                },
                "qa_generation": {
                    "model": llm_model,
                    "estimated_prompt_tokens": qa_prompt_tokens,
                    "estimated_completion_tokens": qa_completion_tokens,
                    "estimated_cost_usd": round(qa_cost, 6),
                },
                "total_estimated_cost_usd": round(embedding_cost + qa_cost, 6),
            }
            print(json.dumps(cost_summary, indent=2))
            return

        if args.mock:
            qa_pipeline = _build_mock_qa_pipeline(config, dimensions=artifacts.vector_index.dimensions)
            # Override retriever client to mock provider so query string can be embedded safely
            artifacts.retriever.client = qa_pipeline.document_pipeline.embedding_client
        else:
            generator = EvidenceBackedAnswerGenerator.from_environment()
            live_embed_client = EmbeddingClient.from_environment()
            artifacts.retriever.client = live_embed_client
            doc_pipeline = DocumentProcessingPipeline.from_environment(is_mocked=False)
            qa_pipeline = QuestionAnsweringPipeline(
                document_pipeline=doc_pipeline,
                answer_generator=generator,
                retrieval_config=config,
                is_mocked=False,
            )

        answer_result, qa_summary = asyncio.run(
            qa_pipeline.ask(args.question, artifacts=artifacts, config=config)
        )

    else:
        paper_paths = [Path(p).resolve() for p in args.papers]
        for path in paper_paths:
            if not path.is_file():
                raise FileNotFoundError(f"Document not found: {path}")

        if args.estimate_cost:
            all_chunks: list[TextChunk] = []
            for path in paper_paths:
                doc = ingest_file(path)
                all_chunks.extend(chunk_document(doc))
            base_est = estimate_processing_cost(
                all_chunks,
                embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large"),
                llm_model=os.environ.get("LLM_MODEL", "gpt-6-luna"),
            )
            qa_prompt_tokens = config.max_total_chunks * 300 + 250
            qa_completion_tokens = 300
            qa_cost = (qa_prompt_tokens / 1_000_000) * 1.25 + (qa_completion_tokens / 1_000_000) * 5.00
            base_est["qa_generation"] = {
                "estimated_prompt_tokens": qa_prompt_tokens,
                "estimated_completion_tokens": qa_completion_tokens,
                "estimated_cost_usd": round(qa_cost, 6),
            }
            base_est["total_estimated_cost_usd"] = round(
                base_est["total_estimated_cost_usd"] + qa_cost, 6
            )
            print(json.dumps(base_est, indent=2))
            return

        if args.mock:
            qa_pipeline = _build_mock_qa_pipeline(config)
        else:
            qa_pipeline = QuestionAnsweringPipeline.from_environment(
                retrieval_config=config,
                is_mocked=False,
            )

        _, answer_result, qa_summary = asyncio.run(
            qa_pipeline.process_and_ask(paper_paths, args.question, config=config)
        )

    print(f"Question: {qa_summary.question}")
    print(f"Mode: {'MOCKED' if qa_summary.is_mocked else 'LIVE API'}")
    print(f"Insufficient Evidence: {qa_summary.insufficient_evidence}")
    print(f"Answer:\n  {qa_summary.answer_text}")
    print(f"Confidence:\n  {qa_summary.confidence_explanation}")
    print(f"Citations ({len(qa_summary.evidence_references)}):")
    for idx, ref in enumerate(qa_summary.evidence_references, 1):
        print(
            f"  [{idx}] chunk_id={ref['chunk_id']} | doc={ref['document_id']} | "
            f"page={ref['page_number']} | quote=\"{ref['quote']}\""
        )

    if args.output:
        qa_summary.save(args.output)
        print(f"Saved QA summary to: {args.output}")


if __name__ == "__main__":
    _qa_cli()
