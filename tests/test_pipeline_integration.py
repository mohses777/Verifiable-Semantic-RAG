"""Integration tests for the end-to-end document processing pipeline."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.embeddings.client import EmbeddingClient
from src.knowledge.extraction import SemanticExtractor
from src.knowledge.schemas import ExtractedEntity, ExtractedRelationship, ExtractionResult
from src.pipeline.runner import (
    DocumentProcessingPipeline,
    estimate_processing_cost,
)


def run_coroutine(coroutine):
    """Run an async coroutine synchronously without event loop I/O."""
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("Mocked coroutine unexpectedly yielded control")


class MockEmbeddingProvider:
    def __init__(self, dimensions: int = 4) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        self.calls.append(texts)
        # Deterministic normalized vectors
        return [[1.0] + [0.0] * (self.dimensions - 1) for _ in texts]


class MockExtractionProvider:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    async def extract(self, *, model: str, messages: list[dict[str, str]], schema: type[ExtractionResult]) -> ExtractionResult:
        self.calls.append(messages)
        user_message = next(m["content"] for m in messages if m["role"] == "user")

        entities: list[ExtractedEntity] = []
        relationships: list[ExtractedRelationship] = []

        if "Retrieval-Augmented Generation" in user_message:
            entities.extend([
                ExtractedEntity(name="RAG", entity_type="architecture", evidence="Retrieval-Augmented Generation"),
                ExtractedEntity(name="Dense Retrieval", entity_type="component", evidence="Dense retrieval"),
            ])
            relationships.append(
                ExtractedRelationship(
                    subject="RAG",
                    predicate="uses",
                    object="Dense Retrieval",
                    evidence="Retrieval-Augmented Generation combines parametric memory with retrieval.",
                    support_level="explicit",
                )
            )

        if "Knowledge graphs" in user_message:
            entities.extend([
                ExtractedEntity(name="Knowledge Graphs", entity_type="data_structure", evidence="Knowledge graphs"),
                ExtractedEntity(name="GraphRAG", entity_type="system", evidence="GraphRAG"),
            ])
            relationships.append(
                ExtractedRelationship(
                    subject="GraphRAG",
                    predicate="connects",
                    object="Knowledge Graphs",
                    evidence="GraphRAG connects passages to entities in the knowledge base.",
                    support_level="explicit",
                )
            )

        return ExtractionResult(entities=entities, relationships=relationships)


class PipelineIntegrationTests(unittest.TestCase):
    fixtures = Path(__file__).parent / "fixtures"

    def setUp(self) -> None:
        self.sample_pdf = self.fixtures / "sample_article.pdf"
        self.sample_md = self.fixtures / "notes.md"
        self.embed_provider = MockEmbeddingProvider(dimensions=4)
        self.extract_provider = MockExtractionProvider()
        self.pipeline = DocumentProcessingPipeline(
            embedding_client=EmbeddingClient(self.embed_provider, model_id="text-embedding-3-large", dimensions=4),
            extractor=SemanticExtractor(self.extract_provider, model_id="gpt-6-luna"),
            is_mocked=True,
        )

    def test_end_to_end_pipeline_processes_pdf_with_full_provenance(self) -> None:
        artifacts = run_coroutine(self.pipeline.process_document(self.sample_pdf))

        # 1. Document & chunk metrics
        self.assertEqual(artifacts.summary.document_count, 1)
        self.assertEqual(artifacts.summary.page_count, 2)
        self.assertGreaterEqual(artifacts.summary.chunk_count, 2)
        self.assertTrue(artifacts.summary.is_mocked)

        # 2. Entity and relationship counts
        self.assertGreaterEqual(artifacts.summary.entity_count, 2)
        self.assertGreaterEqual(artifacts.summary.unique_entity_count, 2)
        self.assertGreaterEqual(artifacts.summary.explicit_relationship_count, 1)

        # 3. Provenance verification: graph relationship links to exact document page
        graph_rag_evidence = artifacts.knowledge_graph.relationships_for("GraphRAG")
        self.assertGreaterEqual(len(graph_rag_evidence), 1)
        rel_evidence = graph_rag_evidence[0]
        self.assertEqual(rel_evidence.relationship.subject, "GraphRAG")
        self.assertEqual(rel_evidence.chunk.page_number, 2)
        self.assertIn("sample_article.pdf", rel_evidence.chunk.source_path)
        self.assertTrue(rel_evidence.chunk.chunk_id.startswith(f"{artifacts.documents[0].document_id}:"))

        # 4. Semantic retrieval verification: vector index holds chunks
        results = artifacts.retriever.index.search((1.0, 0.0, 0.0, 0.0), top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].chunk.document_id, artifacts.documents[0].document_id)

    def test_pipeline_summary_is_json_serializable(self) -> None:
        artifacts = run_coroutine(self.pipeline.process_document(self.sample_pdf))
        summary_json = artifacts.summary.to_json()
        data = json.loads(summary_json)

        self.assertEqual(data["document_count"], 1)
        self.assertEqual(data["page_count"], 2)
        self.assertEqual(data["model_configuration"]["llm_model"], "gpt-6-luna")
        self.assertEqual(data["model_configuration"]["embedding_model"], "text-embedding-3-large")
        self.assertEqual(len(data["documents"]), 1)
        self.assertEqual(data["documents"][0]["page_count"], 2)

    def test_pipeline_multi_document_processing(self) -> None:
        artifacts = run_coroutine(self.pipeline.process_documents([self.sample_pdf, self.sample_md]))

        self.assertEqual(artifacts.summary.document_count, 2)
        self.assertGreaterEqual(artifacts.summary.page_count, 3)  # 2 PDF pages + 1 MD section
        self.assertEqual(len(artifacts.summary.documents), 2)
        doc_ids = {d.document_id for d in artifacts.summary.documents}
        self.assertEqual(len(doc_ids), 2)

    def test_cost_estimator_computes_expected_tokens_and_usd(self) -> None:
        doc = run_coroutine(self.pipeline.process_document(self.sample_pdf))
        cost = estimate_processing_cost(
            doc.chunks,
            embedding_model="text-embedding-3-large",
            llm_model="gpt-6-luna",
        )
        self.assertEqual(cost["chunk_count"], len(doc.chunks))
        self.assertGreater(cost["total_characters"], 0)
        self.assertGreater(cost["total_estimated_cost_usd"], 0)
        self.assertIn("embedding", cost)
        self.assertIn("extraction", cost)


if __name__ == "__main__":
    unittest.main()

