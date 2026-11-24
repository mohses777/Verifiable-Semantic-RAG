"""Integration tests for the end-to-end document processing pipeline."""

from __future__ import annotations

import asyncio
import json
import tempfile
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


class ConcurrentExtractionTests(unittest.TestCase):
    """Concurrent extraction must match sequential output and stop on the first failure."""

    def setUp(self) -> None:
        handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        handle.write("\n\n".join(f"Paragraph {i}: " + ("GraphRAG retrieves evidence. " * 40) for i in range(12)))
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        self.FIXTURE = Path(handle.name)

    def _pipeline(self, concurrency: int, provider) -> DocumentProcessingPipeline:
        return DocumentProcessingPipeline(
            embedding_client=EmbeddingClient(MockEmbeddingProvider(), model_id="text-embedding-3-large", dimensions=4),
            extractor=SemanticExtractor(provider, model_id="gpt-6-luna"),
            is_mocked=True,
            extraction_concurrency=concurrency,
        )

    def test_invalid_concurrency_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._pipeline(0, MockExtractionProvider())

    def test_concurrent_run_matches_sequential_run(self) -> None:
        class Delayed(MockExtractionProvider):
            async def extract(self, **kwargs):
                await asyncio.sleep(0.01)
                return await super().extract(**kwargs)

        seq = asyncio.run(self._pipeline(1, Delayed()).process_document(self.FIXTURE))
        conc = asyncio.run(self._pipeline(4, Delayed()).process_document(self.FIXTURE))
        self.assertGreaterEqual(len(seq.extractions), 10)
        self.assertEqual([c.chunk.chunk_id for c in seq.extractions], [c.chunk.chunk_id for c in conc.extractions])
        self.assertEqual(set(seq.knowledge_graph.graph.nodes), set(conc.knowledge_graph.graph.nodes))
        self.assertEqual(seq.summary.entity_count, conc.summary.entity_count)

    def test_first_failure_cancels_remaining_extractions(self) -> None:
        started: list[int] = []

        class Failing:
            async def extract(self, *, model, messages, schema):
                started.append(1)
                await asyncio.sleep(0)
                raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            asyncio.run(self._pipeline(2, Failing()).process_document(self.FIXTURE))
        # The document has 10+ chunks, but only the bounded in-flight calls may have started.
        self.assertGreaterEqual(len(started), 1)
        self.assertLessEqual(len(started), 2)

    def test_concurrency_actually_overlaps_requests(self) -> None:
        state = {"now": 0, "peak": 0}

        class Tracking(MockExtractionProvider):
            async def extract(self, **kwargs):
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
                await asyncio.sleep(0.01)
                state["now"] -= 1
                return await super().extract(**kwargs)

        asyncio.run(self._pipeline(3, Tracking()).process_document(self.FIXTURE))
        self.assertEqual(state["peak"], 3)  # bounded at exactly the configured limit



class EstimatorCalibrationTests(unittest.TestCase):
    def test_extraction_estimate_is_close_to_the_measured_slice_build(self) -> None:
        from src.core.models import TextChunk

        # The measured run: 132 chunks of about 1,000 characters each cost $0.143365 (run 2) / $0.145711 (run 1).
        chunk = TextChunk(chunk_id="c", document_id="d", text="x" * 997, source_path="s", source_type="txt",
                          page_number=None, section_index=0, start_char=0, end_char=997,
                          document_start_char=0, document_end_char=997, metadata={})
        estimate = estimate_processing_cost([chunk] * 132)
        self.assertAlmostEqual(estimate["total_estimated_cost_usd"], 0.1435, delta=0.1435 * 0.25)


if __name__ == "__main__":
    unittest.main()
