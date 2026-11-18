"""Unit and integration tests for hybrid retrieval and evidence-backed answer generation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.core.models import TextChunk
from src.embeddings.client import EmbeddingClient
from src.embeddings.index import InMemoryVectorIndex
from src.embeddings.pipeline import SemanticRetriever
from src.generation.answering import EvidenceBackedAnswerGenerator
from src.generation.schemas import EvidenceCitation, GeneratedAnswer
from src.knowledge.extraction import ChunkExtraction, SemanticExtractor
from src.knowledge.graph import KnowledgeGraph
from src.knowledge.schemas import ExtractedEntity, ExtractedRelationship, ExtractionResult
from src.pipeline.qa import QuestionAnsweringPipeline, _build_mock_qa_pipeline
from src.retrieval.hybrid import HybridRetrievalConfig, HybridRetriever


def run_sync(coroutine):
    """Execute a mocked coroutine synchronously without event-loop I/O."""
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("Mocked coroutine unexpectedly yielded control")


def make_chunk(chunk_id: str, text: str, *, doc_id: str = "doc-1", page: int = 1) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        text=text,
        source_path=f"{doc_id}.pdf",
        source_type="pdf",
        page_number=page,
        section_index=page - 1,
        start_char=0,
        end_char=len(text),
        document_start_char=0,
        document_end_char=len(text),
        metadata={"filename": f"{doc_id}.pdf"},
    )


class FakeEmbeddingProvider:
    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self.vectors_by_text = vectors_by_text

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        return [self.vectors_by_text[text] for text in texts]


class FakeAnswerProvider:
    def __init__(self, response: GeneratedAnswer) -> None:
        self.response = response
        self.calls: list[dict] = []

    async def generate(self, **kwargs) -> GeneratedAnswer:
        self.calls.append(kwargs)
        return self.response


class HybridRetrievalAndAnsweringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunk_a = make_chunk(
            "doc-1:s0:c0",
            "GraphRAG builds a structured knowledge graph from source documents.",
            page=2,
        )
        self.chunk_b = make_chunk(
            "doc-1:s3:c0",
            "GraphRAG uses Leiden community detection to summarize global themes.",
            page=4,
        )
        self.chunk_c = make_chunk(
            "doc-1:s6:c0",
            "Leiden community detection partitions hierarchical entity clusters.",
            page=7,
        )

        self.query = "How does GraphRAG summarize global themes?"
        self.vectors = {
            self.chunk_a.text: [1.0, 0.0, 0.0],
            self.chunk_b.text: [0.2, 0.9, 0.0],
            self.chunk_c.text: [0.0, 0.1, 1.0],
            self.query: [0.95, 0.05, 0.0],
        }
        client = EmbeddingClient(
            FakeEmbeddingProvider(self.vectors),
            model_id="text-embedding-3-large",
            dimensions=3,
        )
        index = InMemoryVectorIndex(dimensions=3)
        self.semantic_retriever = SemanticRetriever(client, index)
        run_sync(self.semantic_retriever.index_chunks([self.chunk_a, self.chunk_b, self.chunk_c]))

        self.graph = KnowledgeGraph()
        self.graph.add_extraction(
            ChunkExtraction(
                chunk=self.chunk_a,
                result=ExtractionResult(
                    entities=[
                        ExtractedEntity(
                            name="GraphRAG",
                            entity_type="system",
                            evidence="GraphRAG",
                            source_chunk_id=self.chunk_a.chunk_id,
                        ),
                        ExtractedEntity(
                            name="Knowledge Graph",
                            entity_type="structure",
                            evidence="knowledge graph",
                            source_chunk_id=self.chunk_a.chunk_id,
                        ),
                    ],
                    relationships=[
                        ExtractedRelationship(
                            subject="GraphRAG",
                            predicate="builds",
                            object="Knowledge Graph",
                            evidence="GraphRAG builds a structured knowledge graph",
                            support_level="explicit",
                            source_chunk_id=self.chunk_a.chunk_id,
                        )
                    ],
                ),
                model_id="gpt-6-luna",
                execution_ms=1.0,
            )
        )
        self.graph.add_extraction(
            ChunkExtraction(
                chunk=self.chunk_b,
                result=ExtractionResult(
                    entities=[
                        ExtractedEntity(
                            name="GraphRAG",
                            entity_type="system",
                            evidence="GraphRAG",
                            source_chunk_id=self.chunk_b.chunk_id,
                        ),
                        ExtractedEntity(
                            name="Leiden Community Detection",
                            entity_type="algorithm",
                            evidence="Leiden community detection",
                            source_chunk_id=self.chunk_b.chunk_id,
                        ),
                    ],
                    relationships=[
                        ExtractedRelationship(
                            subject="GraphRAG",
                            predicate="uses",
                            object="Leiden Community Detection",
                            evidence="GraphRAG uses Leiden community detection",
                            support_level="explicit",
                            source_chunk_id=self.chunk_b.chunk_id,
                        )
                    ],
                ),
                model_id="gpt-6-luna",
                execution_ms=1.0,
            )
        )
        self.graph.add_extraction(
            ChunkExtraction(
                chunk=self.chunk_c,
                result=ExtractionResult(
                    entities=[
                        ExtractedEntity(
                            name="Leiden Community Detection",
                            entity_type="algorithm",
                            evidence="Leiden community detection",
                            source_chunk_id=self.chunk_c.chunk_id,
                        ),
                        ExtractedEntity(
                            name="Entity Clusters",
                            entity_type="concept",
                            evidence="entity clusters",
                            source_chunk_id=self.chunk_c.chunk_id,
                        ),
                    ],
                    relationships=[
                        ExtractedRelationship(
                            subject="Leiden Community Detection",
                            predicate="partitions",
                            object="Entity Clusters",
                            evidence="Leiden community detection partitions hierarchical entity clusters",
                            support_level="explicit",
                            source_chunk_id=self.chunk_c.chunk_id,
                        )
                    ],
                ),
                model_id="gpt-6-luna",
                execution_ms=1.0,
            )
        )

    def test_hybrid_retrieval_merges_vector_and_graph_evidence_and_preserves_provenance(self) -> None:
        config = HybridRetrievalConfig(vector_top_k=1, graph_traversal_depth=1, max_total_chunks=5)
        hybrid = HybridRetriever(self.semantic_retriever, self.graph, config=config)
        bundle = run_sync(hybrid.retrieve(self.query))

        retrieved_ids = [item.chunk.chunk_id for item in bundle.evidence_chunks]
        # Chunk A is top-1 vector AND in graph; Chunk B is brought in via 1-hop graph expansion on GraphRAG
        self.assertIn(self.chunk_a.chunk_id, retrieved_ids)
        self.assertIn(self.chunk_b.chunk_id, retrieved_ids)
        self.assertEqual(len(retrieved_ids), len(set(retrieved_ids)))

        by_id = {item.chunk.chunk_id: item for item in bundle.evidence_chunks}
        self.assertEqual(by_id[self.chunk_a.chunk_id].retrieval_sources, ("vector", "graph"))
        self.assertEqual(by_id[self.chunk_b.chunk_id].retrieval_sources, ("graph",))
        self.assertEqual(by_id[self.chunk_b.chunk_id].chunk.page_number, 4)
        self.assertEqual(by_id[self.chunk_b.chunk_id].chunk.document_id, "doc-1")

    def test_hybrid_retrieval_depth_controls_multi_hop_expansion(self) -> None:
        hybrid = HybridRetriever(self.semantic_retriever, self.graph)

        depth_1 = run_sync(
            hybrid.retrieve(
                self.query,
                config=HybridRetrievalConfig(vector_top_k=1, graph_traversal_depth=1, max_total_chunks=10),
            )
        )
        rel_predicates_d1 = {r.relationship.predicate for r in depth_1.graph_relationships}
        self.assertIn("uses", rel_predicates_d1)
        self.assertNotIn("partitions", rel_predicates_d1)

        depth_2 = run_sync(
            hybrid.retrieve(
                self.query,
                config=HybridRetrievalConfig(vector_top_k=1, graph_traversal_depth=2, max_total_chunks=10),
            )
        )
        rel_predicates_d2 = {r.relationship.predicate for r in depth_2.graph_relationships}
        self.assertIn("uses", rel_predicates_d2)
        self.assertIn("partitions", rel_predicates_d2)

        with self.assertRaises(ValueError):
            HybridRetrievalConfig(vector_top_k=0)
        with self.assertRaises(ValueError):
            HybridRetrievalConfig(graph_traversal_depth=-1)

    def test_answer_generation_grounds_citations_with_authoritative_provenance(self) -> None:
        hybrid = HybridRetriever(self.semantic_retriever, self.graph)
        bundle = run_sync(hybrid.retrieve(self.query))

        provider = FakeAnswerProvider(
            GeneratedAnswer(
                answer_text="GraphRAG summarizes global themes using Leiden community detection.",
                evidence_references=[
                    EvidenceCitation(
                        chunk_id=self.chunk_b.chunk_id,
                        document_id="spoofed-doc",
                        source_path="spoofed.pdf",
                        page_number=999,
                        quote="GraphRAG uses Leiden community detection to summarize global themes.",
                        claim="GraphRAG uses Leiden community detection for global theme summarization.",
                    )
                ],
                supporting_chunk_ids=["spoofed-id"],
                insufficient_evidence=False,
                confidence_explanation="Directly stated on page 4.",
            )
        )
        generator = EvidenceBackedAnswerGenerator(provider, model_id="gpt-6-luna")
        result = run_sync(generator.answer_question(self.query, bundle))

        self.assertFalse(result.answer.insufficient_evidence)
        self.assertEqual(result.answer.supporting_chunk_ids, [self.chunk_b.chunk_id])
        self.assertEqual(len(result.answer.evidence_references), 1)
        citation = result.answer.evidence_references[0]
        # Provenance overwritten from authoritative TextChunk
        self.assertEqual(citation.document_id, "doc-1")
        self.assertEqual(citation.source_path, "doc-1.pdf")
        self.assertEqual(citation.page_number, 4)

    def test_citation_validation_strips_invented_chunks_and_fabricated_quotes(self) -> None:
        hybrid = HybridRetriever(self.semantic_retriever, self.graph)
        bundle = run_sync(hybrid.retrieve(self.query))

        provider = FakeAnswerProvider(
            GeneratedAnswer(
                answer_text="GraphRAG uses Leiden community detection.",
                evidence_references=[
                    EvidenceCitation(
                        chunk_id="non-existent-chunk-id",
                        quote="GraphRAG uses Leiden community detection",
                        claim="Invented chunk reference.",
                    ),
                    EvidenceCitation(
                        chunk_id=self.chunk_b.chunk_id,
                        quote="This quote does not exist anywhere in the chunk text.",
                        claim="Fabricated quotation.",
                    ),
                    EvidenceCitation(
                        chunk_id=self.chunk_b.chunk_id,
                        quote="GraphRAG uses Leiden community detection",
                        claim="Verified quotation.",
                    ),
                ],
                supporting_chunk_ids=["non-existent-chunk-id", self.chunk_b.chunk_id],
                insufficient_evidence=False,
                confidence_explanation="Verified against chunk B.",
            )
        )
        generator = EvidenceBackedAnswerGenerator(provider, model_id="gpt-6-luna")
        result = run_sync(generator.answer_question(self.query, bundle))

        self.assertFalse(result.answer.insufficient_evidence)
        self.assertEqual(len(result.answer.evidence_references), 1)
        self.assertEqual(result.answer.evidence_references[0].claim, "Verified quotation.")
        self.assertEqual(result.answer.supporting_chunk_ids, [self.chunk_b.chunk_id])

    def test_unverified_answer_fails_closed_to_insufficient_evidence(self) -> None:
        hybrid = HybridRetriever(self.semantic_retriever, self.graph)
        bundle = run_sync(hybrid.retrieve(self.query))

        provider = FakeAnswerProvider(
            GeneratedAnswer(
                answer_text="Unsupported hallucinated answer.",
                evidence_references=[
                    EvidenceCitation(
                        chunk_id=self.chunk_a.chunk_id,
                        quote="Completely fabricated quote not in chunk A.",
                        claim="Hallucinated claim.",
                    )
                ],
                supporting_chunk_ids=[self.chunk_a.chunk_id],
                insufficient_evidence=False,
                confidence_explanation="High confidence.",
            )
        )
        generator = EvidenceBackedAnswerGenerator(provider, model_id="gpt-6-luna")
        result = run_sync(generator.answer_question(self.query, bundle))

        self.assertTrue(result.answer.insufficient_evidence)
        self.assertEqual(result.answer.evidence_references, [])
        self.assertEqual(result.answer.supporting_chunk_ids, [])

    def test_end_to_end_mock_qa_pipeline_on_sample_pdf(self) -> None:
        sample_pdf = Path(__file__).parent / "fixtures" / "sample_article.pdf"
        config = HybridRetrievalConfig(vector_top_k=2, graph_traversal_depth=1, max_total_chunks=4)
        qa_pipeline = _build_mock_qa_pipeline(config)
        artifacts, answer_result, summary = run_sync(
            qa_pipeline.process_and_ask([sample_pdf], "What does GraphRAG connect?", config=config)
        )

        self.assertTrue(summary.is_mocked)
        self.assertFalse(summary.insufficient_evidence)
        self.assertGreaterEqual(len(summary.evidence_references), 1)
        parsed_json = json.loads(summary.to_json())
        self.assertTrue(parsed_json["is_mocked"])
        self.assertEqual(parsed_json["document_count"], 1)


if __name__ == "__main__":
    unittest.main()
