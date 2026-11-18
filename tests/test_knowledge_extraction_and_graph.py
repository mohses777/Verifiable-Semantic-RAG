from __future__ import annotations

import unittest

from src.core.models import TextChunk
from src.knowledge.extraction import SemanticExtractor
from src.knowledge.graph import KnowledgeGraph
from src.knowledge.schemas import ExtractedEntity, ExtractedRelationship, ExtractionResult


def run(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("Mocked coroutine unexpectedly yielded control")


def source_chunk(chunk_id: str, text: str, page: int) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id, document_id="paper-1", text=text, source_path="paper.pdf", source_type="pdf",
        page_number=page, section_index=page - 1, start_char=0, end_char=len(text),
        document_start_char=0, document_end_char=len(text), metadata={"filename": "paper.pdf"},
    )


class FakeProvider:
    def __init__(self, result: ExtractionResult) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def extract(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class KnowledgeExtractionAndGraphTests(unittest.TestCase):
    def test_extraction_parses_structured_response_and_attaches_chunk_provenance(self) -> None:
        text = "A knowledge graph supports RAG retrieval."
        provider = FakeProvider(ExtractionResult(
            entities=[
                ExtractedEntity(name="Knowledge Graph", entity_type="method", evidence="knowledge graph"),
                ExtractedEntity(name="RAG", entity_type="method", evidence="RAG retrieval"),
            ],
            relationships=[ExtractedRelationship(
                subject="Knowledge Graph", predicate="supports", object="RAG",
                evidence="knowledge graph supports RAG retrieval", support_level="explicit",
            )],
        ))
        extraction = run(SemanticExtractor(provider, model_id="gpt-6-luna").extract_chunk(source_chunk("chunk-1", text, 5)))
        self.assertEqual(provider.calls[0]["model"], "gpt-6-luna")
        self.assertEqual(extraction.result.entities[0].source_chunk_id, "chunk-1")
        self.assertEqual(extraction.result.relationships[0].source_chunk_id, "chunk-1")
        self.assertEqual(extraction.result.relationships[0].support_level, "explicit")

    def test_graph_merges_duplicate_entities_and_retains_all_provenance(self) -> None:
        provider = FakeProvider(ExtractionResult(entities=[ExtractedEntity(name="RAG", entity_type="method", evidence="RAG")]))
        extractor = SemanticExtractor(provider, model_id="gpt-6-luna")
        graph = KnowledgeGraph()
        graph.add_extraction(run(extractor.extract_chunk(source_chunk("chunk-a", "RAG is useful.", 1))))
        graph.add_extraction(run(extractor.extract_chunk(source_chunk("chunk-b", "RAG is grounded.", 2))))
        entities = graph.lookup_entity("rag")
        self.assertEqual(len(entities), 2)
        self.assertEqual({item.source_chunk_id for item in entities}, {"chunk-a", "chunk-b"})
        entity_evidence = graph.entity_evidence_for("RAG")
        self.assertEqual({item.chunk.page_number for item in entity_evidence}, {1, 2})
        self.assertEqual({item.chunk.document_id for item in entity_evidence}, {"paper-1"})

    def test_graph_relationship_lookup_returns_document_and_page_evidence(self) -> None:
        text = "GraphRAG uses a knowledge graph."
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="GraphRAG", entity_type="system", evidence="GraphRAG"),
                ExtractedEntity(name="Knowledge Graph", entity_type="concept", evidence="knowledge graph"),
            ],
            relationships=[ExtractedRelationship(
                subject="GraphRAG", predicate="uses", object="Knowledge Graph", evidence="GraphRAG uses a knowledge graph"
            )],
        )
        extraction = run(SemanticExtractor(FakeProvider(result), model_id="gpt-6-luna").extract_chunk(source_chunk("chunk-graph", text, 8)))
        graph = KnowledgeGraph()
        graph.add_extraction(extraction)
        evidence = graph.relationships_for("GraphRAG")
        self.assertEqual(evidence[0].relationship.source_chunk_id, "chunk-graph")
        self.assertEqual(evidence[0].chunk.document_id, "paper-1")
        self.assertEqual(evidence[0].chunk.page_number, 8)

    def test_unsupported_or_uncertain_relationships_do_not_become_edges(self) -> None:
        text = "RAG retrieves documents."
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="RAG", entity_type="method", evidence="RAG"),
                ExtractedEntity(name="Documents", entity_type="object", evidence="documents"),
            ],
            relationships=[ExtractedRelationship(
                subject="RAG", predicate="causes", object="Documents", evidence="RAG causes documents", support_level="explicit"
            )],
        )
        extraction = run(SemanticExtractor(FakeProvider(result), model_id="gpt-6-luna").extract_chunk(source_chunk("chunk-unsupported", text, 3)))
        self.assertEqual(extraction.result.relationships[0].support_level, "uncertain")
        graph = KnowledgeGraph()
        graph.add_extraction(extraction)
        self.assertEqual(graph.relationships_for("RAG"), [])


if __name__ == "__main__":
    unittest.main()
