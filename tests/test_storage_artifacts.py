"""Tests for lightweight research artifact persistence (JSON + NumPy)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.core.models import ChunkEmbedding, DocumentSection, SourceDocument, TextChunk
from src.embeddings.client import EmbeddingClient
from src.embeddings.index import InMemoryVectorIndex
from src.embeddings.pipeline import SemanticRetriever
from src.generation.answering import EvidenceBackedAnswerGenerator
from src.generation.schemas import EvidenceCitation, GeneratedAnswer
from src.knowledge.extraction import ChunkExtraction
from src.knowledge.graph import KnowledgeGraph
from src.knowledge.schemas import ExtractedEntity, ExtractedRelationship, ExtractionResult
from src.pipeline.qa import QuestionAnsweringPipeline
from src.pipeline.runner import DocumentSummary, PipelineArtifacts, PipelineSummary
from src.retrieval.hybrid import HybridRetrievalConfig
from src.storage.artifacts import (
    OfflineEmbeddingProvider,
    load_artifacts,
    save_artifacts,
)


def run_sync(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("Mocked coroutine unexpectedly yielded control")


def make_test_artifacts(dim: int = 3) -> PipelineArtifacts:
    doc = SourceDocument(
        document_id="doc-test-1",
        source_path="test_doc.pdf",
        source_type="pdf",
        checksum_sha256="abc123sha256",
        sections=(
            DocumentSection(text="First page section text.", page_number=1, document_offset=0),
            DocumentSection(text="Second page section text.", page_number=2, document_offset=25),
        ),
        metadata={"filename": "test_doc.pdf"},
    )
    chunk_1 = TextChunk(
        chunk_id="doc-test-1:s0:c0",
        document_id=doc.document_id,
        text="First page chunk text on GraphRAG.",
        source_path=doc.source_path,
        source_type=doc.source_type,
        page_number=1,
        section_index=0,
        start_char=0,
        end_char=34,
        document_start_char=0,
        document_end_char=34,
        metadata=dict(doc.metadata),
    )
    chunk_2 = TextChunk(
        chunk_id="doc-test-1:s1:c0",
        document_id=doc.document_id,
        text="Second page chunk text on Knowledge Graphs.",
        source_path=doc.source_path,
        source_type=doc.source_type,
        page_number=2,
        section_index=1,
        start_char=0,
        end_char=43,
        document_start_char=25,
        document_end_char=68,
        metadata=dict(doc.metadata),
    )

    vec_1 = (1.0, 0.0, 0.0) if dim == 3 else tuple([1.0] + [0.0] * (dim - 1))
    vec_2 = (0.0, 1.0, 0.0) if dim == 3 else tuple([0.0, 1.0] + [0.0] * (dim - 2))

    index = InMemoryVectorIndex(dimensions=dim)
    index.add([
        ChunkEmbedding(chunk=chunk_1, vector=vec_1, model_id="text-embedding-3-large"),
        ChunkEmbedding(chunk=chunk_2, vector=vec_2, model_id="text-embedding-3-large"),
    ])

    kg = KnowledgeGraph()
    ext_1 = ChunkExtraction(
        chunk=chunk_1,
        result=ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="GraphRAG",
                    entity_type="system",
                    evidence="GraphRAG",
                    source_chunk_id=chunk_1.chunk_id,
                )
            ],
            relationships=[],
        ),
        model_id="gpt-6-luna",
        execution_ms=10.5,
    )
    ext_2 = ChunkExtraction(
        chunk=chunk_2,
        result=ExtractionResult(
            entities=[
                ExtractedEntity(
                    name="Knowledge Graph",
                    entity_type="structure",
                    evidence="Knowledge Graphs",
                    source_chunk_id=chunk_2.chunk_id,
                )
            ],
            relationships=[
                ExtractedRelationship(
                    subject="GraphRAG",
                    predicate="uses",
                    object="Knowledge Graph",
                    evidence="Second page chunk text on Knowledge Graphs.",
                    support_level="explicit",
                    source_chunk_id=chunk_2.chunk_id,
                )
            ],
        ),
        model_id="gpt-6-luna",
        execution_ms=12.0,
    )
    kg.add_extraction(ext_1)
    kg.add_extraction(ext_2)

    retriever = SemanticRetriever(
        EmbeddingClient(OfflineEmbeddingProvider(), model_id="text-embedding-3-large", dimensions=dim),
        index,
    )

    summary = PipelineSummary(
        document_count=1,
        page_count=2,
        chunk_count=2,
        entity_count=2,
        unique_entity_count=2,
        relationship_count=1,
        explicit_relationship_count=1,
        documents=[
            DocumentSummary(
                document_id=doc.document_id,
                source_path=doc.source_path,
                source_type=doc.source_type,
                checksum_sha256=doc.checksum_sha256,
                page_count=2,
                chunk_count=2,
            )
        ],
        execution_time_ms=22.5,
        is_mocked=True,
        model_configuration={
            "llm_model": "gpt-6-luna",
            "embedding_model": "text-embedding-3-large",
            "embedding_dimensions": dim,
        },
    )

    return PipelineArtifacts(
        documents=[doc],
        chunks=[chunk_1, chunk_2],
        extractions=[ext_1, ext_2],
        knowledge_graph=kg,
        vector_index=index,
        retriever=retriever,
        summary=summary,
    )


class ArtifactPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = Path(self.temp_dir.name) / "artifacts"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_save_and_load_round_trip_preserves_all_provenance_and_graphs(self) -> None:
        original = make_test_artifacts(dim=3)
        saved_dir = save_artifacts(original, self.storage_path)
        self.assertTrue(saved_dir.is_dir())
        self.assertTrue((saved_dir / "manifest.json").is_file())
        self.assertTrue((saved_dir / "documents.json").is_file())
        self.assertTrue((saved_dir / "chunks.json").is_file())
        self.assertTrue((saved_dir / "extractions.json").is_file())
        self.assertTrue((saved_dir / "vectors.npz").is_file())

        loaded = load_artifacts(saved_dir)

        # 1. Document metrics & checksums
        self.assertEqual(loaded.summary.document_count, 1)
        self.assertEqual(len(loaded.documents), 1)
        self.assertEqual(loaded.documents[0].document_id, "doc-test-1")
        self.assertEqual(loaded.documents[0].checksum_sha256, "abc123sha256")
        self.assertEqual(len(loaded.documents[0].sections), 2)
        self.assertEqual(loaded.documents[0].sections[1].page_number, 2)

        # 2. Chunk provenance
        self.assertEqual(len(loaded.chunks), 2)
        c2 = loaded.chunks[1]
        self.assertEqual(c2.chunk_id, "doc-test-1:s1:c0")
        self.assertEqual(c2.page_number, 2)
        self.assertEqual(c2.document_start_char, 25)
        self.assertEqual(c2.document_end_char, 68)

        # 3. Vector index reconstruction
        rec = loaded.vector_index.get_embedding(c2.chunk_id)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.vector, (0.0, 1.0, 0.0))
        self.assertEqual(rec.model_id, "text-embedding-3-large")

        # 4. Knowledge graph reconstruction
        rels = loaded.knowledge_graph.relationships_for("GraphRAG")
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].relationship.predicate, "uses")
        self.assertEqual(rels[0].chunk.page_number, 2)
        self.assertEqual(rels[0].chunk.document_id, "doc-test-1")

    def test_checksum_tampering_is_rejected(self) -> None:
        original = make_test_artifacts(dim=3)
        saved_dir = save_artifacts(original, self.storage_path)

        # Tamper with chunks.json
        chunks_file = saved_dir / "chunks.json"
        data = json.loads(chunks_file.read_text(encoding="utf-8"))
        data[0]["text"] = "Tampered text!"
        chunks_file.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(ValueError) as ctx:
            load_artifacts(saved_dir)
        self.assertIn("Integrity check failed for chunks.json", str(ctx.exception))

    def test_model_and_dimension_mismatch_is_rejected(self) -> None:
        original = make_test_artifacts(dim=3)
        saved_dir = save_artifacts(original, self.storage_path)

        with self.assertRaises(ValueError) as ctx_dim:
            load_artifacts(saved_dir, expected_embedding_dim=1536)
        self.assertIn("Incompatible embedding dimension", str(ctx_dim.exception))

        with self.assertRaises(ValueError) as ctx_model:
            load_artifacts(saved_dir, expected_embedding_model="text-embedding-ada-002")
        self.assertIn("Incompatible embedding model", str(ctx_model.exception))

    def test_missing_or_corrupt_manifest_is_rejected(self) -> None:
        original = make_test_artifacts(dim=3)
        saved_dir = save_artifacts(original, self.storage_path)

        # Delete manifest
        (saved_dir / "manifest.json").unlink()
        with self.assertRaises(ValueError) as ctx:
            load_artifacts(saved_dir)
        self.assertIn("manifest.json is missing", str(ctx.exception))

        # Missing directory
        with self.assertRaises(FileNotFoundError):
            load_artifacts(saved_dir / "non_existent_subdir")

    def test_qa_pipeline_with_loaded_artifacts_executes_offline(self) -> None:
        original = make_test_artifacts(dim=3)
        saved_dir = save_artifacts(original, self.storage_path)

        class DummyEmbedder:
            async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
                # Deterministic query vector
                return [[0.0, 1.0, 0.0] for _ in texts]

        class DummyAnswerer:
            async def generate(self, **kwargs) -> GeneratedAnswer:
                return GeneratedAnswer(
                    answer_text="GraphRAG uses Knowledge Graphs.",
                    evidence_references=[
                        EvidenceCitation(
                            chunk_id="doc-test-1:s1:c0",
                            quote="Second page chunk text on Knowledge Graphs.",
                            claim="Directly stated in chunk 2.",
                        )
                    ],
                    supporting_chunk_ids=["doc-test-1:s1:c0"],
                    insufficient_evidence=False,
                    confidence_explanation="Supported by page 2 evidence.",
                )

        # Load artifacts with offline provider
        loaded = load_artifacts(saved_dir)
        # Point loaded retriever client to query-only dummy embedder
        loaded.retriever.client = EmbeddingClient(
            DummyEmbedder(),
            model_id="text-embedding-3-large",
            dimensions=3,
        )

        qa_pipeline = QuestionAnsweringPipeline(
            document_pipeline=None,
            answer_generator=EvidenceBackedAnswerGenerator(DummyAnswerer(), model_id="gpt-6-luna"),
            retrieval_config=HybridRetrievalConfig(vector_top_k=2),
            is_mocked=True,
        )

        result, summary = run_sync(
            qa_pipeline.ask("What does GraphRAG use?", artifacts=loaded)
        )
        self.assertFalse(summary.insufficient_evidence)
        self.assertEqual(len(summary.evidence_references), 1)
        self.assertEqual(summary.evidence_references[0]["chunk_id"], "doc-test-1:s1:c0")
        self.assertEqual(summary.evidence_references[0]["page_number"], 2)


if __name__ == "__main__":
    unittest.main()
