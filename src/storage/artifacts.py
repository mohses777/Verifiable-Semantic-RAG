"""Lightweight, reliable local persistence for research artifacts using JSON and NumPy."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.core.models import ChunkEmbedding, DocumentSection, SourceDocument, TextChunk
from src.embeddings.client import EmbeddingClient, EmbeddingProvider
from src.embeddings.index import InMemoryVectorIndex, VectorIndex
from src.embeddings.pipeline import SemanticRetriever
from src.knowledge.extraction import ChunkExtraction
from src.knowledge.graph import KnowledgeGraph
from src.knowledge.schemas import ExtractedEntity, ExtractedRelationship, ExtractionResult
from src.pipeline.runner import DocumentSummary, PipelineArtifacts, PipelineSummary

SCHEMA_VERSION = "1.0.0"


class OfflineEmbeddingProvider:
    """Zero-network embedding provider that rejects live embedding calls.

    Used when loading saved artifacts to guarantee that no accidental paid
    API calls occur during retrieval or question answering.
    """

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        raise RuntimeError(
            "OfflineEmbeddingProvider: Attempted to call embedding API on loaded artifacts. "
            "To query with live text, configure an active embedding client."
        )


@dataclass(frozen=True)
class ArtifactManifest:
    """Metadata, source checksums, and integrity hashes for saved artifacts."""

    schema_version: str
    created_at: str
    document_count: int
    chunk_count: int
    entity_count: int
    unique_entity_count: int
    relationship_count: int
    explicit_relationship_count: int
    embedding_model: str
    embedding_dimension: int
    llm_model: str
    is_mocked: bool
    documents: list[dict[str, Any]]
    checksums: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def save_artifacts(
    artifacts: PipelineArtifacts,
    directory: str | Path,
    *,
    overwrite: bool = True,
) -> Path:
    """Atomically save pipeline artifacts to a local directory using JSON and NumPy.

    Directory structure:
        directory/
            manifest.json     # Schema version, metadata, and SHA-256 integrity checksums
            documents.json    # SourceDocument records with sections and checksums
            chunks.json       # TextChunk records with exact character/page provenance
            extractions.json  # ChunkExtraction records with entities and relationships
            vectors.npz       # NumPy float32 vectors and chunk IDs
    """
    target_dir = Path(directory).resolve()
    parent_dir = target_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    if target_dir.exists() and not overwrite:
        raise FileExistsError(f"Artifact directory already exists: {target_dir}")

    # Create isolated staging directory for atomic writes
    temp_dir = parent_dir / f".tmp_{target_dir.name}_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=False)

    try:
        # 1. Serialize documents
        docs_payload = [
            {
                "document_id": doc.document_id,
                "source_path": doc.source_path,
                "source_type": doc.source_type,
                "checksum_sha256": doc.checksum_sha256,
                "sections": [
                    {
                        "text": s.text,
                        "page_number": s.page_number,
                        "document_offset": s.document_offset,
                    }
                    for s in doc.sections
                ],
                "metadata": dict(doc.metadata),
            }
            for doc in artifacts.documents
        ]
        docs_file = temp_dir / "documents.json"
        docs_file.write_text(json.dumps(docs_payload, indent=2), encoding="utf-8")

        # 2. Serialize chunks
        chunks_payload = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "text": c.text,
                "source_path": c.source_path,
                "source_type": c.source_type,
                "page_number": c.page_number,
                "section_index": c.section_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "document_start_char": c.document_start_char,
                "document_end_char": c.document_end_char,
                "metadata": dict(c.metadata),
            }
            for c in artifacts.chunks
        ]
        chunks_file = temp_dir / "chunks.json"
        chunks_file.write_text(json.dumps(chunks_payload, indent=2), encoding="utf-8")

        # 3. Serialize extractions
        extractions_payload = [
            {
                "chunk_id": ext.chunk.chunk_id,
                "model_id": ext.model_id,
                "execution_ms": ext.execution_ms,
                "entities": [
                    {
                        "name": e.name,
                        "entity_type": e.entity_type,
                        "evidence": e.evidence,
                        "description": e.description,
                        "source_chunk_id": e.source_chunk_id,
                    }
                    for e in ext.result.entities
                ],
                "relationships": [
                    {
                        "subject": r.subject,
                        "predicate": r.predicate,
                        "object": r.object,
                        "evidence": r.evidence,
                        "support_level": r.support_level,
                        "source_chunk_id": r.source_chunk_id,
                    }
                    for r in ext.result.relationships
                ],
            }
            for ext in artifacts.extractions
        ]
        extractions_file = temp_dir / "extractions.json"
        extractions_file.write_text(json.dumps(extractions_payload, indent=2), encoding="utf-8")

        # 4. Serialize vectors to NumPy NPZ
        # Collect vectors in exact chunk order
        records = getattr(artifacts.vector_index, "records", {})
        ordered_vectors: list[tuple[float, ...]] = []
        ordered_chunk_ids: list[str] = []
        embedding_model = artifacts.summary.model_configuration.get("embedding_model", "")
        embedding_dim = artifacts.summary.model_configuration.get(
            "embedding_dimensions", getattr(artifacts.vector_index, "dimensions", 0)
        )

        for c in artifacts.chunks:
            embedding = records.get(c.chunk_id)
            if embedding is not None:
                ordered_vectors.append(embedding.vector)
                ordered_chunk_ids.append(c.chunk_id)
                if not embedding_model:
                    embedding_model = embedding.model_id
            else:
                raise ValueError(f"Missing embedding vector for chunk: {c.chunk_id}")

        if ordered_vectors:
            vector_array = np.array(ordered_vectors, dtype=np.float32)
            if vector_array.shape[1] != embedding_dim:
                raise ValueError(
                    f"Vector dimension mismatch: expected {embedding_dim}, found {vector_array.shape[1]}"
                )
        else:
            vector_array = np.empty((0, embedding_dim), dtype=np.float32)

        vectors_file = temp_dir / "vectors.npz"
        np.savez_compressed(
            vectors_file,
            vectors=vector_array,
            chunk_ids=np.array(ordered_chunk_ids, dtype=np.str_),
            model_id=np.array(embedding_model, dtype=np.str_),
        )

        # 5. Compute file checksums for manifest integrity
        checksums = {
            "documents.json": _sha256_file(docs_file),
            "chunks.json": _sha256_file(chunks_file),
            "extractions.json": _sha256_file(extractions_file),
            "vectors.npz": _sha256_file(vectors_file),
        }

        # 6. Write manifest.json
        manifest = ArtifactManifest(
            schema_version=SCHEMA_VERSION,
            created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            document_count=artifacts.summary.document_count,
            chunk_count=artifacts.summary.chunk_count,
            entity_count=artifacts.summary.entity_count,
            unique_entity_count=artifacts.summary.unique_entity_count,
            relationship_count=artifacts.summary.relationship_count,
            explicit_relationship_count=artifacts.summary.explicit_relationship_count,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dim,
            llm_model=artifacts.summary.model_configuration.get("llm_model", ""),
            is_mocked=artifacts.summary.is_mocked,
            documents=[asdict(d) for d in artifacts.summary.documents],
            checksums=checksums,
        )
        manifest_file = temp_dir / "manifest.json"
        manifest_file.write_text(manifest.to_json(), encoding="utf-8")

        # 7. Atomic directory swap
        if target_dir.exists():
            backup_dir = parent_dir / f".bak_{target_dir.name}_{uuid.uuid4().hex[:8]}"
            os.rename(target_dir, backup_dir)
            try:
                os.rename(temp_dir, target_dir)
                shutil.rmtree(backup_dir, ignore_errors=True)
            except Exception:
                if backup_dir.exists():
                    os.rename(backup_dir, target_dir)
                raise
        else:
            os.rename(temp_dir, target_dir)

        return target_dir

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


def load_artifacts(
    directory: str | Path,
    *,
    expected_embedding_model: str | None = None,
    expected_embedding_dim: int | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> PipelineArtifacts:
    """Load and validate research artifacts, reconstructing the vector index and graph.

    Performs strict schema and SHA-256 integrity validation. Rejects corrupt,
    tampered, or model-mismatched files with descriptive errors.
    """
    path = Path(directory).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {path}")

    manifest_file = path / "manifest.json"
    if not manifest_file.is_file():
        raise ValueError(f"Corrupt artifact: manifest.json is missing in {path}")

    try:
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ValueError(f"Corrupt artifact: manifest.json contains invalid JSON in {path}") from err

    schema_version = manifest_data.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported artifact schema version: {schema_version} (expected {SCHEMA_VERSION})"
        )

    # Validate integrity checksums
    checksums: dict[str, str] = manifest_data.get("checksums", {})
    required_files = ["documents.json", "chunks.json", "extractions.json", "vectors.npz"]
    for fname in required_files:
        fpath = path / fname
        if not fpath.is_file():
            raise ValueError(f"Corrupt artifact: required file {fname} is missing in {path}")
        expected_hash = checksums.get(fname)
        if not expected_hash:
            raise ValueError(f"Corrupt artifact: manifest missing checksum for {fname}")
        actual_hash = _sha256_file(fpath)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Integrity check failed for {fname}: checksum mismatch "
                f"(expected {expected_hash}, computed {actual_hash})"
            )

    # Validate model and dimension compatibility
    dim = int(manifest_data.get("embedding_dimension", 0))
    model_id = str(manifest_data.get("embedding_model", ""))
    if expected_embedding_dim is not None and dim != expected_embedding_dim:
        raise ValueError(
            f"Incompatible embedding dimension: artifact has {dim}, expected {expected_embedding_dim}"
        )
    if expected_embedding_model is not None and model_id != expected_embedding_model:
        raise ValueError(
            f"Incompatible embedding model: artifact has '{model_id}', expected '{expected_embedding_model}'"
        )

    # 1. Load documents
    docs_raw = json.loads((path / "documents.json").read_text(encoding="utf-8"))
    documents: list[SourceDocument] = []
    for d in docs_raw:
        sections = tuple(
            DocumentSection(
                text=s["text"],
                page_number=s["page_number"],
                document_offset=s["document_offset"],
            )
            for s in d["sections"]
        )
        documents.append(
            SourceDocument(
                document_id=d["document_id"],
                source_path=d["source_path"],
                source_type=d["source_type"],
                checksum_sha256=d["checksum_sha256"],
                sections=sections,
                metadata=d.get("metadata", {}),
            )
        )

    # 2. Load chunks
    chunks_raw = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
    chunks: list[TextChunk] = []
    chunks_by_id: dict[str, TextChunk] = {}
    for c in chunks_raw:
        chunk = TextChunk(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            text=c["text"],
            source_path=c["source_path"],
            source_type=c["source_type"],
            page_number=c["page_number"],
            section_index=c["section_index"],
            start_char=c["start_char"],
            end_char=c["end_char"],
            document_start_char=c["document_start_char"],
            document_end_char=c["document_end_char"],
            metadata=c.get("metadata", {}),
        )
        chunks.append(chunk)
        chunks_by_id[chunk.chunk_id] = chunk

    # 3. Load vectors
    with np.load(path / "vectors.npz", allow_pickle=False) as npz:
        vectors_arr = npz["vectors"]
        chunk_ids_arr = [str(cid) for cid in npz["chunk_ids"].tolist()]

    if len(vectors_arr) != len(chunks):
        raise ValueError(
            f"Corrupt artifact: vector count ({len(vectors_arr)}) does not match chunk count ({len(chunks)})"
        )
    if vectors_arr.shape[1] != dim:
        raise ValueError(
            f"Corrupt artifact: vector dimension ({vectors_arr.shape[1]}) does not match manifest dimension ({dim})"
        )
    if chunk_ids_arr != [c.chunk_id for c in chunks]:
        raise ValueError("Corrupt artifact: vector chunk IDs do not match chunks.json order")

    # 4. Load extractions
    extractions_raw = json.loads((path / "extractions.json").read_text(encoding="utf-8"))
    extractions: list[ChunkExtraction] = []
    for ext_dict in extractions_raw:
        cid = ext_dict["chunk_id"]
        chunk = chunks_by_id.get(cid)
        if chunk is None:
            raise ValueError(f"Corrupt artifact: extraction references unknown chunk_id {cid}")
        entities = [
            ExtractedEntity(
                name=e["name"],
                entity_type=e["entity_type"],
                evidence=e["evidence"],
                description=e.get("description"),
                source_chunk_id=e.get("source_chunk_id", cid),
            )
            for e in ext_dict["entities"]
        ]
        relationships = [
            ExtractedRelationship(
                subject=r["subject"],
                predicate=r["predicate"],
                object=r["object"],
                evidence=r["evidence"],
                support_level=r["support_level"],
                source_chunk_id=r.get("source_chunk_id", cid),
            )
            for r in ext_dict["relationships"]
        ]
        extractions.append(
            ChunkExtraction(
                chunk=chunk,
                result=ExtractionResult(entities=entities, relationships=relationships),
                model_id=ext_dict["model_id"],
                execution_ms=float(ext_dict["execution_ms"]),
            )
        )

    # 5. Rebuild InMemoryVectorIndex
    vector_index = InMemoryVectorIndex(dimensions=dim)
    embeddings = [
        ChunkEmbedding(
            chunk=chunk,
            vector=tuple(float(val) for val in vector_row),
            model_id=model_id,
        )
        for chunk, vector_row in zip(chunks, vectors_arr, strict=True)
    ]
    vector_index.add(embeddings)

    # 6. Rebuild KnowledgeGraph
    knowledge_graph = KnowledgeGraph()
    for extraction in extractions:
        knowledge_graph.add_extraction(extraction)

    # 7. Construct SemanticRetriever
    active_client = embedding_client or EmbeddingClient(
        OfflineEmbeddingProvider(),
        model_id=model_id,
        dimensions=dim,
    )
    retriever = SemanticRetriever(active_client, vector_index)

    # 8. Reconstruct PipelineSummary
    doc_summaries = [
        DocumentSummary(
            document_id=d["document_id"],
            source_path=d["source_path"],
            source_type=d["source_type"],
            checksum_sha256=d["checksum_sha256"],
            page_count=d["page_count"],
            chunk_count=d["chunk_count"],
        )
        for d in manifest_data["documents"]
    ]
    summary = PipelineSummary(
        document_count=manifest_data["document_count"],
        page_count=sum(d.page_count for d in doc_summaries),
        chunk_count=manifest_data["chunk_count"],
        entity_count=manifest_data["entity_count"],
        unique_entity_count=manifest_data["unique_entity_count"],
        relationship_count=manifest_data["relationship_count"],
        explicit_relationship_count=manifest_data["explicit_relationship_count"],
        documents=doc_summaries,
        execution_time_ms=0.0,
        is_mocked=manifest_data["is_mocked"],
        model_configuration={
            "llm_model": manifest_data["llm_model"],
            "embedding_model": model_id,
            "embedding_dimensions": dim,
        },
    )

    return PipelineArtifacts(
        documents=documents,
        chunks=chunks,
        extractions=extractions,
        knowledge_graph=knowledge_graph,
        vector_index=vector_index,
        retriever=retriever,
        summary=summary,
    )


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()
