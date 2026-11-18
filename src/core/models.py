"""Immutable records used to preserve source provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class DocumentSection:
    """Readable text from one source unit, normally one PDF page."""

    text: str
    page_number: int | None
    document_offset: int


@dataclass(frozen=True)
class SourceDocument:
    """An ingested source document with stable identity and metadata."""

    document_id: str
    source_path: str
    source_type: str
    checksum_sha256: str
    sections: tuple[DocumentSection, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TextChunk:
    """A source-addressable unit of retrieval evidence."""

    chunk_id: str
    document_id: str
    text: str
    source_path: str
    source_type: str
    page_number: int | None
    section_index: int
    start_char: int
    end_char: int
    document_start_char: int
    document_end_char: int
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkEmbedding:
    """A dense vector that remains permanently linked to its source chunk."""

    chunk: TextChunk
    vector: tuple[float, ...]
    model_id: str


@dataclass(frozen=True)
class RetrievalResult:
    """A ranked evidence chunk returned by semantic retrieval."""

    chunk: TextChunk
    score: float
    model_id: str
