"""Configurable chunking that carries source provenance to every chunk."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.models import SourceDocument, TextChunk


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 1_000
    overlap: int = 200

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be at least zero and smaller than chunk_size")


def chunk_document(document: SourceDocument, config: ChunkingConfig = ChunkingConfig()) -> list[TextChunk]:
    """Split each source section independently, preserving page and offsets."""
    chunks: list[TextChunk] = []
    for section_index, section in enumerate(document.sections):
        start = 0
        sequence = 0
        text = section.text
        while start < len(text):
            end = _choose_end(text, start, config.chunk_size)
            chunk_text = text[start:end]
            if chunk_text:
                chunks.append(
                    TextChunk(
                        chunk_id=f"{document.document_id}:s{section_index}:c{sequence}",
                        document_id=document.document_id,
                        text=chunk_text,
                        source_path=document.source_path,
                        source_type=document.source_type,
                        page_number=section.page_number,
                        section_index=section_index,
                        start_char=start,
                        end_char=end,
                        document_start_char=section.document_offset + start,
                        document_end_char=section.document_offset + end,
                        metadata=document.metadata,
                    )
                )
                sequence += 1
            if end >= len(text):
                break
            start = max(end - config.overlap, start + 1)
    return chunks


def _choose_end(text: str, start: int, chunk_size: int) -> int:
    limit = min(start + chunk_size, len(text))
    if limit == len(text):
        return limit
    boundary = max(text.rfind("\n", start + 1, limit + 1), text.rfind(" ", start + 1, limit + 1))
    return boundary if boundary > start else limit
