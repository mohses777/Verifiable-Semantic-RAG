"""Local, replaceable cosine-similarity vector index."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import sqrt

from src.core.models import ChunkEmbedding, RetrievalResult, TextChunk


class VectorIndex(ABC):
    """Storage boundary for future local or hosted vector backends."""

    @abstractmethod
    def add(self, embeddings: list[ChunkEmbedding]) -> None:
        """Store source-linked vectors."""

    @abstractmethod
    def search(self, query_vector: tuple[float, ...], *, top_k: int) -> list[RetrievalResult]:
        """Return highest-scoring evidence with untouched provenance."""


class InMemoryVectorIndex(VectorIndex):
    """Small deterministic index suitable for tests and early experiments."""

    def __init__(self, *, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")
        self.dimensions = dimensions
        self._records: dict[str, ChunkEmbedding] = {}

    @property
    def records(self) -> dict[str, ChunkEmbedding]:
        """Return a mapping of chunk_id to ChunkEmbedding."""
        return dict(self._records)

    def get_embedding(self, chunk_id: str) -> ChunkEmbedding | None:
        """Return a single stored ChunkEmbedding by chunk_id."""
        return self._records.get(chunk_id)

    def add(self, embeddings: list[ChunkEmbedding]) -> None:
        for embedding in embeddings:
            self._validate_vector(embedding.vector)
            if not embedding.chunk.chunk_id:
                raise ValueError("Chunk embeddings require a chunk_id")
            self._records[embedding.chunk.chunk_id] = embedding

    def search(self, query_vector: tuple[float, ...], *, top_k: int) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self._validate_vector(query_vector)
        ranked = [
            RetrievalResult(chunk=record.chunk, score=_cosine(query_vector, record.vector), model_id=record.model_id)
            for record in self._records.values()
        ]
        return sorted(ranked, key=lambda item: (-item.score, item.chunk.chunk_id))[:top_k]

    def _validate_vector(self, vector: tuple[float, ...]) -> None:
        if len(vector) != self.dimensions:
            raise ValueError(f"Expected a {self.dimensions}-dimension vector")
        if any(value != value or value in (float("inf"), float("-inf")) for value in vector):
            raise ValueError("Vector contains NaN or infinity")


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
