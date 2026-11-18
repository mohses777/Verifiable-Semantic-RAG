"""Configurable OpenAI embedding client with a small provider boundary."""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any, Protocol


class EmbeddingProvider(Protocol):
    """Minimal async provider interface used by the application."""

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        """Return one embedding vector per input text, in order."""


class OpenAIEmbeddingProvider:
    """OpenAI embedding implementation, initialized only when explicitly used."""

    def __init__(self, *, api_key: str, base_url: str | None = None, client: Any | None = None) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        if client is None:
            try:
                async_openai = getattr(import_module("openai"), "AsyncOpenAI")
            except ImportError as error:
                raise RuntimeError("OpenAI embeddings require the 'openai' package.") from error
            client = async_openai(api_key=api_key, base_url=base_url)
        self._client = client
        self.last_usage: dict[str, int] | None = None

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(model=model, input=texts)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0)),
                "total_tokens": int(getattr(usage, "total_tokens", 0)),
            }
        return [list(item.embedding) for item in response.data]


class EmbeddingClient:
    """Validates embedding inputs and batches provider calls deterministically."""

    def __init__(self, provider: EmbeddingProvider, *, model_id: str, dimensions: int, batch_size: int = 100) -> None:
        if not model_id:
            raise ValueError("Embedding model ID is required")
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be greater than zero")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.provider = provider
        self.model_id = model_id
        self.dimensions = dimensions
        self.batch_size = batch_size

    @classmethod
    def from_environment(cls, provider: EmbeddingProvider | None = None, *, batch_size: int = 100) -> "EmbeddingClient":
        model_id = os.environ.get("EMBEDDING_MODEL", "")
        dimensions_text = os.environ.get("EMBEDDING_DIMENSION", "")
        try:
            dimensions = int(dimensions_text)
        except ValueError as error:
            raise ValueError("EMBEDDING_DIMENSION must be an integer") from error
        active_provider = provider or OpenAIEmbeddingProvider(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        return cls(active_provider, model_id=model_id, dimensions=dimensions, batch_size=batch_size)

    async def embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
        if any(not text or not text.strip() for text in texts):
            raise ValueError("Embedding input text must be non-empty")
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            batch_vectors = await self.provider.embed(batch, model=self.model_id)
            if len(batch_vectors) != len(batch):
                raise ValueError("Embedding provider returned a different number of vectors than inputs")
            vectors.extend(self._validate_vector(vector) for vector in batch_vectors)
        return vectors

    def _validate_vector(self, vector: list[float]) -> tuple[float, ...]:
        if len(vector) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} embedding dimensions, received {len(vector)}")
        values = tuple(float(value) for value in vector)
        if any(value != value or value in (float("inf"), float("-inf")) for value in values):
            raise ValueError("Embedding vector contains NaN or infinity")
        return values
