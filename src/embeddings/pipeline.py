"""Bridge from provenance-bearing chunks to the replaceable vector index."""

from __future__ import annotations

from src.core.models import ChunkEmbedding, RetrievalResult, TextChunk
from src.embeddings.client import EmbeddingClient
from src.embeddings.index import VectorIndex


class SemanticRetriever:
    """Indexes chunks and retrieves evidence without dropping provenance."""

    def __init__(self, client: EmbeddingClient, index: VectorIndex) -> None:
        self.client = client
        self.index = index

    async def index_chunks(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            return
        vectors = await self.client.embed_texts([chunk.text for chunk in chunks])
        self.index.add([
            ChunkEmbedding(chunk=chunk, vector=vector, model_id=self.client.model_id)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ])

    async def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievalResult]:
        if not query or not query.strip():
            raise ValueError("Retrieval query must be non-empty")
        query_vector = (await self.client.embed_texts([query]))[0]
        return self.index.search(query_vector, top_k=top_k)
