from __future__ import annotations

import unittest

from src.core.models import TextChunk
from src.embeddings.client import EmbeddingClient
from src.embeddings.index import InMemoryVectorIndex
from src.embeddings.pipeline import SemanticRetriever


class FakeEmbeddingProvider:
    def __init__(self, responses: list[list[float]]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[str], str]] = []
        self.position = 0

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        self.calls.append((texts, model))
        response = self.responses[self.position:self.position + len(texts)]
        self.position += len(texts)
        return response


def run(coroutine):
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("Mocked coroutine unexpectedly yielded control")


def chunk(chunk_id: str, text: str, *, page: int | None = 1) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id, document_id="document-a", text=text, source_path="paper.pdf", source_type="pdf",
        page_number=page, section_index=0, start_char=2, end_char=2 + len(text),
        document_start_char=2, document_end_char=2 + len(text), metadata={"filename": "paper.pdf"},
    )


class EmbeddingsAndRetrievalTests(unittest.TestCase):
    def test_client_batches_and_validates_vectors(self) -> None:
        provider = FakeEmbeddingProvider([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        client = EmbeddingClient(provider, model_id="text-embedding-3-large", dimensions=3, batch_size=2)
        vectors = run(client.embed_texts(["one", "two", "three"]))
        self.assertEqual(vectors, [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)])
        self.assertEqual(provider.calls, [(["one", "two"], "text-embedding-3-large"), (["three"], "text-embedding-3-large")])

    def test_retrieval_ranks_cosine_similarity_and_preserves_provenance(self) -> None:
        provider = FakeEmbeddingProvider([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]])
        client = EmbeddingClient(provider, model_id="text-embedding-3-large", dimensions=3)
        index = InMemoryVectorIndex(dimensions=3)
        retriever = SemanticRetriever(client, index)
        first, second = chunk("chunk-1", "Graph retrieval evidence", page=4), chunk("chunk-2", "Unrelated material", page=9)
        run(retriever.index_chunks([first, second]))
        results = run(retriever.retrieve("How does graph retrieval work?", top_k=1))
        self.assertEqual(results[0].chunk, first)
        self.assertEqual(results[0].chunk.page_number, 4)
        self.assertEqual(results[0].chunk.document_id, "document-a")
        self.assertEqual(results[0].chunk.start_char, 2)
        self.assertEqual(results[0].model_id, "text-embedding-3-large")

    def test_invalid_input_and_vectors_are_rejected(self) -> None:
        client = EmbeddingClient(FakeEmbeddingProvider([[1, 2]]), model_id="model", dimensions=3)
        with self.assertRaises(ValueError):
            run(client.embed_texts([" "]))
        with self.assertRaises(ValueError):
            run(client.embed_texts(["text"]))
        with self.assertRaises(ValueError):
            InMemoryVectorIndex(dimensions=3).search((1, 0), top_k=1)

    def test_empty_index_returns_no_evidence(self) -> None:
        index = InMemoryVectorIndex(dimensions=2)
        self.assertEqual(index.search((1, 0), top_k=3), [])


if __name__ == "__main__":
    unittest.main()
