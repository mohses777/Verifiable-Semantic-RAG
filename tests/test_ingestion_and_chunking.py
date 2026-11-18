from __future__ import annotations

import unittest
from pathlib import Path

from src.ingestion.loader import ingest_file
from src.processing.chunking import ChunkingConfig, chunk_document


class IngestionAndChunkingTests(unittest.TestCase):
    fixtures = Path(__file__).parent / "fixtures"

    def test_txt_ingestion_has_stable_identity_and_metadata(self) -> None:
        path = self.fixtures / "source.txt"
        first = ingest_file(path)
        second = ingest_file(path)
        self.assertEqual(first.document_id, second.document_id)
        self.assertEqual(first.source_type, "text")
        self.assertEqual(first.sections[0].text, "First line.\nSecond line.")
        self.assertEqual(first.metadata["filename"], "source.txt")

    def test_markdown_chunks_preserve_offsets_and_overlap(self) -> None:
        document = ingest_file(self.fixtures / "notes.md")
        chunks = chunk_document(document, ChunkingConfig(chunk_size=18, overlap=6))
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, document.sections[0].text[chunks[0].start_char:chunks[0].end_char])
        self.assertEqual(chunks[0].document_id, document.document_id)
        self.assertIsNone(chunks[0].page_number)
        self.assertLess(chunks[1].start_char, chunks[0].end_char)
        self.assertTrue(all(chunk.chunk_id.startswith(f"{document.document_id}:s0:") for chunk in chunks))

    def test_pdf_pages_keep_page_numbers(self) -> None:
        class Page:
            def __init__(self, text: str) -> None:
                self.text = text
            def extract_text(self) -> str:
                return self.text
        class Reader:
            pages = [Page("Page one evidence."), Page("Page two evidence.")]
        document = ingest_file(self.fixtures / "paper.pdf", pdf_reader_factory=lambda _: Reader())
        chunks = chunk_document(document, ChunkingConfig(chunk_size=100, overlap=0))
        self.assertEqual([section.page_number for section in document.sections], [1, 2])
        self.assertEqual([chunk.page_number for chunk in chunks], [1, 2])
        self.assertLess(chunks[0].document_end_char, chunks[1].document_start_char)

    def test_invalid_chunk_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ChunkingConfig(chunk_size=100, overlap=100)

    def test_unsupported_file_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ingest_file(self.fixtures / "source.docx")


if __name__ == "__main__":
    unittest.main()
