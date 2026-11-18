"""TXT, Markdown, and PDF ingestion with durable source provenance."""

from __future__ import annotations

from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, cast

from src.core.models import DocumentSection, SourceDocument

SUPPORTED_SUFFIXES = {".txt": "text", ".md": "markdown", ".markdown": "markdown", ".pdf": "pdf"}


def clean_text(text: str) -> str:
    """Normalize extracted text while retaining readable paragraph boundaries."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [line.rstrip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def ingest_file(path: str | Path, *, pdf_reader_factory: Callable[[Path], Any] | None = None) -> SourceDocument:
    """Read one supported document and retain source/page-level provenance.

    PDF support uses the optional `pypdf` package only when a PDF is requested.
    """
    source = Path(path).resolve()
    source_type = SUPPORTED_SUFFIXES.get(source.suffix.lower())
    if source_type is None:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported document type '{source.suffix}'. Supported: {allowed}")
    if not source.is_file():
        raise FileNotFoundError(source)

    raw_bytes = source.read_bytes()
    checksum = sha256(raw_bytes).hexdigest()
    document_id = sha256(f"{source.as_posix()}:{checksum}".encode("utf-8")).hexdigest()[:24]
    if source_type == "pdf":
        sections = _read_pdf(source, pdf_reader_factory)
    else:
        text = clean_text(raw_bytes.decode("utf-8"))
        sections = (DocumentSection(text=text, page_number=None, document_offset=0),) if text else ()

    return SourceDocument(
        document_id=document_id,
        source_path=str(source),
        source_type=source_type,
        checksum_sha256=checksum,
        sections=sections,
        metadata={"filename": source.name, "suffix": source.suffix.lower()},
    )


def _read_pdf(source: Path, pdf_reader_factory: Callable[[Path], Any] | None) -> tuple[DocumentSection, ...]:
    reader_factory = pdf_reader_factory
    if reader_factory is None:
        try:
            pypdf = import_module("pypdf")
        except ImportError as error:
            raise RuntimeError("PDF ingestion requires pypdf. Install project dependencies first.") from error
        reader_factory = cast(Callable[[Path], Any], getattr(pypdf, "PdfReader"))

    if reader_factory is None:
        raise RuntimeError("PDF reader factory was not configured")
    reader = reader_factory(source)
    sections: list[DocumentSection] = []
    document_offset = 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            sections.append(DocumentSection(text=text, page_number=page_number, document_offset=document_offset))
        document_offset += len(text) + 1
    return tuple(sections)
