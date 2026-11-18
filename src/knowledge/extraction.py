"""Mockable GPT-6 Luna structured extraction with evidence checks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from time import perf_counter
from typing import Any, Protocol

from src.core.models import TextChunk
from src.knowledge.schemas import ExtractedEntity, ExtractedRelationship, ExtractionResult


class StructuredExtractionProvider(Protocol):
    async def extract(self, *, model: str, messages: list[dict[str, str]], schema: type[ExtractionResult]) -> ExtractionResult:
        """Return a validated structured extraction response."""


class OpenAIStructuredExtractionProvider:
    """OpenAI Chat Completions structured-output provider."""

    def __init__(self, *, api_key: str, base_url: str | None = None, client: Any | None = None) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for semantic extraction")
        if client is None:
            try:
                async_openai = getattr(import_module("openai"), "AsyncOpenAI")
            except ImportError as error:
                raise RuntimeError("OpenAI extraction requires the 'openai' package.") from error
            client = async_openai(api_key=api_key, base_url=base_url)
        self._client = client

    async def extract(self, *, model: str, messages: list[dict[str, str]], schema: type[ExtractionResult]) -> ExtractionResult:
        response = await self._client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=schema,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI declined to return structured extraction data")
        return parsed


@dataclass(frozen=True)
class ChunkExtraction:
    """Validated extraction and the exact chunk from which it was derived."""

    chunk: TextChunk
    result: ExtractionResult
    model_id: str
    execution_ms: float


class SemanticExtractor:
    """Converts a chunk into source-grounded entities and relation claims."""

    def __init__(self, provider: StructuredExtractionProvider, *, model_id: str) -> None:
        if not model_id:
            raise ValueError("LLM model ID is required")
        self.provider = provider
        self.model_id = model_id

    @classmethod
    def from_environment(cls, provider: StructuredExtractionProvider | None = None) -> "SemanticExtractor":
        active_provider = provider or OpenAIStructuredExtractionProvider(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        return cls(active_provider, model_id=os.environ.get("LLM_MODEL", ""))

    async def extract_chunk(self, chunk: TextChunk) -> ChunkExtraction:
        if not chunk.text or not chunk.text.strip():
            raise ValueError("Cannot extract knowledge from an empty chunk")
        started = perf_counter()
        result = await self.provider.extract(
            model=self.model_id,
            schema=ExtractionResult,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract only entities and relationships explicitly supported by the source. "
                        "For every item include an exact quotation in evidence. "
                        "Use support_level='explicit' only when the quotation states the relationship; "
                        "otherwise use 'uncertain' or 'inferred'. Do not invent facts."
                    ),
                },
                {"role": "user", "content": f"Source chunk:\n{chunk.text}"},
            ],
        )
        normalized = _ground_result(result, chunk)
        return ChunkExtraction(
            chunk=chunk,
            result=normalized,
            model_id=self.model_id,
            execution_ms=(perf_counter() - started) * 1_000,
        )


def _ground_result(result: ExtractionResult, chunk: TextChunk) -> ExtractionResult:
    """Attach immutable chunk provenance and demote unquoted relation claims."""
    entities = [
        entity.model_copy(update={"source_chunk_id": chunk.chunk_id})
        for entity in result.entities
        if _is_supported(entity.evidence, chunk.text)
    ]
    names = {_key(entity.name) for entity in entities}
    relationships: list[ExtractedRelationship] = []
    for relationship in result.relationships:
        support_level = relationship.support_level
        if (
            not _is_supported(relationship.evidence, chunk.text)
            or _key(relationship.subject) not in names
            or _key(relationship.object) not in names
        ):
            support_level = "uncertain"
        relationships.append(relationship.model_copy(update={
            "source_chunk_id": chunk.chunk_id,
            "support_level": support_level,
        }))
    return ExtractionResult(entities=entities, relationships=relationships)


def _is_supported(evidence: str, source: str) -> bool:
    return " ".join(evidence.split()).casefold() in " ".join(source.split()).casefold()


def _key(name: str) -> str:
    return " ".join(name.split()).casefold()
