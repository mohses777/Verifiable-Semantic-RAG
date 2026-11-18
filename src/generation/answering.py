"""Evidence-backed answer generation with strict citation and quotation validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from time import perf_counter
from typing import Any, Protocol

from src.core.models import TextChunk
from src.generation.schemas import EvidenceCitation, GeneratedAnswer
from src.retrieval.hybrid import HybridRetrievalBundle


class StructuredAnswerProvider(Protocol):
    """Minimal async structured-output interface for answer generation."""

    async def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: type[GeneratedAnswer],
    ) -> GeneratedAnswer:
        """Return a structured GeneratedAnswer from the LLM."""


class OpenAIStructuredAnswerProvider:
    """OpenAI Chat Completions structured-output provider for answer generation."""

    def __init__(self, *, api_key: str, base_url: str | None = None, client: Any | None = None) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for answer generation")
        if client is None:
            try:
                async_openai = getattr(import_module("openai"), "AsyncOpenAI")
            except ImportError as error:
                raise RuntimeError("OpenAI answer generation requires the 'openai' package.") from error
            client = async_openai(api_key=api_key, base_url=base_url)
        self._client = client

    async def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: type[GeneratedAnswer],
    ) -> GeneratedAnswer:
        response = await self._client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=schema,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("OpenAI declined to return a structured answer")
        return parsed


@dataclass(frozen=True)
class GroundedAnswerResult:
    """Verified answer together with its hybrid retrieval bundle and execution metadata."""

    question: str
    answer: GeneratedAnswer
    retrieval: HybridRetrievalBundle
    model_id: str
    execution_ms: float


class EvidenceBackedAnswerGenerator:
    """Generates answers from retrieved evidence and strips unverified citations."""

    def __init__(self, provider: StructuredAnswerProvider, *, model_id: str) -> None:
        if not model_id:
            raise ValueError("LLM model ID is required")
        self.provider = provider
        self.model_id = model_id

    @classmethod
    def from_environment(
        cls,
        provider: StructuredAnswerProvider | None = None,
    ) -> "EvidenceBackedAnswerGenerator":
        active_provider = provider or OpenAIStructuredAnswerProvider(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        return cls(active_provider, model_id=os.environ.get("LLM_MODEL", ""))

    async def answer_question(
        self,
        question: str,
        retrieval: HybridRetrievalBundle,
    ) -> GroundedAnswerResult:
        if not question or not question.strip():
            raise ValueError("Question must be non-empty")

        started = perf_counter()

        if not retrieval.evidence_chunks:
            empty_answer = GeneratedAnswer(
                answer_text="Insufficient evidence in the retrieved sources to answer this question.",
                evidence_references=[],
                supporting_chunk_ids=[],
                insufficient_evidence=True,
                confidence_explanation="No evidence chunks were retrieved for the question.",
            )
            return GroundedAnswerResult(
                question=question,
                answer=empty_answer,
                retrieval=retrieval,
                model_id=self.model_id,
                execution_ms=(perf_counter() - started) * 1_000,
            )

        evidence_context = _format_evidence_context(retrieval)
        raw_answer = await self.provider.generate(
            model=self.model_id,
            schema=GeneratedAnswer,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the user question using ONLY the supplied source chunks and graph relationships. "
                        "Every factual claim must be backed by an EvidenceCitation containing the exact chunk_id "
                        "and an exact verbatim quote from that chunk. "
                        "If the supplied evidence is insufficient to answer the question, set "
                        "insufficient_evidence=True and state clearly what is missing. "
                        "Never invent facts, chunk IDs, document paths, page numbers, or quotations."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nSupplied Evidence:\n{evidence_context}",
                },
            ],
        )

        grounded_answer = _validate_and_ground_answer(raw_answer, retrieval)
        return GroundedAnswerResult(
            question=question,
            answer=grounded_answer,
            retrieval=retrieval,
            model_id=self.model_id,
            execution_ms=(perf_counter() - started) * 1_000,
        )


def _format_evidence_context(retrieval: HybridRetrievalBundle) -> str:
    sections: list[str] = ["### Source Passages"]
    for idx, item in enumerate(retrieval.evidence_chunks, 1):
        chunk = item.chunk
        page_label = str(chunk.page_number) if chunk.page_number is not None else "N/A"
        sections.append(
            f"[{idx}] chunk_id={chunk.chunk_id} | document_id={chunk.document_id} | "
            f"page={page_label} | sources={','.join(item.retrieval_sources)}\n"
            f"Text: {chunk.text}"
        )

    if retrieval.graph_relationships:
        sections.append("\n### Grounded Knowledge Graph Relationships")
        for rel_ev in retrieval.graph_relationships:
            rel = rel_ev.relationship
            page_label = str(rel_ev.chunk.page_number) if rel_ev.chunk.page_number is not None else "N/A"
            sections.append(
                f"- ({rel.subject}) --[{rel.predicate}]--> ({rel.object}) | "
                f"chunk_id={rel.source_chunk_id} | page={page_label} | quote=\"{rel.evidence}\""
            )

    return "\n\n".join(sections)


def _validate_and_ground_answer(
    raw_answer: GeneratedAnswer,
    retrieval: HybridRetrievalBundle,
) -> GeneratedAnswer:
    """Verify every citation against retrieved chunks and overwrite provenance from source."""
    allowed_chunks: dict[str, TextChunk] = {
        item.chunk.chunk_id: item.chunk for item in retrieval.evidence_chunks
    }

    verified_citations: list[EvidenceCitation] = []
    supporting_ids: list[str] = []

    for citation in raw_answer.evidence_references:
        chunk = allowed_chunks.get(citation.chunk_id)
        if chunk is None:
            continue
        if not _is_quote_supported(citation.quote, chunk.text):
            continue

        grounded_citation = citation.model_copy(
            update={
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source_path": chunk.source_path,
                "page_number": chunk.page_number,
            }
        )
        verified_citations.append(grounded_citation)
        if chunk.chunk_id not in supporting_ids:
            supporting_ids.append(chunk.chunk_id)

    if raw_answer.insufficient_evidence:
        return raw_answer.model_copy(
            update={
                "evidence_references": verified_citations,
                "supporting_chunk_ids": supporting_ids,
                "insufficient_evidence": True,
            }
        )

    if not verified_citations:
        return GeneratedAnswer(
            answer_text="Insufficient verifiable evidence in the retrieved sources to support an answer.",
            evidence_references=[],
            supporting_chunk_ids=[],
            insufficient_evidence=True,
            confidence_explanation=(
                "The candidate answer lacked verifiable quotations matching the retrieved source chunks."
            ),
        )

    return raw_answer.model_copy(
        update={
            "evidence_references": verified_citations,
            "supporting_chunk_ids": supporting_ids,
            "insufficient_evidence": False,
        }
    )


def _is_quote_supported(quote: str, chunk_text: str) -> bool:
    normalized_quote = " ".join(quote.split()).casefold()
    if not normalized_quote:
        return False
    normalized_source = " ".join(chunk_text.split()).casefold()
    return normalized_quote in normalized_source
