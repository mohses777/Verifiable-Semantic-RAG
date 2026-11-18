"""Pydantic schemas for evidence-backed answer generation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceCitation(BaseModel):
    """A verifiable citation linking a factual claim to a retrieved source chunk."""

    chunk_id: str = Field(min_length=1, description="Exact chunk_id of the retrieved evidence passage.")
    document_id: str = Field(default="", description="Originating document identifier.")
    source_path: str = Field(default="", description="Path or filename of the source document.")
    page_number: int | None = Field(
        default=None,
        description="1-based page number of the source document, if available.",
    )
    quote: str = Field(
        min_length=1,
        description="Exact verbatim quotation from the supplied evidence chunk supporting the claim.",
    )
    claim: str = Field(
        default="",
        description="The specific factual claim in the answer supported by this quotation.",
    )


class GeneratedAnswer(BaseModel):
    """Structured answer grounded exclusively in retrieved source evidence."""

    answer_text: str = Field(
        min_length=1,
        description="Grounded answer based exclusively on the supplied evidence.",
    )
    evidence_references: list[EvidenceCitation] = Field(default_factory=list)
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    insufficient_evidence: bool = Field(
        default=False,
        description="True when the retrieved evidence does not contain enough information to answer.",
    )
    confidence_explanation: str = Field(
        min_length=1,
        description="Explanation of confidence level or source uncertainty based strictly on retrieved evidence.",
    )
