"""Pydantic schemas for source-grounded semantic extraction."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


SupportLevel = Literal["explicit", "uncertain", "inferred"]


class ExtractedEntity(BaseModel):
    """An entity mentioned in one source chunk."""

    name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    evidence: str = Field(min_length=1, description="Exact supporting quotation from the source chunk.")
    description: str | None = None
    source_chunk_id: str = ""


class ExtractedRelationship(BaseModel):
    """A relationship claim and the source quotation that supports it."""

    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)
    evidence: str = Field(min_length=1, description="Exact supporting quotation from the source chunk.")
    support_level: SupportLevel = "explicit"
    source_chunk_id: str = ""


class ExtractionResult(BaseModel):
    """Structured response returned for one document chunk."""

    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
