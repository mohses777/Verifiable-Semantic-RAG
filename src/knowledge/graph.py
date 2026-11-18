"""NetworkX graph construction that preserves source evidence."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from src.core.models import TextChunk
from src.knowledge.extraction import ChunkExtraction
from src.knowledge.schemas import ExtractedEntity, ExtractedRelationship


@dataclass(frozen=True)
class RelationshipEvidence:
    relationship: ExtractedRelationship
    chunk: TextChunk


@dataclass(frozen=True)
class EntityEvidence:
    entity: ExtractedEntity
    chunk: TextChunk


class KnowledgeGraph:
    """Graph facade that merges entities while retaining all evidence chunks."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()
        self._chunks: dict[str, TextChunk] = {}

    def add_extraction(self, extraction: ChunkExtraction) -> None:
        chunk = extraction.chunk
        self._chunks[chunk.chunk_id] = chunk
        for entity in extraction.result.entities:
            node_key = _key(entity.name)
            if self.graph.has_node(node_key):
                node = self.graph.nodes[node_key]
                node["entities"].append(entity)
                node["source_chunk_ids"].add(entity.source_chunk_id)
            else:
                self.graph.add_node(
                    node_key,
                    name=entity.name,
                    entity_type=entity.entity_type,
                    entities=[entity],
                    source_chunk_ids={entity.source_chunk_id},
                )

        known_entities = set(self.graph.nodes)
        for relationship in extraction.result.relationships:
            if relationship.support_level != "explicit":
                continue
            subject, object_ = _key(relationship.subject), _key(relationship.object)
            if subject not in known_entities or object_ not in known_entities:
                continue
            self.graph.add_edge(subject, object_, relationship=relationship)

    def lookup_entity(self, name: str) -> tuple[ExtractedEntity, ...]:
        node = self.graph.nodes.get(_key(name))
        return tuple(node["entities"]) if node else ()

    def entity_evidence_for(self, name: str) -> list[EntityEvidence]:
        """Return entity mentions together with the document/page evidence."""
        return [
            EntityEvidence(entity=entity, chunk=self._chunks[entity.source_chunk_id])
            for entity in self.lookup_entity(name)
        ]

    def relationships_for(self, name: str) -> list[RelationshipEvidence]:
        node_key = _key(name)
        if node_key not in self.graph:
            return []
        evidence: list[RelationshipEvidence] = []
        for _, _, attributes in self.graph.out_edges(node_key, data=True):
            relationship = attributes["relationship"]
            chunk = self._chunks[relationship.source_chunk_id]
            evidence.append(RelationshipEvidence(relationship=relationship, chunk=chunk))
        for _, _, attributes in self.graph.in_edges(node_key, data=True):
            relationship = attributes["relationship"]
            chunk = self._chunks[relationship.source_chunk_id]
            evidence.append(RelationshipEvidence(relationship=relationship, chunk=chunk))
        return evidence


def _key(name: str) -> str:
    return " ".join(name.split()).casefold()
