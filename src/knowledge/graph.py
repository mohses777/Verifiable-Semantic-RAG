"""NetworkX graph construction that preserves source evidence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

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

    def get_chunk(self, chunk_id: str) -> TextChunk | None:
        """Retrieve a stored source chunk by its deterministic chunk_id."""
        return self._chunks.get(chunk_id)

    def entities_in_chunk(self, chunk_id: str) -> list[str]:
        """Return canonical entity names mentioned in the specified chunk."""
        names: list[str] = []
        for _, attrs in self.graph.nodes(data=True):
            if chunk_id in attrs.get("source_chunk_ids", set()):
                names.append(attrs["name"])
        return sorted(names, key=_key)

    def match_entities_in_text(self, text: str) -> list[str]:
        """Find graph entities whose normalized names appear in the query text."""
        normalized_text = _key(text)
        if not normalized_text:
            return []
        matched: list[str] = []
        for node_key, attrs in self.graph.nodes(data=True):
            if node_key and node_key in normalized_text:
                matched.append(attrs["name"])
        return sorted(matched, key=_key)

    def traverse(
        self,
        seed_names: Iterable[str],
        *,
        max_depth: int = 1,
    ) -> tuple[list[EntityEvidence], list[RelationshipEvidence]]:
        """Breadth-first traversal up to max_depth hops from seed entities."""
        if max_depth < 0:
            raise ValueError("max_depth must be at least zero")

        visited_nodes: set[str] = set()
        queue: deque[tuple[str, int]] = deque()
        for name in seed_names:
            node_key = _key(name)
            if node_key in self.graph and node_key not in visited_nodes:
                visited_nodes.add(node_key)
                queue.append((node_key, 0))

        entity_evidences: list[EntityEvidence] = []
        relationship_evidences: list[RelationshipEvidence] = []
        seen_rel_keys: set[tuple[str, str, str, str, str]] = set()

        while queue:
            current_key, depth = queue.popleft()
            node_attrs = self.graph.nodes[current_key]
            for entity in node_attrs.get("entities", []):
                chunk = self._chunks.get(entity.source_chunk_id)
                if chunk is not None:
                    entity_evidences.append(EntityEvidence(entity=entity, chunk=chunk))

            if depth >= max_depth:
                continue

            # Traverse outgoing and incoming edges
            edges = list(self.graph.out_edges(current_key, data=True)) + list(
                self.graph.in_edges(current_key, data=True)
            )
            for u, v, edge_attrs in edges:
                rel: ExtractedRelationship = edge_attrs["relationship"]
                rel_key = (
                    _key(rel.subject),
                    _key(rel.predicate),
                    _key(rel.object),
                    rel.source_chunk_id,
                    rel.evidence,
                )
                if rel_key not in seen_rel_keys:
                    seen_rel_keys.add(rel_key)
                    chunk = self._chunks.get(rel.source_chunk_id)
                    if chunk is not None:
                        relationship_evidences.append(RelationshipEvidence(relationship=rel, chunk=chunk))

                neighbor = v if u == current_key else u
                if neighbor in self.graph and neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append((neighbor, depth + 1))

        return entity_evidences, relationship_evidences


def _key(name: str) -> str:
    return " ".join(name.split()).casefold()
