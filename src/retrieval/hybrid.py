"""Hybrid retrieval combining vector similarity and knowledge graph traversal."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.models import TextChunk
from src.embeddings.pipeline import SemanticRetriever
from src.knowledge.graph import EntityEvidence, KnowledgeGraph, RelationshipEvidence
from src.knowledge.schemas import ExtractedEntity, ExtractedRelationship


@dataclass(frozen=True)
class HybridRetrievalConfig:
    """Configuration for combining vector retrieval and knowledge graph expansion."""

    vector_top_k: int = 5
    graph_traversal_depth: int = 1
    max_total_chunks: int = 10
    vector_weight: float = 0.7
    graph_weight: float = 0.3

    def __post_init__(self) -> None:
        if self.vector_top_k <= 0:
            raise ValueError("vector_top_k must be greater than zero")
        if self.graph_traversal_depth < 0:
            raise ValueError("graph_traversal_depth must be at least zero")
        if self.max_total_chunks <= 0:
            raise ValueError("max_total_chunks must be greater than zero")
        if self.vector_weight < 0.0 or self.graph_weight < 0.0:
            raise ValueError("weights must be non-negative")
        if (self.vector_weight + self.graph_weight) <= 0.0:
            raise ValueError("at least one weight must be greater than zero")


@dataclass(frozen=True)
class RankedEvidenceChunk:
    """A deduplicated evidence chunk with combined vector and graph provenance."""

    chunk: TextChunk
    combined_score: float
    vector_score: float
    graph_score: float
    retrieval_sources: tuple[str, ...]
    supporting_entities: tuple[ExtractedEntity, ...]
    supporting_relationships: tuple[ExtractedRelationship, ...]


@dataclass(frozen=True)
class HybridRetrievalBundle:
    """All deduplicated and ranked evidence retrieved for a user question."""

    query: str
    evidence_chunks: tuple[RankedEvidenceChunk, ...]
    graph_entities: tuple[EntityEvidence, ...]
    graph_relationships: tuple[RelationshipEvidence, ...]
    config: HybridRetrievalConfig


class HybridRetriever:
    """Combines vector similarity retrieval with provenance-preserving graph traversal."""

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        knowledge_graph: KnowledgeGraph,
        *,
        config: HybridRetrievalConfig = HybridRetrievalConfig(),
    ) -> None:
        self.semantic_retriever = semantic_retriever
        self.knowledge_graph = knowledge_graph
        self.config = config

    async def retrieve(
        self,
        query: str,
        *,
        config: HybridRetrievalConfig | None = None,
    ) -> HybridRetrievalBundle:
        if not query or not query.strip():
            raise ValueError("Retrieval query must be non-empty")

        active_config = config or self.config

        # 1. Dense vector retrieval
        vector_results = await self.semantic_retriever.retrieve(
            query, top_k=active_config.vector_top_k
        )

        # 2. Identify seed entities from query text and top vector chunks
        seed_names: list[str] = []
        seen_seeds: set[str] = set()

        for name in self.knowledge_graph.match_entities_in_text(query):
            key = name.casefold()
            if key not in seen_seeds:
                seen_seeds.add(key)
                seed_names.append(name)

        for v_res in vector_results:
            for name in self.knowledge_graph.entities_in_chunk(v_res.chunk.chunk_id):
                key = name.casefold()
                if key not in seen_seeds:
                    seen_seeds.add(key)
                    seed_names.append(name)

        # 3. Traverse knowledge graph up to configured depth
        entity_evidences, relationship_evidences = self.knowledge_graph.traverse(
            seed_names, max_depth=active_config.graph_traversal_depth
        )

        # 4. Deduplicate and merge chunk evidence
        chunks_by_id: dict[str, TextChunk] = {}
        vector_scores: dict[str, float] = {}
        sources_by_id: dict[str, list[str]] = {}
        entities_by_id: dict[str, list[ExtractedEntity]] = {}
        relationships_by_id: dict[str, list[ExtractedRelationship]] = {}

        for v_res in vector_results:
            cid = v_res.chunk.chunk_id
            chunks_by_id[cid] = v_res.chunk
            vector_scores[cid] = v_res.score
            sources_by_id.setdefault(cid, []).append("vector")

        for ent_ev in entity_evidences:
            cid = ent_ev.chunk.chunk_id
            chunks_by_id[cid] = ent_ev.chunk
            if "graph" not in sources_by_id.setdefault(cid, []):
                sources_by_id[cid].append("graph")
            ent_list = entities_by_id.setdefault(cid, [])
            if ent_ev.entity not in ent_list:
                ent_list.append(ent_ev.entity)

        for rel_ev in relationship_evidences:
            cid = rel_ev.chunk.chunk_id
            chunks_by_id[cid] = rel_ev.chunk
            if "graph" not in sources_by_id.setdefault(cid, []):
                sources_by_id[cid].append("graph")
            rel_list = relationships_by_id.setdefault(cid, [])
            if rel_ev.relationship not in rel_list:
                rel_list.append(rel_ev.relationship)

        # 5. Compute graph scores and combined scores
        raw_graph_points: dict[str, float] = {}
        for cid in chunks_by_id:
            ent_count = len(entities_by_id.get(cid, []))
            rel_count = len(relationships_by_id.get(cid, []))
            raw_graph_points[cid] = float(ent_count) + (2.0 * float(rel_count))

        max_graph_points = max(raw_graph_points.values(), default=0.0)

        ranked_chunks: list[RankedEvidenceChunk] = []
        for cid, chunk in chunks_by_id.items():
            v_score = vector_scores.get(cid, 0.0)
            g_score = (raw_graph_points[cid] / max_graph_points) if max_graph_points > 0.0 else 0.0
            combined = (
                active_config.vector_weight * max(0.0, v_score)
                + active_config.graph_weight * g_score
            )
            ranked_chunks.append(
                RankedEvidenceChunk(
                    chunk=chunk,
                    combined_score=round(combined, 6),
                    vector_score=round(v_score, 6),
                    graph_score=round(g_score, 6),
                    retrieval_sources=tuple(sources_by_id.get(cid, [])),
                    supporting_entities=tuple(entities_by_id.get(cid, [])),
                    supporting_relationships=tuple(relationships_by_id.get(cid, [])),
                )
            )

        ranked_chunks.sort(
            key=lambda item: (
                -item.combined_score,
                -item.vector_score,
                -item.graph_score,
                item.chunk.chunk_id,
            )
        )
        top_chunks = tuple(ranked_chunks[: active_config.max_total_chunks])

        return HybridRetrievalBundle(
            query=query,
            evidence_chunks=top_chunks,
            graph_entities=tuple(entity_evidences),
            graph_relationships=tuple(relationship_evidences),
            config=active_config,
        )
