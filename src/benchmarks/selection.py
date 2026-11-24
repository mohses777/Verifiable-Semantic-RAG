"""Select a corpus window and a stratified question sample for a reduced-cost benchmark run.

Benchmark questions carry paraphrased evidence, so they cannot be located in the corpus by text
matching. Each question's evidence is therefore matched to its most similar corpus chunk by
embedding cosine similarity. A contiguous window of chunks is then chosen that contains as many
well-matched questions per task type as possible, and a seeded stratified sample is drawn from it.
Everything here is deterministic and offline once the embeddings exist.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WindowChoice:
    start_chunk: int  # inclusive
    end_chunk: int  # exclusive
    score: int
    per_type_available: dict[str, int]


def best_matches(chunk_vectors: np.ndarray, evidence_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (best chunk index, cosine similarity) for every evidence vector."""
    chunks = chunk_vectors / np.linalg.norm(chunk_vectors, axis=1, keepdims=True)
    evidence = evidence_vectors / np.linalg.norm(evidence_vectors, axis=1, keepdims=True)
    similarity = evidence @ chunks.T
    index = similarity.argmax(axis=1)
    return index, similarity[np.arange(len(index)), index]


def select_window(
    best_index: Sequence[int],
    best_similarity: Sequence[float],
    question_types: Sequence[str],
    *,
    chunk_count: int,
    window_chunks: int,
    per_type_target: int,
    min_similarity: float,
) -> WindowChoice:
    """Choose the window maximizing sum over types of min(available, per_type_target).

    Ties resolve to the leftmost window, so the result is deterministic.
    """
    if not 0 < window_chunks <= chunk_count:
        raise ValueError("window_chunks must be between 1 and chunk_count")
    types = sorted(set(question_types))
    counts = {t: np.zeros(chunk_count, dtype=int) for t in types}
    for idx, sim, qtype in zip(best_index, best_similarity, question_types):
        if sim >= min_similarity:
            counts[qtype][idx] += 1
    cumulative = {t: np.concatenate([[0], np.cumsum(counts[t])]) for t in types}
    best: WindowChoice | None = None
    for start in range(0, chunk_count - window_chunks + 1):
        end = start + window_chunks
        available = {t: int(cumulative[t][end] - cumulative[t][start]) for t in types}
        score = sum(min(n, per_type_target) for n in available.values())
        if best is None or score > best.score:
            best = WindowChoice(start, end, score, available)
    assert best is not None
    return best


def sample_questions(
    question_ids: Sequence[str],
    question_types: Sequence[str],
    best_index: Sequence[int],
    best_similarity: Sequence[float],
    *,
    window: WindowChoice,
    per_type_target: int,
    min_similarity: float,
    seed: int,
    eligible_ids: set[str] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Seeded stratified sample of question ids inside the window; also reports shortfalls."""
    pools: dict[str, list[str]] = {}
    for qid, qtype, idx, sim in zip(question_ids, question_types, best_index, best_similarity):
        if eligible_ids is not None and qid not in eligible_ids:
            continue
        if window.start_chunk <= idx < window.end_chunk and sim >= min_similarity:
            pools.setdefault(qtype, []).append(qid)
    rng = random.Random(seed)
    chosen: list[str] = []
    shortfall: dict[str, int] = {}
    for qtype in sorted(set(question_types)):
        pool = sorted(pools.get(qtype, []))
        take = min(per_type_target, len(pool))
        chosen.extend(rng.sample(pool, take))
        shortfall[qtype] = per_type_target - take
    return chosen, shortfall


_STOPWORDS = frozenset(
    "a an the of and or to in for with is are was were be been by on as at from that this it its can may might "
    "which who whom their there than these those such into also not no have has had do does did more most other "
    "any all both each".split()
)


def content_tokens(text: str) -> list[str]:
    """Lower-cased content words (no stopwords, at least 3 characters)."""
    return [w for w in re.findall(r"[a-z0-9\-]+", text.lower()) if w not in _STOPWORDS and len(w) > 2]


def min_statement_coverage(evidence: str, words: set[str]) -> float:
    """Lowest share of a statement's content words found in ``words``, over the evidence's statements.

    Evidence is several statements joined by ``;``. A free, rough answerability check: a statement whose words
    are mostly absent from the slice cannot be supported by it.
    """
    coverages = []
    for statement in re.split(r";\s*", evidence):
        tokens = content_tokens(statement)
        if tokens:
            coverages.append(sum(1 for t in tokens if t in words) / len(tokens))
    return min(coverages) if coverages else 1.0


def similarity_deciles(similarities: Sequence[float]) -> list[float]:
    return [round(float(x), 4) for x in np.quantile(np.asarray(similarities), np.linspace(0.1, 0.9, 9))]


def type_counts(question_types: Sequence[str]) -> dict[str, int]:
    return dict(Counter(question_types))
