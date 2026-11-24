"""Citation verification and rejection diagnostics.

The acceptance rule (``is_quote_supported``) is strict and unchanged: a quote is
accepted only if, after whitespace collapsing and case folding, it is a substring
of the cited chunk. Everything else in this module is *diagnosis only*. Looser
normalizations are used to explain WHY a quote was rejected (formatting artefact
vs unsupported content); they never cause a citation to be accepted.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Mapping

PARTIAL_OVERLAP_THRESHOLD = 0.6
MIN_ELISION_FRAGMENT_CHARS = 8
LONGEST_MATCH_PREVIEW_CHARS = 200

_CHAR_MAP = {
    **dict.fromkeys(map(ord, "‐‑‒–—―−"), "-"),
    **dict.fromkeys(map(ord, "‘’‚‛"), "'"),
    **dict.fromkeys(map(ord, "“”„‟"), '"'),
}
_ELLIPSIS = re.compile(r"\.{3,}|…")

# Diagnoses that indicate a formatting-only difference (text content matches).
FORMATTING_ONLY = (
    "unicode_variant",
    "hyphenation_linebreak",
    "punctuation_difference",
    "spacing_difference",
)


def strict_normalize(text: str) -> str:
    """The acceptance normalization: collapse whitespace and casefold."""
    return " ".join(text.split()).casefold()


def is_quote_supported(quote: str, chunk_text: str) -> bool:
    """Strict acceptance rule (unchanged behaviour)."""
    normalized_quote = strict_normalize(quote)
    if not normalized_quote:
        return False
    return normalized_quote in strict_normalize(chunk_text)


def _unicode_norm(text: str) -> str:
    return strict_normalize(unicodedata.normalize("NFKC", text).translate(_CHAR_MAP))


def _dehyphenate(text: str) -> str:
    # "through- out" (hyphen + collapsed line break) -> "throughout"
    return re.sub(r"(\w)-\s+(\w)", r"\1\2", _unicode_norm(text))


def _strip_punctuation(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", _dehyphenate(text)).split())


def _squash(text: str) -> str:
    return re.sub(r"[\W_]+", "", _dehyphenate(text))


def _longest_match(quote: str, source: str) -> tuple[float, str]:
    if not quote or not source:
        return 0.0, ""
    match = SequenceMatcher(None, quote, source, autojunk=False).find_longest_match(
        0, len(quote), 0, len(source)
    )
    return match.size / len(quote), quote[match.a:match.a + match.size]


def _elision_supported(quote: str, source_norm: str) -> bool:
    if not _ELLIPSIS.search(quote):
        return False
    fragments = [strict_normalize(part) for part in _ELLIPSIS.split(quote)]
    fragments = [f for f in fragments if len(f) >= MIN_ELISION_FRAGMENT_CHARS]
    return bool(fragments) and all(f in source_norm for f in fragments)


def diagnose_rejected_quote(quote: str, chunk_text: str) -> dict[str, Any]:
    """Explain why ``quote`` is not a strict substring of ``chunk_text``.

    Returns the first matching diagnosis in a fixed order from most-benign
    (formatting artefact) to least (unsupported). Diagnosis only; never accepts.
    """
    strict_quote = strict_normalize(quote)
    strict_source = strict_normalize(chunk_text)
    ratio, longest = _longest_match(strict_quote, strict_source)
    result: dict[str, Any] = {
        "normalized_quote": strict_quote,
        "best_match_ratio": round(ratio, 4),
        "longest_matching_text": longest[:LONGEST_MATCH_PREVIEW_CHARS],
    }

    ladder = (
        ("unicode_variant", _unicode_norm),
        ("hyphenation_linebreak", _dehyphenate),
        ("punctuation_difference", _strip_punctuation),
        ("spacing_difference", _squash),
    )
    for name, transform in ladder:
        q, s = transform(quote), transform(chunk_text)
        if q and q in s:
            result.update(diagnosis=name, category="formatting_only")
            return result

    if _elision_supported(quote, strict_source):
        result.update(diagnosis="ellipsis_elision", category="elided_quote")
    elif ratio >= PARTIAL_OVERLAP_THRESHOLD:
        result.update(diagnosis="partial_overlap", category="content_difference")
    else:
        result.update(diagnosis="unsupported_text", category="content_difference")
    return result


def classify_citation(
    citation: Any,
    index: int,
    allowed_texts: Mapping[str, str],
    *,
    authoritative: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the diagnostic record for one model citation.

    ``allowed_texts`` maps retrieved chunk_id -> chunk text. ``authoritative`` is the
    provenance taken from the retrieved chunk (only for accepted citations).
    """
    record: dict[str, Any] = {
        "index": index,
        "chunk_id": citation.chunk_id,
        "quote": citation.quote,
        "claim": citation.claim,
        "model_supplied_provenance": {
            "document_id": citation.document_id,
            "source_path": citation.source_path,
            "page_number": citation.page_number,
        },
    }

    def reject(reason: str, **extra: Any) -> dict[str, Any]:
        record.update(status="rejected", reason=reason, **extra)
        return record

    if not strict_normalize(citation.quote):
        return reject("empty_quote")

    chunk_text = allowed_texts.get(citation.chunk_id)
    if chunk_text is None:
        found_in = [cid for cid, text in allowed_texts.items() if is_quote_supported(citation.quote, text)]
        return reject("invalid_chunk_id", quote_found_in_chunks=found_in)

    if is_quote_supported(citation.quote, chunk_text):
        record.update(status="accepted", reason=None)
        if authoritative is not None:
            record["authoritative_provenance"] = dict(authoritative)
            record["provenance_overridden"] = dict(authoritative) != record["model_supplied_provenance"]
        return record

    elsewhere = [
        cid for cid, text in allowed_texts.items()
        if cid != citation.chunk_id and is_quote_supported(citation.quote, text)
    ]
    if elsewhere:
        return reject("wrong_chunk_id", quote_found_in_chunks=elsewhere)
    return reject("quote_not_in_cited_chunk", **diagnose_rejected_quote(citation.quote, chunk_text))
