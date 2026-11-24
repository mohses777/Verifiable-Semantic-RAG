"""Offline (zero-cost) analysis of how fragile verbatim quoting is on a persisted document.

Counts PDF-text artefacts that make exact quoting fragile and checks how the strict
citation rule treats a historical result's quotes if a model had 'cleaned' line-break
artefacts. This supports or weakens a hypothesis only; it cannot identify the cause of
a specific past failure.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation.diagnostics import diagnose_rejected_quote, is_quote_supported  # noqa: E402
from src.storage.artifacts import load_artifacts  # noqa: E402

FEATURES = {
    "hyphen_linebreak": lambda t: len(re.findall(r"\w-\n\w", t)),
    "hard_newlines": lambda t: t.count("\n"),
    "non_ascii_chars": lambda t: sum(1 for c in t if ord(c) > 127),
    "unicode_dash_or_minus": lambda t: len(re.findall("[‐-―−]", t)),
    "nfkc_changes_text": lambda t: int(unicodedata.normalize("NFKC", t) != t),
    "double_spaces": lambda t: len(re.findall(r"  +", t)),
}


def _clean(quote: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", quote).replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="data/processed/komarov_2026_ragu")
    parser.add_argument("--result", default="results/live_qa_ragu_q1.json", help="historical QA result to test")
    parser.add_argument("--output", default="results/quote_fragility_analysis.json")
    args = parser.parse_args()

    artifacts = load_artifacts(ROOT / args.artifacts)
    chunks = artifacts.chunks
    report: dict = {
        "is_mocked": False,
        "api_calls": 0,
        "artifacts": args.artifacts,
        "chunk_count": len(chunks),
        "features": {
            name: {
                "chunks_with_at_least_one": sum(1 for c in chunks if fn(c.text)),
                "total": sum(fn(c.text) for c in chunks),
            }
            for name, fn in FEATURES.items()
        },
    }

    by_id = {c.chunk_id: c.text for c in chunks}
    result = json.loads((ROOT / args.result).read_text(encoding="utf-8"))
    sensitivity = []
    for ref in result["evidence_references"]:
        quote, text = ref["quote"], by_id[ref["chunk_id"]]
        cleaned = _clean(quote)
        accepted_as_written = is_quote_supported(quote, text)
        accepted_cleaned = is_quote_supported(cleaned, text)
        entry = {
            "chunk_id": ref["chunk_id"],
            "quote_has_hyphen_linebreak": bool(re.search(r"\w-\n\w", quote)),
            "strict_accepted_as_written": accepted_as_written,
            "strict_accepted_if_line_breaks_cleaned": accepted_cleaned,
        }
        if not accepted_cleaned:
            diagnosis = diagnose_rejected_quote(cleaned, text)
            entry["cleaned_diagnosis"] = diagnosis["diagnosis"]
            entry["cleaned_category"] = diagnosis["category"]
        sensitivity.append(entry)
    report["historical_result"] = args.result
    report["sensitivity"] = sensitivity

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
