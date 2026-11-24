"""Tests for citation rejection diagnostics. Acceptance stays strict and fail-closed."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from types import SimpleNamespace

from src.core.config import LLMSettings
from src.core.config import build_run_configuration
from src.core.models import TextChunk
from src.core.redaction import redact_secrets
from src.generation.answering import (
    EvidenceBackedAnswerGenerator,
    OpenAIStructuredAnswerProvider,
    _validate_with_diagnostics,
)
from src.generation.diagnostics import diagnose_rejected_quote, is_quote_supported
from src.generation.schemas import EvidenceCitation, GeneratedAnswer
from src.retrieval.hybrid import HybridRetrievalBundle, HybridRetrievalConfig, RankedEvidenceChunk

PDF_TEXT = (
    "HippoRAG 2 leads: its\npersonalized PageRank pins down single facts\n"
    "(AC 72.4 vs. 54.2 for RAGU, ∆=−18.2 pp). As tasks demand broad synthe-\nsis,\n"
    "RAGU leads through-\nout (57.4 vs. 34.7 on Creative Generation) and  the gap  closes;\n"
    "andre-\nverseson Creative Generation."
)
OTHER_TEXT = "Table 1: Generation quality on GraphRAG-Bench (Medical domain)."


def chunk(chunk_id: str, text: str, page: int) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id, document_id="doc-real", text=text, source_path="real.pdf",
        source_type="pdf", page_number=page, section_index=page - 1, start_char=0,
        end_char=len(text), document_start_char=0, document_end_char=len(text), metadata={},
    )


def bundle() -> HybridRetrievalBundle:
    items = tuple(
        RankedEvidenceChunk(
            chunk=c, combined_score=1.0, vector_score=1.0, graph_score=0.0,
            retrieval_sources=("vector",), supporting_entities=(), supporting_relationships=(),
        )
        for c in (chunk("c-pdf", PDF_TEXT, 3), chunk("c-other", OTHER_TEXT, 4))
    )
    return HybridRetrievalBundle("q", items, (), (), HybridRetrievalConfig())


def answer(*citations: EvidenceCitation, abstain: bool = False) -> GeneratedAnswer:
    return GeneratedAnswer(
        answer_text="candidate answer", evidence_references=list(citations),
        supporting_chunk_ids=[c.chunk_id for c in citations], insufficient_evidence=abstain,
        confidence_explanation="test",
    )


def cite(quote: str, chunk_id: str = "c-pdf", **extra) -> EvidenceCitation:
    return EvidenceCitation(chunk_id=chunk_id, quote=quote, claim="claim", **extra)


def validate(*citations: EvidenceCitation):
    return _validate_with_diagnostics(answer(*citations), bundle())


def only(diagnostics):
    return diagnostics["citations"][0]


class AcceptanceRuleTests(unittest.TestCase):
    def test_verbatim_pdf_line_break_quote_is_accepted(self) -> None:
        final, diag = validate(cite("RAGU leads through-\nout (57.4 vs. 34.7 on Creative Generation)"))
        self.assertFalse(final.insufficient_evidence)
        self.assertEqual(only(diag)["status"], "accepted")

    def test_multiple_whitespace_and_case_differences_are_accepted(self) -> None:
        final, diag = validate(cite("the   GAP\n\ncloses;"))
        self.assertEqual(only(diag)["status"], "accepted")
        self.assertFalse(final.insufficient_evidence)


class RejectionDiagnosisTests(unittest.TestCase):
    """Formatting-only differences are DIAGNOSED but still REJECTED (fail-closed)."""

    def assert_rejected_as(self, quote: str, diagnosis: str, category: str, chunk_id: str = "c-pdf") -> None:
        final, diag = validate(cite(quote, chunk_id))
        record = only(diag)
        self.assertEqual(record["status"], "rejected", quote)
        self.assertEqual(record["reason"], "quote_not_in_cited_chunk")
        self.assertEqual(record["diagnosis"], diagnosis)
        self.assertEqual(record["category"], category)
        self.assertTrue(final.insufficient_evidence)
        self.assertEqual(final.evidence_references, [])
        self.assertTrue(diag["fail_closed"])

    def test_dehyphenated_pdf_line_break_is_formatting_only_but_rejected(self) -> None:
        self.assert_rejected_as("RAGU leads throughout (57.4 vs. 34.7", "hyphenation_linebreak", "formatting_only")

    def test_unicode_minus_and_delta_variant(self) -> None:
        self.assert_rejected_as("∆=-18.2 pp", "unicode_variant", "formatting_only")

    def test_punctuation_difference(self) -> None:
        self.assert_rejected_as(
            "Table 1 Generation quality on GraphRAG-Bench Medical domain", "punctuation_difference",
            "formatting_only", chunk_id="c-other",
        )

    def test_pdf_spacing_artefact(self) -> None:
        self.assert_rejected_as("and reverses on Creative Generation", "spacing_difference", "formatting_only")

    def test_ellipsis_elision_is_diagnosed_but_rejected(self) -> None:
        self.assert_rejected_as("HippoRAG 2 leads ... single facts", "ellipsis_elision", "elided_quote")

    def test_genuinely_unsupported_quote(self) -> None:
        self.assert_rejected_as(
            "RAGU achieves 99% accuracy on every benchmark ever tested", "unsupported_text", "content_difference"
        )

    def test_partial_overlap_is_content_difference(self) -> None:
        diagnosis = diagnose_rejected_quote(
            "personalized PageRank pins down single facts and also all multi-hop questions perfectly", PDF_TEXT
        )
        self.assertEqual(diagnosis["category"], "content_difference")
        self.assertGreater(diagnosis["best_match_ratio"], 0.4)

    def test_strict_rule_does_not_accept_any_loose_variant(self) -> None:
        for quote in ("RAGU leads throughout", "∆=-18.2 pp", "and reverses on"):
            self.assertFalse(is_quote_supported(quote, PDF_TEXT), quote)


class ChunkAndProvenanceTests(unittest.TestCase):
    def test_invalid_chunk_id_is_rejected_and_reports_where_quote_exists(self) -> None:
        final, diag = validate(cite(OTHER_TEXT, chunk_id="not-a-chunk"))
        record = only(diag)
        self.assertEqual(record["reason"], "invalid_chunk_id")
        self.assertEqual(record["quote_found_in_chunks"], ["c-other"])
        self.assertTrue(final.insufficient_evidence)

    def test_wrong_chunk_id_is_rejected_not_silently_corrected(self) -> None:
        final, diag = validate(cite(OTHER_TEXT, chunk_id="c-pdf"))
        record = only(diag)
        self.assertEqual(record["reason"], "wrong_chunk_id")
        self.assertEqual(record["quote_found_in_chunks"], ["c-other"])
        self.assertTrue(final.insufficient_evidence)

    def test_empty_quote_whitespace_only_is_rejected(self) -> None:
        _, diag = validate(cite("   \n "))
        self.assertEqual(only(diag)["reason"], "empty_quote")

    def test_provenance_is_overwritten_from_chunk_and_override_is_recorded(self) -> None:
        final, diag = validate(
            cite("the gap closes;", document_id="spoofed", source_path="spoofed.pdf", page_number=99)
        )
        citation = final.evidence_references[0]
        self.assertEqual((citation.document_id, citation.source_path, citation.page_number), ("doc-real", "real.pdf", 3))
        record = only(diag)
        self.assertTrue(record["provenance_overridden"])
        self.assertEqual(record["model_supplied_provenance"]["page_number"], 99)
        self.assertEqual(record["authoritative_provenance"]["page_number"], 3)
        self.assertEqual(final.supporting_chunk_ids, ["c-pdf"])

    def test_partially_verified_answer_keeps_only_accepted_citations(self) -> None:
        final, diag = validate(cite("the gap closes;"), cite("totally made up quote text here"))
        self.assertFalse(final.insufficient_evidence)
        self.assertEqual(len(final.evidence_references), 1)
        self.assertEqual((diag["accepted_citation_count"], diag["rejected_citation_count"]), (1, 1))
        self.assertFalse(diag["fail_closed"])
        self.assertEqual(diag["rejection_summary"], {"quote_not_in_cited_chunk:unsupported_text": 1})

    def test_model_abstention_is_not_a_fail_closed_event(self) -> None:
        _, diag = _validate_with_diagnostics(answer(abstain=True), bundle())
        self.assertTrue(diag["raw_insufficient_evidence"])
        self.assertFalse(diag["fail_closed"])


class NoReassignmentTests(unittest.TestCase):
    def test_rejected_citation_is_never_reassigned_to_another_chunk(self) -> None:
        # Quote really exists in c-other but is cited against c-pdf, alongside one valid citation.
        final, diag = validate(cite("the gap closes;"), cite(OTHER_TEXT, chunk_id="c-pdf"))
        self.assertEqual([c.chunk_id for c in final.evidence_references], ["c-pdf"])
        self.assertEqual([c.quote for c in final.evidence_references], ["the gap closes;"])
        self.assertEqual(final.supporting_chunk_ids, ["c-pdf"])
        self.assertNotIn("c-other", final.supporting_chunk_ids)
        self.assertEqual(diag["citations"][1]["reason"], "wrong_chunk_id")
        self.assertEqual(diag["citations"][1]["quote_found_in_chunks"], ["c-other"])

    def test_invalid_chunk_id_only_answer_still_fails_closed_without_reassignment(self) -> None:
        final, diag = validate(cite(OTHER_TEXT, chunk_id="made-up-id"))
        self.assertTrue(final.insufficient_evidence)
        self.assertEqual((final.evidence_references, final.supporting_chunk_ids), ([], []))
        self.assertTrue(diag["fail_closed"])


class NoSecretExposureTests(unittest.TestCase):
    FAKE_KEY = "sk-test-FAKE-SECRET-1234567890abcdef"

    def test_diagnostics_and_snapshot_do_not_contain_the_api_key(self) -> None:
        raw = answer(cite("the gap closes;"))

        class Completions:
            async def parse(self, **kwargs):
                message = SimpleNamespace(parsed=raw, refusal=None)
                choice = SimpleNamespace(message=message, finish_reason="stop")
                usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2, completion_tokens_details=None)
                return SimpleNamespace(choices=[choice], usage=usage, model="gpt-6-luna", system_fingerprint=None)

        client = SimpleNamespace(beta=SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
        provider = OpenAIStructuredAnswerProvider(api_key=self.FAKE_KEY, client=client)
        generator = EvidenceBackedAnswerGenerator(provider, model_id="gpt-6-luna")
        result = asyncio.run(generator.answer_question("q", bundle()))
        snapshot = build_run_configuration(
            llm_model="gpt-6-luna", embedding_model="text-embedding-3-large",
            embedding_dimensions=3072, is_mocked=True,
        )
        serialized = json.dumps({"diagnostics": result.diagnostics, "snapshot": snapshot})
        self.assertNotIn(self.FAKE_KEY, serialized)
        self.assertNotIn("api_key", serialized.lower())
        self.assertEqual(result.diagnostics["request"]["params_sent"], {"model": "gpt-6-luna"})

    def test_redaction_masks_key_like_text_and_secret_env_values(self) -> None:
        message = "Incorrect API key provided: sk-proj-abcd****wxyz. Authorization: Bearer abcdefghijklmnop1234"
        cleaned = redact_secrets(message)
        self.assertNotIn("sk-proj", cleaned)
        self.assertNotIn("abcdefghijklmnop1234", cleaned)
        os.environ["TEST_ONLY_SECRET_TOKEN"] = "plainsecretvalue99"
        self.addCleanup(os.environ.pop, "TEST_ONLY_SECRET_TOKEN", None)
        self.assertNotIn("plainsecretvalue99", redact_secrets("value=plainsecretvalue99"))


class GeneratorDiagnosticsTests(unittest.TestCase):
    def test_result_records_raw_output_and_request_parameters(self) -> None:
        raw = answer(cite("RAGU leads throughout"))

        class Provider:
            last_request_params = {"model": "gpt-6-luna", "temperature": 0.0}
            last_response_meta = {"response_model": "gpt-6-luna-2026", "finish_reason": "stop"}

            async def generate(self, **kwargs):
                return raw

        generator = EvidenceBackedAnswerGenerator(Provider(), model_id="gpt-6-luna")
        result = asyncio.run(generator.answer_question("q", bundle()))
        diag = result.diagnostics
        self.assertTrue(result.answer.insufficient_evidence)
        self.assertEqual(diag["raw_model_output"]["evidence_references"][0]["quote"], "RAGU leads throughout")
        self.assertEqual(diag["request"]["params_sent"]["temperature"], 0.0)
        self.assertEqual(diag["request"]["response"]["response_model"], "gpt-6-luna-2026")
        self.assertEqual(diag["fail_closed_reason"], "model asserted an answer but no citation passed strict verification")


class ReproducibilityControlTests(unittest.TestCase):
    def test_enabled_parameters_appear_in_request_params(self) -> None:
        settings = LLMSettings(
            model="gpt-6-luna", temperature=0.0, reasoning_effort="low", max_completion_tokens=2048, seed=1
        )
        self.assertEqual(
            settings.request_params(),
            {"temperature": 0.0, "reasoning_effort": "low", "max_completion_tokens": 2048, "seed": 1},
        )


if __name__ == "__main__":
    unittest.main()
