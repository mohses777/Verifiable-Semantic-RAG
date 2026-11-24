"""Adapter and reconciliation for the official GraphRAG-Bench generation evaluator.

``make_item`` / ``build_items`` turn a system's answers into the evaluator's input format (the evaluator reads
``ground_truth``, whereas the benchmark's question files call it ``answer``). ``reconcile`` compares what was
expected with what the evaluator actually scored, because the evaluator silently drops a sample whose LLM call
fails (it only prints a message) and averages the rest. Any shortfall is reported, never hidden.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

# Mirrors metric_config in Evaluation/generation_eval.py (commit fdbab5959b18c96532580877ffe27d112bccc0ec).
METRIC_CONFIG: dict[str, list[str]] = {
    "Fact Retrieval": ["rouge_score", "answer_correctness"],
    "Complex Reasoning": ["rouge_score", "answer_correctness"],
    "Contextual Summarize": ["answer_correctness", "coverage_score"],
    "Creative Generation": ["answer_correctness", "coverage_score", "faithfulness"],
}


def make_item(question: Mapping[str, Any], generated_answer: str, context: Sequence[str]) -> dict[str, Any]:
    """One evaluator input item. ``context`` is everything the answering model was shown, as a list of strings."""
    qtype = question["question_type"]
    if qtype not in METRIC_CONFIG:
        raise ValueError(f"Question {question['id']} has type '{qtype}', which the evaluator does not score")
    if not isinstance(generated_answer, str) or not generated_answer.strip():
        raise ValueError(f"Question {question['id']}: generated_answer must be a non-empty string")
    if isinstance(context, str) or not all(isinstance(c, str) for c in context):
        raise ValueError(f"Question {question['id']}: context must be a list of strings")
    return {
        "id": question["id"],
        "question": question["question"],
        "ground_truth": question["answer"],
        "generated_answer": generated_answer,
        "context": list(context),
        "question_type": qtype,
    }


def build_items(
    questions: Sequence[Mapping[str, Any]],
    answers: Mapping[str, str],
    contexts: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Items in question order; a missing answer or context is an explicit error, never skipped."""
    missing = [q["id"] for q in questions if q["id"] not in answers or q["id"] not in contexts]
    if missing:
        raise KeyError(f"No answer or context for question ids: {missing}")
    return [make_item(q, answers[q["id"]], contexts[q["id"]]) for q in questions]


def _is_score(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(value)


def reconcile(items: Sequence[Mapping[str, Any]], evaluator_output: Mapping[str, Any]) -> dict[str, Any]:
    """Compare expected samples and metrics with the evaluator's detailed output.

    ``evaluator_output`` is the JSON written with ``--detailed_output``:
    ``{type: {"average_scores": {...}, "detailed": [{"id": ..., "metrics": {...}}, ...]}}``.
    """
    report: dict[str, Any] = {"types": {}, "skipped_items": [], "complete": True}
    by_type: dict[str, list[Mapping[str, Any]]] = {}
    for item in items:
        qtype = item.get("question_type", "Uncategorized")
        if qtype in METRIC_CONFIG:
            by_type.setdefault(qtype, []).append(item)
        else:
            report["skipped_items"].append({"id": item["id"], "question_type": qtype})
            report["complete"] = False

    for qtype, group in by_type.items():
        expected_ids = [item["id"] for item in group]
        result = evaluator_output.get(qtype)
        detailed = result.get("detailed", []) if isinstance(result, Mapping) else []
        returned = {sample["id"]: sample.get("metrics") or {} for sample in detailed}
        missing_ids = [i for i in expected_ids if i not in returned]
        metrics: dict[str, dict[str, Any]] = {}
        for metric in METRIC_CONFIG[qtype]:
            scored = [i for i in expected_ids if i in returned and _is_score(returned[i].get(metric))]
            not_scored = [i for i in expected_ids if i in returned and not _is_score(returned[i].get(metric))]
            metrics[metric] = {
                "expected": len(expected_ids),
                "scored": len(scored),
                "dropped_sample": len(missing_ids),
                "nan_or_missing": len(not_scored),
                "unscored_ids": missing_ids + not_scored,
            }
            if len(scored) != len(expected_ids):
                report["complete"] = False
        report["types"][qtype] = {
            "expected_samples": len(expected_ids),
            "returned_samples": len(expected_ids) - len(missing_ids),
            "missing_ids": missing_ids,
            "metrics": metrics,
        }
    return report
