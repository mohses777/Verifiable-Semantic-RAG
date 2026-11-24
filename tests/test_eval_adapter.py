from __future__ import annotations

import re
import unittest
from pathlib import Path

from src.benchmarks.eval_adapter import METRIC_CONFIG, build_items, make_item, reconcile

EVALUATOR_SOURCE = Path(__file__).resolve().parents[1] / "third_party" / "GraphRAG-Benchmark" / "Evaluation" / "generation_eval.py"


def question(qid: str, qtype: str = "Fact Retrieval") -> dict:
    return {"id": qid, "question": f"Q {qid}?", "answer": f"gold {qid}", "question_type": qtype, "evidence": "e"}


def detailed(entries: dict[str, dict]) -> dict:
    """Evaluator --detailed_output shape for one type."""
    return {"average_scores": {}, "detailed": [{"id": i, "metrics": m} for i, m in entries.items()]}


class ItemTests(unittest.TestCase):
    def test_answer_is_mapped_to_ground_truth_and_context_is_a_list(self) -> None:
        item = make_item(question("a"), "my answer", ["ctx one", "ctx two"])
        self.assertEqual(item["ground_truth"], "gold a")
        self.assertNotIn("answer", item)
        self.assertEqual(item["context"], ["ctx one", "ctx two"])
        self.assertEqual(set(item), {"id", "question", "ground_truth", "generated_answer", "context", "question_type"})

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_item(question("a", "Unknown Type"), "x", ["c"])
        with self.assertRaises(ValueError):
            make_item(question("a"), "   ", ["c"])
        with self.assertRaises(ValueError):
            make_item(question("a"), "x", "a single string, not a list")

    def test_missing_answer_or_context_is_an_explicit_error(self) -> None:
        with self.assertRaises(KeyError):
            build_items([question("a"), question("b")], {"a": "x"}, {"a": ["c"], "b": ["c"]})

    def test_items_keep_question_order(self) -> None:
        qs = [question("b"), question("a")]
        items = build_items(qs, {"a": "x", "b": "y"}, {"a": ["c"], "b": ["c"]})
        self.assertEqual([i["id"] for i in items], ["b", "a"])


class ReconcileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.items = [make_item(question(q), "ans", ["c"]) for q in ("a", "b", "c")]

    def test_complete_run(self) -> None:
        scores = {"rouge_score": 0.5, "answer_correctness": 0.7}
        out = {"Fact Retrieval": detailed({"a": scores, "b": scores, "c": scores})}
        report = reconcile(self.items, out)
        self.assertTrue(report["complete"])
        self.assertEqual(report["types"]["Fact Retrieval"]["metrics"]["answer_correctness"]["scored"], 3)

    def test_silently_dropped_sample_is_reported(self) -> None:
        scores = {"rouge_score": 0.5, "answer_correctness": 0.7}
        out = {"Fact Retrieval": detailed({"a": scores, "c": scores})}  # b was dropped by the evaluator
        report = reconcile(self.items, out)
        self.assertFalse(report["complete"])
        fact = report["types"]["Fact Retrieval"]
        self.assertEqual((fact["expected_samples"], fact["returned_samples"], fact["missing_ids"]), (3, 2, ["b"]))
        self.assertEqual(fact["metrics"]["rouge_score"]["dropped_sample"], 1)

    def test_nan_or_missing_metric_counts_as_not_scored(self) -> None:
        out = {"Fact Retrieval": detailed({
            "a": {"rouge_score": 0.5, "answer_correctness": float("nan")},
            "b": {"rouge_score": 0.5},
            "c": {"rouge_score": 0.5, "answer_correctness": 0.9},
        })}
        report = reconcile(self.items, out)
        metric = report["types"]["Fact Retrieval"]["metrics"]["answer_correctness"]
        self.assertEqual((metric["scored"], metric["nan_or_missing"], sorted(metric["unscored_ids"])), (1, 2, ["a", "b"]))
        self.assertFalse(report["complete"])
        self.assertEqual(report["types"]["Fact Retrieval"]["metrics"]["rouge_score"]["scored"], 3)

    def test_unscorable_type_is_reported_and_type_without_output_is_all_missing(self) -> None:
        items = self.items + [{"id": "z", "question": "q", "ground_truth": "g", "generated_answer": "a",
                               "context": ["c"], "question_type": "Mystery"}]
        report = reconcile(items, {})
        self.assertEqual(report["skipped_items"], [{"id": "z", "question_type": "Mystery"}])
        self.assertEqual(report["types"]["Fact Retrieval"]["missing_ids"], ["a", "b", "c"])
        self.assertFalse(report["complete"])


class EvaluatorDriftGuardTests(unittest.TestCase):
    @unittest.skipUnless(EVALUATOR_SOURCE.is_file(), "evaluator source not downloaded")
    def test_metric_config_matches_downloaded_evaluator_source(self) -> None:
        source = EVALUATOR_SOURCE.read_text(encoding="utf-8")
        block = source[source.index("metric_config = {"):source.index("}", source.index("metric_config = {"))]
        parsed = {m.group(1): re.findall(r'"(\w+)"', m.group(2)) for m in re.finditer(r"'([^']+)':\s*\[([^\]]*)\]", block)}
        self.assertEqual(parsed, METRIC_CONFIG)



class InMemoryDatasetTests(unittest.TestCase):
    def test_behaves_like_the_parts_of_datasets_dataset_the_evaluator_uses(self) -> None:
        from src.benchmarks.eval_shims import InMemoryDataset

        data = {"id": ["a", "b", "c"], "contexts": [["x"], ["y", "z"], []], "question": ["q1", "q2", "q3"]}
        dataset = InMemoryDataset.from_dict(data)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset["id"], ["a", "b", "c"])
        self.assertEqual(dataset["contexts"], [["x"], ["y", "z"], []])  # ragged lists preserved
        subset = dataset.select([2, 0])
        self.assertEqual(subset["id"], ["c", "a"])
        data["id"].append("mutated")  # the dataset owns a copy
        self.assertEqual(len(dataset), 3)

    def test_unequal_columns_and_non_string_keys_are_rejected(self) -> None:
        from src.benchmarks.eval_shims import InMemoryDataset

        with self.assertRaises(ValueError):
            InMemoryDataset.from_dict({"a": [1, 2], "b": [1]})
        with self.assertRaises(TypeError):
            InMemoryDataset.from_dict({"a": [1]})[0]


if __name__ == "__main__":
    unittest.main()
