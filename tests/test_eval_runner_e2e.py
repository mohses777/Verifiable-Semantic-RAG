"""End-to-end test of the evaluation runner with the REAL downloaded evaluator and a fake API (zero cost).

Skipped unless the evaluator virtual environment and the downloaded evaluator code exist.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = Path(os.environ.get("LOCALAPPDATA", "")) / "verifiable-semantic-rag" / "venv-bench-eval-py314" / "Scripts" / "python.exe"
EVALUATOR = ROOT / "third_party" / "GraphRAG-Benchmark" / "Evaluation" / "generation_eval.py"
TYPES = ["Fact Retrieval", "Complex Reasoning", "Contextual Summarize", "Creative Generation"]


@unittest.skipUnless(VENV_PYTHON.is_file() and EVALUATOR.is_file(), "evaluator environment not installed")
class EvaluationRunnerEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        items = [
            {"id": f"mock-{i}", "question": f"Mock question {i}? " + ("FAILME" if i == 5 else ""),
             "ground_truth": f"Gold {i}.", "generated_answer": f"Answer {i}. A statement.",
             "context": [f"Passage {i}."], "question_type": TYPES[i % 4]}
            for i in range(8)
        ]
        self.items = self.tmp / "items.json"
        self.items.write_text(json.dumps(items), encoding="utf-8")

    def run_runner(self, label: str, *extra: str) -> tuple[int, dict]:
        completed = subprocess.run(
            [str(VENV_PYTHON), str(ROOT / "scripts" / "run_benchmark_eval.py"), "--items", str(self.items),
             "--label", label, "--max-usd", "1", "--fake-upstream", "--output-dir", str(self.tmp / "out"), *extra],
            capture_output=True, text=True, cwd=ROOT, env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=240,
        )
        record = json.loads((self.tmp / "out" / label / "eval_record.json").read_text(encoding="utf-8"))
        return completed.returncode, record

    def test_clean_run_is_complete_and_temperature_never_reaches_the_api(self) -> None:
        code, record = self.run_runner("clean")
        self.assertEqual((code, record["status"]), (0, "complete"))
        check = record["fake_upstream_check"]
        self.assertGreater(check["requests_received"], 8)
        self.assertFalse(check["any_request_contained_temperature"])
        ledger = record["judge_ledger"]
        self.assertEqual(ledger["removed_parameters"], {"temperature": ledger["forwarded_paths"]["/v1/chat/completions"]})
        self.assertEqual(set(ledger["models_seen"]), {"gpt-6-luna", "text-embedding-3-large"})
        self.assertEqual(ledger["failed_request_count"], 0)
        self.assertTrue(record["reconciliation"]["complete"])

    def test_failed_judge_call_is_counted_and_the_dropped_sample_is_reported(self) -> None:
        code, record = self.run_runner("failing", "--fail-marker", "FAILME")
        self.assertEqual((code, record["status"]), (2, "incomplete"))
        self.assertEqual(record["judge_ledger"]["failed_request_count"], 1)
        reasoning = record["reconciliation"]["types"]["Complex Reasoning"]
        self.assertEqual((reasoning["expected_samples"], reasoning["returned_samples"], reasoning["missing_ids"]),
                         (2, 1, ["mock-5"]))
        self.assertFalse(record["reconciliation"]["complete"])
        self.assertFalse(record["fake_upstream_check"]["any_request_contained_temperature"])


if __name__ == "__main__":
    unittest.main()
