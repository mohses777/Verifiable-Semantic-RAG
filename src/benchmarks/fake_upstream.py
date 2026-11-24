"""Zero-cost stand-in for the OpenAI API, shaped for the GraphRAG-Bench evaluator's prompts.

Used to test the evaluation plumbing (proxy, embedding substitution, reconciliation) end to end with no network
and no spend. It returns plausible JSON for each judge prompt and deterministic embeddings. If a request's text
contains ``fail_marker`` it answers HTTP 400, which makes the evaluator drop that sample, so the failure path can
be exercised too. Its scores mean nothing and must never be reported as results.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

PROMPT_RESPONSES = (
    ("Given a question and an answer, analyze the complexity", '["statement one", "statement two"]'),
    ("Given a ground truth and an answer statements", json.dumps({
        "TP": [{"statement": "statement one", "reason": "matches"}], "FP": [],
        "FN": [{"statement": "statement two", "reason": "missing"}]})),
    ("You are given a question and a reference answer. Break down", '{"facts": ["fact one", "fact two"]}'),
    ("For each factual statement from the reference, determine", json.dumps({"classifications": [
        {"statement": "fact one", "attributed": 1}, {"statement": "fact two", "attributed": 0}]})),
    ("Your task is to judge the faithfulness", json.dumps([
        {"statement": "statement one", "verdict": 1, "reason": "supported"},
        {"statement": "statement two", "verdict": 0, "reason": "not supported"}])),
)


def _vector(text: str, dims: int = 16) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(digest[i] - 127.5) / 127.5 for i in range(dims)]


class FakeUpstream:
    def __init__(self, fail_marker: str | None = None) -> None:
        self.fail_marker = fail_marker
        self.received: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, path: str, payload: dict[str, Any]) -> tuple[int, bytes]:
        self.received.append((path, payload))
        text = json.dumps(payload)
        if self.fail_marker and self.fail_marker in text:
            return 400, json.dumps({"error": {"message": "fake upstream: forced failure"}}).encode()
        if path.endswith("/embeddings"):
            raw = payload.get("input", "")
            inputs = [raw] if isinstance(raw, str) else list(raw)
            body = {
                "object": "list", "model": payload.get("model"),
                "data": [{"object": "embedding", "index": i, "embedding": _vector(str(t))} for i, t in enumerate(inputs)],
                "usage": {"prompt_tokens": 7 * len(inputs), "total_tokens": 7 * len(inputs)},
            }
            return 200, json.dumps(body).encode()
        prompt = " ".join(str(m.get("content", "")) for m in payload.get("messages", []))
        content = next((out for key, out in PROMPT_RESPONSES if key in prompt), "{}")
        body = {
            "id": "fake", "object": "chat.completion", "created": 0, "model": payload.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }
        return 200, json.dumps(body).encode()
