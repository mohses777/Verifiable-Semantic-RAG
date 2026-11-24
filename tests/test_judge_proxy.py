"""Mocked tests for the judge proxy. No network and no paid API calls: upstream is a fake function."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from src.benchmarks.judge_proxy import (
    DEFAULT_POLICY_PATH, JudgeLedger, JudgeProxyServer, handle_request, load_judge_policy,
)
from src.integrations.ragu_metering import UsageMeter

# The exact request shape the official evaluator constructs (per the reviewed source).
EVALUATOR_PAYLOAD = {
    "model": "gpt-6-luna",
    "messages": [{"role": "user", "content": "Break down each sentence into statements."}],
    "temperature": 0.0,
    "top_p": 1,
    "seed": 42,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "stream": False,
}


def ok_body(prompt: int = 100, completion: int = 50) -> bytes:
    return json.dumps({"choices": [{"message": {"content": "{}"}}],
                       "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}).encode()


class Upstream:
    """Fake API recording exactly what would have reached it."""

    def __init__(self, status: int = 200, body: bytes | None = None, raises: Exception | None = None) -> None:
        self.received: list[dict] = []
        self.paths: list[str] = []
        self.status, self.body, self.raises = status, body if body is not None else ok_body(), raises

    def __call__(self, path: str, payload: dict):
        self.paths.append(path)
        self.received.append(payload)
        if self.raises:
            raise self.raises
        return self.status, self.body


def setup(upstream: Upstream | None = None, max_usd: float = 1.0):
    return load_judge_policy(), JudgeLedger(), UsageMeter(max_usd=max_usd), upstream or Upstream()


def call(policy, ledger, meter, upstream, payload=EVALUATOR_PAYLOAD, path="/v1/chat/completions"):
    return handle_request(path, json.dumps(payload).encode(), policy=policy, ledger=ledger, meter=meter, send=upstream)


class PolicyTests(unittest.TestCase):
    def test_temperature_is_omitted_and_every_other_field_is_forwarded_unchanged(self) -> None:
        policy, ledger, meter, upstream = setup()
        status, _ = call(policy, ledger, meter, upstream)
        self.assertEqual(status, 200)
        sent = upstream.received[0]
        self.assertNotIn("temperature", sent)
        expected = {k: v for k, v in EVALUATOR_PAYLOAD.items() if k != "temperature"}
        self.assertEqual(sent, expected)  # nothing else added, removed or altered
        self.assertEqual(ledger.removed_parameters["temperature"], 1)

    def test_request_without_temperature_is_forwarded_as_is(self) -> None:
        policy, ledger, meter, upstream = setup()
        payload = {k: v for k, v in EVALUATOR_PAYLOAD.items() if k != "temperature"}
        call(policy, ledger, meter, upstream, payload)
        self.assertEqual(upstream.received[0], payload)
        self.assertEqual(sum(ledger.removed_parameters.values()), 0)

    def test_only_temperature_is_in_the_omit_list(self) -> None:
        self.assertEqual(load_judge_policy().omit_parameters, ("temperature",))

    def test_reasoning_effort_is_not_injected_or_changed(self) -> None:
        policy, ledger, meter, upstream = setup()
        call(policy, ledger, meter, upstream)
        self.assertNotIn("reasoning_effort", upstream.received[0])

    def test_other_model_is_refused_explicitly_and_never_forwarded(self) -> None:
        policy, ledger, meter, upstream = setup()
        status, body = call(policy, ledger, meter, upstream, {**EVALUATOR_PAYLOAD, "model": "gpt-4o-mini"})
        self.assertEqual(status, 400)
        self.assertEqual(upstream.received, [])
        self.assertEqual(ledger.refused["model"], 1)
        self.assertIn("not the configured judge", json.loads(body)["error"]["message"])

    def test_streaming_and_unknown_paths_and_bad_json_are_refused(self) -> None:
        policy, ledger, meter, upstream = setup()
        self.assertEqual(call(policy, ledger, meter, upstream, {**EVALUATOR_PAYLOAD, "stream": True})[0], 400)
        self.assertEqual(call(policy, ledger, meter, upstream, path="/v1/models")[0], 404)
        status, _ = handle_request("/v1/chat/completions", b"not json", policy=policy, ledger=ledger,
                                   meter=meter, send=upstream)
        self.assertEqual(status, 400)
        self.assertEqual(upstream.received, [])
        self.assertEqual(dict(ledger.refused), {"streaming": 1, "path": 1, "bad_json": 1})

    def test_embeddings_are_forwarded_unchanged_pinned_to_the_configured_model_and_metered(self) -> None:
        body = json.dumps({"data": [{"embedding": [0.1]}], "usage": {"prompt_tokens": 1000, "total_tokens": 1000}}).encode()
        policy, ledger, meter, upstream = setup(Upstream(body=body))
        payload = {"model": "text-embedding-3-large", "input": ["some answer text"]}
        status, _ = call(policy, ledger, meter, upstream, payload, path="/v1/embeddings")
        self.assertEqual(status, 200)
        self.assertEqual(upstream.received[0], payload)  # unchanged
        self.assertEqual(upstream.paths, ["/v1/embeddings"])
        usage = ledger.summary(policy, meter)["usage"]
        self.assertEqual(usage["api_calls"]["embedding"], 1)
        self.assertAlmostEqual(usage["cost_usd"], 1000 * 0.13 / 1e6, places=9)

    def test_wrong_embedding_model_is_refused_and_models_are_not_interchangeable(self) -> None:
        policy, ledger, meter, upstream = setup()
        self.assertEqual(call(policy, ledger, meter, upstream, {"model": "BAAI/bge-large-en-v1.5", "input": ["x"]},
                              path="/v1/embeddings")[0], 400)
        self.assertEqual(call(policy, ledger, meter, upstream, {"model": "gpt-6-luna", "input": ["x"]},
                              path="/v1/embeddings")[0], 400)
        self.assertEqual(call(policy, ledger, meter, upstream, {**EVALUATOR_PAYLOAD, "model": "text-embedding-3-large"})[0], 400)
        self.assertEqual(upstream.received, [])
        self.assertEqual(ledger.refused["model"], 3)

    def test_policy_model_must_match_configured_model(self) -> None:
        data = json.loads(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
        data["judge_model"] = "gpt-4o-mini"
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(data, handle)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        with self.assertRaises(ValueError):
            load_judge_policy(handle.name)

    def test_policy_hash_is_stable(self) -> None:
        self.assertEqual(load_judge_policy().policy_hash, load_judge_policy().policy_hash)
        self.assertEqual(len(load_judge_policy().policy_hash), 64)


class LedgerAndBudgetTests(unittest.TestCase):
    def test_failed_upstream_responses_are_counted_not_dropped(self) -> None:
        body = json.dumps({"error": {"message": "Unsupported value: 'top_p'"}}).encode()
        policy, ledger, meter, upstream = setup(Upstream(status=400, body=body))
        status, _ = call(policy, ledger, meter, upstream)
        self.assertEqual(status, 400)
        summary = ledger.summary(policy, meter)
        self.assertEqual(summary["failed_request_count"], 1)
        self.assertEqual(summary["upstream_status"], {"400": 1})
        self.assertIn("top_p", summary["failed_requests"][0]["error"])

    def test_network_errors_are_counted_and_returned_as_502(self) -> None:
        policy, ledger, meter, upstream = setup(Upstream(raises=ConnectionError("down")))
        status, _ = call(policy, ledger, meter, upstream)
        self.assertEqual(status, 502)
        self.assertEqual(ledger.summary(policy, meter)["failed_requests"], [{"status": None, "error_type": "ConnectionError"}])

    def test_usage_is_metered_and_cap_refuses_later_calls_before_upstream(self) -> None:
        # One response costs 1,000,000 prompt tokens x $0.10/M = $0.10; cap is $0.15.
        policy, ledger, meter, upstream = setup(Upstream(body=ok_body(prompt=1_000_000, completion=0)), max_usd=0.15)
        statuses = [call(policy, ledger, meter, upstream)[0] for _ in range(4)]
        self.assertEqual(statuses, [200, 200, 402, 402])
        self.assertEqual(len(upstream.received), 2)  # refused calls never reach the API
        self.assertEqual(ledger.refused["budget"], 2)
        self.assertTrue(ledger.summary(policy, meter)["usage"]["budget_exceeded"])


class EndToEndLoopbackTests(unittest.TestCase):
    """Real HTTP through the proxy, including the OpenAI SDK the evaluator uses, with a fake upstream."""

    def setUp(self) -> None:
        self.upstream = Upstream()
        self.policy, self.ledger, self.meter = load_judge_policy(), JudgeLedger(), UsageMeter(max_usd=1.0)
        self.server = JudgeProxyServer(self.policy, self.ledger, self.meter, self.upstream)
        self.server.start()
        self.addCleanup(self.server.stop)

    def test_raw_http_request_reaches_upstream_without_temperature(self) -> None:
        response = httpx.post(self.server.base_url + "/chat/completions", json=EVALUATOR_PAYLOAD,
                              headers={"Authorization": "Bearer dummy-evaluator-key"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("temperature", self.upstream.received[0])
        self.assertNotIn("dummy-evaluator-key", json.dumps(self.upstream.received))  # headers never forwarded

    def test_openai_sdk_call_like_the_evaluator_reaches_upstream_without_temperature(self) -> None:
        from openai import AsyncOpenAI

        async def go():
            client = AsyncOpenAI(api_key="dummy", base_url=self.server.base_url, max_retries=0)
            return await client.chat.completions.create(
                model="gpt-6-luna", messages=EVALUATOR_PAYLOAD["messages"], temperature=0.0, top_p=1,
                seed=42, presence_penalty=0, frequency_penalty=0,
            )

        completion = asyncio.run(go())
        self.assertEqual(completion.usage.prompt_tokens, 100)
        sent = self.upstream.received[0]
        self.assertNotIn("temperature", sent)
        self.assertEqual({k: sent[k] for k in ("top_p", "seed", "presence_penalty", "frequency_penalty")},
                         {"top_p": 1, "seed": 42, "presence_penalty": 0, "frequency_penalty": 0})
        summary = self.ledger.summary(self.policy, self.meter)
        self.assertEqual(summary["removed_parameters"], {"temperature": 1})
        self.assertEqual(summary["models_seen"], {"gpt-6-luna": 1})


if __name__ == "__main__":
    unittest.main()
