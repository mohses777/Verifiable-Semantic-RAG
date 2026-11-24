from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from src.integrations.ragu_metering import BudgetExceededError, MeteredAsyncOpenAI, UsageMeter


def usage(prompt: int = 0, completion: int = 0, total: int = 0, reasoning: int = 0):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=total,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )


class FakeClient:
    """Counts requests that actually reach 'the API'."""

    def __init__(self, chat_usage, embed_usage) -> None:
        self.reached = 0
        outer = self

        async def create(**kwargs):
            outer.reached += 1
            return SimpleNamespace(usage=chat_usage)

        async def parse(**kwargs):
            outer.reached += 1
            return SimpleNamespace(usage=chat_usage)

        async def embed(**kwargs):
            outer.reached += 1
            return SimpleNamespace(usage=embed_usage)

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
        self.embeddings = SimpleNamespace(create=embed)


def run(coro):
    return asyncio.run(coro)


class MeteringTests(unittest.TestCase):
    def test_measured_usage_and_cost_for_chat_parse_and_embeddings(self) -> None:
        fake = FakeClient(usage(prompt=1000, completion=500, reasoning=200), usage(total=2000))
        meter = UsageMeter(max_usd=1.0)
        client = MeteredAsyncOpenAI(fake, meter)
        run(client.chat.completions.create(model="gpt-6-luna", messages=[{"role": "user", "content": "abcd"}]))
        run(client.beta.chat.completions.parse(model="gpt-6-luna", messages=[{"role": "user", "content": "xy"}]))
        run(client.embeddings.create(model="text-embedding-3-large", input=["aa", "bbb"]))
        summary = meter.summary()
        self.assertEqual(summary["api_calls"], {"chat": 2, "embedding": 1})
        self.assertEqual(summary["tokens"], {"prompt": 2000, "completion": 1000, "reasoning": 400, "embedding": 2000})
        expected = 2 * (1000 * 0.10 + 500 * 0.50) / 1e6 + 2000 * 0.13 / 1e6
        self.assertAlmostEqual(summary["cost_usd"], expected, places=6)
        self.assertEqual(summary["request_chars"], {"chat": 6, "embedding": 5})
        self.assertEqual(
            summary["endpoints"],
            {"chat.completions.create": 1, "beta.chat.completions.parse": 1, "embeddings.create": 1},
        )

    def test_cap_blocks_further_calls_before_they_reach_the_api(self) -> None:
        fake = FakeClient(usage(prompt=1_000_000, completion=0), usage(total=0))  # one call costs $0.10
        meter = UsageMeter(max_usd=0.15)
        client = MeteredAsyncOpenAI(fake, meter)
        for _ in range(2):
            run(client.chat.completions.create(model="gpt-6-luna", messages=[]))
        self.assertTrue(meter.exceeded)
        reached_at_cap = fake.reached
        with self.assertRaises(BudgetExceededError):
            run(client.chat.completions.create(model="gpt-6-luna", messages=[]))
        with self.assertRaises(BudgetExceededError):
            run(client.embeddings.create(model="text-embedding-3-large", input=["x"]))
        self.assertEqual(fake.reached, reached_at_cap)  # refused calls never hit the API
        self.assertEqual(meter.summary()["blocked_calls_after_cap"], 2)
        self.assertTrue(meter.summary()["budget_exceeded"])

    def test_unconfigured_model_is_refused_before_any_request(self) -> None:
        fake = FakeClient(usage(), usage())
        client = MeteredAsyncOpenAI(fake, UsageMeter(max_usd=1.0))
        with self.assertRaises(KeyError):
            run(client.chat.completions.create(model="gpt-4o-mini", messages=[]))
        self.assertEqual(fake.reached, 0)

    def test_invalid_cap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            UsageMeter(max_usd=0)


if __name__ == "__main__":
    unittest.main()
