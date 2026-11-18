"""Mocked tests for the project-side RAGU GPT-6 Luna adapter.

The test supplies a narrow fake RAGU module so it can run without installing
RAGU or contacting OpenAI. It verifies the integration boundary, not OpenAI.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from typing import Any, cast


class _Completions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"kind": "chat"}


class _Parse:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        return {"kind": "parsed"}


class _Embeddings:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"kind": "embedding"}


class _Client:
    def __init__(self) -> None:
        self.chat = types.SimpleNamespace(
            completions=_Completions(),
            beta=types.SimpleNamespace(completions=_Parse()),
        )
        self.beta = types.SimpleNamespace(chat=types.SimpleNamespace(completions=self.chat.beta.completions))
        self.embeddings = _Embeddings()


class _FakeCachedAsyncOpenAI:
    def __init__(self, *, client=None, cache=None, **_kwargs) -> None:
        self.client = client or _Client()
        self.cache = cache if cache is not None else {}

    async def chat_completion(self, model_name, conversation, output_schema=str, **kwargs):
        key = repr((model_name, conversation, output_schema.__name__, sorted(kwargs.items())))
        if key in self.cache:
            return self.cache[key]
        response = await self._uncached_chat_completion(
            model_name, conversation, output_schema, **kwargs
        )
        self.cache[key] = response
        return response

    async def _uncached_chat_completion(self, model_name, conversation, output_schema=str, **kwargs):
        if output_schema is str:
            return await self.client.chat.completions.create(model=model_name, messages=conversation, **kwargs)
        if kwargs.get("as_tool"):
            kwargs.pop("as_tool")
            return await self.client.chat.completions.create(
                model=model_name, messages=conversation, tools=[{"type": "function"}],
                tool_choice={"type": "function"}, **kwargs
            )
        return await self.client.beta.chat.completions.parse(
            model=model_name, messages=conversation, response_format=output_schema, **kwargs
        )

    async def embed_text(self, model_name, text):
        return await self.client.embeddings.create(model=model_name, input=text)


def _load_adapter():
    for name in ("ragu", "ragu.models", "ragu.models.openai"):
        sys.modules.pop(name, None)
    ragu = types.ModuleType("ragu")
    models = types.ModuleType("ragu.models")
    openai = cast(Any, types.ModuleType("ragu.models.openai"))
    openai.CachedAsyncOpenAI = _FakeCachedAsyncOpenAI
    sys.modules.update({"ragu": ragu, "ragu.models": models, "ragu.models.openai": openai})
    path = Path(__file__).parents[1] / "src" / "integrations" / "ragu_openai.py"
    spec = importlib.util.spec_from_file_location("adapter_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load adapter module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RAGUAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = _load_adapter()
        self.raw_client = _Client()
        self.cache = {}
        self.subject = self.adapter.GPT6LunaRAGUClient(
            client=self.raw_client, cache=self.cache, llm_model="gpt-6-luna"
        )
        self.messages = [{"role": "user", "content": "hello"}]

    @staticmethod
    def _run(coroutine):
        """Run these fully mocked coroutines without Windows event-loop I/O."""
        try:
            coroutine.send(None)
        except StopIteration as result:
            return result.value
        raise AssertionError("Mocked coroutine unexpectedly yielded control")

    def test_normal_chat_is_unchanged(self):
        self._run(self.subject.chat_completion("gpt-6-luna", self.messages))
        call = self.raw_client.chat.completions.calls[0]
        self.assertNotIn("reasoning_effort", call)
        self.assertEqual(len(self.cache), 1)

    def test_pydantic_structured_parse_is_unchanged(self):
        class Result:
            __name__ = "Result"
        self._run(self.subject.chat_completion("gpt-6-luna", self.messages, Result))
        call = self.raw_client.beta.chat.completions.calls[0]
        self.assertNotIn("reasoning_effort", call)
        self.assertIs(call["response_format"], Result)

    def test_forced_tool_call_gets_reasoning_effort_and_own_cache_key(self):
        class Result:
            __name__ = "Result"
        self._run(self.subject.chat_completion("gpt-6-luna", self.messages, Result, as_tool=True))
        call = self.raw_client.chat.completions.calls[0]
        self.assertEqual(call["reasoning_effort"], "none")
        self.assertNotIn("_project_reasoning_effort", call)
        self.assertTrue(any("_project_reasoning_effort" in key for key in self.cache))

    def test_embedding_request_is_unchanged(self):
        self._run(self.subject.embed_text("text-embedding-3-large", "evidence"))
        call = self.raw_client.embeddings.calls[0]
        self.assertEqual(call, {"model": "text-embedding-3-large", "input": "evidence"})


if __name__ == "__main__":
    unittest.main()
