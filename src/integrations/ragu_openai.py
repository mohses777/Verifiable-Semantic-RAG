"""Project-side GPT-6 Luna adapter for RAGU's OpenAI client."""

from __future__ import annotations

import os
from typing import Any

# RAGU is intentionally cloned under third_party and installed separately for
# experiments. The import is valid in that environment but unavailable to the
# editor before dependencies are installed.
from ragu.models.openai import CachedAsyncOpenAI  # type: ignore[reportMissingImports]


class GPT6LunaRAGUClient(CachedAsyncOpenAI):
    """Keep RAGU caching while injecting reasoning_effort for forced tools.

    The adapter wraps the configured OpenAI client's Chat Completions resource.
    It lets RAGU construct schemas and cache keys through its normal pathways;
    only outgoing tool-call payloads for the configured LLM model are changed.
    """

    def __init__(self, *args: Any, llm_model: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.llm_model = llm_model
        # Preserve the original client and decorate only chat.completions.create.
        self.client = _ReasoningClientProxy(self.client, llm_model)

    async def chat_completion(
        self,
        model_name: str,
        conversation: list[Any],
        output_schema: type[Any] = str,
        **kwargs: Any,
    ) -> Any:
        """Mark forced GPT-6 Luna tool calls before RAGU builds its cache key."""
        if (
            model_name == self.llm_model
            and output_schema is not str
            and kwargs.get("as_tool") is True
        ):
            # RAGU's cache serializes kwargs. This marker prevents a forced-tool
            # result from sharing a key with a request that has other reasoning
            # semantics, while never reaching the upstream OpenAI wrapper.
            kwargs["_project_reasoning_effort"] = "none"
        return await super().chat_completion(
            model_name=model_name,
            conversation=conversation,
            output_schema=output_schema,
            **kwargs,
        )

    async def _uncached_chat_completion(self, *args: Any, **kwargs: Any) -> Any:
        """Consume the cache-key marker before RAGU validates its kwargs."""
        kwargs.pop("_project_reasoning_effort", None)
        return await super()._uncached_chat_completion(*args, **kwargs)


def client_from_environment(**kwargs: Any) -> GPT6LunaRAGUClient:
    """Create the adapter from the project's environment configuration."""
    return GPT6LunaRAGUClient(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        llm_model=os.environ["LLM_MODEL"],
        **kwargs,
    )


class _ReasoningClientProxy:
    def __init__(self, client: Any, llm_model: str) -> None:
        self._client = client
        self.chat = _ChatProxy(client.chat, llm_model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _ChatProxy:
    def __init__(self, chat: Any, llm_model: str) -> None:
        self._chat = chat
        self.completions = _CompletionsProxy(chat.completions, llm_model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class _CompletionsProxy:
    def __init__(self, completions: Any, llm_model: str) -> None:
        self._completions = completions
        self._llm_model = llm_model

    async def create(self, **kwargs: Any) -> Any:
        tools = kwargs.get("tools")
        forced_function = kwargs.get("tool_choice")
        if (
            kwargs.get("model") == self._llm_model
            and tools
            and forced_function
            and forced_function != "auto"
            and forced_function != "none"
        ):
            kwargs.setdefault("reasoning_effort", "none")
        return await self._completions.create(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)
