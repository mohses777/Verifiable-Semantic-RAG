"""Usage metering and a hard spend cap for OpenAI clients used by RAGU.

RAGU does not report token usage or cost. This wrapper sits between RAGU's
``CachedAsyncOpenAI`` and the real ``AsyncOpenAI`` client, so every request that
actually reaches the API is counted (RAGU cache hits never reach it). Once the
configured cap is reached, every further call is refused *before* any request is
sent, so a build cannot keep spending even if RAGU swallows the error.

Records API-reported usage only; unknown models raise rather than being priced
by guesswork.
"""

from __future__ import annotations

from typing import Any

from src.core.pricing import get_model_pricing


class BudgetExceededError(RuntimeError):
    """Raised for any API call attempted after the spend cap was reached."""


class UsageMeter:
    def __init__(self, *, max_usd: float) -> None:
        if max_usd <= 0:
            raise ValueError("max_usd must be greater than zero")
        self.max_usd = max_usd
        self.cost_usd = 0.0
        self.exceeded = False
        self.blocked_calls = 0
        self.calls: dict[str, int] = {"chat": 0, "embedding": 0}
        self.tokens: dict[str, int] = {"prompt": 0, "completion": 0, "reasoning": 0, "embedding": 0}
        self.request_chars: dict[str, int] = {"chat": 0, "embedding": 0}
        self.endpoints: dict[str, int] = {}
        self.errors: list[str] = []

    def before_call(self, model: str) -> None:
        if self.exceeded:
            self.blocked_calls += 1
            raise BudgetExceededError(f"Spend cap ${self.max_usd:.4f} reached; call refused.")
        get_model_pricing(model)  # unknown model: raise before any spend

    def record(self, kind: str, model: str, usage: Any, request_chars: int) -> None:
        pricing = get_model_pricing(model)
        self.calls[kind] += 1
        self.request_chars[kind] += request_chars
        if kind == "embedding":
            tokens = int(getattr(usage, "total_tokens", 0) or 0)
            self.tokens["embedding"] += tokens
            self.cost_usd += pricing.input_cost(tokens)
        else:
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
            details = getattr(usage, "completion_tokens_details", None)
            self.tokens["prompt"] += prompt
            self.tokens["completion"] += completion
            self.tokens["reasoning"] += int(getattr(details, "reasoning_tokens", 0) or 0)
            self.cost_usd += pricing.input_cost(prompt) + pricing.output_cost(completion)
        if self.cost_usd >= self.max_usd:
            self.exceeded = True

    def summary(self) -> dict[str, Any]:
        return {
            "measured": True,
            "max_usd": self.max_usd,
            "cost_usd": round(self.cost_usd, 6),
            "budget_exceeded": self.exceeded,
            "blocked_calls_after_cap": self.blocked_calls,
            "api_calls": dict(self.calls),
            "tokens": dict(self.tokens),
            "request_chars": dict(self.request_chars),
            "endpoints": dict(self.endpoints),
            "errors": list(self.errors),
        }


def _request_chars(kind: str, kwargs: dict[str, Any]) -> int:
    if kind == "embedding":
        value = kwargs.get("input", "")
        return len(value) if isinstance(value, str) else sum(len(str(item)) for item in value)
    return sum(len(str(message.get("content", ""))) for message in kwargs.get("messages", []))


class _MeteredMethod:
    def __init__(self, resource: Any, method: str, kind: str, meter: UsageMeter, label: str) -> None:
        self._resource, self._method, self._kind, self._meter = resource, method, kind, meter
        self._label = label

    async def _call(self, **kwargs: Any) -> Any:
        model = kwargs.get("model", "")
        self._meter.before_call(model)
        try:
            response = await getattr(self._resource, self._method)(**kwargs)
        except Exception as error:
            self._meter.errors.append(f"{type(error).__name__}")
            raise
        self._meter.endpoints[self._label] = self._meter.endpoints.get(self._label, 0) + 1
        self._meter.record(self._kind, model, getattr(response, "usage", None), _request_chars(self._kind, kwargs))
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._resource, name)


class _Namespace:
    def __init__(self, **attrs: Any) -> None:
        self.__dict__.update(attrs)


class MeteredAsyncOpenAI:
    """Drop-in for ``AsyncOpenAI`` as used by RAGU: chat.create, beta.parse, embeddings.create."""

    def __init__(self, client: Any, meter: UsageMeter) -> None:
        self._client = client
        self.meter = meter
        chat_create = _MeteredMethod(client.chat.completions, "create", "chat", meter, "chat.completions.create")
        beta_parse = _MeteredMethod(client.beta.chat.completions, "parse", "chat", meter, "beta.chat.completions.parse")
        embeddings = _MeteredMethod(client.embeddings, "create", "embedding", meter, "embeddings.create")
        self.chat = _Namespace(completions=_Namespace(create=chat_create._call))
        self.beta = _Namespace(chat=_Namespace(completions=_Namespace(parse=beta_parse._call)))
        self.embeddings = _Namespace(create=embeddings._call)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
