"""Version-controlled model pricing used for cost estimates and recorded costs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_PRICING_PATH = Path(__file__).resolve().parents[2] / "config" / "pricing.json"


@dataclass(frozen=True)
class ModelPricing:
    """USD rates per ``per_tokens`` tokens for one model."""

    model: str
    input: float
    output: float
    per_tokens: int

    def input_cost(self, tokens: int) -> float:
        return tokens / self.per_tokens * self.input

    def output_cost(self, tokens: int) -> float:
        return tokens / self.per_tokens * self.output


@lru_cache(maxsize=None)
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_model_pricing(model: str, path: str | Path = DEFAULT_PRICING_PATH) -> ModelPricing:
    """Return pricing for ``model``; raise if it is not configured (no silent fallback)."""
    data = _load(str(Path(path)))
    entry = data["models"].get(model)
    if entry is None:
        raise KeyError(
            f"No pricing configured for model '{model}' in {path}. "
            "Add it to config/pricing.json from the official pricing page."
        )
    return ModelPricing(
        model=model,
        input=float(entry["input"]),
        output=float(entry.get("output", 0.0)),
        per_tokens=int(data["per_tokens"]),
    )
