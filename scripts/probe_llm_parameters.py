"""Probe whether the configured LLM accepts optional request parameters.

DEFAULT IS A DRY RUN: it prints the planned requests and a worst-case cost
estimate and makes NO API calls. Pass --execute (after cost approval) to send one
tiny request per parameter. Results go to a JSON file; update
config/experiment.json ``llm.parameter_support`` by hand from that record.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import build_run_configuration, load_experiment_config, validate_environment  # noqa: E402
from src.core.pricing import get_model_pricing  # noqa: E402
from src.core.redaction import redact_secrets  # noqa: E402
from src.pipeline.runner import load_dotenv_if_present  # noqa: E402

# One isolated request per parameter, using the value we would actually enable.
PROBES = {
    "temperature": {"temperature": 0.0},
    "seed": {"seed": 12345},
    "reasoning_effort": {"reasoning_effort": "low"},
    "max_completion_tokens": {"max_completion_tokens": 256},
}
MESSAGES = [{"role": "user", "content": "Reply with the single word OK."}]
EST_PROMPT_TOKENS = 20
WORST_CASE_COMPLETION_TOKENS = 2_000  # Q1 used ~1.6-2.0k completion tokens on a short answer


async def _run(execute: bool, output: Path) -> None:
    load_dotenv_if_present()
    validate_environment()
    model = load_experiment_config()["llm"]["model"]
    pricing = get_model_pricing(model)
    worst_each = pricing.input_cost(EST_PROMPT_TOKENS) + pricing.output_cost(WORST_CASE_COMPLETION_TOKENS)
    print(f"Model: {model}; probes: {len(PROBES)}; worst-case cost <= ${worst_each * len(PROBES):.6f} USD")
    for name, params in PROBES.items():
        print(f"  planned: {name} -> {json.dumps({'model': model, **params})}")
    if not execute:
        print("DRY RUN: no API calls made. Re-run with --execute after approving the cost.")
        return

    import os
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL"))
    results = []
    for name, params in PROBES.items():
        started = perf_counter()
        entry: dict = {"parameter": name, "request_params": {"model": model, **params}}
        try:
            response = await client.chat.completions.create(model=model, messages=MESSAGES, **params)
            usage = response.usage
            entry.update(
                accepted=True,
                response_model=response.model,
                finish_reason=response.choices[0].finish_reason,
                system_fingerprint=getattr(response, "system_fingerprint", None),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=round(pricing.input_cost(usage.prompt_tokens) + pricing.output_cost(usage.completion_tokens), 6),
            )
        except Exception as error:  # record the exact API error; never substitute a model
            entry.update(accepted=False, error_type=type(error).__name__, error=redact_secrets(str(error)))
        entry["elapsed_ms"] = round((perf_counter() - started) * 1_000, 1)
        results.append(entry)
        print(f"  {name}: {'accepted' if entry['accepted'] else 'REJECTED - ' + entry['error']}")

    record = {
        "is_mocked": False,
        "run_configuration": build_run_configuration(
            llm_model=model,
            embedding_model=load_experiment_config()["embedding"]["model"],
            embedding_dimensions=load_experiment_config()["embedding"]["dimensions"],
            is_mocked=False,
        ),
        "total_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in results), 6),
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved probe record to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Make the paid API calls (default: dry run)")
    parser.add_argument("--output", default="results/llm_parameter_probe.json")
    args = parser.parse_args()
    asyncio.run(_run(args.execute, Path(args.output)))


if __name__ == "__main__":
    main()
