from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.core.config import (
    ConfigMismatchError,
    LLMSettings,
    UnverifiedParameterError,
    build_run_configuration,
    get_llm_settings,
    load_experiment_config,
    validate_environment,
)
from src.embeddings.client import OpenAIEmbeddingProvider
from src.generation.schemas import GeneratedAnswer
from src.knowledge.extraction import OpenAIStructuredExtractionProvider
from src.knowledge.schemas import ExtractionResult
from src.pipeline.runner import measured_processing_usage
from src.storage.artifacts import load_artifacts

ROOT = Path(__file__).resolve().parents[1]


def run(coro):
    return asyncio.run(coro)


class FakeParseClient:
    """Minimal AsyncOpenAI stand-in recording the kwargs passed to parse()."""

    def __init__(self, parsed, usage) -> None:
        self.calls: list[dict] = []
        parse = self._parse
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=parse)))
        self._parsed, self._usage = parsed, usage

    async def _parse(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(parsed=self._parsed)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=self._usage)


class ConfigTests(unittest.TestCase):
    def test_versioned_config_matches_required_models(self) -> None:
        config = load_experiment_config()
        self.assertEqual(config["llm"]["model"], "gpt-6-luna")
        self.assertEqual(config["embedding"]["model"], "text-embedding-3-large")
        self.assertEqual(config["embedding"]["dimensions"], 3072)
        self.assertEqual(len(config["ragu"]["commit_sha"]), 40)

    def test_unset_request_params_are_omitted_and_set_ones_are_sent(self) -> None:
        self.assertEqual(get_llm_settings().request_params(), {})
        settings = LLMSettings(model="m", temperature=0.0, reasoning_effort="low", seed=7)
        self.assertEqual(
            settings.request_params(), {"temperature": 0.0, "reasoning_effort": "low", "seed": 7}
        )

    def test_environment_mismatch_is_rejected_and_unset_is_allowed(self) -> None:
        validate_environment({})
        validate_environment({"LLM_MODEL": "gpt-6-luna", "OPENAI_BASE_URL": "https://api.openai.com/v1/"})
        with self.assertRaises(ConfigMismatchError):
            validate_environment({"LLM_MODEL": "some-other-model"})
        with self.assertRaises(ConfigMismatchError):
            validate_environment({"EMBEDDING_DIMENSION": "1536"})

    def test_run_configuration_snapshot_records_models_settings_pricing_and_code(self) -> None:
        snapshot = build_run_configuration(
            llm_model="gpt-6-luna",
            embedding_model="text-embedding-3-large",
            embedding_dimensions=3072,
            is_mocked=True,
            chunking={"chunk_size": 1000, "overlap": 200},
            retrieval={"vector_top_k": 5},
        )
        self.assertEqual(snapshot["llm"]["model"], "gpt-6-luna")
        self.assertIsNone(snapshot["llm"]["temperature"])
        self.assertIsNone(snapshot["llm"]["seed"])
        self.assertEqual(snapshot["llm_request_params_sent"], {})
        self.assertEqual(snapshot["embedding"]["dimensions"], 3072)
        self.assertEqual(snapshot["pricing"]["rates"]["gpt-6-luna"], {"input": 0.10, "output": 0.50})
        self.assertIn("git_commit", snapshot["code"])
        self.assertIn("openai", snapshot["environment"]["packages"])
        self.assertEqual(snapshot["ragu"]["commit_sha"], load_experiment_config()["ragu"]["commit_sha"])
        self.assertEqual(snapshot["chunking"]["overlap"], 200)
        self.assertEqual(snapshot["retrieval"]["vector_top_k"], 5)


class ParameterGateTests(unittest.TestCase):
    def _config_with(self, **llm_overrides) -> str:
        config = json.loads(json.dumps(load_experiment_config()))
        for key, value in llm_overrides.items():
            if key.startswith("status_"):
                config["llm"]["parameter_support"][key[len("status_"):]]["status"] = value
            else:
                config["llm"][key] = value
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(config, handle)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return handle.name

    def test_enabling_an_unverified_parameter_is_rejected(self) -> None:
        for name, value in (("temperature", 0.0), ("seed", 7), ("reasoning_effort", "low"), ("max_completion_tokens", 512)):
            with self.assertRaises(UnverifiedParameterError, msg=name):
                get_llm_settings(self._config_with(**{name: value, f"status_{name}": "unverified"}))

    def test_verified_parameter_can_be_enabled_and_is_sent(self) -> None:
        path = self._config_with(temperature=0.0, status_temperature="verified")
        self.assertEqual(get_llm_settings(path).request_params(), {"temperature": 0.0})

    def test_unsupported_parameter_cannot_be_enabled(self) -> None:
        # Shipped config records temperature=0.0 as rejected by the API (probe, 2026-11-24).
        with self.assertRaises(UnverifiedParameterError):
            get_llm_settings(self._config_with(temperature=0.0))

    def test_shipped_config_disables_all_parameters_and_records_probe_outcomes(self) -> None:
        llm = load_experiment_config()["llm"]
        for name in ("temperature", "seed", "reasoning_effort", "max_completion_tokens"):
            self.assertIsNone(llm[name], name)
        support = llm["parameter_support"]
        self.assertEqual(support["temperature"]["status"], "unsupported")
        for name in ("seed", "reasoning_effort", "max_completion_tokens"):
            self.assertEqual(support[name]["status"], "verified", name)
        self.assertEqual(get_llm_settings().request_params(), {})


class ProviderTests(unittest.TestCase):
    def test_extraction_provider_forwards_params_and_accumulates_measured_usage(self) -> None:
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=40, total_tokens=140)
        client = FakeParseClient(ExtractionResult(), usage)
        provider = OpenAIStructuredExtractionProvider(
            api_key="test", client=client, request_params={"temperature": 0.0}
        )
        for _ in range(2):
            run(provider.extract(model="gpt-6-luna", messages=[], schema=ExtractionResult))
        self.assertEqual(client.calls[0]["temperature"], 0.0)
        self.assertEqual(
            provider.total_usage,
            {"calls": 2, "prompt_tokens": 200, "completion_tokens": 80, "total_tokens": 280},
        )
        measured = measured_processing_usage(
            None, provider, embedding_model="text-embedding-3-large", llm_model="gpt-6-luna"
        )
        self.assertTrue(measured["extraction"]["measured"])
        self.assertAlmostEqual(measured["extraction"]["cost_usd"], 200 * 0.10 / 1e6 + 80 * 0.50 / 1e6, places=6)

    def test_default_provider_sends_no_extra_params_so_results_are_unchanged(self) -> None:
        client = FakeParseClient(ExtractionResult(), None)
        provider = OpenAIStructuredExtractionProvider(api_key="test", client=client)
        run(provider.extract(model="gpt-6-luna", messages=[], schema=ExtractionResult))
        self.assertEqual(set(client.calls[0]), {"model", "messages", "response_format"})

    def test_embedding_provider_accumulates_usage_across_batches(self) -> None:
        class FakeEmbeddings:
            async def create(self, **kwargs):
                data = [SimpleNamespace(embedding=[0.0]) for _ in kwargs["input"]]
                n = len(kwargs["input"])
                return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=n * 10, total_tokens=n * 10))

        provider = OpenAIEmbeddingProvider(api_key="t", client=SimpleNamespace(embeddings=FakeEmbeddings()))
        run(provider.embed(["a", "b"], model="text-embedding-3-large"))
        run(provider.embed(["c"], model="text-embedding-3-large"))
        self.assertEqual(provider.total_usage, {"calls": 2, "prompt_tokens": 30, "total_tokens": 30})
        self.assertEqual(provider.last_usage["total_tokens"], 10)


class BackwardCompatibilityTests(unittest.TestCase):
    def test_existing_live_artifacts_without_run_configuration_still_load(self) -> None:
        artifacts = load_artifacts(ROOT / "data" / "processed" / "komarov_2026_ragu")
        self.assertEqual(artifacts.summary.chunk_count, 54)
        self.assertEqual(artifacts.summary.run_configuration, {})


if __name__ == "__main__":
    unittest.main()
