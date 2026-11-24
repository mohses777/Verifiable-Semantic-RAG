"""Versioned experiment configuration and per-run configuration snapshots."""

from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

from src.core.pricing import get_model_pricing

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG_PATH = PROJECT_ROOT / "config" / "experiment.json"
TRACKED_PACKAGES = ("openai", "pydantic", "networkx", "numpy", "pypdf")


class ConfigMismatchError(ValueError):
    """Raised when environment variables disagree with the versioned configuration."""


class UnverifiedParameterError(ValueError):
    """Raised when a request parameter is enabled before its API support is verified."""


LLM_REQUEST_PARAMETERS = ("temperature", "reasoning_effort", "max_completion_tokens", "seed")


@lru_cache(maxsize=None)
def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_experiment_config(path: str | Path = EXPERIMENT_CONFIG_PATH) -> dict[str, Any]:
    return _load(str(Path(path)))


@dataclass(frozen=True)
class LLMSettings:
    """LLM request settings. ``None`` means the parameter is not sent to the API."""

    model: str
    temperature: float | None = None
    reasoning_effort: str | None = None
    max_completion_tokens: int | None = None
    seed: int | None = None

    def request_params(self) -> dict[str, Any]:
        """Extra keyword arguments for the API call; omits unset values."""
        params: dict[str, Any] = {}
        for name in ("temperature", "reasoning_effort", "max_completion_tokens", "seed"):
            value = getattr(self, name)
            if value is not None:
                params[name] = value
        return params

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_completion_tokens,
            "seed": self.seed,
        }


def get_llm_settings(path: str | Path = EXPERIMENT_CONFIG_PATH) -> LLMSettings:
    llm = load_experiment_config(path)["llm"]
    support = llm.get("parameter_support", {})
    for name in LLM_REQUEST_PARAMETERS:
        if llm.get(name) is not None and support.get(name, {}).get("status") != "verified":
            raise UnverifiedParameterError(
                f"LLM parameter '{name}' is set in config but its API support is not verified. "
                "Run the approved live probe (scripts/probe_llm_parameters.py), record the result "
                "in parameter_support, then enable it."
            )
    return LLMSettings(
        model=llm["model"],
        temperature=llm.get("temperature"),
        reasoning_effort=llm.get("reasoning_effort"),
        max_completion_tokens=llm.get("max_completion_tokens"),
        seed=llm.get("seed"),
    )


def validate_environment(
    env: Mapping[str, str] | None = None,
    path: str | Path = EXPERIMENT_CONFIG_PATH,
) -> None:
    """Fail loudly if set environment variables contradict the versioned config.

    Unset variables are allowed. This prevents a silent model/provider substitution.
    """
    env = os.environ if env is None else env
    config = load_experiment_config(path)
    expected = {
        "LLM_MODEL": config["llm"]["model"],
        "EMBEDDING_MODEL": config["embedding"]["model"],
        "EMBEDDING_DIMENSION": str(config["embedding"]["dimensions"]),
        "OPENAI_BASE_URL": config["base_url"].rstrip("/"),
    }
    for name, wanted in expected.items():
        actual = env.get(name)
        if actual is None or actual == "":
            continue
        if actual.strip().rstrip("/") != wanted:
            raise ConfigMismatchError(
                f"Environment {name}={actual!r} does not match config/experiment.json value {wanted!r}. "
                "Update the versioned config deliberately; models are never substituted silently."
            )


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _code_version() -> dict[str, Any]:
    commit = _git(["rev-parse", "HEAD"], PROJECT_ROOT)
    status = _git(["status", "--porcelain"], PROJECT_ROOT)
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _ragu_record(config: dict[str, Any]) -> dict[str, Any]:
    ragu = dict(config["ragu"])
    checkout = PROJECT_ROOT / ragu["checkout_path"]
    head = _git(["rev-parse", "HEAD"], checkout) if checkout.is_dir() else None
    ragu["checkout_head"] = head
    ragu["checkout_matches_pin"] = None if head is None else head == ragu["commit_sha"]
    return ragu


def _pricing_record(config: dict[str, Any], models: list[str]) -> dict[str, Any]:
    pricing_path = PROJECT_ROOT / config["pricing_file"]
    raw = _load(str(pricing_path))
    record: dict[str, Any] = {
        "file": config["pricing_file"],
        "source": raw.get("source"),
        "recorded_on": raw.get("recorded_on"),
        "usd_per_tokens": raw.get("per_tokens"),
        "rates": {},
    }
    for model in models:
        try:
            rate = get_model_pricing(model, pricing_path)
            record["rates"][model] = {"input": rate.input, "output": rate.output}
        except KeyError:
            record["rates"][model] = None
    return record


def build_run_configuration(
    *,
    llm_model: str,
    embedding_model: str,
    embedding_dimensions: int,
    is_mocked: bool,
    chunking: Mapping[str, Any] | None = None,
    retrieval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Snapshot of everything needed to defend how a result was produced.

    Model IDs and dimensions come from the objects actually used; the remaining LLM
    request settings come from the versioned config (what the providers send).
    """
    config = load_experiment_config()
    llm = get_llm_settings().to_dict()
    llm["model"] = llm_model
    snapshot: dict[str, Any] = {
        "config_version": config["config_version"],
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "is_mocked": is_mocked,
        "provider": config["provider"],
        "base_url": config["base_url"],
        "llm": llm,
        "llm_request_params_sent": get_llm_settings().request_params(),
        "llm_parameter_support": load_experiment_config()["llm"].get("parameter_support", {}),
        "embedding": {
            "model": embedding_model,
            "dimensions": embedding_dimensions,
            "batch_size": config["embedding"]["batch_size"],
        },
        "pricing": _pricing_record(config, [llm_model, embedding_model]),
        "code": _code_version(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "ragu": _ragu_record(config),
    }
    if chunking is not None:
        snapshot["chunking"] = dict(chunking)
    if retrieval is not None:
        snapshot["retrieval"] = dict(retrieval)
    return snapshot
