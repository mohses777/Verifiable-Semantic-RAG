"""Local proxy that applies the documented judge policy to GraphRAG-Bench evaluator requests.

Carries chat completions (omitting only the configured parameters) and embeddings (text-embedding-3-large
replaces the evaluator's local bge-large model), both metered under one spend cap.

The official evaluator is run UNMODIFIED with ``--base_url http://127.0.0.1:<port>/v1``. This proxy:

* omits exactly the parameters listed in ``config/judge_policy.json`` (currently only ``temperature``)
  and forwards every other field unchanged;
* refuses requests for any model other than the configured judge (no silent substitution), streaming
  requests, and any path other than chat completions;
* counts every request, refusal and failed upstream call in a ledger (nothing is dropped silently);
* meters API-reported usage and refuses all calls after a hard spend cap;
* holds the real API key itself, so the evaluator process never sees it.

The same policy applies to every system being scored, and the policy hash is recorded in the ledger.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from src.core.config import load_experiment_config
from src.core.redaction import redact_secrets
from src.integrations.ragu_metering import BudgetExceededError, UsageMeter

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "judge_policy.json"
Sender = Callable[[str, dict[str, Any]], tuple[int, bytes]]


@dataclass(frozen=True)
class JudgePolicy:
    judge_model: str
    embedding_model: str
    omit_parameters: tuple[str, ...]
    allowed_paths: tuple[str, ...]
    reject_streaming: bool
    refuse_status_code: int
    policy_hash: str


def load_judge_policy(path: str | Path = DEFAULT_POLICY_PATH) -> JudgePolicy:
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    experiment = load_experiment_config()
    configured = experiment["llm"]["model"]
    if data["embedding_model"] != experiment["embedding"]["model"]:
        raise ValueError(
            f"Judge policy embedding model '{data['embedding_model']}' differs from the configured model "
            f"'{experiment['embedding']['model']}'. Change config/experiment.json deliberately."
        )
    if data["judge_model"] != configured:
        raise ValueError(
            f"Judge policy model '{data['judge_model']}' differs from the configured model '{configured}'. "
            "Change config/experiment.json deliberately; the judge is never substituted silently."
        )
    return JudgePolicy(
        judge_model=data["judge_model"],
        embedding_model=data["embedding_model"],
        omit_parameters=tuple(data["omit_parameters"]),
        allowed_paths=tuple(data["allowed_paths"]),
        reject_streaming=bool(data["reject_streaming"]),
        refuse_status_code=int(data["refuse_status_code"]),
        policy_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


class JudgeLedger:
    """Thread-safe record of every request the proxy saw."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests_total = 0
        self.forwarded = 0
        self.forwarded_paths: Counter[str] = Counter()
        self.refused: Counter[str] = Counter()
        self.upstream_status: Counter[int] = Counter()
        self.removed_parameters: Counter[str] = Counter()
        self.forwarded_parameter_keys: Counter[str] = Counter()
        self.models_seen: Counter[str] = Counter()
        self.failed_requests: list[dict[str, Any]] = []

    def summary(self, policy: JudgePolicy, meter: UsageMeter) -> dict[str, Any]:
        with self.lock:
            return {
                "policy_hash": policy.policy_hash,
                "judge_model": policy.judge_model,
                "omit_parameters": list(policy.omit_parameters),
                "requests_total": self.requests_total,
                "forwarded_to_api": self.forwarded,
                "forwarded_paths": dict(self.forwarded_paths),
                "embedding_model": policy.embedding_model,
                "refused": dict(self.refused),
                "upstream_status": {str(k): v for k, v in self.upstream_status.items()},
                "failed_request_count": len(self.failed_requests),
                "failed_requests": list(self.failed_requests),
                "removed_parameters": dict(self.removed_parameters),
                "forwarded_parameter_keys": dict(self.forwarded_parameter_keys),
                "models_seen": dict(self.models_seen),
                "usage": meter.summary(),
            }


def _error(status: int, message: str, kind: str) -> tuple[int, bytes]:
    return status, json.dumps({"error": {"message": message, "type": kind}}).encode("utf-8")


def handle_request(
    path: str,
    body: bytes,
    *,
    policy: JudgePolicy,
    ledger: JudgeLedger,
    meter: UsageMeter,
    send: Sender,
) -> tuple[int, bytes]:
    """Apply the policy to one request. ``send`` receives only the sanitized payload (never headers)."""
    with ledger.lock:
        ledger.requests_total += 1

    def refuse(reason: str, status: int, message: str) -> tuple[int, bytes]:
        with ledger.lock:
            ledger.refused[reason] += 1
        return _error(status, message, f"proxy_refused_{reason}")

    if path not in policy.allowed_paths:
        return refuse("path", 404, f"Path not allowed by judge policy: {path}")
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("body is not a JSON object")
    except ValueError as error:
        return refuse("bad_json", 400, f"Invalid request body: {error}")
    if policy.reject_streaming and payload.get("stream"):
        return refuse("streaming", 400, "Streaming requests are not supported by the judge proxy.")
    model = payload.get("model")
    is_embeddings = path.endswith("/embeddings")
    expected_model = policy.embedding_model if is_embeddings else policy.judge_model
    with ledger.lock:
        ledger.models_seen[str(model)] += 1
    if model != expected_model:
        role = "embedding model" if is_embeddings else "judge"
        return refuse("model", 400, f"Model '{model}' is not the configured {role} '{expected_model}'.")
    try:
        meter.before_call(model)
    except BudgetExceededError as error:
        return refuse("budget", policy.refuse_status_code, str(error))

    removed = [name for name in policy.omit_parameters if name in payload]
    forwarded = {key: value for key, value in payload.items() if key not in policy.omit_parameters}
    with ledger.lock:
        ledger.forwarded += 1
        ledger.forwarded_paths[path] += 1
        ledger.removed_parameters.update(removed)
        ledger.forwarded_parameter_keys.update(forwarded.keys())

    try:
        status, response_body = send(path, forwarded)
    except Exception as error:  # network failure etc.: count it, never drop it silently
        with ledger.lock:
            ledger.failed_requests.append({"status": None, "error_type": type(error).__name__})
        return _error(502, f"Upstream request failed: {type(error).__name__}", "proxy_upstream_error")

    with ledger.lock:
        ledger.upstream_status[status] += 1
    if 200 <= status < 300:
        try:
            usage = json.loads(response_body).get("usage") or {}
            if is_embeddings:
                meter.record("embedding", model, SimpleNamespace(total_tokens=usage.get("total_tokens", 0)),
                             len(str(forwarded.get("input", ""))))
            else:
                details = usage.get("completion_tokens_details") or {}
                meter.record(
                    "chat",
                    model,
                    SimpleNamespace(
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        completion_tokens_details=SimpleNamespace(reasoning_tokens=details.get("reasoning_tokens", 0)),
                    ),
                    sum(len(str(m.get("content", ""))) for m in forwarded.get("messages", []) if isinstance(m, dict)),
                )
        except ValueError:
            with ledger.lock:
                ledger.failed_requests.append({"status": status, "error_type": "unparseable_success_body"})
    else:
        message = ""
        try:
            message = str(json.loads(response_body).get("error", {}).get("message", ""))
        except ValueError:
            pass
        with ledger.lock:
            ledger.failed_requests.append({"status": status, "error": redact_secrets(message)[:300]})
    return status, response_body


def make_openai_sender(base_url: str, api_key: str, timeout: float = 120.0) -> Sender:
    """Real upstream sender. The key is added here and is never logged or returned."""
    import httpx

    client = httpx.Client(timeout=timeout)
    root = base_url.rstrip("/")

    def send(path: str, payload: dict[str, Any]) -> tuple[int, bytes]:
        suffix = path[len("/v1"):] if path.startswith("/v1") else path
        response = client.post(root + suffix, json=payload, headers={"Authorization": f"Bearer {api_key}"})
        return response.status_code, response.content

    return send


class JudgeProxyServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, policy: JudgePolicy, ledger: JudgeLedger, meter: UsageMeter, send: Sender,
                 host: str = "127.0.0.1", port: int = 0) -> None:
        super().__init__((host, port), _Handler)
        self.policy, self.ledger, self.meter, self.send = policy, ledger, meter, send
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> None:
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.shutdown()
        self.server_close()
        if self._thread:
            self._thread.join()


class _Handler(http.server.BaseHTTPRequestHandler):
    server: JudgeProxyServer

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        path = self.path.split("?", 1)[0]
        status, response = handle_request(
            path, body, policy=self.server.policy, ledger=self.server.ledger,
            meter=self.server.meter, send=self.server.send,
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - keep stdout quiet
        return
