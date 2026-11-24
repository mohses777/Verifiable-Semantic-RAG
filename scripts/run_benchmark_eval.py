"""Score one system's answers with the UNMODIFIED official GraphRAG-Bench generation evaluator.

Run with the evaluator virtual environment (it needs langchain; the main environment does not have it):

    <venv>\\Scripts\\python.exe scripts\\run_benchmark_eval.py --items <items.json> --label ours --max-usd 0.20

What this runner does, and the only deviations from the official evaluator (all documented in
docs/benchmark-investigation.md, section 12):
  * judge requests go through the local proxy (config/judge_policy.json): `temperature` is omitted, the judge
    model is pinned, spend is capped, every failed request is counted;
  * the evaluator's local bge-large embedding class is replaced at run time by OpenAI `text-embedding-3-large`,
    also through the proxy (so it is metered and capped);
  * after the run, expected samples and metrics are reconciled with what was actually scored, because the
    evaluator silently drops a sample when a judge call fails. Exit code 2 means the run is incomplete.
The real API key is captured and removed from the environment before the evaluator code runs.
--fake-upstream runs the whole pipeline against a local fake API (zero cost, meaningless scores).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVALUATOR_ROOT = ROOT / "third_party" / "GraphRAG-Benchmark"

from src.benchmarks.eval_adapter import reconcile  # noqa: E402
from src.benchmarks.eval_shims import InMemoryDataset  # noqa: E402
from src.benchmarks.judge_proxy import JudgeLedger, JudgeProxyServer, load_judge_policy, make_openai_sender  # noqa: E402
from src.core.config import build_run_configuration, load_experiment_config, validate_environment  # noqa: E402
from src.integrations.ragu_metering import UsageMeter  # noqa: E402
from src.core.env import load_dotenv_if_present  # noqa: E402

PLACEHOLDER_KEY = "unused-proxy-holds-real-key"


def _embedding_factory(base_url: str, model: str):
    from langchain_openai import OpenAIEmbeddings

    def factory(model_name: str):
        if model_name != model:
            raise ValueError(f"Embedding model '{model_name}' requested; only the configured '{model}' is allowed")
        # Raw strings (no client-side tokenization), sent to the proxy, which meters and pins the model.
        return OpenAIEmbeddings(model=model, base_url=base_url, api_key=PLACEHOLDER_KEY,
                                check_embedding_ctx_length=False, max_retries=0)

    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", required=True, help="Evaluator input JSON (list of items)")
    parser.add_argument("--label", required=True, help="System being scored, for example ours or ragu")
    parser.add_argument("--max-usd", type=float, required=True, help="Hard spend cap for judge + embeddings")
    parser.add_argument("--output-dir", default="results/benchmark_eval")
    parser.add_argument("--fake-upstream", action="store_true", help="Zero-cost plumbing test; scores are meaningless")
    parser.add_argument("--fail-marker", default=None, help="With --fake-upstream: text that forces a 400")
    parser.add_argument("--judge-timeout", type=float, default=None,
                        help="Override the evaluator's hard-coded 30 s judge request timeout (documented deviation)")
    args = parser.parse_args()

    load_dotenv_if_present()
    config = load_experiment_config()
    policy = load_judge_policy()
    items_path = Path(args.items).resolve()
    items = json.loads(items_path.read_text(encoding="utf-8"))

    if args.fake_upstream:
        from src.benchmarks.fake_upstream import FakeUpstream
        send = FakeUpstream(fail_marker=args.fail_marker)
    else:
        validate_environment()
        key = os.environ.pop("OPENAI_API_KEY")  # keep the real key out of the evaluator's environment
        send = make_openai_sender(os.environ.get("OPENAI_BASE_URL") or config["base_url"], key)
        del key
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ["LLM_API_KEY"] = PLACEHOLDER_KEY

    meter, ledger = UsageMeter(max_usd=args.max_usd), JudgeLedger()
    server = JudgeProxyServer(policy, ledger, meter, send)
    server.start()

    sys.path.insert(0, str(EVALUATOR_ROOT))
    import Evaluation.generation_eval as generation_eval  # noqa: E402  (third-party code, audited)

    generation_eval.HuggingFaceBgeEmbeddings = _embedding_factory(server.base_url, policy.embedding_model)
    if args.judge_timeout is not None:
        real_chat_openai = generation_eval.ChatOpenAI

        def chat_openai_with_timeout(**kwargs):  # only the timeout changes; every other argument is the evaluator's
            kwargs["timeout"] = args.judge_timeout
            return real_chat_openai(**kwargs)

        generation_eval.ChatOpenAI = chat_openai_with_timeout
    generation_eval.Dataset = InMemoryDataset  # datasets/dill is broken on Python 3.14; see src/benchmarks/eval_shims.py
    out_dir = (ROOT / args.output_dir / args.label).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluator_output = out_dir / "evaluator_output.json"
    namespace = SimpleNamespace(
        mode="API", model=policy.judge_model, base_url=server.base_url, embedding_model=policy.embedding_model,
        data_file=str(items_path), output_file=str(evaluator_output), num_samples=None, detailed_output=True,
    )

    error = None
    try:
        asyncio.run(generation_eval.main(namespace))
    except Exception as exc:  # record, never hide
        error = f"{type(exc).__name__}: {exc}"
    finally:
        server.stop()

    output = json.loads(evaluator_output.read_text(encoding="utf-8")) if evaluator_output.is_file() else {}
    reconciliation = reconcile(items, output)
    summary = ledger.summary(policy, meter)
    incomplete = (
        error is not None or not reconciliation["complete"] or summary["failed_request_count"] > 0
        or summary["usage"]["budget_exceeded"]
    )
    record = {
        "label": args.label,
        "is_mocked": bool(args.fake_upstream),
        "status": "incomplete" if incomplete else "complete",
        "error": error,
        "items_file": str(items_path),
        "items_sha256": hashlib.sha256(items_path.read_bytes()).hexdigest(),
        "item_count": len(items),
        "evaluator_commit": "fdbab5959b18c96532580877ffe27d112bccc0ec",
        "evaluator_args": {k: v for k, v in vars(namespace).items() if k not in ("data_file", "output_file")},
        "judge_timeout_override_s": args.judge_timeout,
        "judge_ledger": summary,
        "reconciliation": reconciliation,
        "average_scores": {t: v.get("average_scores") for t, v in output.items() if isinstance(v, dict)},
        "fake_upstream_check": (
            {
                "requests_received": len(send.received),
                "any_request_contained_temperature": any("temperature" in payload for _, payload in send.received),
                "paths": sorted({path for path, _ in send.received}),
            }
            if args.fake_upstream else None
        ),
        "run_configuration": build_run_configuration(
            llm_model=policy.judge_model, embedding_model=policy.embedding_model,
            embedding_dimensions=config["embedding"]["dimensions"], is_mocked=bool(args.fake_upstream),
        ),
    }
    record_path = out_dir / "eval_record.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("status", "error", "item_count")}, indent=2))
    print(f"judge failed requests: {summary['failed_request_count']}; cost ${summary['usage']['cost_usd']}; "
          f"reconciliation complete: {reconciliation['complete']}")
    print(f"Record written to {record_path}")
    sys.exit(2 if incomplete else 0)


if __name__ == "__main__":
    main()
