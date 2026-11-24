"""Answer the selected GraphRAG-Bench questions with THIS project's system (hybrid retrieval + grounded QA).

    python scripts/run_benchmark_answers_ours.py --execute --max-usd 0.15      # live, capped
    python scripts/run_benchmark_answers_ours.py --mock                        # zero cost, plumbing only

Questions are answered one at a time so per-question token usage is exact. Every question gets a record:
answered, abstained (insufficient evidence, including fail-closed), or error. Nothing is dropped.

Outputs (third-party benchmark/corpus text stays in the git-ignored data directory):
  data/benchmarks/graphrag_bench/runs/<label>/answers.json   raw per-question records incl. the exact evidence shown
  data/benchmarks/graphrag_bench/runs/<label>/items.json     evaluator input (answered questions only)
  results/benchmark_answers_<label>.json                     aggregate summary (counts, costs, latency, config)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmarks.eval_adapter import build_items  # noqa: E402
from src.core.config import build_run_configuration, load_experiment_config, validate_environment  # noqa: E402
from src.core.env import load_dotenv_if_present  # noqa: E402
from src.core.redaction import redact_secrets  # noqa: E402
from src.generation.answering import format_evidence_context  # noqa: E402
from src.integrations.ragu_metering import BudgetExceededError, UsageMeter  # noqa: E402
from src.retrieval.hybrid import HybridRetrievalConfig  # noqa: E402
from src.storage.artifacts import load_artifacts  # noqa: E402

DATA = ROOT / "data" / "benchmarks" / "graphrag_bench"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_pipeline(args, artifacts, config):
    from src.pipeline.qa import QuestionAnsweringPipeline, _build_mock_qa_pipeline

    retrieval = HybridRetrievalConfig()
    if args.mock:
        qa = _build_mock_qa_pipeline(retrieval, dimensions=artifacts.vector_index.dimensions)
        artifacts.retriever.client = qa.document_pipeline.embedding_client
        return qa, None, retrieval

    from openai import AsyncOpenAI

    from src.embeddings.client import EmbeddingClient, OpenAIEmbeddingProvider
    from src.generation.answering import EvidenceBackedAnswerGenerator, OpenAIStructuredAnswerProvider
    from src.integrations.ragu_metering import MeteredAsyncOpenAI

    validate_environment()
    key = os.environ["OPENAI_API_KEY"]
    meter = UsageMeter(max_usd=args.max_usd)
    client = MeteredAsyncOpenAI(AsyncOpenAI(api_key=key, base_url=os.environ.get("OPENAI_BASE_URL") or config["base_url"]), meter)
    generator = EvidenceBackedAnswerGenerator(OpenAIStructuredAnswerProvider(api_key=key, client=client),
                                              model_id=config["llm"]["model"])
    artifacts.retriever.client = EmbeddingClient(
        OpenAIEmbeddingProvider(api_key=key, client=client),
        model_id=config["embedding"]["model"], dimensions=config["embedding"]["dimensions"],
    )
    qa = QuestionAnsweringPipeline(document_pipeline=None, answer_generator=generator,  # type: ignore[arg-type]
                                   retrieval_config=retrieval, is_mocked=False)
    return qa, meter, retrieval


async def _run(args) -> None:
    config = load_experiment_config()
    questions_path = DATA / "slice" / "selected_questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    artifacts_dir = ROOT / args.artifacts
    artifacts = load_artifacts(artifacts_dir)
    qa, meter, retrieval = _build_pipeline(args, artifacts, config)

    records, started_all = [], perf_counter()
    for number, question in enumerate(questions, 1):
        cost_before = meter.cost_usd if meter else 0.0
        started = perf_counter()
        base = {"id": question["id"], "question_type": question["question_type"], "question": question["question"]}
        try:
            result, summary = await qa.ask(question["question"], artifacts=artifacts, config=retrieval)
        except BudgetExceededError as error:
            records.append({**base, "status": "error", "error": f"BudgetExceededError: {error}"})
            print(f"[{number}/{len(questions)}] {question['id']}: budget cap reached; stopping")
            for rest in questions[number:]:
                records.append({"id": rest["id"], "question_type": rest["question_type"], "question": rest["question"],
                                "status": "not_attempted", "error": "budget cap reached earlier"})
            break
        except Exception as error:  # record exactly, keep going
            records.append({**base, "status": "error", "error": f"{type(error).__name__}: {redact_secrets(str(error))}"})
            print(f"[{number}/{len(questions)}] {question['id']}: ERROR {type(error).__name__}")
            continue
        diagnostics = result.diagnostics
        abstained = bool(summary.insufficient_evidence)
        records.append({
            **base,
            "status": "abstained" if abstained else "answered",
            "answer_text": summary.answer_text,
            "insufficient_evidence": abstained,
            "fail_closed": bool(diagnostics.get("fail_closed")),
            "raw_citations": diagnostics.get("raw_citation_count"),
            "accepted_citations": diagnostics.get("accepted_citation_count"),
            "rejection_summary": diagnostics.get("rejection_summary"),
            "retrieved_chunk_ids": [c["chunk_id"] for c in summary.retrieved_chunks],
            "context": format_evidence_context(result.retrieval),
            "cost_usd": round((meter.cost_usd if meter else 0.0) - cost_before, 6),
            "latency_ms": round((perf_counter() - started) * 1000, 1),
            "qa_summary": summary.to_dict(),
        })
        print(f"[{number}/{len(questions)}] {question['id']}: {records[-1]['status']} "
              f"(citations {diagnostics.get('accepted_citation_count')}/{diagnostics.get('raw_citation_count')})")

    out_dir = DATA / "runs" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "answers.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    usable = [r for r in records if r["status"] in ("answered", "abstained")]
    by_id = {q["id"]: q for q in questions}
    items = build_items([by_id[r["id"]] for r in usable], {r["id"]: r["answer_text"] for r in usable},
                        {r["id"]: [r["context"]] for r in usable})
    items_path = out_dir / "items.json"
    items_path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    status_by_type: dict[str, Counter] = {}
    for r in records:
        status_by_type.setdefault(r["question_type"], Counter())[r["status"]] += 1
    latencies = [r["latency_ms"] for r in usable]
    summary = {
        "system": "this project (hybrid vector + graph retrieval, quote-verified answers)",
        "label": args.label,
        "is_mocked": bool(args.mock),
        "questions_total": len(questions),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "status_by_type": {t: dict(c) for t, c in status_by_type.items()},
        "fail_closed_count": sum(1 for r in usable if r["fail_closed"]),
        "unanswered_ids": [r["id"] for r in records if r["status"] in ("error", "not_attempted")],
        "errors": [{"id": r["id"], "error": r["error"]} for r in records if r["status"] == "error"],
        "usage": meter.summary() if meter else None,
        "total_cost_usd": round(meter.cost_usd, 6) if meter else 0.0,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "wall_time_s": round(perf_counter() - started_all, 1),
        "retrieval_config": asdict(retrieval),
        "artifacts_dir": args.artifacts,
        "artifacts_manifest_sha256": _sha256(artifacts_dir / "manifest.json"),
        "selected_questions_sha256": _sha256(questions_path),
        "items_sha256": _sha256(items_path),
        "items_count": len(items),
        "run_configuration": build_run_configuration(
            llm_model=config["llm"]["model"], embedding_model=config["embedding"]["model"],
            embedding_dimensions=config["embedding"]["dimensions"], is_mocked=bool(args.mock),
            retrieval=asdict(retrieval),
        ),
    }
    result_path = ROOT / "results" / f"benchmark_answers_{args.label}.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status_counts", "fail_closed_count", "total_cost_usd", "mean_latency_ms")}, indent=2))
    print(f"Wrote {result_path} and {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts", default="data/processed/graphrag_bench_slice")
    parser.add_argument("--label", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-usd", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Answer only the first N questions")
    args = parser.parse_args()
    if args.mock == args.execute:
        parser.error("choose exactly one of --mock or --execute")
    if args.execute and args.max_usd is None:
        parser.error("--execute requires --max-usd")
    args.label = args.label or ("ours_mock" if args.mock else "ours")
    load_dotenv_if_present()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
