"""Answer the selected GraphRAG-Bench questions with RAGU (LocalSearchEngine, RAGU defaults) on the slice graph.

Run with the isolated RAGU interpreter:

    <ragu-venv>\\Scripts\\python.exe scripts\\run_benchmark_answers_ragu.py --execute --max-usd 0.15
    <ragu-venv>\\Scripts\\python.exe scripts\\run_benchmark_answers_ragu.py            # dry run: checks only

Questions are answered one at a time so per-question usage is exact. Every question gets a record (answered or
error). RAGU's answer is free text without citations; the context recorded is exactly what RAGU gave its answering
model (``retrieval.to_text()``). Outputs mirror run_benchmark_answers_ours.py.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmarks.eval_adapter import build_items  # noqa: E402
from src.core.config import build_run_configuration, load_experiment_config, validate_environment  # noqa: E402
from src.core.env import load_dotenv_if_present  # noqa: E402
from src.core.redaction import redact_secrets  # noqa: E402
from src.integrations.ragu_metering import BudgetExceededError, MeteredAsyncOpenAI, UsageMeter  # noqa: E402

DATA = ROOT / "data" / "benchmarks" / "graphrag_bench"
GRAPH_FILES = ("knowledge_graph.gml", "kv_chunks.json", "vdb_chunk.json", "vdb_entity.json", "vdb_relation.json")
CLIENT_SETTINGS = {"rate_min_delay": 0, "rate_max_simultaneous": 4, "retry_times_sec": [2, 4, 8]}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def _run(args, questions, questions_path: Path, workdir: Path) -> None:
    from openai import AsyncOpenAI
    from ragu import KnowledgeGraph, LocalSearchEngine, Settings
    from ragu.models.embedder import EmbedderOpenAI
    from ragu.models.llm import LLMOpenAI

    from src.integrations.ragu_openai import GPT6LunaRAGUClient

    config = load_experiment_config()
    llm_model, embedding_model = config["llm"]["model"], config["embedding"]["model"]
    dimensions = config["embedding"]["dimensions"]
    validate_environment()
    meter = UsageMeter(max_usd=args.max_usd)
    real = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"],
                       base_url=os.environ.get("OPENAI_BASE_URL") or config["base_url"], max_retries=0)
    client = GPT6LunaRAGUClient(
        client=MeteredAsyncOpenAI(real, meter), llm_model=llm_model,
        rate_min_delay=CLIENT_SETTINGS["rate_min_delay"], rate_max_simultaneous=CLIENT_SETTINGS["rate_max_simultaneous"],
        retry_times_sec=tuple(CLIENT_SETTINGS["retry_times_sec"]), cache=str(workdir / "llm_cache"),
    )
    llm = LLMOpenAI(client=client, model_name=llm_model)
    embedder = EmbedderOpenAI(client=client, model_name=embedding_model, dim=dimensions)
    Settings.storage_folder = str(workdir / "graph")
    Settings.language = "english"
    engine = LocalSearchEngine(llm=llm, knowledge_graph=KnowledgeGraph(llm=llm, embedder=embedder), embedder=embedder)

    records, started_all = [], perf_counter()
    for number, question in enumerate(questions, 1):
        base = {"id": question["id"], "question_type": question["question_type"], "question": question["question"]}
        cost_before, started = meter.cost_usd, perf_counter()
        try:
            response = await engine.query(question["question"])
        except BudgetExceededError as error:
            records.append({**base, "status": "error", "error": f"BudgetExceededError: {error}"})
            for rest in questions[number:]:
                records.append({"id": rest["id"], "question_type": rest["question_type"], "question": rest["question"],
                                "status": "not_attempted", "error": "budget cap reached earlier"})
            print(f"[{number}/{len(questions)}] {question['id']}: budget cap reached; stopping")
            break
        except Exception as error:  # record exactly, keep going
            records.append({**base, "status": "error", "error": f"{type(error).__name__}: {redact_secrets(str(error))}"})
            print(f"[{number}/{len(questions)}] {question['id']}: ERROR {type(error).__name__}")
            continue
        answer = response.response if isinstance(response.response, str) else json.dumps(response.response.model_dump())
        result = response.retrieval.result
        records.append({
            **base,
            "status": "answered" if answer.strip() else "error",
            "answer_text": answer,
            "context": response.retrieval.to_text(),
            "retrieved": {
                "entities": len(getattr(result, "entities", []) or []),
                "relations": len(getattr(result, "relations", []) or []),
                "chunk_ids": [c.id for c in getattr(result, "chunks", []) or []],
            },
            "cost_usd": round(meter.cost_usd - cost_before, 6),
            "latency_ms": round((perf_counter() - started) * 1000, 1),
            **({"error": "empty answer"} if not answer.strip() else {}),
        })
        print(f"[{number}/{len(questions)}] {question['id']}: {records[-1]['status']}")

    out_dir = DATA / "runs" / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "answers.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    usable = [r for r in records if r["status"] == "answered"]
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
        "system": "RAGU v0.1.0 (abd3f29), LocalSearchEngine with RAGU defaults",
        "label": args.label,
        "is_mocked": False,
        "questions_total": len(questions),
        "status_counts": dict(Counter(r["status"] for r in records)),
        "status_by_type": {t: dict(c) for t, c in status_by_type.items()},
        "unanswered_ids": [r["id"] for r in records if r["status"] != "answered"],
        "errors": [{"id": r["id"], "error": r.get("error")} for r in records if r["status"] != "answered"],
        "usage": meter.summary(),
        "total_cost_usd": round(meter.cost_usd, 6),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "wall_time_s": round(perf_counter() - started_all, 1),
        "engine": "LocalSearchEngine (defaults: top_k=20 entities, chunks included, no community summaries)",
        "ragu_workdir": args.workdir,
        "selected_questions_sha256": _sha256(questions_path),
        "items_sha256": _sha256(items_path),
        "items_count": len(items),
        "run_configuration": build_run_configuration(
            llm_model=llm_model, embedding_model=embedding_model, embedding_dimensions=dimensions, is_mocked=False,
        ),
    }
    result_path = ROOT / "results" / f"benchmark_answers_{args.label}.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status_counts", "total_cost_usd", "mean_latency_ms")}, indent=2))
    print(f"Wrote {result_path} and {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", default="data/ragu/graphrag_bench_slice")
    parser.add_argument("--label", default="ragu")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-usd", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.execute and args.max_usd is None:
        parser.error("--execute requires --max-usd")
    load_dotenv_if_present()
    questions_path = DATA / "slice" / "selected_questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[: args.limit]
    workdir = (ROOT / args.workdir).resolve()
    missing = [n for n in GRAPH_FILES if not (workdir / "graph" / n).is_file()]
    print(f"RAGU graph: {workdir / 'graph'}; missing files: {missing or 'none'}; questions: {len(questions)}")
    if missing:
        sys.exit(1)
    if not args.execute:
        print("DRY RUN: no API calls. Use --execute --max-usd N after approval.")
        return
    asyncio.run(_run(args, questions, questions_path, workdir))


if __name__ == "__main__":
    main()
