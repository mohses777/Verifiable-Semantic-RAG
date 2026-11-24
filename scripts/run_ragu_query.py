"""Query a previously built RAGU graph with metering, a hard spend cap and a full run record.

Run with the isolated RAGU interpreter:

    <venv>\\Scripts\\python.exe scripts\\run_ragu_query.py [--execute --max-usd N]

Default is a dry run (checks the saved graph files; NO API calls). --execute sends one
query embedding and one answer-generation request through the metered client; calls are
refused once the cap is reached. RAGU engine defaults are used and recorded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import build_run_configuration, load_experiment_config, validate_environment  # noqa: E402
from src.core.redaction import redact_secrets  # noqa: E402
from src.integrations.ragu_metering import MeteredAsyncOpenAI, UsageMeter  # noqa: E402
from src.core.env import load_dotenv_if_present  # noqa: E402

DEFAULT_QUESTION = (
    "According to the RAGU paper, what performance does RAGU achieve on GraphRAG-Bench (Medical), "
    "and how does it compare with HippoRAG 2?"
)
GRAPH_FILES = ("knowledge_graph.gml", "kv_chunks.json", "vdb_chunk.json", "vdb_entity.json", "vdb_relation.json")
CLIENT_SETTINGS = {"rate_min_delay": 0, "rate_max_simultaneous": 4, "retry_times_sec": [2, 4, 8]}


def _plain(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _plain(v) for k, v in asdict(value).items()}
    return value


async def _query(args: argparse.Namespace) -> dict:
    from openai import AsyncOpenAI
    from ragu import KnowledgeGraph, LocalSearchEngine, NaiveSearchEngine, Settings
    from ragu.models.embedder import EmbedderOpenAI
    from ragu.models.llm import LLMOpenAI

    from src.integrations.ragu_openai import GPT6LunaRAGUClient

    config = load_experiment_config()
    llm_model, embedding_model = config["llm"]["model"], config["embedding"]["model"]
    dimensions = config["embedding"]["dimensions"]
    validate_environment()

    meter = UsageMeter(max_usd=args.max_usd)
    real = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL") or config["base_url"],
        max_retries=0,
    )
    client = GPT6LunaRAGUClient(
        client=MeteredAsyncOpenAI(real, meter),
        llm_model=llm_model,
        rate_min_delay=CLIENT_SETTINGS["rate_min_delay"],
        rate_max_simultaneous=CLIENT_SETTINGS["rate_max_simultaneous"],
        retry_times_sec=tuple(CLIENT_SETTINGS["retry_times_sec"]),
        cache=str(Path(args.workdir).resolve() / "llm_cache"),
    )
    llm = LLMOpenAI(client=client, model_name=llm_model)
    embedder = EmbedderOpenAI(client=client, model_name=embedding_model, dim=dimensions)

    Settings.storage_folder = str((Path(args.workdir).resolve() / "graph"))
    Settings.language = "english"
    graph = KnowledgeGraph(llm=llm, embedder=embedder)  # loads the existing graph from storage
    if args.engine == "local":
        engine = LocalSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)
    else:
        engine = NaiveSearchEngine(llm=llm, knowledge_graph=graph, embedder=embedder)

    status, error, response = "complete", None, None
    started = perf_counter()
    try:
        response = await engine.query(args.question)
    except Exception as exc:  # record the exact failure; never substitute a model
        status, error = "error", f"{type(exc).__name__}: {redact_secrets(str(exc))}"
    elapsed_ms = (perf_counter() - started) * 1_000
    if meter.exceeded:
        status = "budget_exceeded"

    record: dict = {
        "question": args.question,
        "engine": args.engine,
        "engine_params": "RAGU defaults (not tuned)",
        "status": status,
        "error": error,
        "execution_time_ms": round(elapsed_ms, 1),
        "usage": meter.summary(),
        "is_mocked": False,
        "run_configuration": build_run_configuration(
            llm_model=llm_model, embedding_model=embedding_model, embedding_dimensions=dimensions, is_mocked=False,
        ),
        "tokenizer_note": "RAGU truncates context with its default tokenizer setting (see Settings.tokenizer_llm_name), not a gpt-6-luna tokenizer.",
    }
    if response is not None:
        result = response.retrieval.result
        record["answer"] = response.response if isinstance(response.response, str) else response.response.model_dump()
        record["retrieved_context_chars"] = len(response.retrieval.to_text())
        record["retrieved"] = {
            "entities": [{"id": e.id, "name": e.entity_name, "type": e.entity_type} for e in getattr(result, "entities", [])],
            "relations": [
                {"subject": r.subject_name, "predicate": r.relation_type, "object": r.object_name}
                for r in getattr(result, "relations", [])
            ],
            "chunks": [{"id": c.id, "content": c.content} for c in getattr(result, "chunks", [])],
            "summaries": len(getattr(result, "summaries", []) or []),
        }
        record["retrieval_metrics"] = _plain(response.retrieval.metrics) if response.retrieval.metrics else {}
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", default="data/ragu/komarov_2026_ragu")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--engine", choices=("local", "naive"), default="local")
    parser.add_argument("--output", default="results/ragu_q1_local.json")
    parser.add_argument("--execute", action="store_true", help="LIVE OpenAI API (requires --max-usd)")
    parser.add_argument("--max-usd", type=float, default=None)
    args = parser.parse_args()
    if args.execute and args.max_usd is None:
        parser.error("--execute requires --max-usd (hard spend cap)")
    if args.max_usd is None:
        args.max_usd = 1.0

    load_dotenv_if_present()
    graph_dir = (ROOT / args.workdir / "graph").resolve()
    missing = [name for name in GRAPH_FILES if not (graph_dir / name).is_file()]
    print(f"Graph dir: {graph_dir}; missing files: {missing or 'none'}")
    if missing:
        sys.exit(1)
    if not args.execute:
        print(f"DRY RUN: no API calls. Would run RAGU {args.engine} search on: {args.question}")
        return
    args.workdir = str(ROOT / args.workdir)
    record = asyncio.run(_query(args))
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    shown = {k: record[k] for k in ("status", "error", "execution_time_ms", "usage") if k in record}
    print(json.dumps(shown, indent=2))
    print(f"Saved run record to {output}")


if __name__ == "__main__":
    main()
