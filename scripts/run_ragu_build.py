"""Build a RAGU knowledge graph over one document with metering and a hard spend cap.

Run with the isolated RAGU interpreter (RAGU is not installed in the main environment):

    <venv>\\Scripts\\python.exe scripts\\run_ragu_build.py [mode flags]

Modes (exactly one effective):
  default          dry run: ingest + RAGU chunking statistics only, NO API calls
  --mock-server    full RAGU pipeline against RAGU's local mock OpenAI server (zero cost)
  --execute        LIVE OpenAI API; requires --max-usd. Calls are refused after the cap.

The source text comes from this project's own PDF ingestion, so both systems read
the same document text. RAGU applies its own chunking and extraction (documented in
the run record); the run record also stores models, request settings, measured usage,
cost, errors and timing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import build_run_configuration, load_experiment_config, validate_environment  # noqa: E402
from src.core.redaction import redact_secrets  # noqa: E402
from src.ingestion.loader import ingest_file  # noqa: E402
from src.integrations.ragu_metering import MeteredAsyncOpenAI, UsageMeter  # noqa: E402
from src.pipeline.runner import load_dotenv_if_present  # noqa: E402

# RAGU's documented example configuration (README "Simple example of building knowledge graph").
RAGU_CONFIG = {
    "chunker": {"class": "SimpleChunker", "max_chunk_size": 1000, "overlap": 0},
    "extractor": {
        "class": "TwoStageArtifactsExtractorLLM",
        "icl_enabled": True,
        "icl_num_examples": 2,
        "do_entity_validation": True,
        "do_relation_validation": True,
    },
    "builder": {
        "use_llm_summarization": True,
        "use_clustering": False,
        "build_only_vector_context": False,
        "make_community_summary": True,
        "remove_isolated_nodes": True,
    },
    "language": "english",
    "llm_extra_kwargs": {},  # none: no temperature/top_p (gpt-6-luna rejects temperature != 1)
}
CLIENT_SETTINGS = {"rate_min_delay": 0, "rate_max_simultaneous": 4, "retry_times_sec": [2, 4, 8]}


def _document_text(paper: Path):
    document = ingest_file(paper)
    return document, "\n\n".join(section.text for section in document.sections)


def _chunk_stats(text: str) -> dict:
    from ragu import SimpleChunker

    chunker = SimpleChunker(
        max_chunk_size=RAGU_CONFIG["chunker"]["max_chunk_size"], overlap=RAGU_CONFIG["chunker"]["overlap"]
    )
    chunks = chunker.split(text)
    sizes = [len(c.content) for c in chunks]
    return {"chunk_count": len(chunks), "total_chars": sum(sizes), "max_chunk_chars": max(sizes, default=0)}


async def _build(args: argparse.Namespace, document, text: str, stats: dict) -> dict:
    from openai import AsyncOpenAI
    from ragu import BuilderArguments, KnowledgeGraph, Settings, SimpleChunker, TwoStageArtifactsExtractorLLM
    from ragu.common.logger import logger
    from ragu.common.prompts import ICLConfig
    from ragu.models.embedder import EmbedderOpenAI
    from ragu.models.llm import LLMOpenAI

    from src.integrations.ragu_openai import GPT6LunaRAGUClient

    config = load_experiment_config()
    llm_model, embedding_model = config["llm"]["model"], config["embedding"]["model"]
    dimensions = config["embedding"]["dimensions"]

    server = None
    if args.mock_server:
        from ragu.utils.testing.openai_mock_server import OpenAIMockServer

        server = OpenAIMockServer()
        server.start()
        base_url, api_key = server.base_url, "mock"
    else:
        validate_environment()
        base_url, api_key = os.environ.get("OPENAI_BASE_URL") or config["base_url"], os.environ["OPENAI_API_KEY"]

    meter = UsageMeter(max_usd=args.max_usd)
    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    real = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0)  # RAGU does its own retrying
    client = GPT6LunaRAGUClient(
        client=MeteredAsyncOpenAI(real, meter),
        llm_model=llm_model,
        rate_min_delay=CLIENT_SETTINGS["rate_min_delay"],
        rate_max_simultaneous=CLIENT_SETTINGS["rate_max_simultaneous"],
        retry_times_sec=tuple(CLIENT_SETTINGS["retry_times_sec"]),
        cache=str(workdir / "llm_cache"),
    )
    llm = LLMOpenAI(client=client, model_name=llm_model)
    embedder = EmbedderOpenAI(client=client, model_name=embedding_model, dim=dimensions)

    Settings.storage_folder = str(workdir / "graph")
    Settings.language = RAGU_CONFIG["language"]
    extractor_cfg, builder_cfg = RAGU_CONFIG["extractor"], RAGU_CONFIG["builder"]
    extractor = TwoStageArtifactsExtractorLLM(
        llm=llm,
        embedder=embedder,
        icl_config=ICLConfig(enabled=extractor_cfg["icl_enabled"], num_examples=extractor_cfg["icl_num_examples"]),
        do_entity_validation=extractor_cfg["do_entity_validation"],
        do_relation_validation=extractor_cfg["do_relation_validation"],
    )
    graph = KnowledgeGraph(
        llm=llm,
        embedder=embedder,
        chunker=SimpleChunker(
            max_chunk_size=RAGU_CONFIG["chunker"]["max_chunk_size"], overlap=RAGU_CONFIG["chunker"]["overlap"]
        ),
        artifact_extractor=extractor,
        builder_settings=BuilderArguments(**builder_cfg),
    )

    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
    status, error = "complete", None
    started = perf_counter()
    try:
        await graph.build_from_docs([text])
    except Exception as exc:  # record the exact failure; never substitute a model
        status, error = "error", f"{type(exc).__name__}: {redact_secrets(str(exc))}"
    elapsed_ms = (perf_counter() - started) * 1_000
    logger.remove(sink_id)
    if server is not None:
        server.stop()
    if meter.exceeded:
        status = "budget_exceeded"

    counts = {}
    for message in messages:
        match = re.search(r"entities: (\d+), relations: (\d+), communities: (\d+), chunks: (\d+)", message)
        if match:
            counts = dict(zip(("entities", "relations", "communities", "chunks"), map(int, match.groups())))

    return {
        "is_mocked": bool(args.mock_server),
        "status": status,
        "error": error,
        "execution_time_ms": round(elapsed_ms, 1),
        "graph_counts": counts,
        "usage": meter.summary(),
        "run_configuration": build_run_configuration(
            llm_model=llm_model,
            embedding_model=embedding_model,
            embedding_dimensions=dimensions,
            is_mocked=bool(args.mock_server),
            chunking={"chunker": "ragu.SimpleChunker", **RAGU_CONFIG["chunker"]},
        ),
        "ragu_configuration": {**RAGU_CONFIG, "client": CLIENT_SETTINGS},
        "ragu_workdir": str(workdir),
        "cache_note": "RAGU response cache enabled in the workdir; cache hits cost nothing and are not metered.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paper", default="data/papers/komarov_2026_ragu.pdf")
    parser.add_argument("--workdir", default="data/ragu/komarov_2026_ragu")
    parser.add_argument("--output", default="results/ragu_build_komarov_2026.json")
    parser.add_argument("--mock-server", action="store_true", help="Zero-cost run against RAGU's mock server")
    parser.add_argument("--execute", action="store_true", help="LIVE OpenAI API (requires --max-usd)")
    parser.add_argument("--max-usd", type=float, default=None, help="Hard spend cap for --execute")
    args = parser.parse_args()
    if args.mock_server and args.execute:
        parser.error("choose either --mock-server or --execute")
    if args.execute and args.max_usd is None:
        parser.error("--execute requires --max-usd (hard spend cap)")
    if args.max_usd is None:
        args.max_usd = 1.0  # unused by dry run / mock (mock usage costs nothing)

    load_dotenv_if_present()
    paper = (ROOT / args.paper).resolve()
    document, text = _document_text(paper)
    stats = _chunk_stats(text)
    print(f"Document: {paper.name}: {len(document.sections)} pages, {len(text)} chars; RAGU chunks: {stats}")
    if not (args.mock_server or args.execute):
        print("DRY RUN: no API calls made. Use --mock-server (free) or --execute --max-usd N (paid).")
        return

    result = asyncio.run(_build(args, document, text, stats))
    result.update(
        document={"path": args.paper, "document_id": document.document_id, "checksum_sha256": document.checksum_sha256,
                  "pages": len(document.sections)},
        ragu_chunking=stats,
    )
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "error", "execution_time_ms", "graph_counts", "usage")}, indent=2))
    print(f"Saved run record to {output}")


if __name__ == "__main__":
    main()
