"""Prepare a reduced-cost GraphRAG-Bench (Medical) run: map questions to the corpus, then select a slice.

  embed   Embed corpus chunks and question evidence (PAID, tiny). Dry run unless --execute --max-usd N.
  select  Offline: pick a corpus window and a seeded stratified question sample from those embeddings.

Third-party benchmark text stays under data/benchmarks/ (git-ignored; licence unresolved). Only counts and
parameters are written to results/.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
from collections import Counter
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmarks.selection import (  # noqa: E402
    WindowChoice, best_matches, content_tokens, min_statement_coverage, sample_questions, select_window, similarity_deciles,
    type_counts,
)
from src.core.config import build_run_configuration, load_experiment_config, validate_environment  # noqa: E402
from src.core.pricing import get_model_pricing  # noqa: E402
from src.ingestion.loader import ingest_file  # noqa: E402
from src.core.env import load_dotenv_if_present  # noqa: E402
from src.processing.chunking import ChunkingConfig, chunk_document  # noqa: E402

DATA = ROOT / "data" / "benchmarks" / "graphrag_bench"
EMBEDDINGS = DATA / "mapping_embeddings.npz"
MANIFEST = DATA / "mapping_manifest.json"
SLICE_DIR = DATA / "slice"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_inputs():
    corpus_json = DATA / "medical.json"
    questions_json = DATA / "medical_questions.json"
    context = json.loads(corpus_json.read_text(encoding="utf-8"))["context"]
    corpus_txt = DATA / "medical_corpus.txt"
    corpus_txt.write_text(context, encoding="utf-8")
    document = ingest_file(corpus_txt)
    chunks = chunk_document(document, ChunkingConfig())
    questions = json.loads(questions_json.read_text(encoding="utf-8"))
    usable = [q for q in questions if isinstance(q.get("evidence"), str) and q["evidence"].strip()]
    hashes = {"medical.json": _sha256(corpus_json), "medical_questions.json": _sha256(questions_json)}
    return document, chunks, questions, usable, hashes


async def _embed(args: argparse.Namespace) -> None:
    config = load_experiment_config()
    model, dims = config["embedding"]["model"], config["embedding"]["dimensions"]
    document, chunks, questions, usable, hashes = _load_inputs()
    pricing = get_model_pricing(model)
    chunk_chars = sum(len(c.text) for c in chunks)
    evidence_chars = sum(len(q["evidence"]) for q in usable)
    est_tokens = (chunk_chars + evidence_chars) // 4
    est_cost = pricing.input_cost(est_tokens)
    print(f"Chunks: {len(chunks)} ({chunk_chars} chars); questions with evidence: {len(usable)}/{len(questions)} "
          f"({evidence_chars} chars)")
    print(f"Estimated embedding tokens ~{est_tokens} (chars/4, conservative) -> ~${est_cost:.4f} at ${pricing.input}/M tokens")
    if not args.execute:
        print("DRY RUN: no API calls. Re-run with --execute --max-usd N after approval.")
        return

    from openai import AsyncOpenAI

    from src.embeddings.client import EmbeddingClient, OpenAIEmbeddingProvider
    from src.integrations.ragu_metering import MeteredAsyncOpenAI, UsageMeter

    validate_environment()
    meter = UsageMeter(max_usd=args.max_usd)
    key = os.environ["OPENAI_API_KEY"]
    client = MeteredAsyncOpenAI(
        AsyncOpenAI(api_key=key, base_url=os.environ.get("OPENAI_BASE_URL") or config["base_url"]), meter
    )
    embedder = EmbeddingClient(OpenAIEmbeddingProvider(api_key=key, client=client), model_id=model, dimensions=dims)
    started = perf_counter()
    chunk_vectors = await embedder.embed_texts([c.text for c in chunks])
    evidence_vectors = await embedder.embed_texts([q["evidence"] for q in usable])
    elapsed = perf_counter() - started

    DATA.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        EMBEDDINGS,
        chunk_ids=np.array([c.chunk_id for c in chunks]),
        chunk_vectors=np.array(chunk_vectors, dtype=np.float32),
        question_ids=np.array([q["id"] for q in usable]),
        evidence_vectors=np.array(evidence_vectors, dtype=np.float32),
    )
    manifest = {
        "is_mocked": False,
        "data_hashes": hashes,
        "chunking": {"chunk_size": 1000, "overlap": 200},
        "chunk_count": len(chunks),
        "questions_total": len(questions),
        "questions_with_evidence": len(usable),
        "usage": meter.summary(),
        "execution_time_s": round(elapsed, 1),
        "run_configuration": build_run_configuration(
            llm_model=config["llm"]["model"], embedding_model=model, embedding_dimensions=dims, is_mocked=False,
            chunking={"chunk_size": 1000, "overlap": 200},
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["usage"], indent=2))
    print(f"Saved {EMBEDDINGS.name} and {MANIFEST.name}")


def _best_filtered_window(chunks, text, qids, types, by_id, index, sim, *, window_chunks, per_type, min_sim,
                          min_lexical, stride):
    """Window maximizing sum over types of min(eligible, target); eligible = similarity AND lexical filters.

    Ties go to the window with more eligible questions overall, then to the leftmost.
    """
    best = None
    all_types = sorted(set(types))
    for start in range(0, len(chunks) - window_chunks + 1, stride):
        end = start + window_chunks
        words = set(content_tokens(text[chunks[start].start_char:chunks[end - 1].end_char]))
        counts: Counter = Counter()
        for qid, qtype, idx, s in zip(qids, types, index, sim):
            if start <= idx < end and s >= min_sim and min_statement_coverage(by_id[qid]["evidence"], words) >= min_lexical:
                counts[qtype] += 1
        score = sum(min(c, per_type) for c in counts.values())
        key = (score, sum(counts.values()))
        if best is None or key > best[0]:
            best = (key, WindowChoice(start, end, score, {t: counts.get(t, 0) for t in all_types}))
    return best[1]


def _select(args: argparse.Namespace) -> None:
    document, chunks, questions, usable, hashes = _load_inputs()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["data_hashes"] != hashes:
        sys.exit("Benchmark data changed since embeddings were made; re-run embed.")
    stored = np.load(EMBEDDINGS, allow_pickle=False)
    if list(stored["chunk_ids"]) != [c.chunk_id for c in chunks]:
        sys.exit("Chunk ids differ from the stored embeddings; re-run embed.")
    by_id = {q["id"]: q for q in usable}
    qids = [str(i) for i in stored["question_ids"]]
    types = [by_id[i]["question_type"] for i in qids]

    index, sim = best_matches(stored["chunk_vectors"], stored["evidence_vectors"])
    print("best-match similarity deciles (10%..90%):", similarity_deciles(sim))
    for qtype in sorted(set(types)):
        vals = [s for s, t in zip(sim, types) if t == qtype]
        print(f"  {qtype}: n={len(vals)} median={float(np.median(vals)):.3f}")

    window_chunks = max(1, round(args.fraction * len(chunks)))
    window = _best_filtered_window(
        chunks, document.sections[0].text, qids, types, by_id, index, sim, window_chunks=window_chunks,
        per_type=args.per_type, min_sim=args.min_sim, min_lexical=args.min_lexical, stride=args.stride,
    )
    first, last = chunks[window.start_chunk], chunks[window.end_chunk - 1]
    text = document.sections[0].text
    slice_text = text[first.start_char:last.end_char]
    slice_words = set(content_tokens(slice_text))
    coverage = {qid: min_statement_coverage(by_id[qid]["evidence"], slice_words) for qid in qids}
    eligible = {qid for qid, c in coverage.items() if c >= args.min_lexical}
    in_window = [qid for qid, i in zip(qids, index) if window.start_chunk <= i < window.end_chunk]
    dropped_by_lexical = {t: sum(1 for q in in_window if by_id[q]["question_type"] == t and q not in eligible)
                          for t in sorted(set(types))}
    chosen, shortfall = sample_questions(
        qids, types, index, sim, window=window, per_type_target=args.per_type,
        min_similarity=args.min_sim, seed=args.seed, eligible_ids=eligible,
    )

    SLICE_DIR.mkdir(parents=True, exist_ok=True)
    (SLICE_DIR / "slice_corpus.txt").write_text(slice_text, encoding="utf-8")
    pos = {q: i for i, q in enumerate(qids)}
    selected = []
    for qid in chosen:
        q = dict(by_id[qid])
        q["best_chunk_id"] = chunks[int(index[pos[qid]])].chunk_id
        q["best_similarity"] = round(float(sim[pos[qid]]), 4)
        q["min_statement_coverage"] = round(coverage[qid], 3)
        selected.append(q)
    (SLICE_DIR / "selected_questions.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    spot = [f"# Spot check: does the matched chunk contain the evidence? (min_sim={args.min_sim})\n"]
    for q in selected[: args.spot_check]:
        chunk = chunks[int(index[pos[q["id"]]])]
        spot.append(f"## {q['id']} ({q['question_type']}), similarity {q['best_similarity']}\n"
                    f"**Question:** {q['question']}\n\n**Evidence:** {q['evidence'][:500]}\n\n"
                    f"**Matched chunk:** {' '.join(chunk.text.split())[:700]}\n")
    (SLICE_DIR / "spot_check.md").write_text("\n".join(spot), encoding="utf-8")

    summary = {
        "parameters": {"fraction": args.fraction, "window_chunks": window_chunks, "per_type_target": args.per_type,
                       "min_similarity": args.min_sim, "min_lexical_coverage": args.min_lexical, "seed": args.seed},
        "data_hashes": hashes,
        "window": {"start_chunk": window.start_chunk, "end_chunk": window.end_chunk,
                   "start_char": first.start_char, "end_char": last.end_char, "slice_chars": len(slice_text),
                   "corpus_chars": len(text)},
        "questions_eligible_in_window": window.per_type_available,
        "questions_selected": type_counts([q["question_type"] for q in selected]),
        "shortfall_vs_target": shortfall,
        "in_window_dropped_by_lexical_filter": dropped_by_lexical,
        "selected_question_ids": [q["id"] for q in selected],
        "similarity_deciles": similarity_deciles(sim),
        "note": "Window and sample chosen by embedding similarity (approximate). Slice corpus is easier than the full benchmark.",
    }
    out = ROOT / "results" / "benchmark_selection_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("window", "questions_eligible_in_window", "in_window_dropped_by_lexical_filter",
                                                 "questions_selected", "shortfall_vs_target")}, indent=2))
    print(f"Wrote slice files to {SLICE_DIR} and {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    embed = sub.add_parser("embed")
    embed.add_argument("--execute", action="store_true")
    embed.add_argument("--max-usd", type=float, default=None)
    select = sub.add_parser("select")
    select.add_argument("--fraction", type=float, default=0.10, help="Window size as a fraction of corpus chunks")
    select.add_argument("--per-type", type=int, default=10)
    select.add_argument("--min-sim", type=float, default=0.6, help="Minimum embedding similarity (heuristic)")
    select.add_argument("--min-lexical", type=float, default=0.75,
                        help="Minimum share of every evidence statement's content words found in the slice (heuristic)")
    select.add_argument("--seed", type=int, default=42)
    select.add_argument("--stride", type=int, default=5, help="Window search step in chunks")
    select.add_argument("--spot-check", type=int, default=12)
    args = parser.parse_args()
    load_dotenv_if_present()
    if args.command == "embed":
        if args.execute and args.max_usd is None:
            parser.error("--execute requires --max-usd")
        args.max_usd = args.max_usd or 1.0
        asyncio.run(_embed(args))
    else:
        _select(args)


if __name__ == "__main__":
    main()
