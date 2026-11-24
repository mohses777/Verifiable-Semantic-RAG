"""Summarize GraphRAG-Bench scores for two systems from the evaluator's detailed outputs (offline, no API calls).

Reads results/benchmark_eval/<label>/{evaluator_output.json, eval_record.json} and the answer summaries, and writes
results/benchmark_scores_summary.json with per-type means, standard deviations, sample counts, paired differences on
the questions scored for BOTH systems (with a seeded bootstrap 95% interval), and a breakdown of this project's scores
by whether the answer was flagged "insufficient evidence". Contains no benchmark or corpus text.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmarks" / "graphrag_bench"


def _read_detailed(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    output = json.loads(path.read_text(encoding="utf-8"))
    table: dict[str, dict[str, dict[str, float]]] = {}
    for qtype, block in output.items():
        for sample in block.get("detailed", []):
            metrics = {k: float(v) for k, v in (sample.get("metrics") or {}).items()
                       if isinstance(v, (int, float)) and not math.isnan(v)}
            table.setdefault(qtype, {})[sample["id"]] = metrics
    return table


PER_SAMPLE = ROOT / "results" / "benchmark_per_sample_scores.json"


def _scores(label: str) -> tuple[dict[str, dict[str, dict[str, float]]], dict, list[dict]]:
    """{question_type: {question_id: {metric: score}}} from the main run, completed by supplementary re-scores.

    Supplements (results/benchmark_eval/<label>/rescore_*/) only fill questions the main run did not score; they never
    replace a score the main run produced. If the evaluator's detailed outputs are absent (they are not committed,
    because they contain benchmark text), the committed text-free file results/benchmark_per_sample_scores.json is
    used instead, so the summary can be recomputed from the repository alone.
    Returns the table, a provenance/completeness record, and text-free per-sample rows.
    """
    base = ROOT / "results" / "benchmark_eval" / label
    if (base / "evaluator_output.json").is_file():
        table = _read_detailed(base / "evaluator_output.json")
        source = {qid: "main" for samples in table.values() for qid in samples}
        for supplement in sorted(base.glob("rescore_*")):
            for qtype, samples in _read_detailed(supplement / "evaluator_output.json").items():
                for qid, metrics in samples.items():
                    if qid not in table.get(qtype, {}):
                        table.setdefault(qtype, {})[qid] = metrics
                        source[qid] = supplement.name
        items = json.loads((DATA / "runs" / label / "items.json").read_text(encoding="utf-8"))
        expected = sorted({(i["question_type"], i["id"]) for i in items})
    else:
        stored = json.loads(PER_SAMPLE.read_text(encoding="utf-8"))[label]
        table, source = {}, {}
        for row in stored["samples"]:
            table.setdefault(row["question_type"], {})[row["id"]] = row["metrics"]
            source[row["id"]] = row["source"]
        expected = sorted((t, i) for t, i in stored["expected_items"])
    rows = [{"id": qid, "question_type": qtype, "source": source[qid], "metrics": metrics}
            for qtype, samples in sorted(table.items()) for qid, metrics in sorted(samples.items())]
    still_missing = sorted(qid for qtype, qid in expected if qid not in table.get(qtype, {}))
    provenance = {"expected_items": len(expected),
                  "rescored_ids": {q: s for q, s in source.items() if s != "main"},
                  "still_unscored_ids": still_missing}
    return table, provenance, [{"expected_items": [list(e) for e in expected], "samples": rows}]


def _describe(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 4) if values else None,
        "sd": round(statistics.stdev(values), 4) if len(values) > 1 else None,
    }


def _bootstrap_ci(diffs: list[float], *, seed: int, rounds: int = 10_000) -> list[float] | None:
    if len(diffs) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(rounds))
    return [round(means[int(0.025 * rounds)], 4), round(means[int(0.975 * rounds) - 1], 4)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="ours")
    parser.add_argument("--b", default="ragu")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    (a, provenance_a, rows_a), (b, provenance_b, rows_b) = _scores(args.a), _scores(args.b)
    records = {label: json.loads((ROOT / "results" / "benchmark_eval" / label / "eval_record.json").read_text(encoding="utf-8"))
               for label in (args.a, args.b)}
    summary: dict = {
        "systems": {args.a: "this project", args.b: "RAGU v0.1.0 LocalSearchEngine"},
        "note": ("One run per system on a 10% Medical slice of GraphRAG-Bench (39 questions; 3 Creative Generation are "
                 "descriptive only). Judge gpt-6-luna via the documented proxy (temperature omitted), similarity "
                 "embeddings text-embedding-3-large. Not comparable with the public leaderboard."),
        "eval_status": {label: rec["status"] for label, rec in records.items()},
        "judge_cost_usd": {label: rec["judge_ledger"]["usage"]["cost_usd"] for label, rec in records.items()},
        "dropped_in_main_run": {label: {t: v["missing_ids"] for t, v in rec["reconciliation"]["types"].items()
                                        if v["missing_ids"]} for label, rec in records.items()},
        "completeness_after_supplements": {args.a: provenance_a, args.b: provenance_b},
        "by_type": {},
    }
    for qtype in sorted(set(a) | set(b)):
        metrics = sorted({m for s in list(a.get(qtype, {}).values()) + list(b.get(qtype, {}).values()) for m in s})
        entry: dict = {}
        for metric in metrics:
            va = {q: s[metric] for q, s in a.get(qtype, {}).items() if metric in s}
            vb = {q: s[metric] for q, s in b.get(qtype, {}).items() if metric in s}
            both = sorted(set(va) & set(vb))
            diffs = [va[q] - vb[q] for q in both]
            entry[metric] = {
                args.a: _describe(list(va.values())),
                args.b: _describe(list(vb.values())),
                "paired": {"n": len(both), "mean_difference_a_minus_b": round(statistics.mean(diffs), 4) if diffs else None,
                           "bootstrap_95ci": _bootstrap_ci(diffs, seed=args.seed),
                           "a_higher": sum(d > 0 for d in diffs), "b_higher": sum(d < 0 for d in diffs),
                           "ties": sum(d == 0 for d in diffs)},
            }
        summary["by_type"][qtype] = entry

    # Pooled answer correctness over the three fully scored, analyzable types.
    core = ("Fact Retrieval", "Complex Reasoning", "Contextual Summarize")
    pooled = {}
    for label, table in ((args.a, a), (args.b, b)):
        pooled[label] = {q: s["answer_correctness"] for t in core for q, s in table.get(t, {}).items() if "answer_correctness" in s}
    both = sorted(set(pooled[args.a]) & set(pooled[args.b]))
    diffs = [pooled[args.a][q] - pooled[args.b][q] for q in both]
    summary["pooled_answer_correctness_core_types"] = {
        args.a: _describe(list(pooled[args.a].values())),
        args.b: _describe(list(pooled[args.b].values())),
        "paired": {"n": len(both), "mean_difference_a_minus_b": round(statistics.mean(diffs), 4) if diffs else None,
                   "bootstrap_95ci": _bootstrap_ci(diffs, seed=args.seed)},
    }

    # This project's scores split by the "insufficient evidence" flag.
    answers_file = DATA / "runs" / args.a / "answers.json"
    if answers_file.is_file():
        flagged = sorted(r["id"] for r in json.loads(answers_file.read_text(encoding="utf-8")) if r.get("status") == "abstained")
    else:
        flagged = json.loads(PER_SAMPLE.read_text(encoding="utf-8"))["flagged_insufficient_ids"][args.a]
    split = {"flagged_insufficient": [], "not_flagged": []}
    for q, score in pooled[args.a].items():
        split["flagged_insufficient" if q in flagged else "not_flagged"].append(score)
    summary["ours_answer_correctness_by_insufficient_flag_core_types"] = {k: _describe(v) for k, v in split.items()}

    PER_SAMPLE.write_text(json.dumps({
        "description": "Text-free per-sample scores (question ids, types, metric values, main run or re-score). "
                       "Lets results/benchmark_scores_summary.json be recomputed without the git-ignored evaluator outputs.",
        args.a: rows_a[0], args.b: rows_b[0],
        "flagged_insufficient_ids": {args.a: list(flagged)},
    }, indent=2), encoding="utf-8")
    out = ROOT / "results" / "benchmark_scores_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
