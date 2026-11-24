# GraphRAG-Bench investigation

Written 2026-11-24. Status: investigation plus a measured data download (section 10). No evaluator code has been run and no
API calls were made for this document.

## How to read the evidence levels

The facts below come from three read-only web reads made by sub-agents (the GitHub repository and
Hugging Face dataset card, the arXiv paper, and the evaluator source). I did not re-read the pages
myself. Items are marked **[verbatim]** when the sub-agent quoted them from a page, **[summary]** when
it saw only a model-written summary of that part of a page, and **[not found]** when the sources
did not show it. Before the thesis cites any licence or count, read the primary pages (links in section 8).

## 1. Which benchmark

The benchmark used in the RAGU paper's Medical results is **GraphRAG-Bench** from *When to use Graphs
in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation* (Xiang et al.,
arXiv:2506.05690; v3 dated 22 Feb 2026; accepted at ICLR 2026). The task names match the ones in the RAGU
paper (Fact Retrieval, Complex Reasoning, Contextual Summarize, Creative Generation). Repository:
`github.com/GraphRAG-Bench/GraphRAG-Benchmark` (161 commits on `main`; no release tag or commit is
named in the README). Other benchmarks with similar names exist, so cite the arXiv ID.

## 2. Licence and terms

| Item | Finding | Level |
| --- | --- | --- |
| Repository code | **MIT License**, "Copyright (c) 2025 XMU-DeepLIT" | [verbatim] (raw LICENSE file) |
| Dataset / questions licence | No separate licence statement found in the README or `Datasets/`; the Hugging Face page shows only a "license-mit" badge pointing at the code LICENSE, and asks users to cite the paper | [summary] for the HF part |
| Source corpora | Medical: NCCN clinical guidelines. Novel: pre-20th-century Project Gutenberg novels. Paper says only that the data is "publicly available" and "complies with the terms of the source platforms" | [verbatim] |
| Copyright of NCCN text | **Not stated and not checked.** NCCN guidelines are a third-party publication; whether the benchmark may redistribute them under MIT is unclear | [not found] |
| Paper's own ethics claim | Says "All models used in our evaluation are open-source", which conflicts with its use of GPT-4o-mini and GPT-4.1 | [verbatim] |

**Implication.** The code can be used and, if modified, derived under MIT with attribution. The data
licence is unresolved. Until it is, the corpus and questions should be fetched by script into a git-ignored
directory and **not committed or redistributed** in this repository. The researcher should read the
dataset card and NCCN's terms and tell their supervisor; the thesis should cite the paper and repo.

## 3. Dataset

- Files **[verbatim]**: `Datasets/Corpus/{medical,novel}.{json,parquet}` and
  `Datasets/Questions/{medical,novel}_questions.{json,parquet}`. File sizes were not shown.
- Question counts: the paper reports **4,076** questions in total **[verbatim]**; the Hugging Face viewer shows
  "medical (2.06k rows)" and "novel (2.01k rows)" **[verbatim]**, and one summary says 4,072 **[summary]**.
  **Measured after download (section 10): 2,062 Medical questions**, by task: Fact Retrieval 1,098, Complex Reasoning 509, Contextual Summarize 289, Creative Generation 166.
- Question record fields **[verbatim, HF viewer]**: `id`, `source`, `question`, `answer`, `question_type`,
  `evidence`, `evidence_relations` (in the downloaded JSON both are plain strings, not lists as the viewer suggested; section 10). Example id `Medical-73586ddc`, type "Fact Retrieval".
  `question_type` has 4 values.
- Corpus size: Novel is given in tokens (about 56k small, 603k medium, 1,132k full) **[verbatim]**.
  The Medical corpus size was not in the sources; **measured after download: one document of 1,052,159 characters** (section 10).
- Construction: text turned into ontologies with GPT-4.1, questions generated and refined with GPT-4.1,
  questions discarded if no supporting evidence triple can be inferred, with human checking **[summary]**.
- Authors' stated limits: text-only; only a representative subset of GraphRAG methods and models was tested
  **[verbatim]**.

## 4. Official evaluation

Entry point `python -m Evaluation.generation_eval` (also `retrieval_eval`, `indexing_eval`).

| Task type | Generation metrics |
| --- | --- |
| Fact Retrieval | ROUGE-L, Answer Correctness |
| Complex Reasoning | ROUGE-L, Answer Correctness |
| Contextual Summarize | Answer Correctness, Coverage |
| Creative Generation | Answer Correctness, Coverage, Faithfulness |

- **Answer Correctness** = 0.75 x factual correctness (LLM statement extraction and TP/FP/FN
  classification, F1) + 0.25 x semantic similarity (embedding cosine rescaled to [0,1]). 3 LLM calls and
  2 embedding calls per answer. **Coverage**: at least 2 LLM calls. **Faithfulness**: at least 2 LLM calls
  (answer claims against the supplied `context`). ROUGE-L uses no LLM. **[from source reading]**
- **Input format** the code expects: a JSON **list** of items with `id`, `question`, `ground_truth`,
  `generated_answer`, `context` (a **list of strings**) and `question_type`. A missing `context` key aborts
  the run. The README documents no schema.
- Judge: `langchain_openai.ChatOpenAI`, default model `gpt-4o-mini`, key from `LLM_API_KEY`, model and base
  URL configurable on the command line. The paper does not name the judge model **[not found]**.
- Embeddings: a **local HuggingFace model** (`BAAI/bge-large-en-v1.5` by default, configurable), downloaded at
  run time. No OpenAI embedding endpoint is used.
- Dependencies: pinned `langchain`, `langchain_openai`, `rouge_score` and others; `langchain_community`,
  `sentence-transformers`/`torch` and `aiohttp` are **not listed** but appear to be needed.

## 5. Blocking compatibility problems with the project's models

1. **The evaluator always sends `temperature=0.0`** (plus `top_p=1`, `seed=42`, penalties 0) in API mode, with no
   option to omit it. This project's live probe (2026-11-24) showed `gpt-6-luna` **rejects `temperature=0.0`**
   (only the default 1 is supported). With `gpt-6-luna` as the judge, every judge call would fail.
2. **Failures are silent.** If a call fails, `answer_correctness` drops the whole sample (including its ROUGE
   score); coverage and faithfulness return NaN, which is excluded from averages. The run prints errors but
   still reports averages, so an evaluation could look complete while scoring far fewer samples than intended.
   Any run must count dropped and NaN samples explicitly.
3. **Embeddings differ from the project's.** The semantic-similarity half of Answer Correctness uses a local
   `bge-large` model, not `text-embedding-3-large`. Downloading it (about 1.3 GB, not yet measured) needs the
   researcher's permission.
4. **Published numbers are not comparable.** The benchmark paper's Table 2 reports (GPT-4o-mini, Medical,
   HippoRAG2) Fact Retrieval accuracy 66.28 **[verbatim]**, while the RAGU paper reports 72.4 for HippoRAG 2.
   The two papers used different generation and indexing models and settings, so no number from either paper
   can serve as a target for this project's runs. Comparisons must be run under identical conditions here.

## 6. Options for the judge

| Option | What changes | Cost of choosing it |
| --- | --- | --- |
| **A. Patched official evaluator, judge = `gpt-6-luna`** (recommended) | Omit the unsupported request parameters; keep the metric logic and prompts. Document as a deviation from the published evaluator | Needs a small live probe of `top_p`, penalties and `seed` on `gpt-6-luna` (a few hundredths of a cent); results are not comparable with the public leaderboard but are consistent with the project's model configuration |
| B. Unmodified official evaluator, judge = `gpt-4o-mini` | Judge differs from the project's required model | Closer to the published protocol, but conflicts with the project's single-model rule; the researcher must explicitly approve this substitution |
| C. Re-implement the metrics | Full control | Most work and the largest deviation; not recommended before 30 November |

For any option, **the same judge and embedding model must score both systems**, and the judge must be
recorded in every result. Embeddings for the similarity term: either download `bge-large` (official) or use
`text-embedding-3-large` (project-consistent but a documented deviation).

## 7. Proposed subset and plan (revised with measured corpus size)

**Questions:** the **Medical** domain only (matches the RAGU paper's reported domain), a **stratified random sample by
`question_type`** with a fixed sampling seed and recorded question IDs (for example 10 per type = 40 questions;
Creative Generation reported separately, since one other paper excluded it as hard to score).

**Corpus: use the full Medical corpus.** It is one 1,052,159-character document, and the questions cannot be
mapped to positions in it (only about 1% of evidence strings occur verbatim, section 10). A truncated or
sampled corpus would leave many sampled questions unanswerable, so abstentions would be "correct" and the
metrics would change meaning. A truncated corpus is therefore **not recommended** unless questions are
re-selected by a documented answerability check, which adds cost and a new methodology step.

**Projection for the full corpus** (extrapolated linearly from one 10-page document, so uncertain, perhaps by
a factor of 1.5; chunk counts are exact):

| | This project | RAGU |
| --- | --- | --- |
| Chunks (own chunker) | 1,321 (1000 chars, 200 overlap) | 1,123 (SimpleChunker 1000, overlap 0) |
| Build cost | about $0.7 to $1.5 (not measured; guess from token sizes) plus about $0.05 embeddings | about **$10.6** (measured $0.00941 per chunk x 1,123); plausible range $7 to $16 |
| Build time | about 8 h as the extraction runs today (about 22 s per chunk, sequential); about 1 to 2 h with bounded concurrency (not yet implemented) | about 6.7 h at 4 concurrent requests (measured 21.5 s per chunk); less with higher concurrency |
| Rate-limit risk | unknown | unknown |

Question answering for 40 questions on two systems costs a few cents (about $0.002 per answer). Judging adds
about 7 LLM calls for Creative Generation answers and 3 to 5 for others, plausibly $0.15 to $0.50 in total
(unmeasured). **Rough all-in budget: about $12 to $13, range $9 to $18**, dominated by the RAGU build. Each
paid step still needs its own estimate and approval, and RAGU builds can be capped with `--max-usd`.

**Work needed before the builds:** (a) a preparation script that converts the benchmark JSON into the
documents and question sample both systems use (with the data version recorded); (b) bounded concurrency in this
project's extraction; (c) an evaluation adapter that maps both systems' outputs into the evaluator's input
format (`context` must be a list of strings; RAGU's retrieved chunks and this project's evidence chunks can fill it);
(d) the judge decision in section 6.

## 8. Sources (primary pages to read before citing)

- Paper: arXiv 2506.05690 (HTML v3: `arxiv.org/html/2506.05690v3`).
- Repository and licence: `github.com/GraphRAG-Bench/GraphRAG-Benchmark` (and its `LICENSE`).
- Dataset card: `huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench`.
- Evaluator source: `github.com/GraphRAG-Bench/GraphRAG-Benchmark/tree/main/Evaluation`.

## 9. Decisions and permissions needed from the researcher

1. ~~Download permission~~ granted and done (section 10).
2. Judge option A, B or C (section 6), and embedding choice for the similarity term.
3. Whether the benchmark licence and NCCN terms are acceptable to the supervisor, and whether to wait for that
   before committing anything derived from the data.
4. Subset size (questions per type), and approval of a budget of roughly $12 to $13 (range $9 to $18) for the
   full-corpus builds, mostly the RAGU build.

## 10. Downloaded data and measured facts (2026-11-24)

Downloaded with `curl` from `raw.githubusercontent.com`, **pinned to repository commit
`fdbab5959b18c96532580877ffe27d112bccc0ec`** (committed 2026-06-07), into the git-ignored directory
`data/benchmarks/graphrag_bench/` (licence unresolved, section 2). Sizes match the server's `Content-Length`.
The JSON was parsed as inert data only.

| File | Source path | Bytes | SHA-256 |
| --- | --- | --- | --- |
| `medical.json` | `Datasets/Corpus/medical.json` | 1,055,437 | `e15c3b8af5931610c005cbf465e58c2b9ff3b9fa83b7c4bfb042bd8343cdf002` |
| `medical_questions.json` | `Datasets/Questions/medical_questions.json` | 1,966,196 | `ca6ffc4217c892777bd3ba3009e5539716f3b3213aec9c1b04476cdf7c5fb620` |

- **Corpus:** a dict with `corpus_name` ("Medical") and `context`: **one string of 1,052,159 characters**. Under this
  project's chunker (1000 chars, overlap 200) it makes **1,321 chunks**; under RAGU's `SimpleChunker(1000, 0)`,
  **1,123 chunks**.
- **Questions:** a list of **2,062** items, all `source = "Medical"`, 2,062 unique ids. By `question_type`:
  Fact Retrieval **1,098**, Complex Reasoning **509**, Contextual Summarize **289**, Creative Generation **166**.
  Fields: `id`, `source`, `question`, `answer`, `question_type`, `evidence`, `evidence_relations`.
- **`evidence` and `evidence_relations` are strings**, not lists. Median `answer` length 97 characters, maximum 2,769.
  Example questions: Fact Retrieval "What is the most common type of skin cancer?"; Creative Generation asks the
  model to role-play a dermatology attending.
- **Evidence cannot be located in the corpus.** In a random sample of 200 questions, only 2 of 200 evidence strings
  occurred verbatim in the (whitespace- and case-normalized) corpus, none for Complex Reasoning, Contextual
  Summarize or Creative Generation. The evidence is paraphrased or generated, so there is no question-to-position mapping.
- Counts agree with the Hugging Face viewer (2.06k rows) and not with the 4,072 in one summary (that figure is
  probably both domains).

## 11. Reduced-cost plan after the budget limit (2026-11-24)

The researcher set a ceiling of USD 2.50 for the remaining paid work (`docs/budget.md`), so the full-corpus
builds (about $12) are not used. Instead:

1. `scripts/benchmark_prepare.py embed` embeds the 1,321 corpus chunks and the 2,062 evidence strings (about
   $0.06 estimated; the dry run is free) so each question's evidence can be matched to its most similar chunk.
2. `scripts/benchmark_prepare.py select` (offline) chooses a contiguous window of about 10% of the chunks that
   holds the most well-matched questions per task type, and draws a seeded stratified sample (default 10 per
   type, seed 42, minimum similarity 0.5, to be calibrated from the printed similarity distribution). It writes
   the slice text, the selected questions and a spot-check file for manual review of the matches.
3. Both systems then index the **same slice text**, and answer the same selected questions.

Limitations to disclose: the question-to-chunk mapping is similarity-based and approximate; multi-hop and
summary questions draw on evidence spread across the corpus, so a slice only partly covers them; a 10% corpus
has far fewer distractors, so retrieval is easier than on the full benchmark and absolute scores are not
comparable with published ones. Pure selection logic is unit-tested (`tests/test_benchmark_selection.py`).

Also added: bounded concurrent extraction and an optional `--max-usd` cap for this project's build
(`scripts/process_papers.py --concurrency N --max-usd X`); concurrency 1 keeps the original sequential path,
concurrent runs give the same graph, and the first failure cancels all other in-flight requests (4 tests).

## 12. Judge deviation: omit `temperature` only (decided 2026-11-24)

**Decision (researcher):** keep `gpt-6-luna`'s default reasoning configuration and omit `temperature=0.0` from
the evaluator's requests, rather than changing `reasoning_effort` to accommodate the evaluator. The researcher
reports that the official OpenAI documentation says temperature must be omitted unless reasoning effort is
`none`; this project has **not** verified that statement itself. What this project verified is only that
`temperature=0.0` is rejected under the default configuration (live probe, 2026-11-24).

**Implementation: no edit to the upstream evaluator.** `src/benchmarks/judge_proxy.py` is a loopback proxy
(`127.0.0.1` only) that the unmodified evaluator reaches through `--base_url`. Its rules live in
`config/judge_policy.json` (the policy hash is recorded in every ledger):

| Requirement | How it is met | Test evidence |
| --- | --- | --- |
| Temperature omitted from every judge request | The proxy removes exactly the parameters in `omit_parameters`, currently only `temperature`, before anything leaves the machine | A request carrying `temperature=0.0` reaches the fake upstream without it, through raw HTTP and through the OpenAI SDK |
| No other parameter silently changed | Every other field is forwarded unchanged; nothing is added, so `reasoning_effort` stays at the model default | The forwarded payload equals the evaluator payload minus `temperature`; `reasoning_effort` is absent |
| Same judge and configuration for both systems | The proxy refuses any model other than the configured judge (the policy must equal `config/experiment.json`'s model), and one policy file and hash apply to every run | Other-model requests are refused and never forwarded; a policy with a different model fails to load |
| Deviation documented | This section, `config/judge_policy.json` and the ledger's policy hash | n/a |
| Failed requests counted | The ledger records every request, refusal, upstream non-2xx response (with its redacted error message) and network error | Failed upstream responses and network errors appear in the ledger counts |

Also enforced: only `/v1/chat/completions` is accepted (the evaluator's embeddings run locally), streaming is
refused, the real API key is held by the proxy and never sent to the evaluator process, and a **hard spend cap**
refuses all calls once reached (calls in flight can overshoot by at most the evaluator's concurrency, 3).

**Still unverified or not yet done:**
- The evaluator also sends `top_p=1`, `seed=42`, `presence_penalty=0` and `frequency_penalty=0`, forwarded
  unchanged. Their acceptance by `gpt-6-luna` is untested (only `seed` was probed, with a different value). If
  the API rejects any, the ledger will show failed requests and the policy must be revised deliberately. A
  four-request probe would cost about $0.00004, but it needs approval.
- The evaluator drops a *sample* when an LLM call inside Answer Correctness fails. The ledger counts failed
  requests but does not map them to samples. The evaluation runner (not yet written, because the evaluator
  code has not been downloaded) must compare the number of items in with the number scored, and fail
  loudly on any shortfall.
- Results are **not comparable with the public leaderboard** (different judge, no `temperature`, different
  embedding model unless `bge-large` is used).
- Evaluator code is not yet downloaded; running it requires permission and pinning to a commit. (Done later: section 13.)
- Tests use a fake upstream only. No paid evaluator call has been made.

## 13. Evaluator code downloaded and audited (2026-11-24)

Downloaded with `curl` from `raw.githubusercontent.com`, **pinned to commit
`fdbab5959b18c96532580877ffe27d112bccc0ec`**, into the git-ignored directory `third_party/GraphRAG-Benchmark/`
(created empty for this purpose). Only the 17 text files below were fetched (about 90 KB); the 26
compiled `.pyc` files in the repository were deliberately not downloaded. Nothing has been executed.
`LICENSE` was read directly: MIT, "Copyright (c) 2025 XMU-DeepLIT".

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `Evaluation/generation_eval.py` | 11,168 | `8c5a65224ee9df84cf5268ec5ce95d45a8d1c918f5a0d08ce6de2384ae861674` |
| `Evaluation/indexing_eval.py` | 15,512 | `a890534450cc61d678fcb9455043b45271e1350f75de945d80625fb870e70ad3` |
| `Evaluation/llm/__init__.py` | 53 | `a4be78b8d4972a02ecbb68caab57ded9396478ee3c0d48fdf0a83ee5f93687cc` |
| `Evaluation/llm/ollama_client.py` | 4,464 | `8f7c4d0e242cfa84bcd0acea0a5ccb36974974a03007fa0d9e17569edf508b95` |
| `Evaluation/metrics/__init__.py` | 426 | `5cdd3923d9847c089b0850444337a88dbe1e5f5adbeb20cf919a14742bdb22e8` |
| `Evaluation/metrics/answer_accuracy.py` | 10,859 | `3fec93df5521d71ca0cca5b6acb8b149bac78db91e5d92766cb280907ba9f98b` |
| `Evaluation/metrics/context_relevance.py` | 5,147 | `631af99540a042aa170478b65d5f143b73d1cac0706a5ab02df4cec5dc508984` |
| `Evaluation/metrics/context_relevance_v2.py` | 6,062 | `87081096bafbec99e58006c5e7803968737fc6daf33540ad330966ccff4780ac` |
| `Evaluation/metrics/coverage.py` | 5,842 | `d0019ef32dd3d352e7f895b030c18aa31b85cbf40907dee34c2a6779eaf6f5b9` |
| `Evaluation/metrics/evidence_recall.py` | 4,138 | `7fdf6115ac009453843b64f3c447267fc39ccf0847215736967e32ec6fc58ff6` |
| `Evaluation/metrics/faithfulness.py` | 8,056 | `9f4c127edbb673b516228e5bf991f02fc8b7f800a0d27e7cce249d06402158eb` |
| `Evaluation/metrics/rouge.py` | 1,113 | `d93ef904bc13bc798c740cfff08ebe1fc56318b90bfd058b31d169888eb2de5c` |
| `Evaluation/metrics/utils.py` | 4,206 | `f461b55bc5c4a63b5d36a1edc27ced9e42a5041b2bb38000ba375cc7b25acc66` |
| `Evaluation/README.md` | 2,403 | `041a07bf192d61c7e678eed97e8c08433233d82f08b039c45e0b351a903ea724` |
| `Evaluation/retrieval_eval.py` | 10,099 | `328773cf9ed3fe22fa746b5e07688c46432c9601f6072b51b936c585e0aacee1` |
| `LICENSE` | 1,068 | `f607254fa7fd8fd9e0ab9c2199ee568c44232053f5affa5130a5330b7bbf148a` |
| `requirements.txt` | 191 | `20ee70e3612765c586438338ef7df94e22ddbcd3a55ff5a5e679fd7cf826f386` |

**Static audit (read, not run):**
- No `exec`, `eval`, `subprocess`, `os.system`, sockets or hard-coded credentials. The only network access is the
  configured LLM `--base_url` and the optional Ollama client. File access is limited to reading `--data_file` and
  writing `--output_file`.
- `Evaluation/indexing_eval.py` calls `pickle.load` on graph files supplied by the user. **This project will not
  run `indexing_eval.py`**, and no untrusted pickle is ever loaded.
- `requirements.txt` lists `Evaluation==0.0.2`, a PyPI package with the same name as the local folder. It is
  **not this code and must not be installed**; the evaluator is run from the downloaded folder. `ragas` is pinned but is not imported by the generation path and is not needed. (Correction: `datasets` IS imported,
  as `from datasets import Dataset`; see section 15 for how that is handled.)

**Claims verified first-hand against the source** (previously from a sub-agent): `temperature=0.0` is set
unconditionally and `top_p=1`, `seed`, `presence_penalty=0`, `frequency_penalty=0` are also sent
(`generation_eval.py` lines 151 to 166); the similarity embedding is a local `HuggingFaceBgeEmbeddings(model_name=
args.embedding_model)`; the input items are read as `id`, `question`, `ground_truth`, `generated_answer`,
`context` and `question_type`; a failed sample prints "Sample failed" and is dropped, and averages use `np.nanmean`.

**Consequences for the adapter and setup:**
- Items must use the key **`ground_truth`** (the benchmark's own files call it `answer`), and `context` must be a
  list of strings.
- `generation_eval.py` imports `langchain_community` and `langchain_ollama` at module level, so both must be
  installed even in API mode, and `langchain_community` is not in `requirements.txt`.
- Embeddings for the similarity term are an open decision. (1) Official `BAAI/bge-large-en-v1.5`: no API cost, but
  needs `sentence-transformers`, PyTorch and a model download of about 1.3 GB. (2) `text-embedding-3-large` through
  a documented runtime substitution of the embedding class (the proxy would also have to allow the embeddings path,
  metered): project-consistent, a negligible cost per answer, lighter to install. Either way results are not
  comparable with the public leaderboard.

## 14. Mapping and slice selection (executed 2026-11-24)

**Embedding step (paid, approved cap $0.10):** `scripts/benchmark_prepare.py embed --execute --max-usd 0.10`
embedded the 1321 corpus chunks and 2062 evidence strings with
`text-embedding-3-large`. **Measured: 384,412 tokens in 35
requests, $0.049974** (estimate was $0.06; the characters-divided-by-4 rule overestimates by about 17%).
No errors. The run record is `data/benchmarks/graphrag_bench/mapping_manifest.json` (dirty tree, base commit `1a0ca56`).

**Calibration (free).** Best-match similarity per question ranged from 0.58 (10th percentile) to 0.79 (90th), so the
initial placeholder threshold of 0.5 filtered nothing and ties made the window default to the start of the
document. A spot check of 12 matches showed the matched chunks are topically right (same cancer type and
treatment), but each question's evidence is several short statements joined by semicolons that can come from
different places, so one best chunk cannot show the whole evidence is present. A free lexical check measured
the share of each statement's content words that occur in the slice: 0.85 to 0.96 on average for the selected questions
against 0.66 to 0.70 for random windows of the same size, so the selection beats chance, but 8 of the first 40 selected
questions still had a weak statement. Selection therefore applies two **heuristic** filters (thresholds are not
validated against labels, only by this check): embedding similarity at least **0.6**, and
**every** evidence statement having at least **0.75** of its content words in the slice.
The window search is run over all window positions (stride 5 chunks) using both filters.

**Result** (`results/benchmark_selection_summary.json`; parameters: window = 10% = 132 chunks,
seed 42, per-type target 12):

| | |
| --- | --- |
| Slice | chunks 100 to 232, characters 79,695 to 185,073 (105,378 characters, 10% of the corpus) |
| Questions eligible in the slice (both filters) | {'Complex Reasoning': 46, 'Contextual Summarize': 30, 'Creative Generation': 3, 'Fact Retrieval': 132} |
| In-slice questions removed by the lexical filter | {'Complex Reasoning': 21, 'Contextual Summarize': 15, 'Creative Generation': 29, 'Fact Retrieval': 16} |
| **Selected (39)** | {'Complex Reasoning': 12, 'Contextual Summarize': 12, 'Creative Generation': 3, 'Fact Retrieval': 12} |
| Shortfall against the target of 12 per type | {'Complex Reasoning': 0, 'Contextual Summarize': 0, 'Creative Generation': 9, 'Fact Retrieval': 0} |

**Creative Generation is effectively out of reach for a 10% slice:** these questions need about 7 evidence statements
each, spread across the corpus, and only 3 qualify even in the best window. Those 3 are kept but are
descriptive only (no inference). Because Creative Generation is the only type scored with Faithfulness,
faithfulness cannot be meaningfully analyzed on this slice.

**Limitations to disclose:** the question-to-slice mapping is heuristic and approximate, and some selected questions may still
lack complete supporting evidence in the slice; a 10% corpus has far fewer distractors, so retrieval is
easier than on the full benchmark; 39 questions (12 per analyzable type) give wide uncertainty; sample
selection used embedding similarity computed with the same model as one of the systems under test (a possible
mild bias toward that model's retrieval, since questions were chosen for being easy to match). Selected
question IDs are recorded in the summary file; the question and corpus text stay in the git-ignored data directory.

## 15. Evaluator environment, deviations and mocked end-to-end evidence (2026-11-24)

**Environment** (isolated venv outside the repository, Python 3.14.4; lock in `config/bench_eval_env_requirements.txt`):
the pinned `langchain 0.3.26`, `langchain-core 0.3.69`, `langchain-openai 0.3.28` and `datasets 3.3.2` are installed.
`langchain-community 0.3.27`, `langchain-ollama` and `aiohttp` were added because the evaluator imports them without listing them.
The unrelated PyPI package `Evaluation` is **not** installed, and neither are `ragas` and PyTorch.

**Deviations from the official evaluator** (in addition to the judge policy in section 12):

| Deviation | Reason | Effect on scores |
| --- | --- | --- |
| `pydantic` is 2.14.4 instead of the pinned 2.11.7 | 2.11.7 needs `pydantic-core 2.33.2`, which has no Python 3.14 wheel; pip tried to compile it from Rust and hung | None expected; used for message schemas only |
| `datasets.Dataset` replaced at run time by `src/benchmarks/eval_shims.InMemoryDataset` | `datasets 3.3.2` pins an old `dill` whose pickler raises `TypeError: Pickler._batch_setitems() takes 2 positional arguments but 3 were given` on Python 3.14, so the real evaluator failed on the first item | None; the evaluator only uses `from_dict`, column access and `select` (3 unit tests) |
| `HuggingFaceBgeEmbeddings` replaced by OpenAI `text-embedding-3-large` through the proxy | researcher decision (section 12) | Changes the semantic-similarity term of Answer Correctness; not leaderboard-comparable |
| `temperature` omitted from judge requests | section 12 | Judge sampling differs from the published evaluator |

**Mocked end-to-end evidence** (real evaluator code, fake API, zero cost; reproduced by `tests/test_eval_runner_e2e.py`,
which is skipped if the environment is missing). Clean run, 8 items (2 per type): 52 upstream requests (36 chat, 16
embedding); `temperature` was removed from all 36 chat requests and the fake upstream independently confirms none of the
52 requests contained it; all other fields (`messages`, `model`, `top_p`, `seed`, `presence_penalty`, `frequency_penalty`,
`stream`) were forwarded unchanged; only `gpt-6-luna` and `text-embedding-3-large` were requested; reconciliation complete.
Forced-failure run: one judge call answered HTTP 400; the evaluator printed "Sample failed" and **silently dropped the
whole sample, including its ROUGE score**, which the reconciliation reported exactly (expected 2, returned 1,
missing id), the ledger counted the failed request, and the runner exited with code 2. The scores in these runs are
meaningless (fake responses) and must never be reported.

**Other runner facts:** the real API key is captured and removed from the environment before the third-party evaluator
code runs (the evaluator process holds only a placeholder). Still unverified live: acceptance by `gpt-6-luna` of
`top_p=1`, `presence_penalty=0`, `frequency_penalty=0` and `seed=42` (a 4-request probe, about $0.00004, awaits approval).

## 16. Slice builds (executed 2026-11-24)

**RAGU** (`results/ragu_build_graphrag_bench_slice.json`, cap $1.40): complete, **$0.754863** measured (estimate $1.06),
1,599 s (26.7 min) at 4 concurrent requests, 536 chat calls (all `beta.chat.completions.parse`) and 6 embedding calls,
0 errors; 3,547,165 prompt / 778,326 completion (370,532 reasoning) / 84,486 embedding tokens; graph of **599 entities,
706 relations, 65 communities, 113 chunks**. Graph in `data/ragu/graphrag_bench_slice/` (git-ignored).

**This project, run 1** (`results/pipeline_graphrag_bench_slice_run1_artifacts_lost.json`, renamed from `pipeline_graphrag_bench_slice.json` so it is not mistaken for a result; cap $0.30, 4 concurrent requests): extraction of all
132 chunks completed in 391 s (6.5 min) with **$0.145711 measured** (the first measured cost of this project's extraction:
132 chat calls, 78,202 prompt and 268,422 completion tokens, of which 45,273 reasoning, plus 28,305 embedding tokens),
giving 1,022 unique entities and 412 explicit edges. **Saving the artifacts then failed**: the final atomic folder rename
raised `PermissionError: [WinError 5] Access is denied` (a transient lock on a freshly written folder inside OneDrive), and
the save routine's cleanup deleted the fully written staging copy, so the graph and vectors were lost; only the summary
JSON survives. **Fix:** `save_artifacts` now retries the rename with back-off and, if it still fails, keeps the
hash-verified staging copy and raises `ArtifactSaveError` naming it (3 new tests, including recovery by loading the kept
copy). Run 1's summary is kept as the record of a lost run; it is not an experimental result to compare.

**This project, run 2** (`results/pipeline_graphrag_bench_slice_run2.json`, approved rerun, cap $0.30, 4 concurrent requests):
complete and **saved**; **$0.143365 measured** (132 chat calls, 78,202 prompt and 263,730 completion tokens, of which 47,037
reasoning, plus 28,305 embedding tokens), 397 s (6.6 min); **1,010 unique entities, 368 explicit edges, 132 chunks**.
Artifacts in `data/processed/graphrag_bench_slice/` reload with all hashes verified (132 chunks, 132 vectors of 3,072
dimensions, 1,010 graph nodes, 368 edges, matching the summary). Run 1 and run 2 differ (1,022 against 1,010 entities;
412 against 368 edges) because LLM extraction is not deterministic without a seed. Run 2 is the graph used for the evaluation.

Build comparison on the same slice text (both measured, both at 4 concurrent requests; one sample each): RAGU $0.754863 in
26.7 min with 536 chat calls; this project $0.143365 in 6.6 min with 132 chat calls. The difference reflects RAGU's
four-stage extraction with validation, summarization and community summaries against this project's single extraction
pass, so it is a property of the two designs as configured, not a like-for-like efficiency result. Graph counts are not
comparable as quality measures, since the two systems define and merge entities differently.

## 17. Answering the 39 selected questions (executed 2026-11-24)

Both systems answered the same 39 questions on their slice graphs, one question at a time, with default settings,
no request parameters sent, dirty tree (base commit `1a0ca56`). Summaries: `results/benchmark_answers_ours.json`,
`results/benchmark_answers_ragu.json`; raw answers and evaluator inputs in the git-ignored
`data/benchmarks/graphrag_bench/runs/`.

| Measure | This project | RAGU (LocalSearchEngine, defaults) |
| --- | --- | --- |
| Questions with an answer record | 39 of 39, 0 errors | 39 of 39, 0 errors |
| Flagged "insufficient evidence" | **16** (Fact Retrieval 2, Complex Reasoning 5, Contextual Summarize 6, Creative Generation 3) | not applicable (no such mechanism) |
| Fail-closed (all citations rejected) | 0 | not applicable |
| Citations accepted / proposed | 131 / 139 (rejected: 7 invalid chunk id, 1 wrong chunk id; no quote-text rejection) | none produced |
| Prompt / completion (reasoning) tokens | 336,671 / 36,897 (9,277) | 266,815 / 13,043 (6,019) |
| Cost (measured) | **$0.052214** (cap $0.15) | **$0.033302** (cap $0.20) |
| Mean latency per question | 9.6 s | 4.6 s |
| Answer call | `beta.chat.completions.parse` (structured) | `chat.completions.create` (plain text) |

**The 16 "insufficient evidence" answers are partial answers, not refusals.** Every one contains substantive text
(162 to 2,962 characters, median about 700) supported by verified citations; the model set the flag because part of
the question was not covered by the retrieved evidence. They are passed to the evaluator unchanged and scored like
any other answer. Answered questions carried 2.48 accepted citations on average.

Observations (one run each; not yet scored): per question, this project used about 8.6k prompt tokens and RAGU about
6.8k (the reverse of the single Q1 comparison on the Komarov paper, where the graphs and contexts differed); this
project generated about 2.8 times as many completion tokens, consistent with its structured output carrying quotes
and claims. All 39 of this project's quote checks either passed or failed only on chunk identifiers, so the quote
normalization concern from Q1 did not occur in this run.

## 18. Scoring results (executed 2026-11-24)

**Pilot** (2 items of this project's answers, cap $0.02, `results/benchmark_eval/_validation/pilot_ours/` (a validation run, not a result)): 12 requests, all HTTP 200,
`temperature` removed from all 8 judge calls, $0.002012. This **verified live that `gpt-6-luna` accepts the evaluator's
`top_p=1`, `seed=42`, `presence_penalty=0` and `frequency_penalty=0`**, which had been unverified. (Recorded here rather than
in `config/judge_policy.json`, so that the policy hash stays identical across all scoring runs.)

**Full scoring** (`results/benchmark_eval/{ours,ragu}/eval_record.json`; same judge, policy hash, evaluator commit and
settings for both): 233 requests per system, **all HTTP 200, 0 failed, 0 refused**; judge plus embedding cost $0.065534
(this project) and $0.084986 (RAGU). Both runs are marked **incomplete** because of a client-side limit in the official
evaluator: it hard-codes a 30-second request timeout, and a few long Creative Generation judge calls (reasoning
model) took longer, so the evaluator gave up and dropped those samples even though the API later answered. Dropped:
1 Creative Generation sample for this project and 2 for RAGU, all caught by the reconciliation. **Fact Retrieval, Complex
Reasoning and Contextual Summarize are fully scored for both systems (36 of 36 each).**

**Scores** (`results/benchmark_scores_summary.json`, produced offline by `scripts/summarize_benchmark_scores.py`; paired
over the same questions; bootstrap intervals use 10,000 resamples, seed 7):

| Task type (n) | Metric | This project | RAGU | Paired difference (95% interval) |
| --- | --- | --- | --- | --- |
| Fact Retrieval (12) | Answer Correctness | 0.675 | 0.622 | +0.053 (-0.061 to +0.175) |
| Fact Retrieval (12) | ROUGE-L | 0.474 | 0.407 | +0.068 (-0.028 to +0.155) |
| Complex Reasoning (12) | Answer Correctness | 0.521 | 0.509 | +0.013 (-0.056 to +0.076) |
| Complex Reasoning (12) | ROUGE-L | 0.260 | 0.223 | +0.037 (+0.008 to +0.079) |
| Contextual Summarize (12) | Answer Correctness | 0.562 | 0.633 | -0.071 (-0.171 to +0.030) |
| Contextual Summarize (12) | Coverage | 0.597 | 0.761 | -0.164 (-0.354 to -0.002) |
| **Pooled core types (36)** | **Answer Correctness** | **0.586** | **0.588** | **-0.002 (-0.059 to +0.058)** |
| Creative Generation (3 each, after re-scoring; descriptive only) | AC / Coverage / Faithfulness | 0.440 / 0.386 / 0.738 | 0.526 / 0.534 / 0.390 | not analyzable (n=3) |

**This project's insufficient-evidence flag tracks answer quality:** on the core types, answers it flagged scored 0.408
Answer Correctness (n=13) against 0.686 for answers it did not flag (n=23). The flag is informative rather than noise.
This is an observational split of one run, not a causal result.

**How to read these results (limitations):**
- Both systems score clearly above zero, satisfying the assignment's requirement. Overall Answer Correctness is
  **indistinguishable** between the two systems on this sample; no claim that either architecture is better is supported.
- Per-type differences are small samples (n=12) with nine comparisons; the two intervals that exclude zero (higher
  ROUGE-L for this project on Complex Reasoning; higher Coverage for RAGU on Contextual Summarize) are near the boundary and
  could be chance under multiple comparisons. A plausible but untested reading of the Coverage gap is that this
  project's answers are shorter and stricter about evidence, while RAGU's broader context covers more reference facts.
- One run per system, no seed; a 10% corpus slice with fewer distractors than the full benchmark; questions selected by an
  approximate, heuristic mapping; RAGU with untuned defaults and a different context budget from this project; and
  `gpt-6-luna` is both the answering model and the judge for both systems (any self-preference applies to both).
- Deviations from the official evaluator (sections 12 and 15): `temperature` omitted, `text-embedding-3-large` for
  similarity, `pydantic` version, `datasets` replaced by a minimal in-memory class. Scores are not comparable with the
  public leaderboard or with the numbers in the GraphRAG-Bench or RAGU papers.
- The 3 dropped Creative Generation samples were re-scored afterwards (see below).

**Re-scoring the 3 timed-out samples (executed 2026-11-24, approved by the researcher).** `scripts/run_benchmark_eval.py`
gained `--judge-timeout`, which replaces only the evaluator's hard-coded 30-second `timeout` argument (every other judge
setting, the policy file and its hash are unchanged; recorded as `judge_timeout_override_s` in the run record). The 3
dropped Creative Generation questions were re-scored with a 180-second timeout as supplementary runs, stored inside each
system's folder (`results/benchmark_eval/{ours,ragu}/rescore_cg/`): all requests HTTP 200, 0 failed, reconciliation
complete; cost **$0.008525** (this project, 1 question) and **$0.018316** (RAGU, 2 questions). The estimate given beforehand
was "under $0.01" in total, which was too low by about 2.7 times; long Creative Generation judge calls cost more than assumed.
`scripts/summarize_benchmark_scores.py` merges a supplement only into questions the main run did not score (it never replaces
a main-run score) and records which questions were re-scored. **Result: all 39 questions are scored for both systems**
(`completeness_after_supplements` in `results/benchmark_scores_summary.json`). The main runs' own records remain
"incomplete", because that is what they were; the supplements complete the data set. The core-type results above
are unchanged, since Creative Generation is not part of them.

**Folder layout.** `results/benchmark_eval/ours/` and `results/benchmark_eval/ragu/` hold each system's main scoring run and
its `rescore_cg/` supplement; `results/benchmark_eval/_validation/pilot_ours/` holds the 2-item pilot, which is a check of
the pipeline and not a result.
