# Design note: measuring claim-level evidence support

Status: design only. Nothing here is implemented, and no behaviour of the current pipeline
changes. Written 2026-11-24.

## 1. The gap

Today's guard is **answer-level**. Each model citation is checked mechanically (the cited
chunk was retrieved, and the quote is a whitespace/case-normalized substring of that chunk).
If at least one citation passes, the whole `answer_text` is kept; if none passes, the answer
fails closed. Three consequences:

1. **Verbatim is not the same as supporting.** A quote can be genuine and still not support
   the claim. In the original Q1, citation 1 is a real Table 1 caption but does not contain
   any of the numbers the answer reports. The system counted it as verified.
2. **Claims and citations are not linked.** `answer_text` is free text and citations carry
   only a `claim` string, so nothing records which sentence a citation backs. An answer with
   six statements and one valid citation is kept whole.
3. **Partial verification is invisible.** `answer_diagnostics` records rejected citations, but
   the final answer text is not filtered to match.

So "answer-level citation validity" overstates how verifiable the answer is. It is a necessary
check (it catches fabricated quotes and invented chunk IDs), not a sufficient one.

## 2. Metrics to separate

| Metric | Unit | Definition | How measured |
| --- | --- | --- | --- |
| Answer-level citation validity | answer | At least one citation passes the strict check (the current rule), and optionally all citations pass | Mechanical (exists) |
| Citation precision (mechanical) | citation | Accepted citations / raw citations | Mechanical (exists in diagnostics) |
| Citation support precision | citation | Accepted citations whose quote actually entails the attached claim / accepted citations | Judged |
| Claim-level evidence support | atomic claim | Supported claims / all atomic claims in the answer (also report partial and contradicted) | Judged |
| Citation coverage | atomic claim | Claims with at least one verified, supporting citation / all claims | Mechanical link + judged |
| Abstention rate | question | Share of questions where the system abstains or fails closed; split into correct abstention (evidence truly absent) and false abstention (evidence present) | Mechanical + gold labels |
| Answer correctness | answer | Agreement with the benchmark gold answer | Benchmark metric |

Correctness and support are different axes. An answer can be correct but unsupported (the
model used parametric knowledge) or supported but incorrect (the evidence is wrong or
incomplete). Both should be reported.

## 3. Proposed mechanism (in increasing cost)

1. **Structured claims (schema change).** Have the model emit
   `claims: [{claim_text, citation_indices}]` instead of one free-text answer plus a
   flat citation list. A claim is *verified* only if one of its own citations passes the strict
   quote check. The final answer is assembled from verified claims; the rest are
   **withheld** and recorded with their diagnostics (claim-level fail-closed). This also
   gives the claim-to-citation link that is missing today.
2. **Deterministic consistency checks.** For claims containing numbers, percentages or named
   entities, require each of them to appear in the cited quote (after the normalization used
   for diagnostics). This would have flagged Q1 citation 1, and costs no API calls. It is
   high precision for numeric claims and says nothing about paraphrased ones.
3. **Entailment judgment.** For each (claim, quote) pair, a judge (an LLM or an NLI model)
   labels *supported / partially supported / unsupported / contradicted*. Record the judge's
   model ID and prompt in versioned config. Validate the judge on a human-labelled sample
   (for example 30 to 50 claims, reporting agreement) before trusting aggregate numbers.
4. **Repetition.** Run each question N times and report the mean and spread of every metric,
   because Q1 showed outcomes vary between identical-input runs.

## 4. Fairness with RAGU and the benchmark

Quote-level verification is a property of this system's architecture. Reading RAGU's source
(`docs/ragu-integration.md`, 2026-11-24) shows its query engines return a free-form answer plus
a retrieval context whose chunks have IDs, and no quote-level citations in the answer (RAGU was
not run, so this is source inspection only). A head-to-head comparison on citation metrics is
therefore not available without wrapping RAGU. Where only one system produces citations, a
common footing is needed:

- **Context-support** (claims supported by the retrieved context, whatever the system cites)
  can be judged identically for both systems using the same judge.
- **Citation metrics** can be reported for this system alone and labelled as a property of the
  design, not as a head-to-head win.

Any difference found must be separated into architecture effects and model, prompt or judge
effects, as the project's fairness rules require.

## 5. Threats to validity

- Claim decomposition is itself model-dependent and can change counts.
- A judge can be biased toward fluent text or toward its own model family.
- Strict substring matching may reject correct quotes with PDF artefacts (49 of 54 chunks in
  the development paper contain hyphenated line breaks). Support rates then depend on the
  acceptance rule, so report them alongside the diagnostics breakdown, and decide on any
  normalization change before the final runs, not after seeing results.
- Small question sets give wide uncertainty; report intervals, not just means.

## 6. Suggested scope given the 30 November deadline

Minimum viable (inside the deadline): step 1 (structured claims) and step 2 (deterministic
numeric check), plus a manual support annotation of a small claim sample to estimate
support precision. Steps 3 and 4 are the stronger contribution, and the full-scale version
(LLM judge validated on human labels, repeated runs on the benchmark subset) can be framed
as a limitation or future work if time runs out.

Step 1 changes generation behaviour, so it must ship as a new, separately versioned
experimental configuration. It must not be mixed with the existing Q1 results.

## 7. Open decisions for the researcher

1. Is claim-level support a primary thesis contribution, or a secondary analysis?
2. Which judge, and what budget for judging?
3. How many manually labelled claims are feasible, and is a second annotator available?
4. Whether to keep strict substring acceptance for the final experiments.
