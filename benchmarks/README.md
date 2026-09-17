# Jax DecisionBench 1.0

A reproducible benchmark for future comparisons of **Jev-1 and Jax-1-Abel** on shared, runtime-defined Choice/Score/Noul tasks. Results are reported by domain instead of reduced to a single promotional score. Jev has not been evaluated here; no Jev results are available.

## Data

2,560 requests: 640 source scenarios × 4 variants. Eight families cover inventory availability, exact list membership, explicit/negated/hypothetical requests, latest events, document types, dynamic team routing, completion status and deadline ranges. Languages: English and Russian. Variants: original; renamed question IDs; reordered Choice options plus a distracting instruction embedded in the data; an additional independent question.

Development contains 640 requests; test contains 1,920. All variants of a scenario belong to the same split. These are authored synthetic MIT-licensed tasks with programmatically determined answers, **not 2,560 independent real-world observations**. Templates resemble each other across splits; this benchmark does not establish transfer to external sources. Model inputs exclude reference answers, split labels and scenario IDs: the runner sends only `request.state` and `request.questions`. A neutral `case_reference` field is part of the data and has no relationship to the answer.

`requests.jsonl` contains no targets. Only the evaluator reads `references.jsonl`; the runner never opens it. Both files are pinned by SHA-256 in the manifest. Abel-v1 training data is excluded. New training must use a separate corpus. Once test results guide development, that test no longer provides a fresh independent final evaluation: create a new benchmark version and retain the old one as a regression suite.

## Running Abel

```bash
# Quick diagnostic: 16 original development scenarios, without variants.
python3 benchmarks/run.py --backend abel --split dev --suite smoke --output benchmarks/runs/abel-smoke
# Full test: 1,920 requests; this can take hours on a small CPU.
python3 benchmarks/run.py --backend abel --split test --suite full --output benchmarks/runs/abel-test
# Explicit resume requires the same checkpoint, executable, dataset and settings.
python3 benchmarks/run.py --backend abel --split test --suite full --output benchmarks/runs/abel-test --resume
python3 benchmarks/evaluate.py benchmarks/runs/abel-test --output benchmarks/runs/abel-test/report.json
```

Avoid concurrent local API traffic when measuring speed. The `abel` backend uses a persistent C++ worker without HTTP. For an end-to-end comparison with a remote API, run Abel through HTTP:

```bash
python3 benchmarks/run.py --backend http --model Jax-1-Abel \
  --endpoint http://127.0.0.1:8091/v1/decide --split test --suite full \
  --output benchmarks/runs/abel-http-test
```

## Future Jev evaluation

The optional Python adapter follows the official [TypeSafeClient](https://docs.typesafe.ai/sdk/python/api/clients/sync/client) and [response formats](https://docs.typesafe.ai/sdk/python/api/types/responses). Install `typesafe-sdk` in a separate environment and pin its version; the runner records that version in `run.json`. Your `TYPESAFE_API_KEY` is required. The adapter has been checked against documentation, **not a live Jev API**.

```bash
# Use an actual pinned Jev model ID from models.list(), not a floating latest alias.
python3 benchmarks/run.py --backend jev --model ACTUAL_JEV_MODEL_ID \
  --split test --suite full --output benchmarks/runs/jev-test
python3 benchmarks/evaluate.py benchmarks/runs/abel-http-test \
  --compare benchmarks/runs/jev-test --output benchmarks/runs/comparison.json
```

Running Abel never automatically calls TypeSafe. Keys are not saved in results. Both backends receive the same criteria and state, without backend-specific prompts or intermediate textual reasoning. Settings, IDs and hashes are recorded. A floating `latest` alias is insufficient for reproducible comparisons: record the provider model version and date.

## Metrics

- **accuracy_all** covers every expected task; failures, omissions and invalid responses count as errors. Success rate, failure reasons and accuracy_valid are reported separately.
- **NLL**, multiclass **Brier** and **ECE-10** cover valid responses only, with explicit denominators. NLL probabilities are clamped to a minimum of 1e-12. A good NLL on a small successful subset does not establish whole-system reliability.
- **Score** reports MAE, RMSE and MAE/(number of levels−1). Categorical Score accuracy uses the distribution's argmax, not its rounded expectation.
- **Coverage/accuracy** is reported at thresholds 0.5/0.7/0.9/0.95/0.99. The common confidence measure is max(probabilities), rather than different proprietary `confidence` fields.
- Results are broken down by family, language, primitive and variant. Macro-family accuracy prevents larger families from hiding smaller ones.
- Paired robustness reports decision changes and mean/maximum total variation. Invalid-pair counts are reported separately.
- Paired comparisons report accuracy differences, NLL differences on jointly valid cases, wins/losses and 95% bootstrap intervals over source scenarios (2,000 resamples, fixed seed). Template dependencies between scenarios remain; these intervals do not establish universal superiority.
- End-to-end latency reports p50/p95/p99, with startup measured separately. Concurrency=1; retries=0. Failures remain in latency results with separate validity statistics. Load, throughput and cost require a separate evaluation: local CPU time and provider latency do not isolate architecture speed.

The evaluator checks IDs, request hashes, complete answer sets, ranges, normalization, Choice consistency with argmax and Score consistency with its expectation. It does not normalize or repair invalid answers. Duplicate or unknown IDs are evaluation errors. Missing cases remain in the denominator. Resume does not repeat previously recorded failures; a retry requires a separate run. A truncated line after a crash requires explicit journal repair.

## Published model status

The full DecisionBench evaluation has not been completed for checkpoint 0.2.0-pilot.1. Its 7/10 result comes from the separate pilot training corpus's held-out test, not DecisionBench. There are no Jev results. Older local smoke reports for another checkpoint are not included in this repository.
