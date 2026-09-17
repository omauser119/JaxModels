# Jax-1-Abel — 0.2.0-pilot.1

Experimental trained decision model, published by Jax AI. Frozen pretrained Qwen3.5-0.8B Q8_0 plus a 1026-parameter shared residual head. **Not RLCD-trained, not a standalone new foundation model, not production-qualified.**

The release requires both `backbone.gguf` and `decision_head.json`; `config.json` pins hashes and runtime settings. The backbone is unchanged pretrained Qwen; only the decision head was trained for this release. Total disk size is about 812 MB plus metadata. No teacher weights are required for inference.

## Training

Local teacher: Qwen3.5-2B Q4_K_M via llama.cpp. 100 tasks: 70 train, 10 validation, 10 calibration, 10 test. 80 adapted EN/RU MASSIVE rows and 20 authored synthetic rule tasks. Original source group partitions were preserved. Sixteen previously completed teacher answers were reused from the interrupted parent run. Teacher development result: 9/16, below the required 12; training continued as an explicitly experimental run.

Method: teacher-distribution cross-entropy, residual linear head, L2 regularization and post-fit temperature. No backbone fine-tuning and no reinforcement learning. Selected λ=.001, epoch=30, T=2.8533858187665566. See [full architecture](../../docs/ARCHITECTURE.md), config provenance and training_report.json.

## Held-out ground-truth results

| Metric | Value |
|---|---:|
| Tasks / correct | 10 / 7 |
| Accuracy | 70% |
| Wilson 95% interval | 39.7–89.2% |
| NLL | 0.738826 |
| Multiclass Brier | 0.461048 |
| ECE-10 | 0.238934 |
| Diagnostic gate | Failed |
| Production qualification | False |

See `ground-truth-evaluation.json` for all slices and historical baselines. Teacher-agreement results are separate and must not be described as ground-truth accuracy. The confidence interval ignores correlation between translations. Ten tasks, adapted intent labels and synthetic rules do not establish broad generalization. Potential pretraining overlap is unknown. Full DecisionBench and a Jev comparison have not been completed.

## Intended use and limitations

Local experimentation with runtime-defined Choice/Score/Noul questions and head distillation. No guarantee for arbitrary domains, adversarial text or consequential decisions. Independent per-criterion prefills can be slow on CPU; this is not Jev's parallel architecture. Generic GGUF teacher support does not mean arbitrary student-backbone support. CUDA paths have not been hardware-validated in this release.

## Licenses and provenance

Backbone and head: Apache-2.0 (LICENSE). Source code: MIT (LICENSE-code). MASSIVE adaptations: CC-BY-4.0, credited in the dataset directory and NOTICE. Backbone conversion is credited to Unsloth; original model to Qwen contributors. No affiliation with TypeSafe.

Published provenance replaces local absolute paths with `WORKSPACE/`; historical hashes still refer to original training artifacts. The head's internal `model_version: 0.1.0` is a legacy loader contract, not the public release version. Weight bytes and hashes are preserved unchanged.
