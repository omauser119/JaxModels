# Jax-1-Abel architecture — release 0.2.0-pilot.1

This document describes the actual source and released weights, not an inferred implementation of Jev. Public RLCD context and unknowns are in [RLCD.md](RLCD.md).

## End-to-end data flow

```text
state + runtime question descriptions
                 │
     strict request validation
                 │
  independent binary judgment prompts
                 │
 frozen Qwen3.5-0.8B Q8_0 via llama.cpp
                 │
 binary token-logit difference + normalized final hidden vector
                 │
       shared learned residual head
                 │
  temperature + deterministic probability normalization
                 │
     Choice / Score / Noul JSON
```

There is no generated answer text and no chain-of-thought sampling. The model prefills a prompt and reads next-token logits and hidden features. No fixed task-specific label vocabulary is trained: instructions, choice names/descriptions and score criteria arrive with the request. The shared head is reused across all criteria and tasks.

## Prompt construction and isolation

`include/jax/prompt.hpp` constructs a system instruction requesting exactly the binary token `0` or `1`. A JSON payload carries `state`, `instructions`, `criterion` and, for Choice, the alternatives. State is explicitly described as data; this instruction is not a security guarantee against adversarial inputs.

Question IDs label the output only and are not model context. Questions do not consume each other's answers. A Choice judgment sees the candidate and all alternatives. A Score judgment sees only its individual criterion, without neighboring levels or their index; write each description so it is meaningful alone. A Noul judgment asks a yes/no question, optionally with explicit yes/no definitions.

The native implementation runs independent prefills serially using a shared context protected by a mutex. It does not implement Jev's proprietary parallel sampler, a shared-prefix attention tree, or state KV-cache reuse between judgments. Computational cost grows with criteria count and repeated state length. One HTTP request does not mean one neural network forward pass.

## Backbone and features

The frozen student is Qwen/Qwen3.5-0.8B, converted to Q8_0 GGUF by Unsloth. Exact revisions, size and SHA-256 are in `models/manifest.json` and the model config. llama.cpp is pinned to `7ceed8737fdb4eb09b4760e77bd12d38012de5a8`.

For judgment j, the runtime reads:

- d_j = logit(token `1`) − logit(token `0`), using the tokenizer's binary token IDs.
- h_j: final normalized-layer hidden state at the last prompt token, dimension 1024.
- u_j = h_j / ||h_j||₂ (normalization implementation in `llama_backend.hpp`).

Qwen uses the non-thinking prompt path. Native context is 2048 tokens by default. Long requests can exceed this even when their JSON body is below HTTP limits. Other GGUF teachers may use different chat templates; student features remain tied to this exact backbone and prompt contract.

## Learned head and outputs

The shared residual head computes

    z_j = a d_j + wᵀu_j + b

with w ∈ R¹⁰²⁴, scalar a and scalar b: **1026 trainable parameters**. A separate positive scalar temperature T is fitted after head selection. It is not included in that parameter count.

Choice and Score use p_j = exp(z_j/T) / Σ_k exp(z_k/T), computed with stable softmax. Choice returns argmax(p), the probability map and normalized-entropy confidence. Score returns Σ_j j·p_j (zero-based levels), its legend, probabilities and confidence. Noul uses logits [0,z] so p(yes)=sigmoid(z/T) and returns only `noul`.

For K choices/levels, confidence = 1 − H(p)/log(K). This is a measure of distribution concentration, **not** a separately learned correctness probability. Because b is shared, it cancels in Choice/Score softmax; it still affects Noul. Likewise, calibrating softmax does not establish out-of-domain calibration.

The checkpoint uses `jax.abel.residual-linear.v1`. Its historical `model_version: 0.1.0` is a loader contract checked by existing code, not the public release version; the package version is in config.json. The format pins the backbone hash, normalization, prompt and feature layer. Do not substitute another student GGUF without adapting the feature contract and retraining.

## Distillation and calibration

`train-jax` validates unique IDs and requires nonempty train, validation, calibration and test splits. Duplicate contents and source groups cannot cross splits. It labels tasks using a teacher, extracts frozen student features, then physically separates test features from optimizer input.

Local GGUF teachers produce binary next-token scores converted to task distributions. OpenAI-compatible teachers instead return requested JSON probabilities: those are elicited judgments, not token-logprob measurements. No teacher is universally compatible simply because its file is GGUF: llama.cpp architecture support, chat template, binary-token behavior and available memory must all work.

For teacher distribution q_i and student p_i, optimization minimizes mean soft-target cross-entropy plus

    (λ/2) [ ||w||₂² + (a−1)² + b² ]

at T=1. The regularizer anchors a to 1, not zero. There is no policy rollout, reward model, RL update, PPO, GRPO or RLCD objective.

Implementation: full-batch Adam (learning rate .025, β₁=.9, β₂=.999, ε=1e−8), gradient norm clipping at 5, a constrained to [.05,4]. Each λ ∈ {.001,.01,.1} runs up to 500 epochs; validation CE is checked every 10 epochs to choose the head. Temperature is selected on the calibration split from a 401-point log grid [.05,20], minimizing teacher-distribution cross-entropy. Thus this calibration step optimizes agreement with the teacher, not independently measured correctness.

The released head selected λ=.001, epoch 30 and T=2.8533858187665566. Exact report is `models/Jax-1-Abel/training_report.json`. The 100-task pilot has 70/10/10/10 splits, 80 adapted MASSIVE rows and 20 authored rule tasks. The teacher's development gate failed; the experimental continuation is recorded in config. Head weights and original backbone bytes are unchanged in publication. Historical absolute paths were normalized to `WORKSPACE/` in metadata; historical hashes refer to original training artifacts, not rewritten public metadata.

## Runtime and serving

`src/main.cpp` implements native CLI and persistent line-oriented worker. `service/app.py` owns one persistent worker/context; it validates requests, bounds admission and translates results to TypeSafe-compatible JSON. Limits include 256 KiB request body, 16 questions and 32 judgments at the HTTP boundary. Busy service responses use HTTP 429; there is no unbounded work queue. Worker timeout/crash handling restarts the subprocess. Request IDs, model metadata and actual repeated-prefill input-token accounting are exposed. Text output-token usage is zero.

CPU is the default. CUDA builds support teacher and student feature extraction with `--teacher-gpu-layers` / `--student-gpu-layers`; -1 means all layers, 0 CPU, positive partial offload. GPU requests fail explicitly on a CPU-only build. This release was validated on CPU, not CUDA.

## Source map

| Component | Source |
|---|---|
| Question decomposition and primitive math | `include/jax/jax.hpp` |
| Prompt payload | `include/jax/prompt.hpp` |
| Frozen backbone, hidden extraction | `include/jax/llama_backend.hpp` |
| Residual head and supervised loss | `include/jax/abel_head.hpp` |
| Teacher CE, gradients, calibration | `include/jax/distill.hpp`, `src/distill_train.cpp` |
| Local GGUF teacher | `include/jax/gguf_teacher.hpp`, `src/gguf_teacher.cpp` |
| Feature export | `src/abel_features.cpp` |
| Resumable orchestration / API teacher | `scripts/train-jax`, `scripts/chunk_cache.py`, `scripts/api_teacher.py` |
| Inference executable | `src/main.cpp`, `scripts/jax-abel` |
| TypeSafe-compatible HTTP | `service/app.py`, `scripts/serve-abel` |
| Ground-truth evaluation | `scripts/evaluate-candidate.py`, `benchmarks/evaluate.py` |

## Evidence and limitations

Pilot test accuracy is 7/10, NLL .738826, Brier .461048 and ECE-10 .238934. The 95% Wilson interval is approximately 39.7–89.2%; translated rows are correlated, which this interval does not account for. MASSIVE may have appeared in backbone pretraining. This tiny mixed test cannot establish universal decision capability or production readiness. No measured Jev comparison exists. The full DecisionBench test is a future evaluation, not a completed release result.
