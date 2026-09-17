# Training Abel

Build the runtime and obtain the released student backbone as described in the root README, then run `make -j2 distill`. Python >=3.11 is required. The teacher may change; the current student backbone is fixed by the checkpoint contract.

## Local GGUF teacher

```bash
scripts/train-jax --teacher /absolute/path/teacher.gguf \
  --data data/abel-pilot-100/tasks.jsonl \
  --output training/my-run --chunk-size 8
# Interrupted run, with identical inputs, binaries and settings:
scripts/train-jax --teacher /absolute/path/teacher.gguf \
  --data data/abel-pilot-100/tasks.jsonl \
  --output training/my-run --chunk-size 8 --resume
python3 scripts/evaluate-candidate.py --run training/my-run --corpus data/abel-pilot-100
```

Local teacher weights are not included in the release. The original pilot used `unsloth/Qwen3.5-2B-GGUF`, revision `f6d5376be1edb4d416d56da11e5397a961aca8ae`, file `Qwen3.5-2B-Q4_K_M.gguf`, SHA-256 `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`. Its weak development result is preserved in model config. A stronger teacher does not automatically imply a stronger student: evaluate independently.

Use any **supported** local GGUF teacher, including shard 00001 of a complete sharded model. Architecture support depends on the pinned llama.cpp revision. Auto chat-template selection and binary-label scoring must be checked for each teacher. `--teacher-template raw` is an explicit fallback, not an assurance of quality. Teachers and student run in separate stages; they need not both fit in memory simultaneously.

The pipeline saves provenance, teacher distributions, features, shard hashes, progress, trained head and teacher-agreement metrics. Completed chunks survive interruption. API labeling also checkpoints each accepted row. Exact provenance must match to resume; rebuilds or changed data require a new run. Output is `training/my-run/checkpoint/decision_head.json`; training never silently replaces the serving model.

## OpenAI-compatible API teacher

```bash
# Set the secret in your environment, never in a committed file.
read -rs -p 'Teacher API key: ' JAX_TEACHER_API_KEY; echo
export JAX_TEACHER_API_KEY
scripts/train-jax --teacher-backend openai \
  --base-url https://your-provider.example/v1 --model your-model \
  --data data/abel-pilot-100/tasks.jsonl --output training/api-run --chunk-size 8
unset JAX_TEACHER_API_KEY
```

`--baseurl`/`--apikey` aliases exist, but environment variables avoid exposing keys in command history/process arguments. `--api-minimal` omits optional sampling/token parameters for providers that reject them. Loopback HTTP is supported; remote endpoints require HTTPS. The adapter reads Chat Completions JSON distributions, not token logprobs. Transient failures use bounded retries; authentication errors fail promptly.

Project preference: use a locally licensed GGUF teacher, or a DeepSeek/Qwen/GLM-style deployment whose terms explicitly allow the intended training. Avoid Claude/ChatGPT endpoints for this workflow unless the applicable agreement permits it; model/provider names alone do not establish rights. API training sends your task data to that endpoint and may incur charges. No provider keys are included in this repository.

## CUDA

With a working CUDA toolkit and supported NVIDIA GPU:

```bash
# First fetch/configure the pinned llama.cpp source using scripts/build-native.sh.
python3 scripts/build-cuda.py --jobs 2 --arch native
scripts/train-jax --teacher /path/to/teacher.gguf \
  --teacher-gpu-layers -1 --student-gpu-layers -1 \
  --data data/abel-pilot-100/tasks.jsonl --output training/gpu-run --chunk-size 8
```

Teacher/student GPU layer flags: -1 all, 0 CPU, positive partial. GPU results may differ numerically. CUDA is implemented but not hardware-validated in this release. No Colab dependency is needed.

## Data and evaluation contract

Each task is one JSON line with exactly `id`, `split`, `family`, `group`, `state`, `question`. Split is train/validation/calibration/test. No reference answer belongs in teacher input. Put ground-truth annotations in a separate references file. Validate licenses and preserve source grouping across splits. The included pilot and its license/attribution files reproduce the public input rows; it is a demonstration-sized corpus, not sufficient production coverage.

Head fitting uses train only; validation chooses parameters, calibration chooses temperature, and test is excluded from both. Test teacher agreement is a diagnostic, not ground-truth accuracy. `evaluate-candidate.py` scores held-out references and compares the candidate to the currently bundled Abel and base Qwen. In archived release reports, `deployed_abel` denotes the older local seed checkpoint used at training time, not today's bundled candidate.

DecisionBench has a separate corpus and must not be training input. For future comparisons, use its same fixed requests for Abel and Jev, report failures and latency, and preserve model/version provenance. Once a test set guides development it becomes a regression set; reserve a new unseen test for new headline claims.

## Publishing a new head

Keep the resulting head separate until evaluation is complete. To activate it intentionally, copy it into a new model directory together with the required backbone and config, update the head SHA-256 and config temperature, record evaluation/provenance, and version the package. `scripts/jax-abel` currently loads `models/Jax-1-Abel`; it validates both component hashes and never falls back silently. This repository does not automate a production qualification decision.

## Replay the released head without running the teacher again

Download `training-evidence.tar.gz` from release `v0.2.0-pilot.1`, verify it against `SHA256SUMS`, and extract it into a local working directory. Then:

```bash
build/distill-train fit /path/to/training-evidence/fit-features.jsonl training/replayed-head
sha256sum training/replayed-head/decision_head.json
```

On the release CPU/build, replay produced exactly `281fbdb5c622fa354bfdf211668eceab02c0d56f532fa24f6d263b6be30d98e8`, identical to the published head. Other compilers/platforms may introduce floating-point differences. The evidence archive also includes teacher labels, test features, ground-truth annotations, attribution and training reports. These files are experimental evidence, not additional trained parameters.
