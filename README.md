# JaxModels · Jax-1-Abel

Open-source typed decision inference, training and a local TypeSafe SDK compatibility server. Implemented in C++20 with llama.cpp; Python handles training orchestration and HTTP serving.

**Current release: 0.2.0-pilot.1, experimental.** Abel is a frozen pretrained Qwen3.5-0.8B Q8_0 backbone plus a trained 1,026-parameter shared residual decision head. It is **not a reproduction of TypeSafe RLCD**, a newly pretrained foundation model, or a production-qualified model. The released head was distilled from local Qwen3.5-2B Q4_K_M. Held-out pilot: **7/10 correct**, 95% Wilson interval **39.7–89.2%**. Teacher development gate failed (9/16); this run explicitly continued as an experiment.

- [Architecture, equations and implementation map](docs/ARCHITECTURE.md)
- [RLCD: public information, unknowns and implementation status](docs/RLCD.md)
- [Training with GGUF or OpenAI-compatible teachers](docs/TRAINING.md)
- [HTTP API and official TypeSafe JS SDK](docs/SERVING.md)
- [Model card and evaluation](models/Jax-1-Abel/README.md)
- [DecisionBench: future Jev/Abel comparison](benchmarks/README.md)
- [Model releases](https://github.com/omauser119/JaxModels/releases)

## Build and run (Linux, CPU)

Requires Python >=3.11, C++20, Git, curl, CMake >=3.19, Ninja, json-c and libcurl development headers. CPU build uses native CPU instructions; rebuild on the deployment machine.

```bash
git clone https://github.com/omauser119/JaxModels.git
cd JaxModels
# Debian/Ubuntu:
sudo apt-get update
sudo apt-get install -y build-essential git curl cmake ninja-build pkg-config libjson-c-dev libcurl4-openssl-dev python3-venv
python3 scripts/download-model.py
scripts/build-native.sh
make -j2 distill
scripts/jax-abel infer examples/ticket.json
```

The downloader fetches the pinned backbone from **GitHub Releases**, verifies SHA-256 and activates it beside the head tracked in this repository. The complete model consists of `backbone.gguf`, `decision_head.json` and `config.json`; the head alone cannot run inference. Model files and third-party build outputs are excluded from Git.

## Local HTTP server

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r service/requirements.lock
scripts/serve-abel --port 8091
# Another terminal:
curl -f http://127.0.0.1:8091/readyz
curl -f http://127.0.0.1:8091/v1/systemone -H 'Content-Type: application/json' --data-binary @examples/ticket.json
```

Use `baseURL: 'http://127.0.0.1:8091'` (without `/v1`) and `defaultModel: 'Jax-1-Abel'` with `@typesafe-ai/sdk@0.6.0`. Compatibility covers the documented endpoints and tested primitives; it does not promise Jev-equivalent predictions or performance.

## Verify

```bash
make -j2 check-runtime test-gpu-options
cd integrations/typesafe-js && npm ci && npm test
```

SDK tests require the running server and compare its probabilities with `models/Jax-1-Abel/example-response.json`. CPU runtime and live SDK parity are verified for this release; CUDA code is provided but was not GPU-tested on the release machine.

## Repository scope and licensing

This is the curated public source tree. Local service state, credentials, training caches, old launch scripts, generated videos and vendored build products are excluded. `src/abel_train.cpp` is a separate supervised head trainer; `src/distill_train.cpp` is the teacher-distribution trainer used by this release. Neither changes the backbone.

Code and authored synthetic data: [MIT](LICENSE). Backbone and released head: [Apache-2.0](models/Jax-1-Abel/LICENSE). Adapted MASSIVE pilot rows: CC-BY-4.0 with [attribution](data/abel-pilot-100/NOTICE-MASSIVE.md). Jax AI is independent of TypeSafe and the Google JAX numerical framework.
