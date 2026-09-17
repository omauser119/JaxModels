#!/usr/bin/env bash
set -euo pipefail
project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_dir"
cmake_bin=${JAX_CMAKE:-"$project_dir/.tools/bin/cmake"}
if [[ ! -x "$cmake_bin" ]]; then cmake_bin=$(command -v cmake || true); fi
if [[ -z "$cmake_bin" ]]; then
    echo 'CMake missing. Install it, or run: uv venv .tools && uv pip install --python .tools/bin/python cmake==4.4.3' >&2
    exit 1
fi
revision=$(python3 -c 'import json; print(json.load(open("models/manifest.json"))["llama_cpp_revision"])')
if [[ ! -d vendor/llama.cpp/.git ]]; then
    mkdir -p vendor
    git init vendor/llama.cpp
    git -C vendor/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp.git
    git -C vendor/llama.cpp fetch --depth 1 origin "$revision"
    git -C vendor/llama.cpp checkout --detach FETCH_HEAD
fi
if [[ $(git -C vendor/llama.cpp rev-parse HEAD) != "$revision" ]]; then
    echo "llama.cpp revision differs from models/manifest.json; select $revision explicitly" >&2
    exit 1
fi
"$cmake_bin" -S vendor/llama.cpp -B vendor/llama.cpp/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE='-O2 -DNDEBUG' \
    -DCMAKE_CXX_FLAGS_RELEASE='-O2 -DNDEBUG' -DGGML_NATIVE=ON \
    -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_BACKEND_DL=OFF \
    -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
"$cmake_bin" --build vendor/llama.cpp/build --target llama -j "${JAX_BUILD_JOBS:-2}"
make native
