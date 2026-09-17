CXX ?= c++
CPPFLAGS += -Iinclude $(shell pkg-config --cflags libcurl json-c)
CXXFLAGS ?= -O2 -g
CXXFLAGS += -std=c++20 -Wall -Wextra -Wpedantic -pthread
LDLIBS += $(shell pkg-config --libs libcurl json-c) -pthread
.PHONY: all test clean
all: build/jax
build:
	mkdir -p build
build/jax: src/main.cpp include/jax/jax.hpp include/jax/json.hpp include/jax/prompt.hpp include/jax/requests.hpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDLIBS)
build/test: tests/core.cpp include/jax/jax.hpp | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ -pthread
test: build/jax build/test
	./build/test
	python3 tests/integration.py
clean:
	rm -rf build

LLAMA_DIR ?= vendor/llama.cpp
LLAMA_LIBDIR ?= $(LLAMA_DIR)/build/bin
.PHONY: native
native: build/jax-native
build/jax-native: src/main.cpp include/jax/jax.hpp include/jax/json.hpp include/jax/prompt.hpp include/jax/llama_backend.hpp include/jax/gpu_options.hpp include/jax/requests.hpp include/jax/abel_head.hpp | build
	$(CXX) -Iinclude -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include $(shell pkg-config --cflags json-c) $(CXXFLAGS) -DJAX_WITH_LLAMA $< -o $@ -L$(LLAMA_LIBDIR) -Wl,-rpath,'$$ORIGIN/../$(LLAMA_LIBDIR)' -lllama $(shell pkg-config --libs json-c) -pthread

build/test-native: tests/native.cpp include/jax/llama_backend.hpp include/jax/gpu_options.hpp include/jax/prompt.hpp include/jax/jax.hpp include/jax/json.hpp | build
	$(CXX) -Iinclude -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include $(shell pkg-config --cflags json-c) $(CXXFLAGS) $< -o $@ -L$(LLAMA_LIBDIR) -Wl,-rpath,'$$ORIGIN/../$(LLAMA_LIBDIR)' -lllama $(shell pkg-config --libs json-c) -pthread
.PHONY: test-native
test-native: build/test-native
	./build/test-native models/Jax-1-Abel/backbone.gguf

build/abel-features: src/abel_features.cpp include/jax/llama_backend.hpp include/jax/gpu_options.hpp include/jax/prompt.hpp include/jax/requests.hpp include/jax/jax.hpp include/jax/json.hpp | build
	$(CXX) -Iinclude -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include $(shell pkg-config --cflags json-c) $(CXXFLAGS) $< -o $@ -L$(LLAMA_LIBDIR) -Wl,-rpath,'$$ORIGIN/../$(LLAMA_LIBDIR)' -lllama $(shell pkg-config --libs json-c) -pthread

build/abel-train: src/abel_train.cpp include/jax/abel_head.hpp include/jax/jax.hpp include/jax/json.hpp | build
	$(CXX) -Iinclude $(shell pkg-config --cflags json-c) $(CXXFLAGS) $< -o $@ $(shell pkg-config --libs json-c) -pthread

build/test-abel: tests/abel.cpp include/jax/abel_head.hpp include/jax/jax.hpp include/jax/json.hpp | build
	$(CXX) -Iinclude $(shell pkg-config --cflags json-c) $(CXXFLAGS) $< -o $@ $(shell pkg-config --libs json-c) -pthread
.PHONY: test-abel abel
abel: native build/abel-features build/abel-train
test-abel: build/test-abel
	./build/test-abel

.PHONY: test-abel-cli
test-abel-cli: build/abel-train
	python3 tests/abel_cli.py

.PHONY: test-service test-benchmark check-runtime
test-service:
	python3 tests/service.py
test-benchmark:
	python3 tests/benchmark.py
check-runtime: test test-abel test-abel-cli test-service test-benchmark

build/gguf-teacher: src/gguf_teacher.cpp include/jax/gguf_teacher.hpp include/jax/gpu_options.hpp include/jax/prompt.hpp include/jax/requests.hpp include/jax/jax.hpp include/jax/json.hpp | build
	$(CXX) -Iinclude -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include $(shell pkg-config --cflags json-c) $(CXXFLAGS) $< -o $@ -L$(LLAMA_LIBDIR) -Wl,-rpath,'$$ORIGIN/../$(LLAMA_LIBDIR)' -lllama $(shell pkg-config --libs json-c) -pthread

build/distill-train: src/distill_train.cpp include/jax/distill.hpp include/jax/abel_head.hpp include/jax/jax.hpp include/jax/json.hpp | build
	$(CXX) -Iinclude $(shell pkg-config --cflags json-c) $(CXXFLAGS) $< -o $@ $(shell pkg-config --libs json-c) -pthread
.PHONY: distill
distill: build/gguf-teacher build/abel-features build/distill-train

build/test-distill: tests/distill.cpp include/jax/distill.hpp include/jax/abel_head.hpp include/jax/jax.hpp include/jax/json.hpp | build
	$(CXX) -Iinclude $(shell pkg-config --cflags json-c) $(CXXFLAGS) $< -o $@ $(shell pkg-config --libs json-c) -pthread
.PHONY: test-distill
test-distill: build/test-distill build/distill-train
	./build/test-distill
	python3 tests/distill_cli.py
check-runtime: test-distill

.PHONY: test-chunks
test-chunks:
	python3 tests/chunk_cache.py
check-runtime: test-chunks


.PHONY: test-candidate-evaluation
test-candidate-evaluation:
	python3 tests/candidate_evaluation.py
check-runtime: test-candidate-evaluation

.PHONY: test-api-teacher
test-api-teacher:
	python3 tests/api_teacher.py
check-runtime: test-api-teacher

build/test-gpu-options: tests/gpu_options.cpp include/jax/gpu_options.hpp | build
	$(CXX) -Iinclude -I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include $(CXXFLAGS) $< -o $@ -L$(LLAMA_LIBDIR) -Wl,-rpath,'$$ORIGIN/../$(LLAMA_LIBDIR)' -lllama -pthread
.PHONY: test-gpu-options
test-gpu-options: build/test-gpu-options
	./build/test-gpu-options

.PHONY: test-release
test-release: native
	python3 tests/release_parity.py
