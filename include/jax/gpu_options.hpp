#pragma once
#include <llama.h>
#include <limits>
#include <stdexcept>
#include <string>
namespace jax {
inline int parse_gpu_layers(const std::string& text) {
    size_t used=0;int value=std::stoi(text,&used);
    if(used!=text.size() || value < -1 || value>10000)
        throw std::invalid_argument("GPU layers must be -1 (all), 0 (CPU), or 1..10000");
    return value;
}
inline int gpu_layers(int value) {
    if(value < -1 || value>10000)throw std::invalid_argument("invalid GPU layer count");
    if(value!=0 && !llama_supports_gpu_offload())
        throw std::runtime_error("GPU offload requested but unavailable; build llama.cpp with CUDA and attach a GPU, or use GPU layers 0");
    return value==-1?std::numeric_limits<int>::max():value;
}
}
