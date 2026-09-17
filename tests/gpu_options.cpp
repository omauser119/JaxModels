#include <jax/gpu_options.hpp>
#include <cassert>
#include <iostream>
int main(){
    assert(jax::parse_gpu_layers("-1")==-1);
    assert(jax::parse_gpu_layers("0")==0);
    assert(jax::parse_gpu_layers("32")==32);
    for(const auto& text:{"-2","10001","1oops",""}){
        bool rejected=false;try{jax::parse_gpu_layers(text);}catch(const std::exception&){rejected=true;}
        assert(rejected);
    }
    assert(jax::gpu_layers(0)==0);
    if(!llama_supports_gpu_offload()){
        bool rejected=false;try{jax::gpu_layers(-1);}catch(const std::runtime_error&){rejected=true;}
        assert(rejected);
    }
    std::cout<<"GPU options: parsing, CPU default, unavailable offload rejection OK\n";
}
