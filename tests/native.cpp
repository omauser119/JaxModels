#include <jax/llama_backend.hpp>
#include <iostream>
int main(int argc, char** argv) {
    try {
        if (argc!=2) throw std::runtime_error("model path required");
        jax::LlamaBackend backend(argv[1],512,2);
        jax::Judgment a{"\"The parcel is red.\"","Is the parcel red?","",{}};
        jax::Judgment b{"\"The parcel is blue.\"","Is the parcel red?","",{}};
        double first=backend(a), other=backend(b), again=backend(a);
        auto features=backend.features(a);
        double norm=0;for(double x:features.hidden)norm+=x*x;
        if(features.hidden.size()!=1024 || std::abs(norm-1)>1e-8 || std::abs(features.base_logit-first)>1e-4)
            throw std::runtime_error("hidden feature extraction changed logits or normalization");
        if (!std::isfinite(first) || !std::isfinite(other) || std::abs(first-again)>1e-4)
            throw std::runtime_error("nonfinite logits or leaked recurrent state");
        auto too_long=a; too_long.state="\""+std::string(20000,'x')+"\"";
        bool rejected=false;
        try {backend(too_long);} catch(const std::runtime_error&) {rejected=true;}
        if (!rejected) throw std::runtime_error("overlong state was not rejected");
        std::cout << "native reset/context/1024-d hidden features OK; P(red|red)=" << jax::softmax({0,first})[1]
                  << "; P(red|blue)=" << jax::softmax({0,other})[1] << '\n';
    } catch(const std::exception& e) {std::cerr << e.what()<<'\n';return 1;}
}
