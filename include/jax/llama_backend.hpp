#pragma once
#include <jax/prompt.hpp>
#include <llama.h>
#include <jax/gpu_options.hpp>
#include <memory>
#include <mutex>
#include <limits>
namespace jax {
// Native text-only backend for Qwen3.5. One shared model/context, serialized prefill.
struct DecisionFeatures { double base_logit; std::vector<double> hidden; };
class LlamaBackend {
    struct Runtime {
        Runtime() { llama_backend_init(); }
        ~Runtime() { llama_backend_free(); }
    } runtime;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model{nullptr,llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> context{nullptr,llama_free};
    const llama_vocab* vocab = nullptr;
    llama_token zero{}, one{};
    std::mutex mutex;
    size_t input_tokens_total = 0;
    std::vector<llama_token> tokenize(const std::string& text, bool special) const {
        if (text.size() > size_t(std::numeric_limits<int32_t>::max())) throw std::runtime_error("text too long");
        int n = llama_tokenize(vocab,text.data(),int32_t(text.size()),nullptr,0,false,special);
        if (n == std::numeric_limits<int32_t>::min()) throw std::runtime_error("tokenization overflow");
        if (n == 0) return {};
        if (n > 0) throw std::runtime_error("unexpected tokenizer result");
        std::vector<llama_token> out(size_t(-n));
        int got = llama_tokenize(vocab,text.data(),int32_t(text.size()),out.data(),int32_t(out.size()),false,special);
        if (got < 0) throw std::runtime_error("tokenization failed");
        out.resize(size_t(got)); return out;
    }
public:
    LlamaBackend(const std::string& path, unsigned n_ctx = 2048, unsigned threads = 2, int gpu_layer_count = 0) {
        if (n_ctx < 128 || n_ctx > 32768 || threads < 1 || threads > 64)
            throw std::invalid_argument("JAX_CTX must be 128..32768; JAX_THREADS must be 1..64");
        auto mp = llama_model_default_params(); mp.n_gpu_layers = gpu_layers(gpu_layer_count);
        model.reset(llama_model_load_from_file(path.c_str(),mp));
        if (!model) throw std::runtime_error("cannot load GGUF model: " + path);
        char architecture[128]{};
        if (llama_model_meta_val_str(model.get(),"general.architecture",architecture,sizeof architecture) < 0 ||
            std::string(architecture) != "qwen35") throw std::runtime_error("native template supports qwen35 architecture only");
        vocab = llama_model_get_vocab(model.get());
        auto z = tokenize("0",false), o = tokenize("1",false);
        if (z.size()!=1 || o.size()!=1 || z[0]==o[0]) throw std::runtime_error("decision labels must be distinct single tokens");
        zero=z[0]; one=o[0];
        for (const auto& control : {"<|im_start|>","<|im_end|>","<think>","</think>"})
            if (tokenize(control,true).size()!=1) throw std::runtime_error("unsupported Qwen control tokens");
        auto cp = llama_context_default_params();
        cp.n_ctx=n_ctx; cp.n_batch=256; cp.n_ubatch=128; cp.n_seq_max=1;
        cp.pooling_type=LLAMA_POOLING_TYPE_NONE;
        cp.n_threads=int(threads); cp.n_threads_batch=int(threads);
        context.reset(llama_init_from_model(model.get(),cp));
        if (!context) throw std::runtime_error("cannot initialize llama context");
    }
    DecisionFeatures features(const Judgment& j, bool include_hidden = true) {
        std::lock_guard lock(mutex);
        const auto [system,user] = judgment_messages(j);
        std::vector<llama_token> tokens;
        auto append = [&](const std::string& s, bool special) {
            auto part=tokenize(s,special); tokens.insert(tokens.end(),part.begin(),part.end());
        };
        // Qwen/Qwen3.5-0.8B tokenizer_config.json, enable_thinking=false.
        // Parse control tokens in framing only, not in caller-supplied data.
        append("<|im_start|>system\n",true); append(system,false);
        append("<|im_end|>\n<|im_start|>user\n",true); append(user,false);
        append("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",true);
        if (tokens.size() > llama_n_ctx(context.get())) throw std::runtime_error("judgment exceeds JAX_CTX; shorten state/criteria or increase JAX_CTX");
        llama_memory_clear(llama_get_memory(context.get()),true);
        // Only the final token needs exported hidden features. Exporting embeddings
        // for the whole prompt would also materialize a vocabulary projection per token.
        llama_set_embeddings(context.get(),false);
        const size_t prefix = tokens.size()-1;
        for (size_t offset=0; offset<prefix; offset+=256) {
            int n=int(std::min(size_t(256),prefix-offset));
            auto batch=llama_batch_get_one(tokens.data()+offset,n);
            int rc=llama_decode(context.get(),batch);
            if (rc!=0) throw std::runtime_error("llama_decode failed: " + std::to_string(rc));
        }
        llama_set_embeddings(context.get(),include_hidden);
        auto last=llama_batch_get_one(tokens.data()+prefix,1);
        int rc=llama_decode(context.get(),last);
        if (rc!=0) throw std::runtime_error("final llama_decode failed: " + std::to_string(rc));
        const float* logits=llama_get_logits_ith(context.get(),-1);
        if (!logits) throw std::runtime_error("llama returned no logits");
        double result=double(logits[one])-double(logits[zero]);
        if (!std::isfinite(result)) throw std::runtime_error("llama returned nonfinite logits");
        DecisionFeatures out{result,{}};
        if (include_hidden) {
            const float* hidden=llama_get_embeddings_ith(context.get(),-1);
            if (!hidden) throw std::runtime_error("llama returned no hidden features");
            const int dim=llama_model_n_embd_out(model.get());
            double norm=0;
            for (int i=0;i<dim;++i) {
                if (!std::isfinite(hidden[i])) throw std::runtime_error("nonfinite hidden feature");
                norm+=double(hidden[i])*hidden[i];
            }
            if (norm<1e-20) throw std::runtime_error("zero hidden feature norm");
            out.hidden.resize(size_t(dim)); norm=std::sqrt(norm);
            for (int i=0;i<dim;++i) out.hidden[size_t(i)]=hidden[i]/norm;
        }
        input_tokens_total += tokens.size();
        return out;
    }
    size_t take_input_tokens() {
        std::lock_guard lock(mutex);
        size_t n=input_tokens_total;input_tokens_total=0;return n;
    }
    double operator()(const Judgment& j) {return features(j,false).base_logit;}
};
}
