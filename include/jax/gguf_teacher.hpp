#pragma once
#include <jax/prompt.hpp>
#include <llama.h>
#include <jax/gpu_options.hpp>
#include <memory>
#include <limits>

namespace jax {
// CPU-only local GGUF teacher. No HTTP client, URL, API key or remote model provider.
// Scores complete token sequences for labels 0 and 1, including multi-token labels.
class GgufTeacher {
    struct Runtime { Runtime(){llama_backend_init();} ~Runtime(){llama_backend_free();} } runtime;
    std::unique_ptr<llama_model,decltype(&llama_model_free)> model{nullptr,llama_model_free};
    std::unique_ptr<llama_context,decltype(&llama_free)> ctx{nullptr,llama_free};
    const llama_vocab* vocab=nullptr;
    std::string format;
    std::vector<llama_token> labels[2];
    bool qwen35=false;
    std::vector<llama_token> tokenize(const std::string& s,bool special,bool bos=false) const {
        if(s.size()>8*1024*1024)throw std::runtime_error("teacher text exceeds limit");
        int n=llama_tokenize(vocab,s.data(),int(s.size()),nullptr,0,bos,special);
        if(n==0)return {};
        if(n>0 || n==std::numeric_limits<int>::min())throw std::runtime_error("teacher tokenizer failed");
        std::vector<llama_token> out(size_t(-n));
        int got=llama_tokenize(vocab,s.data(),int(s.size()),out.data(),int(out.size()),bos,special);
        if(got<0)throw std::runtime_error("teacher tokenizer failed");
        out.resize(size_t(got));return out;
    }
    std::vector<llama_token> prompt(const Judgment& j) const {
        auto [system,user]=judgment_messages(j);
        const std::string sys="JAX_SYSTEM_SLOT_794d54e6", usr="JAX_USER_SLOT_9e8b1d29";
        std::string rendered;
        if(format=="raw") {
            rendered="System:\n"+sys+"\nUser:\n"+usr+"\nAssistant:\n";
        } else if(qwen35 && format.empty()) {
            rendered="<|im_start|>system\n"+sys+"<|im_end|>\n<|im_start|>user\n"+usr+
                "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        } else {
            llama_chat_message messages[]={{"system",sys.c_str()},{"user",usr.c_str()}};
            const char* tmpl=format.empty()?llama_model_chat_template(model.get(),nullptr):format.c_str();
            if(!tmpl)throw std::runtime_error("teacher has no chat template; specify --template with a compatible llama.cpp template name");
            int n=llama_chat_apply_template(tmpl,messages,2,true,nullptr,0);
            if(n<0 || n>1024*1024)throw std::runtime_error("unsupported teacher chat template; specify --template explicitly");
            std::vector<char> buffer(size_t(n)+1);
            int got=llama_chat_apply_template(tmpl,messages,2,true,buffer.data(),int(buffer.size()));
            if(got<0 || got>n)throw std::runtime_error("teacher chat template failed");
            rendered.assign(buffer.data(),size_t(got));
        }
        size_t a=rendered.find(sys), b=rendered.find(usr);
        if(a==std::string::npos || b==std::string::npos || a>=b ||
           rendered.find(sys,a+1)!=std::string::npos || rendered.find(usr,b+1)!=std::string::npos)
            throw std::runtime_error("teacher template does not preserve system/user messages");
        std::vector<llama_token> out;
        auto append=[&](const std::string& s,bool special,bool bos=false){auto t=tokenize(s,special,bos);out.insert(out.end(),t.begin(),t.end());};
        append(rendered.substr(0,a),true,!qwen35);append(system,false);
        append(rendered.substr(a+sys.size(),b-a-sys.size()),true);append(user,false);
        append(rendered.substr(b+usr.size()),true);
        if(out.empty())throw std::runtime_error("empty teacher prompt");
        return out;
    }
    void decode(std::vector<llama_token> tokens) {
        for(size_t offset=0;offset<tokens.size();offset+=256){
            auto batch=llama_batch_get_one(tokens.data()+offset,int(std::min(size_t(256),tokens.size()-offset)));
            if(llama_decode(ctx.get(),batch)!=0)throw std::runtime_error("teacher llama_decode failed");
        }
    }
    double likelihood(const std::vector<llama_token>& prefix,const std::vector<llama_token>& suffix) {
        if(prefix.size()+suffix.size()>llama_n_ctx(ctx.get()))throw std::runtime_error("teacher judgment exceeds context; no truncation performed");
        llama_memory_clear(llama_get_memory(ctx.get()),true);
        decode(prefix);double sum=0;
        const int n=llama_vocab_n_tokens(vocab);
        for(size_t i=0;i<suffix.size();++i){
            const float* logits=llama_get_logits_ith(ctx.get(),-1);
            if(!logits)throw std::runtime_error("teacher returned no logits");
            double maximum=-std::numeric_limits<double>::infinity();
            for(int k=0;k<n;++k)maximum=std::max(maximum,double(logits[k]));
            double denominator=0;
            for(int k=0;k<n;++k)denominator+=std::exp(double(logits[k])-maximum);
            double lp=double(logits[suffix[i]])-maximum-std::log(denominator);
            if(!std::isfinite(lp))throw std::runtime_error("teacher returned nonfinite likelihood");
            sum+=lp;
            if(i+1<suffix.size())decode({suffix[i]});
        }
        return sum;
    }
public:
    GgufTeacher(const std::string& path,unsigned context,unsigned threads,const std::string& chat_template="",int gpu_layer_count=0) : format(chat_template) {
        if(context<128 || context>32768 || threads<1 || threads>64)throw std::runtime_error("invalid teacher context/threads");
        auto mp=llama_model_default_params();mp.n_gpu_layers=gpu_layers(gpu_layer_count);
        model.reset(llama_model_load_from_file(path.c_str(),mp));
        if(!model)throw std::runtime_error("cannot load local GGUF teacher");
        if(!llama_model_has_decoder(model.get()) || llama_model_has_encoder(model.get()))
            throw std::runtime_error("teacher must be a supported decoder-only text language model");
        vocab=llama_model_get_vocab(model.get());
        char arch[128]{};llama_model_meta_val_str(model.get(),"general.architecture",arch,sizeof arch);
        qwen35=std::string(arch)=="qwen35";
        labels[0]=tokenize("0",false);labels[1]=tokenize("1",false);
        if(labels[0].empty() || labels[1].empty() || labels[0]==labels[1])throw std::runtime_error("teacher cannot encode distinct decision labels");
        auto cp=llama_context_default_params();cp.n_ctx=context;cp.n_batch=256;cp.n_ubatch=128;
        cp.n_seq_max=1;cp.n_threads=int(threads);cp.n_threads_batch=int(threads);
        ctx.reset(llama_init_from_model(model.get(),cp));
        if(!ctx)throw std::runtime_error("cannot initialize teacher context");
    }
    double operator()(const Judgment& j) {
        auto tokens=prompt(j);
        if(labels[0].size()==1 && labels[1].size()==1) {
            if(tokens.size()>llama_n_ctx(ctx.get()))throw std::runtime_error("teacher judgment exceeds context; no truncation performed");
            llama_memory_clear(llama_get_memory(ctx.get()),true);decode(tokens);
            const float* logits=llama_get_logits_ith(ctx.get(),-1);
            if(!logits)throw std::runtime_error("teacher returned no logits");
            double z=double(logits[labels[1][0]])-double(logits[labels[0][0]]);
            if(!std::isfinite(z))throw std::runtime_error("nonfinite teacher logit");
            return z;
        }
        return likelihood(tokens,labels[1])-likelihood(tokens,labels[0]);
    }
};
}
