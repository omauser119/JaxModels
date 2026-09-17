#include <jax/llama_backend.hpp>
#include <jax/requests.hpp>
#include <fstream>
#include <iostream>
#include <set>
using jax::Json;
int main(int argc,char** argv) {
    try {
        if (argc!=3 && argc!=6) throw std::runtime_error("Usage: abel-features model.gguf dataset.jsonl [context threads gpu-layers] > features.jsonl");
        // Logs must not contaminate the cache and per-tensor debug output is unnecessary.
        llama_log_set([](ggml_log_level level,const char* text,void*){
            if (level==GGML_LOG_LEVEL_WARN || level==GGML_LOG_LEVEL_ERROR) std::fputs(text,stderr);
        },nullptr);
        std::ifstream in(argv[2]); if (!in) throw std::runtime_error("cannot read dataset");
        auto setting=[](const char* text){size_t used=0;auto n=std::stoul(text,&used);if(used!=std::string(text).size() || n>32768)throw std::runtime_error("invalid context/threads");return unsigned(n);};
        jax::LlamaBackend backbone(argv[1],argc==6?setting(argv[3]):2048,argc==6?setting(argv[4]):2,argc==6?jax::parse_gpu_layers(argv[5]):0);
        std::string line; size_t count=0; std::set<std::string> ids;
        while (std::getline(in,line)) {
            if (line.empty()) continue;
            auto row=Json::parse(line); std::string id=row.at("id").string();
            if (!ids.insert(id).second) throw std::runtime_error("duplicate dataset id");
            auto request=Json::object(), questions=Json::object();
            questions.set("judgment",row.at("question")); request.set("questions",questions);
            auto q=jax::parse_questions(request).at(0);
            auto features=Json::array(), base=Json::array(), labels=Json::array();
            size_t target=0;
            if (q.kind==jax::Kind::choice) {
                bool found=false;
                for (size_t i=0;i<q.criteria.size();++i) {
                    labels.push(Json::str(q.criteria[i].name));
                    if (q.criteria[i].name==row.at("target").string()) {target=i;found=true;}
                }
                if (!found) throw std::runtime_error("target not in options");
            } else {
                double y=row.at("target").number();
                size_t classes=q.kind==jax::Kind::noul?2:q.criteria.size();
                if (y<0 || y>=classes || std::floor(y)!=y) throw std::runtime_error("bad target index");
                target=size_t(y);
                for (size_t i=0;i<classes;++i) labels.push(Json::str(std::to_string(i)));
            }
            for (const auto& j:jax::judgments(row.at("state").dump(),q)) {
                auto f=backbone.features(j); auto h=Json::array();
                for (double v:f.hidden) h.push(Json::num(v));
                features.push(h); base.push(Json::num(f.base_logit));
            }
            auto out=Json::object();
            for (const auto& key:{"id","split","family"}) out.set(key,row.at(key));
            out.set("type",row.at("question").at("type")); out.set("target",Json::num(target));
            if(row.has("teacher_probabilities")) out.set("teacher_probabilities",row.at("teacher_probabilities"));
            out.set("labels",labels); out.set("hidden",features); out.set("base_logits",base);
            std::cout<<out.dump()<<'\n'<<std::flush;
            std::cerr<<"extracted "<<++count<<" "<<id<<'\n';
        }
        if (!in.eof()) throw std::runtime_error("dataset read failed");
    } catch(const std::exception& e) {std::cerr<<"abel-features: "<<e.what()<<'\n';return 1;}
}
