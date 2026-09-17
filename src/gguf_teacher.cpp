#include <jax/gguf_teacher.hpp>
#include <jax/requests.hpp>
#include <fstream>
#include <iostream>
#include <set>
using jax::Json;
int main(int argc,char** argv){
    try{
        if(argc!=6 && argc!=7)throw std::runtime_error("Usage: gguf-teacher teacher.gguf tasks.jsonl context threads template-or-auto [gpu-layers]");
        std::ifstream probe(argv[1],std::ios::binary);char magic[4]{};probe.read(magic,4);
        if(!probe || std::string(magic,4)!="GGUF")throw std::runtime_error("teacher must be an existing local GGUF file; API/URL teachers are unsupported");
        auto setting=[](const char* text){size_t used=0;auto n=std::stoul(text,&used);if(used!=std::string(text).size() || n>32768)throw std::runtime_error("invalid numeric setting");return unsigned(n);};
        llama_log_set([](ggml_log_level level,const char* text,void*){if(level==GGML_LOG_LEVEL_WARN || level==GGML_LOG_LEVEL_ERROR)std::fputs(text,stderr);},nullptr);
        jax::GgufTeacher teacher(argv[1],setting(argv[3]),setting(argv[4]),std::string(argv[5])=="auto"?"":argv[5],argc==7?jax::parse_gpu_layers(argv[6]):0);
        std::ifstream input(argv[2]);if(!input)throw std::runtime_error("cannot read tasks");
        std::set<std::string> ids;std::string line;size_t count=0;
        while(std::getline(input,line)){
            if(line.empty())continue;
            auto row=Json::parse(line);auto id=row.at("id").string();
            if(!ids.insert(id).second)throw std::runtime_error("duplicate task ID");
            auto req=Json::object(),qs=Json::object();qs.set("q",row.at("question"));req.set("questions",qs);
            auto questions=jax::parse_questions(req);
            auto answers=jax::decide(row.at("state").dump(),questions,[&](const auto& j){return teacher(j);},1,1);
            const auto& a=answers[0];std::vector<double> p=a.kind==jax::Kind::noul?std::vector<double>{1-a.noul,a.noul}:a.probabilities;
            auto probs=Json::array();for(double v:p)probs.push(Json::num(v));
            size_t target=size_t(std::max_element(p.begin(),p.end())-p.begin());
            row.set("target",a.kind==jax::Kind::choice?Json::str(questions[0].criteria[target].name):Json::num(target));
            row.set("teacher_probabilities",probs);
            std::cout<<row.dump()<<'\n'<<std::flush;
            std::cerr<<"teacher labeled "<<++count<<" "<<id<<'\n';
        }
        if(!input.eof() || count==0)throw std::runtime_error("empty/invalid task file");
    }catch(const std::exception& e){std::cerr<<"gguf-teacher: "<<e.what()<<'\n';return 1;}
}
