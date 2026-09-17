#include <jax/distill.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <limits>
using jax::Json;
static auto read_rows(const std::string& path){
    std::ifstream in(path);if(!in)throw std::runtime_error("cannot read features");
    std::vector<jax::DistillExample> rows;std::set<std::string> ids;std::string line;size_t dim=0;
    while(std::getline(in,line)){
        if(line.empty())continue;
        auto j=Json::parse(line);jax::DistillExample r;auto& e=r.features;
        e.id=j.at("id").string();e.split=j.at("split").string();e.type=j.at("type").string();e.family=j.at("family").string();
        if(!ids.insert(e.id).second)throw std::runtime_error("duplicate feature ID");
        if(e.split!="train" && e.split!="validation" && e.split!="calibration" && e.split!="test")throw std::runtime_error("invalid split");
        auto base=j.at("base_logits"),hidden=j.at("hidden"),target=j.at("teacher_probabilities");
        if(!base.is(json_type_array) || !hidden.is(json_type_array) || base.size()!=hidden.size())throw std::runtime_error("invalid feature shape");
        if((e.type=="noul" && base.size()!=1) || (e.type=="choice" && (base.size()<2 || base.size()>255)) ||
           (e.type=="score" && (base.size()<2 || base.size()>10)) || (e.type!="noul" && e.type!="choice" && e.type!="score"))throw std::runtime_error("invalid primitive shape");
        for(size_t k=0;k<base.size();++k){
            double z=base.at(k).number();if(std::abs(z)>1e6)throw std::runtime_error("bad base logit");e.base.push_back(z);
            auto h=hidden.at(k);if(!h.is(json_type_array) || h.size()==0 || h.size()>8192)throw std::runtime_error("bad hidden vector");
            if(dim==0)dim=h.size();
            if(h.size()!=dim)throw std::runtime_error("inconsistent dimensions");
            std::vector<double> vector;double norm=0;for(size_t i=0;i<h.size();++i){double v=h.at(i).number();vector.push_back(v);norm+=v*v;}
            if(std::abs(norm-1)>1e-5)throw std::runtime_error("non-normalized feature");
            e.hidden.push_back(std::move(vector));
        }
        if(!target.is(json_type_array))throw std::runtime_error("teacher probabilities required");
        for(size_t i=0;i<target.size();++i)r.teacher.push_back(target.at(i).number());
        jax::validate_teacher(r.teacher,e.type=="noul"?2:e.base.size());
        e.target=size_t(std::max_element(r.teacher.begin(),r.teacher.end())-r.teacher.begin());rows.push_back(std::move(r));
    }
    if(!in.eof() || rows.empty())throw std::runtime_error("empty/invalid features");
    return rows;
}
static auto split(const std::vector<jax::DistillExample>& rows,const std::string& name){std::vector<jax::DistillExample> out;for(const auto& r:rows)if(r.features.split==name)out.push_back(r);return out;}
static void save(const std::filesystem::path& path,const Json& value){auto tmp=path;tmp+=".tmp";{std::ofstream out(tmp);out<<value.dump()<<'\n';if(!out)throw std::runtime_error("write failed");}std::filesystem::rename(tmp,path);}
int main(int argc,char** argv){
    try{
        if(argc!=4)throw std::runtime_error("Usage: distill-train fit features.jsonl new-output-directory | evaluate test-features.jsonl head.json");
        auto rows=read_rows(argv[2]);
        if(std::string(argv[1])=="evaluate"){
            std::ifstream in(argv[3]);std::string text{std::istreambuf_iterator<char>(in),{}};
            auto head=jax::AbelHead::load(Json::parse(text));auto test=split(rows,"test");
            std::cout<<jax::distill_metrics(test,head).dump()<<'\n';return 0;
        }
        if(std::string(argv[1])!="fit")throw std::runtime_error("invalid command");
        for(const auto& r:rows)if(r.features.split=="test")throw std::runtime_error("fit input must physically exclude test rows");
        auto train=split(rows,"train"),validation=split(rows,"validation"),cal=split(rows,"calibration");
        if(train.empty() || validation.empty() || cal.empty())throw std::runtime_error("train/validation/calibration required");
        std::filesystem::path dir=argv[3];if(std::filesystem::exists(dir))throw std::runtime_error("output must be a new directory");
        size_t dim=train[0].features.hidden[0].size();jax::AbelHead chosen(dim);
        double best=std::numeric_limits<double>::infinity(),chosen_l2=0;int chosen_epoch=0;
        for(double l2:{.001,.01,.1}){
            jax::AbelHead h(dim);std::vector<double> m(dim+2),v(dim+2);
            for(int epoch=1;epoch<=500;++epoch){
                auto g=jax::distill_gradient(train,h,l2);double norm=0;for(double x:g)norm+=x*x;double scale=norm>25?5/std::sqrt(norm):1;
                for(size_t i=0;i<g.size();++i){double grad=g[i]*scale;m[i]=.9*m[i]+.1*grad;v[i]=.999*v[i]+.001*grad*grad;
                    double step=.025*(m[i]/(1-std::pow(.9,epoch)))/(std::sqrt(v[i]/(1-std::pow(.999,epoch)))+1e-8);
                    if(i<dim)h.weights[i]-=step;else if(i==dim)h.base_scale=std::clamp(h.base_scale-step,.05,4.0);else h.bias-=step;}
                if(epoch%10==0){double loss=jax::distill_loss(validation,h);if(loss<best){best=loss;chosen=h;chosen_l2=l2;chosen_epoch=epoch;}}
            }
        }
        chosen.temperature=jax::distill_temperature(cal,chosen);
        std::filesystem::create_directories(dir);
        auto checkpoint=chosen.json();checkpoint.set("training_method",Json::str("teacher-soft-distribution-distillation"));
        save(dir/"decision_head.json",checkpoint);
        auto report=Json::object();report.set("objective",Json::str("mean CE(teacher probabilities, student probabilities) + L2; held-out soft-target calibration"));
        report.set("teacher_metrics_are_ground_truth_accuracy",Json::boolean(false));report.set("test_used_for_selection",Json::boolean(false));
        report.set("selected_l2",Json::num(chosen_l2));report.set("selected_epoch",Json::num(chosen_epoch));report.set("temperature",Json::num(chosen.temperature));
        report.set("train",jax::distill_metrics(train,chosen));report.set("validation",jax::distill_metrics(validation,chosen));report.set("calibration",jax::distill_metrics(cal,chosen));
        save(dir/"training_report.json",report);std::cout<<report.dump()<<'\n';
    }catch(const std::exception& e){std::cerr<<"distill-train: "<<e.what()<<'\n';return 1;}
}
