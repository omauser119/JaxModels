#include <jax/abel_head.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <set>
#include <limits>
using jax::Json;
static std::vector<jax::AbelExample> load_rows(const std::string& path) {
    std::ifstream in(path);if (!in) throw std::runtime_error("cannot read features");
    std::string line; std::vector<jax::AbelExample> rows;std::set<std::string> ids;
    size_t dim=0;
    while (std::getline(in,line)) {
        if (line.empty()) continue;
        auto j=Json::parse(line);jax::AbelExample e;
        e.id=j.at("id").string();e.split=j.at("split").string();e.type=j.at("type").string();e.family=j.at("family").string();
        if (!ids.insert(e.id).second) throw std::runtime_error("duplicate feature ID");
        if (e.type!="choice" && e.type!="score" && e.type!="noul") throw std::runtime_error("invalid primitive");
        if (e.split!="train" && e.split!="validation" && e.split!="calibration" && e.split!="test") throw std::runtime_error("invalid split");
        auto base=j.at("base_logits"), hidden=j.at("hidden");
        if (!base.is(json_type_array) || !hidden.is(json_type_array) || base.size()!=hidden.size() || base.size()==0 || base.size()>255)
            throw std::runtime_error("invalid feature shape");
        if ((e.type=="noul" && base.size()!=1) || (e.type!="noul" && base.size()<2)) throw std::runtime_error("wrong primitive shape");
        for (size_t k=0;k<base.size();++k) {
            e.base.push_back(base.at(k).number());auto h=hidden.at(k);
            if (!h.is(json_type_array) || h.size()==0 || h.size()>8192) throw std::runtime_error("invalid hidden vector");
            if (dim==0) dim=h.size();
            if (dim!=h.size()) throw std::runtime_error("inconsistent feature dimension");
            std::vector<double> v;double norm=0;
            for (size_t i=0;i<h.size();++i) {double x=h.at(i).number();norm+=x*x;v.push_back(x);}
            if (std::abs(norm-1)>1e-5) throw std::runtime_error("features must have unit L2 norm");
            e.hidden.push_back(std::move(v));
        }
        double label=j.at("target").number();size_t classes=e.type=="noul"?2:e.base.size();
        if (label<0 || label>=classes || std::floor(label)!=label) throw std::runtime_error("invalid label");
        e.target=size_t(label);rows.push_back(std::move(e));
    }
    if (!in.eof() || rows.empty()) throw std::runtime_error("empty/invalid feature file");
    return rows;
}
static auto subset(const std::vector<jax::AbelExample>& rows,const std::string& split,const std::string& type="") {
    std::vector<jax::AbelExample> result;
    for (const auto& row:rows) if (row.split==split && (type.empty() || row.type==type)) result.push_back(row);
    return result;
}
static Json report(const std::vector<jax::AbelExample>& rows,const jax::AbelHead& h,double baseline_temperature=1) {
    auto out=Json::object();jax::AbelHead baseline(h.weights.size());
    out.set("baseline",jax::abel_metrics(rows,baseline,1));
    out.set("baseline_calibrated",jax::abel_metrics(rows,baseline,baseline_temperature));
    auto ablated=h;std::fill(ablated.weights.begin(),ablated.weights.end(),0);
    out.set("abel_without_residual_weights",jax::abel_metrics(rows,ablated,h.temperature));
    out.set("abel_uncalibrated",jax::abel_metrics(rows,h,1));
    out.set("abel",jax::abel_metrics(rows,h,h.temperature));
    auto by_type=Json::object();
    for (const auto& type:{"noul","choice","score"}) {
        std::vector<jax::AbelExample> r;for (const auto& e:rows) if(e.type==type) r.push_back(e);
        if (r.empty()) continue;
        auto pair=Json::object();pair.set("baseline",jax::abel_metrics(r,baseline,1));pair.set("baseline_calibrated",jax::abel_metrics(r,baseline,baseline_temperature));pair.set("abel",jax::abel_metrics(r,h,h.temperature));
        by_type.set(type,pair);
    }
    out.set("by_type",by_type);return out;
}
static void save(const std::filesystem::path& path,const Json& j) {
    auto partial=path;partial+=".tmp";
    {std::ofstream out(partial);out<<j.dump()<<'\n';if(!out)throw std::runtime_error("checkpoint write failed");}
    std::filesystem::rename(partial,path);
}
int main(int argc,char** argv) {
    try {
        if (argc<4) throw std::runtime_error("Usage: abel-train fit features.jsonl output-directory | evaluate features.jsonl head.json split");
        auto rows=load_rows(argv[2]);std::string command=argv[1];
        if(command=="evaluate") {
            if(argc!=5) throw std::runtime_error("evaluate requires checkpoint and split");
            std::ifstream input(argv[3]);std::string text{std::istreambuf_iterator<char>(input),{}};
            auto checkpoint=Json::parse(text);
            auto h=jax::AbelHead::load(checkpoint);
            double baseline_temperature=checkpoint.has("baseline_temperature")?checkpoint.at("baseline_temperature").number():1;
            std::cout<<report(subset(rows,argv[4]),h,baseline_temperature).dump()<<'\n';return 0;
        }
        if(command!="fit" || argc!=4)throw std::runtime_error("invalid command");
        auto train=subset(rows,"train"), val=subset(rows,"validation"), cal=subset(rows,"calibration");
        if(train.empty() || val.empty() || cal.empty())throw std::runtime_error("train, validation, calibration splits are required");
        size_t dim=rows.front().hidden.front().size();
        jax::AbelHead selected(dim);double best=std::numeric_limits<double>::infinity();
        double selected_l2=0;int selected_epoch=0;auto candidates=Json::array();
        for(double l2:{0.001,0.01,0.1}) {
            jax::AbelHead h(dim);std::vector<double> m(dim+2,0),v(dim+2,0);
            double local_best=std::numeric_limits<double>::infinity();int local_epoch=0;
            for(int epoch=1;epoch<=500;++epoch) {
                auto g=jax::abel_gradient(train,h,l2);double norm=0;for(double x:g)norm+=x*x;
                double scale=norm>25?5/std::sqrt(norm):1;
                for(size_t i=0;i<g.size();++i) {
                    double grad=g[i]*scale;
                    m[i]=0.9*m[i]+0.1*grad;v[i]=0.999*v[i]+0.001*grad*grad;
                    double update=0.025*(m[i]/(1-std::pow(0.9,epoch)))/(std::sqrt(v[i]/(1-std::pow(0.999,epoch)))+1e-8);
                    if(i<dim)h.weights[i]-=update;
                    else if(i==dim)h.base_scale=std::clamp(h.base_scale-update,0.05,4.0);
                    else h.bias-=update;
                }
                if(epoch%10!=0)continue;
                double loss=jax::abel_loss(val,h);
                if(loss<local_best){local_best=loss;local_epoch=epoch;}
                if(loss<best){best=loss;selected=h;selected_l2=l2;selected_epoch=epoch;}
            }
            auto c=Json::object();c.set("l2",Json::num(l2));c.set("best_epoch",Json::num(local_epoch));c.set("validation_nll",Json::num(local_best));candidates.push(c);
            std::cerr<<"L2="<<l2<<" val NLL="<<local_best<<" epoch="<<local_epoch<<'\n';
        }
        std::vector<jax::CalibrationRow> calibration;
        for(const auto& row:cal)calibration.push_back({jax::abel_logits(row,selected),row.target});
        selected.temperature=jax::fit_temperature(calibration);
        jax::AbelHead baseline(dim);std::vector<jax::CalibrationRow> baseline_calibration;
        for(const auto& row:cal)baseline_calibration.push_back({jax::abel_logits(row,baseline),row.target});
        double baseline_temperature=jax::fit_temperature(baseline_calibration);
        std::filesystem::path dir=argv[3];std::filesystem::create_directories(dir);
        if(std::filesystem::exists(dir/"decision_head.json"))throw std::runtime_error("output checkpoint already exists; choose a new directory");
        auto checkpoint=selected.json();checkpoint.set("baseline_temperature",Json::num(baseline_temperature));
        save(dir/"decision_head.json",checkpoint);
        auto result=Json::object();result.set("model",Json::str("Jax-1-Abel"));
        result.set("optimizer",Json::str("full-batch Adam, lr=0.025, beta1=0.9, beta2=0.999, gradient norm <=5"));
        result.set("objective",Json::str("task-mean categorical/binary cross-entropy + L2 residual regularization"));
        result.set("selected_l2",Json::num(selected_l2));result.set("selected_epoch",Json::num(selected_epoch));
        result.set("candidates",candidates);result.set("trainable_parameters",Json::num(dim+2));
        result.set("temperature",Json::num(selected.temperature));
        result.set("baseline_temperature",Json::num(baseline_temperature));
        result.set("train",report(train,selected,baseline_temperature));result.set("validation",report(val,selected,baseline_temperature));result.set("calibration",report(cal,selected,baseline_temperature));
        result.set("test_used_for_selection",Json::boolean(false));
        save(dir/"training_report.json",result);
        std::cout<<result.dump()<<'\n';
    }catch(const std::exception& e){std::cerr<<"abel-train: "<<e.what()<<'\n';return 1;}
}
