#pragma once
#include <jax/abel_head.hpp>
namespace jax {
struct DistillExample { AbelExample features; std::vector<double> teacher; };
inline void validate_teacher(const std::vector<double>& p,size_t classes) {
    if(p.size()!=classes)throw std::runtime_error("teacher probability dimension mismatch");
    double sum=0;for(double v:p){if(!std::isfinite(v) || v<0 || v>1)throw std::runtime_error("invalid teacher probability");sum+=v;}
    if(std::abs(sum-1)>1e-6)throw std::runtime_error("teacher probabilities must sum to one");
}
inline double distill_loss(const std::vector<DistillExample>& rows,const AbelHead& h,double t=1) {
    if(rows.empty())throw std::runtime_error("empty distillation split");
    double loss=0;
    for(const auto& r:rows){
        auto z=abel_logits(r.features,h);validate_teacher(r.teacher,z.size());(void)softmax(z,t);
        double max=*std::max_element(z.begin(),z.end()),sum=0;
        for(double v:z)sum+=std::exp((v-max)/t);
        for(size_t i=0;i<z.size();++i)loss+=r.teacher[i]*((max-z[i])/t+std::log(sum));
    }
    return loss/rows.size();
}
inline std::vector<double> distill_gradient(const std::vector<DistillExample>& rows,const AbelHead& h,double l2) {
    if(rows.empty())throw std::runtime_error("empty distillation split");
    std::vector<double> g(h.weights.size()+2,0);
    for(const auto& r:rows){
        auto p=softmax(abel_logits(r.features,h));validate_teacher(r.teacher,p.size());
        for(size_t k=0;k<r.features.base.size();++k){
            size_t i=k+(r.features.type=="noul"?1:0);
            double dz=(p[i]-r.teacher[i])/rows.size();
            for(size_t n=0;n<h.weights.size();++n)g[n]+=dz*r.features.hidden[k][n];
            g[h.weights.size()]+=dz*r.features.base[k];g[h.weights.size()+1]+=dz;
        }
    }
    for(size_t i=0;i<h.weights.size();++i)g[i]+=l2*h.weights[i];
    g[h.weights.size()]+=l2*(h.base_scale-1);g[h.weights.size()+1]+=l2*h.bias;
    return g;
}
inline double distill_temperature(const std::vector<DistillExample>& rows,const AbelHead& h){
    double best=1,loss=distill_loss(rows,h,1);
    for(int i=0;i<=400;++i){double t=std::exp(std::log(.05)+i/400.0*std::log(400.0));double v=distill_loss(rows,h,t);if(v<loss){best=t;loss=v;}}
    return best;
}
inline Json distill_metrics(const std::vector<DistillExample>& rows,const AbelHead& h){
    double entropy=0,agreement=0,brier=0;
    for(const auto& r:rows){
        auto p=softmax(abel_logits(r.features,h),h.temperature);validate_teacher(r.teacher,p.size());
        for(size_t i=0;i<p.size();++i){if(r.teacher[i]>0)entropy-=r.teacher[i]*std::log(r.teacher[i]);brier+=std::pow(p[i]-r.teacher[i],2);}
        agreement+=(std::max_element(p.begin(),p.end())-p.begin())==(std::max_element(r.teacher.begin(),r.teacher.end())-r.teacher.begin());
    }
    auto result=Json::object();double ce=distill_loss(rows,h,h.temperature);
    result.set("tasks",Json::num(rows.size()));result.set("teacher_cross_entropy",Json::num(ce));
    result.set("teacher_kl",Json::num(std::max(0.0,ce-entropy/rows.size())));
    result.set("teacher_agreement",Json::num(agreement/rows.size()));result.set("teacher_probability_mse_sum",Json::num(brier/rows.size()));
    return result;
}
}
