#include <jax/abel_head.hpp>
#include <iostream>
#include <functional>
using namespace jax;
void check(bool b){if(!b)throw std::runtime_error("Abel test failed");}
void rejects(const std::function<void()>& f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}check(caught);}
int main(){
    AbelHead h(2);h.weights={0.3,-0.4};h.base_scale=0.8;h.bias=-0.1;
    std::vector<AbelExample> rows{
        {"n","train","noul","fixture",{{1,0}},{0.7},1},
        {"c","train","choice","fixture",{{0,1},{1,0}},{-0.2,0.3},0},
        {"s","train","score","fixture",{{0.6,0.8},{-0.6,0.8}},{0.1,-0.3},1}
    };
    const double l2=.03,eps=1e-6;
    auto loss=[&](const AbelHead& a){double penalty=(a.base_scale-1)*(a.base_scale-1)+a.bias*a.bias;for(double w:a.weights)penalty+=w*w;return abel_loss(rows,a)+l2*.5*penalty;};
    auto g=abel_gradient(rows,h,l2);
    for(size_t i=0;i<g.size();++i){
        AbelHead a=h,b=h;
        if(i<2){a.weights[i]+=eps;b.weights[i]-=eps;}else if(i==2){a.base_scale+=eps;b.base_scale-=eps;}else{a.bias+=eps;b.bias-=eps;}
        check(std::abs((loss(a)-loss(b))/(2*eps)-g[i])<1e-7);
    }
    auto loaded=AbelHead::load(h.json());check(std::abs(loaded.logit(.3,{.6,.8})-h.logit(.3,{.6,.8}))<1e-12);
    double before=abel_loss(rows,h);
    for(int i=0;i<100;++i){auto grad=abel_gradient(rows,h,0);for(size_t k=0;k<2;++k)h.weights[k]-=.1*grad[k];h.base_scale-=.1*grad[2];h.bias-=.1*grad[3];}
    check(abel_loss(rows,h)<before);
    auto bad=loaded.json();bad.set("temperature",Json::num(0));rejects([&]{AbelHead::load(bad);});
    bad=loaded.json();bad.set("hidden_dim",Json::num(3));rejects([&]{AbelHead::load(bad);});
    rejects([&]{loaded.logit(0,{1});});
    auto metrics=abel_metrics(rows,loaded,1);check(metrics.at("nll").number()>0);
    std::cout<<"Abel: finite-difference gradients, learning, checkpoint roundtrip, rejection OK\n";
}
