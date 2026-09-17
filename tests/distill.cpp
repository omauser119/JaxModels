#include <jax/distill.hpp>
#include <cassert>
#include <iostream>
using namespace jax;
int main(){
    AbelHead h(2);h.weights={.2,-.1};h.base_scale=.8;h.bias=.05;
    AbelExample a{"a","train","noul","f",{{1,0}},{.3},1};
    AbelExample b{"b","train","choice","f",{{1,0},{0,1}},{-.2,.5},0};
    std::vector<DistillExample> rows{{a,{.2,.8}},{b,{.7,.3}}};double l2=.01;
    auto objective=[&](const AbelHead& x){double v=distill_loss(rows,x);for(double w:x.weights)v+=.5*l2*w*w;return v+.5*l2*(std::pow(x.base_scale-1,2)+x.bias*x.bias);};
    auto g=distill_gradient(rows,h,l2);
    for(size_t i=0;i<g.size();++i){
        auto hi=h,lo=h;double* plus=i<2?&hi.weights[i]:i==2?&hi.base_scale:&hi.bias;
        double* minus=i<2?&lo.weights[i]:i==2?&lo.base_scale:&lo.bias;
        *plus+=1e-5;*minus-=1e-5;double numerical=(objective(hi)-objective(lo))/2e-5;
        assert(std::abs(numerical-g[i])<1e-6);
    }
    double initial=distill_loss(rows,h);
    for(int n=0;n<200;++n){auto grad=distill_gradient(rows,h,0);for(size_t i=0;i<2;++i)h.weights[i]-=.1*grad[i];h.base_scale-=.1*grad[2];h.bias-=.1*grad[3];}
    assert(distill_loss(rows,h)<initial);
    auto alternate=rows;alternate[0].teacher={.01,.99};
    assert(std::abs(distill_loss(alternate,h)-distill_loss(rows,h))>1e-3);
    double t=distill_temperature(rows,h);assert(distill_loss(rows,h,t)<=distill_loss(rows,h,1)+1e-12);
    bool rejected=false;try{validate_teacher({.4,.4},2);}catch(...){rejected=true;}assert(rejected);
    std::cout<<"Distillation: soft-target finite-difference gradient, learning, distribution sensitivity, calibration and rejection OK\n";
}
