#pragma once
#include <jax/json.hpp>
#include <jax/jax.hpp>
namespace jax {
// Trainable residual decision head on L2-normalized final hidden features.
struct AbelHead {
    std::vector<double> weights;
    double base_scale=1, bias=0, temperature=1;
    explicit AbelHead(size_t dim=1024) : weights(dim,0) {}
    double logit(double base, const std::vector<double>& hidden) const {
        if (hidden.size()!=weights.size() || hidden.empty()) throw std::runtime_error("Abel hidden dimension mismatch");
        double z=base_scale*base+bias;
        for (size_t i=0;i<weights.size();++i) z+=weights[i]*hidden[i];
        if (!std::isfinite(z) || std::abs(z)>1e6) throw std::runtime_error("Abel produced invalid logit");
        return z;
    }
    Json json() const {
        auto j=Json::object(), w=Json::array();
        for (double v:weights) w.push(Json::num(v));
        j.set("format",Json::str("jax.abel.residual-linear.v1"));
        j.set("model",Json::str("Jax-1-Abel"));
        j.set("model_version",Json::str("0.1.0"));
        j.set("hidden_normalization",Json::str("l2"));
        j.set("prompt_version",Json::str("jax-judgment-v1"));
        j.set("feature_layer",Json::str("final_norm_last_token"));
        j.set("base_gguf_sha256",Json::str("0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c"));
        j.set("hidden_dim",Json::num(weights.size())); j.set("weights",w);
        j.set("base_scale",Json::num(base_scale)); j.set("bias",Json::num(bias));
        j.set("temperature",Json::num(temperature)); return j;
    }
    static AbelHead load(const Json& j) {
        if (j.at("format").string()!="jax.abel.residual-linear.v1" || j.at("model").string()!="Jax-1-Abel" || j.at("model_version").string()!="0.1.0" ||
            j.at("hidden_normalization").string()!="l2" || j.at("prompt_version").string()!="jax-judgment-v1" ||
            j.at("feature_layer").string()!="final_norm_last_token" ||
            j.at("base_gguf_sha256").string()!="0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c")
            throw std::runtime_error("unsupported Abel checkpoint contract");
        double dim=j.at("hidden_dim").number();
        if (dim<1 || dim>8192 || std::floor(dim)!=dim) throw std::runtime_error("invalid hidden dimension");
        auto w=j.at("weights");
        if (!w.is(json_type_array) || w.size()!=size_t(dim)) throw std::runtime_error("invalid Abel weights");
        AbelHead h{size_t(dim)};
        for (size_t i=0;i<w.size();++i) {
            h.weights[i]=w.at(i).number();
            if (std::abs(h.weights[i])>1e4) throw std::runtime_error("Abel weight out of range");
        }
        h.base_scale=j.at("base_scale").number();h.bias=j.at("bias").number();h.temperature=j.at("temperature").number();
        if (h.base_scale<0.05 || h.base_scale>4 || std::abs(h.bias)>1e4 || h.temperature<0.05 || h.temperature>20.000001)
            throw std::runtime_error("Abel parameters out of range");
        return h;
    }
};
struct AbelExample {
    std::string id, split, type, family;
    std::vector<std::vector<double>> hidden;
    std::vector<double> base;
    size_t target;
};
inline std::vector<double> abel_logits(const AbelExample& e,const AbelHead& h) {
    std::vector<double> z;
    if (e.type=="noul") z.push_back(0);
    for (size_t i=0;i<e.base.size();++i) z.push_back(h.logit(e.base[i],e.hidden[i]));
    return z;
}
inline double abel_loss(const std::vector<AbelExample>& rows,const AbelHead& h,double temperature=1) {
    if (rows.empty()) throw std::runtime_error("empty Abel split");
    std::vector<CalibrationRow> r;
    for (const auto& e:rows) r.push_back({abel_logits(e,h),e.target});
    return nll(r,temperature);
}
inline Json abel_metrics(const std::vector<AbelExample>& rows,const AbelHead& head,double temperature) {
    double brier=0,accuracy=0,confidence_sum[10]{},correct_sum[10]{};
    double score_mae=0;size_t score_count=0;
    for (const auto& row:rows) {
        auto p=softmax(abel_logits(row,head),temperature);
        if (row.target>=p.size()) throw std::runtime_error("target out of range");
        size_t best=std::max_element(p.begin(),p.end())-p.begin();
        double correct=best==row.target;accuracy+=correct;
        size_t bin=std::min(size_t(p[best]*10),size_t(9));
        confidence_sum[bin]+=p[best];correct_sum[bin]+=correct;
        for (size_t i=0;i<p.size();++i) brier+=std::pow(p[i]-double(i==row.target),2);
        if (row.type=="score") {
            double expectation=0;for (size_t i=0;i<p.size();++i) expectation+=i*p[i];
            score_mae+=std::abs(expectation-row.target);++score_count;
        }
    }
    auto result=Json::object();
    result.set("nll",Json::num(abel_loss(rows,head,temperature)));
    result.set("tasks",Json::num(rows.size()));result.set("accuracy",Json::num(accuracy/rows.size()));
    result.set("brier",Json::num(brier/rows.size()));
    double ece=0;for (size_t i=0;i<10;++i) ece+=std::abs(confidence_sum[i]-correct_sum[i]);
    result.set("ece_10",Json::num(ece/rows.size()));
    if (score_count) result.set("score_mae",Json::num(score_mae/score_count));
    return result;
}
// Full-batch CE gradient; one task contributes one loss regardless of option count.
inline std::vector<double> abel_gradient(const std::vector<AbelExample>& rows,const AbelHead& h,double l2) {
    std::vector<double> g(h.weights.size()+2,0);
    if (rows.empty()) throw std::runtime_error("empty train split");
    for (const auto& row:rows) {
        auto p=softmax(abel_logits(row,h));
        for (size_t k=0;k<row.base.size();++k) {
            size_t index=k+(row.type=="noul"?1:0);
            double dz=(p[index]-double(index==row.target))/rows.size();
            for (size_t i=0;i<h.weights.size();++i) g[i]+=dz*row.hidden[k][i];
            g[h.weights.size()]+=dz*row.base[k];g[h.weights.size()+1]+=dz;
        }
    }
    for (size_t i=0;i<h.weights.size();++i) g[i]+=l2*h.weights[i];
    g[h.weights.size()]+=l2*(h.base_scale-1);g[h.weights.size()+1]+=l2*h.bias;
    return g;
}
}
