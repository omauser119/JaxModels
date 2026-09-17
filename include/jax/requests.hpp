#pragma once
#include <jax/jax.hpp>
#include <jax/json.hpp>
namespace jax {
inline std::string description(const Json& v) { return v.is(json_type_string) ? v.string() : v.dump(); }
inline std::vector<jax::Question> parse_questions(const Json& req) {
    auto qs = req.at("questions"); std::vector<jax::Question> out;
    for (const auto& id : qs.keys()) {
        auto q = qs.at(id); auto type = q.at("type").string();
        jax::Question item{id,description(q.at("instructions")),jax::Kind::noul,{}};
        if (type == "choice") {
            item.kind = jax::Kind::choice;
            for (const auto& name : q.at("criteria").keys()) item.criteria.push_back({name,description(q.at("criteria").at(name))});
            std::sort(item.criteria.begin(),item.criteria.end(),[](const auto& a, const auto& b){ return a.name < b.name; });
        } else if (type == "score") {
            item.kind = jax::Kind::score; auto c = q.at("criteria");
            if (!c.is(json_type_array)) throw std::runtime_error("Score criteria must be an array");
            for (size_t i = 0; i < c.size(); ++i) item.criteria.push_back({"",description(c.at(i))});
        } else if (type == "noul") {
            if (q.has("criteria")) {
                auto c = q.at("criteria");
                if (c.size() != 2) throw std::runtime_error("Noul requires true/false criteria");
                item.criteria = {{"true",description(c.at("true"))},{"false",description(c.at("false"))}};
            }
        } else throw std::runtime_error("unknown question type: " + type);
        out.push_back(std::move(item));
    }
    jax::validate(out); return out;
}
}
