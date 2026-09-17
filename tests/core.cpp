#include <jax/jax.hpp>
#include <iostream>
#include <limits>
#include <set>
using namespace jax;
void check(bool x) { if (!x) throw std::runtime_error("test failed"); }
void near(double a, double b) { check(std::abs(a-b) < 1e-10); }
template<class F> void rejects(F f) { bool caught = false; try { f(); } catch (const std::exception&) { caught = true; } check(caught); }
int main() {
    auto p = softmax({10000,10000}); near(p[0],0.5); near(confidence(p),0);
    near(confidence({1,0}),1);
    rejects([]{softmax({std::numeric_limits<double>::quiet_NaN()});});
    rejects([]{softmax({0},0);});
    std::vector<Question> qs{
        {"hidden-choice-id","route",Kind::choice,{{"a","alpha"},{"b","beta"}}},
        {"hidden-score-id","severity",Kind::score,{{"","mild"},{"","severe"}}},
        {"hidden-noul-id","refund?",Kind::noul,{}}
    };
    std::mutex mutex; std::vector<Judgment> seen;
    Backend backend = [&](const Judgment& j) {
        std::lock_guard lock(mutex); seen.push_back(j);
        if (j.instructions == "refund?") return std::log(3.0);
        if (j.criterion == "a" || j.criterion == "mild") return 0.0;
        return std::log(3.0);
    };
    auto a = decide("state",qs,backend,4);
    check(a[0].choice == "b"); near(a[0].probabilities[1],0.75);
    near(a[1].score,0.75); near(a[2].noul,0.75); check(seen.size()==5);
    for (const auto& j : seen) {
        check(j.state == "state");
        if (j.instructions == "severity") {check(j.alternatives.empty()); check(j.criterion=="mild" || j.criterion=="severe");}
    }
    auto renamed = qs; renamed[0].id="different";
    auto b = decide("state",renamed,backend,1); near(a[0].probabilities[0],b[0].probabilities[0]);
    auto isolated = decide("state",{qs[1]},backend); near(isolated[0].score,a[1].score);
    std::swap(renamed[1].criteria[0],renamed[1].criteria[1]);
    near(decide("state",renamed,backend)[1].score,0.25);
    auto invalid = qs; invalid[0].criteria[1].name="a";
    rejects([&]{decide("state",invalid,[](auto&)->double{throw std::runtime_error("should not run");});});
    rejects([&]{decide("state",qs,[](auto&)->double{throw std::runtime_error("backend failed");});});
    rejects([&]{decide("state",qs,[](auto&){return std::numeric_limits<double>::infinity();});});
    std::vector<CalibrationRow> rows{{{0,8},1},{{0,8},0},{{0,8},1}};
    auto t = fit_temperature(rows); check(nll(rows,t) < nll(rows,1));
    rejects([]{fit_temperature({});});
    rejects([]{nll({{{1,2},2}},1);});
    std::cout << "core: distributions, isolation, validation, failures, calibration OK\n";
}
