#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <functional>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace jax {
struct Option { std::string name, description; };
enum class Kind { choice, score, noul };
struct Question {
    std::string id, instructions;
    Kind kind;
    std::vector<Option> criteria;
};
// Deliberately excludes question ID, level index and previous answers.
struct Judgment {
    std::string state, instructions, criterion;
    std::vector<Option> alternatives; // Choice only
};
using Backend = std::function<double(const Judgment&)>; // log P(1) - log P(0), thread-safe
struct Answer {
    std::string id;
    Kind kind;
    std::vector<double> probabilities, raw_logits;
    std::string choice;
    double score = 0, noul = 0, confidence = 0;
};
inline std::vector<double> softmax(std::vector<double> x, double temperature = 1) {
    if (x.empty() || !std::isfinite(temperature) || temperature < 0.001)
        throw std::invalid_argument("invalid distribution or temperature (< 0.001)");
    for (double v : x) if (!std::isfinite(v) || std::abs(v) > 1e6)
        throw std::invalid_argument("nonfinite or out-of-range logit");
    double m = *std::max_element(x.begin(), x.end()), sum = 0;
    for (double& v : x) { v = std::exp((v-m)/temperature); sum += v; }
    for (double& v : x) v /= sum;
    return x;
}
inline double confidence(const std::vector<double>& p) {
    if (p.size() == 1) return 1;
    double entropy = 0;
    for (double v : p) if (v > 0) entropy -= v * std::log(v);
    return std::clamp(1 - entropy/std::log(double(p.size())), 0.0, 1.0);
}
inline void validate(const std::vector<Question>& questions) {
    if (questions.empty() || questions.size() > 256)
        throw std::invalid_argument("expected 1..256 questions");
    std::vector<std::string> ids;
    size_t jobs = 0;
    for (const auto& q : questions) {
        if (q.id.empty() || q.instructions.empty()) throw std::invalid_argument("empty ID/instructions");
        if (std::find(ids.begin(), ids.end(), q.id) != ids.end()) throw std::invalid_argument("duplicate question ID");
        ids.push_back(q.id);
        size_t n = q.criteria.size();
        if (q.kind == Kind::choice && (n < 2 || n > 255)) throw std::invalid_argument("Choice requires 2..255 options");
        if (q.kind == Kind::score && (n < 2 || n > 10)) throw std::invalid_argument("Score requires 2..10 levels");
        if (q.kind == Kind::noul && n != 0 && n != 2) throw std::invalid_argument("Noul criteria require true and false");
        std::vector<std::string> names;
        for (const auto& c : q.criteria) {
            if (q.kind == Kind::choice) {
                if (c.name.empty() || std::find(names.begin(), names.end(), c.name) != names.end())
                    throw std::invalid_argument("empty or duplicate option name");
                names.push_back(c.name);
            } else if (c.description.empty()) throw std::invalid_argument("empty criterion");
        }
        jobs += q.kind == Kind::noul ? 1 : n;
    }
    if (jobs > 4096) throw std::invalid_argument("request exceeds 4096 judgments");
}
inline std::vector<Judgment> judgments(const std::string& state, const Question& q) {
    std::vector<Judgment> out;
    size_t n = q.kind == Kind::noul ? 1 : q.criteria.size();
    for (size_t k=0;k<n;++k) {
        Judgment j{state,q.instructions,"",{}};
        if (q.kind==Kind::choice) { j.criterion=q.criteria[k].name; j.alternatives=q.criteria; }
        else if (q.kind==Kind::score) j.criterion=q.criteria[k].description;
        else if (!q.criteria.empty()) j.criterion="YES: "+q.criteria[0].description+"\nNO: "+q.criteria[1].description;
        out.push_back(std::move(j));
    }
    return out;
}
inline std::vector<Answer> decide(const std::string& state, const std::vector<Question>& questions,
                                 const Backend& backend, size_t workers = 4, double temperature = 1) {
    validate(questions);
    (void)softmax({0}, temperature);
    if (workers < 1 || workers > 64) throw std::invalid_argument("workers must be 1..64");
    struct Job { size_t q, k; Judgment judgment; };
    std::vector<Job> jobs;
    std::vector<std::vector<double>> logits(questions.size());
    for (size_t i = 0; i < questions.size(); ++i) {
        const auto& q = questions[i];
        auto items=judgments(state,q);
        logits[i].resize(items.size());
        for (size_t k=0;k<items.size();++k) jobs.push_back({i,k,std::move(items[k])});
    }
    std::atomic<size_t> cursor{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    {
        std::vector<std::jthread> threads;
        for (size_t t = 0; t < std::min(workers, jobs.size()); ++t) threads.emplace_back([&] {
            while (!failed.load()) {
                size_t index = cursor.fetch_add(1);
                if (index >= jobs.size()) break;
                const auto& job = jobs[index];
                try {
                    double z = backend(job.judgment);
                    if (!std::isfinite(z) || std::abs(z) > 1e6) throw std::runtime_error("invalid backend logit");
                    logits[job.q][job.k] = z;
                } catch (...) {
                    std::lock_guard lock(error_mutex);
                    if (!error) error = std::current_exception();
                    failed.store(true);
                }
            }
        });
    } // joins before reading results
    if (error) std::rethrow_exception(error);
    std::vector<Answer> out;
    for (size_t i = 0; i < questions.size(); ++i) {
        const auto& q = questions[i];
        Answer a{}; a.id = q.id; a.kind = q.kind;
        a.raw_logits = q.kind == Kind::noul ? std::vector<double>{0, logits[i][0]} : logits[i];
        if (q.kind == Kind::noul) a.noul = softmax({0, logits[i][0]}, temperature)[1];
        else {
            a.probabilities = softmax(logits[i], temperature);
            a.confidence = confidence(a.probabilities);
            if (q.kind == Kind::choice) {
                size_t k = std::max_element(a.probabilities.begin(), a.probabilities.end()) - a.probabilities.begin();
                a.choice = q.criteria[k].name;
            } else for (size_t k = 0; k < a.probabilities.size(); ++k) a.score += k*a.probabilities[k];
        }
        out.push_back(std::move(a));
    }
    return out;
}
struct CalibrationRow { std::vector<double> logits; size_t label; };
inline double nll(const std::vector<CalibrationRow>& rows, double temperature) {
    if (rows.empty()) throw std::invalid_argument("empty calibration set");
    double loss = 0;
    for (const auto& r : rows) {
        if (r.logits.size() < 2 || r.label >= r.logits.size()) throw std::invalid_argument("invalid calibration row");
        (void)softmax(r.logits, temperature);
        double m = *std::max_element(r.logits.begin(), r.logits.end()), sum = 0;
        for (double v : r.logits) sum += std::exp((v-m)/temperature);
        loss += (m-r.logits[r.label])/temperature + std::log(sum);
    }
    return loss / rows.size();
}
// Fits one shared temperature to cached raw logits; no backbone training, no RLCD.
inline double fit_temperature(const std::vector<CalibrationRow>& rows) {
    double best = 1, loss = nll(rows, 1);
    for (int i = 0; i <= 400; ++i) {
        double t = std::exp(std::log(0.05) + i/400.0*std::log(400.0));
        double candidate = nll(rows, t);
        if (candidate < loss) { loss = candidate; best = t; }
    }
    return best;
}
} // namespace jax
