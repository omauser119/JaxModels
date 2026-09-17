#include <jax/jax.hpp>
#include <jax/json.hpp>
#include <jax/prompt.hpp>
#include <jax/requests.hpp>
#ifdef JAX_WITH_LLAMA
#include <jax/llama_backend.hpp>
#include <jax/abel_head.hpp>
#endif
#ifndef JAX_WITH_LLAMA
#include <curl/curl.h>
#endif
#include <fstream>
#include <iostream>
#include <iterator>
#include <cstdlib>
#include <cstdio>
using jax::Json;
static std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path);
    std::string text; char buffer[4096];
    while (in.read(buffer, sizeof buffer) || in.gcount()) {
        text.append(buffer, size_t(in.gcount()));
        if (text.size() > 8*1024*1024) throw std::runtime_error("input exceeds 8 MiB");
    }
    if (!in.eof()) throw std::runtime_error("read failed");
    return text;
}
static std::string env(const char* key, const char* fallback) { const char* v = std::getenv(key); return v ? v : fallback; }
#ifndef JAX_WITH_LLAMA
static size_t receive(char* ptr, size_t size, size_t count, void* data) noexcept {
    try {
        auto& out = *static_cast<std::string*>(data);
        size_t n = size*count;
        if (n > 8*1024*1024 || out.size() > 8*1024*1024-n) return 0;
        out.append(ptr, n); return n;
    } catch (...) { return 0; }
}
static double binary_logit(const Json& response) {
    auto tokens = response.at("choices").at(size_t(0)).at("logprobs").at("content").at(size_t(0)).at("top_logprobs");
    bool seen[2] = {false,false}; double lp[2] = {};
    for (size_t i = 0; i < tokens.size(); ++i) {
        auto t = tokens.at(i); std::string token = t.at("token").string();
        // Require exact, single-token labels. Do not invent missing probabilities.
        if (token != "0" && token != "1") continue;
        int k = token == "1";
        if (seen[k]) throw std::runtime_error("duplicate decision token");
        lp[k] = t.at("logprob").number(); seen[k] = true;
        if (lp[k] > 0 || lp[k] <= -9999) throw std::runtime_error("invalid/sentinel token logprob");
    }
    if (!seen[0] || !seen[1]) throw std::runtime_error("backend must return both exact tokens 0 and 1 in top_logprobs; use a compatible model/server");
    return lp[1]-lp[0];
}
class HttpBackend {
    std::string endpoint, model, key;
public:
    HttpBackend() : endpoint(env("JAX_ENDPOINT", "http://127.0.0.1:8080/v1/chat/completions")),
                    model(env("JAX_MODEL", "local")), key(env("JAX_API_KEY", "")) {
        if (endpoint.rfind("http://",0) != 0 && endpoint.rfind("https://",0) != 0) throw std::runtime_error("endpoint must use HTTP(S)");
        if (key.find_first_of("\r\n") != std::string::npos) throw std::runtime_error("invalid API key");
    }
    double operator()(const jax::Judgment& j) const {
        auto [system, content] = jax::judgment_messages(j);
        auto messages = Json::array();
        for (const auto& entry : std::vector<std::pair<std::string,std::string>>{{"system",system},{"user",content}}) {
            auto m = Json::object(); m.set("role",Json::str(entry.first)); m.set("content",Json::str(entry.second)); messages.push(m);
        }
        auto request = Json::object();
        request.set("model",Json::str(model)); request.set("messages",messages);
        request.set("max_tokens",Json::num(1)); request.set("temperature",Json::num(1));
        request.set("seed",Json::num(0)); request.set("stream",Json::boolean(false));
        request.set("logprobs",Json::boolean(true)); request.set("top_logprobs",Json::num(20));
        std::string body = request.dump(), result;
        std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> handle(curl_easy_init(), curl_easy_cleanup);
        if (!handle) throw std::runtime_error("curl initialization failed");
        curl_slist* raw = curl_slist_append(nullptr, "Content-Type: application/json");
        if (!raw) throw std::bad_alloc();
        if (!key.empty()) {
            auto next = curl_slist_append(raw, ("Authorization: Bearer " + key).c_str());
            if (!next) { curl_slist_free_all(raw); throw std::bad_alloc(); } raw = next;
        }
        std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> headers(raw, curl_slist_free_all);
        auto h = handle.get();
        curl_easy_setopt(h,CURLOPT_URL,endpoint.c_str()); curl_easy_setopt(h,CURLOPT_HTTPHEADER,headers.get());
        curl_easy_setopt(h,CURLOPT_POSTFIELDS,body.c_str()); curl_easy_setopt(h,CURLOPT_POSTFIELDSIZE,long(body.size()));
        curl_easy_setopt(h,CURLOPT_WRITEFUNCTION,receive); curl_easy_setopt(h,CURLOPT_WRITEDATA,&result);
        curl_easy_setopt(h,CURLOPT_CONNECTTIMEOUT,10L); curl_easy_setopt(h,CURLOPT_TIMEOUT,120L);
        curl_easy_setopt(h,CURLOPT_NOSIGNAL,1L); curl_easy_setopt(h,CURLOPT_PROTOCOLS_STR,"http,https");
        auto status = curl_easy_perform(h); long code = 0; curl_easy_getinfo(h,CURLINFO_RESPONSE_CODE,&code);
        if (status != CURLE_OK) throw std::runtime_error(std::string("backend transport: ")+curl_easy_strerror(status));
        if (code != 200) throw std::runtime_error("backend HTTP " + std::to_string(code));
        return binary_logit(Json::parse(result));
    }
};
#endif
static Json serialize(const std::vector<jax::Answer>& answers, const std::vector<jax::Question>& questions, const std::string& model_id="jax-0.1") {
    auto root = Json::object(), all = Json::object(); root.set("model",Json::str(model_id));
    for (size_t i = 0; i < answers.size(); ++i) {
        const auto& a = answers[i]; auto o = Json::object();
        if (a.kind == jax::Kind::noul) { o.set("type",Json::str("noul")); o.set("noul",Json::num(a.noul)); }
        else {
            bool choice = a.kind == jax::Kind::choice;
            o.set("type",Json::str(choice ? "choice" : "score"));
            o.set("confidence",Json::num(a.confidence));
            auto p = Json::object(), legend = Json::object();
            for (size_t k = 0; k < a.probabilities.size(); ++k) {
                std::string name = choice ? questions[i].criteria[k].name : std::to_string(k);
                p.set(name,Json::num(a.probabilities[k]));
                if (!choice) legend.set(name,Json::str(questions[i].criteria[k].description));
            }
            o.set("probabilities",p);
            if (choice) o.set("choice",Json::str(a.choice));
            else { o.set("score",Json::num(a.score)); o.set("legend",legend); }
        }
        all.set(a.id,o);
    }
    root.set("answers",all);
    if (env("JAX_TRACE", "0") == "1") {
        auto trace = Json::object();
        for (size_t i = 0; i < answers.size(); ++i) {
            auto row = Json::object(), logits = Json::array(), labels = Json::array();
            for (double z : answers[i].raw_logits) logits.push(Json::num(z));
            if (answers[i].kind == jax::Kind::noul) { labels.push(Json::str("false")); labels.push(Json::str("true")); }
            else for (size_t k = 0; k < questions[i].criteria.size(); ++k)
                labels.push(Json::str(answers[i].kind == jax::Kind::choice ? questions[i].criteria[k].name : std::to_string(k)));
            row.set("logits",logits); row.set("labels",labels); trace.set(answers[i].id,row);
        }
        root.set("raw_logits",trace);
    }
    return root;
}

#ifdef JAX_WITH_LLAMA
// Private JSONL worker protocol. Exactly one loaded backbone, no model reload per request.
static int stream_abel() {
    if (env("JAX_LLAMA_VERBOSE","0")!="1") llama_log_set([](ggml_log_level level,const char* text,void*) {
        if (level==GGML_LOG_LEVEL_WARN || level==GGML_LOG_LEVEL_ERROR) std::fputs(text,stderr);
    },nullptr);
    auto setting=[](const char* key,const char* fallback) {
        auto s=env(key,fallback);size_t end=0;auto n=std::stoul(s,&end);
        if (end!=s.size() || n>32768) throw std::runtime_error("invalid runtime setting");
        return unsigned(n);
    };
    auto head_path=env("JAX_HEAD","");
    if (head_path.empty()) throw std::runtime_error("stream requires a verified Abel head");
    auto head=jax::AbelHead::load(Json::parse(read_file(head_path)));
    jax::LlamaBackend backend(env("JAX_MODEL_PATH",""),setting("JAX_CTX","2048"),setting("JAX_THREADS","2"));
    std::cout << "{\"ready\":true,\"protocol\":\"jax.worker.v1\"}" << std::endl;
    while (true) {
        std::string line;char ch;bool overflow=false;
        while (std::cin.get(ch) && ch!='\n') {
            if (line.size()<8*1024*1024) line.push_back(ch);else overflow=true;
        }
        if (!std::cin && line.empty()) break;
        auto response=Json::object();
        (void)backend.take_input_tokens();
        bool validated=false;
        try {
            if (overflow) throw std::invalid_argument("request too large");
            auto input=Json::parse(line);
            auto qs=jax::parse_questions(input);
            auto state=input.at("state").dump();
            validated=true;
            auto answers=jax::decide(state,qs,[&](const auto& j) {
                auto f=backend.features(j);return head.logit(f.base_logit,f.hidden);
            },1,head.temperature);
            auto result=serialize(answers,qs,"Jax-1-Abel");
            auto usage=Json::object();
            usage.set("input_tokens",Json::num(double(backend.take_input_tokens())));
            usage.set("output_tokens",Json::num(0));
            result.set("usage",usage);
            response.set("result",result);
        } catch (const std::exception& e) {
            auto error=Json::object();error.set("message",Json::str(e.what()));
            bool invalid=!validated || std::string(e.what()).rfind("judgment exceeds JAX_CTX",0)==0;
            error.set("code",Json::str(invalid?"invalid_request":"inference_failed"));
            response.set("error",error);
        }
        std::cout << response.dump() << std::endl;
        if (!std::cout) return 1;
    }
    return 0;
}
#endif
int main(int argc, char** argv) {
    try {
#ifdef JAX_WITH_LLAMA
        if (argc==2 && std::string(argv[1])=="stream") return stream_abel();
#endif
        if (argc < 3 || (std::string(argv[1]) != "infer" && std::string(argv[1]) != "calibrate" && std::string(argv[1]) != "evaluate")) {
            std::cerr << "Usage: jax infer request.json [temperature.json]\n       jax calibrate labeled-logits.json\n       jax evaluate heldout-logits.json [temperature.json]\n"; return 2;
        }
        if (argc > 4) throw std::runtime_error("too many arguments");
        auto input = Json::parse(read_file(argv[2]));
        if (std::string(argv[1]) != "infer") {
            bool fitting = std::string(argv[1]) == "calibrate";
            if ((fitting && argc != 3) || !input.is(json_type_array)) throw std::runtime_error("calibrate expects one JSON array file");
            std::vector<jax::CalibrationRow> rows;
            for (size_t i = 0; i < input.size(); ++i) {
                auto row = input.at(i), logits = row.at("logits"); jax::CalibrationRow r{};
                if (!logits.is(json_type_array)) throw std::runtime_error("logits must be an array");
                for (size_t k = 0; k < logits.size(); ++k) r.logits.push_back(logits.at(k).number());
                double label = row.at("label").number();
                if (label < 0 || label >= r.logits.size() || std::floor(label) != label) throw std::runtime_error("invalid calibration label");
                r.label = size_t(label); rows.push_back(std::move(r));
            }
            (void)jax::nll(rows,1);
            double t = fitting ? jax::fit_temperature(rows) : (argc == 4 ? Json::parse(read_file(argv[3])).at("temperature").number() : 1);
            auto result = Json::object();
            if (!fitting) {
                double brier = 0, correct = 0;
                double confidence_sum[10]{}, correct_sum[10]{};
                for (const auto& row : rows) {
                    auto p = jax::softmax(row.logits,t);
                    size_t best = std::max_element(p.begin(),p.end())-p.begin();
                    double hit = best == row.label;
                    correct += hit;
                    size_t bin = std::min(size_t(p[best]*10),size_t(9));
                    confidence_sum[bin] += p[best]; correct_sum[bin] += hit;
                    for (size_t k = 0; k < p.size(); ++k) brier += std::pow(p[k] - double(k == row.label),2);
                }
                double ece = 0;
                for (size_t k = 0; k < 10; ++k) ece += std::abs(confidence_sum[k]-correct_sum[k]);
                result.set("nll",Json::num(jax::nll(rows,t)));
                result.set("brier",Json::num(brier/rows.size()));
                result.set("accuracy",Json::num(correct/rows.size()));
                result.set("ece_10",Json::num(ece/rows.size()));
                std::cout << result.dump() << '\n'; return 0;
            }
            result.set("temperature",Json::num(t)); result.set("fit_nll_before",Json::num(jax::nll(rows,1)));
            result.set("fit_nll_after",Json::num(jax::nll(rows,t))); std::cout << result.dump() << '\n'; return 0;
        }
        auto qs = jax::parse_questions(input); std::string state = input.at("state").dump();
        double temperature = argc == 4 ? Json::parse(read_file(argv[3])).at("temperature").number() : 1;
        (void)jax::softmax({0},temperature);
        auto worker_text = env("JAX_WORKERS","4"); size_t used = 0;
        unsigned long workers = std::stoul(worker_text,&used);
        if (used != worker_text.size() || workers < 1 || workers > 64) throw std::runtime_error("JAX_WORKERS must be 1..64");
        std::vector<jax::Answer> answers;
        std::string model_id="jax-0.1";
#ifdef JAX_WITH_LLAMA
        if (env("JAX_LLAMA_VERBOSE","0")!="1") {
            llama_log_set([](ggml_log_level level,const char* text,void*) {
                if (level==GGML_LOG_LEVEL_WARN || level==GGML_LOG_LEVEL_ERROR) std::fputs(text,stderr);
            },nullptr);
        }
        auto setting = [](const char* name, const char* fallback) {
            auto value=env(name,fallback); size_t consumed=0; auto number=std::stoul(value,&consumed);
            if (consumed!=value.size() || number>32768) throw std::runtime_error(std::string("invalid ")+name);
            return unsigned(number);
        };
        jax::LlamaBackend backend(env("JAX_MODEL_PATH","models/Qwen3.5-0.8B-Q8_0.gguf"),setting("JAX_CTX","2048"),setting("JAX_THREADS","2"));
        std::string head_path=env("JAX_HEAD","");
        if (!head_path.empty()) {
            if (argc==4) throw std::runtime_error("Abel supplies checkpoint temperature; do not pass a second calibration file");
            auto head=jax::AbelHead::load(Json::parse(read_file(head_path)));
            model_id="Jax-1-Abel";
            answers=jax::decide(state,qs,[&](const auto& j){
                auto f=backend.features(j);return head.logit(f.base_logit,f.hidden);
            },1,head.temperature);
        } else answers = jax::decide(state,qs,[&](const auto& j){return backend(j);},1,temperature);
#else
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) throw std::runtime_error("curl global init failed");
        struct Cleanup { ~Cleanup(){curl_global_cleanup();} } cleanup;
        HttpBackend backend;
        answers = jax::decide(state,qs,[&](const auto& j){return backend(j);},workers,temperature);
#endif
        std::cout << serialize(answers,qs,model_id).dump() << '\n';
    } catch (const std::exception& e) { std::cerr << "jax: " << e.what() << '\n'; return 1; }
}
