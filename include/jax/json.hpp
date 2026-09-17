#pragma once
#include <json-c/json.h>
#include <cctype>
#include <cmath>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>
namespace jax {
class Json {
    std::shared_ptr<json_object> p;
public:
    explicit Json(json_object* v = nullptr) : p(v, json_object_put) {}
    static Json object() { return Json(json_object_new_object()); }
    static Json array() { return Json(json_object_new_array()); }
    static Json str(const std::string& v) { return Json(json_object_new_string_len(v.data(), int(v.size()))); }
    static Json num(double v) { return Json(json_object_new_double(v)); }
    static Json boolean(bool v) { return Json(json_object_new_boolean(v)); }
    static Json parse(const std::string& s) {
        if (s.size() > 8*1024*1024) throw std::runtime_error("JSON exceeds 8 MiB");
        std::unique_ptr<json_tokener, decltype(&json_tokener_free)> tok(json_tokener_new(), json_tokener_free);
        json_tokener_set_flags(tok.get(), JSON_TOKENER_STRICT | JSON_TOKENER_VALIDATE_UTF8);
        std::string input = s + " ";
        Json result(json_tokener_parse_ex(tok.get(), input.c_str(), int(input.size())));
        if (json_tokener_get_error(tok.get()) != json_tokener_success) throw std::runtime_error("invalid JSON");
        for (size_t i = json_tokener_get_parse_end(tok.get()); i < s.size(); ++i)
            if (!std::isspace(static_cast<unsigned char>(s[i]))) throw std::runtime_error("trailing JSON content");
        return result;
    }
    bool is(json_type t) const { return json_object_is_type(p.get(), t); }
    bool has(const std::string& key) const { json_object* v; return is(json_type_object) && json_object_object_get_ex(p.get(), key.c_str(), &v); }
    Json at(const std::string& key) const {
        json_object* v;
        if (!is(json_type_object) || !json_object_object_get_ex(p.get(), key.c_str(), &v)) throw std::runtime_error("missing field: " + key);
        return Json(json_object_get(v));
    }
    Json at(size_t i) const {
        if (!is(json_type_array) || i >= size()) throw std::runtime_error("array index out of range");
        return Json(json_object_get(json_object_array_get_idx(p.get(), i)));
    }
    size_t size() const {
        if (is(json_type_array)) return json_object_array_length(p.get());
        if (is(json_type_object)) return json_object_object_length(p.get());
        throw std::runtime_error("expected array/object");
    }
    std::vector<std::string> keys() const {
        if (!is(json_type_object)) throw std::runtime_error("expected object");
        std::vector<std::string> out;
        json_object_object_foreach(p.get(), key, val) { (void)val; out.emplace_back(key); }
        return out;
    }
    std::string string() const {
        if (!is(json_type_string)) throw std::runtime_error("expected string");
        return std::string(json_object_get_string(p.get()), json_object_get_string_len(p.get()));
    }
    double number() const {
        if (!is(json_type_double) && !is(json_type_int)) throw std::runtime_error("expected number");
        double v = json_object_get_double(p.get());
        if (!std::isfinite(v)) throw std::runtime_error("nonfinite number");
        return v;
    }
    void set(const std::string& key, const Json& v) {
        if (!is(json_type_object) || key.find('\0') != std::string::npos) throw std::runtime_error("invalid object key");
        json_object_object_add(p.get(), key.c_str(), json_object_get(v.p.get()));
    }
    void push(const Json& v) { if (!is(json_type_array)) throw std::runtime_error("expected array"); json_object_array_add(p.get(), json_object_get(v.p.get())); }
    std::string dump() const { return json_object_to_json_string_ext(p.get(), JSON_C_TO_STRING_PLAIN); }
};
}
