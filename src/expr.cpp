// Recursive-descent compiler and stack evaluator (see expr.hpp).
#include "expr.hpp"
#include <algorithm>
#include <map>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace {

using namespace exprfn;

const std::map<std::string, int> FUN1 = {
    {"sin", SIN},   {"cos", COS},   {"tan", TAN},     {"asin", ASIN}, {"acos", ACOS},   {"atan", ATAN},
    {"sinh", SINH}, {"cosh", COSH}, {"tanh", TANH},   {"exp", EXP},   {"log", LOG},     {"ln", LOG},
    {"log10", LOG10}, {"sqrt", SQRT}, {"abs", ABS},   {"floor", FLOOR}, {"ceil", CEIL}, {"sign", SIGN},
    {"step", STEP}};
const std::map<std::string, int> FUN2 = {{"atan2", ATAN2}, {"min", MIN}, {"max", MAX},
                                         {"pow", POWF}, {"gauss", GAUSS}, {"rect", RECT}};

class Compiler {
public:
    Compiler(const std::string& s, const std::vector<std::string>& vars, const UnitFn& u, std::vector<Expr::Ins>& code)
            : s_(s), vars_(vars), u_(u), code_(code) {}

    void run() {
        ternary();
        skip();
        if (p_ < s_.size()) fail("unexpected '" + std::string(1, s_[p_]) + "'");
    }

private:
    const std::string& s_;
    const std::vector<std::string>& vars_;
    const UnitFn& u_;
    std::vector<Expr::Ins>& code_;
    size_t p_ = 0;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::invalid_argument("in \"" + s_ + "\" at position " + std::to_string(p_ + 1) + ": " + msg);
    }
    void skip() {
        while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_;
    }
    bool eat(const char* t) {
        skip();
        const size_t n = std::char_traits<char>::length(t);
        if (s_.compare(p_, n, t) == 0) { p_ += n; return true; }
        return false;
    }
    void emit(Expr::Op op, int arg = 0, double val = 0.0) { code_.push_back({op, arg, val}); }

    void ternary() {
        orExpr();
        if (eat("?")) {
            ternary();
            if (!eat(":")) fail("':' expected");
            ternary();
            emit(Expr::SEL);
        }
    }
    void orExpr() {
        andExpr();
        while (eat("||")) { andExpr(); emit(Expr::OR); }
    }
    void andExpr() {
        cmpExpr();
        while (eat("&&")) { cmpExpr(); emit(Expr::AND); }
    }
    void cmpExpr() {
        addExpr();
        for (;;) {
            if (eat("<=")) { addExpr(); emit(Expr::LE); }
            else if (eat(">=")) { addExpr(); emit(Expr::GE); }
            else if (eat("==")) { addExpr(); emit(Expr::EQ); }
            else if (eat("!=")) { addExpr(); emit(Expr::NE); }
            else if (eat("<")) { addExpr(); emit(Expr::LT); }
            else if (eat(">")) { addExpr(); emit(Expr::GT); }
            else break;
        }
    }
    void addExpr() {
        mulExpr();
        for (;;) {
            if (eat("+")) { mulExpr(); emit(Expr::ADD); }
            else if (eat("-")) { mulExpr(); emit(Expr::SUB); }
            else break;
        }
    }
    void mulExpr() {
        unary();
        for (;;) {
            skip();
            if (s_.compare(p_, 2, "**") == 0) break;  // power, handled below
            if (eat("*")) { unary(); emit(Expr::MUL); }
            else if (eat("/")) { unary(); emit(Expr::DIV); }
            else break;
        }
    }
    void unary() {
        if (eat("-")) { unary(); emit(Expr::NEG); return; }
        if (eat("+")) { unary(); return; }
        if (eat("!")) { unary(); emit(Expr::NOT); return; }
        power();
    }
    void power() {  // right-associative, binds tighter than unary minus on its left: -x^2 = -(x^2)
        primary();
        if (eat("^") || eat("**")) { unary(); emit(Expr::POW); }
    }
    // a unit name after a number: letters (and digits) with an optional exponent ^-3, or %
    bool unitAfterNumber(double& factor) {
        if (!u_) return false;
        skip();
        size_t q = p_;
        if (q < s_.size() && s_[q] == '%') { factor = 0.01; p_ = q + 1; return true; }
        if (q >= s_.size() || !std::isalpha(static_cast<unsigned char>(s_[q]))) return false;
        while (q < s_.size() && std::isalnum(static_cast<unsigned char>(s_[q]))) ++q;
        std::string base = s_.substr(p_, q - p_);
        // longest match first: with an exponent (cm^-3, k0^-1), then without
        size_t q2 = q;
        if (q2 < s_.size() && s_[q2] == '^') {
            size_t r = q2 + 1;
            if (r < s_.size() && (s_[r] == '-' || s_[r] == '+')) ++r;
            const size_t d0 = r;
            while (r < s_.size() && std::isdigit(static_cast<unsigned char>(s_[r]))) ++r;
            if (r > d0 && u_(s_.substr(p_, r - p_), factor)) {
                p_ = r;
                return true;
            }
        }
        if (u_(base, factor)) {
            p_ = q;
            return true;
        }
        return false;
    }
    void primary() {
        skip();
        if (p_ >= s_.size()) fail("unexpected end of the expression");
        const char c = s_[p_];
        if (c == '(') {
            ++p_;
            ternary();
            if (!eat(")")) fail("')' expected");
            return;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '.') {
            const char* b = s_.c_str() + p_;
            char* end = nullptr;
            const double x = std::strtod(b, &end);
            if (end == b) fail("number expected");
            p_ += size_t(end - b);
            double f = 1.0;
            const size_t save = p_;
            if (!unitAfterNumber(f)) p_ = save;
            emit(Expr::PUSH, 0, x * f);
            return;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t q = p_;
            while (q < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[q])) || s_[q] == '_')) ++q;
            std::string name = s_.substr(p_, q - p_);
            p_ = q;
            if (eat("(")) {  // function
                if (name == "if") {
                    ternary();
                    if (!eat(",")) fail("if(cond, a, b): ',' expected");
                    ternary();
                    if (!eat(",")) fail("if(cond, a, b): ',' expected");
                    ternary();
                    if (!eat(")")) fail("')' expected");
                    emit(Expr::SEL);
                    return;
                }
                auto f1 = FUN1.find(name);
                auto f2 = FUN2.find(name);
                if (f1 == FUN1.end() && f2 == FUN2.end()) fail("unknown function '" + name + "'");
                ternary();
                if (f2 != FUN2.end()) {
                    if (!eat(",")) fail(name + "(a, b): ',' expected");
                    ternary();
                    if (!eat(")")) fail("')' expected");
                    emit(Expr::F2, f2->second);
                } else {
                    if (!eat(")")) fail("')' expected (" + name + " takes one argument)");
                    emit(Expr::F1, f1->second);
                }
                return;
            }
            for (auto& ch : name) ch = char(std::tolower(static_cast<unsigned char>(ch)));  // deck keys are lower case
            for (size_t k = 0; k < vars_.size(); ++k)
                if (vars_[k] == name) { emit(Expr::VAR, int(k)); return; }
            if (name == "pi") { emit(Expr::PUSH, 0, 3.14159265358979323846); return; }
            std::string list;
            for (const auto& v : vars_) list += (list.empty() ? "" : ", ") + v;
            fail("unknown name '" + name + "' (variables here: " + (list.empty() ? "none" : list) +
                      "; constants: pi)");
        }
        fail("unexpected '" + std::string(1, c) + "'");
    }
};

}  // namespace

Expr::Expr(const std::string& text, const std::vector<std::string>& vars, const UnitFn& units) : src(text) {
    Compiler(src, vars, units, code).run();
    int depth = 0;
    for (const auto& i : code) {
        switch (i.op) {
            case PUSH: case VAR: ++depth; break;
            case NEG: case NOT: case F1: break;
            case SEL: depth -= 2; break;
            default: --depth; break;
        }
        maxStack = std::max(maxStack, depth);
    }
}

double Expr::operator()(const double* v) const {
    if (maxStack > MAX_STACK) throw std::invalid_argument("expression too deeply nested: " + src);
    return expr_eval(code.data(), int(code.size()), v);
}
