// Mathematical expressions in the input deck (taken over from GEM-PIC-RZ src/expr.*, d76ac6f).
// An expression is compiled once into a small stack program and evaluated (expr_eval, on the host or in
// kernels) for given values of its variables, e.g.  density_function = x < 20 um ? x / (20 um) : 1
//
// Syntax
//   numbers     2, 1.5e-3, and numbers with a unit: 10 um, 5 fs, 1e19 cm^-3 (converted to code units
//               by the caller's unit function)
//   variables   given by the context (x, y, z), code units
//   constants   pi
//   operators   + - * / ^ (also **), unary -, comparisons < <= > >= == != (1 or 0), && || !, c ? a : b
//   functions   sin cos tan asin acos atan atan2 sinh cosh tanh exp log log10 sqrt abs floor ceil min
//               max pow sign step (1 for x >= 0), if(cond, a, b), gauss(x, s) = exp(-x^2/(2 s^2)),
//               rect(x, w) = 1 for |x| <= w/2
// Errors are thrown as std::invalid_argument with the position in the text.
#pragma once
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>
#include "portable.hpp"

// factor from one unit (e.g. "um", "cm^-3") to code units; false if the name is no unit
using UnitFn = std::function<bool(const std::string& unit, double& factor)>;

class Expr {
public:
    Expr() = default;
    Expr(const std::string& text, const std::vector<std::string>& vars, const UnitFn& units);
    double operator()(const double* v) const;
    double operator()(std::initializer_list<double> v) const { return (*this)(v.begin()); }
    bool empty() const { return code.empty(); }
    const std::string& text() const { return src; }

    enum Op : int { PUSH, VAR, NEG, NOT, ADD, SUB, MUL, DIV, POW, LT, LE, GT, GE, EQ, NE, AND, OR, SEL, F1, F2 };
    struct Ins { Op op; int arg; double val; };
    static constexpr int MAX_STACK = 64;
    const std::vector<Ins>& program() const { return code; }   // for a device copy (expr_eval)
    int max_stack() const { return maxStack; }

private:
    std::string src;
    std::vector<Ins> code;
    int maxStack = 0;
};

// functions of the F1 / F2 instructions
namespace exprfn {
enum F1 { SIN, COS, TAN, ASIN, ACOS, ATAN, SINH, COSH, TANH, EXP, LOG, LOG10, SQRT, ABS, FLOOR, CEIL, SIGN, STEP };
enum F2 { ATAN2, MIN, MAX, POWF, GAUSS, RECT };

HD inline double f1(int k, double a) {
    switch (k) {
        case SIN: return sin(a);
        case COS: return cos(a);
        case TAN: return tan(a);
        case ASIN: return asin(a);
        case ACOS: return acos(a);
        case ATAN: return atan(a);
        case SINH: return sinh(a);
        case COSH: return cosh(a);
        case TANH: return tanh(a);
        case EXP: return exp(a);
        case LOG: return log(a);
        case LOG10: return log10(a);
        case SQRT: return sqrt(a);
        case ABS: return fabs(a);
        case FLOOR: return floor(a);
        case CEIL: return ceil(a);
        case SIGN: return (a > 0) - (a < 0);
        case STEP: return a >= 0 ? 1.0 : 0.0;
    }
    return 0.0;
}
HD inline double f2(int k, double a, double b) {
    switch (k) {
        case ATAN2: return atan2(a, b);
        case MIN: return fmin(a, b);
        case MAX: return fmax(a, b);
        case POWF: return pow(a, b);
        case GAUSS: return exp(-0.5 * a * a / (b * b));
        case RECT: return fabs(a) <= 0.5 * b ? 1.0 : 0.0;
    }
    return 0.0;
}

}  // namespace exprfn

// The program code[0 .. n) for the variable values v, on the host (Expr::operator()) and in kernels
// (on a device copy of program()). The caller checks max_stack() <= MAX_STACK.
HD inline double expr_eval(const Expr::Ins* code, int n, const double* v) {
    double st[Expr::MAX_STACK];
    double* s = st;  // s points past the top
    for (int q = 0; q < n; ++q) {
        const Expr::Ins& i = code[q];
        switch (i.op) {
            case Expr::PUSH: *s++ = i.val; break;
            case Expr::VAR: *s++ = v[i.arg]; break;
            case Expr::NEG: s[-1] = -s[-1]; break;
            case Expr::NOT: s[-1] = (s[-1] == 0.0) ? 1.0 : 0.0; break;
            case Expr::ADD: --s; s[-1] += s[0]; break;
            case Expr::SUB: --s; s[-1] -= s[0]; break;
            case Expr::MUL: --s; s[-1] *= s[0]; break;
            case Expr::DIV: --s; s[-1] /= s[0]; break;
            case Expr::POW: --s; s[-1] = pow(s[-1], s[0]); break;
            case Expr::LT: --s; s[-1] = s[-1] < s[0]; break;
            case Expr::LE: --s; s[-1] = s[-1] <= s[0]; break;
            case Expr::GT: --s; s[-1] = s[-1] > s[0]; break;
            case Expr::GE: --s; s[-1] = s[-1] >= s[0]; break;
            case Expr::EQ: --s; s[-1] = s[-1] == s[0]; break;
            case Expr::NE: --s; s[-1] = s[-1] != s[0]; break;
            case Expr::AND: --s; s[-1] = (s[-1] != 0.0 && s[0] != 0.0); break;
            case Expr::OR: --s; s[-1] = (s[-1] != 0.0 || s[0] != 0.0); break;
            case Expr::SEL: s -= 2; s[-1] = (s[-1] != 0.0) ? s[0] : s[1]; break;
            case Expr::F1: s[-1] = exprfn::f1(i.arg, s[-1]); break;
            case Expr::F2: --s; s[-1] = exprfn::f2(i.arg, s[-1], s[0]); break;
        }
    }
    return st[0];
}
