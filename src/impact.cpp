// Collisional (impact) ionization: the tables (impact.hpp, docs/specs/collisions.md, decision 94).
#include "impact.hpp"
#include <cmath>
#include <stdexcept>

namespace {
constexpr double MC2_EV = 510998.95;

struct Bind { int Z, k; double e; };
const Bind BINDING_TABLE[] = {
#define BINDING(Z, k, e) {Z, k, e},
#include "../data/binding_energies.def"
#undef BINDING
};
// the neutral-atom binding energy of electron k of element Z [eV] (k = 0 the innermost)
double neutral(int Z, int k) {
    const int off = Z * (Z - 1) / 2;
    const Bind& b = BINDING_TABLE[off + k];
    if (b.Z != Z || b.k != k) throw std::runtime_error("data/binding_energies.def: bad order");
    return b.e;
}
// orbital k of the ion Z* [m c^2]: the neutral energies shifted so that the outermost is I(Z*) (Smilei, Perez)
double orbital(int Z, const std::vector<double>& ip, int zs, int k) {
    return (ip[size_t(zs)] - neutral(Z, Z - zs - 1) + neutral(Z, k)) / MC2_EV;
}
// RBEB of one orbital group (occupation N, binding bp = up, m c^2) at the electron kinetic energy ep; sk the cross-section,
// wk the secondary energy times sk / bp (Smilei's formulas)
void rbeb(double ep, double bp, int N, double norm, double& sk, double& wk) {
    sk = wk = 0;
    const double e = ep / bp;
    if (!(e > 1)) return;
    const double up = bp;
    const double betae2 = 1 - 1 / ((1 + ep) * (1 + ep)), betab2 = 1 - 1 / ((1 + bp) * (1 + bp)), betau2 = 1 - 1 / ((1 + up) * (1 + up));
    const double s0 = norm * N / (bp * (betae2 + betab2 + betau2));
    double ep2 = 1 / (1 + ep * 0.5);
    ep2 *= ep2;
    const double A1 = (1 + 2 * ep) / (1 + e) * ep2, A2 = (e - 1) * bp * bp * 0.5 * ep2;
    const double A3 = std::log(betae2 / (1 - betae2)) - betae2 - std::log(2 * bp);
    sk = s0 * (0.5 * A3 * (1 - 1 / (e * e)) + 1 - 1 / e + A2 - A1 * std::log(e));
    wk = s0 * (0.5 * A3 * (e - 1) * (e - 1) / e / (e + 1) + 2 * std::log(0.5 * (e + 1)) - std::log(e) + 0.25 * A2 * (e - 1) -
               A1 * (e * std::log(e) - (e + 1) * std::log(0.5 * (e + 1))));
}
// the sums over the orbitals of the ion Z* at ep: cs, te (mean secondary energy), le (mean lost energy)
void sums(int Z, const std::vector<double>& ip, double norm, int zs, double ep, double& cs, double& te, double& le) {
    cs = te = le = 0;
    int N = 1;
    for (int k = 0; k < Z - zs; ++k) {
        const double bp = orbital(Z, ip, zs, k);
        if (k < Z - zs - 1 && bp == orbital(Z, ip, zs, k + 1)) { ++N; continue; }
        double sk, wk;
        rbeb(ep, bp, N, norm, sk, wk);
        cs += sk; te += wk * bp; le += (wk + sk) * bp;
        N = 1;
    }
    if (cs > 0) { te /= cs; le /= cs; }
}
}  // namespace

bool impact_element_ok(int Z) { return Z >= 1 && Z <= 100; }

void impact_tables(int Z, const std::vector<double>& ip_eV, double kre, std::vector<double>& cs, std::vector<double>& te,
                   std::vector<double>& le) {
    if (!impact_element_ok(Z) || int(ip_eV.size()) < Z) throw std::runtime_error("impact ionization: elements Z = 1 .. 100");
    const double norm = 0.5 * kre;                    // r_e omega / 2 c (Smilei)
    cs.assign(size_t(Z) * IMPACT_N, 0.0); te = cs; le = cs;
    for (int zs = 0; zs < Z; ++zs)
        for (int i = 0; i < IMPACT_N; ++i) {
            const double ep = std::exp(double(i) / IMPACT_A2) / IMPACT_A1;
            const size_t q = size_t(zs) * IMPACT_N + size_t(i);
            sums(Z, ip_eV, norm, zs, ep, cs[q], te[q], le[q]);
        }
}

double impact_cross_section(int Z, const std::vector<double>& ip_eV, double kre, int zstar, double ek) {
    double cs, te, le;
    sums(Z, ip_eV, 0.5 * kre, zstar, ek, cs, te, le);
    return cs;
}
