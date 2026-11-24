// Deck expressions (src/expr.*) and density profiles (density_function, docs/specs/density_profile.md).
#include <cmath>
#include <cstdio>
#include "common.hpp"
#include "deck.hpp"
#include "expr.hpp"
#include "sim.hpp"

// P1. Expressions: precedence, power, comparisons, functions, units; syntax errors are reported.
static void profile_expr() {
    const double k0 = 2 * M_PI / 0.8e-6;
    const UnitFn u = [k0](const std::string& n, double& f) { return unit_to_code(n, k0, f); };
    struct Case { const char* text; double x, want; };
    const double nc = 8.8541878128e-12 * 9.1093837015e-31 * std::pow(299792458.0 * k0, 2) / std::pow(1.602176634e-19, 2);
    const Case cases[] = {
        {"1 + 2*3^2", 0, 19}, {"-2^2", 0, -4}, {"2**3 - 1", 0, 7}, {"x < 1 ? 5 : 7", 0.5, 5}, {"x < 1 ? 5 : 7", 2, 7},
        {"if(x >= 2 && x < 3, 1, 0)", 2.5, 1}, {"gauss(x, 1)", 1, std::exp(-0.5)}, {"rect(x, 1)", 0.6, 0},
        {"max(1, x) + min(3, 4)", 2, 5}, {"10 um", 0, 10e-6 * k0}, {"5 fs", 0, 5e-15 * 299792458.0 * k0},
        {"1e19 cm^-3", 0, 1e25 / nc}, {"50%", 0, 0.5}, {"x / (2 um)", 1, 1 / (2e-6 * k0)}, {"step(x) * sqrt(x)", 4, 2},
        {"2 * pi", 0, 2 * M_PI}};
    double worst = 0;
    for (const Case& c : cases) {
        const double v = Expr(c.text, {"x"}, u)({c.x});
        worst = std::max(worst, std::abs(v - c.want) / std::max(1.0, std::abs(c.want)));
    }
    int caught = 0;
    for (const char* bad : {"1 +", "foo", "sin(1, 2)", "(1", "3 furlongs"})
        try { Expr(bad, {"x"}, u); } catch (const std::invalid_argument&) { ++caught; }
    check("expressions: values, units, errors", worst < 1e-14 && caught == 5,
          fmt("max rel err %.1e, %.0f of 5 errors caught", worst, caught));
}

// P2. Loading with a density function: a linear ramp is integrated exactly by the regular lattice
//     (midpoints of the sub-cells), and a step keeps exactly the lattice points with f > 0.
static void profile_load(Comm& comm) {
    for (int step : {0, 1}) {
        Config c;
        c.N[0] = 16; c.N[1] = 4; c.N[2] = 4;
        c.h[0] = 0.25; c.h[1] = 0.5; c.h[2] = 0.5;
        c.periodic[0] = false;
        SpeciesConfig e;
        e.name = "e"; e.q = -1; e.m = 1; e.density = 0.02;
        e.ppc[0] = 2; e.ppc[1] = 2;
        for (int a = 0; a < 3; ++a) { e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
        e.density_fn = Expr(step ? "x > 1.3 ? 1 : 0" : "0.25 * x", {"x", "y", "z"}, nullptr);
        c.species.push_back(e);
        c.check();
        Simulation sim(c, comm);
        sim.init();
        const auto w = sim.species[0].get(Species::W);
        double sw = 0;
        for (double q : w) sw += q;
        comm.allreduce_sum(&sw, 1);
        double n = double(w.size());
        comm.allreduce_sum(&n, 1);
        const double L = c.N[0] * c.h[0], A = c.N[1] * c.h[1] * c.N[2] * c.h[2];
        if (!step) {
            const double want = e.density * A * 0.25 * L * L / 2;
            check("density_function: linear ramp loaded exactly", std::abs(sw - want) < 1e-12 * want,
                  fmt("sum w %.10g, exact %.10g", sw, want));
        } else {
            long cnt = 0;                                  // lattice points x = (i + (a + 1/2)/2) h with x > 1.3
            for (int i = 0; i < c.N[0]; ++i)
                for (int a = 0; a < 2; ++a) cnt += (i + (a + 0.5) / 2) * c.h[0] > 1.3;
            const double want = double(cnt) * c.N[1] * c.N[2] * 2;
            check("density_function: step keeps the points with f > 0", n == want, fmt("%.0f particles, want %.0f", n, want));
        }
    }
}

void test_profile(Comm& comm) {
    profile_expr();
    profile_load(comm);
}
