// Time series of energy and Gauss law for the paper (tasks 5 and 8): box iRIP, Cart3D and RZ, vacuum random fields
// and hot plasmas. Built against the VLPL sources (src/*.cpp except main.cpp) by build_series.sh (bench/irip_paper).
//   series <case> <out.txt> [deck overrides]   case = cart_random | cart_hot | rz_random | rz_hot <deck>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "comm.hpp"
#include "deck.hpp"
#include "fields.hpp"
#include "grid.hpp"
#include "particles.hpp"
#include "rz.hpp"
#include "sim.hpp"
#include "species.hpp"

static double maxabs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double e = 0; for (size_t q = 0; q < a.size(); ++q) e = std::max(e, std::abs(a[q] - b[q])); return e;
}
static double maxabs(const std::vector<double>& a) { double e = 0; for (double v : a) e = std::max(e, std::abs(v)); return e; }

static void randomize3(Fields& f, const Domain& d, unsigned seed) {
    std::mt19937_64 rng(seed); std::normal_distribution<double> g;
    for (Field* F : {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz}) {
        std::vector<double> h = F->host();
        for (int k = 0; k < d.n[2]; ++k) for (int j = 0; j < d.n[1]; ++j) for (int i = 0; i < d.n[0]; ++i) h[F->idx(i, j, k)] = g(rng);
        F->upload(h);
    }
}
static void randomize_rz(FieldsRZ& f, const Domain& d, unsigned seed) {   // as tests/test_rz.cpp randomize()
    std::mt19937_64 rng(seed); std::normal_distribution<double> g;
    for (Field* F : {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp}) {
        std::vector<double> h = F->host();
        for (int k = 0; k < d.n[2]; ++k) for (int j = 0; j < d.n[1]; ++j) for (int i = 0; i < d.n[0]; ++i) {
            double v = g(rng);
            if (j == 0 && (F == &f.ep || F == &f.br || (F == &f.ez && k > 0))) v = 0;
            h[F->idx(i, j, k)] = v;
        }
        F->upload(h);
    }
}
// per-mode max |a - b| over the slots k (mode (k + 1) / 2) and the radial rows j < jmax
static std::vector<double> per_mode(const Field& F, const std::vector<double>& a, const std::vector<double>& b,
                                    const Domain& d, int M, int jmax) {
    std::vector<double> e(M, 0.0);
    for (int k = 0; k < d.n[2]; ++k) for (int j = 0; j < std::min(jmax, d.n[1]); ++j) for (int i = 0; i < d.n[0]; ++i) {
        const size_t q = F.idx(i, j, k); e[(k + 1) / 2] = std::max(e[(k + 1) / 2], std::abs(a[q] - b[q]));
    }
    return e;
}

int main(int argc, char** argv) {
    Comm comm; comm.init(&argc, &argv);
    const std::string cs = argv[1];
    FILE* o = std::fopen(argv[2], "w");
    if (cs == "cart_random") {                                         // as tests/test_irip.cpp I3, h_x/h_perp = 10
        Domain d; d.periodic[0] = true; d.N[0] = 12; d.N[1] = 16; d.N[2] = 10; d.h[0] = 1.0; d.h[1] = d.h[2] = 0.1;
        d.ng = 3; d.solver = Solver::IRip; const int want[3] = {0, 0, 0}; comm.decompose(d, want);
        Fields f(d); randomize3(f, d, 5); comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
        Field g(d); f.gauss_residual(g); const std::vector<double> g0 = g.host(); const double gs = maxabs(g0);
        const double b0 = f.max_div_b(), w0 = f.energy();
        std::fprintf(o, "# Cart3D box iRIP, random vacuum fields, 12 x 16 x 10, h_x = c dt = 1, h_y = h_z = 0.1\n"
                        "# step  W/W0-1  max|dG|/max|G0|  (max|divB|-max|divB0|)/max|divB0|\n");
        for (int n = 0; n <= 500; ++n) {
            if (n) { f.zero_currents(); f.step(comm); }
            f.gauss_residual(g);
            std::fprintf(o, "%d %.6e %.6e %.6e\n", n, f.energy() / w0 - 1, maxabs_diff(g.host(), g0) / gs, (f.max_div_b() - b0) / b0);
        }
    } else if (cs == "cart_hot") {                                    // hot pair plasma, h_x/h_perp = 10, moves of ~3 cells
        Config c; c.N[0] = 16; c.N[1] = 24; c.N[2] = 24; c.h[0] = 1.0; c.h[1] = c.h[2] = 0.1;
        c.solver = Solver::IRip; c.t_end = 1000 * c.dt(); c.order = 1;
        SpeciesConfig e; e.name = "e"; e.q = -1; e.m = 1; e.density = 0.02; e.ppc[0] = e.ppc[1] = e.ppc[2] = 2;
        for (int a = 0; a < 3; ++a) { e.uth[a] = 0.3; e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
        SpeciesConfig p = e; p.name = "p"; p.q = 1;
        c.species = {e, p}; c.check();
        Simulation sim(c, comm); sim.init();
        auto kin = [&]() { double k = 0; for (const Species& s : sim.species) k += s.kinetic_energy(); return k; };
        const double wf0 = sim.fields->energy(), k0 = kin();
        std::fprintf(o, "# Cart3D box iRIP, hot e+e- plasma n = 0.02 (w_p dt = 0.14, Debye length 2.1 h_x), u_th = 0.3, 16 x 24 x 24, h_x = 1, h_perp = 0.1, 8 ppc\n"
                        "# step  W_field  W_kin  (W-W0)/W0  max|dGauss|  max|divB|   (W0 = %.6e)\n", wf0 + k0);
        for (int n = 0; ; ++n) {
            const double wf = sim.fields->energy(), k = kin();
            std::fprintf(o, "%d %.6e %.6e %.6e %.3e %.3e\n", n, wf, k, (wf + k) / (wf0 + k0) - 1, n ? sim.gauss_drift() : 0.0, sim.fields->max_div_b());
            if (sim.finished()) break;
            sim.step();
        }
    } else if (cs == "rz_random") {                                   // as tests/test_rz.cpp R2, dz/dr = 4, modes 0..2
        const int M = 3;
        Domain d = rz_domain(comm, 16, 20, M, 0.4, 0.1, true, Solver::IRip);
        FieldsRZ f(d, M); randomize_rz(f, d, 21); comm.exchange(d, {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp});
        Field de(d), db(d); f.div_e(de); f.div_b(db);
        const std::vector<double> e0 = de.host(), b0 = db.host();
        std::vector<double> w0(M); for (int m = 0; m < M; ++m) w0[m] = f.energy_mode(m);
        const double se = maxabs(e0), sb = maxabs(b0);
        std::fprintf(o, "# RZ box iRIP, random fields in modes 0..2, nz x nr = 16 x 20, dz = c dt = 0.4, dr = 0.1\n"
                        "# step  W_m/W_m0-1 (m=0,1,2)  max|d divE_m|/max|divE0| (m=0,1,2)  max|d divB_m|/max|divB0| (m=0,1,2)\n");
        for (int n = 0; n <= 300; ++n) {
            if (n) { f.zero_currents(); f.step(comm); }
            f.div_e(de); f.div_b(db);
            const std::vector<double> ge = per_mode(de, de.host(), e0, d, M, d.n[1] - 1), gb = per_mode(db, db.host(), b0, d, M, d.n[1] - 1);
            std::fprintf(o, "%d", n);
            for (int m = 0; m < M; ++m) std::fprintf(o, " %.6e", f.energy_mode(m) / w0[m] - 1);
            for (int m = 0; m < M; ++m) std::fprintf(o, " %.3e", ge[m] / se);
            for (int m = 0; m < M; ++m) std::fprintf(o, " %.3e", gb[m] / sb);
            std::fprintf(o, "\n");
        }
    } else if (cs == "rz_hot") {                                      // deck path, hot plasma column, the wall absorbs
        Deck dk = Deck::read(argv[3]);
        for (int a = 4; a < argc; ++a) dk.override_from(argv[a]);
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm); sim.init();
        const int M = sim.rz->modes(), order = c.order;
        Field rho(sim.dom), de(sim.dom);
        auto resid = [&]() {
            rho.fill(0); for (Species& s : sim.species) rz_rho(s, *sim.rz, rho, order);
            comm.add_guards(sim.dom, {&rho}); sim.rz->div_e(de);
            std::vector<double> a = de.host(), b = rho.host(); for (size_t q = 0; q < a.size(); ++q) a[q] -= b[q]; return a;
        };
        const std::vector<double> g0 = resid();
        rho.fill(0); rz_rho(sim.species[0], *sim.rz, rho, order); comm.add_guards(sim.dom, {&rho});
        const double rs = maxabs(rho.host());          // scale: max |rho| of the electrons alone
        auto kin = [&]() { double k = 0; for (const Species& s : sim.species) k += s.kinetic_energy(); return k; };
        std::fprintf(o, "# RZ box iRIP deck %s, %d modes, shape %d; scale max|rho_e| = %.3e\n"
                        "# step  W_m (m = 0..M-1)  W_kin  max|dGauss_m|/max|rho| (m = 0..M-1)  electrons\n", argv[3], M, order, rs);
        for (int n = 0; ; ++n) {
            const std::vector<double> g = resid();
            const std::vector<double> gm = per_mode(de, g, g0, sim.dom, M, sim.dom.n[1] - 2);
            std::fprintf(o, "%d", n);
            for (int m = 0; m < M; ++m) std::fprintf(o, " %.6e", sim.rz->energy_mode(m));
            std::fprintf(o, " %.6e", kin());
            for (int m = 0; m < M; ++m) std::fprintf(o, " %.3e", gm[m] / rs);
            std::fprintf(o, " %ld\n", long(sim.species[0].size()));
            if (sim.finished()) break;
            sim.step();
        }
    }
    std::fclose(o);
    return 0;
}
