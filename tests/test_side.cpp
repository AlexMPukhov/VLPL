// Side injection: the TFSF tube with faces y, z = const (docs/specs/side_injection.md, src/side.*).
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "common.hpp"
#include "side.hpp"
#include "sim.hpp"

namespace {
// lambda0 / 16 along x, lambda0 / 10 across; nx dx = 4 lambda0, so k_x = cos(angle) cos(elevation) is on the periodic
// lattice for cos = 0, 1/4, 1/2; CPML of 10 cells in y and z, the faces 2 cells inside them (tube of 24 cells)
const char* kSide =
    "grid {\n lambda0 = 0.8 um\n nx = 64\n ny = 48\n nz = 48\n dx = 0.05 um\n dy = 0.08 um\n dz = 0.08 um\n"
    " boundary_x = periodic\n boundary_y = pml\n boundary_z = pml\n t_end = 20 fs\n}\n"
    "pml {\n cells = 10\n}\n"
    "side {\n faces = yz\n a0 = 1\n angle = 60 deg\n polarization = z\n}\n"
    "output {\n dir = side_test_out\n}\n";

struct Leak { double out = 0, diff = 0, peak = 0; };

// max |E, B| outside the tube (scattered field, outside the CPML layers), max |main - incident| inside, max |E, B|
Leak leakage(Simulation& sim) {
    const Fields& f = *sim.fields;
    const SideTfsf& s = *f.side;
    const Domain& d = sim.dom;
    const int L = sim.cfg.pml.cells;
    Leak r;
    // with a background plasma also the fluid current (j^{t - D/2}; F times the incident current in the tube)
    const Field* a[9] = {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz, &f.jfx, &f.jfy, &f.jfz};
    for (int c = 0; c < (s.fluid() ? 9 : 6); ++c) {
        const std::vector<double> x = a[c]->host();
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    const long ig = d.cell0(0) + i, jg = d.off[1] + j, kg = d.off[2] + k;
                    const double v = x[a[c]->idx(i, j, k)];
                    if (c < 6) r.peak = std::max(r.peak, std::abs(v));
                    const double w = c < 6 ? 1.0 : s.fluid_fraction(c - 6, jg, kg);
                    if (s.total(c, jg, kg)) r.diff = std::max(r.diff, std::abs(v - w * s.incident(c, ig, jg, kg, sim.t)));
                    else if (jg >= L && jg < d.N[1] - L && kg >= L && kg < d.N[2] - L) r.out = std::max(r.out, std::abs(v));
                }
    }
    double v[3] = {r.out, r.diff, r.peak};
    for (double& x : v) x = sim.comm.allreduce_max(x);
    return {v[0], v[1], v[2]};
}
}  // namespace

// T1, T2. Vacuum, the exact discrete plane wave of the solver enters the tube through its y and z faces: at 60 and 75.5 deg
//     to x in the x-y plane (y faces) and at 30 deg elevation in the y-z plane (y and z faces and the corners of the tube),
//     both polarisations, RIP-CC and iRIP; a continuous wave (filling the tube at t = 0) and a pulse (a sum of exact
//     components with the common k_x of the periodic box). Outside the tube the field stays at round-off (no leakage),
//     inside it equals the incident wave to round-off (which also checks the discrete dispersion and polarisation of the
//     components against the solver itself), and the Gauss law and div B hold inside the tube.
void test_side(Comm& comm) {
    const char* path = "side_test_tmp.deck";
    { std::ofstream f(path); f << kSide; }
    struct Case { const char *sv, *ang, *el, *pol, *dur; };
    const Case cases[] = {
        {"rip_cc", "60 deg", "0 deg", "z", "0 fs"},          {"rip_cc", "75.52248781 deg", "0 deg", "y", "0 fs"},
        {"rip_cc", "90 deg", "30 deg", "z", "0 fs"},         {"rip_cc", "90 deg", "30 deg", "y", "0 fs"},
        {"irip", "75.52248781 deg", "0 deg", "z", "0 fs"},   {"irip", "90 deg", "30 deg", "y", "0 fs"},
        {"rip_cc", "60 deg", "0 deg", "y", "12 fs"},          {"irip", "60 deg", "0 deg", "z", "12 fs"},
        // O5 (decision 76): the closure in the one-step matrix of the waves and the tube correction of the closed Ex*, Bx*
        {"irip o5", "75.52248781 deg", "0 deg", "z", "0 fs"}, {"irip o5", "90 deg", "30 deg", "y", "0 fs"},
        {"irip o5", "60 deg", "0 deg", "z", "12 fs"},
    };
    for (const Case& cs : cases) {
        Deck dk = Deck::read(path);
        const bool o5 = std::string(cs.sv) == "irip o5";
        dk.override_from(std::string("numerics.solver=") + (o5 ? "irip" : cs.sv));
        if (o5) dk.override_from("numerics.closure=o5");
        dk.override_from(std::string("side.angle=") + cs.ang);
        dk.override_from(std::string("side.elevation=") + cs.el);
        dk.override_from(std::string("side.polarization=") + cs.pol);
        dk.override_from(std::string("side.duration=") + cs.dur);
        const bool pulse = std::string(cs.dur) != "0 fs";
        if (pulse) { dk.override_from("side.delay=15 fs"); dk.override_from("side.y_center=0.5 um"); }
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        sim.init();
        Leak m = leakage(sim);
        double gd = 0, db = 0;
        while (!sim.finished()) {
            sim.step();
            if (sim.istep % 20 == 0) {
                const Leak l = leakage(sim);
                m.out = std::max(m.out, l.out); m.diff = std::max(m.diff, l.diff); m.peak = std::max(m.peak, l.peak);
                gd = std::max(gd, sim.gauss_drift());
                db = std::max(db, comm.allreduce_max(sim.fields->max_div_b()));
            }
        }
        check(std::string("side tube, ") + cs.sv + ", " + (pulse ? "pulse" : "continuous wave") + " at " + cs.ang + ", elevation " +
                  cs.el + ", polarisation " + cs.pol + ": no leakage, total = incident",
              m.out < 1e-12 * m.peak && m.diff < 1e-12 * m.peak && m.peak > 0.5 && gd < 1e-11 && db < 1e-11,
              fmt("outside %.1e", m.out / m.peak) + fmt(", total - incident %.1e", m.diff / m.peak) + fmt(", peak %.2f", m.peak) +
                  fmt(", Gauss %.1e", gd) + fmt(", div B %.1e", db) + fmt(", %.0f components", double(sim.fields->side->comps().size())));
    }
}

// T3. The flying focus (side.wave = flying_focus): a luminal Bessel pulse (cone 20 deg, 1/e half length 1.8 um, a0 = 1
//     on the axis) of about 5000 exact discrete components in a vacuum window of 24 um moving at c, CPML in y and z,
//     RIP-CC and iRIP. The reference is the incident field of the run's components, summed over the whole box
//     (SideTfsf::add_incident unmasked). Outside the tube the field stays at round-off, inside it equals the incident
//     pulse (which also checks that the tabulated corrections in the window frame and the separable initial field agree
//     with the components); the pulse keeps its peak on the axis, the Gauss law and div B hold. The same in a plasma of
//     0.01 n_c, the cold fluid everywhere (side.fluid = all, decision 63): the fluid current too equals that of the
//     incident wave (RIP-CC: TE and TM waves of their own k_perp).
void test_side_ff(Comm& comm) {
    const char* path = "side_ff_tmp.deck";
    {
        std::ofstream f(path);
        f << "grid {\n lambda0 = 0.8 um\n nx = 480\n ny = 48\n nz = 48\n dx = 0.05 um\n dy = 0.1 um\n dz = 0.1 um\n"
             " boundary_x = open\n boundary_y = pml\n boundary_z = pml\n t_end = 12 fs\n}\n"
             "pml {\n cells = 8\n}\n"
             "window {\n start = 0 fs\n velocity = 1\n}\n"
             "side {\n wave = flying_focus\n a0 = 1\n cone_angle = 20 deg\n length_1e = 1.8 um\n x_center = 12 um\n"
             " polarization = y\n}\n"
             "output {\n dir = side_test_out\n}\n";
    }
    struct Case { const char* sv; double n; };
    for (const Case cs : {Case{"rip_cc", 0}, Case{"irip", 0}, Case{"rip_cc", 0.01}, Case{"irip", 0.01}, Case{"irip o5", 0},
                          Case{"irip o5", 0.01}}) {
        const char* sv = cs.sv;
        const bool pl = cs.n > 0;
        Deck dk = Deck::read(path);
        const bool o5 = std::string(sv) == "irip o5";
        dk.override_from(std::string("numerics.solver=") + (o5 ? "irip" : sv));
        if (o5) dk.override_from("numerics.closure=o5");
        if (pl) { dk.override_from(fmt("side.background_density=%g", cs.n)); dk.override_from("side.fluid=all"); }
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        sim.init();
        Fields ref(sim.dom);
        if (pl) { ref.jfx.resize(sim.dom); ref.jfy.resize(sim.dom); ref.jfz.resize(sim.dom); }
        const Domain& d = sim.dom;
        const int L = c.pml.cells;
        double out = 0, diff = 0, peak = 0, gd = 0, db = 0;
        long wo[4] = {0, 0, 0, 0}, wd[4] = {0, 0, 0, 0};       // where (component, global i, j, k)
        auto measure = [&]() {
            for (Field* f : {&ref.ex, &ref.ey, &ref.ez, &ref.bx, &ref.by, &ref.bz}) f->fill(0);
            if (pl) for (Field* f : {&ref.jfx, &ref.jfy, &ref.jfz}) f->fill(0);
            const FV r[6] = {ref.ex.view(), ref.ey.view(), ref.ez.view(), ref.bx.view(), ref.by.view(), ref.bz.view()};
            const FV rj[3] = {ref.jfx.view(), ref.jfy.view(), ref.jfz.view()};
            sim.fields->side->add_incident(r, sim.t, false, pl ? rj : nullptr);
            const Fields& F = *sim.fields;
            // in a plasma also the fluid current against F j of the incident wave
            const Field* a[9] = {&F.ex, &F.ey, &F.ez, &F.bx, &F.by, &F.bz, &F.jfx, &F.jfy, &F.jfz};
            const Field* b[9] = {&ref.ex, &ref.ey, &ref.ez, &ref.bx, &ref.by, &ref.bz, &ref.jfx, &ref.jfy, &ref.jfz};
            for (int q = 0; q < (pl ? 9 : 6); ++q) {
                const std::vector<double> x = a[q]->host(), y = b[q]->host();
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = 0; i < d.n[0]; ++i) {
                            const long jg = d.off[1] + j, kg = d.off[2] + k;
                            const size_t p = a[q]->idx(i, j, k);
                            if (q < 6) peak = std::max(peak, std::abs(x[p]));
                            if (sim.fields->side->total(q, jg, kg)) {
                                if (std::abs(x[p] - y[p]) > diff) { diff = std::abs(x[p] - y[p]); wd[0] = q; wd[1] = d.off[0] + i; wd[2] = jg; wd[3] = kg; }
                            } else if (jg >= L && jg < d.N[1] - L && kg >= L && kg < d.N[2] - L && std::abs(x[p]) > out) {
                                out = std::abs(x[p]); wo[0] = q; wo[1] = d.off[0] + i; wo[2] = jg; wo[3] = kg;
                            }
                        }
            }
        };
        measure();
        while (!sim.finished()) {
            sim.step();
            if (sim.istep % 24 == 0) {
                measure();
                gd = std::max(gd, sim.gauss_drift());
                db = std::max(db, comm.allreduce_max(sim.fields->max_div_b()));
            }
        }
        double v[3] = {out, diff, peak};
        for (double& x : v) x = comm.allreduce_max(x);
        check(std::string("side tube, ") + sv + ", flying focus in the moving window (" +
                  (pl ? fmt("plasma %g n_c, cold fluid", cs.n) : std::string("vacuum")) + "): no leakage, total = incident",
              v[0] < 1e-12 * v[2] && v[1] < 1e-12 * v[2] && v[2] > 0.8 && gd < 1e-11 && db < 1e-11,
              fmt("outside %.1e", v[0] / v[2]) + fmt(", total - incident %.1e", v[1] / v[2]) + fmt(", peak %.2f", v[2]) +
                  fmt(", Gauss %.1e", gd) + fmt(", div B %.1e", db) +
                  fmt(", %.0f components", double(sim.fields->side->comps().size())) +
                  fmt(" (%.0f frequencies)", double(sim.fields->side->frequencies())));
        if (comm.size == 1)
            std::printf("    largest outside: component %ld at (%ld %ld %ld); largest difference: component %ld at (%ld %ld %ld)\n",
                        wo[0], wo[1], wo[2], wo[3], wd[0], wd[1], wd[2], wd[3]);
    }
}

// T4. Background plasma (side.background_density, decision 63). (a) The cold fluid everywhere (side.fluid = all): the
//     plane waves of T1 in a plasma of 0.05 n_c (omega_p = 0.22 omega0), continuous and pulsed; the incident wave is the
//     exact discrete wave of the solver with the fluid (RIP-CC: a TE and a TM wave of their own wave vectors), so outside
//     the tube the field and the fluid current stay at round-off and inside both equal the incident ones.
//     (b) A kinetic plasma: cold electrons of 0.05 n_c (2 x 2 x 2 per cell, immobile ions) fill the kinetic region (2 cells
//     inside the faces), the fluid shell outside it; a weak pulse (a0 = 1e-3, linear) crosses the tube at 60 deg. The
//     leakage is now the difference between the particle and the fluid response (deposit, gather, the kinetic edge);
//     reference: the same with the particles but the incident wave of vacuum and no fluid (decision 61 setup).
void test_side_plasma(Comm& comm) {
    const char* path = "side_test_tmp.deck";
    { std::ofstream f(path); f << kSide; }
    struct Case { const char *sv, *ang, *el, *pol, *dur; };
    const Case cases[] = {
        {"rip_cc", "60 deg", "0 deg", "z", "0 fs"},          {"rip_cc", "75.52248781 deg", "0 deg", "y", "0 fs"},
        {"rip_cc", "90 deg", "30 deg", "z", "0 fs"},         {"irip", "60 deg", "0 deg", "z", "0 fs"},
        {"irip", "90 deg", "30 deg", "y", "0 fs"},           {"rip_cc", "60 deg", "0 deg", "y", "12 fs"},
        {"irip", "60 deg", "0 deg", "z", "12 fs"},
    };
    for (const Case& cs : cases) {
        Deck dk = Deck::read(path);
        dk.override_from(std::string("numerics.solver=") + cs.sv);
        dk.override_from(std::string("side.angle=") + cs.ang);
        dk.override_from(std::string("side.elevation=") + cs.el);
        dk.override_from(std::string("side.polarization=") + cs.pol);
        dk.override_from(std::string("side.duration=") + cs.dur);
        dk.override_from("side.background_density=0.05");
        dk.override_from("side.fluid=all");
        const bool pulse = std::string(cs.dur) != "0 fs";
        if (pulse) { dk.override_from("side.delay=15 fs"); dk.override_from("side.y_center=0.5 um"); }
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        sim.init();
        Leak m = leakage(sim);
        double gd = 0, db = 0;
        while (!sim.finished()) {
            sim.step();
            if (sim.istep % 20 == 0) {
                const Leak l = leakage(sim);
                m.out = std::max(m.out, l.out); m.diff = std::max(m.diff, l.diff); m.peak = std::max(m.peak, l.peak);
                gd = std::max(gd, sim.gauss_drift());
                db = std::max(db, comm.allreduce_max(sim.fields->max_div_b()));
            }
        }
        check(std::string("side tube in a plasma of 0.05 n_c (cold fluid), ") + cs.sv + ", " + (pulse ? "pulse" : "continuous wave") +
                  " at " + cs.ang + ", elevation " + cs.el + ", polarisation " + cs.pol + ": no leakage, total = incident",
              m.out < 1e-12 * m.peak && m.diff < 1e-12 * m.peak && m.peak > 0.5 && gd < 1e-11 && db < 1e-11,
              fmt("outside %.1e", m.out / m.peak) + fmt(", total - incident %.1e", m.diff / m.peak) + fmt(", peak %.2f", m.peak) +
                  fmt(", Gauss %.1e", gd) + fmt(", div B %.1e", db) + fmt(", %.0f components", double(sim.fields->side->comps().size())) +
                  fmt(", mode residual %.1e", sim.fields->side->mode_residual()));
    }
    // (b) the kinetic region with the fluid shell, against the same particles in a vacuum shell
    const char* plasma =
        "species electrons {\n charge = -1\n mass = 1\n density = 0.05\n ppc_x = 2\n ppc_y = 2\n ppc_z = 2\n"
        " density_function = (y >= 1.12 um) * (y <= 2.64 um) * (z >= 1.12 um) * (z <= 2.64 um)\n}\n"
        "species ions {\n charge = 1\n mass = 1836\n density = 0.05\n mobile = false\n ppc_x = 2\n ppc_y = 2\n ppc_z = 2\n"
        " density_function = (y >= 1.12 um) * (y <= 2.64 um) * (z >= 1.12 um) * (z <= 2.64 um)\n}\n";
    { std::ofstream f(path); f << kSide << plasma; }
    double out[2] = {0, 0}, peak[2] = {0, 0};
    for (int fl = 1; fl >= 0; --fl) {
        Deck dk = Deck::read(path);
        dk.override_from("side.a0=1e-3");
        dk.override_from("side.polarization=y");
        dk.override_from("side.duration=12 fs");
        dk.override_from("side.delay=15 fs");
        dk.override_from("side.y_center=0.5 um");
        if (fl) dk.override_from("side.background_density=0.05");
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        sim.init();
        while (!sim.finished()) {
            sim.step();
            if (sim.istep % 20 == 0) {
                const Leak l = leakage(sim);
                out[fl] = std::max(out[fl], l.out); peak[fl] = std::max(peak[fl], l.peak);
            }
        }
    }
    const double r1 = out[1] / peak[1], r0 = out[0] / peak[0];
    check("side tube, kinetic plasma of 0.05 n_c in the tube, cold-fluid shell: leakage far below the vacuum shell",
          r1 < 0.1 * r0 && peak[1] > 0.5e-3,
          fmt("outside %.1e of the peak with the fluid shell", r1) + fmt(", %.1e with a vacuum shell", r0));
}
