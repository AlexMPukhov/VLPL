// O1: the TFSF laser source (docs/specs/tfsf.md, src/tfsf.*).
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "common.hpp"
#include "sim.hpp"
#include "tfsf.hpp"

namespace {
const char* kVac =
    "grid {\n lambda0 = 0.8 um\n nx = 480\n ny = 96\n nz = 4\n dx = 0.025 um\n dy = 0.1 um\n dz = 0.4 um\n"
    " boundary_x = open\n t_end = 70 fs\n}\n"
    "laser {\n a0 = 1\n polarization = y\n duration = 6 fs\n delay = 25 fs\n waist = 1.5 um\n focus = 1 um\n"
    " tfsf_x = 8.5 um\n angle = 30 deg\n}\n"
    "output {\n dir = tfsf_test_out\n}\n";

struct Leak { double left = 0, diff = 0, peak = 0; };

// max |E, B| left of the plane (scattered region), max |main - incident| right of it, max |E, B| overall
Leak leakage(Simulation& sim) {
    const Fields& f = *sim.fields;
    const Fields& g = sim.fields->tfsf->incident();
    const Domain& d = sim.dom;
    const int s = sim.fields->tfsf->plane() - d.off[0];
    Leak r;
    const Field* a[6] = {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz};
    const Field* b[6] = {&g.ex, &g.ey, &g.ez, &g.bx, &g.by, &g.bz};
    for (int c = 0; c < 6; ++c) {
        const std::vector<double> x = a[c]->host(), y = b[c]->host();
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    const size_t q = a[c]->idx(i, j, k);
                    r.peak = std::max(r.peak, std::abs(x[q]));
                    if (i < s) r.left = std::max(r.left, std::abs(x[q]));
                    else r.diff = std::max(r.diff, std::abs(x[q] - y[q]));
                }
    }
    double v[3] = {r.left, r.diff, r.peak};
    for (double& x : v) x = sim.comm.allreduce_max(x);
    return {v[0], v[1], v[2]};
}
}  // namespace

// T1. Vacuum, a focused Gaussian beam (waist 1.5 um, focus 1 um beyond the plane) through the TFSF plane at 0 and 30 deg,
//     both polarisations, RIP-CC and iRIP: the field left of the plane stays at round-off (no leakage) and right of it
//     equals the incident grid to round-off; the beam does get through (peak ~ a0). With the cut envelope (zero beyond
//     3 sigma) nothing of the pulse reaches across the plane at t = 0 (the overlap is exactly 0). The Gauss law holds
//     to round-off away from the plane (where E jumps by the incident field: skipped, Fields::tfsf_point). The same with
//     CPML layers in y (12 cells) for the 30 deg beam: the incident grid has the same layers, the corrections go through
//     their own CPML memories. The incident grid starts with div E = div B = 0 on the grid (its Ex, Bx from the discrete
//     constraints outside the CPML layers, Tfsf): no static field stays where the pulse started. At 0 deg the incident
//     grid holds < 1e-2 of the peak after the pulse has left (4e-3: waves partly reflected at the open x edges, trapped in
//     the periodic y box); before, the beam formula lacked the longitudinal field of the focusing, and its div E, div B left
//     a static Ex, Bx of 7e-2 (0.2-0.3 at 30 deg).
static void tfsf_vacuum(Comm& comm) {
    const char* path = "tfsf_test_tmp.deck";
    { std::ofstream f(path); f << kVac; }
    for (const char* lay : {"periodic", "pml"})
    for (const char* sv : {"rip_cc", "irip", "irip o5"})
    for (const char* w : {"1e6 um", "1.5 um"})
    for (const char* pol : {"y", "z"})
        for (const char* ang : {"0 deg", "30 deg"}) {
            if (std::string(w) == "1e6 um" && std::string(ang) != "0 deg") continue;   // an oblique plane wave overlaps the plane
            if (std::string(lay) == "pml" && (std::string(w) != "1.5 um" || std::string(ang) != "30 deg")) continue;
            Deck dk = Deck::read(path);
            dk.override_from(std::string("laser.polarization=") + pol);
            dk.override_from(std::string("laser.angle=") + ang);
            dk.override_from(std::string("laser.waist=") + w);
            const bool o5 = std::string(sv) == "irip o5";
            dk.override_from(std::string("numerics.solver=") + (o5 ? "irip" : sv));
            if (o5) dk.override_from("numerics.closure=o5");
            if (std::string(lay) == "pml") { dk.override_from("grid.boundary_y=pml"); dk.override_from("pml.cells=12"); }
            Config c = Config::from_deck(dk);
            Simulation sim(c, comm);
            sim.init();
            const Fields& inc = sim.fields->tfsf->incident();
            const double amax = sim.fields->tfsf->max_incident();
            Field gr;
            gr.resize(sim.dom);
            inc.gauss_residual(gr);
            double div = 0;
            for (double v : gr.host()) div = std::max(div, std::abs(v));
            div = sim.comm.allreduce_max(std::max(div, inc.max_div_b())) * sim.dom.h[0] / amax;
            Leak m;
            double gd = 0;
            int n = 0;
            while (!sim.finished()) {
                sim.step();
                if (++n % 20 == 0) {
                    const Leak l = leakage(sim);
                    m.left = std::max(m.left, l.left); m.diff = std::max(m.diff, l.diff); m.peak = std::max(m.peak, l.peak);
                    gd = std::max(gd, sim.gauss_drift());
                }
            }
            double rest = 0;                               // what the incident grid still holds at the end
            for (const Field* f : {&inc.ex, &inc.ey, &inc.ez, &inc.bx, &inc.by, &inc.bz})
                for (double v : f->host()) rest = std::max(rest, std::abs(v));
            rest = sim.comm.allreduce_max(rest) / amax;
            check(std::string("tfsf, ") + sv + ", " + lay + " y, vacuum, waist " + w + ", at " + ang + ", polarisation " + pol + ": no leakage",
                  m.left < 1e-13 * m.peak && m.diff < 1e-13 * m.peak && m.peak > 0.5 && gd < 1e-12 &&
                      (std::string(ang) != "0 deg" || sim.fields->tfsf->overlap() == 0) && div < 1e-12 &&
                      (std::string(ang) != "0 deg" || rest < 1e-2),
                  fmt("left of the plane %.1e", m.left / m.peak) + fmt(", total - incident %.1e", m.diff / m.peak) +
                      fmt(", peak %.2f", m.peak) + fmt(", initial overlap %.1e", sim.fields->tfsf->overlap()) + fmt(", Gauss %.1e", gd) +
                      fmt(", initial div E, div B (dx / peak) %.1e", div) + fmt(", left in the incident grid %.1e", rest));
        }
}

// T2 (O3). grid.boundary_x = absorbing with the default thickness 4 lambda0 (64 cells at lambda0/16, R0 = 1e-4): a focused
//     beam at 0, 30, 45 deg from the TFSF plane runs into the right layer. The reflection is the difference to the same run
//     in a box 15 um longer (its own layer too far away for anything to come back in time), max over the cells left of
//     the layer and over time, relative to the peak field. The main grid keeps the leakage across the TFSF plane at
//     round-off, and nothing comes out through the left measuring plane (E_net_left, xflux.txt, every step). RIP-CC at
//     the three angles, iRIP at 45 deg. (The reflection is measured on the main grid, not by the energy left in the
//     incident grid.)
void test_x_absorber(Comm& comm) {
    if (comm.size > 1) return;                           // the reference box has another decomposition
    const char* path = "tfsf_test_xabs.deck";
    {
        std::ofstream f(path);
        f << "grid {\n lambda0 = 0.8 um\n nx = 400\n ny = 96\n nz = 4\n dx = 0.05 um\n dy = 0.1 um\n dz = 0.4 um\n"
             " boundary_x = absorbing\n t_end = 95 fs\n}\n"
             "laser {\n a0 = 1\n polarization = z\n duration = 6 fs\n delay = 21 fs\n waist = 1.5 um\n focus = 1 um\n"
             " tfsf_x = 13 um\n angle = 30 deg\n}\n"
             "output {\n dir = tfsf_test_out\n every = 1\n xflux = true\n}\n";
    }
    for (const char* sv : {"rip_cc", "irip"})
        for (const char* ang : {"0 deg", "30 deg", "45 deg"}) {
            if (std::string(sv) == "irip" && std::string(ang) != "45 deg") continue;
            auto config = [&](bool ref) {
                Deck dk = Deck::read(path);
                dk.override_from(std::string("laser.angle=") + ang);
                dk.override_from(std::string("numerics.solver=") + sv);
                if (ref) { dk.override_from("grid.nx=700"); dk.override_from("output.dir=tfsf_test_ref"); }
                return Config::from_deck(dk);
            };
            const Config c = config(false), cr = config(true);
            Simulation sim(c, comm), ref(cr, comm);
            sim.init();
            ref.init();
            const int nl = c.N[0] - c.x_abs_cells;          // the cells left of the right layer
            double diff = 0, peak = 0, left = 0, lpeak = 0;
            while (!sim.finished()) {
                sim.step();
                ref.step();
                if (sim.istep % 5) continue;
                const Fields& f = *sim.fields;
                const Fields& g = *ref.fields;
                const Field* a[6] = {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz};
                const Field* b[6] = {&g.ex, &g.ey, &g.ez, &g.bx, &g.by, &g.bz};
                for (int q = 0; q < 6; ++q) {
                    const std::vector<double> x = a[q]->host(), y = b[q]->host();
                    for (int k = 0; k < sim.dom.n[2]; ++k)
                        for (int j = 0; j < sim.dom.n[1]; ++j)
                            for (int i = 0; i < nl; ++i) {
                                const double u = x[a[q]->idx(i, j, k)], v = y[b[q]->idx(i, j, k)];
                                diff = std::max(diff, std::abs(u - v));
                                peak = std::max(peak, std::abs(v));
                            }
                }
                if (sim.istep % 25 == 0) {
                    const Leak l = leakage(sim);
                    left = std::max(left, l.left); lpeak = std::max(lpeak, l.peak);
                }
            }
            // the net outflow through the left measuring plane from xflux.txt (every step)
            std::ifstream in(c.dir + "/xflux.txt");
            double bal = 0, eright = 0;
            for (std::string line; std::getline(in, line);) {
                if (line.empty() || line[0] == '#') continue;
                double v[8];
                if (std::sscanf(line.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf", v, v + 1, v + 2, v + 3, v + 4, v + 5, v + 6, v + 7) == 8) {
                    bal = std::max(bal, std::abs(v[6]));
                    eright = v[7];
                }
            }
            const double r = diff / peak;
            check(std::string("tfsf, ") + sv + ", x absorber, beam at " + ang + ": reflection, no leakage, nothing out at the left",
                  r < 2e-3 && left < 1e-13 * lpeak && bal < 1e-12 * eright,
                  fmt("reflected field %.1e of the peak", r) + fmt(", leakage %.1e", left / lpeak) +
                      fmt(", E_net_left / E_net_right %.1e", bal / eright));
        }
}

void test_tfsf(Comm& comm) {
    tfsf_vacuum(comm);
}

// T3. The TFSF plane with exact discrete plane waves (laser.tfsf_incident = modes, tfsf_modes.hpp): no incident grid.
//   a) Vacuum, the deck of T1 (periodic y, open x) 22.5 um long: beams at 0, 30, 45 deg, both polarisations, RIP-CC and
//      iRIP, and a round beam (waist_z): the field left of the plane stays at round-off (no leakage) until 50 fs. (The open
//      right edge reflects an oblique beam by (1 - cos a) / (1 + cos a); the plane waves are only the incident +x waves, so
//      that echo crosses into the scattered region, while the incident grid contained it: keep the pulse off that edge, or
//      use absorbing x as in c.)
//   b) The same beam as with the incident grid (30 deg, polarisation y): the fields right of the plane at 40 fs agree to
//      the difference of the beam models (paraxial entrance formula with a cut envelope against the exact angular spectrum
//      of a Gaussian focus).
//   c) Absorbing x: the plane two cells right of the left layer, the pulse starting outside the box (delay 60 fs): no
//      leakage, nothing out through the left measuring plane.
static double max_left(Simulation& sim, double& peak) {
    const Fields& f = *sim.fields;
    const Domain& d = sim.dom;
    const int s = sim.fields->tfsf->plane() - d.off[0];
    double left = 0, pk = 0;
    for (const Field* a : {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz}) {
        const std::vector<double> x = a->host();
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    const double v = std::abs(x[a->idx(i, j, k)]);
                    pk = std::max(pk, v);
                    if (i < s - 1) left = std::max(left, v);    // the cell s - 1 holds the scattered field next to the plane
                }
    }
    peak = std::max(peak, sim.comm.allreduce_max(pk));
    return sim.comm.allreduce_max(left);
}

void test_tfsf_modes(Comm& comm) {
    const char* path = "tfsf_test_tmp.deck";
    { std::ofstream f(path); f << kVac; }
    struct Case { const char* sv; const char* pol; const char* ang; bool round; };
    const Case cases[] = {{"rip_cc", "y", "0 deg", false}, {"rip_cc", "y", "30 deg", false}, {"rip_cc", "z", "30 deg", false},
                          {"rip_cc", "y", "45 deg", false}, {"irip", "y", "45 deg", false}, {"irip", "z", "30 deg", false},
                          {"rip_cc", "y", "45 deg", true}, {"irip o5", "y", "45 deg", false}, {"irip o5", "z", "30 deg", false},
                          {"irip o5", "y", "45 deg", true}};
    for (const Case& K : cases) {
        Deck dk = Deck::read(path);
        dk.override_from("laser.tfsf_incident=modes");
        dk.override_from("grid.nx=900");                  // 22.5 um: the pulse stays clear of the open right edge, which
        dk.override_from("grid.t_end=40fs");               // reflects oblique waves ((1 - cos a) / (1 + cos a), decision 59)
        dk.override_from(std::string("laser.polarization=") + K.pol);
        dk.override_from(std::string("laser.angle=") + K.ang);
        const bool o5 = std::string(K.sv) == "irip o5";
        dk.override_from(std::string("numerics.solver=") + (o5 ? "irip" : K.sv));
        if (o5) dk.override_from("numerics.closure=o5");
        if (K.round) { dk.override_from("grid.nz=32"); dk.override_from("grid.dz=0.1um"); dk.override_from("laser.waist_z=1um"); }
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        sim.init();
        double left = 0, peak = 0, gd = 0;
        int n = 0;
        while (!sim.finished()) {
            sim.step();
            if (++n % 20 == 0) { left = std::max(left, max_left(sim, peak)); gd = std::max(gd, sim.gauss_drift()); }
        }
        const size_t nc = sim.fields->tfsf->plane_waves()->count();
        check(std::string("tfsf modes, ") + K.sv + ", pol " + K.pol + ", " + K.ang + (K.round ? ", round" : "") + ": no leakage",
              left < 1e-13 * peak && peak > 0.5 && gd < 1e-12,
              fmt("left of the plane %.1e", left / peak) + fmt(", peak %.2f, Gauss %.1e", peak, gd) + fmt(", %.0f plane waves", double(nc)));
    }
    for (const char* ang : {"0 deg"}) {   // b) against the incident grid (at 30 deg its initial beam is cut by the y box)
        std::vector<double> ey[2];
        double pk[2] = {0, 0}, num = 0, den = 0;
        for (int m = 0; m < 2; ++m) {
            Deck dk = Deck::read(path);
            if (m) dk.override_from("laser.tfsf_incident=modes");
            dk.override_from("grid.nx=900");
            dk.override_from("laser.tfsf_x=12um");     // the pulse of the incident grid must fit between x = 0 and the plane
            dk.override_from("grid.t_end=40fs");
            dk.override_from(std::string("laser.angle=") + ang);
            Config c = Config::from_deck(dk);
            Simulation sim(c, comm);
            sim.init();
            int n = 0;
            while (!sim.finished()) {
                sim.step();
                if (++n % 10) continue;
                for (double v : sim.fields->ey.host()) pk[m] = std::max(pk[m], std::abs(v));
            }
            ey[m] = sim.fields->ey.host();
            if (m == 1) {
                const Domain& d = sim.dom;
                const int s = sim.fields->tfsf->plane() - d.off[0];
                for (int k = 0; k < d.n[2]; ++k)
                    for (int j = 0; j < d.n[1]; ++j)
                        for (int i = s < 0 ? 0 : s; i < d.n[0]; ++i) {
                            const size_t q = sim.fields->ey.idx(i, j, k);
                            num += (ey[1][q] - ey[0][q]) * (ey[1][q] - ey[0][q]); den += ey[0][q] * ey[0][q];
                        }
            }
        }
        comm.allreduce_sum(&num, 1); comm.allreduce_sum(&den, 1);
        pk[0] = comm.allreduce_max(pk[0]); pk[1] = comm.allreduce_max(pk[1]);
        const double rel = std::sqrt(num / den);
        check(std::string("tfsf modes = incident grid (") + ang + ", waist 1.5 um, Ey at 40 fs) to the beam model",
              rel < 6e-2 && std::abs(pk[1] / pk[0] - 1) < 1e-2,
              fmt("relative L2 difference %.1e", rel) + fmt(", peak over the run modes / grid %.4f (grid %.4f)", pk[1] / pk[0], pk[0]));
    }
    for (int abs = 0; abs < 2; ++abs) {   // c) the plane next to the left x edge (open) or layer (absorbing, 4 lambda0), long box
        Deck dk = Deck::read(path);
        dk.override_from("laser.tfsf_incident=modes");
        dk.override_from("laser.angle=45deg");
        dk.override_from(abs ? "laser.tfsf_x=3.4um" : "laser.tfsf_x=0.2um");
        dk.override_from("laser.delay=40fs");
        dk.override_from("grid.nx=1800");                  // 45 um: no echo of the right edge returns to the plane within 60 fs
        if (abs) dk.override_from("grid.boundary_x=absorbing");
        dk.override_from("grid.t_end=60fs");
        Config c = Config::from_deck(dk);
        Simulation sim(c, comm);
        sim.init();
        double left = 0, peak = 0;
        int n = 0;
        while (!sim.finished()) {
            sim.step();
            if (++n % 20 == 0) left = std::max(left, max_left(sim, peak));
        }
        check(std::string("tfsf modes, plane 8 cells from the ") + (abs ? "left x layer" : "open left edge") +
                  ", 45 deg, pulse from outside the box: no leakage",
              left < 1e-13 * peak && peak > 0.5, fmt("left of the plane %.1e, peak %.2f", left / peak, peak));
    }
}
