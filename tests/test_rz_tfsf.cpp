// O1 and O3 in RZ (docs/specs/tfsf.md "RZ", docs/specs/x_absorber.md "RZ"): the TFSF plane with its incident grid, the
// z absorbing layers; and the laser energy injected through an open left edge in xflux.txt (docs/specs/xflux.md).
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "common.hpp"
#include "sim.hpp"

namespace {
Config deck_config(const std::string& text, const std::vector<std::string>& ov) {
    const char* path = "rz_tfsf_test_tmp.deck";
    { std::ofstream f(path); f << text; }
    Deck dk = Deck::read(path);
    for (const std::string& o : ov) dk.override_from(o);
    return Config::from_deck(dk);
}

// the rows of xflux.txt (all columns)
std::vector<std::vector<double>> xflux_rows(const std::string& dir) {
    std::vector<std::vector<double>> r;
    std::ifstream in(dir + "/xflux.txt");
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<double> v;
        double x;
        size_t p = 0;
        while (p < line.size()) {
            int n = 0;
            if (std::sscanf(line.c_str() + p, "%lf%n", &x, &n) != 1) break;
            v.push_back(x);
            p += size_t(n);
        }
        r.push_back(v);
    }
    return r;
}

const char* kBeam = R"(
grid {
  geometry = rz
  modes = 2
  lambda0 = 0.8 um
  nz = 240
  nr = 64
  dz = 0.025 um
  dr = 0.05 um
  boundary_z = absorbing
  z_absorber_width = 1 um
  boundary_r = absorbing
  r_absorber_cells = 16
  t_end = 40 fs
}
laser {
  a0 = 1
  polarization = x
  duration = 6 fs
  delay = 12 fs
  waist = 1 um
  focus = 3 um
  tfsf_x = 1.5 um
}
output {
  dir = rz_tfsf_test_out
  every = 10
  xflux = true
}
)";
}  // namespace

void test_rz_tfsf(Comm& comm) {
    if (comm.size > 1) return;                       // host copies of the whole grid; tests/mpi.sh compares ranks
    // R1: a focused beam (waist 1 um, focus 3 um from z = 0) through the plane at 1.5 um, z layers of 1 um at both ends, a
    // radial absorber: in vacuum the main grid holds nothing left of the plane (scattered = 0) and right of it exactly the
    // incident grid (the same scheme and layers there), RIP-CC and iRIP. And the energy balance with the injected energy:
    // W_in + Enet_left + Enet_right + E_abs_r = E_inj once the pulse has crossed the plane.
    // -5.7e-4: E_inj is counted where the laser enters the incident grid at z = 0; its own radial absorber takes 3e-4 of it
    // before the plane (-2.5e-4 with nr = 128 or 256), the rest diffracts at low frequency.
    // irip o5 (decision 78): the incident grid runs the closure, the main grid's Cz across the plane gets the incident far cell.
    for (const char* sv : {"rip_cc", "irip", "irip o5"}) {
        const bool o5 = std::string(sv) == "irip o5";
        std::vector<std::string> ov{std::string("numerics.solver=") + (o5 ? "irip" : sv)};
        if (o5) ov.push_back("numerics.closure=o5");
        const Config c = deck_config(kBeam, ov);
        Simulation sim(c, comm);
        sim.init();
        const int s = sim.rz_tfsf()->plane();
        double left = 0, diff = 0, peak = 0;
        while (!sim.finished()) {
            sim.step();
            FieldsRZ& f = *sim.rz;
            FieldsRZ& g = sim.rz_tfsf()->incident();
            const Field* a[6] = {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp};
            const Field* b[6] = {&g.ez, &g.er, &g.ep, &g.bz, &g.br, &g.bp};
            for (int q = 0; q < 6; ++q) {
                const std::vector<double> x = a[q]->host(), y = b[q]->host();
                for (int k = 0; k < sim.dom.n[2]; ++k)
                    for (int j = 0; j < sim.dom.n[1]; ++j)
                        for (int i = 0; i < sim.dom.n[0]; ++i) {
                            const double u = x[a[q]->idx(i, j, k)], v = y[b[q]->idx(i, j, k)];
                            peak = std::max(peak, std::abs(v));
                            if (i < s) left = std::max(left, std::abs(u));
                            else diff = std::max(diff, std::abs(u - v));
                        }
            }
        }
        const auto r = xflux_rows(c.dir);
        const std::vector<double>& e = r.back();      // step t P_l P_r E_l E_r Enet_l Enet_r E_abs_r W_in E_part E_inj
        const double bal = (e[9] + e[6] + e[7] + e[8]) / e[11] - 1;
        std::printf("  R1 RZ TFSF plane, %s, vacuum beam: left of the plane %.1e of the peak, total - incident %.1e, energy balance "
                    "against the injected %.1e (E_inj %.4g, out right %.4g, radial absorber %.2g)\n",
                    sv, left / peak, diff / peak, bal, e[11], e[7], e[8]);
        check(std::string("R1 RZ TFSF: no leakage, total = incident (") + sv + ")", left < 1e-13 * peak && diff < 1e-13 * peak &&
              std::abs(bal) < 1e-3, fmt("%.1e %.1e", left / peak, diff / peak));
    }
    // R2: the right z layer (1 um = 40 cells) against the same run in a box 10 um longer (its layer too far away for anything
    // to come back in time): the largest field difference left of the short box's layer over the run, relative to the peak.
    {
        const Config c = deck_config(kBeam, {}), cr = deck_config(kBeam, {"grid.nz=640", "output.dir=rz_tfsf_test_ref"});
        Simulation sim(c, comm), ref(cr, comm);
        sim.init();
        ref.init();
        const int nl = c.N[0] - c.x_abs_cells;
        double diff = 0, peak = 0;
        while (!sim.finished()) {
            sim.step();
            ref.step();
            if (sim.istep % 5) continue;
            const Field* a[6] = {&sim.rz->ez, &sim.rz->er, &sim.rz->ep, &sim.rz->bz, &sim.rz->br, &sim.rz->bp};
            const Field* b[6] = {&ref.rz->ez, &ref.rz->er, &ref.rz->ep, &ref.rz->bz, &ref.rz->br, &ref.rz->bp};
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
        }
        std::printf("  R2 RZ z layer of 1 um (40 cells) against a box 10 um longer: reflected field %.1e of the peak\n", diff / peak);
        check("R2 RZ z absorber", diff < 2e-3 * peak, fmt("%.1e", diff / peak));
    }
    // R3: the laser through the open left edge, vacuum: Enet_left counts the injected energy as inbound, so
    // W_in + Enet_left + Enet_right (+ E_abs_r) stays 0 (before the fix: the field energy itself) and E_inj = E_left - Enet_left
    // is the laser energy. RZ: examples/rz_laser.deck (O(dz^2) with transverse structure); Cart3D: a plane pulse (exact).
    {
        std::string edge;                            // the beam deck with open z and the laser at the left edge
        for (size_t p = 0, q; p < std::string(kBeam).size(); p = q + 1) {
            q = std::string(kBeam).find('\n', p);
            const std::string line = std::string(kBeam).substr(p, q - p);
            if (line.find("z_absorber_width") == std::string::npos && line.find("tfsf_x") == std::string::npos) edge += line + "\n";
        }
        const Config c = deck_config(edge, {"grid.boundary_z=open", "output.every=1", "output.dir=rz_inj_test"});
        Simulation sim(c, comm);
        sim.init();
        while (!sim.finished()) sim.step();
        double m = 0, inj = 0;
        for (const auto& e : xflux_rows(c.dir)) {
            m = std::max(m, std::abs(e[9] + e[6] + e[7] + e[8]));
            inj = e[11];
        }
        const std::string cart = R"(
grid {
  lambda0 = 0.8 um
  nx = 200
  ny = 4
  nz = 4
  dx = 0.05 um
  dy = 0.2 um
  dz = 0.2 um
  boundary_x = open
  t_end = 50 fs
}
laser {
  a0 = 1
  polarization = z
  duration = 6 fs
  waist = 1000 um
}
output {
  dir = cart_inj_test
  every = 1
  xflux = true
}
)";
        const Config cc = deck_config(cart, {});
        Simulation cs(cc, comm);
        cs.init();
        while (!cs.finished()) cs.step();
        double mc = 0, injc = 0;
        for (const auto& e : xflux_rows(cc.dir)) {
            mc = std::max(mc, std::abs(e[9] + e[6] + e[7]));
            injc = e[11];
        }
        std::printf("  R3 laser through the open left edge, vacuum: max |W_in + Enet_left + Enet_right| / E_inj: RZ beam %.1e, "
                    "Cart3D plane pulse %.1e (E_inj %.5g, %.5g)\n", m / inj, mc / injc, inj, injc);
        check("R3 injected laser energy in xflux (RZ, Cart3D)", inj > 0 && injc > 0 && m < 5e-4 * inj && mc < 1e-12 * injc,
              fmt("%.1e %.1e", m / inj, mc / injc));
    }
    // R4: E_abs_r only between the measuring planes. A strongly diverging beam (waist 0.4 um, focus 2 um) reaches the radial
    // absorber inside the right z layer, beyond the right plane, where Enet_right has already counted it as gone; once the
    // pulse is in the box (30 fs), W_in + Enet_left + Enet_right + E_abs_r must stay constant. Counting the radial damping
    // over the whole z range (before 2026-11-03) added that corner a second time: +2.9e-2 by 50 fs.
    {
        const Config c = deck_config(kBeam, {"laser.waist=0.4 um", "laser.focus=2 um", "grid.t_end=50 fs"});
        Simulation sim(c, comm);
        sim.init();
        while (!sim.finished()) sim.step();
        const double t_in = 30e-15 * 2 * M_PI * 299792458.0 / 0.8e-6;          // 30 fs in 1/omega0
        double b_in = 0, dev = 0, abs_r = 0;
        for (const auto& e : xflux_rows(c.dir)) {
            const double b = e[9] + e[6] + e[7] + e[8];
            if (e[1] < t_in) continue;
            if (b_in == 0) b_in = b;
            dev = std::max(dev, std::abs(b - b_in));
            abs_r = e[8];
        }
        std::printf("  R4 diverging beam into the corner of the radial absorber and the right z layer: max |B(t) - B(30 fs)| / B "
                    "%.1e (radial absorber %.3f of B)\n", dev / b_in, abs_r / b_in);
        check("R4 RZ radial absorber energy between the measuring planes", b_in > 0 && abs_r > 0.05 * b_in && dev < 1e-3 * b_in,
              fmt("%.1e", dev / b_in));
    }
}
