// O2: the field leaving through the open x edges (docs/specs/xflux.md, src/xflux.*).
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "common.hpp"
#include "sim.hpp"

namespace {
struct Run {
    std::vector<double> W, El, Er;   // after step n + 1: field + kinetic energy, energy out through left / right
    std::vector<double> left;        // with xflux_every = 1: the record of the left edge (rank 0), [step][2][plane]
    std::vector<double> right;
    long plane = 0;
    Config c;
};

Run run_deck(Comm& comm, const std::string& text, const std::vector<std::string>& ov) {
    const char* path = "xflux_test_tmp.deck";
    { std::ofstream f(path); f << text; }
    Deck dk = Deck::read(path);
    dk.override_from("output.every=1");
    for (const std::string& o : ov) dk.override_from(o);
    Run r;
    r.c = Config::from_deck(dk);
    Simulation sim(r.c, comm);
    sim.init();
    while (!sim.finished()) {
        sim.step();
        double v[2] = {sim.cfg.rz ? sim.rz->energy() : sim.fields->energy(), 0};
        for (const Species& s : sim.species) v[1] += s.kinetic_energy();
        comm.allreduce_sum(v, 2);
        r.W.push_back(v[0] + v[1]);
    }
    std::ifstream in(r.c.dir + "/xflux.txt");
    for (std::string line; std::getline(in, line);) {
        if (line.empty() || line[0] == '#') continue;
        long step; double t, pl, pr, el, er;
        if (std::sscanf(line.c_str(), "%ld %lf %lf %lf %lf %lf", &step, &t, &pl, &pr, &el, &er) == 6) {
            r.El.push_back(el); r.Er.push_back(er);
        }
    }
    r.plane = sim.cfg.rz ? long(sim.dom.n[1]) * sim.dom.n[2] : long(sim.dom.n[1]) * sim.dom.n[2];
    for (int e = 0; e < 2 && r.c.xflux.every > 0; ++e) {
        std::ifstream b(r.c.dir + (e ? "/xflux/right_r0.bin" : "/xflux/left_r0.bin"), std::ios::binary);
        std::vector<double>& dst = e ? r.right : r.left;
        for (double x; b.read(reinterpret_cast<char*>(&x), sizeof x);) dst.push_back(x);
    }
    return r;
}

// largest relative change of W + E_left + E_right after step n0, and the totals at n0 and at the end
double balance(const Run& r, size_t n0, double& w0, double& w1) {
    const size_t n = std::min(r.W.size(), r.El.size());
    w0 = r.W[n0] + r.El[n0] + r.Er[n0];
    double m = 0;
    for (size_t q = n0; q < n; ++q) m = std::max(m, std::abs(r.W[q] + r.El[q] + r.Er[q] - w0));
    w1 = r.W[n - 1] + r.El[n - 1] + r.Er[n - 1];
    return m / w0;
}

// complex amplitude at omega = omega0 of the plane mean of the first component (E_y +- B_z) in a record
std::complex<double> amp(const std::vector<double>& rec, long plane, double dt) {
    std::complex<double> a = 0;
    const size_t n = rec.size() / size_t(2 * plane);
    for (size_t q = 0; q < n; ++q) {
        double s = 0;
        for (long p = 0; p < plane; ++p) s += rec[q * 2 * plane + size_t(p)];
        a += s / double(plane) * std::exp(std::complex<double>(0, dt * double(q + 1)));
    }
    return a;
}

const char* kVacuum =
    "grid {\n lambda0 = 0.8 um\n nx = 200\n ny = 4\n nz = 4\n dx = 0.05 um\n dy = 0.5 um\n dz = 0.5 um\n"
    " boundary_x = open\n t_end = 55 fs\n}\n"
    "laser {\n a0 = 0.01\n polarization = y\n duration = 4 fs\n delay = 9 fs\n waist = 1e6 um\n focus = 0 um\n}\n"
    "output {\n dir = xflux_test_out\n xflux = true\n}\n";

const char* kRZ =
    "grid {\n geometry = rz\n modes = 2\n lambda0 = 0.8 um\n nz = 200\n nr = 40\n dz = 0.05 um\n dr = 0.25 um\n"
    " boundary_z = open\n boundary_r = wall\n t_end = 55 fs\n}\n"
    "laser {\n a0 = 0.01\n polarization = x\n duration = 4 fs\n delay = 9 fs\n waist = 3 um\n focus = 5 um\n}\n"
    "output {\n dir = xflux_test_out\n xflux = true\n}\n";

// a cold overdense step (9 n_c, electrons on immobile ions) from 1.5 um to 3 um in a 4 um box
const char* kMirror =
    "grid {\n lambda0 = 0.8 um\n nx = 500\n ny = 4\n nz = 4\n dx = 0.008 um\n dy = 0.1 um\n dz = 0.1 um\n"
    " boundary_x = open\n t_end = 45 fs\n}\n"
    "laser {\n a0 = 0.001\n polarization = y\n duration = 5 fs\n delay = 12 fs\n waist = 1e6 um\n focus = 0 um\n}\n"
    "species electrons {\n charge = -1\n mass = 1\n density = 9\n xmin = 1.5 um\n xmax = 3 um\n ppc_x = 2\n}\n"
    "species ions {\n charge = 1\n mass = 1836\n mobile = false\n density = 9\n xmin = 1.5 um\n xmax = 3 um\n ppc_x = 2\n}\n"
    "output {\n dir = xflux_test_out\n xflux_every = 1\n}\n";
}  // namespace

// X1. A plane pulse in vacuum enters at the left and leaves at the right: once it is inside, W + E_left +
//     E_right is constant to round-off (the transport is an exact shift of T+- for a plane wave), it all
//     leaves through the right edge, nothing through the left (RIP-CC and iRIP).
static void xflux_vacuum(Comm& comm) {
    for (const char* sv : {"rip_cc", "irip"}) {
        Run r = run_deck(comm, kVacuum, {std::string("numerics.solver=") + sv});
        const size_t n0 = size_t(30e-15 / (0.05e-6 / 2.99792458e8));        // the pulse is inside at 30 fs
        double w0, w1;
        const double d = balance(r, n0, w0, w1);
        const double wend = r.W.back(), el = r.El.back();
        check(std::string("xflux, plane pulse in vacuum (") + sv + "): energy out = energy in", d < 1e-11 && wend < 1e-12 * w0 &&
              el < 1e-12 * w0 && r.Er.back() > 0.999 * w0,
              fmt("max |d(W + E_out)| / W %.1e", d) + fmt(", left %.1e", el / w0) + fmt(", left in the box %.1e", wend / w0));
    }
}

// X2. RZ, a focused Gaussian beam (waist 3 um, modes 0..1, dz = lambda0/16): the balance holds to a discretisation
//     term at the edge, which falls as dz^2 (1.1e-4 .. 1.6e-4 here, 2-3e-5 at lambda0/32): bound 3e-4.
static void xflux_rz(Comm& comm) {
    for (const char* sv : {"rip_cc", "irip"}) {
        Run r = run_deck(comm, kRZ, {std::string("numerics.solver=") + sv});
        const size_t n0 = size_t(30e-15 / (0.05e-6 / 2.99792458e8));
        double w0, w1;
        const double d = balance(r, n0, w0, w1);
        check(std::string("xflux, RZ Gaussian beam (") + sv + "): energy out = energy in to O(dz^2)",
              d < 3e-4 && r.W.back() < 1e-5 * w0 && r.Er.back() > 0.99 * w0,
              fmt("max |d(W + E_out)| / W %.1e", d) + fmt(", left %.1e", r.El.back() / w0) + fmt(", in the box %.1e", r.W.back() / w0));
    }
}

// X3. Normal incidence on a cold overdense step (n = 9 n_c, a0 = 1e-3): the reflected energy (left edge)
//     equals the incident one (the right edge of the same run without plasma), and the phase of the reflected
//     field at omega0 is the Fresnel phase -2 atan(sqrt(n/n_c - 1)) (|r| = 1 for a lossless overdense step).
static void xflux_mirror(Comm& comm) {
    if (comm.size > 1) return;                       // reads the records of rank 0
    Run p = run_deck(comm, kMirror, {});
    Run v = run_deck(comm, kMirror, {"species electrons.density=0", "species ions.density=0"});
    const double R = p.El.back() / v.Er.back();
    const double dt = p.c.dt(), k0 = 2 * M_PI / 0.8e-6;
    const double x0 = 1.5e-6 * k0, L = (p.c.N[0] - 1) * p.c.h[0];
    const std::complex<double> r = amp(p.left, p.plane, dt) / amp(v.right, v.plane, dt) *
                                   std::exp(std::complex<double>(0, L - 2 * x0));
    const double fres = -2 * std::atan(std::sqrt(9.0 - 1));
    check("xflux, overdense step at normal incidence: R = 1, Fresnel phase", std::abs(R - 1) < 1e-4 &&
              std::abs(std::abs(r) - 1) < 1e-3 && std::abs(std::arg(r) - fres) < 0.01 && p.Er.back() < 1e-8 * v.Er.back(),
          fmt("R %.6f", R) + fmt(", |r| %.5f", std::abs(r)) + fmt(", phase %.4f", std::arg(r)) + fmt(" (Fresnel %.4f)", fres) +
              fmt(", transmitted %.1e", p.Er.back() / v.Er.back()));
}

// X4. With a moving window the edge cell does not hold all of the outgoing wave: refused.
static void xflux_window_refused(Comm& comm) {
    bool refused = false;
    std::string msg = "no exception";
    try {
        run_deck(comm, kVacuum, {"window.start=20 fs", "window.velocity=1"});
    } catch (const std::exception& e) { msg = e.what(); refused = msg.find("xflux") != std::string::npos; }
    check("xflux: refused with a moving window", refused, refused ? "" : msg);
}

void test_xflux(Comm& comm) {
    xflux_vacuum(comm);
    xflux_rz(comm);
    xflux_mirror(comm);
    xflux_window_refused(comm);
}

// F1. Oblique Fresnel reflection (track O): a plane wave of one k_y (laser.waist = 0, periodic y with sin(theta) L_y =
//     lambda0) from the TFSF plane onto a cold plasma step (electrons on immobile ions, a0 = 1e-3), absorbing x. The
//     reflection coefficient at omega0 is the ratio of the outbound component at the left measuring plane (only the
//     reflected wave is left of the TFSF plane) to the one at the right plane of the same run without plasma (the
//     incident wave), both projected on exp(i k_y y). The surface lies half way between the two planes, so the
//     propagation phases cancel exactly, whatever the numerical k_x. Against the Fresnel formulas with
//     eps = 1 - n / n_c: s r = (cos - k) / (cos + k), p (E_y over E_y) r = (k - eps cos) / (k + eps cos),
//     k = sqrt(eps - sin^2). The underdense slab runs into the right absorber (no rear face); docs/specs/fresnel.md.
namespace {
// shift: the plasma surface at xs + shift dx (0: on a node; 0.5: half way between nodes)
std::string fresnel_deck(double ang, char pol, double n, double dxum, const char* solver, double shift = 0) {
    const double lam = 0.8, s = std::sin(ang * M_PI / 180);
    const int ny = s > 0 ? 32 : 4;
    const double dy = s > 0 ? lam / s / ny : 0.1;
    const double xs = 10.8, xp = 10.3;                       // the surface (box centre) and the TFSF plane (um)
    const int nx = 2 * int(std::lround(xs / dxum)) + 1;
    const double xmax = n > 1 ? xs + 1.5 : 2 * xs + 1;       // overdense: 1.5 um; underdense: into the right absorber (no rear face)
    // the record ends when the reflected pulse has passed, its slow tail near the cutoff omega = c k_y included (the 12 fs
    // pulse keeps little spectrum there; in the underdense slab also near the cutoff of the transmitted wave)
    const double tend = n > 1 ? 130 : 160;
    char b[2048];
    std::snprintf(b, sizeof b,
        "grid {\n lambda0 = 0.8 um\n nx = %d\n ny = %d\n nz = 4\n dx = %.10g um\n dy = %.10g um\n dz = 0.1 um\n"
        " boundary_x = absorbing\n t_end = %g fs\n}\n"
        "numerics {\n solver = %s\n}\n"
        "laser {\n a0 = 0.001\n polarization = %c\n duration = 12 fs\n delay = 24 fs\n waist = 0 um\n tfsf_x = %g um\n angle = %g deg\n}\n"
        "species electrons {\n charge = -1\n mass = 1\n density = %g\n xmin = %g um\n xmax = %g um\n ppc_x = 2\n}\n"
        "species ions {\n charge = 1\n mass = 1836\n mobile = false\n density = %g\n xmin = %g um\n xmax = %g um\n ppc_x = 2\n}\n"
        "output {\n dir = xflux_test_out\n xflux_every = 1\n}\n",
        nx, ny, dxum, dy, tend, solver, pol, xp, ang, n, xs + shift * dxum, xmax, n, xs + shift * dxum, xmax);
    return b;
}

// complex amplitude at omega0 of component comp of a record, projected on exp(i ky y)
std::complex<double> amp_ky(const std::vector<double>& rec, long plane, int ny, int comp, double ky, double hy, double dt) {
    std::complex<double> a = 0;
    const size_t n = rec.size() / size_t(2 * plane);
    for (size_t q = 0; q < n; ++q) {
        std::complex<double> s = 0;
        for (long p = 0; p < plane; ++p)
            s += rec[(q * 2 + size_t(comp)) * size_t(plane) + size_t(p)] * std::exp(std::complex<double>(0, -ky * double(p % ny) * hy));
        a += s / double(plane) * std::exp(std::complex<double>(0, dt * double(q + 1)));
    }
    return a;
}

std::complex<double> fresnel_r(double ang, char pol, double n) {
    const double c = std::cos(ang * M_PI / 180), s = std::sin(ang * M_PI / 180), eps = 1 - n;
    const std::complex<double> k = std::sqrt(std::complex<double>(eps - s * s, 0.0));
    return pol == 'z' ? (c - k) / (c + k) : (k - eps * c) / (k + eps * c);
}
}  // namespace

void test_fresnel(Comm& comm) {
    if (comm.size > 1) return;                       // reads the records of rank 0
    // lambda0/32 (the convergence to lambda0/64 and /128 is in docs/specs/fresnel.md): bounds on |r| - |r_F| (relative
    // for the underdense slab) and arg(r / r_F). p sees the one-cell transition layer that the deposition makes of the
    // step at first order (Drude-Lekner); with the surface half way between nodes (sh = 0.5) most of it cancels
    struct Case { double n, ang; char pol; double dr, da, sh; };
    const Case cases[] = {{9, 45, 'z', 1e-3, 2e-2, 0}, {9, 30, 'y', 1e-3, 1e-2, 0.5}, {9, 45, 'y', 1e-3, 3e-2, 0.5},
                          {0.5, 0, 'z', 3e-3, 5e-3, 0}, {0.25, 30, 'z', 2e-2, 1e-2, 0}};
    const double dxum = 0.025;
    for (const Case& cs : cases) {
        const std::string dk = fresnel_deck(cs.ang, cs.pol, cs.n, dxum, "rip_cc", cs.sh);
        Run p = run_deck(comm, dk, {});
        Run v = run_deck(comm, dk, {"species electrons.density=0", "species ions.density=0"});
        const int ny = int(p.c.N[1]), comp = cs.pol == 'y' ? 0 : 1;
        const double ky = std::sin(cs.ang * M_PI / 180), dt = p.c.dt();
        // r referred to the shifted surface: remove the vacuum path 2 cos(theta) shift dx of the reflected wave
        const std::complex<double> r = amp_ky(p.left, p.plane, ny, comp, ky, p.c.h[1], dt) /
                                       amp_ky(v.right, v.plane, ny, comp, ky, v.c.h[1], dt) *
                                       std::exp(std::complex<double>(0, -2 * std::cos(cs.ang * M_PI / 180) * cs.sh * p.c.h[0]));
        const std::complex<double> f = fresnel_r(cs.ang, cs.pol, cs.n);
        // the record must have ended after the reflected pulse: max over the last 40 steps / max
        double mx = 0, tail = 0;
        const size_t ns = p.left.size() / size_t(2 * p.plane);
        for (size_t q = 0; q < ns; ++q)
            for (long j = 0; j < p.plane; ++j) {
                const double x = std::abs(p.left[(q * 2 + size_t(comp)) * size_t(p.plane) + size_t(j)]);
                mx = std::max(mx, x);
                if (q + 40 >= ns) tail = std::max(tail, x);
            }
        const double dr = (std::abs(r) - std::abs(f)) / (cs.n > 1 ? 1 : std::abs(f)), da = std::arg(r / f);
        char name[160], msg[256];
        std::snprintf(name, sizeof name, "fresnel, %g n_c, %g deg, pol %c%s: |r| and phase against the Fresnel formula",
                      cs.n, cs.ang, cs.pol, cs.sh > 0 ? ", surface between nodes" : "");
        std::snprintf(msg, sizeof msg, "|r| %.5f (Fresnel %.5f), d|r| %.1e, darg %.1e rad, tail %.1e", std::abs(r),
                      std::abs(f), dr, da, tail / mx);
        check(name, std::abs(dr) < cs.dr && std::abs(da) < cs.da && tail < 2e-2 * mx, msg);
    }
}
