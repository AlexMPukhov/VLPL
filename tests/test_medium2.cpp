// O7: linear media, part 2 (docs/specs/linear_medium.md, decision 84): the deck block and the sub-cell weighting, the Gauss law
// with particles plus a medium (explicit and implicit movers), RZ (energy per mode, Gauss law) and the parabolic plasma mirror
// against physical optics.
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include "common.hpp"
#include "deck.hpp"
#include "implicit_mover.hpp"
#include "medium.hpp"
#include "rz.hpp"
#include "sim.hpp"

namespace {
Config deck_config(const std::string& text, const char* path) {
    if (FILE* f = std::fopen(path, "w")) { std::fputs(text.c_str(), f); std::fclose(f); }
    Deck dk = Deck::read(path);
    return Config::from_deck(dk);
}
double sim_total(Simulation& s, Comm& comm) {
    double w = s.rz ? s.rz->energy() : s.fields->energy(), k = 0;
    for (const Species& sp : s.species) k += sp.kinetic_energy();
    comm.allreduce_sum(&w, 1);
    comm.allreduce_sum(&k, 1);
    return w + k + s.medium_energy() + s.medium->dissipated;
}
}  // namespace

// L5. The deck block: permittivity -> wp2 = (eps - 1)(w_r^2 - 1); the sub-cell volume fraction at an edge between grid points
//     (supersample 4: exactly 1/4 at the Jx node a quarter cell before the edge, 3/4 at Jy half a cell further); unsupported
//     combinations rejected.
static void medium_deck(Comm& comm) {
    const std::string base =
        "grid {\n geometry = cartesian2d\n lambda0 = 1 um\n nx = 16\n ny = 4\n dx = 0.1 um\n dy = 0.1 um\n t_end = 1 fs\n}\n"
        "numerics {\n solver = irip\n}\n"
        "medium glass {\n model = lorentz\n permittivity = 2.25\n resonance = 3\n profile = x >= 0.825 um\n}\n"
        "output {\n dir = medium_tmp_out\n}\n";
    Config c = deck_config(base, "medium_tmp.deck");
    const double k0 = 2 * M_PI;
    const bool dens = c.media.size() == 1 && std::abs(c.media[0].density - 10) < 1e-12 && c.media[0].w0 == 3;
    Simulation sim(c, comm);
    sim.init();
    // Jx at x_i = i dx: the node i = 8 (x = 0.8 um) samples [0.75, 0.85] um: 1/4 of it beyond 0.825; Jy at x_8 + dx/2: 3/4
    double fx = -1, fy = -1;
    const Domain& d = sim.dom;
    const int i8 = 8 - d.off[0];
    if (i8 >= 0 && i8 < d.n[0]) {
        fx = H(sim.medium->wp2[0])(i8, 0, 0) / 10;
        fy = H(sim.medium->wp2[1])(i8, 0, 0) / 10;
    }
    fx = comm.allreduce_max(fx); fy = comm.allreduce_max(fy);
    check("medium: deck block (permittivity -> density 10), sub-cell volume fraction at the edge (1/4 at Jx, 3/4 at Jy)",
          dens && std::abs(fx - 0.25) < 1e-14 && std::abs(fy - 0.75) < 1e-14, fmt("Jx %.4f", fx) + fmt(", Jy %.4f", fy));
    (void)k0;
    int rejected = 0;
    Deck dk = Deck::read("medium_tmp.deck");
    for (const char* o : {"numerics.solver=rip_cc", "numerics.closure=o5", "medium glass.resonance=0", "medium glass.model=drude",
                          "medium glass.supersample=0"}) {
        Deck x = dk;
        x.override_from(o);
        try { Config::from_deck(x); } catch (const std::exception&) { ++rejected; }
    }
    check("medium: rip_cc, o5, resonance 0, drude with permittivity, supersample 0 rejected", rejected == 5, fmt("rejected %.0f of 5", rejected));
    if (comm.rank == 0) std::remove("medium_tmp.deck");
}

// L6. The Gauss law with particles plus media (the medium charge -div P): a hot plasma half overlapping a 50 n_c Drude layer
//     (omega_p dt = 2.2 in the medium) and a Lorentz layer, 3D, shape 2; the explicit mover, then the implicit one (O6), where
//     W + K_particles + K_medium + dissipated is conserved to the solve tolerances.
static void medium_gauss_particles(Comm& comm) {
    for (int implicit = 0; implicit < 2; ++implicit) {
        const std::string t =
            std::string("grid {\n lambda0 = 1 um\n nx = 16\n ny = 8\n nz = 6\n dx = 0.05 um\n dy = 0.04 um\n dz = 0.05 um\n t_end = 3 fs\n}\n") +
            "numerics {\n solver = irip\n shape = 2\n" + (implicit ? " mover = implicit\n mover_tol = 1e-12\n mover_maxit = 200\n" : "") +
            " medium_tol = 1e-12\n}\n"
            "species e {\n charge = -1\n mass = 1\n density = 2\n xmax = 0.5 um\n ppc_x = 2\n ppc_y = 2\n ppc_z = 2\n"
            " uth_x = 0.2\n uth_y = 0.2\n uth_z = 0.2\n}\n"
            "species i {\n charge = 1\n mass = 1836\n density = 2\n xmax = 0.5 um\n ppc_x = 2\n ppc_y = 2\n ppc_z = 2\n mobile = false\n}\n"
            "medium {\n density = 50\n collision_frequency = 0.3\n profile = x > 0.3 um && x < 0.62 um\n}\n"
            "medium glass {\n model = lorentz\n density = 4\n resonance = 2\n profile = x > 0.62 um && y < 0.2 um\n}\n"
            "output {\n dir = medium_tmp_out\n}\n";
        Config c = deck_config(t, "medium_tmp.deck");
        Simulation sim(c, comm);
        sim.init();
        const double e0 = sim_total(sim, comm);
        double g = 0, dev = 0;
        while (!sim.finished()) {
            sim.step();
            g = std::max(g, sim.gauss_drift());
            dev = std::max(dev, std::abs(sim_total(sim, comm) / e0 - 1));
        }
        const MediumStats& ms = sim.medium->stats;
        std::string info = fmt("Gauss %.1e", g) + fmt(", GMRES %.1f per solve", double(ms.iterations) / ms.solves);
        bool ok = g < 1e-12 && ms.unconverged == 0;
        if (implicit) {
            const MoverStats& st = implicit_mover_stats(sim);
            info += fmt(", max |E/E0 - 1| %.1e", dev) + fmt(", mover %.1f iterations per step", double(st.iterations) / st.steps);
            ok = ok && dev < 1e-9 && st.unconverged == 0;
        }
        check(std::string("medium: Gauss law with particles plus media, 3D, ") + (implicit ? "implicit mover (W + K conserved)" : "explicit mover"),
              ok, info);
    }
    if (comm.rank == 0) std::remove("medium_tmp.deck");
}

// L7. RZ per mode: random fields in modes 0..2 and random axisymmetric media (wp dt <= 10, collisions, Lorentz) on a stretched
//     radial grid at dz = 2.5 dr: W + K + dissipated constant to the solve tolerance; then the Gauss law with a plasma column
//     and a medium (deck, m = 0, 1, explicit mover).
static void medium_rz(Comm& comm) {
    const int M = 3;
    Domain d = rz_domain(comm, 10, 14, M, 0.4, 0.16, true, Solver::IRip);
    FieldsRZ F(d, M, rz_radial_grid(14, 0.16, 0.8, 1.15));
    std::mt19937_64 rng(17);
    std::normal_distribution<double> nd;
    std::uniform_real_distribution<double> ud;
    for (Field* f : {&F.ez, &F.er, &F.ep, &F.bz, &F.br, &F.bp}) {
        H h(*f);
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    double v = nd(rng);
                    if (j == 0 && ((f == &F.ez && k > 0) || f == &F.ep || f == &F.br)) v = 0;
                    h(i, j, k) = v;
                }
        h.put(*f);
    }
    comm.exchange(d, {&F.ez, &F.er, &F.ep, &F.bz, &F.br, &F.bp});
    Medium Md;
    Md.resize(d);
    Md.rz = true;
    Md.tol = 1e-13;
    const RadialGrid& g = F.grid();
    Md.weights.resize(size_t(3 * d.n[1]));
    for (int j = 0; j < d.n[1]; ++j) {
        Md.weights[size_t(j)] = Md.weights[size_t(2 * d.n[1] + j)] = 2 * M_PI * g.A[size_t(j)] * d.h[0];
        Md.weights[size_t(d.n[1] + j)] = 2 * M_PI * g.B[size_t(j)] * d.h[0];
    }
    const double D = d.dt();
    for (int c = 0; c < 3; ++c) {
        H w(Md.wp2[c]), o(Md.w02[c]), n(Md.nu[c]);
        for (int j = 0; j < d.n[1]; ++j)
            for (int i = 0; i < d.n[0]; ++i) {
                double a = 0, b = 0, e = 0;
                if (ud(rng) < 0.6 && !(c == 2 && j == 0)) {
                    a = 100 / (D * D) * ud(rng);
                    b = ud(rng) < 0.5 ? 4 / (D * D) : 0;
                    e = ud(rng) < 0.5 ? 0.5 / D : 0;
                }
                for (int k = 0; k < d.n[2]; ++k) { w(i, j, k) = a; o(i, j, k) = b; n(i, j, k) = e; }
            }
        w.put(Md.wp2[c]); o.put(Md.w02[c]); n.put(Md.nu[c]);
    }
    F.medium = &Md;
    auto tot = [&]() {
        double v[2] = {F.energy(), Md.energy(d, &Md.weights)};
        comm.allreduce_sum(v, 2);
        return v[0] + v[1] + Md.dissipated;
    };
    const double e0 = tot();
    double worst = 0;
    for (int n = 0; n < 30; ++n) {
        F.step(comm);
        worst = std::max(worst, std::abs(tot() / e0 - 1));
    }
    check("medium RZ: W + K + dissipated, random fields in modes 0..2, random media (wp dt <= 10), stretched grid, dz = 2.5 dr",
          worst < 1e-10 && Md.stats.unconverged == 0,
          fmt("max |E/E0 - 1| %.1e", worst) + fmt(", dissipated %.3f E0", Md.dissipated / e0) +
              fmt(", GMRES %.1f per step", double(Md.stats.iterations) / Md.stats.solves));
    // the Gauss law with particles and a medium (m = 0, 1)
    const std::string t =
        "grid {\n geometry = rz\n modes = 2\n lambda0 = 1 um\n nz = 16\n nr = 12\n dz = 0.1 um\n dr = 0.05 um\n t_end = 3 fs\n}\n"
        "numerics {\n solver = irip\n shape = 2\n medium_tol = 1e-12\n}\n"
        "species e {\n charge = -1\n mass = 1\n density = 2\n rmax = 0.3 um\n zmax = 0.9 um\n ppc_r = 4\n ppc_theta = 8\n"
        " uth_z = 0.2\n uth_x = 0.2\n uth_y = 0.2\n}\n"
        "species i {\n charge = 1\n mass = 1836\n density = 2\n rmax = 0.3 um\n zmax = 0.9 um\n ppc_r = 4\n ppc_theta = 8\n mobile = false\n}\n"
        "medium {\n density = 30\n collision_frequency = 0.2\n profile = z > 0.6 um && z < 1.2 um && r < 0.4 um\n}\n"
        "output {\n dir = medium_tmp_out\n}\n";
    Config c = deck_config(t, "medium_tmp.deck");
    Simulation sim(c, comm);
    sim.init();
    double gd = 0;
    while (!sim.finished()) {
        sim.step();
        gd = std::max(gd, sim.gauss_drift());
    }
    check("medium RZ: Gauss law with a plasma column plus a medium, m = 0, 1", gd < 1e-12 && sim.medium->stats.unconverged == 0,
          fmt("Gauss %.1e", gd) + fmt(", GMRES %.1f per solve", double(sim.medium->stats.iterations) / sim.medium->stats.solves));
    if (comm.rank == 0) std::remove("medium_tmp.deck");
}

// L8. A parabolic plasma mirror (100 n_c, 2D, s-pol; omega_p dt = 5.2 / 3.9 at lambda/12 / 16): a wide pulse along +x (2D Gaussian
//     beam, waist 3 lambda at x_b, envelope 1.5 lambda) focused by x = x_v - y^2 / (4 f), f = 4 lambda (NA 0.75). The focus is the
//     peak of the time-largest envelope of Ez of the reflected pulse. Reference: physical optics (Kirchhoff) of a perfectly reflecting
//     parabola (the 100 n_c surface reflects with |r| = 1 and a skin depth of 0.016 lambda), s-pol (Dirichlet):
//       E_r(P, k) = (k/2) int E_inc(S, k) H0(k |P - S|) dy_S    ((k^.n) ds = dy on the parabola),
//     E_inc the 2D Gaussian beam propagated from x_b to the surface (amplitude sqrt(w0/w), curvature, half the Gouy phase) for each
//     k of the pulse spectrum; the time-largest |sum_k E_r e^{-i k t}|^2 on the axis. Physical optics: -0.037 lambda from the
//     geometric focus (the beam diverges: the collimated amplitude alone gives +0.135). The simulation converges with second order
//     (-0.353, -0.223, -0.143 lambda at lambda/12, 16, 24): the Richardson value of lambda/12 and 16 within lambda/20 of physical
//     optics, the error at lambda/16 below that at lambda/12, and the focus on the axis.
namespace {
struct Focus { double x, y, its; };
Focus parabolic_sim(Comm& comm, int N) {
    const double lam = 2 * M_PI, dx = lam / N, f = 4 * lam, xv = 14 * lam, yc = 7 * lam, xb = 6 * lam;
    const int nx = int(std::round(17 * lam / dx)), ny = int(std::round(14 * lam / dx));
    Domain d;
    d.N[0] = nx; d.N[1] = ny; d.N[2] = 1;
    d.h[0] = dx; d.h[1] = dx; d.h[2] = 1;
    d.flat = true; d.periodic[0] = false; d.ng = 3; d.solver = Solver::IRip;
    const int want[3] = {0, 1, 1};
    comm.decompose(d, want);
    Fields F(d);
    Medium M;
    MediumConfig m;
    m.density = 100;
    char t[160];
    std::snprintf(t, sizeof t, "x > %.17g - (y - %.17g)^2 / %.17g", xv, yc, 4 * f);
    m.profile = Expr(t, {"x", "y", "z"}, [](const std::string&, double&) { return false; });
    medium_build(M, d, {m}, 1);
    M.tol = 1e-8;
    {
        H ez(F.ez), by(F.by);
        for (int j = 0; j < d.n[1]; ++j)
            for (int i = 0; i < d.n[0]; ++i) {
                const double x = d.lo(0) + i * dx, y = j * dx;
                const double g = std::exp(-std::pow((x - xb) / (1.5 * lam), 2) - std::pow((y - yc) / (3 * lam), 2)) * std::cos(x - xb);
                ez(i, j, 0) = g; by(i, j, 0) = -g;                     // along +x: By = -Ez
            }
        ez.put(F.ez); by.put(F.by);
    }
    comm.exchange(d, {&F.ez, &F.by});
    F.medium = &M;
    // the envelope of Ez per node, Ez_n^2 + ((Ez_{n+1} - Ez_{n-1}) / (2 sin(omega0 dt)))^2 (exact for a sinusoid; the plain max of
    // Ez^2 over the steps depends on the sampling of the carrier), its largest value over the reflected pulse
    std::vector<double> peak(size_t(d.n[0]) * d.n[1], 0.0), em, e0;
    const int steps = int(std::round(16 * lam / dx));
    const double sw = 2 * std::sin(dx);
    for (int n = 0; n < steps; ++n) {
        F.step(comm);
        std::vector<double> e = F.ez.host();
        if (n * dx >= 9 * lam && !em.empty())                           // the reflected pulse only (the incident one has passed)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    const size_t q = F.ez.idx(i, j, 0);
                    const double a = e0[q], b = (e[q] - em[q]) / sw;
                    double& p = peak[size_t(j) * d.n[0] + i];
                    p = std::max(p, a * a + b * b);
                }
        em = std::move(e0);
        e0 = std::move(e);
    }
    // the maximum in front of the mirror (x < x_v - 1.5 lambda), refined by a parabola through the neighbours per axis
    auto ref = [](double a, double b, double c) { const double q = a - 2 * b + c; return q < 0 ? 0.5 * (a - c) / q : 0.0; };
    double best = 0;
    Focus r{0, 0, double(M.stats.iterations) / M.stats.solves};
    for (int j = 1; j + 1 < d.n[1]; ++j)
        for (int i = 1; i + 1 < d.n[0]; ++i) {
            const double x = d.lo(0) + i * dx;
            if (x > xv - 1.5 * lam) continue;
            const double p = peak[size_t(j) * d.n[0] + i];
            if (p <= best) continue;
            best = p;
            r.x = (x + dx * ref(peak[size_t(j) * d.n[0] + i - 1], p, peak[size_t(j) * d.n[0] + i + 1]) - (xv - f)) / lam;
            r.y = (j * dx + dx * ref(peak[size_t(j - 1) * d.n[0] + i], p, peak[size_t(j + 1) * d.n[0] + i]) - yc) / lam;
        }
    return r;
}
// physical optics (above): the focus on the axis from the geometric one, in lambda
double parabolic_po() {
    const double lam = 2 * M_PI, f = 4 * lam, xv = 14 * lam, xb = 6 * lam, w0 = 3 * lam;
    auto H0 = [](double x) {                                            // Hankel H0^(1), asymptotic (x > 15: error < 1e-5)
        const std::complex<double> c(1.0, -1.0 / (8 * x));
        return std::sqrt(2 / (M_PI * x)) * std::exp(std::complex<double>(0, x - M_PI / 4)) * (c - 9.0 / (128 * x * x));
    };
    auto ref = [](double a, double b, double c) { const double q = a - 2 * b + c; return q < 0 ? 0.5 * (a - c) / q : 0.0; };
    const int nk = 41, ns = 1121;
    std::vector<double> ks(static_cast<size_t>(nk)), ak(static_cast<size_t>(nk));
    for (int q = 0; q < nk; ++q) {
        ks[size_t(q)] = 0.6 + 0.8 * q / (nk - 1);
        ak[size_t(q)] = std::exp(-std::pow((ks[size_t(q)] - 1) * 1.5 * lam / 2, 2));
    }
    std::vector<double> prof;
    const double xa = xv - f - 1.5 * lam, dxa = lam / 40;
    for (int ip = 0; ip <= 90; ++ip) {
        const double xp = xa + ip * dxa;
        std::vector<std::complex<double>> Ek(static_cast<size_t>(nk));
        for (int s = 0; s < ns; ++s) {
            const double u = -7 * lam + 14 * lam * s / (ns - 1), xs = xv - u * u / (4 * f), R = std::hypot(xp - xs, u);
            const double tw = s == 0 || s == ns - 1 ? 0.5 : 1.0, z = xs - xb;
            for (int q = 0; q < nk; ++q) {
                const double k = ks[size_t(q)], zr = k * w0 * w0 / 2, w = w0 * std::sqrt(1 + z * z / (zr * zr));
                const double iR = z / (z * z + zr * zr), psi = std::atan(z / zr);
                const std::complex<double> inc = std::sqrt(w0 / w) * std::exp(-u * u / (w * w)) *
                                                 std::exp(std::complex<double>(0, k * z + 0.5 * k * u * u * iR - 0.5 * psi));
                Ek[size_t(q)] += tw * 0.5 * k * ak[size_t(q)] * inc * H0(k * R);
            }
        }
        double mx = 0;
        const double t0 = (xv - xb) + (xv - xp);
        for (int it = -400; it <= 400; ++it) {
            const double tt = t0 + it * lam / 100;
            std::complex<double> s = 0;
            for (int q = 0; q < nk; ++q) s += Ek[size_t(q)] * std::exp(std::complex<double>(0, -ks[size_t(q)] * tt));
            mx = std::max(mx, std::norm(s));
        }
        prof.push_back(mx);
    }
    double pbest = 0, px = 0;
    for (size_t ip = 1; ip + 1 < prof.size(); ++ip)
        if (prof[ip] > pbest) { pbest = prof[ip]; px = xa + ip * dxa + dxa * ref(prof[ip - 1], prof[ip], prof[ip + 1]); }
    return (px - (xv - f)) / lam;
}
}  // namespace

static void medium_parabolic(Comm& comm) {
    const double po = parabolic_po();
    const Focus a = parabolic_sim(comm, 12), b = parabolic_sim(comm, 16);
    const double rich = b.x + (b.x - a.x) / (256.0 / 144.0 - 1);     // x(N) = x_inf + C / N^2
    const bool ok = std::abs(rich - po) < 0.05 && std::abs(b.x - po) < std::abs(a.x - po) && std::abs(a.y) < 0.01 && std::abs(b.y) < 0.01;
    check("medium: parabolic plasma mirror, 100 n_c, f = 4 lambda: focus against physical optics (" + fmt("%.3f lambda)", po), ok,
          fmt("lambda/12 %.3f", a.x) + fmt(", lambda/16 %.3f", b.x) + fmt(", Richardson %.3f lambda", rich) +
              fmt(" from the geometric focus; y - y_c %.1e", std::max(std::abs(a.y), std::abs(b.y))) +
              fmt(" lambda; GMRES %.1f", a.its) + fmt(" / %.1f per step", b.its));
}

// L9. The block metal with numerics.solver = irip (decision 124): the conductor is a Drude medium with omega_p dt = 10 by
//     default. (a) the deck: one medium named metal with that density; particles = absorb rejected. (b) a pulse at normal
//     incidence on a metal slab comes back whole, the field energy deeper than two cells inside stays small. (c) a hot plasma
//     crossing a metal slab, 3D, shape 2: the Gauss law (with the medium charge) at round-off, no particle lost.
static void metal_irip(Comm& comm) {
    {
        const double um = 2 * M_PI;                               // 1 um in code units (lambda0 = 1 um)
        const double x0 = 7 * um, xs = 14 * um, w = 2 * um;
        const std::string t =
            "grid {\n geometry = cartesian2d\n lambda0 = 1 um\n nx = 420\n ny = 4\n dx = 0.05 um\n dy = 0.05 um\n t_end = 100 fs\n}\n"
            "numerics {\n solver = irip\n medium_tol = 1e-12\n}\n"
            "metal {\n function = x - 14 um\n}\n"
            "output {\n dir = medium_tmp_out\n}\n";
        Config c = deck_config(t, "medium_tmp.deck");
        const double wd = 10.0 / c.dt();
        const bool cfg_ok = c.media.size() == 1 && c.media[0].name == "metal" && std::abs(c.media[0].density / (wd * wd) - 1) < 1e-12 &&
                            c.metal.medium && !c.metal.on;
        bool rejected = false;                         // particles = absorb: accepted in Cart3D / 2D (L9e)
        {
            Deck x = Deck::read("medium_tmp.deck");
            x.override_from("metal.particles=absorb");
            try {
                const Config ca = Config::from_deck(x);
                rejected = ca.metal.medium && ca.metal.absorb && ca.media.size() == 1;
            } catch (const std::exception&) { rejected = false; }
        }
        check("metal in irip L9a: the block becomes a Drude medium with omega_p dt = 10; particles = absorb accepted", cfg_ok && rejected,
              fmt("density %.4g n_c", c.media.empty() ? 0.0 : c.media[0].density));
        Simulation sim(c, comm);
        sim.init();
        const Domain& d = sim.dom;
        {
            H ey(sim.fields->ey), bz(sim.fields->bz);
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    const double x = d.lo(0) + i * d.h[0], g = std::exp(-std::pow((x - x0) / w, 2)) * std::cos(x - x0);
                    ey(i, j, 0) = g; bz(i, j, 0) = g;
                }
            ey.put(sim.fields->ey); bz.put(sim.fields->bz);
        }
        comm.exchange(d, {&sim.fields->ey, &sim.fields->bz});
        auto energy = [&](double xa, double xb) {                // field energy of the owned cells with xa <= x < xb
            double s = 0;
            for (const Field* f : {&sim.fields->ex, &sim.fields->ey, &sim.fields->ez, &sim.fields->bx, &sim.fields->by, &sim.fields->bz}) {
                const std::vector<double> h = f->host();
                for (int j = 0; j < d.n[1]; ++j)
                    for (int i = 0; i < d.n[0]; ++i) {
                        const double x = d.lo(0) + i * d.h[0];
                        if (x >= xa && x < xb) s += h[f->idx(i, j, 0)] * h[f->idx(i, j, 0)];
                    }
            }
            comm.allreduce_sum(&s, 1);
            return s;
        };
        const double w0 = energy(-1e300, 1e300);
        const int steps = int(std::round(2 * (xs - x0) / d.h[0]));
        double deep = 0;
        for (int n = 0; n < steps; ++n) {
            sim.step();
            deep = std::max(deep, energy(xs + 2.5 * d.h[0], 20 * um) / w0);
        }
        const double R = energy(-1e300, xs) / w0;
        check("metal in irip L9b: a pulse at normal incidence on the metal comes back whole; no field deeper than two cells inside",
              std::abs(R - 1) < 1e-6 && deep < 1e-3 && sim.medium->stats.unconverged == 0,
              fmt("|R - 1| %.1e", std::abs(R - 1)) + fmt(", energy deeper than two cells <= %.1e W0", deep) +
                  fmt(", GMRES %.1f per step", double(sim.medium->stats.iterations) / sim.medium->stats.solves));
    }
    {
        const std::string t =
            "grid {\n lambda0 = 1 um\n nx = 16\n ny = 8\n nz = 6\n dx = 0.05 um\n dy = 0.04 um\n dz = 0.05 um\n t_end = 3 fs\n}\n"
            "numerics {\n solver = irip\n shape = 2\n medium_tol = 1e-12\n}\n"
            "species e {\n charge = -1\n mass = 1\n density = 2\n xmax = 0.5 um\n ppc_x = 2\n ppc_y = 2\n ppc_z = 2\n"
            " uth_x = 0.2\n uth_y = 0.2\n uth_z = 0.2\n}\n"
            "species i {\n charge = 1\n mass = 1836\n density = 2\n xmax = 0.5 um\n ppc_x = 2\n ppc_y = 2\n ppc_z = 2\n mobile = false\n}\n"
            "metal {\n function = (x - 0.3 um) * (0.62 um - x)\n}\n"
            "output {\n dir = medium_tmp_out\n}\n";
        Config c = deck_config(t, "medium_tmp.deck");
        Simulation sim(c, comm);
        sim.init();
        auto count = [&] {
            double n = 0;
            for (const Species& sp : sim.species) n += double(sp.size());
            comm.allreduce_sum(&n, 1);
            return n;
        };
        const double n0 = count();
        double g = 0;
        while (!sim.finished()) {
            sim.step();
            g = std::max(g, sim.gauss_drift());
        }
        const double n1 = count();
        const MediumStats& ms = sim.medium->stats;
        check("metal in irip L9c: a hot plasma crossing a metal slab (omega_p dt = 10), 3D: Gauss law, no particle lost",
              g < 1e-12 && n1 == n0 && ms.unconverged == 0,
              fmt("Gauss %.1e", g) + fmt(", particles %.0f -> %.0f", n0, n1) + fmt(", GMRES %.1f per solve", double(ms.iterations) / ms.solves));
        // (e) the same with particles = absorb, 3D and 2D: the electrons that enter the slab are removed, their charge stays at rest
        //     (Simulation::rho_abs_): the Gauss law still at round-off, and the total charge on the grid (particles + absorbed +
        //     medium) stays 0 although the electrons that are left no longer balance the ions
        for (int flat = 0; flat < 2; ++flat) {
            std::string ta = t;                        // 2D: no z cells, no ppc_z
            auto cut = [&](const std::string& a, const std::string& b) {
                for (size_t p = ta.find(a); p != std::string::npos; p = ta.find(a, p + b.size())) ta.replace(p, a.size(), b);
            };
            cut("metal {\n", "metal {\n particles = absorb\n");
            if (flat) { cut(" nz = 6\n", ""); cut(" dz = 0.05 um\n", ""); cut(" ppc_z = 2\n", ""); cut("grid {\n", "grid {\n geometry = cartesian2d\n"); }
            Config ca = deck_config(ta, "medium_tmp.deck");
            Simulation sa(ca, comm);
            sa.init();
            auto cnt = [&](int q) {
                double n = double(sa.species[size_t(q)].size());
                comm.allreduce_sum(&n, 1);
                return n;
            };
            auto qsum = [&](double& tot, double& mx) {       // sum and max |rho| over the owned cells (after gauss_drift)
                const std::vector<double> h = sa.fields->rho.host();
                tot = 0; mx = 0;
                for (int k = 0; k < sa.dom.n[2]; ++k)
                    for (int j = 0; j < sa.dom.n[1]; ++j)
                        for (int i = 0; i < sa.dom.n[0]; ++i) {
                            const double v = h[sa.fields->rho.idx(i, j, k)];
                            tot += v; mx = std::max(mx, std::abs(v));
                        }
                comm.allreduce_sum(&tot, 1);
                mx = comm.allreduce_max(mx);
            };
            const double e0 = cnt(0), i0 = cnt(1);
            double ga = 0, tot = 0, mx = 0, worst = 0;
            while (!sa.finished()) {
                sa.step();
                ga = std::max(ga, sa.gauss_drift());
                qsum(tot, mx);
                worst = std::max(worst, std::abs(tot));
            }
            const double e1 = cnt(0), i1 = cnt(1);
            const double cells = double(sa.dom.N[0]) * sa.dom.N[1] * sa.dom.N[2];
            check(std::string("metal in irip L9e (") + (flat ? "2D" : "3D") + "): particles = absorb: electrons removed in the slab, their charge "
                      "stays at rest: Gauss law, total charge on the grid",
                  ga < 1e-12 && e1 < e0 && i1 == i0 && worst / (cells * std::max(mx, 1e-300)) < 1e-12 && sa.medium->stats.unconverged == 0,
                  fmt("Gauss %.1e", ga) + fmt(", electrons %.0f -> %.0f", e0, e1) + fmt(", ions %.0f -> %.0f", i0, i1) +
                      fmt(", |sum rho| / (cells max |rho|) %.1e", worst / (cells * std::max(mx, 1e-300))) + fmt(" (max |rho| %.2g)", mx));
        }
    }
    // (d) RZ (irip, modes 0, 1): the rings block and an axisymmetric metal function as media; a plasma column through them: the
    //     Gauss law; the coefficient inside a ring and in a gap; a function of phi rejected
    for (int v = 0; v < 2; ++v) {
        const std::string t =
            std::string("grid {\n geometry = rz\n modes = 2\n lambda0 = 1 um\n nz = 16\n nr = 12\n dz = 0.1 um\n dr = 0.05 um\n t_end = 3 fs\n}\n"
                        "numerics {\n solver = irip\n shape = 2\n medium_tol = 1e-12\n}\n"
                        "species e {\n charge = -1\n mass = 1\n density = 2\n rmax = 0.3 um\n zmax = 0.9 um\n ppc_r = 4\n ppc_theta = 8\n"
                        " uth_z = 0.2\n uth_x = 0.2\n uth_y = 0.2\n}\n"
                        "species i {\n charge = 1\n mass = 1836\n density = 2\n rmax = 0.3 um\n zmax = 0.9 um\n ppc_r = 4\n ppc_theta = 8\n mobile = false\n}\n") +
            (v == 0 ? "rings {\n r_in = 0.2 um\n r_out = 0.45 um\n period = 0.6 um\n gap = 0.2 um\n z_start = 0.2 um\n z_end = 1.4 um\n}\n"
                    : "metal {\n function = min(min(r - 0.2 um, 0.45 um - r), min(z - 0.2 um, 0.6 um - z))\n}\n") +
            "output {\n dir = medium_tmp_out\n}\n";
        Config c = deck_config(t, "medium_tmp.deck");
        bool rejected = false;                         // RZ: a function of phi; particles = absorb (not in RZ irip yet)
        {
            Deck x = Deck::read("medium_tmp.deck");
            x.override_from(v == 1 ? "metal.function=min(r - 0.2 um, 0.45 um - r) * cos(2 * phi)" : "rings.particles=absorb");
            try { Config::from_deck(x); } catch (const std::exception&) { rejected = true; }
        }
        Simulation sim(c, comm);
        sim.init();
        const Domain& d = sim.dom;
        double in = -1, out = -1;                     // Jz at (z, r) = (0.4, 0.3) um: inside; (0.7, 0.3) um: the gap (v = 1: vacuum)
        {
            H w(sim.medium->wp2[0]);
            const int i4 = 4 - d.off[0], i7 = 7 - d.off[0];
            if (i4 >= 0 && i4 < d.n[0]) in = w(i4, 6, 0);
            if (i7 >= 0 && i7 < d.n[0]) out = w(i7, 6, 0);
        }
        in = comm.allreduce_max(in); out = comm.allreduce_max(out);
        const double wd = 10.0 / c.dt();
        double g = 0;
        while (!sim.finished()) {
            sim.step();
            g = std::max(g, sim.gauss_drift());
        }
        check(std::string("metal in irip L9d (RZ, modes 0, 1): ") + (v == 0 ? "rings" : "an axisymmetric metal function; a function of phi rejected") +
                  ": omega_p dt = 10 inside, 0 in the gap, Gauss law with a plasma column through it",
              c.media.size() == 1 && std::abs(in / (wd * wd) - 1) < 1e-12 && out == 0 && g < 1e-12 && rejected && sim.medium->stats.unconverged == 0,
              fmt("wp^2 inside %.4g, in the gap %.1g", in, out) + fmt(", Gauss %.1e", g) +
                  fmt(", GMRES %.1f per solve", double(sim.medium->stats.iterations) / sim.medium->stats.solves));
    }
    if (comm.rank == 0) std::remove("medium_tmp.deck");
}

void test_medium2(Comm& comm) {
    medium_deck(comm);
    medium_gauss_particles(comm);
    metal_irip(comm);
    medium_rz(comm);
    if (comm.size > 1) return;
    medium_parabolic(comm);
}
