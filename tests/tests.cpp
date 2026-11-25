// Checks against exact or analytic references (serial build); more in tests/test_*.cpp.
#include "env.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "sim.hpp"
#include "geometry.hpp"
#include "stagger.hpp"

#include "common.hpp"

int failures = 0;
void check(const std::string& name, bool ok, const std::string& info) {
    std::printf("%-58s %s  %s\n", name.c_str(), ok ? "ok  " : "FAIL", info.c_str());
    if (!ok) ++failures;
}
std::string fmt(const char* f, double a, double b) {
    char s[160]; std::snprintf(s, sizeof s, f, a, b); return s;
}

static Domain box(int nx, int ny, int nz, double hx, double hy, double hz, bool px = true) {
    Domain d;
    d.N[0] = nx; d.N[1] = ny; d.N[2] = nz;
    d.h[0] = hx; d.h[1] = hy; d.h[2] = hz;
    d.periodic[0] = px;
    d.ng = 3;
    return d;
}

// 0. Reductions (portable.hpp; on the GPU block-wise partials combined on the host): sum and max|.| of
//    values with a known exact result, for sizes below one block, not a multiple of the block, and
//    above the 1024 x 256 grid of the GPU reduction (grid-stride loop).
static void test_reductions() {
    double worst = 0;
    for (long n : {1L, 37L, 1000003L, 3000000L}) {
        Buf<double> b{size_t(n)};
        double* p = b.data();
        par_for(n, [=] HD(long i) { p[i] = (i % 7 == 3) ? -0.5 : 0.25; });   // exactly representable
        long n3 = 0;
        for (long i = 0; i < n; ++i) n3 += i % 7 == 3;
        const double want = 0.25 * double(n - n3) - 0.5 * double(n3);
        const double s = par_sum(n, [=] HD(long i) { return p[i]; });
        const double m = par_max_abs(n, [=] HD(long i) { return p[i]; });
        worst = std::max(worst, std::abs(s - want) + std::abs(m - (n3 ? 0.5 : 0.25)));
    }
    check("reductions: par_sum and par_max_abs exact", worst == 0, fmt("max error %.1e", worst));
}

// The fixed-point add of the box deposit (two 32-bit adds with carry, decision 26): many threads of
// one block add random int64 (both signs, low words that carry) into 4 shared cells; the result is the
// exact 64-bit sum. On the GPU this runs the native shared atomics concurrently.
static void test_block_add() {
    const long n = 20000;
    std::vector<long long> v(static_cast<size_t>(n));
    std::mt19937_64 rng(7);
    unsigned long long want[4] = {0, 0, 0, 0};
    for (long i = 0; i < n; ++i) {
        v[size_t(i)] = (long long)(rng() >> (i % 3 == 0 ? 1 : 20)) * (i % 2 ? -1 : 1);
        want[i % 4] += (unsigned long long)v[size_t(i)];
    }
    Buf<long long> dv{size_t(n)}, out{4};
    dv.from_host(v.data(), 0, size_t(n));
    const long long* V = dv.data();
    long long* O = out.data();
    par_blocks(1, 4,
        [=] HD(long, int tid, int nt, double* s) {
            for (long i = tid; i < n; i += nt) block_add(reinterpret_cast<long long*>(s) + i % 4, V[i]);
        },
        [=] HD(long, int tid, int nt, double* s) {
            for (int c = tid; c < 4; c += nt) O[c] = reinterpret_cast<const long long*>(s)[c];
        });
    long long got[4];
    out.to_host(got, 0, 4);
    int bad = 0;
    for (int c = 0; c < 4; ++c) bad += (unsigned long long)got[c] != want[c];
    check("block_add: 64-bit sums from 32-bit adds with carry, exact", bad == 0, fmt("%.0f wrong cells", bad));
}

// 1. A plane wave along +x (Ey, Bz) and one along -x (Ez, By) move exactly one cell per step.
static void test_vacuum_shift(Comm& comm) {
    Domain d = box(64, 6, 5, 0.2, 0.5, 0.5);
    const int want[3] = {1, 1, 1};
    comm.decompose(d, want);
    Fields f(d);
    auto prof = [](double i) { return std::exp(-0.02 * (i - 20) * (i - 20)) * std::sin(0.9 * i); };
    H hey(f.ey), hbz(f.bz), hez(f.ez), hby(f.by);
    for (int k = -3; k < d.n[2] + 3; ++k)
        for (int j = -3; j < d.n[1] + 3; ++j)
            for (int i = -3; i < d.n[0] + 3; ++i) {
                const int ii = ((i % 64) + 64) % 64;
                hey(i, j, k) = hbz(i, j, k) = prof(ii);
                hez(i, j, k) = hby(i, j, k) = 2 * prof(ii);
            }
    hey.put(f.ey); hbz.put(f.bz); hez.put(f.ez); hby.put(f.by);
    const int steps = 25;
    for (int n = 0; n < steps; ++n) { f.zero_currents(); f.step(comm); }
    double e1 = 0, e2 = 0;
    H fey(f.ey), fez(f.ez);
    for (int i = 0; i < 64; ++i) {
        e1 = std::max(e1, std::abs(fey(i, 2, 2) - prof((i - steps + 64) % 64)));
        e2 = std::max(e2, std::abs(fez(i, 2, 2) - 2 * prof((i + steps) % 64)));
    }
    check("vacuum: +x wave (Ey,Bz) shifted exactly 1 cell/step", e1 < 1e-14, fmt("max err %.1e", e1));
    check("vacuum: -x wave (Ez,By) shifted exactly 1 cell/step", e2 < 1e-14, fmt("max err %.1e", e2));
}

// 2. Random vacuum fields: energy stays bounded, div B stays at its initial value (here: of random data).
static void test_vacuum_random(Comm& comm) {
    Domain d = box(16, 12, 10, 0.3, 0.5, 0.6);
    const int want[3] = {1, 1, 1};
    comm.decompose(d, want);
    Fields f(d);
    std::mt19937_64 rng(3);
    std::normal_distribution<double> g;
    for (Field* F : {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz}) {
        H h(*F);
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) h(i, j, k) = g(rng);
        h.put(*F);
    }
    comm.exchange(d, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
    // div B of random data is not zero; store it and check that it does not change
    auto divb = [&](std::vector<double>& out) {
        out.clear();
        H bx(f.bx), by(f.by), bz(f.bz);
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i)
                    out.push_back((bx(i + 1, j, k) - bx(i, j, k)) / d.h[0] +
                                  0.5 * (by(i, j + 1, k) + by(i + 1, j + 1, k) - by(i, j, k) - by(i + 1, j, k)) / d.h[1] +
                                  0.5 * (bz(i, j, k + 1) + bz(i + 1, j, k + 1) - bz(i, j, k) - bz(i + 1, j, k)) / d.h[2]);
    };
    std::vector<double> b0, b1;
    divb(b0);
    const double w0 = f.energy();
    double wmax = w0;
    for (int n = 0; n < 2000; ++n) {
        f.zero_currents(); f.step(comm);
        wmax = std::max(wmax, f.energy());
    }
    divb(b1);
    double e = 0;
    for (size_t q = 0; q < b0.size(); ++q) e = std::max(e, std::abs(b1[q] - b0[q]));
    check("vacuum random fields: energy bounded over 2000 steps", wmax / w0 < 1.2, fmt("max W/W0 = %.4f", wmax / w0));
    check("vacuum random fields: div B unchanged", e < 1e-11, fmt("max change %.1e", e));
}

// 3. Plasma: Gauss law and div B conserved to round-off with moving, hot particles (both shapes,
//    neutral and non-neutral, with and without a drift crossing the periodic boundary).
static Config plasma_config(int order, bool neutral, double drift) {
    Config c;
    c.N[0] = 12; c.N[1] = 10; c.N[2] = 8;
    c.h[0] = 0.3; c.h[1] = 0.5; c.h[2] = 0.6;
    c.t_end = 60 * c.h[0];
    c.order = order;
    SpeciesConfig e;
    e.name = "e"; e.q = -1; e.m = 1; e.density = 0.5;
    e.ppc[0] = e.ppc[1] = e.ppc[2] = 2;
    for (int a = 0; a < 3; ++a) { e.uth[a] = 0.3; e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
    e.u0[0] = drift;
    c.species.push_back(e);
    if (neutral) {
        SpeciesConfig p = e;
        p.name = "p"; p.q = 1; p.m = 10; p.uth[0] = p.uth[1] = p.uth[2] = 0.05;
        c.species.push_back(p);
    } else {
        c.species[0].rmin[0] = 0.5; c.species[0].rmax[0] = 2.0;     // a slab: rho != 0 initially
    }
    c.check();
    return c;
}

static void test_plasma_conservation(Comm& comm) {
    for (int order : {1, 2, 3})
        for (int neutral : {1, 0}) {
            const double drift = neutral ? -3.0 : 0.0;
            Simulation sim(plasma_config(order, neutral, drift), comm);
            sim.init();
            double g = 0, rmax = 0, db = 0;
            while (!sim.finished()) {
                sim.step();
                g = std::max(g, sim.gauss_drift());
                db = std::max(db, sim.fields->max_div_b());
            }
            H rho(sim.fields->rho);
            for (int k = 0; k < sim.dom.n[2]; ++k)
                for (int j = 0; j < sim.dom.n[1]; ++j)
                    for (int i = 0; i < sim.dom.n[0]; ++i) rmax = std::max(rmax, std::abs(rho(i, j, k)));
            const double wf = sim.fields->energy();
            const std::string tag = std::string("plasma shape ") + std::to_string(order) +
                                    (neutral ? ", neutral, drift -3" : ", electron slab");
            check(tag + ": Gauss law conserved", g < 1e-12 && wf > 0, fmt("max|dGauss| %.1e (|rho| ~ %.2g)", g, rmax));
            check(tag + ": div B = 0", db < 1e-12, fmt("max|divB| %.1e", db));
        }
}

// 4. Pushers: gyration in a uniform B (gamma conserved, rotation angle 2 atan(B dt / 2 gamma)),
//    acceleration in a uniform E (u = q E t exactly).
static void test_pushers() {
    for (Pusher p : {Pusher::Boris, Pusher::HigueraCary}) {
        const std::string nm = p == Pusher::Boris ? "Boris" : "Higuera-Cary";
        double u[3] = {10, 0, 0};
        const double E[3] = {0, 0, 0}, B[3] = {0, 0, 2.0}, dt = 0.1;
        const int n = 1000;
        for (int s = 0; s < n; ++s) push_momentum(p, -dt, E, B, u);
        const double g = std::sqrt(1 + u[0] * u[0] + u[1] * u[1] + u[2] * u[2]), g0 = std::sqrt(101.0);
        // rotation per step 2 atan(b/g*), b = B dt/2; Boris: g* = gamma, Higuera-Cary: gamma of the mean velocity
        const double b = B[2] * dt / 2, sg = g0 * g0 - b * b;
        const double gs = p == Pusher::Boris ? g0 : std::sqrt(0.5 * (sg + std::sqrt(sg * sg + 4 * b * b)));
        const double ang = n * 2 * std::atan(b / gs);
        const double ex = 10 * std::cos(ang), ey = 10 * std::sin(ang);
        const double err = std::hypot(u[0] - ex, u[1] - ey);
        check(nm + ": gyration, |u| conserved and phase exact", std::abs(g - g0) < 1e-12 && err < 1e-9,
              fmt("dgamma %.1e, phase err %.1e", g - g0, err));
        double v[3] = {0, 0, 0};
        const double E2[3] = {0.1, 0, 0}, B0[3] = {0, 0, 0};
        for (int s = 0; s < n; ++s) push_momentum(p, -dt, E2, B0, v);
        check(nm + ": uniform E, u = q E t", std::abs(v[0] + 0.1 * dt * n) < 1e-12, fmt("err %.1e", v[0] + 0.1 * dt * n));
    }
    // E x B drift with E = B/2: guiding centre velocity E/B = 0.5 (Higuera-Cary gets the drift right)
    double u[3] = {0, 0, 0}, x = 0;
    const double E[3] = {0, 0.5, 0}, B[3] = {0, 0, 1.0}, dt = 0.05;
    const int n = 200000;
    for (int s = 0; s < n; ++s) {
        push_momentum(Pusher::HigueraCary, dt, E, B, u);
        x += dt * u[0] / std::sqrt(1 + u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    }
    const double vd = x / (n * dt);
    check("Higuera-Cary: E x B drift velocity E/B", std::abs(vd - 0.5) < 2e-3, fmt("v_d = %.5f (0.5)", vd));
}

// 5. Langmuir oscillation of cold electrons (immobile ions): frequency wp up to the leapfrog
//    correction sin(w dt/2) = wp dt/2.
static void test_plasma_oscillation(Comm& comm) {
    for (int order : {1, 3}) {                          // shape 3: 6 cells in y, z (guard width 5)
        Config c;
        c.N[0] = 32; c.N[1] = c.N[2] = order == 3 ? 6 : 4;
        c.order = order;
        c.h[0] = 0.2; c.h[1] = 0.5; c.h[2] = 0.5;
        const double n0 = 4.0, wp = 2.0;
        c.t_end = 6 * 2 * M_PI / wp;
        SpeciesConfig e;
        e.name = "e"; e.q = -1; e.m = 1; e.density = n0; e.ppc[0] = 4;
        for (int a = 0; a < 3; ++a) { e.rmin[a] = -1e300; e.rmax[a] = 1e300; }
        SpeciesConfig ion = e; ion.name = "i"; ion.q = 1; ion.m = 1e12;
        c.species = {e, ion};
        c.check();
        Simulation sim(c, comm);
        sim.init();
        const double L = c.N[0] * c.h[0], k = 2 * M_PI / L;
        auto& el = sim.species[0];                                            // velocity kick: Gauss holds at t = 0
        {
            const std::vector<double> x = el.get(Species::X);
            std::vector<double> ux(x.size());
            for (size_t p = 0; p < x.size(); ++p) ux[p] = 1e-3 * std::sin(k * x[p]);
            el.set(Species::UX, ux);
        }
        std::vector<double> ex;
        while (!sim.finished()) { sim.step(); ex.push_back(H(sim.fields->ex)(5, 1, 1)); }
        std::vector<double> zc;                                                // zero crossings (linear interp.)
        for (size_t n = 1; n < ex.size(); ++n)
            if (ex[n - 1] * ex[n] < 0) zc.push_back(n - 1 + ex[n - 1] / (ex[n - 1] - ex[n]));
        const std::string tag = "Langmuir oscillation frequency, shape " + std::to_string(order);
        if (zc.size() < 3) { check(tag, false, "no oscillation"); continue; }
        const double w = M_PI * (zc.size() - 1) / ((zc.back() - zc.front()) * c.h[0]);
        const double wth = 2 / c.h[0] * std::asin(wp * c.h[0] / 2);
        check(tag, std::abs(w / wth - 1) < 0.01, fmt("w = %.4f, expected %.4f", w, wth));
    }
}

// 6. Staggering descriptor: the J positions derived from rho are those of src/grid.hpp, and the
//    Esirkepov deposition satisfies the discrete continuity equation (rho^{n+1} - rho^n)/dt + div J = 0
//    on the charge grid for random moves (independent of the field solver).
static void test_continuity(Comm& comm) {
    const double want_j[3][3] = {{0, 0, 0}, {0.5, 0.5, 0}, {0.5, 0, 0.5}};
    bool ok = true;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) ok = ok && current_loc<RipCC>(a).o[b] == want_j[a][b];
    check("staggering: J_a = rho + e_a/2 matches grid.hpp (RIP-CC)", ok, "");
    for (int order : {1, 2, 3}) {
        Domain d = box(12, 10, 9, 0.3, 0.5, 0.45);
        const int want[3] = {1, 1, 1};
        comm.decompose(d, want);
        Fields f(d);
        Field r0(d), r1(d);
        Species sp;
        sp.q = -1; sp.m = 1;
        std::mt19937_64 rng(11 + order);
        std::uniform_real_distribution<double> u01(0, 1), uu(-3, 3);
        std::vector<double> h[Species::NC];
        for (int p = 0; p < 500; ++p) {                     // well inside: no wrap, guards unused
            const double v[Species::NC] = {(4 + 4 * u01(rng)) * d.h[0], (3 + 4 * u01(rng)) * d.h[1],
                                           (3 + 3 * u01(rng)) * d.h[2], uu(rng), uu(rng), uu(rng), u01(rng)};
            for (int c = 0; c < Species::NC; ++c) h[c].push_back(v[c]);
        }
        sp.append(h);
        deposit_rho(sp, r0, d, order);
        f.zero_currents();
        push_and_deposit(sp, f, d, Pusher::Boris, order);    // E = B = 0: straight moves
        deposit_rho(sp, r1, d, order);
        H R0(r0), R1(r1), Jx(f.jx), Jy(f.jy), Jz(f.jz);
        constexpr int cx = current_shift<RipCC>(0), cy = current_shift<RipCC>(1), cz = current_shift<RipCC>(2);
        double err = 0, scale = 0;
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    const double dr = (R1(i, j, k) - R0(i, j, k)) / d.dt();
                    const double div = (Jx(i + cx, j, k) - Jx(i + cx - 1, j, k)) / d.h[0] +
                                       (Jy(i, j + cy, k) - Jy(i, j + cy - 1, k)) / d.h[1] +
                                       (Jz(i, j, k + cz) - Jz(i, j, k + cz - 1)) / d.h[2];
                    err = std::max(err, std::abs(dr + div));
                    scale = std::max(scale, std::abs(dr));
                }
        check("continuity (rho1-rho0)/dt + div J = 0, shape " + std::to_string(order), err < 1e-13 * scale,
              fmt("max err %.1e (|drho/dt| ~ %.2g)", err, scale));
    }
    // 6b. Moves over many cells (an implicit solver allows c dt > h_perp, decision 28): h_y = h_z = h_x / 6,
    //     so straight moves cross up to ~6 cells in y and z and are split into segments of < 1 cell. The
    //     continuity equation still holds at round-off.
    for (int order : {1, 2, 3}) {
        Domain d = box(12, 48, 48, 0.3, 0.05, 0.05);
        const int want[3] = {1, 1, 1};
        comm.decompose(d, want);
        Fields f(d);
        Field r0(d), r1(d);
        Species sp;
        sp.q = -1; sp.m = 1;
        std::mt19937_64 rng(21 + order);
        std::uniform_real_distribution<double> u01(0, 1), uu(-3, 3);
        std::vector<double> h[Species::NC];
        for (int p = 0; p < 500; ++p) {                     // well inside: the moves stay 10 cells from the edges
            const double v[Species::NC] = {(4 + 4 * u01(rng)) * d.h[0], (16 + 16 * u01(rng)) * d.h[1],
                                           (16 + 16 * u01(rng)) * d.h[2], uu(rng), uu(rng), uu(rng), u01(rng)};
            for (int c = 0; c < Species::NC; ++c) h[c].push_back(v[c]);
        }
        sp.append(h);
        const std::vector<double> y0 = sp.get(Species::Y);
        deposit_rho(sp, r0, d, order);
        f.zero_currents();
        push_and_deposit(sp, f, d, Pusher::Boris, order);
        deposit_rho(sp, r1, d, order);
        const std::vector<double> y1 = sp.get(Species::Y);
        double cells = 0;
        for (size_t q = 0; q < y0.size(); ++q) cells = std::max(cells, std::abs(y1[q] - y0[q]) / d.h[1]);
        H R0(r0), R1(r1), Jx(f.jx), Jy(f.jy), Jz(f.jz);
        constexpr int cx = current_shift<RipCC>(0), cy = current_shift<RipCC>(1), cz = current_shift<RipCC>(2);
        double err = 0, scale = 0;
        for (int k = 0; k < d.n[2]; ++k)
            for (int j = 0; j < d.n[1]; ++j)
                for (int i = 0; i < d.n[0]; ++i) {
                    const double dr = (R1(i, j, k) - R0(i, j, k)) / d.dt();
                    const double div = (Jx(i + cx, j, k) - Jx(i + cx - 1, j, k)) / d.h[0] +
                                       (Jy(i, j + cy, k) - Jy(i, j + cy - 1, k)) / d.h[1] +
                                       (Jz(i, j, k + cz) - Jz(i, j, k + cz - 1)) / d.h[2];
                    err = std::max(err, std::abs(dr + div));
                    scale = std::max(scale, std::abs(dr));
                }
        check("continuity with moves over several cells (split), shape " + std::to_string(order),
              cells > 4 && err < 1e-13 * scale, fmt("moves up to %.1f cells, max err %.1e", cells, err) +
              fmt(" (|drho/dt| ~ %.2g)", scale));
    }
}

// 7. Attribute registry: extra attributes of every type keep their values through append and
//    migration (periodic wrap in y, z and particles leaving through the open x boundary); require()
//    is idempotent, gives existing particles the init value and rejects a type change.
static void test_registry(Comm& comm) {
    Domain d = box(10, 6, 5, 0.3, 0.5, 0.6, false);            // open x: the general migration path
    const int want[3] = {1, 1, 1};
    comm.decompose(d, want);
    Fields f(d);
    Species sp;
    sp.q = -1; sp.m = 1;
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<double> u01(0, 1), uu(-2, 2);
    std::vector<double> h[Species::NC];
    const int np = 4000;
    for (int p = 0; p < np; ++p) {
        const double v[Species::NC] = {u01(rng) * d.L(0), u01(rng) * d.L(1), u01(rng) * d.L(2), uu(rng), uu(rng),
                                       uu(rng), 1.0};
        for (int c = 0; c < Species::NC; ++c) h[c].push_back(v[c]);
    }
    sp.append(h);
    const int sid = sp.require("id", Attr::I64), stag = sp.require("tag", Attr::F64, 7.5);
    const int sflag = sp.require("flag", Attr::I32, -3), sf32 = sp.require("f32", Attr::F32);
    bool ok = sp.require("tag", Attr::F64) == stag && sp.n_attr() == Species::NC + 4;
    for (double t : sp.get(stag)) ok = ok && t == 7.5;
    for (int g : sp.get_as<int>(sflag)) ok = ok && g == -3;
    bool threw = false;
    try { sp.require("id", Attr::F64); } catch (const std::exception&) { threw = true; }
    check("registry: require idempotent, init values, type change rejected", ok && threw, "");
    std::vector<long long> id(np);
    std::vector<double> tag(np);
    std::vector<int> flag(np);
    std::vector<float> f32(np);
    for (int p = 0; p < np; ++p) { id[p] = 1000000000000LL + p; tag[p] = 0.25 * p; flag[p] = p % 97; f32[p] = 0.5f * p; }
    sp.set_as(sid, id); sp.set(stag, tag); sp.set_as(sflag, flag); sp.set_as(sf32, f32);
    for (int n = 0; n < 12; ++n) { push_only(sp, f, d, Pusher::Boris, 1); migrate(sp, d, comm); }
    const auto id1 = sp.get_as<long long>(sid);
    const auto tag1 = sp.get(stag);
    const auto flag1 = sp.get_as<int>(sflag);
    const auto f321 = sp.get_as<float>(sf32);
    const auto x = sp.get(Species::X), y = sp.get(Species::Y), z = sp.get(Species::Z);
    std::vector<char> seen(np, 0);
    int bad = 0, inside = 0;
    for (size_t q = 0; q < id1.size(); ++q) {
        const long long p = id1[q] - 1000000000000LL;
        if (p < 0 || p >= np || seen[p]) { ++bad; continue; }
        seen[p] = 1;
        if (tag1[q] != 0.25 * p || flag1[q] != p % 97 || f321[q] != 0.5f * p) ++bad;
        inside += x[q] >= d.keep_lo(0) && x[q] < d.L(0) && y[q] >= 0 && y[q] < d.L(1) && z[q] >= 0 && z[q] < d.L(2);
    }
    const int left = np - int(id1.size());
    check("registry: attributes survive migration (wrap and open x)",
          bad == 0 && inside == int(id1.size()) && left > 0 && left < np,
          fmt("%.0f left through x, %.0f mismatches", left, bad));
}

// 8. Pipeline: operators run in stage order, in insertion order within a stage, setup once each;
//    the default pipeline of a simulation; a boundary hook is called once per step with t = t_n.
struct Probe : Operator {
    std::string nm; Stage st; std::vector<std::string>* log; int setups = 0;
    Probe(std::string n, Stage s, std::vector<std::string>* l) : nm(std::move(n)), st(s), log(l) {}
    const char* name() const override { return nm.c_str(); }
    Stage stage() const override { return st; }
    void setup(Simulation&) override { ++setups; }
    void apply(Simulation&) override { log->push_back(nm); }
};

static void test_pipeline(Comm& comm) {
    // sort_every = auto (decision 20): tile_sort only with a deposit that needs sorted particles
    Config cg = plasma_config(1, true, 0.0), cs = cg, ct = cg, cb = cg;
    cb.deposit = Config::Deposit::Box;
    cg.deposit = Config::Deposit::Global;
    cs.deposit = Config::Deposit::Global; cs.sort_every = 2;                      // explicit wins
    ct.deposit = Config::Deposit::Tiled;
    Simulation sim(cg, comm), srt(cs, comm), til(ct, comm), bxs(cb, comm);
    const std::vector<std::string> core = sim.pipeline.names(), three = {"push_deposit", "rip_cc", "migrate"};
    std::vector<std::string> four = three;
    four.push_back("tile_sort");
    check("pipeline: default operators; tile_sort only when needed (sort_every = auto)",
          core == three && sim.sort_every == 0 && srt.pipeline.names() == four && srt.sort_every == 2 &&
              (RIP_GPU || (til.pipeline.names() == four && til.sort_every == 1)) && bxs.pipeline.names() == four &&
              bxs.sort_every == 1,
          core.size() == 3 ? core[0] + " -> " + core[1] + " -> " + core[2] : "");
    std::vector<std::string> log;
    Pipeline p;
    std::vector<Probe*> probes;
    const std::vector<std::pair<std::string, Stage>> added = {
        {"diag", Stage::Diagnostics}, {"qed", Stage::Qed},     {"inject_a", Stage::Inject},
        {"coll", Stage::Collisions},  {"inject_b", Stage::Inject}, {"ionize", Stage::Ionize}};
    for (const auto& a : added) {
        auto op = std::make_unique<Probe>(a.first, a.second, &log);
        probes.push_back(op.get());
        p.add(std::move(op));
    }
    p.setup(sim);
    p.run(sim);
    p.run(sim);
    const std::vector<std::string> one = {"inject_a", "inject_b", "ionize", "qed", "coll", "diag"};
    std::vector<std::string> two = one;
    two.insert(two.end(), one.begin(), one.end());
    bool ok = log == two && p.names() == one;
    for (Probe* q : probes) ok = ok && q->setups == 1;
    check("pipeline: stage order, insertion order within a stage, setup once", ok, "");
    std::vector<double> seen;
    sim.hooks.inflow.push_back([&seen](Fields&, double t) { seen.push_back(t); });
    sim.init();
    for (int n = 0; n < 3; ++n) sim.step();
    const double dt = sim.dom.dt();
    check("pipeline: boundary hook once per step at t_n",
          seen.size() == 3 && seen[0] == 0 && seen[1] == dt && seen[2] == 2 * dt, fmt("%.0f calls", double(seen.size())));
}

// 9. Geometry: Cart3D gather of fields that are linear in x, y, z, stored at the stagger positions of
//    RipCC, is exact for shapes 1 and 2 at random positions (both B-splines reproduce linear functions).
//    Runs as a kernel (device memory in the GPU build).
void gather_kernel(const Cart3D::FieldViews F, const double* R, double* out, long n, double h0, double h1, double h2,
                   int order) {
    par_for(n, [=] HD(long p) {
        const double ih[3] = {1 / h0, 1 / h1, 1 / h2};
        const double r[3] = {R[3 * p], R[3 * p + 1], R[3 * p + 2]};
        double E[3], B[3];
        if (order == 1) Cart3D::gather<RipCC, 1>(F, r, ih, E, B);
        else if (order == 2) Cart3D::gather<RipCC, 2>(F, r, ih, E, B);
        else Cart3D::gather<RipCC, 3>(F, r, ih, E, B);
        for (int c = 0; c < 3; ++c) { out[6 * p + c] = E[c]; out[6 * p + 3 + c] = B[c]; }
    });
}

static void test_geometry(Comm& comm) {
    Domain d = box(10, 9, 8, 0.3, 0.5, 0.45);
    const int want[3] = {1, 1, 1};
    comm.decompose(d, want);
    Fields f(d);
    auto lin = [](int c, double x, double y, double z) { return 0.1 * c + (1 + c) * x - (2 - 0.5 * c) * y + 0.3 * c * z; };
    Field* comp[6] = {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz};
    for (int c = 0; c < 6; ++c) {
        const Loc l = c < 3 ? RipCC::E[c] : RipCC::B[c - 3];
        H h(*comp[c]);
        for (int k = -3; k < d.n[2] + 3; ++k)
            for (int j = -3; j < d.n[1] + 3; ++j)
                for (int i = -3; i < d.n[0] + 3; ++i)
                    h(i, j, k) = lin(c, (i + l.o[0]) * d.h[0], (j + l.o[1]) * d.h[1], (k + l.o[2]) * d.h[2]);
        h.put(*comp[c]);
    }
    std::mt19937_64 rng(17);
    std::uniform_real_distribution<double> u01(0, 1);
    const long n = 300;
    std::vector<double> r(3 * n);
    for (long p = 0; p < n; ++p)
        for (int a = 0; a < 3; ++a) r[3 * p + a] = (1 + (d.n[a] - 2) * u01(rng)) * d.h[a];
    Buf<double> dr(r.size()), dout(6 * n);
    dr.from_host(r.data(), 0, r.size());
    for (int order : {1, 2, 3}) {
        gather_kernel(Cart3D::views(f), dr.data(), dout.data(), n, d.h[0], d.h[1], d.h[2], order);
        const std::vector<double> out = dout.host();
        double err = 0;
        for (long p = 0; p < n; ++p)
            for (int c = 0; c < 6; ++c)
                err = std::max(err, std::abs(out[6 * p + c] - lin(c, r[3 * p], r[3 * p + 1], r[3 * p + 2])));
        check("geometry Cart3D: gather of linear fields exact, shape " + std::to_string(order), err < 1e-12,
              fmt("max err %.1e", err));
    }
}

// 10. Tiles: after sort_by_tile every particle lies in its tile range, the ranges cover all particles,
//     the permutation keeps every attribute with its particle and is stable (ids ascending inside a
//     tile); sort_every = 0 removes the operator.
static void test_tiles(Comm& comm) {
    Domain d = box(20, 12, 10, 0.3, 0.5, 0.6);
    const int want[3] = {1, 1, 1};
    comm.decompose(d, want);
    const int cells[3] = {8, 5, 4};                            // incomplete tiles at the upper edges
    const TileGrid g = tile_grid(d, cells);
    Species sp;
    std::mt19937_64 rng(23);
    std::uniform_real_distribution<double> u01(0, 1);
    std::vector<double> h[Species::NC];
    const int np = 5000;
    for (int p = 0; p < np; ++p) {
        const double v[Species::NC] = {u01(rng) * d.L(0), u01(rng) * d.L(1), u01(rng) * d.L(2), u01(rng), 0, 0, 1};
        for (int c = 0; c < Species::NC; ++c) h[c].push_back(v[c]);
    }
    sp.append(h);
    const int sid = sp.require("id", Attr::I64), sv = sp.require("v", Attr::F32);
    std::vector<long long> id(np);
    std::vector<float> v(np);
    for (int p = 0; p < np; ++p) { id[p] = p; v[p] = float(h[Species::UX][p]); }
    sp.set_as(sid, id); sp.set_as(sv, v);
    sort_by_tile(sp, d, g);
    const auto x = sp.get(Species::X), y = sp.get(Species::Y), z = sp.get(Species::Z), ux = sp.get(Species::UX);
    const auto id1 = sp.get_as<long long>(sid);
    const auto v1 = sp.get_as<float>(sv);
    const auto& ts = sp.tile_start;
    bool ok = long(ts.size()) == g.count() + 1 && ts.front() == 0 && ts.back() == np && g.nt[0] == 3 && g.nt[1] == 3;
    int bad = 0;
    std::vector<char> seen(np, 0);
    for (long t = 0; ok && t < g.count(); ++t)
        for (int q = ts[t]; q < ts[t + 1]; ++q) {
            const int i = int(x[q] / d.h[0]), j = int(y[q] / d.h[1]), k = int(z[q] / d.h[2]);
            const long p = id1[q];
            if (tile_of(g, i, j, k) != t || p < 0 || p >= np || seen[p] || ux[q] != h[Species::UX][p] ||
                v1[q] != v[p] || (q > ts[t] && id1[q - 1] > p))
                ++bad;
            else seen[p] = 1;
        }
    check("tiles: sorted ranges, attributes kept, stable order", ok && bad == 0, fmt("%.0f tiles, %.0f bad", double(g.count()), bad));
    Config c = plasma_config(1, true, 0.0);
    c.sort_every = 0;
    Simulation off(c, comm);
    const auto nm = off.pipeline.names();
    check("tiles: sort_every = 0 removes the tile_sort operator", nm.size() == 3 && nm.back() == "migrate", "");
}

// 11. A laser pulse leaving through the open x boundary: div B and the Gauss residual stay at
//     round-off (both diagnostics skip the cell beyond the open edge, where the guards are zero) and the
//     pulse leaves the box. Polarisation y: its y-profile needs an Ex component (div E = 0).
static void test_open_exit(Comm& comm) {
    for (char pol : {'z', 'y'}) {
        Config c;
        c.N[0] = 64; c.N[1] = 48; c.N[2] = 4;
        c.h[0] = 0.25; c.h[1] = 0.5; c.h[2] = 1.0;
        c.periodic[0] = false;
        c.laser.enabled = true;
        c.laser.a0 = 1; c.laser.tau = 4; c.laser.tc = 8; c.laser.w0 = 5; c.laser.yc = 12; c.laser.pol = pol;
        c.t_end = 2 * c.laser.tc + 2 * c.N[0] * c.h[0] + 4 * c.laser.tau;  // pulse in, across, out
        c.sort_every = 0;
        c.check();
        Simulation sim(c, comm);
        sim.init();
        double db = 0, gd = 0, wmax = 0;
        while (!sim.finished()) {
            sim.step();
            db = std::max(db, sim.fields->max_div_b());
            gd = std::max(gd, sim.gauss_drift());
            wmax = std::max(wmax, sim.fields->energy());
        }
        const double wend = sim.fields->energy();
        const std::string tag = std::string("open x, pol ") + pol + ": pulse leaves, ";
        check(tag + "div B, Gauss at round-off", db < 1e-12 && gd < 1e-12 && wmax > 1 && wend < 1e-3 * wmax,
              fmt("max|divB| %.1e, max|dGauss| %.1e", db, gd) + fmt(", W_end/W_max %.1e", wend / wmax));
    }
}

// 11b. Particles leaving through the open x boundaries (docs/specs/open_boundary.md): a hot drifting
//      plasma streams out to the left (drift -3) and to the right (drift +3), shapes 1 and 2, global and
//      tiled deposit. The Gauss law stays at round-off in every checked cell: a particle beyond the
//      left edge is kept until its shape has left cell 0 (x < lo - h_x), so its current takes its
//      charge out; at the right edge the last cell, the only one a removed particle still touches, is
//      not a Gauss point.
static void test_open_particles(Comm& comm) {
    for (int order : {1, 2, 3})
        for (double drift : {-3.0, 3.0})
            for (Config::Deposit dep : {Config::Deposit::Global, Config::Deposit::Tiled}) {
                if (RIP_GPU && dep == Config::Deposit::Tiled) continue;
                Config c = plasma_config(order, true, drift);
                c.N[0] = 24;
                c.periodic[0] = false;
                c.deposit = dep;
                c.tile[0] = 6; c.tile[1] = 10; c.tile[2] = 8;
                c.check();
                Simulation sim(c, comm);
                sim.init();
                const double n0 = double(sim.species[0].size());
                double g = 0;
                while (!sim.finished()) { sim.step(); g = std::max(g, sim.gauss_drift()); }
                const double left = double(sim.species[0].size()) / n0;
                const std::string tag = std::string("open x, shape ") + std::to_string(order) +
                                        (drift < 0 ? ", leaving left" : ", leaving right") +
                                        (dep == Config::Deposit::Tiled ? ", tiled" : ", global");
                check(tag + ": Gauss law conserved", g < 1e-12 && left < 0.5,
                      fmt("max|dGauss| %.1e, electrons left %.2f", g, left));
            }
}

// 12. Coloured-tile deposit (docs/specs/tile_deposit.md) on 4 x 3 x 3 tiles of 6 cells: the colouring
//     visits each tile once and never puts neighbours in one colour; tiled = global to round-off with
//     the Gauss law at round-off; the tiled run does not depend on the thread count (bit-identical);
//     sort_every = 3 makes strays, which take the atomic path.
static Config tiled_config(int order, Config::Deposit dep, long sort_every) {
    Config c = plasma_config(order, true, -3.0);
    c.N[0] = 24; c.N[1] = 18; c.N[2] = 18;
    c.tile[0] = c.tile[1] = c.tile[2] = 6;
    c.t_end = 20 * c.h[0];
    c.deposit = dep;
    c.sort_every = sort_every;
    return c;
}

struct TiledRun { std::vector<double> jx, ex, by; double w = 0, gauss = 0; long strays = 0; bool tiled = false; };

static TiledRun run_tiled(Comm& comm, const Config& c) {
    Simulation sim(c, comm);
    sim.init();
    TiledRun r;
    while (!sim.finished()) { sim.step(); r.gauss = std::max(r.gauss, sim.gauss_drift()); }
    r.jx = sim.fields->jx.host(); r.ex = sim.fields->ex.host(); r.by = sim.fields->by.host();
    r.w = sim.fields->energy();
    r.strays = sim.strays;
    r.tiled = sim.tiled_deposit;
    return r;
}

static double max_rel(const std::vector<double>& a, const std::vector<double>& b) {
    double d = 0, m = 0;
    for (size_t q = 0; q < a.size(); ++q) { d = std::max(d, std::abs(a[q] - b[q])); m = std::max(m, std::abs(a[q])); }
    return m > 0 ? d / m : d;
}

static void test_tiled_deposit(Comm& comm) {
    if (RIP_GPU) {      // the coloured-tile path is the CPU deposit; forced on the GPU it runs one thread per tile
        std::printf("%-58s skipped (GPU build)\n", "tiled deposit tests");
        return;
    }
    {
        Domain d = box(24, 18, 18, 0.3, 0.5, 0.6);
        const int want[3] = {1, 1, 1};
        comm.decompose(d, want);
        const int cells[3] = {6, 6, 6};
        const TileGrid g = tile_grid(d, cells);
        std::vector<int> order, start;
        tile_colours(g, order, start);
        std::vector<int> seen(size_t(g.count()), 0);
        bool ok = long(order.size()) == g.count() && start[8] == int(order.size()) && tiled_deposit_safe(g, d);
        for (int t : order) ok = ok && ++seen[size_t(t)] == 1;
        for (int col = 0; col < 8; ++col)
            for (int a = start[col]; a < start[col + 1]; ++a)
                for (int b = a + 1; b < start[col + 1]; ++b) {
                    const int ta = order[a], tb = order[b];
                    const int dx = std::abs(ta % g.nt[0] - tb % g.nt[0]), dy = std::abs(ta / g.nt[0] % g.nt[1] - tb / g.nt[0] % g.nt[1]),
                              dz = std::abs(ta / (g.nt[0] * g.nt[1]) - tb / (g.nt[0] * g.nt[1]));
                    ok = ok && std::max(dx, std::max(dy, dz)) >= 2;      // never neighbours, not even diagonally
                }
        const int small[3] = {4, 6, 6};
        ok = ok && !tiled_deposit_safe(tile_grid(d, small), d);
        Domain fine = d;                                         // h_y < c dt (implicit solver): > 1 cell per step
        fine.h[1] = 0.2;
        ok = ok && !tiled_deposit_safe(g, fine);
        check("tiled deposit: colouring (each tile once, no neighbours in a colour), safety", ok,
              fmt("%.0f tiles", double(g.count())));
    }
    for (int order : {1, 2, 3}) {
        const TiledRun gl = run_tiled(comm, tiled_config(order, Config::Deposit::Global, 1));
        const TiledRun ti = run_tiled(comm, tiled_config(order, Config::Deposit::Tiled, 1));
        const double dj = max_rel(gl.jx, ti.jx), dw = std::abs(gl.w - ti.w) / gl.w;
        check("tiled deposit = global, shape " + std::to_string(order) + ", Gauss at round-off",
              ti.tiled && !gl.tiled && dj < 1e-9 && dw < 1e-10 && ti.gauss < 1e-12 && ti.strays == 0,
              fmt("max rel dJx %.1e, dW %.1e", dj, dw) + fmt(", max|dGauss| %.1e", ti.gauss));
    }
    TiledRun one, three;
#ifdef _OPENMP
    const int nthr = omp_get_max_threads();
    omp_set_num_threads(1);
    one = run_tiled(comm, tiled_config(2, Config::Deposit::Tiled, 1));
    omp_set_num_threads(3);
    three = run_tiled(comm, tiled_config(2, Config::Deposit::Tiled, 1));
    omp_set_num_threads(nthr);
#else
    one = run_tiled(comm, tiled_config(2, Config::Deposit::Tiled, 1));
    three = run_tiled(comm, tiled_config(2, Config::Deposit::Tiled, 1));
#endif
    check("tiled deposit: 1 and 3 threads bit-identical", one.jx == three.jx && one.ex == three.ex && one.by == three.by,
          fmt("max rel dJx %.1e, dEx %.1e", max_rel(one.jx, three.jx), max_rel(one.ex, three.ex)));
    const TiledRun st = run_tiled(comm, tiled_config(1, Config::Deposit::Tiled, 3));
    check("tiled deposit: sort_every = 3, strays deposited atomically, Gauss at round-off",
          st.strays > 0 && st.gauss < 1e-12, fmt("%.0f strays, max|dGauss| %.1e", double(st.strays), st.gauss));
}

// 13. deposit = auto picks the coloured tiles only when every colour has at least as many tiles as
//     there are threads (otherwise the threads wait: 4x slower on a small grid with 20 threads).
static void test_tiled_auto(Comm& comm) {
#ifdef _OPENMP
    Config c = plasma_config(1, true, 0.0);
    c.N[0] = 32; c.N[1] = 16; c.N[2] = 16;               // 8^3 tiles: 4 x 2 x 2, 2 tiles per colour
    c.deposit = Config::Deposit::Auto;
    const int nthr = omp_get_max_threads();
    omp_set_num_threads(1);
    Simulation one(c, comm);
    omp_set_num_threads(4);
    Simulation four(c, comm);
    omp_set_num_threads(nthr);
    check("deposit = auto: tiled with 1 thread, global with 4 threads (2 tiles per colour)",
          RIP_GPU ? !one.tiled_deposit && !four.tiled_deposit : one.tiled_deposit && !four.tiled_deposit,
          fmt("min tiles per colour %.0f", double(min_tiles_per_colour(one.tiles))));
#else
    (void)comm;
#endif
}

// 14. Reach of the Esirkepov stencil (the basis of the colouring and of the box size): for random moves
//     of less than one cell (|dx| < h_x along every axis), the written cells lie in [c-2, c+3] around
//     the start cell c, and shape 2 reaches both ends along y and z (the bound is tight).
static void test_reach() {
    for (int order : {1, 2, 3}) {
        int lo[3] = {99, 99, 99}, hi[3] = {-99, -99, -99};
        std::mt19937_64 rng(7 + order);
        std::uniform_real_distribution<double> u01(0, 1), du(-0.999, 0.999);
        Domain d = box(12, 12, 12, 0.3, 0.5, 0.45);
        Comm one;
        const int want[3] = {1, 1, 1};
        one.decompose(d, want);
        Fields f(d);
        const int c[3] = {6, 6, 6};
        const double ih[3] = {1 / d.h[0], 1 / d.h[1], 1 / d.h[2]}, coef[3] = {1, 1, 1};
        for (int trial = 0; trial < 3000; ++trial) {
            f.zero_currents();
            double r0[3], r1[3];
            for (int a = 0; a < 3; ++a) { r0[a] = (c[a] + u01(rng)) * d.h[a]; r1[a] = r0[a] + du(rng) * d.h[0]; }
            Buf<double> dr(6);
            const double hr[6] = {r0[0], r0[1], r0[2], r1[0], r1[1], r1[2]};
            dr.from_host(hr, 0, 6);
            const double* R = dr.data();
            const auto F = Cart3D::views(f);
            par_for(1, [=] HD(long) {
                const double a0[3] = {R[0], R[1], R[2]}, a1[3] = {R[3], R[4], R[5]};
                if (order == 1) Cart3D::deposit<RipCC, 1>(F, a0, a1, ih, coef);
                else if (order == 2) Cart3D::deposit<RipCC, 2>(F, a0, a1, ih, coef);
                else Cart3D::deposit<RipCC, 3>(F, a0, a1, ih, coef);
            });
            for (Field* J : {&f.jx, &f.jy, &f.jz}) {
                const auto h = J->host();
                for (int k = -3; k < 15; ++k)
                    for (int j = -3; j < 15; ++j)
                        for (int i = -3; i < 15; ++i)
                            if (h[J->idx(i, j, k)] != 0) {
                                const int g[3] = {i - c[0], j - c[1], k - c[2]};
                                for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], g[a]); hi[a] = std::max(hi[a], g[a]); }
                            }
            }
        }
        bool ok = true;
        // shape 3 (support 2 cells): one cell lower along x, where the charge grid is at i + 1/2 (box and colours: [a - 3, b + 2])
        for (int a = 0; a < 3; ++a) ok = ok && lo[a] >= (order == 3 && a == 0 ? -3 : -2) && hi[a] <= 3;
        if (order >= 2) ok = ok && lo[1] == -2 && hi[1] == 3 && lo[2] == -2 && hi[2] == 3;
        if (order == 3) ok = ok && lo[0] == -3;
        char info[160];
        std::snprintf(info, sizeof info, "x [%d,%d] y [%d,%d] z [%d,%d]", lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
        check(std::string("reach of the deposit within [c-") + (order == 3 ? "3" : "2") + ", c+3], shape " + std::to_string(order), ok, info);
    }
}

// 15. Box deposit (one block per tile with a scratch box; the GPU design, here through the CPU fallback
//     of par_blocks): box = global to round-off with the Gauss law at round-off, shapes 1 and 2, also
//     with partial tiles at the upper edges (26 x 17 x 16 cells, tiles of 6), with a transversely hot
//     plasma that uses the full reach; strays with sort_every = 3.
static void test_box_deposit(Comm& comm) {
    for (int order : {1, 2, 3})
        for (int partial : {0, 1}) {
            Config cg = tiled_config(order, Config::Deposit::Global, 1), cb = tiled_config(order, Config::Deposit::Box, 1);
            for (Config* c : {&cg, &cb}) {
                if (partial) { c->N[0] = 26; c->N[1] = 17; c->N[2] = 16; }
                for (auto& sc : c->species) sc.uth[1] = sc.uth[2] = 3.0;    // moves of > 1/2 cell in y, z: full reach
            }
            const TiledRun gl = run_tiled(comm, cg);
            Simulation probe(cb, comm);
            const TiledRun bx = run_tiled(comm, cb);
            const double dj = max_rel(gl.jx, bx.jx), dw = std::abs(gl.w - bx.w) / gl.w;
            check("box deposit = global, shape " + std::to_string(order) + (partial ? ", partial tiles" : "") +
                      ", Gauss at round-off",
                  probe.box_deposit && dj < 1e-9 && dw < 1e-10 && bx.gauss < 1e-12 && bx.strays == 0,
                  fmt("max rel dJx %.1e, dW %.1e", dj, dw) + fmt(", max|dGauss| %.1e", bx.gauss));
        }
    {   // fixed-point box (decision 26): weights over 8 decades inside one tile (density e^(-6 y), tiles of
        // 3 in y): the error is absolute, round-off of the heaviest content of a tile; reported per cell
        Config cg = tiled_config(2, Config::Deposit::Global, 1), cb = tiled_config(2, Config::Deposit::Box, 1);
        for (Config* c : {&cg, &cb})
            for (auto& sc : c->species) sc.density_fn = Expr("exp(-6 * y)", {"x", "y", "z"}, nullptr);
        const TiledRun gl = run_tiled(comm, cg), bx = run_tiled(comm, cb);
        double m = 0, loc = 0, lo = 1e300;
        for (double v : gl.jx) m = std::max(m, std::abs(v));
        for (size_t q = 0; q < gl.jx.size(); ++q)
            if (std::abs(gl.jx[q]) > 1e-9 * m) {
                loc = std::max(loc, std::abs(gl.jx[q] - bx.jx[q]) / std::abs(gl.jx[q]));
                lo = std::min(lo, std::abs(gl.jx[q]) / m);
            }
        check("box deposit, weights over 8 decades: box = global to round-off of max|J|, Gauss",
              max_rel(gl.jx, bx.jx) < 1e-13 && bx.gauss < 1e-12,
              fmt("max rel dJx %.1e", max_rel(gl.jx, bx.jx)) + fmt(", per cell %.1e", loc) +
                  fmt(" (|J| down to %.0e of max)", lo) + fmt(", max|dGauss| %.1e", bx.gauss));
    }
    Config cs = tiled_config(2, Config::Deposit::Box, 3);
    for (auto& sc : cs.species) sc.uth[1] = sc.uth[2] = 3.0;
    const TiledRun st = run_tiled(comm, cs);
    check("box deposit: sort_every = 3, strays deposited globally, Gauss at round-off",
          st.strays > 0 && st.gauss < 1e-12, fmt("%.0f strays, max|dGauss| %.1e", double(st.strays), st.gauss));
}

// Environment variables VLPL_<X> with the old names RIP_<X> as deprecated aliases (decision 88, src/env.hpp): the new name
// wins, the old one is read when the new one is unset, neither gives nullptr.
static void test_env_alias() {
    unsetenv("VLPL_ENVTEST"); unsetenv("RIP_ENVTEST");
    const bool none = vlpl_getenv("VLPL_ENVTEST") == nullptr;
    setenv("RIP_ENVTEST", "7", 1);
    const char* a = vlpl_getenv("VLPL_ENVTEST");
    const bool old_ok = a && std::string(a) == "7";
    setenv("VLPL_ENVTEST", "8", 1);
    const char* b = vlpl_getenv("VLPL_ENVTEST");
    const bool new_ok = b && std::string(b) == "8";
    unsetenv("VLPL_ENVTEST"); unsetenv("RIP_ENVTEST");
    check("env: VLPL_<X>, old RIP_<X> as alias", none && old_ok && new_ok,
          std::string(none ? "" : "unset not null; ") + (old_ok ? "" : "alias not read; ") + (new_ok ? "" : "new name not first"));
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);       // line-buffered: a crash in CI shows the last test reached
    Comm comm;
    comm.init(&argc, &argv);
    // build/tests [name ...]: only the groups whose function name contains one of the names
    auto run = [&](const char* name, const std::function<void()>& f) {
        bool sel = argc < 2;
        for (int i = 1; i < argc; ++i) sel = sel || std::strstr(name, argv[i]);
        if (sel) f();
    };
    run("test_env_alias", [&] { test_env_alias(); });
    run("test_reductions", [&] { test_reductions(); });
    run("test_block_add", [&] { test_block_add(); });
    run("test_vacuum_shift", [&] { test_vacuum_shift(comm); });
    run("test_vacuum_random", [&] { test_vacuum_random(comm); });
    run("test_pushers", [&] { test_pushers(); });
    run("test_plasma_conservation", [&] { test_plasma_conservation(comm); });
    run("test_plasma_oscillation", [&] { test_plasma_oscillation(comm); });
    run("test_continuity", [&] { test_continuity(comm); });
    run("test_registry", [&] { test_registry(comm); });
    run("test_pipeline", [&] { test_pipeline(comm); });
    run("test_geometry", [&] { test_geometry(comm); });
    run("test_tiles", [&] { test_tiles(comm); });
    run("test_open_exit", [&] { test_open_exit(comm); });
    run("test_open_particles", [&] { test_open_particles(comm); });
    run("test_tiled_deposit", [&] { test_tiled_deposit(comm); });
    run("test_tiled_auto", [&] { test_tiled_auto(comm); });
    run("test_reach", [&] { test_reach(); });
    run("test_box_deposit", [&] { test_box_deposit(comm); });
    run("test_window", [&] { test_window(comm); });
    run("test_profile", [&] { test_profile(comm); });
    run("test_beam", [&] { test_beam(comm); });
    run("test_diag", [&] { test_diag(comm); });
    run("test_pml", [&] { test_pml(comm); });
    run("test_laser", [&] { test_laser(comm); });
    run("test_openpmd", [&] { test_openpmd(comm); });
    run("test_lasy", [&] { test_lasy(comm); });
    run("test_yee", [&] { test_yee(comm); });
    run("test_irip", [&] { test_irip(comm); });
    run("test_rz", [&] { test_rz(comm); });
    run("test_rz_grid", [&] { test_rz_grid(comm); });
    run("test_xflux", [&] { test_xflux(comm); });
    run("test_tfsf", [&] { test_tfsf(comm); });
    run("test_x_absorber", [&] { test_x_absorber(comm); });
    run("test_tfsf_modes", [&] { test_tfsf_modes(comm); });
    run("test_fresnel", [&] { test_fresnel(comm); });
    run("test_side", [&] { test_side(comm); });
    run("test_side_ff", [&] { test_side_ff(comm); });
    run("test_side_plasma", [&] { test_side_plasma(comm); });
    run("test_solid", [&] { test_solid(comm); });
    run("test_cart2d", [&] { test_cart2d(comm); });
    run("test_mover", [&] { test_mover(comm); });
    run("test_boost", [&] { test_boost(comm); });
    run("test_boost_antenna", [&] { test_boost_antenna(comm); });
    run("test_overlap", [&] { test_overlap(comm); });
    run("test_medium", [&] { test_medium(comm); });
    run("test_medium2", [&] { test_medium2(comm); });
    run("test_bound", [&] { test_bound(comm); });
    run("test_hfocus", [&] { test_hfocus(comm); });
    run("test_rz_tfsf", [&] { test_rz_tfsf(comm); });
    run("test_rz_shape", [&] { test_rz_shape(comm); });
    run("test_rz_axis", [&] { test_rz_axis(comm); });
    run("test_rz_metal", [&] { test_rz_metal(comm); });
    run("test_rz_metal_phi", [&] { test_rz_metal_phi(comm); });
    run("test_structure", [&] { test_structure(comm); });
    run("test_rz_structure", [&] { test_rz_structure(comm); });
    run("test_rz_dielectric", [&] { test_rz_dielectric(comm); });
    run("test_dielectric3d", [&] { test_dielectric3d(comm); });
    run("test_ionization", [&] { test_ionization(comm); });
    run("test_qed", [&] { test_qed(comm); });
    run("test_collisions", [&] { test_collisions(comm); });
    run("test_pairs", [&] { test_pairs(comm); });
    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "all passed", failures);
    comm.finalize();
    return failures ? 1 : 0;
}
