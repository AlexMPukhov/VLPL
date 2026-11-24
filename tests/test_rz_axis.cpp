// The axis cells of RZ in the modes m >= 1 (decision 106): the density-conserving radial map of decision 53 is for m = 0 only.
// A1. A uniform plasma moving rigidly along x is a uniform current J_x = q n v: mode 1 with J_r1 = J_x / 2, J_phi1 = i J_x / 2
//     at every radial face and node, also next to the axis. With the map in every mode (numerics.rz_axis_map = all, the code
//     before decision 106) the first face has 5/6 of it (shape 1: the flux weight dxi/du = 3/2 - u in the axis cell).
// A2. The dipole of a charge next to the axis: a charge q at x0 < dr puts q x0 / dr on node 1 of mode 1 (linear in x0, shapes 1
//     and 2); with the map in every mode it is q xi(x0) (shape 1: 1.45 x at x0 = 0.1 dr; shape 2: (x0/dr)^0.33).
// A3. The gather of a linear focusing field E_r = g r (m = 0): exact with rz_gather = plain; printed for matched (decision 80).
// A6. numerics.rz_axis_current = regular (decision 112): the first-cell r^2 part of the mode-1 current removed, continuity at round-off,
//     the transverse current moment exact, a uniform transverse current unchanged.
// A4. Betatron oscillation of a particle in that field (q E_r = -m w^2 r, amplitude 0.3 dr and 0.8 dr, no self-field): the frequency
//     against w. Plain: the leapfrog value; matched: too high and amplitude dependent (the force is g dr xi(r) in the axis cell).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "common.hpp"
#include "particles.hpp"
#include "rz.hpp"
#include "species.hpp"

namespace {
HD inline int slot(int m, int c) { return m == 0 ? 0 : 2 * m - 1 + c; }

// max |J / J_ref - 1| over the faces / nodes j < jmax of the column i; first: the value at the first face
void uniform_current(Comm& comm, int order, bool all, double& e_r, double& e_p, double& first) {
    const int nz = 8, nr = 20, M = 2;
    const double dz = 0.3, dr = 0.2, n0 = 0.7, ux = 1e-4;
    Domain d = rz_domain(comm, nz, nr, M, dz, dr, true, Solver::RipCC);
    FieldsRZ f(d, M);
    f.map_all = all;
    Species sp;
    sp.q = -1; sp.m = 1;
    std::vector<double> h[Species::NC];
    const int sub = 64, na = 16, nzp = 4;
    for (int i = 0; i < nzp * nz; ++i)
        for (int j = 0; j < sub * 14; ++j)                          // a column of radius 14 dr, midpoints in r
            for (int a = 0; a < na; ++a) {
                const double z = (i + 0.5) * dz / nzp, r = (j + 0.5) * dr / sub, ph = 2 * M_PI * (a + 0.31) / na;
                h[Species::X].push_back(z); h[Species::Y].push_back(r * std::cos(ph)); h[Species::Z].push_back(r * std::sin(ph));
                h[Species::UX].push_back(0); h[Species::UY].push_back(ux); h[Species::UZ].push_back(0);
                h[Species::W].push_back(n0 * 2 * M_PI * r * (dr / sub) * (dz / nzp) / na);
            }
    sp.append(h);
    f.jz.fill(0); f.jr.fill(0); f.jp.fill(0);
    rz_push_deposit(sp, f, Pusher::None, true, order);
    const std::vector<double> jr = f.jr.host(), jp = f.jp.host();
    const double ref = 0.5 * sp.q * n0 * ux / std::sqrt(1 + ux * ux);
    const int i = nz / 2;
    e_r = e_p = 0;
    for (int j = 0; j < 8; ++j) {
        e_r = std::max(e_r, std::abs(jr[f.jr.idx(i, j, slot(1, 0))] / ref - 1) + std::abs(jr[f.jr.idx(i, j, slot(1, 1))] / ref));
        if (j > 0) e_p = std::max(e_p, std::abs(jp[f.jp.idx(i, j, slot(1, 1))] / ref - 1) + std::abs(jp[f.jp.idx(i, j, slot(1, 0))] / ref));
    }
    first = jr[f.jr.idx(i, 0, slot(1, 0))] / ref;
}
void uniform_current_regular(Comm& comm, int order, double& e_r, double& e_p, double& first) {
    const bool all = false;
    const int nz = 8, nr = 20, M = 2;
    const double dz = 0.3, dr = 0.2, n0 = 0.7, ux = 1e-4;
    Domain d = rz_domain(comm, nz, nr, M, dz, dr, true, Solver::RipCC);
    FieldsRZ f(d, M);
    f.map_all = all;
    Species sp;
    sp.q = -1; sp.m = 1;
    std::vector<double> h[Species::NC];
    const int sub = 64, na = 16, nzp = 4;
    for (int i = 0; i < nzp * nz; ++i)
        for (int j = 0; j < sub * 14; ++j)                          // a column of radius 14 dr, midpoints in r
            for (int a = 0; a < na; ++a) {
                const double z = (i + 0.5) * dz / nzp, r = (j + 0.5) * dr / sub, ph = 2 * M_PI * (a + 0.31) / na;
                h[Species::X].push_back(z); h[Species::Y].push_back(r * std::cos(ph)); h[Species::Z].push_back(r * std::sin(ph));
                h[Species::UX].push_back(0); h[Species::UY].push_back(ux); h[Species::UZ].push_back(0);
                h[Species::W].push_back(n0 * 2 * M_PI * r * (dr / sub) * (dz / nzp) / na);
            }
    sp.append(h);
    f.jz.fill(0); f.jr.fill(0); f.jp.fill(0);
    rz_push_deposit(sp, f, Pusher::None, true, order);
    rz_axis_regular_current(f);
    const std::vector<double> jr = f.jr.host(), jp = f.jp.host();
    const double ref = 0.5 * sp.q * n0 * ux / std::sqrt(1 + ux * ux);
    const int i = nz / 2;
    e_r = e_p = 0;
    for (int j = 0; j < 8; ++j) {
        e_r = std::max(e_r, std::abs(jr[f.jr.idx(i, j, slot(1, 0))] / ref - 1) + std::abs(jr[f.jr.idx(i, j, slot(1, 1))] / ref));
        if (j > 0) e_p = std::max(e_p, std::abs(jp[f.jp.idx(i, j, slot(1, 1))] / ref - 1) + std::abs(jp[f.jp.idx(i, j, slot(1, 0))] / ref));
    }
    first = jr[f.jr.idx(i, 0, slot(1, 0))] / ref;
}

// the charge of mode 1 on node 1 of a unit charge at x = x0, in units of q
double dipole(Comm& comm, int order, bool all, double x0) {
    const int nz = 8, nr = 12, M = 2;
    const double dz = 0.3, dr = 0.2;
    Domain d = rz_domain(comm, nz, nr, M, dz, dr, true, Solver::RipCC);
    FieldsRZ f(d, M);
    f.map_all = all;
    Species sp;
    sp.q = 1; sp.m = 1;
    std::vector<double> h[Species::NC];
    h[Species::X].push_back(4.2 * dz); h[Species::Y].push_back(x0 * dr); h[Species::Z].push_back(0);
    h[Species::UX].push_back(0); h[Species::UY].push_back(0); h[Species::UZ].push_back(0);
    h[Species::W].push_back(1);
    sp.append(h);
    Field rho(d);
    rho.fill(0);
    rz_rho(sp, f, rho, order);
    const std::vector<double> r = rho.host();
    double s = 0;
    for (int i = 0; i < nz; ++i) s += r[rho.idx(i, 1, slot(1, 0))] * 2 * M_PI * f.grid().A[1] * dz;
    return s;
}

// E_x / (g x) gathered at x = u dr in the field E_r = g r of mode 0
double linear_gather(Comm& comm, int order, int gather, double u) {
    const int nz = 8, nr = 12;
    const double dz = 0.3, dr = 0.2, g0 = 1.7;
    Domain d = rz_domain(comm, nz, nr, 1, dz, dr, true, Solver::RipCC);
    FieldsRZ f(d, 1);
    f.gather_matched = gather > 0; f.gather_energy = gather == 2;
    std::vector<double> h = f.er.host();
    for (int j = 0; j < d.n[1]; ++j)
        for (int i = -d.ng; i < d.n[0] + d.ng; ++i) h[f.er.idx(i, j, 0)] = g0 * f.grid().rh[size_t(j)];
    f.er.upload(h);
    double E[3], B[3];
    rz_gather_point(f, 4.2 * dz, u * dr, 0, E, B, order);
    return E[1] / (g0 * u * dr);
}
// w_numerical / w of the betatron oscillation with the amplitude a dr
double betatron(Comm& comm, int order, int gather, double a) {
    const int nz = 8, nr = 12;
    const double dz = 0.3, dr = 0.2, w = 0.05;
    Domain d = rz_domain(comm, nz, nr, 1, dz, dr, true, Solver::RipCC);
    FieldsRZ f(d, 1);
    f.gather_matched = gather > 0;
    std::vector<double> h = f.er.host();
    for (int j = 0; j < d.n[1]; ++j)
        for (int i = -d.ng; i < d.n[0] + d.ng; ++i) h[f.er.idx(i, j, 0)] = -w * w * f.grid().rh[size_t(j)];
    f.er.upload(h);
    Species sp;
    sp.q = 1; sp.m = 1;
    std::vector<double> p[Species::NC];
    p[Species::X].push_back(4.2 * dz); p[Species::Y].push_back(a * dr); p[Species::Z].push_back(0);
    p[Species::UX].push_back(0); p[Species::UY].push_back(0); p[Species::UZ].push_back(0);
    p[Species::W].push_back(1);
    sp.append(p);
    const double dt = d.dt();
    double xo = a * dr, t1 = -1, t2 = -1;
    int n = 0;
    for (int s = 1; s <= 4000 && n < 9; ++s) {
        rz_push_deposit(sp, f, Pusher::Boris, false, order);
        const double x = sp.get(Species::Y)[0];
        if (xo > 0 && x <= 0) {                                   // downward zero crossing, linear in the step
            const double t = (s - 1 + xo / (xo - x)) * dt;
            if (n == 0) t1 = t;
            t2 = t; ++n;
        }
        xo = x;
    }
    return n > 1 ? 2 * M_PI * (n - 1) / (t2 - t1) / w : 0.0;
}
// A6 (numerics.rz_axis_current = regular, decision 112): one particle on the straight line y = b (b = 0 .. 2 dr) crossing the axis region
// in N = 3, 10, 30 steps per cell (u_z = 0), shapes 1, 2, phase and chord. Per step: the first-cell r^2 part D = J_r,0 + i J_phi,1 of
// mode 1 before and after rz_axis_regular_current, the m = 1 continuity residual at the rows 1 .. 3 (rho from rz_rho before and after the
// move, the divergence stencil of the field solver) relative to the largest |d rho / dt|, and the total x moment of the transverse current
// over the passage against the exact q dx.
struct PassResult { double dbefore = 0, dafter = 0, jplus = 0, cont = 0, mom = 0; };
PassResult axis_passage(Comm& comm, int order, bool chord, double b, int N) {
    const int nz = 16, nr = 14, M = 2;
    const double dr = M_PI, dz = 2.3397496101291093;
    Domain d = rz_domain(comm, nz, nr, M, dz, dr, true, Solver::IRip);
    FieldsRZ f(d, M);
    f.deposit_chord = chord;
    const double v = dr / (N * dz), u = v / std::sqrt(1 - v * v);
    std::vector<double> h[Species::NC];
    h[Species::X].push_back(5.3 * dz); h[Species::Y].push_back(-4 * dr); h[Species::Z].push_back(b * dr);
    h[Species::UX].push_back(0); h[Species::UY].push_back(u); h[Species::UZ].push_back(0); h[Species::W].push_back(1);
    Species sp; sp.q = 1; sp.m = 1; sp.append(h);
    const RadialGrid& g = f.grid();
    const double dt = d.dt();
    Field r0(d), r1(d);
    PassResult R;
    double mom = 0;
    // the continuity stencil (sign and the z face) chosen once on the rows 2, 3 of the raw deposit, where the rule changes nothing
    int sz = 0, sd = 0;
    double best = 1e300, rho_scale = 0;
    std::vector<double> res[4];
    for (int s = 0; s < 8 * N; ++s) {
        r0.fill(0); rz_rho(sp, f, r0, order); comm.add_guards(d, {&r0});
        f.jz.fill(0); f.jr.fill(0); f.jp.fill(0);
        rz_push_deposit(sp, f, Pusher::None, true, order);
        comm.add_guards(d, {&f.jz, &f.jr, &f.jp});
        r1.fill(0); rz_rho(sp, f, r1, order); comm.add_guards(d, {&r1});
        auto Dmax = [&]() {
            const std::vector<double> a = f.jr.host(), c = f.jp.host();
            double m = 0, p = 0;
            for (int i = 0; i < nz; ++i) {
                const double rr = a[f.jr.idx(i, 0, slot(1, 0))], ri = a[f.jr.idx(i, 0, slot(1, 1))], pr = c[f.jp.idx(i, 1, slot(1, 0))], pi = c[f.jp.idx(i, 1, slot(1, 1))];
                m = std::max(m, std::hypot(rr - pi, ri + pr)); p = std::max(p, std::hypot(rr + pi, ri - pr));
            }
            return std::make_pair(m, p);
        };
        const auto before = Dmax();
        rz_axis_regular_current(f);
        const auto after = Dmax();
        R.dbefore = std::max(R.dbefore, before.first); R.dafter = std::max(R.dafter, after.first); R.jplus = std::max(R.jplus, after.second);
        const std::vector<double> JZ = f.jz.host(), JR = f.jr.host(), JP = f.jp.host(), A0 = r0.host(), A1 = r1.host();
        for (int i = 0; i < nz; ++i)
            for (int j = 0; j < nr; ++j) {
                const double V = 2 * M_PI * dz;
                mom += V * (g.B[size_t(j)] * JR[f.jr.idx(i, j, slot(1, 0))] + g.A[size_t(j)] * JP[f.jp.idx(i, j, slot(1, 1))]) * dt;
            }
        // residual variants: z face (0: Jz(i+1) - Jz(i), 1: Jz(i) - Jz(i-1)), sign of the divergence (0: +, 1: -)
        for (int vz = 0; vz < 2; ++vz)
            for (int vs = 0; vs < 2; ++vs) {
                double rmax[4] = {0, 0, 0, 0};
                for (int i = 0; i < nz; ++i)
                    for (int j = 1; j <= 3; ++j)
                        for (int c = 0; c < 2; ++c) {
                            const int sl = slot(1, c);
                            const double drho = (A1[r1.idx(i, j, sl)] - A0[r0.idx(i, j, sl)]) / dt;
                            rho_scale = std::max(rho_scale, std::abs(drho));
                            const double djz = vz == 0 ? JZ[f.jz.idx(i + 1, j, sl)] - JZ[f.jz.idx(i, j, sl)] : JZ[f.jz.idx(i, j, sl)] - JZ[f.jz.idx(i - 1, j, sl)];
                            const double djr = (g.rh[size_t(j)] * JR[f.jr.idx(i, j, sl)] - g.rh[size_t(j - 1)] * JR[f.jr.idx(i, j - 1, sl)]) / g.A[size_t(j)];
                            // i m_f / r_j J_phi: real part -m_f Im / r, imaginary part m_f Re / r
                            const double jph = c == 0 ? -g.mf(1) * JP[f.jp.idx(i, j, slot(1, 1))] / g.r[size_t(j)] : g.mf(1) * JP[f.jp.idx(i, j, slot(1, 0))] / g.r[size_t(j)];
                            const double div = djz / dz + djr + jph;
                            rmax[j] = std::max(rmax[j], std::abs(drho + (vs ? -div : div)));
                        }
                const int key = 2 * vz + vs;
                res[key].push_back(std::max(rmax[1], std::max(rmax[2], rmax[3])));
                if (s == 8 * N - 1) {
                    double m23 = 0, m1 = 0;
                    for (double x : res[key]) m1 = std::max(m1, x);
                    (void)m23;
                    if (m1 < best) { best = m1; sz = vz; sd = vs; }
                }
            }
    }
    (void)sz; (void)sd;
    R.cont = best / std::max(rho_scale, 1e-300);
    R.mom = mom / (8 * dr);                                 // q = 1, dx = 8 dr
    return R;
}

}  // namespace

void test_rz_axis(Comm& comm) {
    if (comm.size > 1) return;                       // host copies of the whole grid (MPI: tests/mpi.sh)
    for (int order = 1; order <= 2; ++order) {
        double er, ep, f1, ero, epo, f1o;
        uniform_current(comm, order, false, er, ep, f1);
        uniform_current(comm, order, true, ero, epo, f1o);
        std::printf("  A1 uniform J_x, mode 1, shape %d: max |J_r1 / ref - 1| %.2e, |J_phi1 / ref - 1| %.2e, first face %.4f "
                    "(map in every mode: %.2e, %.2e, first face %.4f)\n", order, er, ep, f1, ero, epo, f1o);
        check("A1 rz axis: uniform transverse current in mode 1 next to the axis, shape " + std::to_string(order),
              er < (order == 1 ? 1e-3 : 0.1) && ep < (order == 1 ? 1e-3 : 0.1), fmt("%.1e %.1e", er, ep));
    }
    for (int order = 1; order <= 2; ++order) {
        const double a = dipole(comm, order, false, 0.1), b = dipole(comm, order, false, 0.4), ao = dipole(comm, order, true, 0.1);
        std::printf("  A2 dipole of a charge at x0 = 0.1 dr, shape %d: node-1 charge of mode 1 / (q x0 / dr) = %.4f (0.4 dr: %.4f; "
                    "map in every mode: %.3f)\n", order, a / 0.1, b / 0.4, ao / 0.1);
        check("A2 rz axis: dipole of a charge next to the axis linear in its offset, shape " + std::to_string(order),
              std::abs(a / 0.1 - 1) < 1e-12 && std::abs(b / 0.4 - 1) < 1e-12, fmt("%.1e %.1e", a / 0.1 - 1, b / 0.4 - 1));
    }
    double ep = 0;
    for (int order = 1; order <= 2; ++order)
        for (double u : {0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5}) {
            const double p = linear_gather(comm, order, 0, u), m = linear_gather(comm, order, 1, u), e = linear_gather(comm, order, 2, u);
            std::printf("  A3 linear E_r = g r (m = 0), shape %d, r = %.2f dr: gathered / exact: plain %.4f, matched %.4f, energy %.4f\n",
                        order, u, p, m, e);
            ep = std::max(ep, std::abs(p - 1));
        }
    check("A3 rz axis: gather of a linear radial field with rz_gather = plain (exact)", ep < 1e-12, fmt("%.1e", ep));
    double eb = 0;
    for (int order = 1; order <= 2; ++order)
        for (double a : {0.3, 0.8, 2.5}) {
            const double p = betatron(comm, order, 0, a), m = betatron(comm, order, 1, a);
            std::printf("  A4 betatron frequency / exact, shape %d, amplitude %.1f dr: plain %.5f, matched %.5f\n", order, a, p, m);
            eb = std::max(eb, std::abs(p - 1));
        }
    check("A4 rz axis: betatron frequency in a linear focusing field with rz_gather = plain", eb < 1e-3, fmt("%.1e", eb));
    {   // A6: numerics.rz_axis_current = regular (decision 112)
        double dworst = 0, cworst = 0, mworst = 0, draw = 0;
        for (int order = 1; order <= 2; ++order)
            for (int chord = 0; chord <= 1; ++chord)
                for (int N : {3, 10, 30})
                    for (double b : {0.0, 0.01, 0.1, 0.25, 0.5, 1.0, 2.0}) {
                        const PassResult R = axis_passage(comm, order, chord == 1, b, N);
                        if (N == 10 && chord == 1)
                            std::printf("  A6 shape %d chord, N %d, b = %.2f dr: first-cell r^2 part D before %.3e, after %.1e (J_+ %.3e); continuity / max|drho/dt| %.1e; "
                                        "x moment / q dx %.12f\n", order, N, b, R.dbefore, R.dafter, R.jplus, R.cont, R.mom);
                        dworst = std::max(dworst, R.dafter / std::max(R.jplus, 1e-300)); cworst = std::max(cworst, R.cont);
                        mworst = std::max(mworst, std::abs(R.mom - 1)); draw = std::max(draw, R.dbefore / std::max(R.jplus, 1e-300));
                    }
        check("A6 rz axis current regular: first-cell r^2 part of mode 1 removed (straight passages b = 0 .. 2 dr, shapes 1, 2, phase, chord)",
              dworst < 1e-12, fmt("max |D| / |J_+| after %.1e", dworst) + fmt(" (before %.2f)", draw));
        check("A6 rz axis current regular: m = 1 continuity at round-off", cworst < 1e-12, fmt("%.1e", cworst));
        check("A6 rz axis current regular: total transverse current moment = q dx", mworst < 1e-10, fmt("%.1e", mworst));
        // a uniform transverse current is unchanged (the regular part only)
        double er, ep, f1, er2, ep2, f12;
        uniform_current(comm, 1, false, er, ep, f1);
        uniform_current_regular(comm, 1, er2, ep2, f12);
        std::printf("  A6 uniform J_x, shape 1: max |J / ref - 1| raw %.2e %.2e, regular %.2e %.2e\n", er, ep, er2, ep2);
        check("A6 rz axis current regular: a uniform transverse current next to the axis kept (no worse than raw)", er2 <= er + 1e-12 && ep2 <= ep + 1e-12,
              fmt("regular %.1e", std::max(er2, ep2)) + fmt(", raw %.1e", std::max(er, ep)));
    }
}
