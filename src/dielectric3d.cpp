// Dielectrics in the RIP-CC step of Cart3D and 2D (dielectric3d.hpp, docs/specs/dielectric3d.md, decision 100).
#include "dielectric3d.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "comm.hpp"
#include "fields.hpp"
#include "metal3d.hpp"

namespace {
constexpr int BS = 3;                  // block record: chi (constant), code offset, code length (0: the constant)
// chi of the blocks at the point (x, y, z) (absolute, code units)
HD inline double chi3(const double* B, int nb, const Expr::Ins* code, double x, double y, double z) {
    double s = 0;
    for (int b = 0; b < nb; ++b) {
        const double* p = B + BS * b;
        if (p[2] > 0) {
            const double v[3] = {x, y, z};
            s += expr_eval(code + long(p[1]), int(p[2]), v) - 1;
        } else s += p[0];
    }
    return s;
}
// eps over the box [x0, x1] x [y0, y1] x [z0, z1] of a component along axis `dir`: harmonic along dir, arithmetic across
// (4 x 4 x 4 sub-samples; flat: z0 = z1, one sample in z). Lp[a] > 0: axis a periodic with that length, every sample wrapped
// into [0, Lp) (the guards then hold the chi of their periodic partners)
HD inline double eps_box(const double* B, int nb, const Expr::Ins* code, double x0, double x1, double y0, double y1, double z0,
                         double z1, int dir, bool flat, const double* Lp) {
    const int S = 4, SZ = flat ? 1 : 4;
    const int n[3] = {S, S, SZ};
    const double a[3] = {x0, y0, z0}, w[3] = {x1 - x0, y1 - y0, z1 - z0};
    const int u = dir == 0 ? 1 : 0, v = dir == 2 ? 1 : 2;    // the two axes across
    double acc = 0;
    for (int p = 0; p < n[u]; ++p)
        for (int q = 0; q < n[v]; ++q) {
            double h = 0;
            for (int t = 0; t < n[dir]; ++t) {
                double c[3];
                c[dir] = a[dir] + (t + 0.5) * w[dir] / n[dir];
                c[u] = a[u] + (p + 0.5) * w[u] / n[u];
                c[v] = a[v] + (q + 0.5) * w[v] / n[v];
                for (int e = 0; e < 3; ++e) if (Lp[e] > 0) c[e] -= Lp[e] * floor(c[e] / Lp[e]);
                h += 1 / (1 + chi3(B, nb, code, c[0], c[1], c[2]));
            }
            acc += n[dir] / h;
        }
    return acc / (n[u] * n[v]);
}
// the masks (0: E held at 0 by the metal, or a guard node beyond an open x end) and the line coefficients
struct Line3 {
    FV E, nE, chi;
    Metal3View mv;
    bool me;
    int j, k, c;                       // c: 0 E_y, 1 E_z
    int lo, hi;                        // open x: the global nodes [lo, hi) in local indices
    HD double mask(int i) const {
        if (i < lo || i >= hi) return 0.0;
        if (!me) return 1.0;
        return (c == 0 ? mv.ey(i, j, k) : mv.ez(i, j, k)) ? 0.0 : 1.0;
    }
    HD void coef(int a, double& l, double& dd, double& u, double& r) const {
        const double ch = chi(a, j, k);
        if (ch == 0) { l = u = r = 0; dd = 1; return; }
        const double ma = mask(a), mb = mask(a + 1);
        l = 0.25 * ch * ma; u = 0.25 * ch * mb; dd = 1 + l + u;
        r = 0.5 * ch * (ma * (nE(a, j, k) - E(a, j, k)) + mb * (nE(a + 1, j, k) - E(a + 1, j, k)));
    }
};
// the face type of a metal E_y / E_z edge at node i: 0 free, 1 nothing (inside, or one cell thick: B held at 0), 2 left x face
// (vacuum at i - 1), 3 right x face
HD inline int face(const Metal3View& mv, bool me, int c, int i, int j, int k) {
    if (!me) return 0;
    auto m = [&](int ii) { return c == 0 ? mv.ey(ii, j, k) : mv.ez(ii, j, k); };
    if (!m(i)) return 0;
    const bool l = m(i - 1), r = m(i + 1);
    if (l == r) return 1;
    return !l ? 2 : 3;
}
}  // namespace

// max |f| over the whole array (a free function outside any class or unnamed namespace: nvcc refuses device lambdas in
// constructors)
double dielectric3_max_abs(const Field& f) {
    const FV X = f.view();
    return par_max_abs(long(f.size()), [=] HD(long q) { return X.p[q]; });
}

Dielectric3::Dielectric3(const std::vector<Dielectric3Config>& b, const Domain& d) : blocks(b) {
    if (d.n[0] < 2) throw std::runtime_error("dielectric: at least 2 x cells per rank");
    std::vector<double> h;
    std::vector<Expr::Ins> prog;
    for (const Dielectric3Config& c : blocks) {
        const bool fn = !c.fn.empty();
        if (!fn && !(c.eps >= 1)) throw std::runtime_error("dielectric " + c.name + ": eps >= 1 (lossless, non-dispersive)");
        if (fn && c.fn.max_stack() > Expr::MAX_STACK) throw std::runtime_error("dielectric " + c.name + ".eps: expression too deeply nested");
        const double off = double(prog.size()), len = fn ? double(c.fn.program().size()) : 0.0;
        if (fn) prog.insert(prog.end(), c.fn.program().begin(), c.fn.program().end());
        h.insert(h.end(), {fn ? 0.0 : c.eps - 1, off, len});
    }
    bk_.upload(h);
    if (!prog.empty()) { code_.resize(prog.size()); code_.from_host(prog.data(), 0, prog.size()); }
    for (Field* f : {&cx_, &cy_, &cz_}) f->resize(d);
    fill(d, -d.ng, d.n[0] + d.ng);
    max_chi = dielectric3_max_abs(cx_);
}

void Dielectric3::fill(const Domain& d, int i0, int i1) {
    if (i1 <= i0) return;
    const int gy = d.gw(1), gz = d.gw(2), ey = d.ext(1), ez = d.ext(2), nx = i1 - i0, nb = int(blocks.size());
    const long c0 = d.cell0(0), c1 = d.cell0(1), c2 = d.cell0(2), N0 = d.N[0], off = d.off[0];
    const bool p0 = d.periodic[0], flat = d.flat;
    const double h0 = d.h[0], h1 = d.h[1], h2 = d.h[2];
    // periodic axes: positions wrapped (x only in the window frame of a periodic box: no window then)
    const double L0 = p0 ? d.N[0] * h0 : 0.0, L1 = d.periodic[1] ? d.N[1] * h1 : 0.0, L2 = d.periodic[2] && !flat ? d.N[2] * h2 : 0.0;
    const double* B = bk_.data();
    const Expr::Ins* code = code_.data();
    const FV X = cx_.view(), Y = cy_.view(), Z = cz_.view();
    par_for(long(nx) * ey * ez, [=] HD(long q) {
        const int i = i0 + int(q % nx), j = int((q / nx) % ey) - gy, k = int(q / (long(nx) * ey)) - gz;
        // absolute node position (lab frame; periodic axes are not wrapped: the function is evaluated where the cell is)
        const double x = double(c0 + i) * h0, y = double(c1 + j) * h1, z = flat ? 0.0 : double(c2 + k) * h2;
        const double hz = flat ? 0.0 : 0.5 * h2;
        const double Lp[3] = {L0, L1, L2};
        // E_x at the node: [x -+ hx/2] x [y -+ hy/2] x [z -+ hz/2], harmonic along x
        const double sx = eps_box(B, nb, code, x - 0.5 * h0, x + 0.5 * h0, y - 0.5 * h1, y + 0.5 * h1, z - hz, z + hz, 0, flat, Lp) - 1;
        // J_y at (x + hx/2, y + hy/2, z): [x, x + hx] x [y, y + hy] x [z -+ hz/2], harmonic along y
        const double sy = eps_box(B, nb, code, x, x + h0, y, y + h1, z - hz, z + hz, 1, flat, Lp) - 1;
        // J_z at (x + hx/2, y, z + hz/2): [x, x + hx] x [y -+ hy/2] x [z, z + hz], harmonic along z
        const double sz = eps_box(B, nb, code, x, x + h0, y - 0.5 * h1, y + 0.5 * h1, z, flat ? z : z + h2, 2, flat, Lp) - 1;
        // open x: nothing beyond the global nodes [0, N); the last half point (to the guard node, held at 0) keeps its chi
        const long I = off + i;
        const bool out = !p0 && (I < 0 || I >= N0);
        X(i, j, k) = out ? 0.0 : sx; Y(i, j, k) = out ? 0.0 : sy; Z(i, j, k) = out ? 0.0 : sz;
    });
    const double low = par_max_abs(long(nx) * ey * ez, [=] HD(long q) {
        const int i = i0 + int(q % nx), j = int((q / nx) % ey) - gy, k = int(q / (long(nx) * ey)) - gz;
        return fmin(fmin(X(i, j, k), Y(i, j, k)), fmin(Z(i, j, k), 0.0));
    });
    if (low > 1e-12) throw std::runtime_error("dielectric: eps < 1 (lossless dielectrics only, eps >= 1)");
}

void Dielectric3::shift(const Domain& d, Field& tmp) {
    for (Field* f : {&cx_, &cy_, &cz_}) f->shift_x(tmp);
    fill(d, -d.ng, 1);                         // the left guards and the first column (the out-of-domain rule at rank 0)
    fill(d, d.n[0] - 2, d.n[0] + d.ng);        // the entering columns and the right guards
}

std::string Dielectric3::describe() const {
    std::string s;
    char b[512];
    for (const Dielectric3Config& c : blocks) {
        if (!c.fn.empty()) std::snprintf(b, sizeof b, "  dielectric %s: eps = %s (x, y, z in code units)\n", c.name.c_str(), c.fn.text().c_str());
        else std::snprintf(b, sizeof b, "  dielectric %s: eps = %.6g everywhere\n", c.name.c_str(), c.eps);
        s += b;
    }
    std::snprintf(b, sizeof b, "  dielectric: max chi %.4g; time-centred polarization current, lines along x (decision 100); chi per component from "
                  "4 x 4 x 4 sub-samples, harmonic along its direction, arithmetic across\n", max_chi);
    return s + b;
}

// The lines (component c: 0 E_y / B_z, 1 E_z / B_y; owned (j, k)) of rank p of P along x, as DielectricRZ::correct: the interior
// half points by Thomas with three right-hand sides, one interface unknown per rank (its last half point), the P x P
// tridiagonal (cyclic) system from 10 values per line and rank, gathered in the row of ranks along x.
void Dielectric3::correct(Comm& comm, const Domain& d, const Fields& f, Field& ney, Field& nez, Field& nby, Field& nbz) {
    const int n = d.n[0], ny = d.n[1], nz = d.n[2];
    const long plane = long(ny) * nz, L = 2 * plane;
    const int P = comm.dims[0], pr = comm.coords[0];
    const bool per = d.periodic[0];
    if (cpr_.size() != size_t(L * n)) {
        cpr_.resize(size_t(L * n)); x_.resize(size_t(3 * L * n)); y_.resize(size_t(L * (n + 1)));
        ex_.resize(size_t(10 * L)); ga_.resize(size_t(10 * L * P)); red_.resize(size_t(3 * L * P));
        exh_.resize(size_t(10 * L)); gah_.resize(size_t(10 * L * P));
    }
    const bool me = f.metal != nullptr;
    const Metal3View mv = me ? f.metal->view() : Metal3View{};
    const int lo = per ? -(1 << 29) : -d.off[0], hi = per ? (1 << 29) : d.N[0] - d.off[0];   // open x: the global nodes
    const FV Ey = f.ey.view(), Ez = f.ez.view(), nEy = ney.view(), nEz = nez.view(), CY = cy_.view(), CZ = cz_.view();
    double *gw = cpr_.data(), *xs = x_.data(), *ys = y_.data(), *ex = ex_.data();
    // A. the local solves of the interior and the edge values
    par_for(L, [=] HD(long q) {
        const int c = int(q / plane);
        const long jk = q % plane;
        const Line3 ln{c ? Ez : Ey, c ? nEz : nEy, c ? CZ : CY, mv, me, int(jk % ny), int(jk / ny), c, lo, hi};
        const int ni = n - 1;
        double* g = gw + q * n;
        double* x0 = xs + 3 * q * n;
        double* xl = x0 + n;
        double* xr = xl + n;
        double l, dd, u, r, bet = 0, ulast = 0;
        for (int a = 0; a < ni; ++a) {
            ln.coef(a, l, dd, u, r);
            if (a == ni - 1) ulast = u;
            const double rl = a == 0 ? -l : 0.0;
            if (a == 0) { bet = dd; x0[0] = r / bet; xl[0] = rl / bet; xr[0] = 0; g[0] = u; continue; }
            const double gg = g[a - 1] / bet;
            bet = dd - l * gg;
            x0[a] = (r - l * x0[a - 1]) / bet;
            xl[a] = (rl - l * xl[a - 1]) / bet;
            xr[a] = (0.0 - l * xr[a - 1]) / bet;
            g[a - 1] = gg; g[a] = u;
        }
        xr[ni - 1] += -ulast / bet;
        for (int a = ni - 2; a >= 0; --a) {
            x0[a] -= g[a] * x0[a + 1]; xl[a] -= g[a] * xl[a + 1]; xr[a] -= g[a] * xr[a + 1];
        }
        ln.coef(n - 1, l, dd, u, r);
        double* e = ex + 10 * q;
        e[0] = l; e[1] = dd; e[2] = u; e[3] = r;
        e[4] = x0[ni - 1]; e[5] = xl[ni - 1]; e[6] = xr[ni - 1];
        e[7] = x0[0]; e[8] = xl[0]; e[9] = xr[0];
    });
    // B. the edge values of the ranks along x (the same (j, k) range on every rank of the row)
    ex_.to_host(exh_.data(), 0, exh_.size());
    comm.allgather_x(exh_.data(), int(10 * L), gah_.data());
    ga_.from_host(gah_.data(), 0, gah_.size());
    const double* ga = ga_.data();
    double* red = red_.data();
    // C. per line the reduced system, then y of the own half points (y[0] = y_{-1})
    par_for(L, [=] HD(long q) {
        auto G = [&](int p, int c) { return ga[(long(p) * L + q) * 10 + c]; };
        double* pv = red + 3 * q * P;
        double* sv = pv + P;
        double* gm = sv + P;
        auto coefP = [&](int p, double& A, double& B, double& C, double& R) {
            const bool nx = per || p < P - 1;
            const int pn = (p + 1) % P;
            A = G(p, 0) * G(p, 5);
            B = G(p, 1) + G(p, 0) * G(p, 6) + (nx ? G(p, 2) * G(pn, 8) : 0.0);
            C = nx ? G(p, 2) * G(pn, 9) : 0.0;
            R = G(p, 3) - G(p, 0) * G(p, 4) - (nx ? G(p, 2) * G(pn, 7) : 0.0);
        };
        double A, B, C, R;
        if (!per) {
            double bet = 0, Cprev = 0;
            for (int p = 0; p < P; ++p) {
                coefP(p, A, B, C, R);
                if (p == 0) { bet = B; pv[0] = R / bet; }
                else { gm[p] = Cprev / bet; bet = B - A * gm[p]; pv[p] = (R - A * pv[p - 1]) / bet; }
                Cprev = C;
            }
            for (int p = P - 2; p >= 0; --p) pv[p] -= gm[p + 1] * pv[p + 1];
        } else if (P == 1) {
            coefP(0, A, B, C, R);
            pv[0] = R / (A + B + C);
        } else {
            double bet = 0, Cprev = 0;
            for (int p = 1; p < P; ++p) {
                coefP(p, A, B, C, R);
                const double s = -(p == 1 ? A : 0.0) - (p == P - 1 ? C : 0.0);
                if (p == 1) { bet = B; pv[1] = R / bet; sv[1] = s / bet; }
                else { gm[p] = Cprev / bet; bet = B - A * gm[p]; pv[p] = (R - A * pv[p - 1]) / bet; sv[p] = (s - A * sv[p - 1]) / bet; }
                Cprev = C;
            }
            for (int p = P - 2; p >= 1; --p) { pv[p] -= gm[p + 1] * pv[p + 1]; sv[p] -= gm[p + 1] * sv[p + 1]; }
            coefP(0, A, B, C, R);
            const double l0 = (R - A * pv[P - 1] - C * pv[1]) / (B + A * sv[P - 1] + C * sv[1]);
            pv[0] = l0;
            for (int p = 1; p < P; ++p) pv[p] += l0 * sv[p];
        }
        const double Lp = pr > 0 ? pv[pr - 1] : (per ? pv[P - 1] : 0.0), Ls = pv[pr];
        const double* x0 = xs + 3 * q * n;
        const double* xl = x0 + n;
        const double* xr = xl + n;
        double* y = ys + q * (n + 1);
        y[0] = Lp;
        for (int a = 0; a < n - 1; ++a) y[a + 1] = x0[a] + Lp * xl[a] + Ls * xr[a];
        y[n] = Ls;
    });
    // D. the owned nodes: E_y -= (ym + yp) / 2, B_z -= (ym - yp) / 2; E_z -= (ym + yp) / 2, B_y += (ym - yp) / 2;
    //    metal x faces: B from the characteristic of the vacuum side (metal3d.hpp), E stays 0
    const FV nBy = nby.view(), nBz = nbz.view();
    par_for(long(n) * plane, [=] HD(long w) {
        const int i = int(w % n);
        const long jk = w / n;
        const int j = int(jk % ny), k = int(jk / ny);
        {
            const double* y = ys + jk * (n + 1);
            const double ym = y[i], yp = y[i + 1];
            const int h = face(mv, me, 0, i, j, k);
            if (h == 0) { nEy(i, j, k) -= 0.5 * (ym + yp); nBz(i, j, k) -= 0.5 * (ym - yp); }
            else if (h == 2) nBz(i, j, k) -= ym;
            else if (h == 3) nBz(i, j, k) += yp;
        }
        {
            const double* y = ys + (plane + jk) * (n + 1);
            const double ym = y[i], yp = y[i + 1];
            const int h = face(mv, me, 1, i, j, k);
            if (h == 0) { nEz(i, j, k) -= 0.5 * (ym + yp); nBy(i, j, k) += 0.5 * (ym - yp); }
            else if (h == 2) nBy(i, j, k) += ym;
            else if (h == 3) nBy(i, j, k) -= yp;
        }
    });
}

// Moving window (RIP-CC): E_x of the column mi + 1 from the Gauss point (mi + 1/2) with div D at its initial value G0:
//   (1 + chi_x(mi+1)) E_x(mi+1) = (1 + chi_x(mi)) E_x(mi) + h_x (G0 - div_perp D(mi + 1/2)),
//   D_y(mi + 1/2, j + 1/2) = (1 + chi_y(mi)) (E_y(mi) + E_y(mi + 1)) / 2 (the J_y point), D_z likewise;
// sequentially from the front-layer start to the entering column (layer > 0), else the entering column only. Metal: E_x 0
// where the rule ex_zero says so.
void dielectric3_window_entry(Fields& f, const Domain& d, const Field& gauss0, int layer, const Dielectric3& diel) {
    const int nx = d.n[0], i0 = layer > 0 ? std::max(nx - 2 - layer, 0) : nx - 2;
    const int ny = d.n[1], nz = d.n[2];
    const double hx = d.h[0], iy = 1.0 / d.h[1], iz = 1.0 / d.h[2];
    const Diel3View dv = diel.view();
    const bool me = f.metal != nullptr;
    const Metal3View mv = me ? f.metal->view() : Metal3View{};
    const FV Ex = f.ex.view(), Ey = f.ey.view(), Ez = f.ez.view(), G0 = gauss0.view();
    par_for(long(ny) * nz, [=] HD(long q) {
        const int j = int(q % ny), k = int(q / ny);
        for (int mi = i0; mi <= nx - 2; ++mi) {
            auto Dy = [&](int jj) { return 0.5 * (1 + dv.cy(mi, jj, k)) * (Ey(mi, jj, k) + Ey(mi + 1, jj, k)); };
            auto Dz = [&](int kk) { return 0.5 * (1 + dv.cz(mi, j, kk)) * (Ez(mi, j, kk) + Ez(mi + 1, j, kk)); };
            const double t = (Dy(j) - Dy(j - 1)) * iy + (Dz(k) - Dz(k - 1)) * iz;
            double e = ((1 + dv.cx(mi, j, k)) * Ex(mi, j, k) + hx * (G0(mi, j, k) - t)) / (1 + dv.cx(mi + 1, j, k));
            if (me && mv.ex_zero(mi + 1, j, k)) e = 0;
            Ex(mi + 1, j, k) = e;
        }
    });
}

void dielectric3_absorb_front(Fields& f, const Domain& d, int cells, double kappa) {
    const int g = d.ng, gz = d.gw(2), sx = d.n[0] + 2 * g, sy = d.n[1] + 2 * g, sz = d.ext(2), N = d.N[0];
    const long c0 = d.cell0(0) - d.wx;          // window-relative global index
    const FV A = f.ex.view(), B = f.ey.view(), C = f.ez.view(), D = f.bx.view(), E = f.by.view(), F = f.bz.view();
    par_for(long(sx) * sy * sz, [=] HD(long q) {
        const int i = int(q % sx) - g, j = int((q / sx) % sy) - g, k = int(q / (long(sx) * sy)) - gz;
        const long gi = c0 + i;
        const double s = gi > N - 1 - cells ? double(gi - (N - 1 - cells)) / cells : 0.0;
        if (s <= 0) return;
        const double fa = exp(-kappa * s * s);
        A(i, j, k) *= fa; B(i, j, k) *= fa; C(i, j, k) *= fa; D(i, j, k) *= fa; E(i, j, k) *= fa; F(i, j, k) *= fa;
    });
}
