// Cart3D current filter that keeps the Gauss law exact (filter.hpp, docs/specs/solid_target.md, decision 64).
// The Gauss nodes along an axis: all along a periodic axis; along an open one the global nodes [0, N), and along x
// [0, N - 1) (the last cell at an open right edge is no Gauss point, decision 21). Only faces between two Gauss nodes
// are changed, so no charge is moved across an open edge (a leaving particle's charge in the cell below the edge stays
// where its own current put it).
#include "filter.hpp"
#include <complex>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <stdexcept>
#include "xlines.hpp"

namespace {

struct GaussSet { int lo[3], hi[3]; };      // local node c is a Gauss node along a iff lo[a] <= c < hi[a]

GaussSet gauss_set(const Domain& d) {
    GaussSet G;
    for (int a = 0; a < 3; ++a) {
        if (d.periodic[a]) { G.lo[a] = -(1 << 29); G.hi[a] = 1 << 29; continue; }
        G.lo[a] = -d.off[a];
        G.hi[a] = d.N[a] - (a == 0 ? 1 : 0) - d.off[a];
    }
    return G;
}
HD inline bool gnode(const GaussSet& G, int i, int j, int k) {
    return i >= G.lo[0] && i < G.hi[0] && j >= G.lo[1] && j < G.hi[1] && k >= G.lo[2] && k < G.hi[2];
}

// w = D J on the owned nodes and one guard layer (J with valid guards), 0 at the nodes that are no Gauss nodes
void divergence(const Domain& d, const GaussSet G, const FV jx, const FV jy, const FV jz, const FV w) {
    const int gz = d.flat ? 0 : 1, e0 = d.n[0] + 2, e1 = d.n[1] + 2, e2 = d.n[2] + 2 * gz;   // flat: one plane
    const double i0 = 1 / d.h[0], i1 = 1 / d.h[1], i2 = 1 / d.h[2];
    par_for(long(e0) * e1 * e2, [=] HD(long q) {
        const int i = int(q % e0) - 1, j = int((q / e0) % e1) - 1, k = int(q / (long(e0) * e1)) - gz;
        w(i, j, k) = gnode(G, i, j, k) ? (jx(i + 1, j, k) - jx(i, j, k)) * i0 + (jy(i, j, k) - jy(i, j - 1, k)) * i1 +
                                             (jz(i, j, k) - jz(i, j, k - 1)) * i2
                                       : 0.0;
    });
}

// J_a += (h_a / 4) (w_+ - w_-) on the owned faces between two Gauss nodes; the face of J_a at local index c lies
// between the nodes c - 1 and c (x) or c and c + 1 (y, z)
void step_axis(const Domain& d, const GaussSet G, const FV J, const FV w, int a, double cf = 1.0) {
    const int n0 = d.n[0], n1 = d.n[1];
    const double q4 = 0.25 * cf * d.h[a];
    const int s0 = a == 0 ? -1 : 0, s1 = a == 0 ? 0 : 1;     // the two nodes of a face: c + s0 and c + s1 along a
    par_for(d.owned(), [=] HD(long q) {
        const int i = int(q % n0), j = int((q / n0) % n1), k = int(q / (long(n0) * n1));
        int m[3] = {i, j, k}, p[3] = {i, j, k};
        m[a] += s0; p[a] += s1;
        if (!gnode(G, m[0], m[1], m[2]) || !gnode(G, p[0], p[1], p[2])) return;
        J(i, j, k) += q4 * (w(p[0], p[1], p[2]) - w(m[0], m[1], m[2]));
    });
}

// out = T_a rho on the owned Gauss nodes (rho with valid guards), rho elsewhere
void binomial_axis(const Domain& d, const GaussSet G, const FV rho, const FV out, int a, double cf = 1.0) {
    const double qc = 0.25 * cf;
    const int n0 = d.n[0], n1 = d.n[1];
    par_for(d.owned(), [=] HD(long q) {
        const int i = int(q % n0), j = int((q / n0) % n1), k = int(q / (long(n0) * n1));
        const double r = rho(i, j, k);
        double v = r;
        if (gnode(G, i, j, k)) {
            int m[3] = {i, j, k}, p[3] = {i, j, k};
            m[a] -= 1; p[a] += 1;
            if (gnode(G, m[0], m[1], m[2])) v += qc * (rho(m[0], m[1], m[2]) - r);
            if (gnode(G, p[0], p[1], p[2])) v += qc * (rho(p[0], p[1], p[2]) - r);
        }
        out(i, j, k) = v;
    });
}

// w = d_a J_a (the a part of D J) on the owned nodes and one guard layer, 0 at the nodes that are no Gauss nodes
void divergence_axis(const Domain& d, const GaussSet G, const FV J, const FV w, int a) {
    const int gz = d.flat ? 0 : 1, e0 = d.n[0] + 2, e1 = d.n[1] + 2, e2 = d.n[2] + 2 * gz;
    const double ih = 1 / d.h[a];
    const int dx = a == 0, dy = a == 1, dz = a == 2;
    const int p0 = a == 0 ? 1 : 0;                       // Jx between the nodes c - 1 and c at c; Jy, Jz between c and c + 1 at c
    par_for(long(e0) * e1 * e2, [=] HD(long q) {
        const int i = int(q % e0) - 1, j = int((q / e0) % e1) - 1, k = int(q / (long(e0) * e1)) - gz;
        w(i, j, k) = gnode(G, i, j, k) ? (J(i + p0 * dx, j, k) - J(i - (1 - p0) * dx, j - dy, k - dz)) * ih : 0.0;
    });
}

// out = T_a X on the owned values of a component X != J_a: the binomial (1, 2, 1) / 4 along a over the Gauss range
// of a (Neumann at its ends; a value outside the range is copied). X lies at the a-position of the charge nodes, so
// T_a commutes with its difference along its own axis: D_b T_a J_b = T_a D_b J_b on every Gauss node.
void binomial_face(const Domain& d, const GaussSet G, const FV X, const FV out, int a, double cf = 1.0) {
    const int n0 = d.n[0], n1 = d.n[1];
    const double qf = 0.25 * cf;
    const int lo = G.lo[a], hi = G.hi[a];
    par_for(d.owned(), [=] HD(long q) {
        const int i = int(q % n0), j = int((q / n0) % n1), k = int(q / (long(n0) * n1));
        int c[3] = {i, j, k};
        const double x = X(i, j, k);
        double v = x;
        if (c[a] >= lo && c[a] < hi) {
            int m[3] = {i, j, k}, p[3] = {i, j, k};
            m[a] -= 1; p[a] += 1;
            if (m[a] >= lo) v += qf * (X(m[0], m[1], m[2]) - x);
            if (p[a] < hi) v += qf * (X(p[0], p[1], p[2]) - x);
        }
        out(i, j, k) = v;
    });
}

void copy_owned(const Domain& d, const FV from, const FV to) {
    const int n0 = d.n[0], n1 = d.n[1];
    par_for(d.owned(), [=] HD(long q) {
        const int i = int(q % n0), j = int((q / n0) % n1), k = int(q / (long(n0) * n1));
        to(i, j, k) = from(i, j, k);
    });
}

void zero_guards(const Domain& d, Field& f) {
    const FV v = f.view();
    const int n0 = d.n[0], n1 = d.n[1], n2 = d.n[2], g = d.ng, gz = d.gw(2), s0 = n0 + 2 * g, s1 = n1 + 2 * g;
    par_for(long(f.size()), [=] HD(long q) {
        const int i = int(q % s0) - g, j = int((q / s0) % s1) - g, k = int(q / (long(s0) * s1)) - gz;
        if (i < 0 || i >= n0 || j < 0 || j >= n1 || k < 0 || k >= n2) v(i, j, k) = 0.0;
    });
}

// psi on the Gauss nodes: J += G psi on the owned faces between two Gauss nodes, G = (dx, mu_x^2 dy, mu_x^2 dz) the
// gradient of RIP-CC's Gauss law (D G = dxx + mu_x^2 (dyy + dzz), the Laplacian of docs/specs/beam_self_field.md):
// Jx += (psi_c - psi_c-1) / hx; Jy, Jz += the (1, 2, 1) / 4 average along x of (psi_c+1 - psi_c) / h_a, psi reflected
// at an open x edge (psi_-1 = psi_0; psi_N-1 = psi_N-2 holds from the solve). The plain gradient (dx, dy, dz) is not
// free of curl for RIP-CC: it radiates (test S6).
void grad_axis(const Domain& d, const GaussSet G, const FV J, const FV psi, int a) {
    const int n0 = d.n[0], n1 = d.n[1];
    const double ih = 1 / d.h[a];
    const int s0 = a == 0 ? -1 : 0, s1 = a == 0 ? 0 : 1;
    const bool lo_open = !d.periodic[0] && d.off[0] == 0;
    par_for(d.owned(), [=] HD(long q) {
        const int i = int(q % n0), j = int((q / n0) % n1), k = int(q / (long(n0) * n1));
        int m[3] = {i, j, k}, p[3] = {i, j, k};
        m[a] += s0; p[a] += s1;
        if (!gnode(G, m[0], m[1], m[2]) || !gnode(G, p[0], p[1], p[2])) return;
        if (a == 0) { J(i, j, k) += (psi(p[0], p[1], p[2]) - psi(m[0], m[1], m[2])) * ih; return; }
        auto g = [&](int ii) { return psi(ii, p[1], p[2]) - psi(ii, m[1], m[2]); };
        const int il = lo_open && i == 0 ? 0 : i - 1;
        J(i, j, k) += 0.25 * (g(il) + 2 * g(i) + g(i + 1)) * ih;
    });
}

// L psi = r per transverse mode (qy, qz), L = dxx + mu_x^2 (dyy + dzz), scaled by hx^2: s (psi_i-1 + psi_i+1) + b psi_i =
// hx^2 r_i with s = 1 - hx^2 lambda / 4, b = -2 - hx^2 lambda / 2, lambda = 4/hy^2 sin^2(pi qy/ny) + 4/hz^2 sin^2(pi qz/nz)
// (s > 0 under RIP-CC's stability limit). Periodic x: cyclic. Open x: the Gauss nodes are [0, N - 1); Neumann at node 0
// (psi_-1 = psi_0: first diagonal b + s), and the last node N - 1 (no Gauss node, r = 0) gets psi = psi_N-2 (row
// s psi_N-2 - s psi_N-1 = 0), which is Neumann for node N - 2. The mode lambda = 0 is singular (psi + const): psi_0 is pinned
// to 0 (first diagonal 1e30); its row is then dropped, which holds since sum r = 0 (T keeps the charge); periodic x then
// is the open system with zeros beyond the last node (psi_N = psi_0 = 0).
struct ProjCoef {
    int ny, nz;
    double hx, hy, hz;
    bool cyc;
    HD xlines::LineCoef operator()(long M) const {
        const int qy = int(M % ny), qz = int(M / ny);
        const double sy = sin(M_PI * qy / ny), sz = sin(M_PI * qz / nz);
        const double lam = 4 * sy * sy / (hy * hy) + 4 * sz * sz / (hz * hz);
        const double s = 1 - 0.25 * hx * hx * lam, b = -2 - 0.5 * hx * hx * lam, pin = 1e30;
        if (cyc) return lam > 0 ? xlines::LineCoef{b, s, b, b, true} : xlines::LineCoef{b, s, pin, b, false};
        return {b, s, lam > 0 ? b + s : pin, -s, false};
    }
};

}  // namespace

struct FilterProjWork : xlines::LineWork {
    Field r, psi;
};

void filter_currents_proj(Fields& f, Field& w, std::shared_ptr<FilterProjWork>& W, const Domain& d, Comm& comm, int passes) {
    if (!W) {
        if (comm.dims[1] != 1 || comm.dims[2] != 1 || !d.periodic[1] || !d.periodic[2] || f.pml || f.side || d.n[0] < 3)
            throw std::runtime_error("numerics.filter = projection: y and z periodic (no pml, no side tube), ranks along x only, "
                                     "nx >= 3 per rank (numerics.filter = local otherwise)");
        W = std::make_shared<FilterProjWork>();
        W->resize(d);
        W->r.resize(d); W->psi.resize(d);
    }
    const GaussSet G = gauss_set(d);
    Field* J[3] = {&f.jx, &f.jy, &f.jz};
    const int nx = d.n[0], ny = d.n[1], nz = d.n[2];
    const double hx2 = d.h[0] * d.h[0], norm = 1.0 / (double(ny) * nz);
    double* sp = W->w.data();
    double* t = W->t.data();
    const double* ty = W->twy.data();
    const double* tz = W->twz.data();
    for (int p = 0; p < passes; ++p) {
        comm.exchange(d, {&f.jx, &f.jy, &f.jz});
        divergence(d, G, f.jx.view(), f.jy.view(), f.jz.view(), w.view());
        // r = T w (the binomial of filter_rho_cart, axis by axis), then hx^2 (T w - w) into the spectrum
        copy_owned(d, w.view(), W->r.view());
        for (int a = 0; a < (d.flat ? 2 : 3); ++a) {
            comm.exchange(d, {&W->r});
            binomial_axis(d, G, W->r.view(), W->psi.view(), a);
            copy_owned(d, W->psi.view(), W->r.view());
        }
        const FV R = W->r.view(), Wv = w.view(), Psi = W->psi.view();
        for_owned(d, [=] HD(int i, int j, int k) {
            const long q = 2 * ((long(k) * ny + j) * nx + i);
            sp[q] = hx2 * (R(i, j, k) - Wv(i, j, k)); sp[q + 1] = 0;
        });
        xlines::transform_y(sp, t, nx, ny, nz, nullptr, ty, -1);
        xlines::transform_z(sp, t, nx, ny, nz, nullptr, tz, -1);
        xlines::solve_lines(*W, d, comm, ProjCoef{ny, nz, d.h[0], d.h[1], d.h[2], d.periodic[0]});
        xlines::transform_z(sp, t, nx, ny, nz, nullptr, tz, +1);
        xlines::transform_y(sp, t, nx, ny, nz, nullptr, ty, +1);
        for_owned(d, [=] HD(int i, int j, int k) { Psi(i, j, k) = sp[2 * ((long(k) * ny + j) * nx + i)] * norm; });
        comm.exchange(d, {&W->psi});
        for (int a = 0; a < (d.flat ? 2 : 3); ++a) grad_axis(d, G, J[a]->view(), Psi, a);
    }
    for (Field* x : J) zero_guards(d, *x);
}

void filter_currents_cart(Fields& f, Field& w, const Domain& d, Comm& comm, int passes) {
    const GaussSet G = gauss_set(d);
    Field* J[3] = {&f.jx, &f.jy, &f.jz};
    for (int p = 0; p < passes; ++p)
        for (int a = 0; a < (d.flat ? 2 : 3); ++a) {   // flat: d/dz = 0, the z pass would change nothing
            comm.exchange(d, {&f.jx, &f.jy, &f.jz});
            divergence(d, G, f.jx.view(), f.jy.view(), f.jz.view(), w.view());
            step_axis(d, G, J[a]->view(), w.view(), a);
        }
    for (Field* x : J) zero_guards(d, *x);
}

// One step along a with coefficient cf: J_a <- J_a + cf (S_a - 1) J_a, J_b <- J_b + cf (T_a - 1) J_b. cf = 1 is a binomial
// pass; after N passes cf = -N is the compensator (Lee and Vay): the response T^N ((N + 1) - N T), flat to O(k^4), zero
// at Nyquist. Any polynomial in S_a keeps d_a P(S_a) = P(T_a) d_a, so the Gauss law stays exact (bench/nci/mechanism).
static void binomial_step(Fields& f, Field& w, const Domain& d, Comm& comm, const GaussSet& G, int a, double cf) {
    Field* J[3] = {&f.jx, &f.jy, &f.jz};
    comm.exchange(d, {&f.jx, &f.jy, &f.jz});
    divergence_axis(d, G, J[a]->view(), w.view(), a);
    step_axis(d, G, J[a]->view(), w.view(), a, cf);       // S_a J_a: d_a S_a J_a = T_a d_a J_a
    for (int b = 0; b < 3; ++b) {
        if (b == a) continue;
        binomial_face(d, G, J[b]->view(), w.view(), a, cf);  // T_a J_b (guards of J_b valid: exchanged above)
        copy_owned(d, w.view(), J[b]->view());
    }
}

void set_flat_filter(Domain& d, int passes, int L) {
    d.filt_nflat = 0;
    if (L <= 0) return;
    if (L < 2 || L > 9) throw std::runtime_error("numerics.filter_flat: 2 .. 9");
    const int n = L - 1;                                  // degree of P
    std::vector<double> c(L);
    for (int j = 0; j < L; ++j) {                         // binom(N - 1 + j, j)
        double b = 1;
        for (int i = 1; i <= j; ++i) b = b * (passes - 1 + i) / i;
        c[j] = b;
    }
    using cd = std::complex<double>;
    std::vector<cd> r(n);
    for (int i = 0; i < n; ++i) r[i] = std::polar(1.0 / passes, 0.4 + 2 * M_PI * i / n);
    auto P = [&](cd x) { cd v = 0; for (int j = n; j >= 0; --j) v = v * x + c[j]; return v; };
    for (int it = 0; it < 500; ++it) {                    // Durand-Kerner
        double moved = 0;
        for (int i = 0; i < n; ++i) {
            cd den = c[n];
            for (int j = 0; j < n; ++j) if (j != i) den *= r[i] - r[j];
            const cd dr = P(r[i]) / den;
            r[i] -= dr; moved = std::max(moved, std::abs(dr) / std::abs(r[i]));
        }
        if (moved < 1e-15) break;
    }
    std::vector<bool> used(n, false);
    for (int i = 0; i < n; ++i) {
        if (used[i]) continue;
        used[i] = true;
        const cd u = 1.0 / r[i];
        if (std::abs(r[i].imag()) < 1e-9 * std::abs(r[i])) {          // real root: (1 - S / r)
            d.filt_fk[d.filt_nflat] = 1; d.filt_fa[d.filt_nflat] = u.real(); d.filt_fb[d.filt_nflat] = 0; ++d.filt_nflat;
            continue;
        }
        int m = -1; double best = 1e300;                                // its conjugate
        for (int j = 0; j < n; ++j)
            if (!used[j] && std::abs(r[j] - std::conj(r[i])) < best) { best = std::abs(r[j] - std::conj(r[i])); m = j; }
        if (m < 0 || best > 1e-6 * std::abs(r[i])) throw std::runtime_error("numerics.filter_flat: root pairing failed");
        used[m] = true;
        d.filt_fk[d.filt_nflat] = 2; d.filt_fa[d.filt_nflat] = 2 * u.real(); d.filt_fb[d.filt_nflat] = std::norm(u); ++d.filt_nflat;
    }
}

double flat_filter_response(const Domain& d, int passes, double S) {
    double t = std::pow(1 - S, passes);
    for (int s = 0; s < d.filt_nflat; ++s)
        t *= d.filt_fk[s] == 1 ? 1 - d.filt_fa[s] * S : 1 - d.filt_fa[s] * S + d.filt_fb[s] * S * S;
    return t;
}

void filter_currents_binomial(Fields& f, Field& w, const Domain& d, Comm& comm, int passes) {
    const GaussSet G = gauss_set(d);
    Field* J[3] = {&f.jx, &f.jy, &f.jz};
    const int na = d.filt_xonly ? 1 : d.flat ? 2 : 3;    // flat: d/dz = 0, a z pass would change nothing
    for (int p = 0; p < passes; ++p)
        for (int a = 0; a < na; ++a) binomial_step(f, w, d, comm, G, a, 1.0);
    if (d.filt_comp && passes > 0) binomial_step(f, w, d, comm, G, 0, -double(passes));
    if (d.filt_nflat > 0 && passes > 0)                  // numerics.filter_flat (decision 109)
        flat_filter_steps(d, J, 3, [&](double cf) { binomial_step(f, w, d, comm, G, 0, cf); });
    for (Field* x : J) zero_guards(d, *x);
}

void filter_rho_cart(Field& rho, Field& w, const Domain& d, Comm& comm, int passes) {
    const GaussSet G = gauss_set(d);
    const int na = d.filt_xonly ? 1 : d.flat ? 2 : 3;    // flat: d/dz = 0, the z pass would change nothing
    for (int p = 0; p <= passes; ++p) {
        const bool comp = p == passes;                   // the compensator (numerics.filter_compensate) after the passes
        if (comp && !(d.filt_comp && passes > 0)) break;
        for (int a = 0; a < (comp ? 1 : na); ++a) {
            comm.exchange(d, {&rho});
            binomial_axis(d, G, rho.view(), w.view(), a, comp ? -double(passes) : 1.0);
            copy_owned(d, w.view(), rho.view());
        }
    }
    if (d.filt_nflat > 0 && passes > 0) {                // numerics.filter_flat (decision 109)
        Field* X[1] = {&rho};
        flat_filter_steps(d, X, 1, [&](double cf) {
            comm.exchange(d, {&rho});
            binomial_axis(d, G, rho.view(), w.view(), 0, cf);
            copy_owned(d, w.view(), rho.view());
        });
    }
    zero_guards(d, rho);
}
