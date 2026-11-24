// Dielectric blocks in the RIP-CC RZ step (rz_dielectric.hpp, docs/specs/rz_dielectric.md, decision 100).
#include "rz_dielectric.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "comm.hpp"
#include "rz.hpp"

namespace {
// chi of the blocks at the point (r, z); per block at most its own chi (periodic: the copies one period away)
HD inline double chi_point(const double* B, int nb, bool per, double Lz, double r, double z) {
    double s = 0;
    for (int b = 0; b < nb; ++b) {
        const double* p = B + 6 * b;
        if (r < p[0] || r >= p[1]) continue;
        double f = 0;
        for (int w = per ? -1 : 0; w <= (per ? 1 : 0); ++w) {
            const double c = p[2] + w * Lz, e = p[3] + w * Lz, L = p[5];
            f += L > 0 ? fmin(fmax((z - c) / L + 0.5, 0.0), 1.0) * fmin(fmax((e - z) / L + 0.5, 0.0), 1.0) : (z >= c && z < e ? 1.0 : 0.0);
        }
        s += p[4] * fmin(f, 1.0);
    }
    return s;
}
// eps over the control volume [za, zb] x [ra, rb] of a field component (kind 0: E_z, 1: E_r, 2: E_phi; see prepare); inside
// or outside every block: exact, else 8 x 8 sub-samples
HD inline double eps_cv(const double* B, int nb, bool per, double Lz, double za, double zb, double ra, double rb, int kind) {
    bool cut = false;
    double inside = 0;
    for (int b = 0; b < nb && !cut; ++b) {
        const double* p = B + 6 * b;
        const double h = 0.5 * p[5];
        const bool rin = ra >= p[0] && rb <= p[1], rout = rb <= p[0] || ra >= p[1];
        bool zin = false, zout = true;
        for (int w = per ? -1 : 0; w <= (per ? 1 : 0); ++w) {
            const double c = p[2] + w * Lz, e = p[3] + w * Lz;
            zin = zin || (za >= c + h && zb <= e - h);
            zout = zout && (zb <= c - h || za >= e + h);
        }
        if (rout || zout) continue;
        if (rin && zin) inside += p[4];
        else cut = true;
    }
    if (!cut) return 1 + inside;
    const int S = 8;
    double acc = 0, wacc = 0;
    if (kind == 1) {                                  // E_r: harmonic in r (weight dr / r), arithmetic in z
        for (int a = 0; a < S; ++a) {
            const double z = za + (a + 0.5) * (zb - za) / S;
            double hs = 0, hw = 0;
            for (int c = 0; c < S; ++c) {
                const double r = ra + (c + 0.5) * (rb - ra) / S;
                hs += 1 / (r * (1 + chi_point(B, nb, per, Lz, r, z))); hw += 1 / r;
            }
            acc += hw / hs;
        }
        return acc / S;
    }
    for (int c = 0; c < S; ++c) {                     // E_z: harmonic in z, r-weighted in r; E_phi: arithmetic, r-weighted
        const double r = ra + (c + 0.5) * (rb - ra) / S;
        double v = 0;
        for (int a = 0; a < S; ++a) {
            const double e = 1 + chi_point(B, nb, per, Lz, r, za + (a + 0.5) * (zb - za) / S);
            v += kind == 0 ? 1 / e : e;
        }
        acc += r * (kind == 0 ? S / v : v / S); wacc += r;
    }
    return acc / wacc;
}
// the r-weighted fraction of [ra, rb] inside [c, e]
HD inline double rfrac(double ra, double rb, double c, double e) {
    const double lo = fmax(ra, c), hi = fmin(rb, e);
    return hi > lo ? (hi * hi - lo * lo) / (rb * rb - ra * ra) : 0.0;
}
// One line along z (component c: 0 E_r, 1 E_phi; row j; slot k): the coefficients (l, d, u, r) of half point a from the
// masks (0 where the metal holds E at 0, and E_phi on the axis) and the vacuum changes dEv of the nodes a, a + 1.
struct Line {
    FV E, nE;
    const double* chi;
    DielView dv;
    MetalView mv;
    long c0;
    int j, k, c;
    int lo, hi;                        // open z: the global nodes [lo, hi) in local indices (beyond: the guards, held at 0)
    HD double mask(int i) const {
        if (i < lo || i >= hi) return 0.0;
        if (c == 0) return mv.n && mv.half_at(c0 + i, j) ? 0.0 : 1.0;
        return j == 0 || (mv.n && mv.node_at(c0 + i, j)) ? 0.0 : 1.0;
    }
    HD void coef(int a, double& l, double& dd, double& u, double& r) const {
        const double ch = chi[dv.at(a, j)];
        if (ch == 0) { l = u = r = 0; dd = 1; return; }
        const double ma = mask(a), mb = mask(a + 1);
        l = 0.25 * ch * ma; u = 0.25 * ch * mb; dd = 1 + l + u;
        r = 0.5 * ch * (ma * (nE(a, j, k) - E(a, j, k)) + mb * (nE(a + 1, j, k) - E(a + 1, j, k)));
    }
};
// line q of rows x S per component
HD inline Line make_line(long q, int rows, int S, int J0, const FV& Er, const FV& Ep, const FV& nEr, const FV& nEp,
                         const DielView& dv, const MetalView& mv, long c0, int lo, int hi) {
    const long rs = long(rows) * S;
    const int c = int(q / rs), j = J0 + int((q % rs) % rows), k = int((q % rs) / rows);
    return Line{c ? Ep : Er, c ? nEp : nEr, c ? dv.cp : dv.cr, dv, mv, c0, j, k, c, lo, hi};
}
}  // namespace

DielectricRZ::DielectricRZ(const std::vector<DielectricConfig>& b, const Domain& d, const RadialGrid& g) : blocks(b) {
    nz_ = d.n[0]; nr_ = d.n[1];
    if (nz_ < 2) throw std::runtime_error("dielectric: at least 2 z cells per rank");
    std::vector<double> h;
    jlo = nr_; jhi = 0;
    for (const DielectricConfig& c : blocks) {
        if (!(c.eps >= 1)) throw std::runtime_error("dielectric " + c.name + ": eps >= 1 (lossless, non-dispersive)");
        if (!(c.r_out > c.r_in) || c.r_in < 0 || !(c.z_end > c.z_start))
            throw std::runtime_error("dielectric " + c.name + ": 0 <= r_in < r_out and z_start < z_end");
        h.insert(h.end(), {c.r_in, c.r_out, c.z_start, c.z_end, c.eps - 1, c.ramp});
        for (int j = 0; j < nr_; ++j) {             // rows whose node or half-node control volume meets the block
            const double na = j == 0 ? 0.0 : g.rh[size_t(j - 1)], nb = g.rh[size_t(j)];
            if (c.eps > 1 && (rfrac(na, nb, c.r_in, c.r_out) > 0 || rfrac(g.r[size_t(j)], g.r[size_t(j + 1)], c.r_in, c.r_out) > 0)) {
                jlo = std::min(jlo, j); jhi = std::max(jhi, j + 1);
            }
        }
    }
    if (jhi <= jlo) jlo = jhi = 0;
    bk_.upload(h);
    const size_t n = size_t(nz_ + 2) * nr_;
    cz_.resize(n); cr_.resize(n); cp_.resize(n);
    prepare(d, g);
}

void DielectricRZ::prepare(const Domain& d, const RadialGrid& g) {
    const long c0 = d.cell0(0);
    if (c0 == c0_) return;
    c0_ = c0;
    const int nz = nz_, nr = nr_, nb = int(blocks.size()), off = d.off[0], N = d.N[0];
    const double dz = d.h[0], Lz = N * dz;
    const bool per = d.periodic[0];
    const RadialView rg = g.view();
    const double* B = bk_.data();
    double *cz = cz_.data(), *cr = cr_.data(), *cp = cp_.data();
    const int J0 = jlo > 0 ? jlo - 1 : 0, J1 = jhi < nr ? jhi + 1 : nr;   // rows that can hold dielectric
    par_for(long(nz + 2) * nr, [=] HD(long q) {
        const int i = int(q % (nz + 2)) - 1, j = int(q / (nz + 2));
        if (j < J0 || j >= J1) { cz[q] = cr[q] = cp[q] = 0; return; }
        const double na = j == 0 ? 0.0 : rg.rh[j - 1], nbr = rg.rh[j], ha = rg.r[j], hb = rg.r[j + 1];
        const double zn = double(c0 + i) * dz;      // node i (lab frame); the half point a = i at zn + dz / 2
        // eps of each component over its control volume: E_z [zn -+ dz/2] x [rh_{j-1}, rh_j] harmonic along z (its normal) and
        // r-weighted arithmetic along r; E_r [zn, zn + dz] x [r_j, r_{j+1}] harmonic along r (weight 1/r: D_r r is constant
        // across a shell) and arithmetic along z; E_phi [zn, zn + dz] x [rh_{j-1}, rh_j] arithmetic (tangential to both)
        const double sz = eps_cv(B, nb, per, Lz, zn - 0.5 * dz, zn + 0.5 * dz, na, nbr, 0) - 1;
        const double sr = eps_cv(B, nb, per, Lz, zn, zn + dz, ha, hb, 1) - 1;
        const double sp = eps_cv(B, nb, per, Lz, zn, zn + dz, na, nbr, 2) - 1;
        // open z: nothing beyond the global nodes [0, N); the last half point (to the guard node N, held at 0) keeps its chi, so
        // a column entering a moving window brings P = chi E (E = 0 there) and the Gauss law holds across the shift
        const int I = off + i;
        const bool out = !per && (I < 0 || I >= N);
        cz[q] = out ? 0.0 : sz; cr[q] = out ? 0.0 : sr; cp[q] = out ? 0.0 : sp;
    });
}

std::string DielectricRZ::describe(const RadialGrid& g, double dz) const {
    std::string s;
    char b[512];
    for (const DielectricConfig& c : blocks) {
        std::snprintf(b, sizeof b, "  dielectric %s: eps %.6g, r %.6g .. %.6g, z %.6g .. %.6g (code units; dz %.4g; sub-cell average per component: harmonic along "
                      "its normal, arithmetic along the interface)\n", c.name.c_str(), c.eps, c.r_in, c.r_out, c.z_start, c.z_end, dz);
        s += b;
    }
    std::snprintf(b, sizeof b, "  dielectric: radial rows %d .. %d (r %.6g .. %.6g) solved along z (Crank-Nicolson polarization current, "
                  "decision 100)\n", jlo, jhi - 1, g.r[size_t(jlo)], g.r[size_t(jhi)]);
    return s + b;
}

// The lines (component c: 0 E_r / B_phi, 1 E_phi / B_r; row j; slot k) of rank p of P along z: the owned half points
// a = 0 .. n - 1 (a = i + 1/2). Partition: the interior a <= n - 2 with the Dirichlet values y_{-1} = L_{p-1} (the last
// value of the rank before) and y_{n-1} = L_p: y = x0 + L_{p-1} xL + L_p xR (three Thomas solves); the equations of the
// last points form the tridiagonal (cyclic if periodic) system A_p L_{p-1} + B_p L_p + C_p L_{p+1} = R_p of P unknowns
// per line, assembled on every rank from 10 values per line and rank (allgather).
void DielectricRZ::correct(Comm& comm, FieldsRZ& f) {
    const Domain& d = f.dom();
    const int n = d.n[0], S = d.n[2], rows = jhi - jlo, J0 = jlo;
    const long L = 2L * rows * S;
    if (L == 0) return;
    const int P = comm.dims[0], pr = comm.coords[0];
    const bool per = d.periodic[0];
    if (cpr_.size() != size_t(L * n)) {
        cpr_.resize(size_t(L * n)); x_.resize(size_t(3 * L * n)); y_.resize(size_t(L * (n + 1)));
        ex_.resize(size_t(10 * L)); ga_.resize(size_t(10 * L * P)); red_.resize(size_t(3 * L * P));
        exh_.resize(size_t(10 * L)); gah_.resize(size_t(10 * L * P));
    }
    const DielView dv = view();
    const MetalView mv = f.metal ? f.metal->view() : MetalView{};
    const int lo = per ? -(1 << 29) : -d.off[0], hi = per ? (1 << 29) : d.N[0] - d.off[0];   // open z: the global nodes
    const long c0 = d.cell0(0);
    const FV Er = f.er.view(), Ep = f.ep.view(), nEr = f.ner_.view(), nEp = f.nep_.view();
    double *gw = cpr_.data(), *xs = x_.data(), *ys = y_.data(), *ex = ex_.data();
    // A. the local solves of the interior and the edge values
    par_for(L, [=] HD(long q) {
        const Line ln = make_line(q, rows, S, J0, Er, Ep, nEr, nEp, dv, mv, c0, lo, hi);
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
            const double gg = g[a - 1] / bet;        // u_{a-1} / bet_{a-1}
            bet = dd - l * gg;
            x0[a] = (r - l * x0[a - 1]) / bet;
            xl[a] = (rl - l * xl[a - 1]) / bet;
            xr[a] = (0.0 - l * xr[a - 1]) / bet;
            g[a - 1] = gg; g[a] = u;
        }
        xr[ni - 1] += -ulast / bet;                  // the unit response to y_{n-1}: rhs -u_{ni-1} e_{ni-1}
        for (int a = ni - 2; a >= 0; --a) {
            x0[a] -= g[a] * x0[a + 1]; xl[a] -= g[a] * xl[a + 1]; xr[a] -= g[a] * xr[a + 1];
        }
        ln.coef(n - 1, l, dd, u, r);
        double* e = ex + 10 * q;
        e[0] = l; e[1] = dd; e[2] = u; e[3] = r;
        e[4] = x0[ni - 1]; e[5] = xl[ni - 1]; e[6] = xr[ni - 1];
        e[7] = x0[0]; e[8] = xl[0]; e[9] = xr[0];
    });
    // B. the edge values of all ranks
    ex_.to_host(exh_.data(), 0, exh_.size());
    comm.allgather(exh_.data(), int(10 * L), gah_.data());
    ga_.from_host(gah_.data(), 0, gah_.size());
    const double* ga = ga_.data();
    double* red = red_.data();
    // C. per line the reduced system on every rank, then y of the own half points (y[0] = y_{-1})
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
        if (!per) {                                  // Thomas over the ranks (L_{-1} = 0, nothing beyond the last)
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
        } else {                                     // cyclic: L_q = p_q + L_0 s_q for q >= 1, then the equation of rank 0
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
    // D. the owned nodes: E -= (y_{i-1/2} + y_{i+1/2}) / 2, B_phi -= (y_{i-1/2} - y_{i+1/2}) / 2, B_r += (y_{i-1/2} - y_{i+1/2}) / 2;
    //    metal faces: B from the characteristic of the vacuum side (rz_metal.hpp), E stays 0
    const FV nBp = f.nbp_.view(), nBr = f.nbr_.view();
    const long rs = long(rows) * S;
    par_for(long(n) * rs, [=] HD(long w) {
        const int i = int(w % n);
        const long qq = w / n;                       // the line of component 0; component 1 is qq + rs
        const int j = J0 + int(qq % rows), k = int(qq / rows);
        {
            const double* y = ys + qq * (n + 1);
            const double ym = y[i], yp = y[i + 1];
            const int h = mv.n ? mv.half_at(c0 + i, j) : 0;
            if (h == 0) { nEr(i, j, k) -= 0.5 * (ym + yp); nBp(i, j, k) -= 0.5 * (ym - yp); }
            else if (h == 2) nBp(i, j, k) -= ym;
            else if (h == 3) nBp(i, j, k) += yp;
        }
        if (j > 0) {
            const double* y = ys + (qq + rs) * (n + 1);
            const double ym = y[i], yp = y[i + 1];
            const int nd = mv.n ? mv.node_at(c0 + i, j) : 0;
            if (nd == 0) { nEp(i, j, k) -= 0.5 * (ym + yp); nBr(i, j, k) += 0.5 * (ym - yp); }
            else if (nd == 2) nBr(i, j, k) += ym;
            else if (nd == 3) nBr(i, j, k) -= yp;
        }
    });
}

// Moving window with dielectric blocks (RIP-CC): the entering last column mi + 1 = n - 1 has E_perp = 0; its E_z is set so that
// div D at the Gauss point (mi + 1/2, r_j) keeps its initial value G0 (as rz_window_entry does for iRIP with div E):
//   (1 + chi_z(mi+1)) Ez(mi+1) = (1 + chi_z(mi)) Ez(mi) + dz (G0 - div_perp D(mi + 1/2)),  D_perp(mi+1/2) = (1 + chi) E_perp(mi) / 2.
// Without it a field reaching the open front edge (the grid-scale branch of the scheme runs ahead of c in a dielectric, decision
// 100) leaves its charge in every column the window shifts in.
void rz_window_entry_dielectric(FieldsRZ& f, const Field& gauss0, int layer) {
    const Domain& d = f.dom();
    f.diel->prepare(d, f.grid());
    const DielView dv = f.diel->view();
    const MetalView mv = f.metal ? f.metal->view() : MetalView{};
    const double dz = d.h[0];
    const int nr = d.n[1], M = f.modes(), nz = d.n[0], i0 = layer > 0 ? (nz - 2 - layer > 0 ? nz - 2 - layer : 0) : nz - 2;
    const long c0 = d.cell0(0);
    const RadialView g = f.grid().view();
    const FV Ez = f.ez.view(), Er = f.er.view(), Ep = f.ep.view(), G0 = gauss0.view();
    par_for(long(nr) * M, [=] HD(long q) {
        const int j = int(q % nr), m = int(q / nr);
        if (m >= 1 && j == 0) return;
        auto sl = [&](int mm, int c) { return mm == 0 ? 0 : 2 * mm - 1 + c; };
        for (int mi = i0; mi <= nz - 2; ++mi) {                  // sequential along z: E_z(mi + 1) from the point mi + 1/2
            double t[2];
            for (int c = 0; c < (m ? 2 : 1); ++c) {               // Re, Im
                auto Dr = [&](int jj) { return 0.5 * (1 + dv.r(mi, jj)) * (Er(mi, jj, sl(m, c)) + Er(mi + 1, jj, sl(m, c))); };
                // (i m / r) D_phi: the real part of i x is -Im x
                const double dp = m ? 0.5 * (1 + dv.p(mi, j)) * (Ep(mi, j, sl(m, 1 - c)) + Ep(mi + 1, j, sl(m, 1 - c))) : 0.0;
                if (j == 0) t[c] = m == 0 ? g.rh[0] * Dr(0) / g.A[0] : 0.0;
                else t[c] = (g.rh[j] * Dr(j) - g.rh[j - 1] * Dr(j - 1)) / g.A[j] + (c == 0 ? -1.0 : 1.0) * g.mf(m) / g.r[j] * dp;
            }
            for (int c = 0; c < (m ? 2 : 1); ++c) {
                const int k = sl(m, c);
                double e = ((1 + dv.z(mi, j)) * Ez(mi, j, k) + dz * (G0(mi, j, k) - t[c])) / (1 + dv.z(mi + 1, j));
                if (mv.n && mv.node_at(c0 + mi + 1, j)) e = 0;
                Ez(mi + 1, j, k) = e;
            }
        }
    });
}
