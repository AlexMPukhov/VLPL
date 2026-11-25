// Dielectric blocks in the RIP-CC RZ step (rz_dielectric.hpp, docs/specs/rz_dielectric.md, decision 100).
#include "rz_dielectric.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "comm.hpp"
#include "rz.hpp"

namespace {
constexpr int BS = 8;                  // block record: r_in, r_out, z_start, z_end, chi, ramp, code offset, code length (0: none)
// chi of the blocks at the point (r, z, phi); per block at most its own chi (periodic: the copies one period away). A function
// block: eps(x, y, z, r, phi) - 1 inside its bounding box (periodic z: the function of z wrapped into [0, Lz))
HD inline double chi_point(const double* B, int nb, bool per, double Lz, double r, double z, const Expr::Ins* code, double phi = 0) {
    double s = 0;
    for (int b = 0; b < nb; ++b) {
        const double* p = B + BS * b;
        if (r < p[0] || r >= p[1]) continue;
        if (p[7] > 0) {
            const double zz = per ? z - Lz * floor(z / Lz) : z;
            if (zz < p[2] || zz >= p[3]) continue;
            const double v[5] = {r * cos(phi), r * sin(phi), zz, r, phi};
            s += expr_eval(code + long(p[6]), int(p[7]), v) - 1;
            continue;
        }
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
HD inline double eps_cv(const double* B, int nb, bool per, double Lz, double za, double zb, double ra, double rb, int kind,
                        const Expr::Ins* code) {
    bool cut = false;
    double inside = 0;
    for (int b = 0; b < nb && !cut; ++b) {
        const double* p = B + BS * b;
        if (p[7] > 0) {                               // a function: sampled wherever the box meets the control volume
            if (rb <= p[0] || ra >= p[1]) continue;
            if (!per && (zb <= p[2] || za >= p[3])) continue;
            cut = true;
            continue;
        }
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
                hs += 1 / (r * (1 + chi_point(B, nb, per, Lz, r, z, code))); hw += 1 / r;
            }
            acc += hw / hs;
        }
        return acc / S;
    }
    for (int c = 0; c < S; ++c) {                     // E_z: harmonic in z, r-weighted in r; E_phi: arithmetic, r-weighted
        const double r = ra + (c + 0.5) * (rb - ra) / S;
        double v = 0;
        for (int a = 0; a < S; ++a) {
            const double e = 1 + chi_point(B, nb, per, Lz, r, za + (a + 0.5) * (zb - za) / S, code);
            v += kind == 0 ? 1 / e : e;
        }
        acc += r * (kind == 0 ? S / v : v / S); wacc += r;
    }
    return acc / wacc;
}
// chi per angle: eps over the control volume [za, zb] x [ra, rb] x [pa, pb] (the azimuth of the component's angle); where no
// function block meets it, the axisymmetric eps_cv; else 8 x 8 x 4 sub-samples, harmonic along the component's own direction
// (E_z: z, E_r: r with weight 1/r, E_phi: phi), arithmetic (r-weighted) across
HD inline double eps_cv_ang(const double* B, int nb, bool per, double Lz, double za, double zb, double ra, double rb, double pa, double pb,
                            int kind, const Expr::Ins* code) {
    bool fn = false;
    for (int b = 0; b < nb && !fn; ++b) {
        const double* p = B + BS * b;
        if (p[7] > 0 && !(rb <= p[0] || ra >= p[1]) && (per || !(zb <= p[2] || za >= p[3]))) fn = true;
    }
    if (!fn) return eps_cv(B, nb, per, Lz, za, zb, ra, rb, kind, code);
    const int S = 8, SP = 4;
    if (kind == 2) {                                  // E_phi: harmonic over the azimuth per (r, z), r-weighted arithmetic across
        double acc = 0, wacc = 0;
        for (int c = 0; c < S; ++c) {
            const double r = ra + (c + 0.5) * (rb - ra) / S;
            for (int a = 0; a < S; ++a) {
                const double z = za + (a + 0.5) * (zb - za) / S;
                double h = 0;
                for (int e = 0; e < SP; ++e) h += 1 / (1 + chi_point(B, nb, per, Lz, r, z, code, pa + (e + 0.5) * (pb - pa) / SP));
                acc += r * SP / h; wacc += r;
            }
        }
        return acc / wacc;
    }
    if (kind == 1) {                                  // E_r: harmonic in r (weight 1/r) per (z, phi), arithmetic across
        double t = 0;
        for (int a = 0; a < S; ++a) {
            const double z = za + (a + 0.5) * (zb - za) / S;
            for (int e = 0; e < SP; ++e) {
                const double ph = pa + (e + 0.5) * (pb - pa) / SP;
                double hs = 0, hw = 0;
                for (int c = 0; c < S; ++c) {
                    const double r = ra + (c + 0.5) * (rb - ra) / S;
                    hs += 1 / (r * (1 + chi_point(B, nb, per, Lz, r, z, code, ph))); hw += 1 / r;
                }
                t += hw / hs;
            }
        }
        return t / (S * SP);
    }
    double t = 0, tw = 0;                             // E_z: harmonic in z per (r, phi), r-weighted arithmetic across
    for (int c = 0; c < S; ++c) {
        const double r = ra + (c + 0.5) * (rb - ra) / S;
        for (int e = 0; e < SP; ++e) {
            const double ph = pa + (e + 0.5) * (pb - pa) / SP;
            double v = 0;
            for (int a = 0; a < S; ++a) v += 1 / (1 + chi_point(B, nb, per, Lz, r, za + (a + 0.5) * (zb - za) / S, code, ph));
            t += r * S / v; tw += r;
        }
    }
    return t / tw;
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
    MetalPhiView pv;                   // block metal: k is the angle (E, nE hold the angle values)
    long c0;
    int j, k, c;
    int lo, hi;                        // open z: the global nodes [lo, hi) in local indices (beyond: the guards, held at 0)
    HD double mask(int i) const {
        if (i < lo || i >= hi) return 0.0;
        if (pv.np) {                   // E_r at (j+1/2, k): Mh; E_phi at (j, k+1/2): Mp (and the axis)
            if (c == 0) return pv.Mh(c0 + i, j, k) ? 0.0 : 1.0;
            return j == 0 || pv.Mp(c0 + i, j, k) ? 0.0 : 1.0;
        }
        if (c == 0) return mv.n && mv.half_at(c0 + i, j) ? 0.0 : 1.0;
        return j == 0 || (mv.n && mv.node_at(c0 + i, j)) ? 0.0 : 1.0;
    }
    HD void coef(int a, double& l, double& dd, double& u, double& r) const {
        const double ch = chi[dv.np ? dv.at(a, j, k) : dv.at(a, j)];   // chi per angle: k is the angle
        if (ch == 0) { l = u = r = 0; dd = 1; return; }
        const double ma = mask(a), mb = mask(a + 1);
        l = 0.25 * ch * ma; u = 0.25 * ch * mb; dd = 1 + l + u;
        r = 0.5 * ch * (ma * (nE(a, j, k) - E(a, j, k)) + mb * (nE(a + 1, j, k) - E(a + 1, j, k)));
    }
};
// line q of rows x S per component
HD inline Line make_line(long q, int rows, int S, int J0, const FV& Er, const FV& Ep, const FV& nEr, const FV& nEp,
                         const DielView& dv, const MetalView& mv, const MetalPhiView& pv, long c0, int lo, int hi) {
    const long rs = long(rows) * S;
    const int c = int(q / rs), j = J0 + int((q % rs) % rows), k = int((q % rs) / rows);
    return Line{c ? Ep : Er, c ? nEp : nEr, c ? dv.cp : dv.cr, dv, mv, pv, c0, j, k, c, lo, hi};
}
}  // namespace

DielectricRZ::DielectricRZ(const std::vector<DielectricConfig>& b, const Domain& d, const RadialGrid& g) : blocks(b) {
    nz_ = d.n[0]; nr_ = d.n[1];
    if (nz_ < 2) throw std::runtime_error("dielectric: at least 2 z cells per rank");
    std::vector<double> h;
    std::vector<Expr::Ins> prog;
    jlo = nr_; jhi = 0;
    for (const DielectricConfig& c : blocks) {
        const bool fn = !c.fn.empty();
        if (!fn && !(c.eps >= 1)) throw std::runtime_error("dielectric " + c.name + ": eps >= 1 (lossless, non-dispersive)");
        if (!(c.r_out > c.r_in) || c.r_in < 0 || !(c.z_end > c.z_start))
            throw std::runtime_error("dielectric " + c.name + ": 0 <= r_in < r_out and z_start < z_end");
        if (fn && c.ramp > 0) throw std::runtime_error("dielectric " + c.name + ": ramp only with a constant eps (a function can ramp itself)");
        if (fn && c.fn.max_stack() > Expr::MAX_STACK) throw std::runtime_error("dielectric " + c.name + ".eps: expression too deeply nested");
        const double off = double(prog.size()), len = fn ? double(c.fn.program().size()) : 0.0;
        if (fn) prog.insert(prog.end(), c.fn.program().begin(), c.fn.program().end());
        h.insert(h.end(), {c.r_in, c.r_out, c.z_start, c.z_end, fn ? 0.0 : c.eps - 1, c.ramp, off, len});
        fn_ = fn_ || fn;
        for (int j = 0; j < nr_; ++j) {             // rows whose node or half-node control volume meets the block
            const double na = j == 0 ? 0.0 : g.rh[size_t(j - 1)], nb = g.rh[size_t(j)];
            if ((fn || c.eps > 1) && (rfrac(na, nb, c.r_in, c.r_out) > 0 || rfrac(g.r[size_t(j)], g.r[size_t(j + 1)], c.r_in, c.r_out) > 0)) {
                jlo = std::min(jlo, j); jhi = std::max(jhi, j + 1);
            }
        }
    }
    if (jhi <= jlo) jlo = jhi = 0;
    for (const DielectricConfig& c : blocks) phi_ = phi_ || (!c.fn.empty() && c.phi);
    bk_.upload(h);
    if (!prog.empty()) { code_.resize(prog.size()); code_.from_host(prog.data(), 0, prog.size()); }
    const size_t n = size_t(nz_ + 2) * nr_;
    cz_.resize(n); cr_.resize(n); cp_.resize(n);
    if (fn_) { tz_.resize(n); tr_.resize(n); tp_.resize(n); }
    if (!phi_) prepare(d, g);          // chi per angle: after set_angles
}

void DielectricRZ::set_angles(const Domain& d, const RadialGrid& g, int M, bool staggered) {
    if (!phi_) return;
    if (2 * M - 1 > RZ_PHI_MAXNP) throw std::runtime_error("dielectric: eps depending on phi with at most 128 modes");
    M_ = M; np_ = 2 * M - 1; stag_ = staggered;
    std::vector<double> cs(size_t(np_) * M_), sn(cs.size()), ch(cs.size()), sh(cs.size());
    const double off = stag_ ? 0.5 : 0.0;    // the angles of E_phi (as MetalPhi)
    for (int k = 0; k < np_; ++k)
        for (int m = 0; m < M_; ++m) {
            const double a = 2 * M_PI * double(k) * m / np_;
            cs[size_t(k) * M_ + m] = std::cos(a);
            sn[size_t(k) * M_ + m] = std::sin(a);
            ch[size_t(k) * M_ + m] = std::cos(2 * M_PI * (k + off) * m / np_);
            sh[size_t(k) * M_ + m] = std::sin(2 * M_PI * (k + off) * m / np_);
        }
    cs_.upload(cs); sn_.upload(sn); ch_.upload(ch); sh_.upload(sh);
    const size_t n = size_t(nz_ + 2) * nr_ * np_;
    cz_.resize(n); cr_.resize(n); cp_.resize(n);
    tz_.resize(n); tr_.resize(n); tp_.resize(n);
    c0_ = -(1L << 60);
    prepare(d, g);
}

// chi of the local columns i0 .. i1 (the window position of d), on every angle if chi depends on phi
void DielectricRZ::fill(const Domain& d, const RadialGrid& g, int i0, int i1) {
    const long c0 = d.cell0(0);
    const int nz = nz_, nr = nr_, nb = int(blocks.size()), off = d.off[0], N = d.N[0], nc = i1 - i0 + 1;
    if (nc <= 0) return;
    const double dz = d.h[0], Lz = N * dz;
    const bool per = d.periodic[0];
    const RadialView rg = g.view();
    const double* B = bk_.data();
    const Expr::Ins* code = code_.data();
    double *cz = cz_.data(), *cr = cr_.data(), *cp = cp_.data();
    const int J0 = jlo > 0 ? jlo - 1 : 0, J1 = jhi < nr ? jhi + 1 : nr;   // rows that can hold dielectric
    const int np = np_, na_ = np > 0 ? np : 1;
    const double dph = np > 0 ? 2 * M_PI / np : 0.0, poff = stag_ ? 0.5 : 0.0;
    par_for(long(nc) * nr * na_, [=] HD(long w) {
        const int i = i0 + int(w % nc), j = int((w / nc) % nr), k = int(w / (long(nc) * nr));
        const long q = long(i + 1) + long(nz + 2) * (j + long(nr) * k);
        if (j < J0 || j >= J1) { cz[q] = cr[q] = cp[q] = 0; return; }
        const double na = j == 0 ? 0.0 : rg.rh[j - 1], nbr = rg.rh[j], ha = rg.r[j], hb = rg.r[j + 1];
        const double zn = double(c0 + i) * dz;      // node i (lab frame); the half point a = i at zn + dz / 2
        // eps of each component over its control volume: E_z [zn -+ dz/2] x [rh_{j-1}, rh_j] harmonic along z (its normal) and
        // r-weighted arithmetic along r; E_r [zn, zn + dz] x [r_j, r_{j+1}] harmonic along r (weight 1/r: D_r r is constant
        // across a shell) and arithmetic along z; E_phi [zn, zn + dz] x [rh_{j-1}, rh_j] arithmetic (tangential to both);
        // per angle: the azimuth [phi_k -+ dphi / 2] too (E_phi: its own angle, harmonic over it)
        double sz, sr, sp;
        if (np == 0) {
            sz = eps_cv(B, nb, per, Lz, zn - 0.5 * dz, zn + 0.5 * dz, na, nbr, 0, code) - 1;
            sr = eps_cv(B, nb, per, Lz, zn, zn + dz, ha, hb, 1, code) - 1;
            sp = eps_cv(B, nb, per, Lz, zn, zn + dz, na, nbr, 2, code) - 1;
        } else {
            const double pk = k * dph, pq = (k + poff) * dph;
            sz = eps_cv_ang(B, nb, per, Lz, zn - 0.5 * dz, zn + 0.5 * dz, na, nbr, pk - 0.5 * dph, pk + 0.5 * dph, 0, code) - 1;
            sr = eps_cv_ang(B, nb, per, Lz, zn, zn + dz, ha, hb, pk - 0.5 * dph, pk + 0.5 * dph, 1, code) - 1;
            sp = eps_cv_ang(B, nb, per, Lz, zn, zn + dz, na, nbr, pq - 0.5 * dph, pq + 0.5 * dph, 2, code) - 1;
        }
        // open z: nothing beyond the global nodes [0, N); the last half point (to the guard node N, held at 0) keeps its chi, so
        // a column entering a moving window brings P = chi E (E = 0 there) and the Gauss law holds across the shift
        const int I = off + i;
        const bool out = !per && (I < 0 || I >= N);
        cz[q] = out ? 0.0 : sz; cr[q] = out ? 0.0 : sr; cp[q] = out ? 0.0 : sp;
    });
    if (fn_) {                         // eps >= 1 everywhere (a function may go below 1)
        const double low = par_max_abs(long(nc) * nr * na_, [=] HD(long w) {
            const long q = long(i0 + int(w % nc) + 1) + long(nz + 2) * (int((w / nc) % nr) + long(nr) * int(w / (long(nc) * nr)));
            return fmin(fmin(cz[q], cr[q]), fmin(cp[q], 0.0));
        });
        if (low > 1e-12) throw std::runtime_error("dielectric: eps < 1 from an eps function (lossless dielectrics only, eps >= 1)");
    }
}

void DielectricRZ::prepare(const Domain& d, const RadialGrid& g) {
    const long c0 = d.cell0(0);
    if (c0 == c0_) return;
    if (phi_ && np_ == 0) throw std::runtime_error("dielectric: eps depending on phi needs the angles (DielectricRZ::set_angles)");
    if (fn_ && c0 == c0_ + 1) {        // a window shift: the columns move by one; the edge columns (out-of-domain rule) and the new
        const int nz = nz_;            // last ones are filled (an eps function is sampled only there)
        const long rows = long(nr_) * (np_ > 0 ? np_ : 1);
        double *cz = cz_.data(), *cr = cr_.data(), *cp = cp_.data(), *tz = tz_.data(), *tr = tr_.data(), *tp = tp_.data();
        par_for(long(nz + 1) * rows, [=] HD(long w) {
            const long q = long(w % (nz + 1)) + long(nz + 2) * (w / (nz + 1));
            tz[q] = cz[q + 1]; tr[q] = cr[q + 1]; tp[q] = cp[q + 1];
        });
        par_for(long(nz + 1) * rows, [=] HD(long w) {
            const long q = long(w % (nz + 1)) + long(nz + 2) * (w / (nz + 1));
            cz[q] = tz[q]; cr[q] = tr[q]; cp[q] = tp[q];
        });
        c0_ = c0;
        fill(d, g, -1, 0);
        fill(d, g, nz - 2, nz);
        return;
    }
    c0_ = c0;
    fill(d, g, -1, nz_);
}

std::string DielectricRZ::describe(const RadialGrid& g, double dz) const {
    std::string s;
    char b[512];
    for (const DielectricConfig& c : blocks) {
        if (!c.fn.empty()) {
            std::snprintf(b, sizeof b, "  dielectric %s: eps = %s (z, r in code units)", c.name.c_str(), c.fn.text().c_str());
            s += b;
            if (c.r_in > 0 || c.r_out < 1e299 || c.z_start > -1e299 || c.z_end < 1e299) {
                std::snprintf(b, sizeof b, " inside r %.6g .. %.6g, z %.6g .. %.6g", c.r_in, c.r_out, c.z_start, c.z_end);
                s += b;
            }
            std::snprintf(b, sizeof b, " (8 x 8 sub-samples per control volume: harmonic along each component's direction, arithmetic across)\n");
            s += b;
            continue;
        }
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
    const MetalPhiView pv = f.metal_phi ? f.metal_phi->view() : MetalPhiView{};   // block metal: the lines on the angles
    const int lo = per ? -(1 << 29) : -d.off[0], hi = per ? (1 << 29) : d.N[0] - d.off[0];   // open z: the global nodes
    const long c0 = d.cell0(0);
    FV Er = f.er.view(), Ep = f.ep.view(), nEr = f.ner_.view(), nEp = f.nep_.view();
    // the lines on the angles: with the block metal (angle-dependent masks) or chi depending on phi; the tables of the metal
    // (the same angles as the chi of set_angles) or of the dielectric
    const bool ang = pv.np > 0 || dv.np > 0;
    const int NA = pv.np ? pv.np : dv.np, AM = pv.np ? pv.M : dv.M;
    const double *tcs = pv.np ? pv.cs : dv.cs, *tsn = pv.np ? pv.sn : dv.sn, *tch = pv.np ? pv.ch : dv.ch, *tsh = pv.np ? pv.sh : dv.sh;
    if (ang) {                         // the fields of the rows on the angles (nodes 0 .. n: a line reads a and a + 1)
        if (!aer_) for (auto* p : {&aer_, &aep_, &aner_, &anep_}) { *p = std::make_shared<Field>(); (*p)->resize(d); }
        const FV aEr = aer_->view(), aEp = aep_->view(), anEr = aner_->view(), anEp = anep_->view();
        const FV sEr = Er, sEp = Ep, snEr = nEr, snEp = nEp;
        par_for(long(n + 1) * rows * S, [=] HD(long w) {
            const int i = int(w % (n + 1)), j = J0 + int((w / (n + 1)) % rows), k = int(w / (long(n + 1) * rows));
            aEr(i, j, k) = rz_phi_at_angle(sEr, i, j, k, tcs, tsn, AM);
            anEr(i, j, k) = rz_phi_at_angle(snEr, i, j, k, tcs, tsn, AM);
            aEp(i, j, k) = rz_phi_at_angle(sEp, i, j, k, tch, tsh, AM);
            anEp(i, j, k) = rz_phi_at_angle(snEp, i, j, k, tch, tsh, AM);
        });
        Er = aEr; Ep = aEp; nEr = anEr; nEp = anEp;
    }
    double *gw = cpr_.data(), *xs = x_.data(), *ys = y_.data(), *ex = ex_.data();
    // A. the local solves of the interior and the edge values
    par_for(L, [=] HD(long q) {
        const Line ln = make_line(q, rows, S, J0, Er, Ep, nEr, nEp, dv, mv, pv, c0, lo, hi);
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
    if (ang) {                         // per angle (the metal rules rBp, rBr; the rings rules; none), then back to the modes
        const FV mEr = f.ner_.view(), mEp = f.nep_.view();
        par_for(long(n) * rows, [=] HD(long w) {
            const int i = int(w % n), jr = int(w / n), j = J0 + jr;
            const long I = c0 + i;
            double d1[RZ_PHI_MAXNP], d2[RZ_PHI_MAXNP];
            for (int k = 0; k < NA; ++k) {                // E_r, B_phi at (j+1/2, k)
                const double* y = ys + (long(k) * rows + jr) * (n + 1);
                const double ym = y[i], yp = y[i + 1];
                const int h = pv.np ? pv.rBp(I, j, k) : (mv.n ? mv.half_at(I, j) : 0);
                d1[k] = h == 0 ? -0.5 * (ym + yp) : 0.0;
                d2[k] = h == 0 ? -0.5 * (ym - yp) : (h == 2 ? -ym : (h == 3 ? yp : 0.0));
            }
            rz_phi_add_back(mEr, i, j, nullptr, d1, NA, tcs, tsn, AM, NA);
            rz_phi_add_back(nBp, i, j, nullptr, d2, NA, tcs, tsn, AM, NA);
            if (j == 0) return;
            for (int k = 0; k < NA; ++k) {                // E_phi, B_r at (j, k+1/2)
                const double* y = ys + (rs + long(k) * rows + jr) * (n + 1);
                const double ym = y[i], yp = y[i + 1];
                const int r = pv.np ? pv.rBr(I, j, k) : (mv.n ? mv.node_at(I, j) : 0);
                d1[k] = r == 0 ? -0.5 * (ym + yp) : 0.0;
                d2[k] = r == 0 ? 0.5 * (ym - yp) : (r == 2 ? ym : (r == 3 ? -yp : 0.0));
            }
            rz_phi_add_back(mEp, i, j, nullptr, d1, NA, tch, tsh, AM, NA);
            rz_phi_add_back(nBr, i, j, nullptr, d2, NA, tch, tsh, AM, NA);
        });
        return;
    }
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
    if (f.metal_phi || dv.np) {        // the block metal or chi per angle: per row all modes, column by column; D on the angles,
        if (f.metal_phi) f.metal_phi->ensure(d);   // E_z of each new column 0 at its metal angles (rEz) before the next reads it
        const MetalPhiView pv = f.metal_phi ? f.metal_phi->view() : MetalPhiView{};
        const int NA = 2 * M - 1;
        const double *tcs = pv.np ? pv.cs : dv.cs, *tsn = pv.np ? pv.sn : dv.sn, *tch = pv.np ? pv.ch : dv.ch, *tsh = pv.np ? pv.sh : dv.sh;
        const bool ang = dv.np > 0;
        par_for(long(nr), [=] HD(long q) {
            const int j = int(q);
            auto sl = [&](int mm, int c) { return mm == 0 ? 0 : 2 * mm - 1 + c; };
            // slots of a quantity from its values on the angles; the value on angle k of local slots
            auto back = [&](const double* a, const double* tc, const double* ts, double* s) {
                double s0 = 0;
                for (int k = 0; k < NA; ++k) s0 += a[k];
                s[0] = s0 / NA;
                for (int m = 1; m < M; ++m) {
                    double re = 0, im = 0;
                    for (int k = 0; k < NA; ++k) { re += a[k] * tc[size_t(k) * M + m]; im -= a[k] * ts[size_t(k) * M + m]; }
                    s[2 * m - 1] = re / NA; s[2 * m] = im / NA;
                }
            };
            auto angv = [&](const double* s, int k, const double* tc, const double* ts) {
                double v = s[0];
                for (int m = 1; m < M; ++m) v += 2 * (s[2 * m - 1] * tc[size_t(k) * M + m] - s[2 * m] * ts[size_t(k) * M + m]);
                return v;
            };
            double dzs[RZ_PHI_MAXNP], drj[RZ_PHI_MAXNP], drm[RZ_PHI_MAXNP], dps[RZ_PHI_MAXNP], av[RZ_PHI_MAXNP];
            for (int mi = i0; mi <= nz - 2; ++mi) {
                // D_z at (mi, j), D_r at (mi + 1/2, j and j - 1), D_phi at (mi + 1/2, j): slots
                if (ang) {
                    for (int k = 0; k < NA; ++k) av[k] = (1 + dv.z(mi, j, k)) * rz_phi_at_angle(Ez, mi, j, k, tcs, tsn, M);
                    back(av, tcs, tsn, dzs);
                    for (int k = 0; k < NA; ++k)
                        av[k] = 0.5 * (1 + dv.r(mi, j, k)) * (rz_phi_at_angle(Er, mi, j, k, tcs, tsn, M) + rz_phi_at_angle(Er, mi + 1, j, k, tcs, tsn, M));
                    back(av, tcs, tsn, drj);
                    if (j > 0) {
                        for (int k = 0; k < NA; ++k)
                            av[k] = 0.5 * (1 + dv.r(mi, j - 1, k)) * (rz_phi_at_angle(Er, mi, j - 1, k, tcs, tsn, M) + rz_phi_at_angle(Er, mi + 1, j - 1, k, tcs, tsn, M));
                        back(av, tcs, tsn, drm);
                    }
                    for (int k = 0; k < NA; ++k)
                        av[k] = 0.5 * (1 + dv.p(mi, j, k)) * (rz_phi_at_angle(Ep, mi, j, k, tch, tsh, M) + rz_phi_at_angle(Ep, mi + 1, j, k, tch, tsh, M));
                    back(av, tch, tsh, dps);
                } else {
                    for (int s = 0; s < NA; ++s) {
                        dzs[s] = (1 + dv.z(mi, j)) * Ez(mi, j, s);
                        drj[s] = 0.5 * (1 + dv.r(mi, j)) * (Er(mi, j, s) + Er(mi + 1, j, s));
                        drm[s] = j > 0 ? 0.5 * (1 + dv.r(mi, j - 1)) * (Er(mi, j - 1, s) + Er(mi + 1, j - 1, s)) : 0.0;
                        dps[s] = 0.5 * (1 + dv.p(mi, j)) * (Ep(mi, j, s) + Ep(mi + 1, j, s));
                    }
                }
                for (int m = 0; m < M; ++m)                   // D_z(mi + 1) = D_z(mi) + dz (G0 - div_perp D)
                    for (int c = 0; c < (m ? 2 : 1); ++c) {
                        const int k = sl(m, c);
                        double t;
                        if (j == 0) t = m == 0 ? g.rh[0] * drj[0] / g.A[0] : 0.0;
                        else t = (g.rh[j] * drj[k] - g.rh[j - 1] * drm[k]) / g.A[j] + (c == 0 ? -1.0 : 1.0) * g.mf(m) / g.r[j] * (m ? dps[sl(m, 1 - c)] : 0.0);
                        dzs[k] = m >= 1 && j == 0 ? 0.0 : dzs[k] + dz * (G0(mi, j, k) - t);
                    }
                if (ang) {                                    // E_z = D_z / (1 + chi_z) per angle
                    for (int k = 0; k < NA; ++k) av[k] = angv(dzs, k, tcs, tsn) / (1 + dv.z(mi + 1, j, k));
                    back(av, tcs, tsn, dzs);
                    for (int s = 0; s < NA; ++s) Ez(mi + 1, j, s) = j == 0 && s > 0 ? 0.0 : dzs[s];
                } else {
                    for (int s = 0; s < NA; ++s) Ez(mi + 1, j, s) = dzs[s] / (1 + dv.z(mi + 1, j));
                }
                if (mv.n && mv.node_at(c0 + mi + 1, j)) for (int s = 0; s < NA; ++s) Ez(mi + 1, j, s) = 0;
                if (pv.np) {
                    int ks[RZ_PHI_MAXNP];
                    int nk = 0;
                    for (int k = 0; k < pv.np; ++k) if (pv.rEz(c0 + mi + 1, j, k)) ks[nk++] = k;
                    if (nk == pv.np) { for (int s = 0; s < NA; ++s) Ez(mi + 1, j, s) = 0; }
                    else if (nk > 0) {
                        for (int q2 = 0; q2 < nk; ++q2) av[q2] = -rz_phi_at_angle(Ez, mi + 1, j, ks[q2], pv.cs, pv.sn, pv.M);
                        rz_phi_add_back(Ez, mi + 1, j, ks, av, nk, pv.cs, pv.sn, pv.M, pv.np);
                    }
                }
            }
        });
        return;
    }
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

// ---- chi per angle (DielView::np > 0): E_z, the polarization energy and div D through the angles ----
namespace {
// slots s of the values a on the NA = 2M - 1 angles of table (tc, ts); the value on angle k of the slots s
HD inline void ang_back(const double* a, const double* tc, const double* ts, int M, double* s) {
    const int NA = 2 * M - 1;
    double s0 = 0;
    for (int k = 0; k < NA; ++k) s0 += a[k];
    s[0] = s0 / NA;
    for (int m = 1; m < M; ++m) {
        double re = 0, im = 0;
        for (int k = 0; k < NA; ++k) { re += a[k] * tc[size_t(k) * M + m]; im -= a[k] * ts[size_t(k) * M + m]; }
        s[2 * m - 1] = re / NA; s[2 * m] = im / NA;
    }
}
HD inline double ang_val(const double* s, int k, const double* tc, const double* ts, int M) {
    double v = s[0];
    for (int m = 1; m < M; ++m) v += 2 * (s[2 * m - 1] * tc[size_t(k) * M + m] - s[2 * m] * ts[size_t(k) * M + m]);
    return v;
}
}  // namespace

void rz_diel_ez_angles(FieldsRZ& f, Field& out, const Field& base, const Field& vac) {
    const Domain& d = f.dom();
    const DielView dv = f.diel->view();
    const MetalView mv = f.metal ? f.metal->view() : MetalView{};
    const int nz = d.n[0], nr = d.n[1], M = f.modes(), NA = 2 * M - 1;
    const long c0 = d.cell0(0);
    const FV O = out.view(), B = base.view(), V = vac.view();
    par_for(long(nz) * nr, [=] HD(long q) {
        const int i = int(q % nz), j = int(q / nz);
        double b[RZ_PHI_MAXNP], inc[RZ_PHI_MAXNP], a[RZ_PHI_MAXNP];
        for (int s = 0; s < NA; ++s) { b[s] = B(i, j, s); inc[s] = V(i, j, s) - b[s]; }
        for (int k = 0; k < NA; ++k) a[k] = ang_val(inc, k, dv.cs, dv.sn, M) / (1 + dv.z(i, j, k));
        ang_back(a, dv.cs, dv.sn, M, inc);
        const bool zero = mv.n && mv.node_at(c0 + i, j) == 1;
        for (int s = 0; s < NA; ++s) O(i, j, s) = zero || (j == 0 && s > 0) ? 0.0 : b[s] + inc[s];
    });
}

double rz_diel_pol_energy(const FieldsRZ& f, int i0, int i1) {
    const Domain& d = f.dom();
    const DielView dv = f.diel->view();
    const RadialView g = f.grid().view();
    const int nz = d.n[0], nr = d.n[1], M = f.modes(), NA = 2 * M - 1;
    const FV Ez = f.ez.view(), Er = f.er.view(), Ep = f.ep.view();
    const double s = par_sum(long(nz) * nr * NA, [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), k = int(q / (long(nz) * nr));
        if (i < i0 || i > i1) return 0.0;
        const double ez = rz_phi_at_angle(Ez, i, j, k, dv.cs, dv.sn, M);
        const double mr = 0.5 * (rz_phi_at_angle(Er, i, j, k, dv.cs, dv.sn, M) + rz_phi_at_angle(Er, i + 1, j, k, dv.cs, dv.sn, M));
        const double mp = 0.5 * (rz_phi_at_angle(Ep, i, j, k, dv.ch, dv.sh, M) + rz_phi_at_angle(Ep, i + 1, j, k, dv.ch, dv.sh, M));
        return 2 * M_PI / NA * (g.A[j] * (dv.z(i, j, k) * ez * ez + dv.p(i, j, k) * mp * mp) + g.B[j] * dv.r(i, j, k) * mr * mr);
    });
    return 0.5 * s * d.h[0];
}

double rz_diel_pol_energy_mode(const FieldsRZ& f, int m) {
    const Domain& d = f.dom();
    const DielView dv = f.diel->view();
    const RadialView g = f.grid().view();
    const int nz = d.n[0], nr = d.n[1], M = f.modes(), NA = 2 * M - 1, s0 = m == 0 ? 0 : 2 * m - 1, ns = m == 0 ? 1 : 2;
    const FV Ez = f.ez.view(), Er = f.er.view(), Ep = f.ep.view();
    const double s = par_sum(long(nz) * nr, [=] HD(long q) {
        const int i = int(q % nz), j = int(q / nz);
        double a[RZ_PHI_MAXNP], p[RZ_PHI_MAXNP];
        double w = 0;
        // E_m . P_m with P = chi E on the angles (taken back to the modes)
        for (int k = 0; k < NA; ++k) a[k] = dv.z(i, j, k) * rz_phi_at_angle(Ez, i, j, k, dv.cs, dv.sn, M);
        ang_back(a, dv.cs, dv.sn, M, p);
        for (int c = 0; c < ns; ++c) w += g.A[j] * Ez(i, j, s0 + c) * p[s0 + c];
        for (int k = 0; k < NA; ++k)
            a[k] = 0.25 * dv.p(i, j, k) * (rz_phi_at_angle(Ep, i, j, k, dv.ch, dv.sh, M) + rz_phi_at_angle(Ep, i + 1, j, k, dv.ch, dv.sh, M));
        ang_back(a, dv.ch, dv.sh, M, p);
        for (int c = 0; c < ns; ++c) w += g.A[j] * (Ep(i, j, s0 + c) + Ep(i + 1, j, s0 + c)) * p[s0 + c];
        for (int k = 0; k < NA; ++k)
            a[k] = 0.25 * dv.r(i, j, k) * (rz_phi_at_angle(Er, i, j, k, dv.cs, dv.sn, M) + rz_phi_at_angle(Er, i + 1, j, k, dv.cs, dv.sn, M));
        ang_back(a, dv.cs, dv.sn, M, p);
        for (int c = 0; c < ns; ++c) w += g.B[j] * (Er(i, j, s0 + c) + Er(i + 1, j, s0 + c)) * p[s0 + c];
        return (m ? 2.0 : 1.0) * 2 * M_PI * w;
    });
    return 0.5 * s * d.h[0];
}

void rz_diel_div_d(const FieldsRZ& f, Field& out) {
    const Domain& d = f.dom();
    const DielView dv = f.diel->view();
    const RadialView g = f.grid().view();
    const double dz = d.h[0];
    const int nz = d.n[0], nr = d.n[1], M = f.modes(), NA = 2 * M - 1;
    const FV Ez = f.ez.view(), Er = f.er.view(), Ep = f.ep.view(), O = out.view();
    par_for(long(nz) * nr, [=] HD(long q) {
        const int i = int(q % nz), j = int(q / nz);
        auto sl = [&](int mm, int c) { return mm == 0 ? 0 : 2 * mm - 1 + c; };
        double a[RZ_PHI_MAXNP] = {}, z0[RZ_PHI_MAXNP], z1[RZ_PHI_MAXNP], rj[RZ_PHI_MAXNP], rm[RZ_PHI_MAXNP] = {}, pj[RZ_PHI_MAXNP];
        for (int k = 0; k < NA; ++k) a[k] = (1 + dv.z(i, j, k)) * rz_phi_at_angle(Ez, i, j, k, dv.cs, dv.sn, M);
        ang_back(a, dv.cs, dv.sn, M, z0);
        for (int k = 0; k < NA; ++k) a[k] = (1 + dv.z(i + 1, j, k)) * rz_phi_at_angle(Ez, i + 1, j, k, dv.cs, dv.sn, M);
        ang_back(a, dv.cs, dv.sn, M, z1);
        for (int k = 0; k < NA; ++k)
            a[k] = 0.5 * (1 + dv.r(i, j, k)) * (rz_phi_at_angle(Er, i, j, k, dv.cs, dv.sn, M) + rz_phi_at_angle(Er, i + 1, j, k, dv.cs, dv.sn, M));
        ang_back(a, dv.cs, dv.sn, M, rj);
        if (j > 0) {
            for (int k = 0; k < NA; ++k)
                a[k] = 0.5 * (1 + dv.r(i, j - 1, k)) * (rz_phi_at_angle(Er, i, j - 1, k, dv.cs, dv.sn, M) + rz_phi_at_angle(Er, i + 1, j - 1, k, dv.cs, dv.sn, M));
            ang_back(a, dv.cs, dv.sn, M, rm);
        }
        for (int k = 0; k < NA; ++k)
            a[k] = 0.5 * (1 + dv.p(i, j, k)) * (rz_phi_at_angle(Ep, i, j, k, dv.ch, dv.sh, M) + rz_phi_at_angle(Ep, i + 1, j, k, dv.ch, dv.sh, M));
        ang_back(a, dv.ch, dv.sh, M, pj);
        for (int m = 0; m < M; ++m)
            for (int c = 0; c < (m ? 2 : 1); ++c) {
                const int k = sl(m, c);
                double t;
                if (j == 0) t = m == 0 ? g.rh[0] * rj[0] / g.A[0] : 0.0;
                else t = (g.rh[j] * rj[k] - g.rh[j - 1] * rm[k]) / g.A[j] + (c == 0 ? -1.0 : 1.0) * g.mf(m) / g.r[j] * (m ? pj[sl(m, 1 - c)] : 0.0);
                O(i, j, k) = m >= 1 && j == 0 ? 0.0 : (z1[k] - z0[k]) / dz + t;
            }
    });
}
