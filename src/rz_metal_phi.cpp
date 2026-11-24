// Embedded perfect conductors that depend on the azimuth in RZ (block `metal`, docs/specs/rz_metal_phi.md): the mask on
// the collocation angles, the rules after each pass of the RIP-CC step and the removal of particles in the metal.
#include "rz_metal_phi.hpp"
#include <climits>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include "grid.hpp"
#include "rz.hpp"
#include "species.hpp"

namespace {
constexpr int MAXNP = 255;             // angles per kernel thread (modes <= 128)
}

MetalPhi::MetalPhi(const MetalPhiConfig& c, const Domain& d, int modes, const std::vector<double>& rnodes)
    : cfg(c), np_(2 * modes - 1), M_(modes), r_(rnodes) {
    if (np_ > MAXNP) throw std::runtime_error("metal: at most 128 modes");
    if (cfg.fn.empty()) throw std::runtime_error("metal: function = <expression in x, y, z, r, phi> is required");
    if (int(r_.size()) < d.N[1] + 1) throw std::runtime_error("metal: radial nodes missing");
    r_.resize(size_t(d.N[1]) + 1);    // nodes 0 .. nr (the wall)
    nrn_ = int(r_.size());
    dz_ = d.h[0];
    n_ = d.n[0];
    N_ = d.periodic[0] ? d.N[0] : 0;
    L_ = N_ > 0 ? N_ : n_ + 6;
    host_.assign(size_t(L_) * nrn_ * np_, 0);
    have_.assign(size_t(L_), LONG_MIN);
    bits_.resize(host_.size());
    std::vector<double> cs(size_t(np_) * M_), sn(size_t(np_) * M_), ch(cs.size()), sh(cs.size());
    const double off = cfg.spectral ? 0.0 : 0.5;           // the staggered angles of E_phi, B_r, B_z
    for (int k = 0; k < np_; ++k)
        for (int m = 0; m < M_; ++m) {
            const double a = 2 * M_PI * double(k) * m / np_;
            cs[size_t(k) * M_ + m] = std::cos(a);
            sn[size_t(k) * M_ + m] = std::sin(a);
            ch[size_t(k) * M_ + m] = std::cos(2 * M_PI * (k + off) * m / np_);
            sh[size_t(k) * M_ + m] = std::sin(2 * M_PI * (k + off) * m / np_);
        }
    cs_.upload(cs); sn_.upload(sn); ch_.upload(ch); sh_.upload(sh);
    ensure(d);
}

MetalPhiView MetalPhi::view() const {
    MetalPhiView v;
    v.bits = bits_.data(); v.cs = cs_.data(); v.sn = sn_.data(); v.ch = ch_.data(); v.sh = sh_.data();
    v.stag = !cfg.spectral;
    v.L = L_; v.nrn = nrn_; v.np = np_; v.M = M_; v.N = N_;
    return v;
}

void MetalPhi::fill(long I, long s) {
    const double z = double(I) * dz_;
    unsigned char* b = host_.data() + size_t(s) * nrn_ * np_;
    for (int j = 0; j < nrn_; ++j)
        for (int k = 0; k < np_; ++k) {
            const double phi = 2 * M_PI * k / np_, r = r_[size_t(j)];
            const double v[5] = {r * std::cos(phi), r * std::sin(phi), z, r, phi};
            b[size_t(j) * np_ + k] = cfg.fn(v) > 0 ? 1 : 0;
        }
    have_[size_t(s)] = I;
    bits_.from_host(b, size_t(s) * nrn_ * np_, size_t(nrn_) * np_);
}

void MetalPhi::ensure(const Domain& d) {
    if (d.n[0] != n_) throw std::runtime_error("metal: the domain changed its local size");
    const long c0 = d.cell0(0);
    for (long I = c0 - 2; I <= c0 + n_ + 1; ++I) {
        const long Iw = N_ > 0 ? ((I % N_) + N_) % N_ : I;
        const long s = ((Iw % L_) + L_) % L_;
        if (have_[size_t(s)] != Iw) fill(Iw, s);
    }
}

long MetalPhi::metal_nodes_local() const {
    long c = 0;
    for (long s = 0; s < L_; ++s)
        if (have_[size_t(s)] != LONG_MIN)
            for (size_t q = 0; q < size_t(nrn_) * np_; ++q) c += host_[size_t(s) * nrn_ * np_ + q];
    return c;
}

MetalPhiView MetalPhi::host_view() const {
    MetalPhiView v = view();
    v.bits = host_.data(); v.cs = v.sn = v.ch = v.sh = nullptr;
    return v;
}

bool MetalPhi::has_phi_surfaces() const {
    for (long s = 0; s < L_; ++s)
        if (have_[size_t(s)] != LONG_MIN)
            for (int j = 0; j < nrn_; ++j) {
                const unsigned char* b = host_.data() + (size_t(s) * nrn_ + j) * np_;
                for (int k = 1; k < np_; ++k) if (b[k] != b[0]) return true;
            }
    return false;
}

std::string MetalPhi::describe() const {
    char s[400];
    std::snprintf(s, sizeof s, " on %d angles (modes 0..%d, dphi = %.4g rad, phi derivative %s); metal nodes in the first columns: %ld of %ld\n",
                  np_, M_ - 1, 2 * M_PI / np_, cfg.spectral ? "spectral" : "difference (staggered angles)", metal_nodes_local(),
                  long(L_) * nrn_ * np_);
    return "  metal: function \"" + cfg.text + "\"" + s;
}

namespace {
// field value at angle k (table c, s: integer or staggered angles) from the slots of (i, j): F_0 + 2 sum (Re F_m cos - Im F_m sin)
HD inline double at_angle(const FV& F, int i, int j, int k, const double* tc, const double* ts, int M) {
    double s = F(i, j, 0);
    const double* c = tc + size_t(k) * M;
    const double* sn = ts + size_t(k) * M;
    for (int m = 1; m < M; ++m) s += 2 * (F(i, j, 2 * m - 1) * c[m] - F(i, j, 2 * m) * sn[m]);
    return s;
}
// adds the changes d at the angles ks[0 .. n) back to the modes: dF_0 = sum d / Np, dRe F_m = sum d cos / Np,
// dIm F_m = -sum d sin / Np (exact for degree M - 1 on 2M - 1 angles, also on the staggered ones)
HD inline void add_back(const FV& F, int i, int j, const int* ks, const double* d, int n, const double* tc, const double* ts, int M,
                        int np) {
    const double w = 1.0 / np;
    double s0 = 0;
    for (int q = 0; q < n; ++q) s0 += d[q];
    F(i, j, 0) += w * s0;
    for (int m = 1; m < M; ++m) {
        double a = 0, b = 0;
        for (int q = 0; q < n; ++q) {
            a += d[q] * tc[size_t(ks[q]) * M + m];
            b -= d[q] * ts[size_t(ks[q]) * M + m];
        }
        F(i, j, 2 * m - 1) += w * a;
        F(i, j, 2 * m) += w * b;
    }
}
}  // namespace

void rz_metal_phi_apply(const MetalPhi& mp, FieldsRZ& f, int stage, Field& fa, Field& fb, Field* fc, Field* fe) {
    const Domain& d = f.dom();
    const int nz = d.n[0], nr = d.n[1];
    const long c0 = d.cell0(0);
    const MetalPhiView v = mp.view();
    if (stage == 0 || stage == 2) {
        const FV Ez = fa.view(), Bz = fb.view();
        par_for(long(nz) * nr, [=] HD(long q) {
            const int i = int(q % nz), j = int(q / nz);
            const long I = c0 + i;
            int ks[MAXNP];
            double dv[MAXNP];
            int n = 0;
            for (int k = 0; k < v.np; ++k) if (v.rEz(I, j, k)) ks[n++] = k;      // E_z at (j, k)
            if (n == v.np) { for (int s = 0; s < 2 * v.M - 1; ++s) Ez(i, j, s) = 0; }
            else if (n > 0) {
                for (int q2 = 0; q2 < n; ++q2) dv[q2] = -at_angle(Ez, i, j, ks[q2], v.cs, v.sn, v.M);
                add_back(Ez, i, j, ks, dv, n, v.cs, v.sn, v.M, v.np);
            }
            n = 0;
            for (int k = 0; k < v.np; ++k) if (v.Mc(I, j, k)) ks[n++] = k;       // B_z at (j+1/2, k+1/2): 0 in and on the metal
            if (n == v.np) { for (int s = 0; s < 2 * v.M - 1; ++s) Bz(i, j, s) = 0; }
            else if (n > 0) {
                for (int q2 = 0; q2 < n; ++q2) dv[q2] = -at_angle(Bz, i, j, ks[q2], v.ch, v.sh, v.M);
                add_back(Bz, i, j, ks, dv, n, v.ch, v.sh, v.M, v.np);
            }
        });
        return;
    }
    // stage 1: the transported fields. (E_r, B_phi) at (j+1/2, k): E_r = 0, B_phi by rBp (2: T+ = E_r + B_phi, 3: -T- =
    // B_phi - E_r). (E_phi, B_r) at (j, k+1/2): E_phi = 0, B_r by rBr (2: -T-_z = B_r - E_phi, 3: T+_z = E_phi + B_r).
    const FV Er = fa.view(), Ep = fb.view(), Br = fc->view(), Bp = fe->view();
    par_for(long(nz) * nr, [=] HD(long q) {
        const int i = int(q % nz), j = int(q / nz);
        const long I = c0 + i;
        int ks[MAXNP];
        double d1[MAXNP], d2[MAXNP];
        int n = 0;
        for (int k = 0; k < v.np; ++k) if (v.Mh(I, j, k)) ks[n++] = k;
        if (n > 0) {
            for (int q2 = 0; q2 < n; ++q2) {
                const int k = ks[q2], h = v.rBp(I, j, k);
                const double er = at_angle(Er, i, j, k, v.cs, v.sn, v.M), bp = at_angle(Bp, i, j, k, v.cs, v.sn, v.M);
                d1[q2] = -er;
                d2[q2] = h == 1 ? -bp : (h == 2 ? er : -er);
            }
            add_back(Er, i, j, ks, d1, n, v.cs, v.sn, v.M, v.np);
            add_back(Bp, i, j, ks, d2, n, v.cs, v.sn, v.M, v.np);
        }
        n = 0;
        for (int k = 0; k < v.np; ++k) if (v.Mp(I, j, k)) ks[n++] = k;
        if (n > 0) {
            for (int q2 = 0; q2 < n; ++q2) {
                const int k = ks[q2], r = v.rBr(I, j, k);
                const double ep = at_angle(Ep, i, j, k, v.ch, v.sh, v.M), br = at_angle(Br, i, j, k, v.ch, v.sh, v.M);
                d1[q2] = -ep;
                d2[q2] = r == 1 ? -br : (r == 2 ? -ep : ep);
            }
            add_back(Ep, i, j, ks, d1, n, v.ch, v.sh, v.M, v.np);
            add_back(Br, i, j, ks, d2, n, v.ch, v.sh, v.M, v.np);
        }
    });
}

void rz_metal_phi_remove(Species& sp, const FieldsRZ& f, const MetalPhi& mp, double rwall, int order) {
    if (sp.size() == 0) return;
    const MetalPhiView mv = mp.view();
    const RadialView g = f.grid().view();
    const double dz = f.dom().h[0], mg = 0.5 * (order - 1), out = 2 * rwall;
    const bool all = f.map_all || f.c1_map || f.modes() == 1;
    double* X = sp.ptr<double>(Species::X);
    double* Y = sp.ptr<double>(Species::Y);
    double* Z = sp.ptr<double>(Species::Z);
    par_for(long(sp.size()), [=] HD(long p) {
        const double r = sqrt(Y[p] * Y[p] + Z[p] * Z[p]);
        double ph = atan2(Z[p], Y[p]);
        if (ph < 0) ph += 2 * M_PI;
        const int k = int(floor(ph * mv.np / (2 * M_PI) + 0.5)) % mv.np;     // the nearest angle
        // the nodes of the shape in the radial map of the deposit; the modes m >= 1 deposit in the plain xi <= the mapped
        // one (decision 106): the lowest node is taken from it
        const double xi = rz_xi(g, r, order), xp = all ? xi : rz_xi(g, r, 0);
        const int ja = order == 1 ? int(floor(xp)) : int(floor(xp + 0.5)) - 1;
        const int jb = order == 1 ? int(floor(xi)) + 1 : int(floor(xi + 0.5)) + 1;
        const double zc = X[p] / dz;
        const long lo = long(floor(zc - mg)), hi = long(ceil(zc + mg));
        for (long I = lo; I <= hi; ++I)
            for (int j = ja < 0 ? 0 : ja; j <= jb; ++j)
                if (!mv.Mn(I, j, k)) return;
        Y[p] = out; Z[p] = 0;                      // migrate() absorbs it like the wall
    });
}
