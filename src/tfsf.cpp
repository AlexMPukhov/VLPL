#include "tfsf.hpp"
#include <cmath>
#include <algorithm>
#include "irip.hpp"
#include <cstdio>
#include <stdexcept>

// ---------------------------------------------------------------- kernels (free functions)
// The beam of L, propagating along (cos a, sin a, 0) through the point (xs, yc, zc), its peak there at t = tc:
// the field of the entrance-plane formula of LaserConfig in the rotated frame, retarded by the distance along the
// beam and with the focus distance shifted by it (paraxial). At t = 0 it lies left of the plane (xs - tc along the
// beam). Waist 0: a plane wave of one k_y = sin a (periodic y), at the plane a(t - sin a (y - yc)) with the envelope of
// LaserConfig. pol 'z': E = a z, B = a (sin a, -cos a, 0); pol 'y': E = a (-sin a, cos a, 0), B = a z. The longitudinal
// fields of this formula (the tilt only) are replaced outside the CPML layers by those of the discrete constraints
// (tfsf_longitudinal_kernel).
void tfsf_init_kernel(const Domain& d, const LaserConfig L, double xs, const FV Ex, const FV Ey, const FV Ez,
                      const FV Bx, const FV By, const FV Bz) {
    const double hx = d.h[0], hy = d.h[1], hz = d.h[2], ca = std::cos(L.angle), sa = std::sin(L.angle);
    const long c0 = d.cell0(0), o1 = d.off[1], o2 = d.off[2];
    for_owned(d, [=] HD(int i, int j, int k) {
        const double x = (c0 + i) * hx, y0 = (o1 + j) * hy, z0 = (o2 + k) * hz;
        auto a = [&](double y, double z) {
            if (L.w0 <= 0) {          // waist = 0: the pulse of one k_y = sin a, its envelope moving along x at cos a
                const double te = -(x - xs) / ca - L.tc;
                return L.a0 * exp(-1.3862943611198906 * te * te / (L.tau * L.tau)) * L.taper(te) *
                       sin(-(ca * (x - xs) + sa * (y - L.yc)) - L.tc + L.phase);
            }
            const double xp = (x - xs) * ca + (y - L.yc) * sa, yp = -(x - xs) * sa + (y - L.yc) * ca + L.yc;
            LaserConfig M = L;
            M.xf = L.xf - xp;
            return M.field(-xp, yp, z);
        };
        if (L.pol == 'z') {
            const double ez = a(y0, z0 + 0.5 * hz), bx = a(y0 + 0.5 * hy, z0 + 0.5 * hz);
            Ez(i, j, k) += ez; By(i, j, k) += -ca * ez; Bx(i, j, k) += sa * bx;
        } else {
            const double ey = a(y0 + 0.5 * hy, z0), ex = a(y0, z0);
            Ey(i, j, k) += ca * ey; Bz(i, j, k) += ey; Ex(i, j, k) += -sa * ex;
        }
    });
}

// dst = src over the whole array, guards included
void tfsf_copy(const Domain& d, const FV src, const FV dst) {
    const int g = d.ng, gz = d.gw(2), sx = d.n[0] + 2 * g, sy = d.n[1] + 2 * g, sz = d.ext(2);
    par_for(long(sx) * sy * sz, [=] HD(long q) {
        const int i = int(q % sx) - g, j = int((q / sx) % sy) - g, k = int(q / (long(sx) * sy)) - gz;
        dst(i, j, k) = src(i, j, k);
    });
}

// Ex*, Bx* of the incident field (no currents): kind 0 the RIP-CC predictor of (Ex, .. Bz) at t_n, kind 1 half of
// t_n, kind 2 the mean of t_n (Ex, Bx) and t_n+1 (Ex1, Bx1)
void tfsf_averages(const Domain& d, int kind, const FV Ex, const FV Ey, const FV Ez, const FV Bx, const FV By, const FV Bz,
                   const FV Ex1, const FV Bx1, const FV XS, const FV BS) {
    const double D = d.h[0], iy = 1.0 / d.h[1], iz = 1.0 / d.h[2];
    for_owned(d, [=] HD(int i, int j, int k) {
        if (kind == 0) {
            XS(i, j, k) = Ex(i, j, k) + 0.5 * D * ((Bz(i, j, k) - Bz(i, j - 1, k)) * iy - (By(i, j, k) - By(i, j, k - 1)) * iz);
            BS(i, j, k) = Bx(i, j, k) - 0.5 * D * ((Ez(i, j + 1, k) - Ez(i, j, k)) * iy - (Ey(i, j, k + 1) - Ey(i, j, k)) * iz);
        } else if (kind == 1) {
            XS(i, j, k) = 0.5 * Ex(i, j, k); BS(i, j, k) = 0.5 * Bx(i, j, k);
        } else {
            XS(i, j, k) = 0.5 * (Ex(i, j, k) + Ex1(i, j, k)); BS(i, j, k) = 0.5 * (Bx(i, j, k) + Bx1(i, j, k));
        }
    });
}

// The transport across the plane between the local cells s - 1 (scattered) and s (total). The cell s takes T+ of
// s - 1 and the sources at s - 1/2, whose x averages mix both sides: + the incident T+ of s - 1 and half the incident
// Ex*, Bx* of s - 1. The cell s - 1 takes T- of s: - the incident T- of s and half the incident Ex*, Bx* of s.
// (n = (T+ + T-) / 2 and (T+ - T-) / 2: a change of T+_y by c changes nEy, nBz by c / 2; of T-_y: nEy +c/2, nBz -c/2;
// the z pair: T+_z = Ez + By (along -x), T-_z = Ez - By (along +x), nEz = (T+_z + T-_z) / 2, nBy = (T+_z - T-_z) / 2.)
// In a CPML layer every transverse derivative g of the sources becomes g + psi, psi <- b psi + a g (pml.hpp); the
// map is linear, so the half incident source goes through its own psi (8 per transverse point: 4 sources, 2 sides),
// with the coefficients of the layers; commit = false (first iRIP pass) reads psi without updating it.
struct TfCp {
    const double *a, *b;
    int off;
    HD double operator()(double g, double* psi, int jl, bool commit) const {
        if (!a) return g;
        const double aa = a[off + jl];
        if (aa == 0.0) return g;
        const double p = b[off + jl] * *psi + aa * g;
        if (commit) *psi = p;
        return g + p;
    }
};
void tfsf_correct_kernel(const Domain& d, int s, int io, const FV Ey, const FV Ez, const FV By, const FV Bz, const FV XS,
                         const FV BS, const FV nEy, const FV nEz, const FV nBy, const FV nBz, const TfCp yn, const TfCp yh,
                         const TfCp zn, const TfCp zh, double* psi, bool commit) {
    const double D = d.h[0], iy = 1.0 / d.h[1], iz = 1.0 / d.h[2];
    const int ny = d.n[1], nx = d.n[0];
    const long np = long(ny) * d.n[2];
    const bool tf = s >= 0 && s < nx, sf = s - 1 >= 0 && s - 1 < nx;
    par_for(np, [=] HD(long q) {
        const int j = int(q % ny), k = int(q / ny);
        // sources of the cell a: psi_y = dz Bx*, psi_z = -dy Bx*, phi_y = -dz Ex*, phi_z = dy Ex* (half, one side;
        // side 0: the total cell s takes a = s - 1, side 1: the scattered cell s - 1 takes a = s)
        // the incident arrays are indexed from io along x (the slab of the plane waves; the incident grid: io = 0)
        const int ia = s - 1 - io, ib = s - io;
        auto psy = [&](int a, int side) { a -= io; return zn(0.5 * (BS(a, j, k) - BS(a, j, k - 1)) * iz, psi + (4 * side + 0) * np + q, k, commit); };
        auto psz = [&](int a, int side) { a -= io; return -yn(0.5 * (BS(a, j, k) - BS(a, j - 1, k)) * iy, psi + (4 * side + 1) * np + q, j, commit); };
        auto phy = [&](int a, int side) { a -= io; return -zh(0.5 * (XS(a, j, k + 1) - XS(a, j, k)) * iz, psi + (4 * side + 2) * np + q, k, commit); };
        auto phz = [&](int a, int side) { a -= io; return yh(0.5 * (XS(a, j + 1, k) - XS(a, j, k)) * iy, psi + (4 * side + 3) * np + q, j, commit); };
        if (tf) {
            const int a = s - 1;
            const double sy = psy(a, 0), sz = psz(a, 0), fy = phy(a, 0), fz = phz(a, 0);
            const double dp = Ey(ia, j, k) + Bz(ia, j, k) + D * (sy + fz);      // T+_y into s
            const double dm = Ez(ia, j, k) - By(ia, j, k) + D * (sz - fy);      // T-_z into s
            nEy(s, j, k) += 0.5 * dp; nBz(s, j, k) += 0.5 * dp;
            nEz(s, j, k) += 0.5 * dm; nBy(s, j, k) -= 0.5 * dm;
        }
        if (sf) {
            const int b = s, c = s - 1;
            const double sy = psy(b, 1), sz = psz(b, 1), fy = phy(b, 1), fz = phz(b, 1);
            const double dm = -(Ey(ib, j, k) - Bz(ib, j, k) + D * (sy - fz));   // T-_y into s - 1
            const double dp = -(Ez(ib, j, k) + By(ib, j, k) + D * (sz + fy));   // T+_z into s - 1
            nEy(c, j, k) += 0.5 * dm; nBz(c, j, k) -= 0.5 * dm;
            nEz(c, j, k) += 0.5 * dp; nBy(c, j, k) += 0.5 * dp;
        }
    });
}

// O5: X = Cx P (the x stencil of the closure, P already C_perp-closed, guards valid)
void tfsf_cx(const Domain& d, const FV P, const FV X) {
    for_owned(d, [=] HD(int i, int j, int k) { X(i, j, k) = P(i, j, k) - (P(i + 1, j, k) - 2 * P(i, j, k) + P(i - 1, j, k)) / 3; });
}

// O5: the Cx stencil across the plane (local cells s - 1 scattered, s total): the far cell's incident value
// (io: the x index of PE, PB starts there: 0 for the incident grid, s - 1 for the slab of the plane waves)
void tfsf_o5_fix_kernel(const Domain& d, int s, int io, const FV PE, const FV PB, const FV Exs, const FV Bxs) {
    const int ny = d.n[1], nx = d.n[0];
    const bool tf = s >= 0 && s < nx, sf = s - 1 >= 0 && s - 1 < nx;
    par_for(long(ny) * d.n[2], [=] HD(long q) {
        const int j = int(q % ny), k = int(q / ny);
        if (sf) { Exs(s - 1, j, k) += PE(s - io, j, k) / 3; Bxs(s - 1, j, k) += PB(s - io, j, k) / 3; }
        if (tf) { Exs(s, j, k) -= PE(s - 1 - io, j, k) / 3; Bxs(s, j, k) -= PB(s - 1 - io, j, k) / 3; }
    });
}

// max over the owned cells i >= from of |a| + |b| + |c| (local)
double tfsf_max(const Domain& d, const FV a, const FV b, const FV c, int from) {
    const int nx = d.n[0], ny = d.n[1];
    return par_max_abs(d.owned(), [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        return i >= from ? fabs(a(i, j, k)) + fabs(b(i, j, k)) + fabs(c(i, j, k)) : 0.0;
    });
}

// max over the owned cells i < to of |a| + |b| + |c| (local)
double tfsf_max_below(const Domain& d, const FV a, const FV b, const FV c, int to) {
    const int nx = d.n[0], ny = d.n[1];
    return par_max_abs(d.owned(), [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        return i < to ? fabs(a(i, j, k)) + fabs(b(i, j, k)) + fabs(c(i, j, k)) : 0.0;
    });
}

// F = 0 in the owned cells i >= s
void tfsf_zero_from(const Domain& d, const FV F, int s) {
    for_owned(d, [=] HD(int i, int j, int k) { if (i >= s) F(i, j, k) = 0; });
}

// The longitudinal fields of the incident grid from its discrete constraints (Fields::gauss_residual, max_div_b, no
// charge): (Ex(i + 1) - Ex(i)) / dx + mu_x div_perp E = 0, the same for B, summed from the plane (where the cut field is 0)
// towards -x. The beam formula has only the longitudinal field of the tilt, not that of the focusing (~ a / (k w0)); the
// div E, div B it left were conserved by the scheme, a static Ex, Bx where the pulse started. The local cells i < s,
// starting from the guard i = n (the right neighbour's first cell, valid after the previous pass) or from 0 at the plane
// and at the right edge: one pass per rank along x. Only the columns outside the CPML layers (lo, hi: local y, z ranges,
// those of the Gauss law check), where these constraints hold; a pulse reaching into a layer or up to the zero guard of
// an open y, z edge would otherwise turn that cut into a large Ex, Bx (the layers keep the formula's fields).
void tfsf_longitudinal_kernel(const Domain& d, int s, const int lo[3], const int hi[3], const FV Ex, const FV Ey,
                              const FV Ez, const FV Bx, const FV By, const FV Bz) {
    const double D = d.h[0], iy = 1.0 / d.h[1], iz = 1.0 / d.h[2];
    const int nx = d.n[0], ny = d.n[1], iend = s < nx ? s : nx;
    const bool from_guard = iend == nx && d.off[0] + nx < d.N[0];
    const int jlo = lo[1], jhi = hi[1], klo = lo[2], khi = hi[2];
    if (iend <= 0) return;
    par_for(long(ny) * d.n[2], [=] HD(long q) {
        const int j = int(q % ny), k = int(q / ny);
        if (j < jlo || j >= jhi || k < klo || k >= khi) return;
        auto de = [&](int i) { return (Ey(i, j, k) - Ey(i, j - 1, k)) * iy + (Ez(i, j, k) - Ez(i, j, k - 1)) * iz; };
        auto db = [&](int i) { return (By(i, j + 1, k) - By(i, j, k)) * iy + (Bz(i, j, k + 1) - Bz(i, j, k)) * iz; };
        double ex = from_guard ? Ex(nx, j, k) : 0.0, bx = from_guard ? Bx(nx, j, k) : 0.0;
        double ge = de(iend), gb = db(iend);
        for (int i = iend - 1; i >= 0; --i) {
            const double ge0 = de(i), gb0 = db(i);
            ex += 0.5 * D * (ge0 + ge); bx += 0.5 * D * (gb0 + gb);
            Ex(i, j, k) = ex; Bx(i, j, k) = bx;
            ge = ge0; gb = gb0;
        }
    });
}

// ---------------------------------------------------------------- Tfsf
Tfsf::Tfsf(const Domain& d, Comm& comm, const std::vector<LaserConfig>& lasers, const PmlConfig* pml, double bmin, double t_span)
    : d_(d) {
    double xs = -1;
    int nm = 0, nl = 0;
    for (const LaserConfig& L : lasers) {
        if (!L.enabled || L.tfsf_x < 0) continue;
        if (xs >= 0 && std::abs(L.tfsf_x - xs) > 1e-12 * (1 + xs)) throw std::runtime_error("laser.tfsf_x: one plane for all lasers");
        xs = L.tfsf_x;
        ++nl; nm += L.tfsf_modes;
    }
    if (xs < 0) throw std::runtime_error("Tfsf: no laser with tfsf_x");
    if (nm != 0 && nm != nl) throw std::runtime_error("laser.tfsf_incident: the same for all lasers of the TFSF plane");
    is_ = int(std::lround(xs / d.h[0]));
    if (is_ < 1 || is_ > d.N[0] - 1) throw std::runtime_error("laser.tfsf_x: the plane must lie inside the box (1 .. nx - 1 cells)");
    if (nm > 0) {                                      // exact plane waves (tfsf_modes.hpp): no incident grid
        if (pml) throw std::runtime_error("laser.tfsf_incident = modes: periodic y and z (no pml) so far");
        modes_ = std::make_unique<TfsfModes>(d, lasers, is_, t_span);
        amax_ = modes_->peak();
        // the part of the pulse already right of the plane at t = 0 is added to the main grid by start(): the run begins as the
        // plain scheme for (total right, scattered + incident left); over_ reports how much was at the plane
        const double p0 = comm.allreduce_max(modes_->plane_max(0.0));
        over_ = amax_ > 0 ? p0 / amax_ : 0;
        if (comm.rank == 0)
            std::printf("  tfsf: %zu exact plane waves (polarisation within %.1e of the wanted one), at the plane at t = 0: %.1e "
                        "of the peak\n", modes_->count(), modes_->mismatch(), over_);
        return;
    }
    inc_ = std::make_unique<Fields>(d);
    for (Field* f : {&ex0_, &ey0_, &ez0_, &bx0_, &by0_, &bz0_, &xs_, &bs_}) f->resize(d);
    if (pml) {                               // the incident grid with the same layers: the same discrete scheme
        pml_ = std::make_unique<Pml>(d, *pml, d.dt(), bmin);
        inc_->pml = pml_.get();
        psi_.resize(size_t(8) * d.n[1] * d.n[2]);
    }
    Fields& g = *inc_;
    const double xp = is_ * d.h[0];                    // the plane on the grid
    for (const LaserConfig& L : lasers)
        if (L.enabled && L.tfsf_x >= 0)
            tfsf_init_kernel(d, L, xp, g.ex.view(), g.ey.view(), g.ez.view(), g.bx.view(), g.by.view(), g.bz.view());
    comm.exchange(d, {&g.ex, &g.ey, &g.ez, &g.bx, &g.by, &g.bz});
    const FV a = g.ey.view(), b = g.ez.view(), c = g.ex.view();
    amax_ = comm.allreduce_max(tfsf_max(d, a, b, c, -d.ng));
    // The scheme with the corrections is the plain scheme for (total in i >= i_s, scattered + incident in i < i_s), which
    // starts from the incident grid only if that is zero from the plane on at t = 0 (a part already across the plane would
    // be a free error wave, carried back across the plane by diffraction or obliqueness: leakage)
    const int s = is_ - d.off[0];
    const double over = comm.allreduce_max(tfsf_max(d, a, b, c, s));
    over_ = amax_ > 0 ? over / amax_ : 0;
    if (comm.rank == 0 && over_ > 1e-4)
        std::fprintf(stderr, "warning: laser.tfsf_x: the incident pulse reaches across the TFSF plane at t = 0 (%.1e of its peak) and is cut there: "
                             "increase laser.delay (it is the time at which the peak crosses the plane)\n", over_);
    // ... and is cut there, so that the source is exact for any beam (the cut is the overlap, < 1e-4 of the peak
    // unless warned)
    {
        for (Field* f : {&g.ex, &g.ey, &g.ez, &g.bx, &g.by, &g.bz}) tfsf_zero_from(d, f->view(), s);
        comm.exchange(d, {&g.ex, &g.ey, &g.ez, &g.bx, &g.by, &g.bz});
    }
    // Ex, Bx from the constraints (the incident grid starts with div E = div B = 0 on the grid outside the CPML layers),
    // rank by rank from the plane
    int lo[3] = {0, 0, 0}, hi[3] = {d.n[0], d.n[1], d.n[2]};
    for (int a = 1; a < 3; ++a)
        if (pml_) { lo[a] = std::max(0, pml_->check_lo(a) - d.off[a]); hi[a] = std::min(d.n[a], pml_->check_hi(a, d.N[a]) - d.off[a]); }
    for (int r = 0; r < comm.dims[0]; ++r) {
        tfsf_longitudinal_kernel(d, s, lo, hi, g.ex.view(), g.ey.view(), g.ez.view(), g.bx.view(), g.by.view(), g.bz.view());
        comm.exchange(d, {&g.ex, &g.bx});
    }
}

void Tfsf::begin(Comm& comm, double t) {
    if (modes_) { modes_->begin(t); return; }
    const Field* src[6] = {&inc_->ex, &inc_->ey, &inc_->ez, &inc_->bx, &inc_->by, &inc_->bz};
    Field* dst[6] = {&ex0_, &ey0_, &ez0_, &bx0_, &by0_, &bz0_};
    for (int q = 0; q < 6; ++q) tfsf_copy(d_, src[q]->view(), dst[q]->view());
    inc_->step(comm);
    if (o5_) {                                         // the closed means of t_n and t_n+1, for both passes and the corrections
        const Domain& d = d_;
        tfsf_averages(d, 2, ex0_.view(), ey0_.view(), ez0_.view(), bx0_.view(), by0_.view(), bz0_.view(), inc_->ex.view(),
                      inc_->bx.view(), pe_.view(), pb_.view());
        comm.exchange(d, {&pe_, &pb_});
        irip_o5_closure(*inc_, d, comm, pe_, 0, false);
        irip_o5_closure(*inc_, d, comm, pb_, 1, false);
        tfsf_cx(d, pe_.view(), xs_.view());
        tfsf_cx(d, pb_.view(), bs_.view());
        comm.exchange(d, {&xs_, &bs_});
    }
}

void Tfsf::set_o5(Comm& comm) {
    (void)comm;
    o5_ = true;
    if (modes_) return;                                // the plane waves carry the closure (d.o5, decision 76)
    inc_->o5 = true;
    pe_.resize(d_); pb_.resize(d_);
}

void Tfsf::o5_fix(Comm& comm, Field& exs, Field& bxs) {
    const int s = is_ - d_.off[0];
    if (modes_) tfsf_o5_fix_kernel(d_, s, s - 1, modes_->pe(), modes_->pb(), exs.view(), bxs.view());
    else tfsf_o5_fix_kernel(d_, s, 0, pe_.view(), pb_.view(), exs.view(), bxs.view());
    comm.exchange(d_, {&exs, &bxs});
}

void Tfsf::correct(Comm& comm, Field& ney, Field& nez, Field& nby, Field& nbz, int kind, bool commit) {
    const Domain& d = d_;
    const int s = is_ - d.off[0];
    if (modes_) {                                      // the plane waves of the solver (kind 0 RIP-CC, 2 iRIP: tfsf_modes.cpp)
        if ((kind == 2) != (d.solver == Solver::IRip)) throw std::runtime_error("Tfsf::correct: kind of another solver (modes)");
        const TfsfModes& M = *modes_;
        const TfCp none{nullptr, nullptr, 0};
        tfsf_correct_kernel(d, s, s - 1, M.ey(), M.ez(), M.by(), M.bz(), M.xs(), M.bs(), ney.view(), nez.view(), nby.view(),
                            nbz.view(), none, none, none, none, nullptr, commit);
        return;
    }
    // RIP-CC: the predictor the incident grid computed in its own step from t_n (in a layer with its CPML terms);
    // iRIP: the box means, local in each cell
    const Field& X = kind == 0 ? inc_->exs_ : xs_;
    const Field& B = kind == 0 ? inc_->bxs_ : bs_;
    if (kind != 0 && !o5_) {                           // o5: the closed means of begin()
        tfsf_averages(d, kind, ex0_.view(), ey0_.view(), ez0_.view(), bx0_.view(), by0_.view(), bz0_.view(), inc_->ex.view(),
                      inc_->bx.view(), xs_.view(), bs_.view());
        comm.exchange(d, {&xs_, &bs_});
    }
    TfCp yn{nullptr, nullptr, 0}, yh = yn, zn = yn, zh = yn;
    if (pml_) {
        const Pml& P = *pml_;
        if (P.cfg.on[1]) { yn = {P.a_n[1].data(), P.b_n[1].data(), d.off[1]}; yh = {P.a_h[1].data(), P.b_h[1].data(), d.off[1]}; }
        if (P.cfg.on[2]) { zn = {P.a_n[2].data(), P.b_n[2].data(), d.off[2]}; zh = {P.a_h[2].data(), P.b_h[2].data(), d.off[2]}; }
    }
    tfsf_correct_kernel(d, s, 0, ey0_.view(), ez0_.view(), by0_.view(), bz0_.view(), X.view(), B.view(),
                        ney.view(), nez.view(), nby.view(), nbz.view(), yn, yh, zn, zh, psi_.data(), commit);
}

// O3: the incident grid gets the same x layers (the same discrete scheme). At t = 0 the pulse must lie between the
// left layer and the plane, else the layer damps part of it before it reaches the plane. The plane waves need nothing:
// they are given at the plane, which must lie right of the left layer.
void Tfsf::set_x_absorber(Comm& comm, const Fields::XAbsorber& x) {
    if (modes_) {
        if (is_ <= x.cells) throw std::runtime_error("laser.tfsf_x: the TFSF plane must lie right of the left x absorber");
        return;
    }
    inc_->xabs = x;
    const Domain& d = d_;
    const double in = comm.allreduce_max(tfsf_max_below(d, inc_->ey.view(), inc_->ez.view(), inc_->ex.view(), x.cells + 1 - d.off[0]));
    const double rel = amax_ > 0 ? in / amax_ : 0;
    if (comm.rank == 0 && rel > 1e-4)
        std::fprintf(stderr, "warning: laser.tfsf_x: at t = 0 the incident pulse reaches into the left x absorber (%.1e of its peak), "
                             "which damps it before it reaches the plane: move the plane right or shorten laser.delay\n", rel);
}

// t = 0 (not on a restart): with the plane waves, the incident field of t = 0 is added to the total-field region of the main
// grid (the incident grid instead starts with the pulse left of the plane and cuts what reaches across)
void Tfsf::start(Fields& f, Comm& comm) {
    if (!modes_) return;
    const FV v[6] = {f.ex.view(), f.ey.view(), f.ez.view(), f.bx.view(), f.by.view(), f.bz.view()};
    modes_->add_incident(v, 0.0);
    comm.exchange(d_, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
}
