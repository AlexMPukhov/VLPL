// grid.absorber_particles (decision 111; Alexander 2026-11-20, nci/trap/d3): the particles that enter the transverse absorbing layers.
// Cart3D: the CPML layers at the y, z edges (grid.boundary_y/z = pml: depth > 0 at y < L h and y > (N - L) h, pml.cpp); RZ: the radial
// absorber of grid.boundary_r = absorbing (r > r_{j0}, j0 = nr - cells, rz_absorb).
// Why: a plasma streaming through the window of a boosted run (u = -5.9, gamma 6) that fills the box up to the PML drives a
// near-uniform transverse field that grows along the plasma's path (x10 per 100 fs, 4e10 V/m near the axis); with periodic y, z or
// with the plasma kept off the layers the same run stays at the noise level (7e6 V/m) for 1500 fs.
//   remove (the default, Alexander: "keep the current deposition out of the PML"): no current inside the layers: after the deposit
//     and the filter every current component at a point of positive depth is set to zero (Cart3D: J_y at the half nodes, J_x, J_z
//     at the nodes along y; likewise along z; RZ: J_z, J_phi at r_j > r_{j0}, J_r at r_{j+1/2} > r_{j0}); a particle is removed
//     once it is in a layer (y < L h, y > (N - L) h; RZ r >= r_{j0}). The current of a particle stops at the face, so the charge
//     continuity is broken there (charge vanishes at the face).
//   conserve: the current is deposited into the layer (charge conserving); a particle is removed once its shape no longer touches a
//     node of the physical domain, plus one cell (Cart3D: y/h < L - 2 - (s - 1)/2 or > N - L + 1 + (s - 1)/2; RZ r >= r_{j0} +
//     (1 + (s - 1)/2) dn): the Gauss law in the physical domain holds, the removed charge stays in the layer.
//   keep: the particles cross the layers and leave at the outer edge (the behaviour before decision 111).
// The removed number and charge q w Z are counted per species (Species::lost_count, lost_charge, local) and printed at the end.
#include <cmath>
#include <cstdio>
#include "rz.hpp"
#include "sim.hpp"

namespace {
constexpr double FAR = 1e30;                          // moved there, migrate() drops it at the open y or z edge
const double e_si = 1.602176634e-19, me_si = 9.1093837015e-31, eps0_si = 8.8541878128e-12, c_si = 299792458.0;
}

double absorber_remove_cart(Species& sp, const Domain& d, const int L[3], int order, int mode, double& count) {
    count = 0;
    const long n = long(sp.size());
    if (n == 0 || (L[1] <= 0 && L[2] <= 0) || mode == ABSORBER_KEEP) return 0;
    double* Y = sp.ptr<double>(Species::Y);
    double* Z = sp.ptr<double>(Species::Z);
    const double* W = sp.ptr<double>(Species::W);
    const int* Zs = sp.zslot >= 0 ? sp.ptr<int>(sp.zslot) : nullptr;
    const double mg = 0.5 * (order - 1), q = sp.q;
    const double ih1 = 1 / d.h[1], ih2 = 1 / d.h[2];
    const bool cut = mode == ABSORBER_REMOVE;
    auto lo_of = [&](int a) { return L[a] > 0 ? (cut ? double(L[a]) : L[a] - 2 - mg) : -FAR; };
    auto hi_of = [&](int a) { return L[a] > 0 ? (cut ? double(d.N[a] - L[a]) : d.N[a] - L[a] + 1 + mg) : FAR; };
    const double lo1 = lo_of(1), hi1 = hi_of(1), lo2 = lo_of(2), hi2 = hi_of(2);
    const double Ly = d.L(1), Lz = d.L(2);
    const double qs = par_sum(n, [=] HD(long p) {
        const double y = Y[p] * ih1, z = Z[p] * ih2;
        if (!(y < lo1 || y > hi1 || z < lo2 || z > hi2)) return 0.0;
        if (y < lo1) Y[p] = -FAR; else if (y > hi1) Y[p] = Ly + FAR;
        else if (z < lo2) Z[p] = -FAR; else Z[p] = Lz + FAR;
        return q * W[p] * (Zs ? double(Zs[p]) : 1.0);
    });
    count = par_sum(n, [=] HD(long p) { return (Y[p] < -0.5 * FAR || Y[p] > 0.5 * FAR || Z[p] < -0.5 * FAR || Z[p] > 0.5 * FAR) ? 1.0 : 0.0; });
    return qs;
}

void absorber_zero_currents_cart(Fields& f, const Domain& d, const int L[3]) {
    if (L[1] <= 0 && L[2] <= 0) return;
    const FV Jx = f.jx.view(), Jy = f.jy.view(), Jz = f.jz.view();
    const int o1 = d.off[1], o2 = d.off[2], N1 = d.N[1], N2 = d.N[2], L1 = L[1], L2 = L[2];
    // depth > 0: node g at g < L or g > N - L; half node g + 1/2 at g + 1/2 < L or g + 1/2 > N - L
    for_owned_halo(d, d.ng, [=] HD(int i, int j, int k) {
        const int gy = o1 + j, gz = o2 + k;
        const bool ny = L1 > 0 && (gy < L1 || gy > N1 - L1), hy = L1 > 0 && (gy < L1 || gy >= N1 - L1),
                   nz = L2 > 0 && (gz < L2 || gz > N2 - L2), hz = L2 > 0 && (gz < L2 || gz >= N2 - L2);
        if (ny || nz) Jx(i, j, k) = 0;                  // J_x: nodes in y and z
        if (hy || nz) Jy(i, j, k) = 0;                  // J_y: half node in y, node in z
        if (ny || hz) Jz(i, j, k) = 0;                  // J_z: node in y, half node in z
    });
}

double rz_absorber_radius(const RadialGrid& g, int cells, int order, int mode) {
    const int jf = g.nr - cells;                      // the inner face node of the radial absorber
    if (mode == ABSORBER_REMOVE) return g.r[size_t(jf)];
    return g.r[size_t(jf)] + (1 + 0.5 * (order - 1)) * g.dn[size_t(jf)];
}

void rz_absorber_zero_currents(FieldsRZ& f, int cells) {
    const Domain& d = f.dom();
    const int nz = d.n[0], nr = d.n[1], S = d.n[2], j0 = nr - cells, g = d.ng;
    const FV Jz = f.jz.view(), Jr = f.jr.view(), Jp = f.jp.view();
    // rows j0 .. nr + guards: J_z, J_phi at r_j (j > j0), J_r at r_{j+1/2} (j >= j0); every column with its z guards
    const int rows = nr + g - j0, cols = nz + 2 * g;
    par_for(long(cols) * rows * S, [=] HD(long q) {
        const int i = int(q % cols) - g, j = j0 + int((q / cols) % rows), k = int(q / (long(cols) * rows));
        Jr(i, j, k) = 0;
        if (j > j0) { Jz(i, j, k) = 0; Jp(i, j, k) = 0; }
    });
}

double rz_absorber_count(Species& sp, double rrm, double& count) {
    count = 0;
    const long n = long(sp.size());
    if (n == 0) return 0;
    const double* Y = sp.ptr<double>(Species::Y);
    const double* Z = sp.ptr<double>(Species::Z);
    const double* W = sp.ptr<double>(Species::W);
    const int* Zs = sp.zslot >= 0 ? sp.ptr<int>(sp.zslot) : nullptr;
    const double q = sp.q, r2 = rrm * rrm;
    count = par_sum(n, [=] HD(long p) { return Y[p] * Y[p] + Z[p] * Z[p] >= r2 ? 1.0 : 0.0; });
    return par_sum(n, [=] HD(long p) { return Y[p] * Y[p] + Z[p] * Z[p] >= r2 ? q * W[p] * (Zs ? double(Zs[p]) : 1.0) : 0.0; });
}

void absorber_report(Simulation& s, FILE* f) {
    const bool on = s.cfg.absorber_mode != ABSORBER_KEEP && (s.cfg.rz ? s.cfg.rz_abs_cells > 0 : (s.cfg.pml.on[1] || s.cfg.pml.on[2]));
    if (!on) return;
    const size_t S = s.species.size();
    std::vector<double> v(2 * S);
    for (size_t i = 0; i < S; ++i) { v[2 * i] = s.species[i].lost_count; v[2 * i + 1] = s.species[i].lost_charge; }
    if (S) s.comm.allreduce_sum(v.data(), int(2 * S));           // collective
    if (s.comm.rank != 0 || !f) return;
    const double k0 = 2 * M_PI / s.cfg.lambda0, w0 = c_si * k0;
    const double n_c = eps0_si * me_si * w0 * w0 / (e_si * e_si);
    const double pc = e_si * n_c / (k0 * k0 * k0) * 1e12;            // pC per unit charge x weight
    std::fprintf(f, "removed in the absorbing layers (grid.absorber_particles = %s):",
                 s.cfg.absorber_mode == ABSORBER_REMOVE ? "remove" : "conserve");
    for (size_t i = 0; i < S; ++i)
        std::fprintf(f, " %s %.0f (%.4g pC)%s", s.species[i].name.c_str(), v[2 * i], v[2 * i + 1] * pc, i + 1 < S ? "," : "");
    std::fprintf(f, "\n");
}
