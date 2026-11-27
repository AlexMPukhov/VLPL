// Embedded perfect conductors that depend on the azimuth in RZ (block `metal`, docs/specs/rz_metal_phi.md, decision 103):
// metal where a deck function f(x, y, z, r, phi) > 0 (lab frame, fixed; x = r cos phi, y = r sin phi, z the axis).
//
// Angular grid. The M modes are represented exactly by their values on Np = 2M - 1 equidistant angles. With
// phi_derivative = difference (the default) the phi derivative of the RZ solver becomes the local staggered difference
// over dphi = 2 pi / Np, i m -> i m_eff, m_eff = (2 / dphi) sin(m dphi / 2) (RadialGrid::set_meff; diagonal in the modes,
// so every mode keeps its own radial operators), and the fields sit on a cylindrical Yee grid in phi:
//   E_z, E_r, B_phi (and rho, J_z, J_r) at phi_k = k dphi;   E_phi, B_r, B_z (and J_phi) at phi_k + dphi / 2.
// Then the metal is a staircase as in a Cartesian Yee code: tangential E = 0 on the metal surfaces, normal B = 0. With
// the spectral phi derivative (phi_derivative = spectral: m unchanged, every component at phi_k) the zeros inside the metal
// enter the vacuum through the global derivative (Gibbs) and B_z loses its Neumann condition at phi surfaces: the TE modes
// of a square guide come out wrong (research/rz_metal_phi/xsection.py). Spectral is right for axisymmetric masks only, where
// it reproduces the rings block to round-off.
//
// Masks from node bits Mn(I, j, k) (column I, node r_j, angle phi_k): Mh = Mn(j) Mn(j+1) at (j+1/2, k) (E_r, B_phi);
// Mp = Mn(k) Mn(k+1) at (j, k+1/2) (E_phi, B_r); Mc = Mh(k) Mh(k+1) at (j+1/2, k+1/2) (B_z). Spectral: Mp = Mn, Mc = Mh.
// After each pass of the RIP-CC step the fields are taken to their angles and, per point with its own mask X:
//   not metal: free;  both z neighbours metal (X(I-1) X(I+1)): zero;  else a z face (left: vacuum at I - 1) with the
//   rules of the rings block (decision 83): E_r = E_phi = B_z = 0; B_phi, B_r from the characteristic of the vacuum side
//   (B_phi = E_r + B_phi, B_r = B_r - E_phi at a left face; B_phi - E_r, E_phi + B_r at a right face) unless the component
//   is normal to a transverse metal surface through the point (B_phi: an angular neighbour B_z cell (j+1/2, k +- 1/2) in
//   vacuum; B_r: a radial neighbour cell (j +- 1/2, k+1/2) in vacuum, or the axis): then 0; E_z from the corrector unless
//   tangential to such a surface (any of its four B_z cells in vacuum): then 0.
// For an axisymmetric mask these are the rules of rings / jacket (the same at every angle).
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "expr.hpp"
#include "grid.hpp"
#include "portable.hpp"

struct MetalPhiConfig {
    bool on = false;
    bool absorb = false;               // metal.particles = absorb (default pass, decision 122)
    Expr fn;                           // metal where fn(x, y, z, r, phi) > 0 (code units)
    std::string text;
    bool spectral = false;             // phi_derivative = spectral (axisymmetric masks only); default: difference
};

// The plain view the kernels capture (np = 0: no metal).
struct MetalPhiView {
    const unsigned char* bits = nullptr;         // per (column slot, node j = 0 .. nr, angle k): 1 metal node
    const double *cs = nullptr, *sn = nullptr;   // cos, sin(m phi_k): [k * M + m]
    const double *ch = nullptr, *sh = nullptr;   // at the staggered angles phi_k + dphi/2 (spectral: = cs, sn)
    long L = 0;                        // column slots of the ring buffer (absolute column I at slot I mod L)
    int nrn = 0, np = 0, M = 0;        // nodes 0 .. nr (nrn = nr + 1), angles, modes
    long N = 0;                        // periodic z: the global cells (absolute columns wrap); open: 0
    bool stag = true;                  // staggered angles (phi_derivative = difference)
    HD long slot(long I) const { if (N > 0) I = ((I % N) + N) % N; return ((I % L) + L) % L; }
    HD bool Mn(long I, int j, int k) const {
        if (j < 0 || j >= nrn) return false;
        k = ((k % np) + np) % np;
        return bits[(size_t(slot(I)) * nrn + j) * np + k] != 0;
    }
    HD bool Mh(long I, int j, int k) const { return Mn(I, j, k) && Mn(I, j + 1, k); }
    HD bool Mp(long I, int j, int k) const { return Mn(I, j, k) && (!stag || Mn(I, j, k + 1)); }
    HD bool Mc(long I, int j, int k) const { return Mh(I, j, k) && (!stag || Mh(I, j, k + 1)); }
    // E_z at (j, k): 1 zero, 0 free (corrector)
    HD int rEz(long I, int j, int k) const {
        if (!Mn(I, j, k)) return 0;
        if (Mn(I - 1, j, k) && Mn(I + 1, j, k)) return 1;
        const int kl = stag ? k - 1 : k;           // the B_z cells around the node
        if (!Mc(I, j, k) || !Mc(I, j, kl)) return 1;
        if (j > 0 && (!Mc(I, j - 1, k) || !Mc(I, j - 1, kl))) return 1;
        return 0;
    }
    // B_phi at (j+1/2, k) (E_r = 0 wherever Mh): 0 free, 1 zero, 2 left face, 3 right face
    HD int rBp(long I, int j, int k) const {
        if (!Mh(I, j, k)) return 0;
        const bool l = Mh(I - 1, j, k), r = Mh(I + 1, j, k);
        if (l && r) return 1;
        if (!Mc(I, j, k) || !Mc(I, j, stag ? k - 1 : k)) return 1;     // normal to a phi surface
        return !l ? 2 : 3;
    }
    // B_r at (j, k+1/2) (E_phi = 0 wherever Mp): 0 free, 1 zero, 2 left face, 3 right face
    HD int rBr(long I, int j, int k) const {
        if (!Mp(I, j, k)) return 0;
        const bool l = Mp(I - 1, j, k), r = Mp(I + 1, j, k);
        if (l && r) return 1;
        if (j == 0 || !Mc(I, j, k) || !Mc(I, j - 1, k)) return 1;     // normal to an r surface (or the axis)
        return !l ? 2 : 3;
    }
    HD bool any_node(long I, int j) const {
        if (np == 0) return false;
        for (int k = 0; k < np; ++k) if (Mn(I, j, k)) return true;
        return false;
    }
    // a Gauss point (i+1/2, r_j) touched by the metal at any angle (surface charge, all modes): node j of column I or I + 1
    HD bool gauss_skip(long I, int j) const { return np > 0 && (any_node(I, j) || any_node(I + 1, j)); }
    // a div B point (i+1/2, r_{j+1/2}): one of its corner nodes in the metal at any angle
    HD bool divb_skip(long I, int j) const {
        return np > 0 && (any_node(I, j) || any_node(I, j + 1) || any_node(I + 1, j) || any_node(I + 1, j + 1));
    }
};

// Mode slots <-> collocation angles (also used by the dielectric line solves on the angles, rz_dielectric.cpp):
constexpr int RZ_PHI_MAXNP = 255;      // angles per kernel thread (modes <= 128)
// field value at angle k (table c, s: integer or staggered angles) from the slots of (i, j): F_0 + 2 sum (Re F_m cos - Im F_m sin)
HD inline double rz_phi_at_angle(const FV& F, int i, int j, int k, const double* tc, const double* ts, int M) {
    double s = F(i, j, 0);
    const double* c = tc + size_t(k) * M;
    const double* sn = ts + size_t(k) * M;
    for (int m = 1; m < M; ++m) s += 2 * (F(i, j, 2 * m - 1) * c[m] - F(i, j, 2 * m) * sn[m]);
    return s;
}
// adds the changes d at the angles ks[0 .. n) (ks = nullptr: all angles 0 .. np - 1, n = np) back to the modes:
// dF_0 = sum d / Np, dRe F_m = sum d cos / Np, dIm F_m = -sum d sin / Np (exact for degree M - 1 on 2M - 1 angles, also on
// the staggered ones)
HD inline void rz_phi_add_back(const FV& F, int i, int j, const int* ks, const double* d, int n, const double* tc, const double* ts,
                               int M, int np) {
    const double w = 1.0 / np;
    double s0 = 0;
    for (int q = 0; q < n; ++q) s0 += d[q];
    F(i, j, 0) += w * s0;
    for (int m = 1; m < M; ++m) {
        double a = 0, b = 0;
        for (int q = 0; q < n; ++q) {
            const size_t kk = size_t(ks ? ks[q] : q) * M + m;
            a += d[q] * tc[kk];
            b -= d[q] * ts[kk];
        }
        F(i, j, 2 * m - 1) += w * a;
        F(i, j, 2 * m) += w * b;
    }
}

class FieldsRZ;
class Species;
class Field;
struct Domain;
class MetalPhi {
public:
    // the domain's z decomposition, M modes, the node radii r_0 .. r_nr (more are ignored); the local columns get filled
    MetalPhi(const MetalPhiConfig& c, const Domain& d, int modes, const std::vector<double>& rnodes);
    MetalPhiView view() const;
    MetalPhiView host_view() const;    // the same on the host copy of the bits (host code, tests; no angle tables)
    // the columns c0 - 2 .. c0 + n + 1 of the domain (absolute) present in the ring buffer (a window shift adds one)
    void ensure(const Domain& d);
    std::string describe() const;
    int angles() const { return np_; }
    long metal_nodes_local() const;    // metal (column, node, angle) points held (diagnostics)
    bool has_phi_surfaces() const;     // a column held whose mask depends on the angle
    MetalPhiConfig cfg;
private:
    void fill(long I, long s);
    int np_ = 0, M_ = 0, nrn_ = 0;
    long L_ = 0, N_ = 0, n_ = 0;
    double dz_ = 0;
    std::vector<double> r_;
    std::vector<unsigned char> host_;
    std::vector<long> have_;           // absolute column held by each slot (LONG_MIN: none)
    Buf<unsigned char> bits_;
    Buf<double> cs_, sn_, ch_, sh_;
};

// Applies the rules after a pass of the RIP-CC step (src/rz_metal_phi.cpp):
//   stage 0 (predictor): a = Ez*, b = Bz*;   stage 2 (corrector): a = Ez, b = Bz;
//   stage 1 (transport): a = E_r, b = E_phi, c = B_r, e = B_phi.
void rz_metal_phi_apply(const MetalPhi& mp, FieldsRZ& f, int stage, Field& a, Field& b, Field* c = nullptr, Field* e = nullptr);
// Particles whose shape lies on metal nodes at their nearest angle are moved beyond rwall (migrate() absorbs them).
void rz_metal_phi_remove(Species& sp, const FieldsRZ& f, const MetalPhi& mp, double rwall, int order);
