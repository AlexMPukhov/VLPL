// Embedded perfect conductors in Cart3D (block `metal`, docs/specs/metal3d.md, decision 102): metal where the deck
// function metal(x, y, z) > 0 (absolute coordinates, fixed in the lab frame), as a staircase on the RIP-CC grid.
//
// The metal is a set of nodes (i, j, k) (the positions of Ex): M = 1 where the function is > 0 at the node. A field point
// is metal when all nodes of its transverse element are (the 2D Yee layout of the transverse plane):
//   Ex (node)                 M(i, j, k)
//   Ey, Bz (y edge)           M(i, j, k) M(i, j+1, k)
//   Ez, By (z edge)           M(i, j, k) M(i, j, k+1)
//   Bx (face)                 M at the four nodes j..j+1, k..k+1
// Rules (applied after the predictor, the transport and the corrector of the RIP-CC step, plain and CPML):
//   Ey, Ez of a metal edge: 0 (tangential E). Their partners Bz, By: 0 if the same edge is metal at i - 1 and i + 1
//   (inside along x; then they are the normal B of a y or z surface), otherwise the characteristic arriving from the
//   vacuum side (x face: Bz = T+_y, By = -T-_z at a left face, Bz = -T-_y, By = T+_z at a right face: total reflection of
//   the RIP transport, exact at normal incidence, as the RZ rings, docs/specs/rz_metal.md). A metal edge without metal
//   at i - 1 and i + 1 (one cell thick along x) cannot reflect from both sides: it is set to 0 (the setup warns).
//   Ex of a metal node: kept (from the corrector: the normal E of an x face, the surface charge) if the node is an x face
//   (vacuum at i - 1 or i + 1) and its four transverse neighbours are metal; otherwise 0 (tangential to a y or z surface,
//   or inside). The predictor Ex* follows the same rule.
//   Bx of a metal face: 0 (normal B of an x face, or inside); Bx* too.
// The Gauss law changes only at the rho points whose stencil touches a metal point (the surface charge): the checks skip
// (i+1/2, j, k) with any metal node in i..i+1, j-1..j+1, k-1..k+1, and the div B points (i+1/2, j+1/2, k+1/2) with any
// metal node in i..i+1, j..j+1, k..k+1.
// Particles (metal.particles, decision 122): pass (the default): they cross the metal; the fields at the metal points are 0, so
// they move on straight lines there, and their charge and current inside touch only skipped Gauss points (entry and exit
// radiate: transition radiation). absorb: a particle whose shape lies inside the metal (every node of the box around it
// metal) is removed (its kinetic energy goes to Species::lost_energy); the charge it carried in stays on the conductor
// as surface charge. The Gauss law at the checked points stays exact in both modes.
#pragma once
#include <string>
#include "expr.hpp"
#include "grid.hpp"

class Species;

struct Metal3Config {
    bool on = false;
    Expr fn;                                 // metal(x, y, z) > 0 inside (code units, absolute positions)
    bool absorb = false;                     // metal.particles = absorb (default pass: particles cross the metal, decision 122)
    // numerics.solver = irip (decision 124): the conductor is a Drude medium block named metal (Config::media) with the profile
    // function > 0; `on` is then false (no staircase conductor), density = n / n_c (default: omega_p dt = 10), nu / omega0
    bool medium = false;
    double density = 0, nu = 0;
};

// The plain view the kernels capture: the node mask (0 / 1 as double, the layout of the fields, guards valid)
struct Metal3View {
    FV m;
    HD bool M(int i, int j, int k) const { return m(i, j, k) != 0.0; }
    HD bool ey(int i, int j, int k) const { return M(i, j, k) && M(i, j + 1, k); }
    HD bool ez(int i, int j, int k) const { return M(i, j, k) && M(i, j, k + 1); }
    HD bool bx(int i, int j, int k) const { return M(i, j, k) && M(i, j + 1, k) && M(i, j, k + 1) && M(i, j + 1, k + 1); }
    // Ex (and Ex*) of node (i, j, k) is forced to 0
    HD bool ex_zero(int i, int j, int k) const {
        if (!M(i, j, k)) return false;
        const bool tin = M(i, j - 1, k) && M(i, j + 1, k) && M(i, j, k - 1) && M(i, j, k + 1);
        const bool face = !M(i - 1, j, k) || !M(i + 1, j, k);
        return !(tin && face);
    }
    HD bool any(int i0, int i1, int j0, int j1, int k0, int k1) const {
        for (int k = k0; k <= k1; ++k)
            for (int j = j0; j <= j1; ++j)
                for (int i = i0; i <= i1; ++i)
                    if (M(i, j, k)) return true;
        return false;
    }
    HD bool gauss_skip(int i, int j, int k) const { return any(i, i + 1, j - 1, j + 1, k - 1, k + 1); }
    HD bool divb_skip(int i, int j, int k) const { return any(i, i + 1, j, j + 1, k, k + 1); }
};

class Metal3 {
public:
    Metal3(const Domain& d, const Metal3Config& c);
    // evaluates the function at the nodes of the local columns [i0, i1) (all y, z including the guards)
    void fill(const Domain& d, int i0, int i1);
    // after a window shift (Domain::wx already advanced): the mask moves with the fields, the new columns are evaluated
    void shift(const Domain& d, Field& tmp);
    Metal3View view() const { return {mask.view()}; }
    long owned_nodes(const Domain& d) const;  // metal nodes among the owned ones (local)
    long thin_edges(const Domain& d) const;   // metal y / z edges with no metal at i - 1 and i + 1 (local)
    std::string describe() const;
    Field mask;
private:
    Buf<Expr::Ins> code_;
    int ncode_ = 0;
    std::string text_;
};

// The rules above (src/metal3d.cpp). halo: the predictor is computed one cell beyond the owned ones.
void metal_fix_pred(const Domain& d, const Metal3View& mv, const FV Exs, const FV Bxs, int halo);
void metal_fix_transport(const Domain& d, const Metal3View& mv, const FV nEy, const FV nEz, const FV nBy, const FV nBz);
void metal_fix_corr(const Domain& d, const Metal3View& mv, const FV Ex, const FV Bx);
// every forced point of E, B to 0 (the face B too), owned cells and guards: the initial fields
void metal_zero_fields(const Domain& d, const Metal3View& mv, const FV Ex, const FV Ey, const FV Ez, const FV Bx, const FV By, const FV Bz);
// absorbs the particles inside the metal (weight and momentum 0); returns the kinetic energy removed, w m (gamma - 1)
// remove = false: the particles are only flagged (removal_flags of species.hpp; the caller removes them with remove_flagged);
// flagged: their number (local)
double metal_absorb(Species& sp, const Domain& d, const Metal3View& mv, int order, bool remove = true, long* flagged = nullptr);
