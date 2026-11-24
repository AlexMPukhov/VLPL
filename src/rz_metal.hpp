// Embedded perfect conductors in RZ (block `rings`, docs/specs/rz_metal.md, decision 83): a stack of metal rings
// r_in < r < r_out, fixed in the lab frame, as a staircase on the RIP-CC grid. The metal of ring k occupies the closed
// index rectangle [IA_k, IB_k] x [J, K]: IA_k, IB_k the absolute z cells (lab frame, also with a moving window) of its
// faces, J, K the radial nodes of r_in, r_out. Rules per point (applied inside FieldsRZ::pass, RIP-CC only):
//   inside (columns IA < I < IB, or the corners / the r surfaces of the face columns): every field 0;
//   z face (I = IA: vacuum at I - 1; I = IB: vacuum at I + 1), J < j < K: E_r, E_phi, B_z = 0 (tangential E, normal B),
//   B_phi, B_r from the characteristic arriving from the vacuum side (B_phi = T+_y, B_r = -T-_z at a left face;
//   B_phi = -T-_y, B_r = T+_z at a right face: perfect reflection of the RIP transport), E_z (the normal E) from the
//   corrector. Particles whose shape lies entirely on metal and surface nodes are removed (as at the wall).
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "portable.hpp"

struct RingsConfig {
    bool on = false;
    double r_in = 0, r_out = 0, period = 0, gap = 0, z_start = 0, z_end = 0;   // code units
    double jacket = 0;                 // > 0: a solid tube r_out < r < r_out + jacket over [z_start, z_end] closes the gaps
                                       // (a closed corrugated pipe with an open end at z_end); 0: no jacket
};

// The plain view the kernels capture (n = 0: no metal)
struct MetalView {
    const long* ia = nullptr;
    const long* ib = nullptr;
    int n = 0;                         // rings
    int J = 0, K = 0;                  // radial nodes of r_in, r_out
    long N = 0;                        // periodic z: the global cells (absolute indices wrap); open: 0
    long ta = 0, tb = -1;              // jacket: absolute columns [ta, tb] (tb < ta: none)
    int KJ = 0;                        // jacket: radial nodes K..KJ
    HD long wrap(long I) const { return N > 0 ? ((I % N) + N) % N : I; }
    HD bool in(long I) const {         // column I (absolute) inside a ring's closed z range
        if (n == 0) return false;
        I = wrap(I);
        if (I < ia[0]) return false;
        int lo = 0, hi = n;            // ia[lo] <= I < ia[hi]
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (ia[mid] <= I) lo = mid; else hi = mid;
        }
        return I <= ib[lo];
    }
    // the column type of the rings alone: 0 vacuum, 1 interior, 2 left face, 3 right face (tests, diagnostics)
    HD int col(long I) const {
        if (!in(I)) return 0;
        const bool l = in(I - 1), r = in(I + 1);
        return l && r ? 1 : (!l ? 2 : 3);
    }
    HD bool tube(long I) const { I = wrap(I); return I >= ta && I <= tb; }
    // node (I, r_j) inside or on the metal: a ring's closed rectangle [IA_k, IB_k] x [J, K] or the jacket [ta, tb] x [K, KJ]
    HD bool M(long I, int j) const { return (j >= J && j <= K && in(I)) || (tb >= ta && j >= K && j <= KJ && tube(I)); }
    // half node (I, r_{j+1/2}) inside the metal: both radial nodes
    HD bool MH(long I, int j) const { return M(I, j) && M(I, j + 1); }
    // the rule of a node point (E_z, E_phi, B_r at r_j) / half-node point (E_r, B_z, B_phi at r_{j+1/2}) in column I
    // (absolute): 0 free, 1 zero (inside, or on an r surface), 2 left z face (vacuum at I - 1), 3 right z face
    // (vacuum at I + 1). Without the jacket the same rules as the column types of decision 83.
    HD int node_at(long I, int j) const {
        if (n == 0 || !M(I, j)) return 0;
        const bool l = M(I - 1, j), r = M(I + 1, j);
        if (l && r) return 1;
        if (j > 0 && (!M(I, j - 1) || !M(I, j + 1))) return 1;
        return !l ? 2 : 3;
    }
    HD int half_at(long I, int j) const {
        if (n == 0 || !MH(I, j)) return 0;
        const bool l = MH(I - 1, j), r = MH(I + 1, j);
        return l && r ? 1 : (!l ? 2 : 3);
    }
    // a Gauss point (i+1/2, r_j) touched by the metal (surface charge): node j of column I or I + 1 in the metal
    HD bool gauss_skip(long I, int j) const { return n > 0 && (M(I, j) || M(I + 1, j)); }
    // a div B point (i+1/2, r_{j+1/2}): one of its four corner nodes in the metal
    HD bool divb_skip(long I, int j) const { return n > 0 && (M(I, j) || M(I, j + 1) || M(I + 1, j) || M(I + 1, j + 1)); }
};

struct RadialGrid;
class FieldsRZ;
class Species;
struct MetalRZ {
    RingsConfig cfg;
    std::vector<long> ia, ib;          // host copies
    Buf<long> dia, dib;
    int J = 0, K = 0;
    long N = 0;
    long ta = 0, tb = -1;              // jacket columns (tb < ta: none)
    int KJ = 0;
    MetalView view() const { return {dia.data(), dib.data(), int(ia.size()), J, K, N, ta, tb, KJ}; }
    MetalView host_view() const { return {ia.data(), ib.data(), int(ia.size()), J, K, N, ta, tb, KJ}; }   // host code only
    std::string describe(double dz, const RadialGrid& g) const;
};
// Snaps the rings onto the grid (nz global cells, dz, the radial grid); periodic: indices within [0, nz).
std::shared_ptr<MetalRZ> rz_metal_setup(const RingsConfig& c, int nz, double dz, bool periodic, const RadialGrid& g);
// Removes the particles whose shape (order 1, 2) lies entirely in a ring's closed index rectangle: they are moved beyond
// rwall, so the following migrate() absorbs them (their kinetic energy counted in Species::lost_energy).
void rz_metal_remove(Species& sp, const FieldsRZ& f, double rwall, int order);
