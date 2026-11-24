// Loading of cold plasma in kernels (Simulation::load): the cells of a box get ppc particles at regular
// positions with the momentum u0, filtered by the bounds, the slab and the density function (evaluated
// in the kernels, expr.hpp). Same particles in the same order as the host loop of Simulation::load,
// which keeps the hot species (their momenta come from a generator per cell). The host loop of the
// entering column was 5 ms of 6.5 ms per window shift in bench/lwfa_bubble on the RTX 5090 (2026-10-28).
#pragma once
#include "expr.hpp"
#include "slab.hpp"
#include "species.hpp"

struct ColdLoad {                  // plain values for the kernels
    int ppc[3];
    double h[3];
    double u0[3];
    double w;                      // weight of a particle at density factor 1
    double rmin[3], rmax[3];
    SlabProfile slab;              // slab.T 0: no slab (slab.hpp)
    // the lattice along x (boost.hpp Lattice): lattice coordinate q in [qlo, qhi), stored x = q + sx, the bounds, slab and
    // density function at x = ax q + bx (plain load: 1, 0, 0 and no bounds)
    double ax = 1, bx = 0, sx = 0, qlo = -1e300, qhi = 1e300;
};

// Appends the particles of the cells with absolute indices g0 + (i, j, k), 0 <= i, j, k < n. fn: the
// density function (empty: 1). Returns the largest density factor (slab profile times fn) of the loaded particles
// (0 if none).
double load_cold(Species& s, const ColdLoad& L, const Expr& fn, const long g0[3], const int n[3]);
