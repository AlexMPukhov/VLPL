// Macro-particles: field gather, momentum push, charge-conserving current deposition.
//
// Units: c = 1, lengths in 1/k0, times in 1/omega0, momenta u = p/(m c) = gamma v, fields in
// m_e c omega0 / e, densities in n_c. Charge q in e, mass m in m_e. A macro-particle of weight w
// carries w n_c / k0^3 physical particles, so rho = sum q w S / dV and wp^2 = n / n_c.
//
// Time levels: positions at n, momenta at n-1/2. One push: gather E^n, B^n at x^n, u^{n-1/2} ->
// u^{n+1/2}, x^{n+1} = x^n + dt v^{n+1/2}; the Esirkepov current j^{n+1/2} of the move x^n -> x^{n+1}
// is deposited on the charge grid (rho at (i+1/2, j, k)), which is exactly the continuity equation
// the RIP-CC solver needs: (rho^{n+1} - rho^n)/dt + dx Jx + dy Jy + dz Jz = 0.
//
// Storage and migration: src/species.hpp (structure of arrays with an attribute registry).
#pragma once
#include <string>
#include <vector>
#include "comm.hpp"
#include "fields.hpp"
#include "grid.hpp"
#include "species.hpp"
#include "tiles.hpp"

// None: the momentum is left as it is (the move + deposit of a species pushed before by push_store, decision 15)
enum class Pusher { Boris, HigueraCary, None };

// Shape order 1 (linear, CIC), 2 (quadratic, TSC) or 3 (cubic); the guard width must be >= 3 (shape 3: 5,
// Simulation sets it).
void push_and_deposit(Species& s, Fields& f, const Domain& d, Pusher pusher, int order);
// The same for part 1 (the boundary particles: cell within 2 guard widths of a face; it writes in[p] = 1 for the others)
// or part 2 (the interior ones, in[p] = 1); in: one byte per particle (device). particles.cpp, decision 92
void push_and_deposit_part(Species& s, Fields& f, const Domain& d, Pusher pusher, int order, int part, unsigned char* in);
// The same with the coloured-tile deposit (docs/specs/tile_deposit.md); needs s sorted by g (else it
// falls back to push_and_deposit). Returns the number of strays deposited with atomics.
long push_and_deposit_tiled(Species& s, Fields& f, const Domain& d, Pusher pusher, int order, const TileGrid& g);
// Box deposit (one block per tile, scratch box in shared memory on the GPU); needs s sorted by g (else
// push_and_deposit). Returns the number of strays deposited with global atomics.
long push_and_deposit_box(Species& s, Fields& f, const Domain& d, Pusher pusher, int order, const TileGrid& g);
void push_only(Species& s, const Fields& f, const Domain& d, Pusher pusher, int order);   // no deposit (tests)
// Gather and push only (no move): u^{n-1/2} -> u^{n+1/2}; EB gets per particle E^n, B^n at x^n and u^{n-1/2} (9 doubles;
// zero for the entry columns of a moving window, which are not pushed). The move and deposit follow with Pusher::None.
void push_store(Species& s, Fields& f, const Domain& d, Pusher pusher, int order, double* EB);
void deposit_rho(const Species& s, Field& rho, const Domain& d, int order);           // adds; caller reduces
// only the particles with x0 <= x < x1 (the window's entering columns, decision 91)
void deposit_rho(const Species& s, Field& rho, const Domain& d, int order, double x0, double x1);

// single-particle momentum update (exposed for tests); qmdt = q dt / m
HD void push_momentum(Pusher p, double qmdt, const double E[3], const double B[3], double u[3]);
