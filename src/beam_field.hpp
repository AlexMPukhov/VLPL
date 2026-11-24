// Steady self-field of a charge density moving along x with velocity beta (docs/specs/beam_self_field.md):
// the discrete RIP-CC form of (dxx / gamma^2 + dyy + dzz) phi = -rho, Ex = -dx phi / gamma^2,
// E_perp = -grad_perp phi, B = beta x E, so that the Gauss law holds to round-off. beta = 0 is the
// electrostatic (Poisson) initialisation. Open x: DFT in y, z (periodic), tridiagonal solve along x with
// phi = 0 just beyond both x edges. Periodic x (RIP-CC, iRIP, Yee; decision 98): 3D DFT, Yee's own Gauss law for
// solver = yee. Host code for the setup; every rank solves the global problem.
#pragma once
#include "comm.hpp"
#include "fields.hpp"

// Adds the fields of rho (owned cells, guards reduced) to f; refreshes the guards of E and B. steady (RIP-CC, beta > 0;
// decision 99): the exact discrete steady state of the RIP-CC step for the rigid beam instead of the Gauss-law field.
void add_beam_field(Fields& f, const Field& rho, double beta, const Domain& d, Comm& comm, bool steady = false);

// A fully periodic box with a self_field beam carries a net charge and current; its k = 0 part has no field in the
// init (no neutralizing background is possible otherwise). Each step the mean of Jx, Jy, Jz is removed: a uniform
// background that co-moves with the mean current (div J unchanged, so the Gauss law stays exact). Without it the
// uniform Ex grows as -<Jx> t and decelerates the beam (decision 98).
void remove_mean_current(Fields& f, const Domain& d, Comm& comm);
