// O6: the energy-conserving implicit particle mover with the iRIP box closure (numerics.mover = implicit;
// docs/specs/implicit_mover.md, research/o6_mover/).
//
// Crank-Nicolson in the particles, momenta at integer times (u^n, not u^{n-1/2} as with the explicit push):
//   u1 = u0 + q/m dt (Ebar + vbar x Bbar),  vbar = (u0 + u1) / (g0 + g1),  x1 = x0 + dt vbar,
//   J = the Esirkepov current of the straight move x0 -> x1 (unchanged: the Gauss law stays exact),
//   Ebar_a = h_a S_a / (x1 - x0)_a, S_a = sum of the work field Et_a of the box closure with the deposit's own weights,
// so m (g1 - g0) = q Ebar . (x1 - x0) = the work of Et on the particle's current, and W + K is conserved to the
// tolerance of the nonlinear solve (any Bbar: vbar x Bbar does no work). Bbar = (B^n + B^{n+1}) / 2 at the path midpoint.
// Nonlinear solve: a fixed point in the grid current J. One iteration: the iRIP trial step with J (irip_trial) gives Et
// and Bbar; one particle pass updates u1 and deposits J' from the new paths; the residual P^{-1} (J' - J) with the lumped
// cold response P = 1 + dt^2/4 sum q^2 w / (m g V) S, accelerated by Anderson. The step is committed with the deposit of
// the final paths (exact Gauss law whatever the residual), then the particles move.
#pragma once
#include <memory>

class Simulation;

struct MoverStats {
    long steps = 0, iterations = 0, max_iterations = 0, unconverged = 0;
    double last_residual = 0;
};

// One step of particles and fields (replaces push_deposit + the field step in the pipeline).
void implicit_mover_step(Simulation& s);
const MoverStats& implicit_mover_stats(const Simulation& s);
