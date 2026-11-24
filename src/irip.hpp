// Internal interface between src/irip.cpp and src/irip_pml.cpp (iRIP with CPML layers, decision 41).
#pragma once
#include <vector>
#include "comm.hpp"
#include "grid.hpp"

class Pml;
class Fields;

// O5 closure (src/irip.cpp) with the work arrays of the iRIP grid F (after its first step): f <- Cx C_perp f (cx = false:
// C_perp only), kind 0 (Ex*) or 1 (Bx*); the guards exchanged.
void irip_o5_closure(Fields& F, const Domain& d, Comm& comm, Field& f, int kind, bool cx = true);

// Eigenbasis of the transverse operator along a pml axis (1 or 2) for kind 0 (Ex, nodes) or 1 (Bx, half
// nodes), global size n = d.N[axis]: V (n x n, column q = mode q), Vi = V^-1, lam = eigenvalues of -L.
void irip_axis_basis(const Domain& d, const Pml& P, int axis, int kind, std::vector<double>& V, std::vector<double>& Vi,
                     std::vector<double>& lam);
// One pass of the box step with the layers (RIP-CC's CPML step with the given Ex*, Bx*): transverse
// fields into ney..nbz (exchanged), Ex, Bx into oEx, oBx; psi updated only if commit.
void irip_pml_pass(const Domain& d, Comm& comm, Pml& P, const FV Ex, const FV Ey, const FV Ez, const FV Bx, const FV By,
                   const FV Bz, const FV Jx, const FV Jy, const FV Jz, const FV Exs, const FV Bxs, Field& ney, Field& nez,
                   Field& nby, Field& nbz, const FV oEx, const FV oBx, bool commit, class Tfsf* tf = nullptr,
                   int kind = 0, const class SideTfsf* sd = nullptr);   // tf: the TFSF plane (tfsf.hpp, kind as Tfsf::correct);
                                                                     // sd: the side tube (side.hpp)
