// Wake diagnostics for structure runs (docs/specs/structure3d.md, decision 102), Cart3D:
//
// `wake [<name>] { y; z; every; start }`: the force per unit charge on a test particle moving with c along +x at the probe
// (y, z) (absolute; default the box centre; the nearest node), F/q = (Ex, Ey - Bz, Ez + By), along the whole window, and
// its integral over time at fixed xi = x - c t (the wake potential seen by a witness at that xi:
// V(xi) = int F(xi + c t, t) c dt, volts). Needs a moving window at v = c; accumulated from max(start, the first shift)
// on. Written every `every` steps and at the end to dir/wake_<name>.txt: xi [um], Ex, Fy, Fz [V/m], Vx, Vy, Vz [V].
// The wake potential per charge of the driver Q is W = -V / Q (W_par > 0 decelerates); with a rigid driver
// (species.rigid_until) and the probe at the driver's offset, Vy, Vz give the transverse (dipole) wake.
//
// `beam_slices [<name>] { species; dx; every }`: slices of width dx (default h_x) along xi of one species: charge,
// current, centroids, rms sizes, mean and rms kinetic energy, mean transverse momenta -> dir/slices_<name>.txt.
//
// `side_flux { y_lo; y_hi; z_lo; z_hi; every }`: the Poynting energy S = E x B leaving the box y_lo < y < y_hi,
// z_lo < z < z_hi (node planes; default 2 cells inside the CPML layers or the edges) through its four side faces over
// the whole window length, per face and accumulated (trapezoidal in time), and the field energy inside -> dir/side_flux.txt.
#pragma once
#include <memory>
#include <string>
#include <vector>

struct WakeConfig { std::string name; double y = -1, z = -1, start = 0; long every = 0; };
struct SliceDiagConfig { std::string name, species; double dx = 0; long every = 0; };
struct SideFluxConfig { bool on = false; double lo[3] = {-1, -1, -1}, hi[3] = {-1, -1, -1}; long every = 0; };

class Operator;
std::unique_ptr<Operator> make_wake_operator(const WakeConfig& c);
std::unique_ptr<Operator> make_slices_operator(const SliceDiagConfig& c);
std::unique_ptr<Operator> make_side_flux_operator(const SideFluxConfig& c);
