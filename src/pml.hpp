// CPML absorbing layers at the y and z edges (docs/specs/pml.md, decision 31). In a layer of L cells
// every transverse derivative of the RIP-CC step becomes d/dy F + psi, psi <- b psi + a d/dy F, with
// b = exp(-sigma dt), a = b - 1 (kappa = 1, alpha = 0: the best choice of the 2D prototype,
// research/pml/), sigma = sigma_max (depth / L)^m. Each of the twelve derivative call sites has its own
// psi; zero fields behind the layer. Without layers the plain step of Fields runs (bit-identical).
#pragma once
#include <vector>
#include "comm.hpp"
#include "grid.hpp"

struct PmlConfig {
    bool on[3] = {false, false, false};   // axes 1 (y) and 2 (z)
    int cells = 16;
    double R0 = 1e-6;                     // reflection coefficient at normal incidence of the continuous layer
    int order = 3;                        // grading exponent m
};

class Fields;
struct BoundaryHooks;

class Pml {
public:
    // bmin > 0 floors b = exp(-sigma dt) (iRIP, decision 41: keeps its eigenbasis well conditioned)
    Pml(const Domain& d, const PmlConfig& c, double dt, double bmin = 0);
    PmlConfig cfg;
    // a, b per global index along y and z, at the node (position j h) and the half node ((j + 1/2) h);
    // a = 0 outside the layers
    Buf<double> a_n[3], b_n[3], a_h[3], b_h[3];
    // psi of the call sites: predictor, transport sources, corrector (y terms on ranks with a y layer, z
    // terms on ranks with a z layer; empty otherwise and never read)
    Field p_bz_y, p_ez_y, p_by_z, p_ey_z;      // predictor: dy Bz, dy Ez, dz By, dz Ey
    Field t_bx_y, t_ex_y, t_bx_z, t_ex_z;      // transport: dy mu Bx*, dy mu Ex*, dz mu Bx*, dz mu Ex*
    Field c_bz_y, c_ez_y, c_by_z, c_ey_z;      // corrector: dy <Bz>, dy <Ez>, dz <By>, dz <Ey>
    Field s_py, s_pz, s_fy, s_fz;              // transport sources Psi_y, Psi_z, Phi_y, Phi_z at (i+1/2)
    std::vector<Field*> psi_fields();          // the psi arrays (shifted with the moving window)
    // Gauss law and div B are checked only at global indices [lo, hi) along y and z (outside the layers
    // and one cell more)
    int check_lo(int a) const { return cfg.on[a] ? cfg.cells + 1 : 0; }
    int check_hi(int a, int N) const { return cfg.on[a] ? N - cfg.cells - 1 : N; }
};

// One RIP-CC step with the layers (the same as Fields::step otherwise).
void rip_cc_step_pml(Fields& f, Pml& p, Comm& comm, const BoundaryHooks& hooks, double t);
