// The TFSF x plane of O1 with the incident field as a sum of exact discrete plane waves (`laser.tfsf_incident = modes`,
// docs/specs/tfsf.md "Exact plane waves", decision 65). The corrections at the plane (tfsf.cpp) need the incident
// Ey, Ez, By, Bz and Ex*, Bx* only in the two cells next to the plane; here they are sums of plane waves, each an exact
// discrete solution of the solver (plane_mode, side.cpp), so the leakage is at round-off as with the incident grid,
// without that grid: the pulse may start outside the box and the plane may sit next to the left x absorber.
//
// A Gaussian beam (waist w0 in y', w0z in z; 0: uniform) focused at r_f = plane point + focus n, its peak there at
// t_f = delay + focus, is its angular spectrum on the lattice of the periodic y, z box:
//   E = Re sum A e^{i (k . (r - r_f) - w (t - t_f))},  A = i a0 e^{-i phase} (dnu / 2 pi) sqrt(pi / al) e^{-nu^2 / (4 al)}
//       x (sqrt(pi) w0 J / Ly) e^{-q_y'^2 w0^2 / 4} (sqrt(pi) w0z / Lz) e^{-k_z^2 w0z^2 / 4},
// w = 1 + nu, al = 2 ln 2 / tau^2 (the field envelope of LaserConfig), q_y' = -kx sin a + ky cos a of the continuum kx,
// J = |dq_y'/dk_y|; k_x of each component from the discrete dispersion. Evaluated per step separably: the sums over w
// per (k_y, k_z) on the host, then two device passes over k_z and k_y for the plane points.
#pragma once
#include <vector>
#include "comm.hpp"
#include "grid.hpp"
#include "laser.hpp"
#include "side.hpp"

class TfsfModes {
public:
    // the lasers with tfsf_x >= 0; is: the plane (global index); t_span: no periodic image of a pulse within it
    TfsfModes(const Domain& d, const std::vector<LaserConfig>& lasers, int is, double t_span);
    // the incident field of t_n (t = t_n) in the cells is - 1, is: Ey, Ez, By, Bz and the Ex*, Bx* of the solver (RIP-CC the
    // predictor of t_n, iRIP the mean of t_n and t_n+1), on the slab (local index 0 = global is - 1, transverse guards of 1)
    void begin(double t);
    FV ey() const { return f_[0].view(); }
    FV ez() const { return f_[1].view(); }
    FV by() const { return f_[2].view(); }
    FV bz() const { return f_[3].view(); }
    FV xs() const { return f_[4].view(); }
    FV bs() const { return f_[5].view(); }
    FV pe() const { return f_[6].view(); }         // o5: g_perp of the incident mean of Ex, Bx (Tfsf::o5_fix)
    FV pb() const { return f_[7].view(); }
    size_t count() const { return comps_.size(); }
    double peak() const { return peak_; }          // largest a0
    double mismatch() const { return mism_; }      // largest polarisation mismatch of a component
    // largest |E| of the incident field in the plane cells at t (local; host): the start check
    double plane_max(double t);
    // t = 0: the incident field (Ex .. Bz, main-grid views) added to the owned cells from the plane on (the total-field
    // region), so the run starts from an exact incident field whatever part of the pulse is already there; exchange after
    void add_incident(const FV f[6], double t);
private:
    void evaluate(double t, int nq, const long* ig, const int* sel, const void* outs);
    const Domain& d_;
    Domain ds_;                                    // the slab: two cells along x
    std::vector<SideTfsf::Comp> comps_;
    std::vector<int> ga_, gb_;                     // the (k_y, k_z) group of each component
    int is_ = 0;                                   // the plane (global x index, no moving window)
    int na_ = 0, nb_ = 0;
    std::vector<double> c_;                        // per step: [12][na][nb] complex sums over w (host)
    Buf<double> cd_, ty_, tz_, p_;                 // the sums, e^{i ky y_j}, e^{i kz z_k} tables, the first pass
    Field f_[8];
    double peak_ = 0, mism_ = 0;
    bool irip_ = false, o5_ = false;
};
