// Lossless, non-dispersive dielectrics eps(r, z) in the explicit RIP-CC RZ step (blocks `dielectric [<name>]`,
// docs/specs/rz_dielectric.md, decision 100). Each block is a rectangle r_in < r < r_out, z_start < z < z_end, fixed in
// the lab frame (it moves through a moving window as the rings do), with chi = eps - 1; blocks add their chi.
// The medium is the polarization current J_P = d_t P, P = chi E, time-centred (Crank-Nicolson), added to the free
// current of every component at its own position:
//   E_z  (i, r_j):           (1 + chi_z) (E_z^{n+1} - E_z^n) = dt (curl B - J_z): local (predictor: (1 + chi_z) too);
//   J_r  (i+1/2, r_{j+1/2}): y_a = dt J_P = chi_a mu_z(E_r^{n+1} - E_r^n) at a = i + 1/2,
//   J_phi(i+1/2, r_j):       the same with E_phi.
// The transverse fields of node i get -(y_{i-1/2} + y_{i+1/2}) / 2 (E) and -+(y_{i-1/2} - y_{i+1/2}) / 2 (B_phi, B_r) through
// the characteristics; with dE_i = m_i (dEv_i - (y_{i-1/2} + y_{i+1/2}) / 2) (dEv: the change of the vacuum transport, m_i = 0
// where the metal of the rings holds E at 0) every (radial row, mode slot, component) is the tridiagonal line along z
//   y_a (1 + chi_a (m_a + m_{a+1}) / 4) + chi_a / 4 (m_a y_{a-1} + m_{a+1} y_{a+1}) = chi_a / 2 (m_a dEv_a + m_{a+1} dEv_{a+1}),
// solved directly (Thomas, with the ranks along z: a partition solve with one allgather per step). chi enters every point
// as the r-weighted volume fraction of its control volume inside the blocks. Then exactly div(E + P) = rho (P = chi E with
// the same staggering), the scheme is unconditionally stable and lossless (von Neumann analysis in the decision).
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "portable.hpp"

struct DielectricConfig {
    std::string name;
    double r_in = 0, r_out = 0, z_start = 0, z_end = 0;   // code units
    double eps = 1;
    double ramp = 0;                    // > 0: chi rises linearly over this length centred on z_start (falls at z_end)
};

// The view the kernels capture: chi at the E_z nodes (cz), the J_r points (cr) and the J_phi points (cp) of the local
// columns i = -1 .. nz (half points: a = i + 1/2), index (i + 1) + (nz + 2) j. on = false: no dielectric.
struct DielView {
    const double *cz = nullptr, *cr = nullptr, *cp = nullptr;
    int nz = 0;
    bool on = false;
    HD long at(int i, int j) const { return long(i + 1) + long(nz + 2) * j; }
    HD double z(int i, int j) const { return cz[at(i, j)]; }
    HD double r(int a, int j) const { return cr[at(a, j)]; }
    HD double p(int a, int j) const { return cp[at(a, j)]; }
};

struct Domain;
struct RadialGrid;
class Comm;
class FieldsRZ;
class Field;
class DielectricRZ {
public:
    DielectricRZ(const std::vector<DielectricConfig>& blocks, const Domain& d, const RadialGrid& g);
    // chi of the local columns for the window position of d (recomputed when the window has moved)
    void prepare(const Domain& d, const RadialGrid& g);
    DielView view() const { return {cz_.data(), cr_.data(), cp_.data(), nz_, true}; }
    // After the vacuum transport into f.ner_ .. f.nbp_ (guards exchanged): solves the lines and corrects the transverse
    // fields of the owned nodes; the caller exchanges them again.
    void correct(Comm& comm, FieldsRZ& f);
    std::string describe(const RadialGrid& g, double dz) const;
    int jlo = 0, jhi = 0;              // the radial rows with dielectric (nodes or half nodes)
    std::vector<DielectricConfig> blocks;
private:
    int nz_ = 0, nr_ = 0;
    long c0_ = -(1L << 60);            // the window position of chi
    Buf<double> cz_, cr_, cp_, bk_;    // chi; the blocks (r_in, r_out, z_start, z_end, chi) on the device
    Buf<double> cpr_, x_, y_, ex_, ga_, red_;   // Thomas scratch, the three local solutions, y, edge values, reduced systems
    std::vector<double> exh_, gah_;
};
// RIP-CC moving window with dielectric blocks: E_z of the entering column from div D of the column before it (rz_dielectric.cpp)
// layer > 0: also the columns of the front absorbing layer (their E_z from div D, sequentially from the layer start), so the damping
// there leaves no charge
void rz_window_entry_dielectric(FieldsRZ& f, const Field& gauss0, int layer = 0);
