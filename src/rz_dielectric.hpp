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
// solved directly (Thomas, with the ranks along z: a partition solve with one allgather per step). With the block metal
// (conductors depending on phi, decision 103) the masks depend on the angle: then the lines are solved on the 2M - 1
// collocation angles (E_r at phi_k, E_phi at its own, staggered angles; the modes and these angles are a bijection) and the
// corrections go back to the modes. chi enters every point
// as the r-weighted volume fraction of its control volume inside the blocks. Then exactly div(E + P) = rho (P = chi E with
// the same staggering), the scheme is unconditionally stable and lossless (von Neumann analysis in the decision).
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "expr.hpp"
#include "portable.hpp"

struct DielectricConfig {
    std::string name;
    double r_in = 0, r_out = 0, z_start = 0, z_end = 0;   // code units (with fn: the bounding box, default everything)
    double eps = 1;
    double ramp = 0;                    // > 0: chi rises linearly over this length centred on z_start (falls at z_end)
    Expr fn;                            // eps as a deck expression of (x, y, z, r, phi) (lab frame, code units); empty: constant eps
    bool phi = false;                   // fn depends on the azimuth (uses x, y or phi)
};

// The view the kernels capture: chi at the E_z nodes (cz), the J_r points (cr) and the J_phi points (cp) of the local
// columns i = -1 .. nz (half points: a = i + 1/2), index (i + 1) + (nz + 2) j (+ (nz + 2) nr k on angle k). on = false: no
// dielectric. np > 0: chi depends on the azimuth, given on the np = 2M - 1 collocation angles (E_z, E_r at phi_k, E_phi at
// its own angles: staggered by dphi / 2 with the metal block's phi_derivative = difference), with the transform tables.
struct DielView {
    const double *cz = nullptr, *cr = nullptr, *cp = nullptr;
    int nz = 0, nr = 0;
    bool on = false;
    int np = 0, M = 0;
    const double *cs = nullptr, *sn = nullptr, *ch = nullptr, *sh = nullptr;   // cos, sin(m phi_k) [k M + m]; staggered
    HD long at(int i, int j) const { return long(i + 1) + long(nz + 2) * j; }
    HD long at(int i, int j, int k) const { return at(i, j) + long(nz + 2) * nr * k; }
    HD double z(int i, int j) const { return cz[at(i, j)]; }
    HD double r(int a, int j) const { return cr[at(a, j)]; }
    HD double p(int a, int j) const { return cp[at(a, j)]; }
    HD double z(int i, int j, int k) const { return cz[at(i, j, k)]; }
    HD double r(int a, int j, int k) const { return cr[at(a, j, k)]; }
    HD double p(int a, int j, int k) const { return cp[at(a, j, k)]; }
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
    // The angles (before the first step; FieldsRZ::set_diel does it): M modes, staggered angles of E_phi (the metal block
    // with phi_derivative = difference). Needed only when an eps function depends on the azimuth (then chi is per angle).
    void set_angles(const Domain& d, const RadialGrid& g, int M, bool staggered);
    bool angular() const { return np_ > 0; }
    DielView view() const {
        DielView v;
        v.cz = cz_.data(); v.cr = cr_.data(); v.cp = cp_.data(); v.nz = nz_; v.nr = nr_; v.on = true;
        v.np = np_; v.M = M_;
        if (np_ > 0) { v.cs = cs_.data(); v.sn = sn_.data(); v.ch = ch_.data(); v.sh = sh_.data(); }
        return v;
    }
    // After the vacuum transport into f.ner_ .. f.nbp_ (guards exchanged): solves the lines and corrects the transverse
    // fields of the owned nodes; the caller exchanges them again.
    void correct(Comm& comm, FieldsRZ& f);
    std::string describe(const RadialGrid& g, double dz) const;
    // chi of component c (0 E_z, 1 E_r, 2 E_phi) in the layout of DielView (tests)
    std::vector<double> chi_host(int c) const {
        const Buf<double>& b = c == 0 ? cz_ : (c == 1 ? cr_ : cp_);
        std::vector<double> h(size_t(nz_ + 2) * nr_ * (np_ > 0 ? np_ : 1));
        b.to_host(h.data(), 0, h.size());
        return h;
    }
    int jlo = 0, jhi = 0;              // the radial rows with dielectric (nodes or half nodes)
    std::vector<DielectricConfig> blocks;
    // chi of the local columns i0 .. i1 (-1 .. nz) for the window at c0 (absolute cell of local column 0); public: nvcc refuses
    // device lambdas in private member functions
    void fill(const Domain& d, const RadialGrid& g, int i0, int i1);
private:
    int nz_ = 0, nr_ = 0;
    long c0_ = -(1L << 60);            // the window position of chi
    bool fn_ = false;                  // some block has eps from a deck function (then a window shift moves chi and fills the edges)
    bool phi_ = false;                 // some function depends on the azimuth: chi per angle
    int np_ = 0, M_ = 0;               // angles of chi (0: axisymmetric), modes
    bool stag_ = false;
    Buf<double> cs_, sn_, ch_, sh_;    // the transform tables of the angles
    Buf<double> cz_, cr_, cp_, bk_;    // chi; the blocks (r_in, r_out, z_start, z_end, chi, ramp, code offset, code length)
    Buf<double> tz_, tr_, tp_;         // scratch of the shift
    Buf<Expr::Ins> code_;              // the programs of the eps functions (device)
    Buf<double> cpr_, x_, y_, ex_, ga_, red_;   // Thomas scratch, the three local solutions, y, edge values, reduced systems
    std::vector<double> exh_, gah_;
    // block metal (rz_metal_phi.hpp): the lines are solved on the 2M - 1 collocation angles (Alexander, 2026-11-24): E_r, E_phi
    // and their transported values there (slot k = angle k; the same layout as the fields)
    std::shared_ptr<Field> aer_, aep_, aner_, anep_;
};
// RIP-CC moving window with dielectric blocks: E_z of the entering column from div D of the column before it (rz_dielectric.cpp)
// layer > 0: also the columns of the front absorbing layer (their E_z from div D, sequentially from the layer start), so the damping
// there leaves no charge
void rz_window_entry_dielectric(FieldsRZ& f, const Field& gauss0, int layer = 0);
// chi per angle (DielView::np > 0):
//   out = base + T^-1 [ T(inc) / (1 + chi_z) ] at the owned nodes (the E_z increment of the predictor / corrector divided per
//   angle; T: modes -> angles); out may be base; E_z of m >= 1 on the axis 0; rings: E_z 0 where the rule says so
void rz_diel_ez_angles(FieldsRZ& f, Field& out, const Field& base, const Field& inc);
//   the polarization energy sum E.P / 2 of the columns i0 .. i1 (as FieldsRZ::energy adds it; per angle, Parseval), and of mode m
double rz_diel_pol_energy(const FieldsRZ& f, int i0, int i1);
double rz_diel_pol_energy_mode(const FieldsRZ& f, int m);
//   div D per mode (D = (1 + chi) E on the angles, back to the modes)
void rz_diel_div_d(const FieldsRZ& f, Field& out);
