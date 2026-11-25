// Lossless, non-dispersive dielectrics eps(x, y, z) in the explicit RIP-CC step of Cart3D and 2D (blocks `dielectric [<name>]`,
// docs/specs/dielectric3d.md, decision 100: the scheme of the RZ blocks, Alexander 2026-11-24: "dielectrics and metals have to
// be implemented in general"). eps is a number (everywhere) or a deck expression of x, y, z (lab frame, fixed; it moves through
// a moving window), blocks add their chi = eps - 1.
// The medium is the polarization current J_P = d_t P, P = chi E, time-centred (Crank-Nicolson), at the positions of the free
// current of each component:
//   E_x (i, j, k):              (1 + chi_x) (E_x^{n+1} - E_x^n) = dt (curl B - J_x): local (predictor: (1 + chi_x) too);
//   J_y (i+1/2, j+1/2, k):      y_a = dt J_P = chi_a mu_x(E_y^{n+1} - E_y^n) at a = i + 1/2,
//   J_z (i+1/2, j, k+1/2):      the same with E_z.
// Through the characteristics (T+_y = E_y + B_z to +x, T-_z = E_z - B_y to +x) node i gets -(y_{i-1/2} + y_{i+1/2}) / 2 in E,
// -(y_{i-1/2} - y_{i+1/2}) / 2 in B_z (E_y lines) and +(y_{i-1/2} - y_{i+1/2}) / 2 in B_y (E_z lines); every (j, k, component)
// is then the tridiagonal line along x
//   y_a (1 + chi_a (m_a + m_{a+1}) / 4) + chi_a / 4 (m_a y_{a-1} + m_{a+1} y_{a+1}) = chi_a / 2 (m_a dEv_a + m_{a+1} dEv_{a+1})
// (m = 0 where the metal block holds E at 0 and at the guard nodes beyond an open x end), solved directly with the ranks along
// x (partition solve, one allgather per step in the row of ranks along x; cyclic for periodic x). div(E + P) = rho holds to
// round-off, the scheme is unconditionally stable and lossless (decision 100). chi per component over its control volume:
// 4 x 4 x 4 sub-samples, harmonic along the component's own direction, arithmetic across (Alexander, 2026-11-24).
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "expr.hpp"
#include "grid.hpp"
#include "portable.hpp"

struct Dielectric3Config {
    std::string name;
    double eps = 1;                    // constant eps (fn empty)
    Expr fn;                           // eps(x, y, z) (code units, absolute lab positions)
};

// chi at E_x (cx), at the J_y points (cy) and the J_z points (cz) in the layout of the fields (guards valid)
struct Diel3View {
    FV cx, cy, cz;
    bool on = false;
};

class Comm;
class Fields;
class Metal3;
class Dielectric3 {
public:
    Dielectric3(const std::vector<Dielectric3Config>& blocks, const Domain& d);
    Diel3View view() const { return {cx_.view(), cy_.view(), cz_.view(), true}; }
    // After the vacuum transport into the transported fields of f (guards exchanged): solves the lines, corrects E_y, E_z,
    // B_y, B_z of the owned nodes; the caller exchanges them again.
    void correct(Comm& comm, const Domain& d, const Fields& f, Field& ney, Field& nez, Field& nby, Field& nbz);
    // a moving-window shift (after ++wx): chi moves with the fields, the entering columns and the left guards are filled
    void shift(const Domain& d, Field& tmp);
    std::string describe() const;
    std::vector<Dielectric3Config> blocks;
    double max_chi = 0;                // over the whole box at the start (describe)
    // local columns i0 .. i1 - 1 (guards in y, z included); public: nvcc refuses device lambdas in private member functions
    void fill(const Domain& d, int i0, int i1);
private:
    Field cx_, cy_, cz_;
    Buf<double> bk_;                   // per block: chi (constant), code offset, code length
    Buf<Expr::Ins> code_;
    Buf<double> cpr_, x_, y_, ex_, ga_, red_;
    std::vector<double> exh_, gah_;
};

// Moving window with dielectric (RIP-CC, Cart3D / 2D): E_x of the entering column and of the front-layer columns from div D of
// the column before (sequentially from the layer start), so the damping and the entry leave no charge (as RZ, decision 100)
void dielectric3_window_entry(Fields& f, const Domain& d, const Field& gauss0, int layer, const Dielectric3& diel);
// the front absorbing layer (the last `cells` columns at the right x edge, kappa): all fields * exp(-kappa s^2)
void dielectric3_absorb_front(Fields& f, const Domain& d, int cells, double kappa);
