// O7: linear-response media without particles for iRIP (numerics.solver = irip, closure = box; docs/specs/linear_medium.md,
// decision 84). Cold Drude plasma (resonance 0), collisional plasma (nu > 0) and Lorentz dielectrics, per current component at
// the position of that current (Cart3D: Jx (i, j, k), Jy (i+1/2, j+1/2, k), Jz (i+1/2, j, k+1/2); RZ: Jz (i, r_j),
// Jr (i+1/2, r_{j+1/2}), Jphi (i+1/2, r_j), every mode with the same axisymmetric coefficients):
//   (P^{n+1} - P^n) / dt = mu_t J,    (J^{n+1} - J^n) / dt = wp2 Et - w02 mu_t P - nu mu_t J,
// the field step gets the current u = mu_t J. Et is the work field of the box closure (irip_trial, decision 81), so
//   [W + K]^{n+1} - [W + K]^n = -dt dV sum nu u^2 / wp2,   K = dV sum (J^2 + w02 P^2) / (2 wp2)
// exactly (to the tolerance of the linear solve), for any omega_p dt. The medium's own source in the transverse transport
// (-dt/2 u in Et_y, Et_z) is eliminated locally:
//   u = (2 J + dt wp2 Et0 - dt w02 P) / (2 + dt nu + dt^2 w02 / 2 + kap dt^2 wp2 / 2),   kap = 0 for Jx (Jz in RZ), 1 else,
// Et0 = Et without the medium current. The unknowns of the step stay Ex^{n+1}, Bx^{n+1} (RZ: Ez, Bz per mode); the system is
// linear with variable coefficients and solved by restarted GMRES, right-preconditioned by the vacuum box solver after scaling
// the Ex equation by its local diagonal 1 + dt^2 wp2_x / (2 (2 + dt nu + dt^2 w02 / 2)). The medium charge is -div P, so the
// Gauss law with particles plus the medium holds to round-off (Simulation::compute_rho adds it).
// model = bound (decision 85, bloch.hpp): the atoms of a gas as V-type Maxwell-Bloch atoms on the positions of one current
// component (the dipole axis); their current u_b(Et) is nonlinear, linearized at Et* (u_b = u* + s (Et - Et*)) inside the linear
// solve, Et* updated until it is the Et of the solution (Newton). Energy: the field's work goes into ef n_a sum w_j' p_j.
#pragma once
#include <functional>
#include <string>
#include <vector>
#include "comm.hpp"
#include "expr.hpp"
#include "grid.hpp"
#include "bloch.hpp"

// deck block `medium [<name>]`
struct MediumConfig {
    std::string name;
    double density = 0;          // wp^2 / omega0^2 = n / n_c of the (bound or free) electrons
    double nu = 0;               // collision (damping) frequency / omega0
    double w0 = 0;               // resonance frequency / omega0 (0: Drude)
    Expr profile;                // factor f(x, y, z) of the density (RZ: f(z, r)), absolute code units; empty: 1
    int supersample = 4;         // sub-samples per axis of the cell around each current position (the volume fraction)
    // model = bound (bloch.hpp, decision 85): a V-type Maxwell-Bloch atom, density = n_a / n_c of the atoms
    bool bound = false;
    int lines = 0;               // excited states (1 or 2)
    double bw[2] = {0, 0};       // transition frequencies / omega0
    double bf[2] = {0, 0};       // oscillator strengths
    double bd[2] = {0, 0};       // effective dipoles d_j / (e a0), d^2 = f hbar e^2 / (2 m omega_j)
    int axis = 2;                // the dipole axis (0 x, 1 y, 2 z)
    double a0k0 = 0;             // Bohr radius times k0
    double bT1 = 0, bT2 = 0;     // relaxation: lifetime T1 of the excited states, pure dephasing time T2 (1 / omega0; 0: none)
    bool ionize = false;         // ionization = adk: ADK per level (bloch.hpp), depopulation; pairs into the species below
    double ip[3] = {0, 0, 0};    // ionization energies of the levels 0, 1, 2 (eV)
    std::string electrons, ions; // species of the freed electrons and of the ions (both or none)
    int ion_ppc = 8;             // pairs per atom position if all of its atoms are ionized (the weight quantum n_a dV / ion_ppc)
    int nphi = 0;                // RZ: azimuthal samples of the atoms (0: 4 times the number of modes; at least 2 m_max + 1)
};

class Deck;
// the block b (`medium` or `medium <name>`): model = drude | lorentz, density (n / n_c) or, lorentz, permittivity (the real eps
// at omega0 without damping: wp2 = (eps - 1)(resonance^2 - 1)), collision_frequency, resonance (omega0 units), profile (an
// expression of x, y, z; RZ z, r), supersample
MediumConfig medium_block(const Deck& d, const std::string& b, double k0, bool rz);

struct MediumStats {
    long solves = 0, iterations = 0, max_iterations = 0, unconverged = 0;
    long newton = 0, newton_steps = 0, newton_max = 0, newton_unconverged = 0;   // model = bound: outer iterations
    double last_residual = 0;
};

class Medium {
public:
    // coefficients and state per current component (Cart3D x, y, z; RZ z, r, phi with the mode slots of FieldsRZ)
    Field wp2[3], w02[3], nu[3], J[3], P[3];
    Field u[3], jt[3];           // scratch: the medium current of the iterate, the total current (particles + medium)
    Field sx, sy, gx, gy;        // scratch of the solve: the iterate as fields, G(iterate)
    double dissipated = 0;       // energy dissipated by nu since the start (summed over the ranks)
    double tol = 1e-10;          // relative residual of the linear solve (relative to max(|G(0)|, 1e-4 fnorm))
    double fnorm = 0;            // the norm of the whole field state of the step (set by the caller): a floor of the residual scale
    int maxit = 200, restart = 30;
    bool rz = false;
    std::vector<double> weights; // RZ: the volume of each current position (3 nr: Jz, Jr, Jphi), for K and the dissipation
    MediumStats stats;
    // model = bound (bloch.hpp): the atoms on the positions of the current component bax
    bool bound = false;
    int bax = 2;
    BlochC bc;
    Field bs[BS_N], bn;          // the density matrix per node, n_a
    Field bE, bEs, bu0, bsg;     // Et of the iterate, the linearization point, u(bEs), du/dEt there
    Field bU, bP;                // the atoms' current of the iterate, their polarization n_a P (for rho)
    Field bacc;                  // ionization: the ionized density not yet emitted as particles (n / n_c)
    // RZ (medium_rz_bound.cpp): the atoms at the Jr positions (i+1/2, rh_j) on nphi angles, raw arrays [quantity][angle][j][i]
    // (owned cells): the density matrix, Et*, Et, u*, s, u_b, the ionization accumulator; per mode (slots): the lagged atom current
    // (its own source in Et), Et of the iterate after the linear media, the deposited atom current, the atoms' polarization
    int nphi = 0;
    std::vector<Buf<double>> ra;
    Field bLr, bLp, bEr, bEp, bUr, bUp, bPr, bPp;
    double ionized = 0;          // ionization: sum over the ranks of the ionized atoms (n / n_c times volume) since the start
    double emitted = 0;          // ionization: the weight emitted as electron-ion pairs on this rank (n / n_c times volume)
    double btol = 1e-10;         // the outer iteration: max |Et - Et_lin| / max |Et|
    int bmaxit = 30;
    double bt = -1e300;          // the time of the last outer iteration (a new step starts from the last Et)
    // warm start within one step (O6 calls the medium solve once per mover iteration): the last solution and its time
    double last_t = -1e300;
    bool have_last = false;
    std::vector<Buf<double>> V; Buf<double> x, w, z, adiag;   // GMRES (flat 2N vectors of the owned cells)
    void resize(const Domain& d);
    double energy(const Domain& d, const std::vector<double>* weights = nullptr) const;   // local K (RZ: weights per (j, comp))
};

// Fills wp2, w02, nu from the blocks (sub-cell averages of the profiles; overlapping blocks: wp2 summed, w02 and nu weighted
// by wp2). geom: 0 Cart3D, 1 cartesian2d, 2 RZ (positions from the radial grid r, rh). Returns the largest wp2 (local).
double medium_build(Medium& M, const Domain& d, const std::vector<MediumConfig>& cfg, int geom, const double* r = nullptr,
                    const double* rh = nullptr, int i0 = 0);   // i0 > 0: only the cells i >= i0 (a moving window; zeroed before)

// The linear operator of the step for the solve: apply(X, Y) sets the box averages from the iterate, computes the medium current
// and the total current and runs the step's pass without committing: G(X, Y) into gx, gy (owned cells). precond(f, kind): the
// vacuum solve of kind 0 (Ex) or 1 (Bx) in place on the owned cells.
struct MediumOps {
    std::function<void(Field& X, Field& Y, Field& gx, Field& gy)> apply;
    std::function<void(Field& f, int kind)> precond;
};
// Solves X = G(X, Y), Y = G_B(X, Y) for the iterate (sx, sy on return, owned cells; x0: X0, Y0 used without a warm start).
void medium_solve(Medium& M, const Domain& d, Comm& comm, const MediumOps& ops, const Field& X0, const Field& Y0, double t);

// Cart3D: u and jt = Jext + u from the box averages Exs, Bxs and the fields at t_n (owned cells; jt not exchanged).
void medium_current_cart(Medium& M, const Domain& d, const FV Ey, const FV Ez, const FV By, const FV Bz, const FV Jx,
                         const FV Jy, const FV Jz, const FV Exs, const FV Bxs);
// Commit after the step: J <- 2 u - J, P <- P + dt u, the dissipated energy (weights: RZ volumes, else dV).
void medium_commit(Medium& M, const Domain& d, Comm& comm, const std::vector<double>* weights = nullptr);
// The medium charge -div P added to rho (Cart3D, the RIP-CC Gauss stencil at (i+1/2, j, k)).
void medium_add_rho_cart(const Medium& M, const Domain& d, Field& rho);

// model = bound: the outer iteration around medium_solve. solve() runs medium_solve, prepare() sets bE, bU from the solution
// (medium_current_cart). Newton with the local tangent du/dEt (central differences of the node step).
void medium_bound_solve(Medium& M, const Domain& d, Comm& comm, double t, const std::function<void()>& solve,
                        const std::function<void()>& prepare);
// model = bound: the atoms to rho^{n+1} with bE, bP += dt bU (after the final pass)
void medium_bound_commit(Medium& M, const Domain& d, Comm& comm);

// RZ (src/medium_rz.cpp): u and jt per mode from the box means Exs = Ez*, Bxs = Bz* and the fields at t_n (all modes, owned cells)
struct RadialView;
void medium_current_rz(Medium& M, const Domain& d, const RadialView g, int modes, const FV Er, const FV Ep, const FV Br,
                       const FV Bp, const FV Jz, const FV Jr, const FV Jp, const FV Exs, const FV Bxs);
// RZ: the medium charge -div P per mode added to rho (the Gauss stencil of FieldsRZ::div_e at the J positions)
void medium_add_rho_rz(const Medium& M, const Domain& d, const RadialView g, int modes, Field& rho);
// RZ, model = bound (medium_rz_bound.cpp): set up the atoms (n_phi angles, n_a at the Jr positions in bn), the atom current of the
// iterate deposited per mode into jt (after medium_current_rz), the outer iteration, the commit, the energy (local), the charge
// -div bP per mode, the pairs of the ionization (host: positions z, r cos, r sin; weight acc 2 pi B_j dz / nphi)
enum { RA_ES = BS_N, RA_E, RA_U0, RA_SG, RA_U, RA_ACC, RA_N };
void medium_bound_rz_setup(Medium& M, const Domain& d, const RadialView g, int modes, int nphi);
void medium_bound_rz_current(Medium& M, const Domain& d, const RadialView g, int modes);
void medium_bound_rz_solve(Medium& M, const Domain& d, Comm& comm, double t, const std::function<void()>& solve,
                           const std::function<void()>& prepare);
void medium_bound_rz_commit(Medium& M, const Domain& d, Comm& comm, const RadialView g);
double medium_bound_rz_energy(const Medium& M, const Domain& d, const RadialView g);
void medium_bound_rz_add_rho(const Medium& M, const Domain& d, const RadialView g, int modes, Field& rho);
