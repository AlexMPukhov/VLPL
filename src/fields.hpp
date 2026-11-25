// RIP-CC Maxwell solver: X-dispersionless (c tau = h_x), exactly charge conserving. Fields also runs the
// Yee solver when Domain::solver says so (src/yee.cpp).
//
// One step n -> n+1 (J = j^{n+1/2} from the particles, D = h_x = c tau, mu_x / mu_t = averages):
//   1. predictor  Ex* = Ex + D/2 (dy Bz - dz By - Jx),   Bx* = Bx - D/2 (dy Ez - dz Ey)
//   2. transport  T+_y = Ey+Bz -> +x,  T-_y = Ey-Bz -> -x,  T+_z = Ez+By -> -x,  T-_z = Ez-By -> +x,
//                 one cell per step, with the sources at (i+1/2, n+1/2)
//                 Psi_y = dz mu_x Bx* - Jy,  Psi_z = -dy mu_x Bx* - Jz,  Phi_y = -dz mu_x Ex*,  Phi_z = dy mu_x Ex*
//   3. corrector  Ex += D (dy mu_t Bz - dz mu_t By - Jx),  Bx -= D (dy mu_t Ez - dz mu_t Ey)
// The corrector uses the same time-centred transverse fields as the trapezoidal x-transport, so the
// Gauss law  dx Ex + mu_x dy Ey + mu_x dz Ez = rho  and div B are conserved to round-off for any
// predictor. Stable for D^2 (1/hy^2 + 1/hz^2) + D^2 wp^2/4 < 1 (vacuum dispersion as Eq. 57 of
// Pukhov, JCP 418, 109622 (2020)).
#pragma once
#include <functional>
#include <memory>
#include <vector>
#include "comm.hpp"
#include "grid.hpp"
#include "pml.hpp"

class Fields;
class Tfsf;
class SideTfsf;

// Boundary hooks of the field step.
struct BoundaryHooks {
    // Called once per step right before the transport, on every rank, with t = t_n: may set the
    // x-guard layer of the transverse fields (inbound T at an open boundary), e.g. to inject a laser.
    std::vector<std::function<void(Fields&, double)>> inflow;
};

class Fields;
// Yee (src/yee.cpp, docs/specs/yee.md), used by Fields when Domain::solver is Yee.
void yee_step(Fields& f, const Domain& d, Comm& comm);
double yee_max_div_b(const Fields& f, const Domain& d);
void yee_gauss_residual(const Fields& f, const Domain& d, Field& out);
// iRIP, box closure (src/irip.cpp, docs/specs/implicit_rip.md), when Domain::solver is IRip.
void irip_step(Fields& f, const Domain& d, Comm& comm, const BoundaryHooks& hooks, double t);
// O6 (docs/specs/implicit_mover.md): the iRIP step with the current f.jx.. as a trial, F unchanged: et[3] the work field
// of the box closure at the positions of Jx, Jy, Jz, bb[3] = (B^n + B^{n+1}) / 2, guards valid. Box closure, no pml/TFSF.
void irip_trial(Fields& f, const Domain& d, Comm& comm, const BoundaryHooks& hooks, double t, Field* et, Field* bb);
void irip_step_impl(Fields& f, const Domain& d, Comm& comm, const BoundaryHooks& hooks, double t, Field* et, Field* bb);
// max |Ex - X|, |Bx - Y| over the owned cells after an iRIP step: the corrector against the solve (the self-consistency of
// the implicit system; round-off when the solved operator is the step's), local (reduce in the caller)
double irip_residual(const Fields& f, const Domain& d);

class Fields {
public:
    Field ex, ey, ez, bx, by, bz;    // E^n, B^n (guards valid between steps)
    Field jx, jy, jz;                // j^{n+1/2} deposited by the particles
    Field rho;                       // diagnostics only
    Field jfx, jfy, jfz;             // side tube in a background plasma: the cold-fluid current j^{n-1/2} (side.hpp; else empty)
    explicit Fields(const Domain& d);
    void zero_currents();
    // requires J reduced (add_guards); leaves E, B with valid guards; t = t_n for the hooks
    void step(Comm& comm, const BoundaryHooks& hooks = {}, double t = 0);

    // diagnostics (owned cells, local sums / maxima; reduce over ranks in the caller)
    double energy() const;                                     // (1/2) sum (E^2 + B^2) dV
    double max_div_b() const;
    int tfsf_point() const;                                    // local cell whose Gauss point is on the TFSF plane (fields.cpp)
    void gauss_residual(Field& out) const;                     // dx Ex + mu_x dy Ey + mu_x dz Ez - rho
    Pml* pml = nullptr;              // CPML layers in y, z (pml.hpp); null: periodic or plain step
    bool j_ready = false;            // the guards of jx, jy, jz are exchanged already (the overlapped push): step_core skips it
    Tfsf* tfsf = nullptr;            // O1 TFSF laser source (tfsf.hpp); null: none
    SideTfsf* side = nullptr;        // TFSF tube with faces y, z = const (side.hpp, docs/specs/side_injection.md); null: none
    // O3 (docs/specs/x_absorber.md): graded damping layers of `cells` cells at both x ends (grid.boundary_x = absorbing):
    // after every step all fields there are multiplied by exp(-kappa s^2), s the depth in the layer / cells (0: none)
    struct XAbsorber { int cells = 0; double kappa = 0; } xabs;
    std::shared_ptr<struct IripWork> irip;   // scratch of the implicit solver (src/irip.cpp), on first use
    bool o5 = false;                 // numerics.closure = o5: the dispersion-corrected closure of iRIP (decision 72)
    class Medium* medium = nullptr;  // O7: linear-response media in iRIP (medium.hpp, decision 84); null: none
    class Metal3* metal = nullptr;   // embedded perfect conductor (block metal, metal3d.hpp, decision 102); null: none
    class Dielectric3* diel = nullptr;   // dielectric blocks (dielectric3d.hpp, decision 100); null: none
    int dfront = 0;                  // dielectric + moving window: the front absorbing layer (cells at the right x edge, decision 100)
    double dfront_kappa = 0;
    // the solver step without the x layers (public: nvcc refuses device lambdas in private member functions)
    void step_core(Comm& comm, const BoundaryHooks& hooks, double t);
    // the local cells of the Gauss and div B checks (without the CPML and x layers, and one cell more); public for the
    // window bookkeeping of decision 91
    void check_range(int lo[3], int hi[3]) const;
private:
    friend class Tfsf;               // reads the predictor of the incident grid (tfsf.cpp)
    friend void irip_step_impl(Fields& f, const Domain& d, Comm& comm, const BoundaryHooks& hooks, double t, Field* et, Field* bb);

    friend void rip_cc_step_pml(Fields& f, Pml& p, Comm& comm, const BoundaryHooks& hooks, double t);
    const Domain& d_;
    Field exs_, bxs_, ney_, nez_, nby_, nbz_;
};
