// O1 (docs/specs/tfsf.md): total-field / scattered-field laser source. A plane between the cells i_s - 1 and i_s:
// to its right the total field, to its left the scattered (reflected) field. The incident field is the discrete
// solution of an auxiliary vacuum grid (the same domain and solver, no particles): the beam is set up there once
// as initial fields left of the plane, at any angle in the x-y plane, and advanced every step. The corrections at
// the plane read that grid, so the incident field satisfies the discrete equations exactly (no leakage beyond
// round-off) for any beam shape and both RIP-CC and iRIP.
#pragma once
#include <memory>
#include <vector>
#include "comm.hpp"
#include "fields.hpp"
#include "laser.hpp"
#include "pml.hpp"
#include "tfsf_modes.hpp"

class Tfsf {
public:
    // lasers with tfsf_x >= 0 (the plane x_s = tfsf_x, rotated by L.angle about z, the peak at (x_s, yc, zc) at t = tc)
    // pml: the CPML layers of the main grid (null: none); the incident grid gets the same (bmin: Pml). With
    // laser.tfsf_incident = modes the incident field is a sum of exact discrete plane waves instead (tfsf_modes.hpp; periodic
    // y, z, no CPML); t_span: the run time plus the box diagonal (no periodic image of the pulse within it).
    Tfsf(const Domain& d, Comm& comm, const std::vector<LaserConfig>& lasers, const PmlConfig* pml = nullptr, double bmin = 0,
         double t_span = 0);
    bool modes() const { return bool(modes_); }
    void start(Fields& f, Comm& comm);         // t = 0: modes add the incident field right of the plane to f
    int plane() const { return is_; }           // global index i_s
    // At the start of the field step t_n -> t_n+1 (t = t_n): keeps the incident field of t_n and advances the grid to t_n+1
    // (modes: evaluates the plane waves of t_n in the two cells next to the plane).
    void begin(Comm& comm, double t);
    // Adds the incident transport components across the plane to the transported fields (cells i_s - 1, i_s),
    // with the longitudinal averages Ex*, Bx* of the incident field: kind 0 the RIP-CC predictor of t_n, kind 2 the mean
    // of t_n and t_n+1 (iRIP, both passes: in the first, the incident Ex, Bx of t_n+1 across the plane are known and go
    // into the right-hand side, so the implicit solve of the main grid needs no change; kind 1 half of t_n, unused).
    // commit = false: the CPML memories of the corrections are read, not updated (the first iRIP pass, as irip_pml_pass)
    void correct(Comm& comm, Field& ney, Field& nez, Field& nby, Field& nbz, int kind, bool commit = true);
    double max_incident() const { return amax_; }  // largest initial |E| of the incident grid (diagnostics)
    double overlap() const { return over_; }       // the initial incident field from the plane on, relative (should be 0)
    // O3: the incident grid with the x layers of the main grid; warns if the pulse starts inside the left one
    void set_x_absorber(Comm& comm, const Fields::XAbsorber& x);
    const Fields& incident() const { return *inc_; }  // the incident grid (at t_n after a full step; tests; not with modes)
    // O5 (numerics.closure = o5, decision 75): the incident grid runs the closure too; its Ex*, Bx* of the corrections get it
    // (computed in begin), and o5_fix adds to the main grid's closed Ex*, Bx* the incident value of the far cell where the
    // Cx stencil crosses the plane: + C_perp X_inc(s) / 3 in the cell s - 1, - C_perp X_inc(s - 1) / 3 in s (both passes;
    // the guards exchanged). Not with the plane waves (modes) so far.
    void set_o5(Comm& comm);
    void o5_fix(Comm& comm, Field& exs, Field& bxs);
    const TfsfModes* plane_waves() const { return modes_.get(); }
private:
    const Domain& d_;
    std::unique_ptr<Fields> inc_;                 // the incident grid at t_n+1 after begin (null with modes)
    std::unique_ptr<TfsfModes> modes_;            // the exact plane waves (laser.tfsf_incident = modes)
    Field ex0_, ey0_, ez0_, bx0_, by0_, bz0_;     // the incident field at t_n
    Field xs_, bs_;                               // Ex*, Bx* of the incident field for the corrections
    bool o5_ = false;
    Field pe_, pb_;                               // O5: C_perp of the incident means
    std::unique_ptr<Pml> pml_;                    // the layers of the incident grid
    Buf<double> psi_;                             // CPML memories of the corrections (8 per transverse point)
    int is_ = 0;
    double amax_ = 0, over_ = 0;
};
