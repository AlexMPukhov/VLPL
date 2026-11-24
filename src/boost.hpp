// Lorentz-boosted frame (docs/specs/boosted_frame.md, decision 87): the run is done in a frame moving at beta c along
// +x (RZ: +z) with respect to the lab. The deck stays in the lab frame; apply_boost turns it into the boosted run,
// operator boost_inject feeds the plasma (at rest in the lab) through a plane 3 + filter reach cells behind the right edge, and
// block `lab_diagnostics` writes fields and particles back-transformed to fixed lab times (boost_diag.cpp).
//
// Coordinates (code units, c = 1): lab (x, t) and boosted (x', t') events are related by
//   x = gamma (x' + beta t') + X0,   t = gamma (t' + beta x') + T0,   X0 = T0 = -gamma beta t'_ref,
// with t'_ref the boosted time at which the box's left edge x' = 0 reaches the lab plane x = 0 (the window start, or
// the end of the laser injection without a window): the lasers enter at lab events x <= 0, before any plasma of the
// lab deck, and after the window starts the box covers exactly the lab window (lab light line t - x = window.start).
// The grid along the boost axis is the co-moving lab grid stretched by gamma (1 + beta) (as FBPIC's copropag_length):
// the same cells per laser wavelength; transverse cells are unchanged.
#pragma once
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <cstdio>
#include "grid.hpp"
#include "pipeline.hpp"
#include "portable.hpp"

struct BoostConfig {
    bool on = false;
    double gamma = 1, beta = 0;
    double X0 = 0, T0 = 0;             // lab event of the boosted origin (code units)
    double t_ref = 0;                  // boosted time at which x' = 0 is at the lab plane x = 0
    double dx_lab = 0;                 // the deck's cell along the boost axis (lab)
    double t_end_lab = 0, start_lab = 0;   // the deck's grid.t_end and window.start (lab)
    HD double x_lab(double xp, double tp) const { return gamma * (xp + beta * tp) + X0; }
    HD double t_lab(double xp, double tp) const { return gamma * (tp + beta * xp) + T0; }
    HD double xp_of(double x, double t) const { return gamma * ((x - X0) - beta * (t - T0)); }
    HD double tp_of(double x, double t) const { return gamma * ((t - T0) - beta * (x - X0)); }
    double stretch() const { return gamma * (1 + beta); }   // boosted / lab cell along the boost axis
};

// The particle lattice of a load along the boost axis (Simulation::load_cells, load_rz_cells): the lattice coordinate q
// of a particle (regular in cells of h[0], ppc per cell) gives the stored position q + sx and the lab position ax q + bx
// at which the bounds, the slab and the density function of the deck are evaluated. boost: the momenta of the deck are
// lab momenta, transformed per particle, and the weight carries the density factor gamma (1 - beta v_x) of the boost.
// Default: the plain load (bit-identical).
struct Lattice {
    double ax = 1, bx = 0, sx = 0;
    double qlo = -1e300, qhi = 1e300;  // lattice coordinates loaded: [qlo, qhi)
    bool boost = false;
    double g = 1, b = 0;               // gamma, beta of the boost
};

// Boosted momentum (u along the boost axis first) and the density factor gamma (1 - beta v_x) of a lab momentum.
HD inline double boost_momentum(double g, double b, const double ul[3], double up[3]) {
    const double gl = sqrt(1 + ul[0] * ul[0] + ul[1] * ul[1] + ul[2] * ul[2]);
    up[0] = g * (ul[0] - b * gl); up[1] = ul[1]; up[2] = ul[2];
    return g * (1 - b * ul[0] / gl);
}

// Lab-frame diagnostics of a boosted run (block lab_diagnostics): snapshots at the lab times t_first + k t_every,
// k < count; fields (E, B in the lab frame) on the planes x' where the boosted step meets the lab time, and the particles
// of the listed species crossing that lab time, with their beam moments in the lab frame.
struct LabDiagConfig {
    std::vector<double> times;         // lab times (code units)
    bool fields = true;
    std::vector<std::string> species;  // particles at the lab times (none: no particle snapshots)
    double gamma_min = 1;              // beam moments of the particles with lab gamma >= gamma_min
    bool particles = true;             // write the particles (else only lab_beams.txt)
    double select_uz = -1e300;         // record only crossings with the lab u along the boost axis >= select_uz
    double select_r = 1e300;           // and |transverse coordinates| < select_r (code units); FBPIC select uz, x, y
};

struct Config;
struct SpeciesConfig;
class Deck;
// boost.gamma > 1: the deck's lab values -> the boosted run (grid along x, t_end, window start, lasers, species);
// throws for what the boosted frame does not support yet. Idempotent.
void apply_boost(Config& c);
void boost_from_deck(const Deck& d, Config& c, double k0);   // blocks boost and lab_diagnostics
void boost_summary(FILE* f, const Config& c);
struct LaserConfig;
// The laser beam at its lab entrance events (lab x in [X0, 0]; the antenna: x = 0): its largest width w at the largest
// focus distance d, the transverse half box (RZ: the wall); true (the summary warns) if the box cuts it there although it
// holds the beam at its focus (2.5 w > half >= 2.5 w0). A beam wider than the box at its focus (a plane wave over a periodic y) is no warning.
bool boost_beam_cut(const Config& c, const LaserConfig& L, double& w, double& d, double& half);
// The streaming plasma of a boosted run and the numerical Cherenkov instability (bench/nci/boost_stability, decision 104):
// empty when the numerics are those measured stable, else the reason (the summary warns with it).
std::string boost_nci_risk(const Config& c);
double boost_net_charge(const Config& c, double* where = nullptr, const SpeciesConfig* key = nullptr);   // relative net charge of
                                                 // the injected plasma (0: neutral); key: only its lattice group (decision 114)
bool boost_same_lattice(const SpeciesConfig& s, const SpeciesConfig& t);   // the same ppc, bounds and slab: one lattice group
void boost_check_neutral(const Config& c);   // throws unless the boosted injection is neutral at every lattice point (decision 108)
// Boosted beams (boost_kind 2), checks of the loading (decision 118; the loading itself unchanged): the lattice planes across
// the beam along the boost axis, (xmax - xmin) a ppc_x / dx' with a = 1 / (gamma (1 - beta v)) (a beam moving against the boost
// is gamma^2 (1 + beta)^2 shorter in boosted cells); apply_boost throws below 1, the summary warns below 20 (boost_beam_warning).
double boost_beam_planes(const Config& c, const SpeciesConfig& sc);
int boost_beam_ppc_for(const Config& c, const SpeciesConfig& sc, double planes);   // the ppc_x for that many planes
std::string boost_beam_warning(const Config& c);                // empty, or the beams with fewer than 20 planes
// The charge (sum of the weights) the deck asks for: exact along the boost axis, transversely on the loader's lattice (Cart3D,
// 2D; RZ: exact in r). boost_beam_charge_check (after the load of species i, collective): prints the loaded charge against it
// and warns above 1 %; returns the relative difference.
double boost_beam_deck_charge(const Config& c, const SpeciesConfig& sc);
class Simulation;
double boost_beam_charge_check(Simulation& s, size_t i);
// The particle lattice of species sc loaded at the boosted time t (plasma: comoving with it; beam: ballistic map).
Lattice boost_lattice(const Config& c, const SpeciesConfig& sc, const Domain& d, double t);
// The lattice coordinate q = x' + beta t' up to which the plasma has entered at step n (the injection plane).
double boost_inject_q(const Config& c, const Domain& d, long n);
std::unique_ptr<Operator> make_boost_inject_operator();      // the plasma through the injection plane (stage Fields)
std::unique_ptr<Operator> make_lab_field_operator();         // lab_diagnostics fields (stage Diagnostics)
std::unique_ptr<Operator> make_lab_particle_operator();      // lab_diagnostics particles (stage Fields, before the window)
// laser.injection = antenna (boost_antenna.cpp, decision 89): the lab-fixed current sheet with its charge. The state (null: no
// antenna laser), the operator (stage Push, after the deposit) that adds its current, and its charge at step s.istep (Gauss law).
class Simulation;
class Field;
struct BoostAntenna;
std::shared_ptr<BoostAntenna> make_boost_antenna(const Simulation& s);
std::unique_ptr<Operator> make_boost_antenna_operator();
void boost_antenna_add_rho(Simulation& s, Field& rho);
double boost_antenna_start(const Config& c);                 // x' of the antenna when it starts to emit (lab t = 0)
