// Simulation setup from the deck and the main loop.
#pragma once
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include "comm.hpp"
#include "deck.hpp"
#include "expr.hpp"
#include "fields.hpp"
#include "beam_diag.hpp"
#include "laser.hpp"
#include "laser_file.hpp"
#include "medium.hpp"
#include "openpmd.hpp"
#include "particles.hpp"
#include "pipeline.hpp"
#include "rz.hpp"
#include "tiles.hpp"
#include "window.hpp"
#include "side.hpp"
#include "boost.hpp"
#include "slab.hpp"
#include "tfsf.hpp"
#include "xflux.hpp"
#include "hfocus.hpp"
#include "ionization.hpp"
#include "qed.hpp"
#include "collisions.hpp"
#include "metal3d.hpp"
#include "dielectric3d.hpp"
#include "external.hpp"
#include "beam_import.hpp"
#include "wake.hpp"

struct SpeciesConfig {
    std::string name;
    double q = -1, m = 1, density = 0;           // density in n_c
    int ppc[3] = {1, 1, 1};                      // RZ: z, r, theta
    double u0[3] = {0, 0, 0}, uth[3] = {0, 0, 0};   // RZ: z, x, y (the particle axes)
    double rmin[3], rmax[3];                     // code units (RZ: [0] z, [1] r)
    // optional tilted slab: 0 <= (x - x0) cos(angle) + (y - y0) sin(angle) < thickness, with an exponential pre-plasma
    // in front (scale length slab_ramp, cut at slab_ramp_extent) and behind (slab_rear_ramp, slab_rear_extent); with a
    // ramp or slab_exact the particles get the exact mean of the profile over their sub-cell (docs/specs/solid_target.md)
    double slab_x0 = 0, slab_y0 = 0, slab_thickness = 0, slab_angle = 0;
    double slab_ramp = 0, slab_ramp_extent = 0, slab_rear_ramp = 0, slab_rear_extent = 0;
    bool slab_exact = false;
    SlabProfile slab() const;                    // the plain values of the loaders (slab.hpp)
    Expr density_fn;                             // optional factor f(x, y, z) of the density (absolute x); RZ: f(z, r)
    bool self_field = false;                     // steady fields of the species moving with u0 when loaded
    BeamFileConfig file;                         // species.file: the particles from an openPMD file (beam_import.hpp)
    double t_start = 0;                          // load time (code units); > 0: loaded then, in the window
    double rigid_until = 0;                      // before this time (code units) the momenta stay frozen: ballistic motion,
                                                 // current deposited (fields relax to the solver's steady state; decision 98)
    bool mobile = true;                          // false: never pushed (a static background, e.g. ions; no current)
    // field ionization (ionization.hpp, decision 86): q = 1 (the unit) and the charge state per particle
    IonizationConfig ion;
    SpinConfig spin;                             // spin = sx sy sz: attributes s0 s1 s2, T-BMT (qed.hpp, decision 93)
    bool photon = false;                         // kind = photon: q = 0, empty at the start, filled by qed.photons
    double q2() const { return ion.on || ion.states ? double(ion.zmax) * ion.zmax : q * q; }   // q^2 for the omega_p checks (the highest state)
    double wp2_electrons() const { return ion.on || ion.states ? density * (ion.zmax - ion.z0) : 0.0; }   // the electrons it can free (n_c)
    // boosted frame (boost.hpp): 1 a plasma at rest in the lab (fed through the injection plane), 2 a beam (loaded once,
    // ballistic map from the lab time beam_t0, lab velocity beam_v); u0_lab the deck's momentum (u0 is then the boosted one)
    int boost_kind = 0;
    double u0_lab[3] = {0, 0, 0};
    double beam_v = 0, beam_t0 = 0;
};

struct Config {
    double lambda0 = 0.8e-6;                     // SI, sets the unit of length 1/k0
    // grid.geometry = rz (docs/specs/rz_modes.md): N = {nz, nr, 2 modes - 1}, h = {dz, dr, 1}, z the
    // transport axis (as x in Cart3D), a conducting wall at r = nr dr
    bool rz = false;
    // grid.geometry = cartesian2d (docs/specs/cart2d.md): Cart3D with one z cell of thickness h[2] = grid.dz (default 1 um)
    // and the z stride 0 (Domain::flat); totals (energies, charges, spectra) are per slab of thickness dz
    bool flat = false;
    int modes = 1;
    int absorber_mode = 2;                       // grid.absorber_particles = remove (1) | conserve (2, default) | keep (0): particles
                                                 // entering the y/z PML (Cart3D) or the radial absorber (RZ) (absorber_particles.cpp)
    int rz_abs_cells = 0;                        // grid.boundary_r = absorbing: layer cells (0: conducting wall)
    double rz_abs_R0 = 1e-4;                     // its target round-trip attenuation
    int x_abs_cells = 0;                         // grid.boundary_x = absorbing: damping layers at both x ends (O3, 0: none)
    double x_abs_R0 = 1e-4;                      // their target round-trip attenuation (docs/specs/x_absorber.md)
    // stretched radial grid (docs/specs/rz_modes.md step 6): dr up to r_uniform, then cells growing by
    // stretch per cell up to dr_max (stretch = 1: uniform)
    double rz_r_uniform = 0, rz_stretch = 1, rz_dr_max = 0;
    RadialGrid radial_grid() const;
    int N[3] = {32, 32, 32};
    double h[3] = {0.1, 0.4, 0.4};               // code units
    bool periodic[3] = {true, true, true};
    int mpi[3] = {0, 0, 0};
    double t_end = 10;                           // code units
    int order = 1;
    int filter_passes = 0;                       // current filter passes (numerics.filter_passes; RZ decision 54, Cart3D 62)
    bool filter_comp = false;                    // numerics.filter_compensate (binomial): T^N ((N + 1) - N T) along x (z), bench/nci/mechanism
    bool filter_xonly = false;                   // numerics.filter_axes = x (binomial): the passes along x (z in RZ) only
    int filter_flat = 0;                         // numerics.filter_flat = L (binomial): maximally flat factor, decision 109
    int filter_reach() const {
        return filter_passes > 0 ? filter_passes + (filter_comp ? 1 : 0) + (filter_flat > 1 ? filter_flat - 1 : 0) : 0;
    }   // cells of the stencil
    bool o5 = false;                             // numerics.closure = o5: iRIP's dispersion-corrected closure (decision 72)
    bool rz_gather_matched = false;              // numerics.rz_gather = matched (decision 80: the gather in the radial map of the deposit; default plain, decision 106)
    bool rz_gather_energy = false;               // numerics.rz_gather = energy (matched; E_r energy-consistent, decision 80)
    bool rz_map_all = false;                     // numerics.rz_axis_map = all (m0, the default: the density-conserving radial map for m = 0 only, decision 106)
    bool rz_deposit_chord = false;               // numerics.rz_deposit = chord (phase: deposit_seg, the default; chord: the splits of seg_currents)
    double rz_shape_width = 0;                   // numerics.rz_transverse_shape = gaussian: width (rz_shape_width, dr at the axis); 0 = none (default)
    bool rz_axis_regular = true;                 // numerics.rz_axis_current = regular (default) | raw: the m = 1 first-cell current (decision 112)
    double rz_axis_cut = 0;                      // numerics.rz_axis_cut: mode m >= 2 zero inside r < m_eff dr / kappa (0: off; RadialGrid::set_rod)
    bool filter_local = false;                   // numerics.filter = local (Cart3D decision 64, RZ 54) instead of projection (69, 70)
    bool beam_steady = true;                     // numerics.beam_init = steady (RIP-CC: the discrete steady state of the self_field
                                                 // beam, decision 99) | gauss (the Gauss-law field, decision 24; iRIP and Yee always)
    bool filter_binomial = true;                 // numerics.filter = binomial (the default): every component smoothed (decision 91); filter_local wins
    Solver solver = Solver::RipCC;               // numerics.solver (docs/specs/yee.md)
    double cfl = 0.98;                           // Yee: c dt / (1/hx^2 + 1/hy^2 + 1/hz^2)^{-1/2}
    double dt() const;                           // RIP-CC: h_x; Yee: from cfl
    Pusher pusher = Pusher::HigueraCary;
    // numerics.mover = implicit (O6, docs/specs/implicit_mover.md): Crank-Nicolson particles with the box closure, W + K
    // conserved to mover_tol; irip (box closure), shape 2 or 3, Cart3D / cartesian2d; no omega_p dt limit
    bool mover_implicit = false;
    double mover_tol = 1e-6;                     // relative residual of the current (preconditioned)
    int mover_maxit = 100;
    int mover_anderson = 5;                      // Anderson depth (0: preconditioned Picard)
    unsigned long seed = 1;
    int tile[3] = {8, 8, 8};                     // cells per tile
    long sort_every = -1;                        // tile sort every n steps, 0: never, -1: auto (decision 20)
    enum class Deposit { Auto, Global, Tiled, Box } deposit = Deposit::Auto;   // docs/specs/tile_deposit.md
    std::vector<SpeciesConfig> species;
    // O7 (docs/specs/linear_medium.md, decision 84): blocks `medium [<name>]`, linear media without particles (irip, box)
    std::vector<MediumConfig> media;
    double medium_tol = 1e-10;                   // numerics.medium_tol: relative residual of the GMRES solve
    int medium_maxit = 200;                      // numerics.medium_maxit: GMRES iterations per step at most
    double medium_bound_tol = 1e-8;              // numerics.medium_bound_tol: model = bound, the outer iteration (max |dEt| / max |Et|)
    int medium_bound_maxit = 30;                 // numerics.medium_bound_maxit
    int medium_restart = 30;                     // numerics.medium_restart: Krylov vectors (memory: restart + 2 vectors of 2 N)
    void check_media() const;
    LaserConfig laser;                           // block `laser`
    std::vector<LaserConfig> extra_lasers;       // blocks `laser <name>`, summed with `laser`
    std::vector<LaserConfig> all_lasers() const {
        std::vector<LaserConfig> v{laser};
        v.insert(v.end(), extra_lasers.begin(), extra_lasers.end());
        return v;
    }
    std::vector<LaserFileConfig> file_lasers;    // laser blocks with `file` (docs/specs/laser_file.md)
    WindowConfig window;                         // moving window (docs/specs/moving_window.md)
    PmlConfig pml;                               // CPML layers in y, z (docs/specs/pml.md)
    SideConfig side;                             // block `side`: the TFSF tube (docs/specs/side_injection.md)
    std::string dir = "out";
    long every = 10;
    long slice_every = 0;                        // 0: no field slices
    double slice_z = -1;                         // z of the x-y slice (code units), < 0: middle
    std::vector<std::string> slice_fields;       // Ex .. Bz, rho, n_<species>
    std::vector<BeamDiagConfig> diags;           // blocks `diagnostic <name>` (docs/specs/beam_diagnostics.md)
    OpenPmdConfig openpmd;                       // output.openpmd_* (docs/specs/openpmd.md)
    XFluxConfig xflux;                           // output.xflux, xflux_every (docs/specs/xflux.md)
    std::vector<HFocusConfig> hfocus;            // blocks harmonic_focus [<name>] (docs/specs/harmonic_focus.md)
    CheckpointConfig checkpoint;                 // block `checkpoint`
    RingsConfig rings;                           // block `rings`: embedded metal rings in RZ (docs/specs/rz_metal.md)
    MetalPhiConfig metal_rz;                      // block `metal`: conductors depending on phi in RZ (docs/specs/rz_metal_phi.md)
    // structure runs in Cart3D (docs/specs/structure3d.md, decision 102, src/structure.cpp)
    Metal3Config metal;                          // block metal: embedded perfect conductor (metal3d.hpp)
    std::vector<Quad> quads;                     // blocks quadrupole [<name>], quadrupole_lattice (external.hpp)
    std::vector<WakeConfig> wakes;               // blocks wake [<name>] (wake.hpp)
    std::vector<SliceDiagConfig> slice_diags;    // blocks beam_slices [<name>]
    SideFluxConfig side_flux;                    // block side_flux
    void check_structure() const;
    std::vector<DielectricConfig> dielectrics;   // blocks `dielectric [<name>]`: eps(r, z) in RZ, RIP-CC (rz_dielectric.hpp, decision 100)
    std::vector<Dielectric3Config> dielectrics3; // the same blocks in Cart3D / 2D: eps(x, y, z), RIP-CC (dielectric3d.hpp)
    double plane_z = -1;                         // RZ output.plane_z: the field crossing this lab plane every step (< 0: none)
    long axis_every = 0;                         // RZ output.axis_every: Ez on the axis of the window (0: none)
    BoostConfig boost;                           // block boost (docs/specs/boosted_frame.md, decision 87)
    LabDiagConfig labdiag;                       // block lab_diagnostics (boosted frame)
    QedConfig qed;                               // block qed (qed.hpp, docs/specs/qed.md, decision 93)
    std::vector<CollisionConfig> collisions;     // blocks collisions [<name>] (collisions.hpp, docs/specs/collisions.md, decision 94)
    // mobile = false in a collisions block: frozen (never pushed, no current, momenta changed only by the collisions; Smilei's
    // time_frozen), so it may carry momenta and does not count in the omega_p dt limits
    bool frozen(const SpeciesConfig& s) const {
        if (s.mobile || s.photon) return false;
        for (const CollisionConfig& k : collisions)
            if (k.sp1 == s.name || k.sp2 == s.name) return true;
        return false;
    }
    // the factor from the deck (lab) density to the density in the frame of the run, for the omega_p dt start-up checks
    // (decision 120). A boosted beam (boost_kind 2) has n' = n gamma_b (1 - beta_b beta_beam) (lab velocity beta_beam; 0.050 n for
    // a gamma 1e4 beam at gamma_b = 10). A plasma at rest in the lab (boost_kind 1) keeps its lab density n: its boosted density
    // is gamma_b n, but the transverse response of the streaming plasma is n' / gamma_b = n (unchanged, decision 120).
    double check_density_factor(const SpeciesConfig& s) const {
        return boost.on && s.boost_kind == 2 ? boost.gamma * (1 - boost.beta * s.beam_v) : 1.0;
    }

    static Config from_deck(const Deck& d);
    void summary(FILE* f, int ranks) const;
    void check() const;                          // stability and consistency
    void check_rz() const;
    void check_flat() const;
    void check_window() const;
    void check_mover() const;                    // numerics.mover = implicit (O6)
};

unsigned long long mix64(unsigned long long x);   // splitmix64 (seeds per cell)
// structure runs (src/structure.cpp, docs/specs/structure3d.md)
void structure_from_deck(const Deck& d, Config& c, double k0);
void species_file_from_deck(const Deck& d, const std::string& b, const Config& c, SpeciesConfig& s, double k0);
void species_bunch_from_deck(const Deck& d, const std::string& b, const Config& c, SpeciesConfig& s, double k0);
void structure_summary(FILE* f, const Config& c);
std::unique_ptr<Operator> make_metal_absorb_operator();
// the key particles = pass (default) | absorb of the blocks metal and rings (decision 122): true for absorb
bool metal_particles_absorb(const Deck& d, const std::string& block);
// the block (metal, rings) of an irip run as a Drude medium, without its profile (src/structure.cpp, decision 124)
MediumConfig metal_medium(const Deck& d, const std::string& block, double dt, bool absorb_ok = false);
// grid.absorber_particles = remove (src/absorber_particles.cpp): Cart3D: the particles past the y/z PML faces (L[a] layer cells, 0: none)
// are moved out of the box (migrate drops them); returns their charge q w Z, count = their number. RZ: the removal radius of the
// radial absorber (cells layer cells) and the charge / number of the particles at r >= rrm. absorber_report: totals (collective).
struct RadialGrid;
class Simulation;
class FieldsRZ;
enum { ABSORBER_KEEP = 0, ABSORBER_REMOVE = 1, ABSORBER_CONSERVE = 2 };
double absorber_remove_cart(Species& sp, const Domain& d, const int L[3], int order, int mode, double& count);
void absorber_zero_currents_cart(Fields& f, const Domain& d, const int L[3]);   // remove: no current in the layers
double rz_absorber_radius(const RadialGrid& g, int cells, int order, int mode);
void rz_absorber_zero_currents(FieldsRZ& f, int cells);
double rz_absorber_count(Species& sp, double rrm, double& count);
void absorber_report(Simulation& s, FILE* f);
std::unique_ptr<Operator> make_rz_plane_operator();   // RZ output.plane_z / axis_every (src/rz_sim.cpp)
std::unique_ptr<Operator> make_front_layer_watch();   // dielectric + window: charged particles in the front layer (src/rz_sim.cpp)
std::unique_ptr<Operator> make_rz_move_deposit_operator();   // RZ stage MoveDeposit with spin / QED (src/rz_sim.cpp)
// Cart3D push and deposit of species i with the deck's deposit path (Pusher::None: move and deposit only), and the rest of the
// push stage after all species (ionization currents, guards, filter, side fluid)
void push_and_deposit_species(Simulation& s, size_t i, Pusher pusher);
void push_deposit_tail(Simulation& s);
class Simulation;
// O7: dir/medium.txt (medium_sim.cpp): K, the dissipated energy, the balance with W_field + W_kin = wtot, the GMRES statistics
void medium_diagnostics(Simulation& s, bool root, double wtot);
void medium_summary(FILE* f, const Config& c);       // the media of the deck (medium_sim.cpp)
// model = bound with ionization = adk: electron-ion pairs from the ionized atoms (medium_sim.cpp; an operator after the field step)
void medium_emit(Simulation& s);
std::unique_ptr<Operator> make_medium_ionize_operator();

class Simulation {
public:
    Simulation(const Config& c, Comm& comm);
    // Loads the species, sets up hooks and operators; with a checkpoint path the state comes from it
    // instead (openpmd.hpp), and restart_step is its step.
    void init(const std::string& restart = "");
    void step();                                 // runs the pipeline once: t_n -> t_{n+1}
    void diagnostics(FILE* f, bool header);
    void write_slices();                         // x-y slices as float32 files in dir/slices
    bool finished() const { return t >= cfg.t_end - 0.5 * dom.dt(); }
    long istep = 0;
    long front_layer_warnings = 0;               // dielectric + window (RZ, Cart3D, 2D): warnings of charged particles in the front layer (decision 100)
    double t = 0;
    long restart_step = -1;                      // step of the checkpoint this run started from, -1: none
    std::string restart_path;                    // the checkpoint this run started from ("": none; read_op_state)
    double absorbed_r = 0;                       // RZ: energy removed by the radial absorber since the start (local; with xflux)
    double peak_wpdt = 0;                        // omega_p dt at the peak density loaded at the start (Cart3D, O4)
    Config cfg;
    Domain dom;
    Comm& comm;
    Fields* fields = nullptr;                    // Cart3D
    FieldsRZ* rz = nullptr;                      // RZ (cfg.rz): fields per mode (src/rz_sim.cpp)
    std::vector<Species> species;
    std::shared_ptr<struct ImplicitMover> mover; // numerics.mover = implicit: its work arrays (implicit_mover.cpp)
    std::unique_ptr<Medium> medium;              // O7: blocks `medium` (medium.hpp); null: none
    std::shared_ptr<struct Ionization> ionization;   // field ionization (ionization.cpp); null: none
    std::shared_ptr<struct Qed> qed;             // spin and QED (qed.cpp); null: none
    std::shared_ptr<struct Collisions> collisions;   // binary collisions (collisions.cpp); null: none
    std::vector<char> split;                     // per species: pushed by qed_push in stage Push, moved in stage MoveDeposit (decision 15)
    std::shared_ptr<struct BoostAntenna> antenna;    // laser.injection = antenna in a boosted run (boost_antenna.cpp); null: none
    void setup_medium();                         // builds it (Cart3D and RZ) after the fields
    double medium_wp2_max = 0;                   // the largest wp^2 of the media (omega_p dt of the summary)
    FILE* medium_f = nullptr;                    // dir/medium.txt (rank 0)
    double medium_energy();                      // K of the media, summed over the ranks (medium_sim.cpp)
    ~Simulation();
    // Gauss-law residual (dx Ex + mu_x dy Ey + mu_x dz Ez - rho) relative to the initial one
    double gauss_drift();
    void compute_rho();                          // with numerics.filter_passes the filtered charge (filter.hpp)
    void filter_currents();                      // Cart3D current filter (numerics.filter_passes, docs/specs/solid_target.md)
    // species.rigid_until (decision 98): the momenta of a rigid species are saved before its push and restored after it
    Buf<double> rigid_u[3];
    bool rigid_save(size_t i);                   // true if species i is rigid now (then its momenta are saved)
    void rigid_restore(size_t i);
    BoundaryHooks hooks;                         // field boundary hooks (laser inflow)
    std::vector<std::shared_ptr<FileLaser>> file_lasers;   // on the ranks at the left x edge
    std::vector<std::shared_ptr<RzFileLaser>> rz_file_lasers;   // RZ: lab runs on the rank at the left z edge, boosted runs everywhere
    Pipeline pipeline;                           // operators of one step (pipeline.hpp)
    TileGrid tiles;                              // tile layout of the owned box
    bool tiled_deposit = false;                  // resolved numerics.deposit: coloured tiles (CPU)
    bool box_deposit = false;                    // resolved numerics.deposit: box per tile (GPU design)
    // Overlap of the current guard exchanges with the interior push (sim.cpp PushDeposit; VLPL_OVERLAP=0 turns it off):
    // 0 none, 1 the guard sums, 2 also the copy back that starts a plain RIP-CC field step
    int push_overlap() const;
    long sort_every = 0;                         // resolved numerics.sort_every (auto: 1 if the deposit needs it)
    long strays = 0;                             // particles deposited atomically by the tiled path
    // Appends the particles of species `index` in the local cells [i0, i1) along x (all of y, z).
    void load(const SpeciesConfig& sc, Species& s, int index, int i0, int i1);
    // the lattice cells gi0 .. gi0 + nx - 1 along x (boost.hpp Lattice; plain: absolute cells) and every owned y, z cell
    void load_cells(const SpeciesConfig& sc, Species& s, int index, long gi0, int nx, const Lattice& lat);
    void load_rz_cells(const SpeciesConfig& sc, Species& s, int index, long gi0, int nz, const Lattice& lat);
    void shift_window();                         // one cell along +x: fields, then new plasma (window.cpp)
    void load_species(size_t i);                 // all owned cells, plus the self field if requested
    std::vector<char> loaded;                    // species already loaded (t_start reached)
    Pml* pml() { return pml_.get(); }
    // RZ (src/rz_sim.cpp)
    void setup_rz();                             // domain, fields, operators
    void load_rz(const SpeciesConfig& sc, Species& s, int index, int i0, int i1);   // local z cells [i0, i1)
    void shift_window_rz();
    // The plasma entering a window at v = c (decision 91): it is frozen in the last window_freeze_cells() cells (3, plus the
    // filter passes); a column that leaves them has its charge (filtered as rho) folded into gauss0_ as a background, and
    // window_frozen_rho gives the charge of the columns still frozen (false: none). The Gauss check then measures the
    // drift of div E - (rho - the charge that entered), round-off also for a non-neutral entering column.
    int window_freeze_cells() const;
    void window_fold_entered();
    bool window_frozen_rho(Field& out);
    void shift_window_medium();                  // O7: the media state moves with the window, the entering column from the profiles (medium_sim.cpp)
    void warn_window_start();                    // at the start of a run with a moving window
    void rz_rho_all(Field& rho);                 // rho per mode of all species (guards added)
    void rz_gauss_residual(Field& out);          // div E - rho per mode (rho in rz_rho_)
    void rz_gauss_check(Field& out);             // rz_gauss_residual + the frozen entry columns of a window (decision 91)
    void rz_diagnostics(double& wfield, double& gauss, double& rhomax, double& divb);
    // per mode m (over all ranks): the field energy and the largest change of the Gauss residual
    void rz_mode_diagnostics(std::vector<double>& w, std::vector<double>& gauss);
    Field& gauss0() { return gauss0_; }          // initial Gauss residual (checkpoints)
    // side tube: plasma check, initial incident field, x-end warning (sim.cpp; public: nvcc refuses device lambdas in
    // private member functions)
    void side_setup(bool fresh);
    RzTfsf* rz_tfsf() { return rz_tfsf_.get(); }   // RZ: the TFSF plane (null: none)
    // structure runs (src/structure.cpp): the metal, the quadrupole table, beams from files, the wake diagnostics
    std::unique_ptr<Metal3> metal3;              // block metal; null: none
    std::unique_ptr<Metal3> metal_mask_;         // irip with metal.particles = absorb: the node mask of the medium metal (decision 124)
    Field rho_abs_, absorb_tmp_, absorb_tmp2_;   // its absorbed charge at rest (owned cells, guards 0; added in compute_rho; in the
                                                 // checkpoint as vlpl_metal_absorbed); scratch
    std::unique_ptr<Dielectric3> diel3;          // blocks dielectric in Cart3D / 2D; null: none
    void setup_structure();                      // in the constructor (after the fields)
    void load_file_species(size_t i);            // species.file
    double metal_absorb_all();                   // particles inside the metal absorbed (kinetic energy removed, local)
    void structure_operators();                  // the diagnostics operators (init)
private:
    std::unique_ptr<RzTfsf> rz_tfsf_;            // RZ: O1 TFSF plane (laser.tfsf_x, src/rz_tfsf.cpp)
    std::unique_ptr<Pml> pml_;
    std::unique_ptr<Tfsf> tfsf_;                 // O1 TFSF source (laser.tfsf_x)
    std::unique_ptr<SideTfsf> side_;             // the side TFSF tube (block side)
    Field gauss0_, work_, rz_rho_, filt_;
    Field entry_rho_;                            // window_frozen_rho's scratch (window at v = c only)
    std::shared_ptr<struct FilterProjWork> fproj_;   // the projection filter's work arrays (filter.hpp)
    bool warned_fmax_ = false;
    std::vector<double> peak_;                   // largest density factor loaded per species (local)
    void check_peak_density();                   // omega_p dt at the peak density after the initial load (O4)
    FILE* modes_f_ = nullptr;                    // RZ: dir/rz_modes.txt (rank 0)
};
