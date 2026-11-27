// RZ with azimuthal modes in the Simulation (docs/specs/rz_modes.md, step 2b): grid.geometry = rz sets up
// the RZ domain and fields, loads the species in the cylinder and runs push/deposit, the field step and
// the migration along z, where the conducting wall at r = nr dr absorbs particles.
#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <stdexcept>
#include "filter.hpp"
#include "implicit_mover.hpp"
#include <cstdio>
#include <string>
#include "fsutil.hpp"
#include "laser.hpp"
#include "sim.hpp"

namespace {
// after the deposit of all species: ionization currents, guards, transverse shape, filter
void rz_push_tail(Simulation& s) {
    ionization_currents(s);                        // the displacements of this step's new electrons (ionization.hpp)
    s.comm.add_guards(s.dom, {&s.rz->jz, &s.rz->jr, &s.rz->jp});
    rz_smooth_currents(*s.rz, s.comm);             // S J (decision 79), then the filter
    rz_axis_zero_jz(*s.rz);                        // J_z of m >= 1 zero on the axis before the filter spreads it (J_z ~ r^m)
    rz_filter_currents(*s.rz, s.comm, s.cfg.filter_passes, s.cfg.filter_local, s.cfg.filter_binomial);
    rz_axis_zero_jz(*s.rz);
    if (s.cfg.absorber_mode == ABSORBER_REMOVE && s.cfg.rz_abs_cells > 0) rz_absorber_zero_currents(*s.rz, s.cfg.rz_abs_cells);   // decision 111
}

class RZPushDeposit : public Operator {
public:
    const char* name() const override { return "rz_push_deposit"; }
    Stage stage() const override { return Stage::Push; }
    void apply(Simulation& s) override {
        rz_smooth_e(*s.rz, s.comm);                // the transverse shape: S E for the gather (decision 79)
        s.rz->zero_currents();
        for (size_t i = 0; i < s.species.size(); ++i) {
            if (!s.cfg.species[i].mobile) continue;                                    // static: no push
            if (!s.split.empty() && s.split[i]) { qed_push(s, i); continue; }          // spin / QED: moved in stage MoveDeposit
            const bool rigid = s.rigid_save(i);       // species.rigid_until (decision 98)
            rz_push_deposit(s.species[i], *s.rz, s.cfg.pusher, true, s.cfg.order);
            if (rigid) s.rigid_restore(i);
        }
        if (!s.qed) rz_push_tail(s);               // with spin / QED after the move (RZMoveDeposit, decision 15)
    }
};

class RZMoveDeposit : public Operator {
public:
    const char* name() const override { return "rz_move_deposit"; }
    Stage stage() const override { return Stage::MoveDeposit; }
    void apply(Simulation& s) override {
        for (size_t i = 0; i < s.species.size(); ++i)
            if (s.cfg.species[i].mobile && s.split[i]) rz_push_deposit(s.species[i], *s.rz, Pusher::None, true, s.cfg.order);
        rz_push_tail(s);
    }
};

// the radial absorber after the field step (grid.boundary_r = absorbing)
void rz_radial_absorber(Simulation& s) {
    if (s.cfg.rz_abs_cells > 0) {
        // kappa: a wave crossing the layer twice (a steps each way) is damped by R0: exp(-2 kappa a / 3) = R0
        const RadialGrid& g = s.rz->grid();
        const double a = (g.R() - g.r[size_t(g.nr - s.cfg.rz_abs_cells)]) / s.dom.dt();
        // the energy balance in xflux.txt: only between its measuring planes (the inner faces of the z layers); what
        // the radial absorber takes beyond them (the corners with the z layers) has been counted by Enet already
        const int L = s.cfg.x_abs_cells, i0 = L - s.dom.off[0], i1 = s.dom.N[0] - 1 - L - s.dom.off[0];
        const double w0 = s.cfg.xflux.on ? s.rz->energy(i0, i1) : 0.0;
        rz_absorb(*s.rz, s.cfg.rz_abs_cells, 1.5 * std::log(1 / s.cfg.rz_abs_R0) / a);
        if (s.cfg.xflux.on) s.absorbed_r += w0 - s.rz->energy(i0, i1);
        s.comm.exchange(s.dom, {&s.rz->ez, &s.rz->er, &s.rz->ep, &s.rz->bz, &s.rz->br, &s.rz->bp});
    }
}

class RZFieldStep : public Operator {
public:
    const char* name() const override { return "rz_fields"; }
    Stage stage() const override { return Stage::Fields; }
    void apply(Simulation& s) override {
        s.rz->step(s.comm, s.t);
        rz_radial_absorber(s);
    }
};

// O6 (numerics.mover = implicit, decision 82): particles and fields in one nonlinear solve, then the radial absorber
class RZImplicitStep : public Operator {
public:
    const char* name() const override { return "implicit_mover"; }
    Stage stage() const override { return Stage::Fields; }
    void apply(Simulation& s) override {
        implicit_mover_step(s);
        rz_radial_absorber(s);
    }
};

// Along z as in Cart3D; particles leave through the wall once their shape has no weight left on the last
// node r_{nr-1}: at xi >= nr for shape 1, xi >= nr + 1/2 for shape 2 (their current up to there was
// deposited, so the Gauss law inside holds; removed at nr dr, a shape-2 particle took charge from the last
// node without a current, 6e-4 in the LWFA benchmark).
class RZMigrate : public Operator {
public:
    const char* name() const override { return "migrate"; }
    Stage stage() const override { return Stage::Migrate; }
    void apply(Simulation& s) override {
        const RadialGrid& g = s.rz->grid();
        double rwall = g.R() + 0.5 * (s.cfg.order - 1) * g.dn[size_t(g.nr)];
        const bool rm = s.cfg.absorber_mode != ABSORBER_KEEP && s.cfg.rz_abs_cells > 0;   // grid.absorber_particles (absorber_particles.cpp)
        if (rm) rwall = std::min(rwall, rz_absorber_radius(g, s.cfg.rz_abs_cells, s.cfg.order, s.cfg.absorber_mode));
        std::vector<Species*> v;
        for (auto& sp : s.species) {
            if (rm) { double n = 0; sp.lost_charge += rz_absorber_count(sp, rwall, n); sp.lost_count += n; }
            // particles = absorb of the blocks rings / metal (default pass: they cross the metal, decision 122)
            if (s.rz->metal && s.cfg.rings.absorb) rz_metal_remove(sp, *s.rz, rwall, s.cfg.order);
            if (s.rz->metal_phi && s.cfg.metal_rz.absorb) rz_metal_phi_remove(sp, *s.rz, *s.rz->metal_phi, rwall, s.cfg.order);
            v.push_back(&sp);
        }
        migrate(v, s.dom, s.comm, rwall);              // all species in one message per direction
    }
};

// output.plane_z / axis_every (docs/specs/rz_metal.md): after the field step (before a window shift). plane: every step,
// on the rank that owns the lab column I_p = round(plane_z / dz), the record [step, t, then Ez Er Ephi Bz Br Bphi, each
// [slots][nr]] (float64) into dir/plane/plane_r<rank>.bin; axis: every axis_every steps every rank writes [step, t,
// first absolute cell, cells, Ez(m = 0, r = 0) of its owned cells] into dir/plane/axis_r<rank>.bin. dir/plane/meta.txt:
// the layout, lambda0, dz, dt, I_p and the radii r_j, r_{j+1/2} (code units).
class RZPlane : public Operator {
public:
    const char* name() const override { return "rz_plane"; }
    Stage stage() const override { return Stage::Fields; }
    void setup(Simulation& s) override {
        const std::string dir = s.cfg.dir + "/plane";
        if (s.comm.rank == 0) {
            make_dirs(dir);
            FILE* f = std::fopen((dir + "/meta.txt").c_str(), "w");
            const RadialGrid& g = s.rz->grid();
            std::fprintf(f, "nr %d\nslots %d\nlambda0 %.17g\ndz %.17g\ndt %.17g\nplane_cell %ld\naxis_every %ld\n",
                         s.dom.N[1], s.dom.N[2], s.cfg.lambda0, s.dom.h[0], s.dom.dt(), plane_cell(s), s.cfg.axis_every);
            std::fprintf(f, "components Ez Er Ephi Bz Br Bphi\nr");
            for (int j = 0; j < g.nr; ++j) std::fprintf(f, " %.17g", g.r[size_t(j)]);
            std::fprintf(f, "\nrh");
            for (int j = 0; j < g.nr; ++j) std::fprintf(f, " %.17g", g.rh[size_t(j)]);
            std::fprintf(f, "\n");
            std::fclose(f);
        }
        s.comm.barrier();
    }
    void apply(Simulation& s) override {
        const Domain& d = s.dom;
        const int nz = d.n[0], nr = d.n[1], S = d.n[2];
        const long step = s.istep + 1;
        const double t = step * d.dt();
        const std::string dir = s.cfg.dir + "/plane";
        if (s.cfg.plane_z >= 0) {
            const long i = plane_cell(s) - d.cell0(0);
            if (i >= 0 && i < nz) {
                buf_.resize(size_t(6) * S * nr);
                double* B = buf_.data();
                const FieldsRZ& f = *s.rz;
                const FV F[6] = {f.ez.view(), f.er.view(), f.ep.view(), f.bz.view(), f.br.view(), f.bp.view()};
                const FV F0 = F[0], F1 = F[1], F2 = F[2], F3 = F[3], F4 = F[4], F5 = F[5];
                const int ii = int(i);
                par_for(long(6) * S * nr, [=] HD(long q) {
                    const int j = int(q % nr), k = int((q / nr) % S), c = int(q / (long(S) * nr));
                    const FV& V = c == 0 ? F0 : c == 1 ? F1 : c == 2 ? F2 : c == 3 ? F3 : c == 4 ? F4 : F5;
                    B[q] = V(ii, j, k);
                });
                std::vector<double> h(size_t(2) + buf_.size());
                h[0] = double(step); h[1] = t;
                buf_.to_host(h.data() + 2, 0, buf_.size());
                if (!fp_) fp_ = std::fopen((dir + "/plane_r" + std::to_string(s.comm.rank) + ".bin").c_str(), "wb");
                std::fwrite(h.data(), sizeof(double), h.size(), fp_);
                std::fflush(fp_);
            }
        }
        if (s.cfg.axis_every > 0 && step % s.cfg.axis_every == 0) {
            buf_.resize(size_t(nz));
            double* B = buf_.data();
            const FV E = s.rz->ez.view();
            par_for(nz, [=] HD(long i) { B[i] = E(int(i), 0, 0); });
            std::vector<double> h(size_t(4) + size_t(nz));
            h[0] = double(step); h[1] = t; h[2] = double(d.cell0(0)); h[3] = nz;
            buf_.to_host(h.data() + 4, 0, size_t(nz));
            if (!fa_) fa_ = std::fopen((dir + "/axis_r" + std::to_string(s.comm.rank) + ".bin").c_str(), "wb");
            std::fwrite(h.data(), sizeof(double), h.size(), fa_);
            std::fflush(fa_);
        }
    }
    ~RZPlane() override { if (fp_) std::fclose(fp_); if (fa_) std::fclose(fa_); }
private:
    static long plane_cell(const Simulation& s) { return s.cfg.plane_z >= 0 ? std::lround(s.cfg.plane_z / s.dom.h[0]) : -1; }
    Buf<double> buf_;
    FILE* fp_ = nullptr;
    FILE* fa_ = nullptr;
};

// Dielectric blocks with a moving window (decision 100): the 32-cell front absorbing layer damps every field there, so charged
// particles must stay behind it. Warns (rank 0) at the start for every charged species whose load region reaches into the
// layer (a plasma that keeps entering through the front edge: one note, harmless for a neutral plasma at rest), and during
// the run when particles of the other charged species are inside it: on entry, then at most every 2000 steps while it lasts.
class RZFrontLayerWatch : public Operator {
public:
    const char* name() const override { return "rz_front_layer_watch"; }
    Stage stage() const override { return Stage::Fields; }
    void setup(Simulation& s) override {
        const int L = s.cfg.rz ? s.rz->zabs_hi : s.fields->dfront;   // Cart3D / 2D: x is the window axis
        const double dz = s.dom.h[0], front = (s.dom.wx + s.dom.N[0]) * dz, zl = front - L * dz;
        entering_.assign(s.cfg.species.size(), false);
        in_.assign(s.cfg.species.size(), false);
        last_.assign(s.cfg.species.size(), -(1L << 40));
        for (size_t i = 0; i < s.cfg.species.size(); ++i) {
            const SpeciesConfig& c = s.cfg.species[i];
            if (c.q == 0 || c.photon || !(c.density > 0) || c.rmax[0] < zl) continue;
            entering_[i] = c.rmax[0] >= front && c.t_start <= 0;
            ++s.front_layer_warnings;
            if (s.comm.rank == 0)
                std::fprintf(stderr, entering_[i]
                    ? "warning: species %s enters the window through its front edge: in the last %d cells (the absorbing layer of the "
                      "dielectric blocks, decision 100) its fields are damped; harmless for a neutral plasma at rest ahead of the beam\n"
                    : "warning: species %s is loaded within the last %d cells of the window (%s > %.6g): the front absorbing layer of the "
                      "dielectric blocks (decision 100) damps its fields there; start it further behind the front edge\n",
                    c.name.c_str(), L, s.cfg.rz ? "z" : "x", zl);
        }
    }
    void apply(Simulation& s) override {
        if (s.istep % 20 != 0) return;
        const int L = s.cfg.rz ? s.rz->zabs_hi : s.fields->dfront;   // Cart3D / 2D: x is the window axis
        const double zl = (s.dom.wx + s.dom.N[0] - L) * s.dom.h[0];
        for (size_t i = 0; i < s.species.size(); ++i) {
            if (entering_[i] || s.species[i].q == 0) continue;
            const long n = long(s.species[i].size());
            const double* X = n ? s.species[i].ptr<double>(Species::X) : nullptr;
            double c = n ? par_sum(n, [=] HD(long q) { return X[q] >= zl ? 1.0 : 0.0; }) : 0.0;
            s.comm.allreduce_sum(&c, 1);
            if (c == 0) { in_[i] = false; continue; }
            if (in_[i] && s.istep - last_[i] < 2000) continue;
            if (s.comm.rank == 0)
                std::fprintf(stderr, "warning: step %ld: %.0f particles of species %s %s the last %d cells of the window (%s > %.6g), "
                             "the front absorbing layer of the dielectric blocks (decision 100): their fields are damped there\n",
                             s.istep, c, s.species[i].name.c_str(), in_[i] ? "still in" : "inside", L, s.cfg.rz ? "z" : "x", zl);
            ++s.front_layer_warnings;
            in_[i] = true;
            last_[i] = s.istep;
        }
    }
private:
    std::vector<bool> entering_, in_;
    std::vector<long> last_;
};
}  // namespace

std::unique_ptr<Operator> make_rz_plane_operator() { return std::make_unique<RZPlane>(); }
std::unique_ptr<Operator> make_front_layer_watch() { return std::make_unique<RZFrontLayerWatch>(); }
std::unique_ptr<Operator> make_rz_move_deposit_operator() { return std::make_unique<RZMoveDeposit>(); }

void Simulation::setup_rz() {
    dom = rz_domain(comm, cfg.N[0], cfg.N[1], cfg.modes, cfg.h[0], cfg.h[1], cfg.periodic[0], cfg.solver, cfg.mpi[0]);
    dom.filt_comp = cfg.filter_comp; dom.filt_xonly = cfg.filter_xonly;
    set_flat_filter(dom, cfg.filter_passes, cfg.filter_flat);
    rz = new FieldsRZ(dom, cfg.modes, cfg.radial_grid());
    rz->o5 = cfg.o5;
    if (cfg.mover_implicit) rz->use_c1_map();     // the C1 radial map of the implicit mover (decision 82)
    if (cfg.rings.on) {                            // embedded metal rings (docs/specs/rz_metal.md)
        rz->metal = rz_metal_setup(cfg.rings, cfg.N[0], cfg.h[0], cfg.periodic[0], rz->grid());
        if (comm.rank == 0) std::printf("%s", rz->metal->describe(cfg.h[0], rz->grid()).c_str());
    }
    set_quadrupoles(cfg.quads);                    // external quadrupoles in the push (external.hpp; RZ axes z, x, y)
    if (cfg.metal_rz.on) {                            // conductors depending on phi (docs/specs/rz_metal_phi.md)
        rz->set_metal_phi(std::make_shared<MetalPhi>(cfg.metal_rz, dom, cfg.modes, rz->grid().r));
        if (cfg.metal_rz.spectral && rz->metal_phi->has_phi_surfaces() && comm.rank == 0)
            std::fprintf(stderr, "warning: metal.phi_derivative = spectral with a mask that depends on phi: the TE fields at phi "
                         "surfaces are wrong (decision 103); use the default phi_derivative = difference\n");
        if (comm.rank == 0) std::printf("%s", rz->metal_phi->describe().c_str());
    }
    if (!cfg.dielectrics.empty()) {                // dielectric regions (rz_dielectric.hpp, decision 100)
        rz->diel = std::make_shared<DielectricRZ>(cfg.dielectrics, dom, rz->grid());
        // eps depending on phi: chi on the 2M - 1 angles (E_phi at the staggered angles of the metal block, if any)
        rz->diel->set_angles(dom, rz->grid(), cfg.modes, rz->metal_phi && !rz->metal_phi->cfg.spectral);
        // moving window: an absorbing layer of 32 cells at the front edge (decision 100). In a dielectric the grid-scale branch of the
        // scheme runs ahead of c (mu_z E of the z-Nyquist mode is 0: no polarization there) and grows at the moving open edge; ahead
        // of a beam at v ~ c nothing physical is there
        if (cfg.window.enabled && !cfg.periodic[0]) { rz->zabs_hi = 32; rz->zabs_kappa = 1.5 * std::log(1e3) / 32; }
        if (comm.rank == 0) std::printf("%s", rz->diel->describe(rz->grid(), cfg.h[0]).c_str());
    }
    rz->gather_matched = cfg.rz_gather_matched;
    rz->gather_energy = cfg.rz_gather_energy;
    rz->map_all = cfg.rz_map_all;
    rz->deposit_chord = cfg.rz_deposit_chord;
    // numerics.rz_axis_current (decision 112); not with the implicit mover (decision 82): its fixed-point iteration does not converge with
    // the first-cell projection (mover test, m = 0, 1 column: 30 of 30 steps unconverged, raw 7 of 30): it keeps the raw current (open)
    rz->axis_regular = cfg.rz_axis_regular && !cfg.mover_implicit;
    rz->smooth_a = cfg.rz_shape_width * rz->grid().dr;          // the transverse shape (decision 79; 0: none)
    if (comm.rank == 0) {                          // the gather and the filter against the plasma of the run (decision 106)
        bool dense = false, streaming = false;     // (a boosted run: boost_nci_risk in src/boost.cpp warns, decision 104)
        for (const auto& s : cfg.species) {
            if (!s.mobile || s.density <= 0) continue;
            const double kd = std::sqrt(s.density * s.q * s.q / s.m) * rz->grid().dr;
            if (!s.self_field && std::abs(s.u0[0]) >= 1) streaming = true;    // a plasma drifting through the grid (gamma >~ 1.4)
            if (kd <= 1.5) continue;
            dense = true;                          // the axis instability of dense plasma (decision 80): k_p dr <~ 1.5 at the axis
            std::fprintf(stderr, "warning: species %s: k_p dr = %.2f at the axis (density %g n_c): a dense plasma next to the "
                         "axis is unstable for k_p dr >~ 1.5 (decision 80, bench/noise_heating); refine dr near the axis "
                         "(grid.r_uniform, stretch, dr_max) to k_p dr <= 1%s\n",
                         s.name.c_str(), kd, s.density,
                         cfg.rz_gather_matched ? "" : "; numerics.rz_gather = plain (the default) heats about twice as fast there as "
                         "numerics.rz_gather = matched: set matched for a dense target");
        }
        if (cfg.rz_gather_matched && !dense)
            std::fprintf(stderr, "warning: numerics.rz_gather = %s is meant for dense plasma at the axis (k_p dr > 1.5; none here): it "
                         "overestimates a field that grows with r in the cell next to the axis (a linear focusing field x 1.25 at "
                         "r = dr/2 with shape 1, x 1.6 with shape 2: wrong betatron frequency of beams narrower than ~2 dr); "
                         "numerics.rz_gather = plain is exact there (decision 106)\n", cfg.rz_gather_energy ? "energy" : "matched");
        if (streaming && !cfg.boost.on && cfg.filter_passes == 0 && rz->grid().dr < dom.dt())
            std::fprintf(stderr, "warning: a plasma streams through the grid (a drift u >= 1) with dr = %.3g < "
                         "c dt = %.3g and no current filter: the numerical Cherenkov instability grows (W_field / W_kin 1.3e-3 at "
                         "dr = c dt / 4, gamma 10, shape 1; 5e-10 with one binomial pass, bench/nci): set numerics.filter_passes = 1\n",
                         rz->grid().dr, dom.dt());
    }
    if (comm.rank == 0 && cfg.rz_shape_width == 0 && cfg.modes > 1)  // the noise heating grows with 2M - 1 at fixed ppc_theta
        for (const auto& s : cfg.species)
            if (s.mobile && s.density > 0 && s.ppc[2] < 2 * (2 * cfg.modes - 1))   // density 0: filled by ionization (O7 bound)
                std::fprintf(stderr, "warning: species %s: ppc_theta = %d with %d modes samples %d azimuthal components: noise "
                             "heating (decision 79, bench/noise_heating); use ppc_theta >= %d, or numerics.rz_transverse_shape = "
                             "gaussian when the physics has no large k_perp (it damps harmonics)\n",
                             s.name.c_str(), s.ppc[2], cfg.modes, 2 * cfg.modes - 1, 2 * (2 * cfg.modes - 1));
    if (comm.rank == 0 && cfg.solver == Solver::RipCC) {   // how close the run is to the RIP-CC limit (Config::check_rz)
        double wp2 = 0;
        for (auto& s : cfg.species) wp2 += s.density * s.q * s.q / s.m;
        const double D = cfg.h[0], c = 0.25 * D * D * (rz_lambda_max(cfg.radial_grid(), cfg.modes) + wp2);
        std::printf("  rz rip_cc: dz^2 (lambda_max + wp^2) / 4 = %.3f (stable below 1)\n", c);
    }
    std::vector<LaserConfig> edge;                 // lasers through the open left edge; tfsf_x: through the TFSF plane
    bool plane = false;
    for (const LaserConfig& L : cfg.all_lasers()) {
        if (L.enabled && L.tfsf_x >= 0) plane = true;
        else edge.push_back(L);
    }
    // LASY lasers (laser_file_rz.cpp): lab runs on the rank at the left z edge; boosted runs on every rank (the antenna sheet
    // crosses them all)
    if (cfg.boost.on || comm.nbr[0][0] < 0)
        for (const LaserFileConfig& F : cfg.file_lasers) {
            rz_file_lasers.push_back(std::make_shared<RzFileLaser>(F, cfg.lambda0, cfg.modes, rz->grid().R()));
            const RzFileLaser& L = *rz_file_lasers.back();
            if (comm.rank == 0 && restart_step < 0) {
                for (const std::string& line : L.info) std::printf("  %s: %s\n", F.name.c_str(), line.c_str());
                if (L.edge > 1e-3)
                    std::fprintf(stderr, "warning: %s: the envelope at the edge of its grid is %.2g of the peak (a pulse cut "
                                         "by the file, or wrapped around by the propagation)\n", F.name.c_str(), L.edge);
                const double ws = cfg.boost.on ? cfg.boost.start_lab : cfg.window.start;
                if (cfg.window.enabled && ws < L.t_last)
                    std::fprintf(stderr, "warning: %s enters z = 0 until (lab) t = %.4g, but the window moves from t = %.4g on and "
                                         "the injection stops then\n", F.name.c_str(), L.t_last, ws);
            }
        }
    rz_add_laser_inflow(*rz, edge, comm, cfg.boost.on ? std::vector<std::shared_ptr<RzFileLaser>>{} : rz_file_lasers);
    double zkappa = 0;
    if (cfg.x_abs_cells > 0) {                     // z layers: kappa as the x layers of Cart3D (a = cells steps across)
        zkappa = 1.5 * std::log(1 / cfg.x_abs_R0) / cfg.x_abs_cells;
        rz->zabs_lo = rz->zabs_hi = cfg.x_abs_cells;
        rz->zabs_kappa = zkappa;
    }
    if (plane) {
        double rkappa = 0;
        if (cfg.rz_abs_cells > 0) {                // as RZFieldStep
            const RadialGrid& g = rz->grid();
            rkappa = 1.5 * std::log(1 / cfg.rz_abs_R0) / ((g.R() - g.r[size_t(g.nr - cfg.rz_abs_cells)]) / dom.dt());
        }
        rz_tfsf_ = std::make_unique<RzTfsf>(dom, comm, cfg.modes, cfg.radial_grid(), cfg.all_lasers(), rz->zabs_hi, zkappa,
                                            cfg.rz_abs_cells, rkappa);
        rz->tfsf = rz_tfsf_.get();
        if (cfg.o5) rz_tfsf_->set_o5();               // the incident grid with the closure (decision 78)
    }
    if (!cfg.media.empty()) setup_medium();       // O7 (medium.hpp)
    gauss0_.resize(dom); work_.resize(dom); rz_rho_.resize(dom);
    if (rz->diel && rz->zabs_hi > 0 && cfg.x_abs_cells == 0) pipeline.add(std::make_unique<RZFrontLayerWatch>());   // decision 100
    if (cfg.mover_implicit) pipeline.add(std::make_unique<RZImplicitStep>());   // O6: particles and fields together
    else {
        pipeline.add(std::make_unique<RZPushDeposit>());
        if ((antenna = make_boost_antenna(*this))) pipeline.add(make_boost_antenna_operator());   // its current after the deposit
        pipeline.add(std::make_unique<RZFieldStep>());
    }
    if (medium && medium->bound && medium->bc.ion) pipeline.add(make_medium_ionize_operator());   // O7 bound: electron-ion pairs
    pipeline.add(std::make_unique<RZMigrate>());
}

// Gauss-Legendre points and weights of order n on [0, 1]
static void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
    x.assign(size_t(n), 0.0); w.assign(size_t(n), 0.0);
    for (int i = 0; i < n; ++i) {
        double t = std::cos(M_PI * (i + 0.75) / (n + 0.5)), dp = 0;
        for (int it = 0; it < 100; ++it) {                       // Newton on P_n
            double p0 = 1, p1 = t;
            for (int k = 2; k <= n; ++k) { const double p2 = ((2 * k - 1) * t * p1 - (k - 1) * p0) / k; p0 = p1; p1 = p2; }
            if (n == 1) p0 = 1;
            dp = n * (t * p1 - p0) / (t * t - 1);
            const double dt = p1 / dp;
            t -= dt;
            if (std::abs(dt) < 1e-16) break;
        }
        x[size_t(i)] = 0.5 * (1 - t); w[size_t(i)] = 1.0 / ((1 - t * t) * dp * dp);   // on [0, 1]: weight sum 1
    }
}

// Regular positions in (z, theta) and Gauss-Legendre points in r within each owned cell (r_j .. r_{j+1}); the
// weight of a particle is the density times its share of the cell volume, 2 pi r dn_j g_b dz / (ppc_z ppc_theta)
// (g_b the Gauss weights, sum 1). With the shape-1 Ruyten weights the deposited charge of a cell is a cubic
// in the position: from ppc_r >= 2 points the uniform density is loaded exactly onto every node, the axis
// node too (decision 53). Momenta as in Cart3D, from a generator seeded by the absolute cell (independent
// of the decomposition).
void Simulation::load_rz(const SpeciesConfig& sc, Species& s, int index, int i0, int i1) {
    load_rz_cells(sc, s, index, dom.cell0(0) + i0, i1 - i0, Lattice{});
}

// The lattice cells gi0 .. gi0 + nz - 1 along z (boost.hpp; plain: the absolute cells), every owned radial cell; with
// lat.boost lab momenta boosted per particle and the density factor gamma (1 - beta v_z) in the weight (as Cart3D).
void Simulation::load_rz_cells(const SpeciesConfig& sc, Species& s, int index, long gi0, int nz, const Lattice& lat) {
    if (nz <= 0 || sc.photon) return;              // photons: filled by the qed block
    if (sc.density == 0) return;                       // an empty species (filled by ionization), as Cart3D's load_cells
    const int pz = sc.ppc[0], pr = sc.ppc[1], pt = sc.ppc[2];
    const double dz = dom.h[0];
    const RadialGrid& rg = rz->grid();
    const bool hot = sc.uth[0] != 0 || sc.uth[1] != 0 || sc.uth[2] != 0;
    std::vector<double> gx, gw;
    gauss_legendre(pr, gx, gw);
    std::mt19937_64 rng;
    std::vector<double> h[Species::NC];
    double fmax = 0;
    for (int j = 0; j < dom.n[1]; ++j)
        for (int i = 0; i < nz; ++i) {
            const long gi = gi0 + i;
            if (hot) rng.seed(mix64(cfg.seed ^ mix64(index + 1000003ULL * (gi + 1000003ULL * j))));
            std::normal_distribution<double> g(0.0, 1.0);
            for (int c = 0; c < pt; ++c)
                for (int b = 0; b < pr; ++b)
                    for (int a = 0; a < pz; ++a) {
                        const double q0 = (gi + (a + 0.5) / pz) * dz, z = lat.ax * q0 + lat.bx;   // z: the deck's (lab) z
                        const double dr = rg.dn[size_t(j)], r = rg.r[size_t(j)] + gx[size_t(b)] * dr;
                        const double ph = 2 * M_PI * (c + 0.5) / pt;
                        double u[3], wb = 1;
                        if (lat.boost) {
                            double ul[3];
                            for (int q = 0; q < 3; ++q) ul[q] = hot ? sc.u0_lab[q] + sc.uth[q] * g(rng) : sc.u0_lab[q];
                            boost_momentum(lat.g, lat.b, ul, u);   // the density factor of the drift (below): neutral loading
                            double up0[3];
                            wb = boost_momentum(lat.g, lat.b, sc.u0_lab, up0);
                        } else
                            for (int q = 0; q < 3; ++q) u[q] = hot ? sc.u0[q] + sc.uth[q] * g(rng) : sc.u0[q];
                        if (!(q0 >= lat.qlo && q0 < lat.qhi)) continue;
                        if (z < sc.rmin[0] || z >= sc.rmax[0] || r < sc.rmin[1] || r >= sc.rmax[1]) continue;
                        double f = 1;
                        if (!sc.density_fn.empty()) {
                            f = sc.density_fn({z, r, r * std::cos(ph), r * std::sin(ph), ph});   // also off-axis beams
                            if (!(f > 0)) continue;
                            fmax = std::max(fmax, f);
                        }
                        const double w = sc.density * f * 2 * M_PI * r * (dr * gw[size_t(b)]) * (dz / pz) / pt * wb;
                        const double v[Species::NC] = {q0 + lat.sx, r * std::cos(ph), r * std::sin(ph), u[0], u[1], u[2], w};
                        for (int q = 0; q < Species::NC; ++q) h[q].push_back(v[q]);
                    }
        }
    s.append(h);
    if (fmax > 1 + 1e-12 && !warned_fmax_) {
        warned_fmax_ = true;
        std::fprintf(stderr, "warning: species %s: density_function reaches %.4g > 1; the stability check used "
                             "density alone\n", sc.name.c_str(), fmax);
    }
}

// Moving window in RZ (as Simulation::shift_window in Cart3D): the fields and the initial Gauss
// residual move one cell towards -z, iRIP sets the entering cell from the constraints, and the plasma of
// the entering column is loaded.
void Simulation::shift_window_rz() {
    FieldsRZ& f = *rz;
    comm.exchange_x(dom, {&gauss0_});
    shift_fields_x(dom, {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp, &gauss0_}, work_);
    ++dom.wx;
    comm.exchange_x(dom, {&f.ez, &f.er, &f.ep, &f.bz, &f.br, &f.bp});
    const bool right = dom.off[0] + dom.n[0] == dom.N[0];
    Field* rhom = nullptr;
    if (medium) {                                  // O7: the media state moves too; their charge enters the iRIP entry cell
        shift_window_medium();
        work_.fill(0);
        medium_add_rho_rz(*medium, dom, rz->grid().view(), cfg.modes, work_);
        if (medium->bound) medium_bound_rz_add_rho(*medium, dom, rz->grid().view(), cfg.modes, work_);
        rhom = &work_;
    }
    if (dom.solver == Solver::IRip) {
        if (right) rz_window_entry(f, gauss0_, rhom);
        comm.exchange(dom, {&f.ez, &f.bz});                              // on every rank
    }
    else if (f.diel) {                             // dielectric blocks: div D kept across the entering column (decision 100)
        if (right) rz_window_entry_dielectric(f, gauss0_, f.zabs_hi);
        comm.exchange(dom, {&f.ez});
    }
    if (right && !cfg.boost.on)                                       // boosted frame: boost_inject
        for (size_t i = 0; i < species.size(); ++i)
            if (loaded[i]) load_rz(cfg.species[i], species[i], int(i), dom.n[0] - 1, dom.n[0]);
    window_fold_entered();
}

void Simulation::rz_rho_all(Field& rho) {
    rho.fill(0);
    for (auto& s : species) rz_rho(s, *rz, rho, cfg.order);
    comm.add_guards(dom, {&rho});
    rz_smooth_rho(*rz, comm, rho);                                  // the charge of S J (decision 79)
    rz_filter_rho(*rz, comm, rho, cfg.filter_passes);              // the charge of the filtered current
    if (medium) {                                                   // O7: the polarization charge -div P per mode
        comm.exchange(dom, {&medium->P[0], &medium->P[1], &medium->P[2]});
        medium_add_rho_rz(*medium, dom, rz->grid().view(), cfg.modes, rho);
        if (medium->bound) {                                        // the atoms' polarization per mode
            comm.exchange(dom, {&medium->bPr, &medium->bPp});
            medium_bound_rz_add_rho(*medium, dom, rz->grid().view(), cfg.modes, rho);
        }
    }
    boost_antenna_add_rho(*this, rho);           // the charge of the boosted antenna sheet per mode (boost_antenna.cpp)
}

void Simulation::rz_gauss_residual(Field& out) {
    rz->div_e(out);
    rz_rho_all(rz_rho_);
    const FV O = out.view(), R = rz_rho_.view();
    for_owned(dom, [=] HD(int i, int j, int k) { O(i, j, k) -= R(i, j, k); });
}

// The Gauss residual of the diagnostics: rz_gauss_residual plus the charge of the frozen entry columns of a window
// (window_frozen_rho, decision 91; their earlier columns are in gauss0_)
void Simulation::rz_gauss_check(Field& out) {
    rz_gauss_residual(out);
    if (!window_frozen_rho(entry_rho_)) return;
    const FV O = out.view(), R = entry_rho_.view();
    for_owned(dom, [=] HD(int i, int j, int k) { O(i, j, k) += R(i, j, k); });
}

// Field energy (local), and over all ranks: the change of the Gauss residual per mode since the start
// (the axis node is a Gauss point of m = 0 only; the last cell at an open right z edge is none, its point
// needs fields beyond the edge, as in Cart3D), max |rho| and max |div B| per mode. With the current filter and a
// moving window the first filter_passes cells at the left edge are not checked either: each shift makes interior
// cells edge cells, whose fields hold the charge filtered as interior (the edge closure differs, decision 54); the
// offset is frozen in them (it cannot propagate) and leaves with them.
static int rz_gauss_istart(const Domain& d, const Config& c) {
    return c.window.enabled && c.filter_passes > 0 && !d.periodic[0] && d.off[0] == 0 ? c.filter_reach() : 0;
}
// The local z range [i0, iend) of the checks, without the z layers and one cell more (as Cart3D, docs/specs/x_absorber.md);
// sp: the local Gauss point between the scattered and the total cell of the TFSF plane (skipped, as Fields::tfsf_point).
static void rz_check_range(const Domain& d, const Config& c, const FieldsRZ& f, int& i0, int& iend, int& sp) {
    const int nz = d.n[0];
    i0 = rz_gauss_istart(d, c);
    iend = !d.periodic[0] && d.off[0] + nz == d.N[0] ? nz - 1 : nz;
    if (c.x_abs_cells > 0) {
        i0 = std::max(i0, c.x_abs_cells + 1 - d.off[0]);
        iend = std::min(iend, d.N[0] - c.x_abs_cells - 1 - d.off[0]);
    }
    if (f.diel && c.x_abs_cells == 0 && f.zabs_hi > 0) iend = std::min(iend, d.N[0] - f.zabs_hi - 1 - d.off[0]);   // the front layer
    sp = f.tfsf ? f.tfsf->plane() - d.off[0] - 1 : -(1 << 30);
}
// Per mode: W_m and the largest Gauss change (the points of rz_diagnostics), for dir/rz_modes.txt.
void Simulation::rz_mode_diagnostics(std::vector<double>& w, std::vector<double>& gauss) {
    const int M = cfg.modes, nz = dom.n[0], nr = dom.n[1];
    w.assign(size_t(M), 0.0); gauss.assign(size_t(M), 0.0);
    for (int m = 0; m < M; ++m) w[size_t(m)] = rz->energy_mode(m);
    comm.allreduce_sum(w.data(), M);
    rz_gauss_check(work_);
    int i0, iend, sp;
    rz_check_range(dom, cfg, *rz, i0, iend, sp);
    const int jend = cfg.rz_abs_cells > 0 ? nr - cfg.rz_abs_cells - 1 : nr;
    const FV W = work_.view(), G = gauss0_.view();
    const MetalView mv = rz->metal ? rz->metal->view() : MetalView{};   // the surface charge of the rings: not checked
    const MetalPhiView pv = rz->metal_phi ? rz->metal_phi->view() : MetalPhiView{};   // block metal: all angles
    const RadialView gv = rz->grid().view();          // numerics.rz_axis_cut: the rods of the modes are not checked
    const long c0 = dom.cell0(0);
    for (int m = 0; m < M; ++m) {
        const int k0 = m == 0 ? 0 : 2 * m - 1;
        gauss[size_t(m)] = comm.allreduce_max(par_max_abs(long(nz) * nr * (m ? 2 : 1), [=] HD(long q) {
            const int i = int(q % nz), j = int((q / nz) % nr), k = k0 + int(q / (long(nz) * nr));
            return (j == 0 && k > 0) || i < i0 || i >= iend || i == sp || j >= jend || mv.gauss_skip(c0 + i, j) || pv.gauss_skip(c0 + i, j) || j <= gv.jc((k + 1) / 2) ? 0.0 : W(i, j, k) - G(i, j, k);
        }));
    }
}

void Simulation::rz_diagnostics(double& wfield, double& gauss, double& rhomax, double& divb) {
    wfield = rz->energy();
    rz_gauss_check(work_);
    const int nz = dom.n[0], nr = dom.n[1];
    int i0, iend, sp;
    rz_check_range(dom, cfg, *rz, i0, iend, sp);
    const FV W = work_.view(), G = gauss0_.view(), R = rz_rho_.view();
    // with the radial absorber: only inside it (and one cell more), as with the Cart3D CPML
    const int jend = cfg.rz_abs_cells > 0 ? nr - cfg.rz_abs_cells - 1 : nr;
    const MetalView mv = rz->metal ? rz->metal->view() : MetalView{};   // the surface charge of the rings: not checked
    const MetalPhiView pv = rz->metal_phi ? rz->metal_phi->view() : MetalPhiView{};   // block metal: all angles
    const RadialView gv = rz->grid().view();          // numerics.rz_axis_cut: the rods of the modes are not checked
    const long c0 = dom.cell0(0);
    gauss = comm.allreduce_max(par_max_abs(dom.owned(), [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), k = int(q / (long(nz) * nr));
        return (j == 0 && k > 0) || i < i0 || i >= iend || i == sp || j >= jend || mv.gauss_skip(c0 + i, j) || pv.gauss_skip(c0 + i, j) || j <= gv.jc((k + 1) / 2) ? 0.0 : W(i, j, k) - G(i, j, k);
    }));
    rhomax = comm.allreduce_max(par_max_abs(dom.owned(), [=] HD(long q) {
        return R(int(q % nz), int((q / nz) % nr), int(q / (long(nz) * nr)));
    }));
    rz->div_b(work_);
    // div B: the z layers and the point on the TFSF plane left out (as Fields::max_div_b)
    const int ib0 = cfg.x_abs_cells > 0 ? cfg.x_abs_cells + 1 - dom.off[0] : 0;
    divb = comm.allreduce_max(par_max_abs(dom.owned(), [=] HD(long q) {
        const int i = int(q % nz), j = int((q / nz) % nr), k = int(q / (long(nz) * nr));
        return i < ib0 || i >= iend || i == sp || j >= jend || mv.divb_skip(c0 + i, j) || pv.divb_skip(c0 + i, j) || j <= gv.jc((k + 1) / 2) ? 0.0 : W(i, j, k);
    }));
}
