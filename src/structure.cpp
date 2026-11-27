// Structure runs (docs/specs/structure3d.md, decision 102; RZ: decision 105): the deck blocks metal, quadrupole [<name>],
// quadrupole_lattice, wake [<name>], beam_slices [<name>], side_flux and the species key file, and their setup.
// RZ (particle axes (z, x, y), z the transport axis): quadrupole { z_start; length; gradient; x_center; y_center },
// quadrupole_lattice { file; z_offset; x_center; y_center }, wake { x; y } (default the axis), species { file; z_center;
// x_offset; y_offset }; metal is the RZ block of decision 103 (Config::metal_rz) and side_flux is Cart3D only.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "sim.hpp"
#include "beam_field.hpp"

namespace {
const double CL = 299792458.0, QE = 1.602176634e-19, ME = 9.1093837015e-31;
std::string unquote(std::string t) {
    if (t.size() >= 2 && t.front() == '"' && t.back() == '"') t = t.substr(1, t.size() - 2);
    return t;
}
std::string near_deck(const Deck& d, const std::string& p) {      // relative to the working directory or to the deck
    if (p.empty() || p[0] == '/' || std::ifstream(p)) return p;
    const size_t sl = d.file.rfind('/');
    if (sl != std::string::npos && std::ifstream(d.file.substr(0, sl + 1) + p)) return d.file.substr(0, sl + 1) + p;
    return p;
}
}  // namespace

bool metal_particles_absorb(const Deck& d, const std::string& block) {
    const std::string p = d.str(block, "particles", "pass");
    if (p != "pass" && p != "absorb") throw std::runtime_error(block + ".particles: pass or absorb");
    return p == "absorb";
}

// decision 124: the conductor of the block in an irip run, a Drude medium named after the block (the caller sets the profile):
// density = n / n_c (default: omega_p dt = 10, a skin depth of dx / 10: a mirror), collision_frequency = nu / omega0 (default 0)
MediumConfig metal_medium(const Deck& d, const std::string& block, double dt, bool absorb_ok) {
    if (!absorb_ok && metal_particles_absorb(d, block))
        throw std::runtime_error(block + ".particles = absorb with irip: Cart3D / 2D only so far (decision 124)");
    MediumConfig m;
    m.name = block;
    const double wd = 10.0 / dt;
    m.density = d.num(block, "density", Kind::None, wd * wd);
    m.nu = d.num(block, "collision_frequency", Kind::None, 0.0);
    if (!(m.density > 0) || m.nu < 0) throw std::runtime_error(block + ": density (n / n_c) > 0 and collision_frequency >= 0");
    return m;
}

void structure_from_deck(const Deck& d, Config& c, double k0) {
    // RZ: the block metal belongs to the RZ conductor (Config::metal_rz, parsed in Config::from_deck), not to this one
    const bool any = (!c.rz && d.has_block("metal")) || d.has_block("quadrupole_lattice") || d.has_block("side_flux");
    bool named = false;
    for (const std::string& b : d.block_names())
        named = named || b.rfind("quadrupole", 0) == 0 || b.rfind("wake", 0) == 0 || b.rfind("beam_slices", 0) == 0;
    if (!any && !named) return;
    if (c.rz && d.has_block("side_flux")) throw std::runtime_error("side_flux: Cart3D only");
    const UnitFn u = [k0](const std::string& n, double& f) { return unit_to_code(n, k0, f); };
    // the default transverse axis: the box centre (Cart3D), the axis r = 0 (RZ)
    const double ybox = c.rz ? 0.0 : 0.5 * c.N[1] * c.h[1], zbox = c.rz ? 0.0 : 0.5 * c.N[2] * c.h[2];
    if (!c.rz && d.has_block("metal")) {             // RZ: the metal block of decision 103 (Config::metal_rz)
        c.metal.on = d.flag("metal", "enabled", true);
        try { c.metal.fn = Expr(unquote(d.str("metal", "function")), {"x", "y", "z"}, u); }
        catch (const std::exception& ex) { throw std::runtime_error(std::string("metal.function: ") + ex.what()); }
        c.metal.absorb = metal_particles_absorb(d, "metal");
        if (c.metal.on && c.solver == Solver::IRip) {   // decision 124: in iRIP the conductor is a dense Drude medium
            MediumConfig m = metal_medium(d, "metal", c.dt(), true);
            try { m.profile = Expr("(" + unquote(d.str("metal", "function")) + ") > 0", {"x", "y", "z"}, u); }
            catch (const std::exception& ex) { throw std::runtime_error(std::string("metal.function: ") + ex.what()); }
            c.media.push_back(m);
            c.metal.medium = true; c.metal.density = m.density; c.metal.nu = m.nu;
            c.metal.on = false;                       // no staircase conductor: the medium block does the work (medium.hpp)
        }
    }
    // gradient [T/m] -> code: B in m omega0 / e = m c k0 / e per 1/k0
    const double gunit = ME * CL * k0 / QE * k0;
    for (const std::string& b : d.block_names()) {
        if (b != "quadrupole" && b.rfind("quadrupole ", 0) != 0) continue;
        Quad q;
        q.x0 = d.num(b, c.rz ? "z_start" : "x_start", Kind::Length) * k0;
        q.x1 = q.x0 + d.num(b, "length", Kind::Length) * k0;
        q.g = d.num(b, "gradient", Kind::None) / gunit;
        q.yc = d.num(b, c.rz ? "x_center" : "y_center", Kind::Length, ybox / k0) * k0;   // RZ: the particle axes (z, x, y)
        q.zc = d.num(b, c.rz ? "y_center" : "z_center", Kind::Length, zbox / k0) * k0;
        if (q.x1 <= q.x0) throw std::runtime_error(b + ".length must be > 0");
        c.quads.push_back(q);
    }
    if (d.has_block("quadrupole_lattice")) {             // lines: x_start [m]  length [m]  gradient [T/m]
        const std::string path = near_deck(d, unquote(d.str("quadrupole_lattice", "file")));
        const double x0 = d.num("quadrupole_lattice", c.rz ? "z_offset" : "x_offset", Kind::Length, 0.0) * k0;
        const double yc = d.num("quadrupole_lattice", c.rz ? "x_center" : "y_center", Kind::Length, ybox / k0) * k0;
        const double zc = d.num("quadrupole_lattice", c.rz ? "y_center" : "z_center", Kind::Length, zbox / k0) * k0;
        std::ifstream in(path);
        if (!in) throw std::runtime_error("quadrupole_lattice.file: cannot open " + path);
        std::string line;
        int n = 0;
        while (std::getline(in, line)) {
            ++n;
            const size_t h = line.find('#');
            if (h != std::string::npos) line.resize(h);
            std::istringstream is(line);
            double a, l, g;
            if (!(is >> a)) continue;
            if (!(is >> l >> g) || l <= 0) throw std::runtime_error(path + ":" + std::to_string(n) + ": x_start [m] length [m] gradient [T/m]");
            c.quads.push_back(Quad{x0 + a * k0, x0 + (a + l) * k0, g / gunit, yc, zc});
        }
    }
    for (const std::string& b : d.block_names()) {
        if (b == "wake" || b.rfind("wake ", 0) == 0) {
            WakeConfig w;
            w.name = b == "wake" ? "" : b.substr(5);
            if (c.rz) {                              // the probe (x, y) off the axis (RZ), default the axis
                w.y = d.num(b, "x", Kind::Length, 0.0) * k0;
                w.z = d.num(b, "y", Kind::Length, 0.0) * k0;
            } else {
                w.y = d.num(b, "y", Kind::Length, -1.0 / k0) * k0;
                w.z = d.num(b, "z", Kind::Length, -1.0 / k0) * k0;
            }
            w.start = d.num(b, "start", Kind::Time, 0.0) * CL * k0;
            w.every = d.integer(b, "every", c.every);
            c.wakes.push_back(w);
        } else if (b == "beam_slices" || b.rfind("beam_slices ", 0) == 0) {
            SliceDiagConfig s;
            s.name = b == "beam_slices" ? "" : b.substr(12);
            s.species = d.str(b, "species");
            s.dx = d.num(b, "dx", Kind::Length, 0.0) * k0;
            s.every = d.integer(b, "every", c.every);
            c.slice_diags.push_back(s);
        }
    }
    if (d.has_block("side_flux") && d.flag("side_flux", "enabled", true)) {
        SideFluxConfig& F = c.side_flux;
        F.on = true;
        const char* lo[3] = {"", "y_lo", "z_lo"};
        const char* hi[3] = {"", "y_hi", "z_hi"};
        for (int a = 1; a < 3; ++a) {
            F.lo[a] = d.num("side_flux", lo[a], Kind::Length, -1.0 / k0) * k0;
            F.hi[a] = d.num("side_flux", hi[a], Kind::Length, -1.0 / k0) * k0;
        }
        F.every = d.integer("side_flux", "every", c.every);
    }
}

// species with file = ... (beam_import.hpp): charge from the file (electrons or positrons), m = m_e, density 0 (no lattice
// load, nothing enters with the window)
void species_file_from_deck(const Deck& d, const std::string& b, const Config& c, SpeciesConfig& s, double k0) {
    BeamFileConfig& F = s.file;
    F.path = near_deck(d, unquote(d.str(b, "file")));
    F.species = d.str(b, "file_species", "");
    F.iteration = d.integer(b, "file_iteration", -1);
    if (c.rz) {                                      // particle axes (z, x, y): the file's (z, x, y) on the axis r = 0
        F.x_center = d.num(b, "z_center", Kind::Length) * k0;
        F.y_center = 0; F.z_center = 0;
        F.y_offset = d.num(b, "x_offset", Kind::Length, 0.0) * k0;
        F.z_offset = d.num(b, "y_offset", Kind::Length, 0.0) * k0;
    } else {
        F.x_center = d.num(b, "x_center", Kind::Length) * k0;
        F.y_center = d.num(b, "y_center", Kind::Length, 0.5 * c.N[1] * c.h[1] / k0) * k0;
        F.z_center = d.num(b, "z_center", Kind::Length, 0.5 * c.N[2] * c.h[2] / k0) * k0;
        F.y_offset = d.num(b, "y_offset", Kind::Length, 0.0) * k0;
        F.z_offset = d.num(b, "z_offset", Kind::Length, 0.0) * k0;
    }
    F.charge = d.num(b, "total_charge", Kind::Charge, 0.0);
    F.macroparticles = d.integer(b, "macroparticles", 0);
    F.smoothing = d.num(b, "resample_smoothing", Kind::None, 0.0);
    s.q = -1; s.m = 1; s.density = 0;
    for (int a = 0; a < 3; ++a) { s.rmin[a] = -1e300; s.rmax[a] = 1e300; }
    s.self_field = d.flag(b, "self_field", false);
    s.t_start = d.num(b, "t_start", Kind::Time, 0.0) * CL * k0;
    s.rigid_until = d.num(b, "rigid_until", Kind::Time, 0.0) * CL * k0;
    s.mobile = d.flag(b, "mobile", true);
}

// bunch <name> { ... } (docs/specs/deck_standard.md): a Gaussian bunch without a file, in the coordinates of the
// beam whatever the geometry of the run: z along the propagation (the x of a Cartesian run), x and y across it.
//   species = electron | positron, charge (total), energy (kinetic) or gamma, energy_spread (relative rms),
//   sigma_x sigma_y sigma_z, emittance_x emittance_y (normalized rms), z_center (the centroid at t = 0, lab),
//   x_offset y_offset (from the axis: r = 0 in RZ, the centre of the box in a Cartesian run), macroparticles,
//   and as for a file beam: self_field, t_start, rigid_until, mobile
void species_bunch_from_deck(const Deck& d, const std::string& b, const Config& c, SpeciesConfig& s, double k0) {
    BeamFileConfig& F = s.file;
    BunchShape& G = F.gen;
    G.on = true;
    F.path = "bunch " + s.name;
    const std::string kind = d.str(b, "species", "electron");
    if (kind != "electron" && kind != "positron")
        throw std::runtime_error(b + ".species: electron or positron (other particles: a species block with charge and mass)");
    G.sign = kind == "electron" ? -1 : 1;
    G.charge = d.num(b, "charge", Kind::Charge);
    if (d.has(b, "energy") == d.has(b, "gamma")) throw std::runtime_error(b + ": energy (kinetic, with its unit) or gamma (one of them)");
    G.gamma = d.has(b, "gamma") ? d.num(b, "gamma", Kind::None) : 1 + d.num(b, "energy", Kind::Energy) / 0.51099895000;
    G.de = d.num(b, "energy_spread", Kind::None, 0.0);
    // the beam's (z, x, y) are the run's (x, y, z)
    G.sigma[0] = d.num(b, "sigma_z", Kind::Length) * k0;
    G.sigma[1] = d.num(b, "sigma_x", Kind::Length) * k0;
    G.sigma[2] = d.num(b, "sigma_y", Kind::Length, G.sigma[1] / k0) * k0;
    G.eps[0] = d.num(b, "emittance_x", Kind::Length, 0.0) * k0;
    G.eps[1] = d.num(b, "emittance_y", Kind::Length, G.eps[0] / k0) * k0;
    G.n = d.integer(b, "macroparticles", 100000);
    if (!(G.charge > 0) || !(G.gamma > 1) || G.de < 0 || G.n < 1 || G.sigma[0] < 0 || G.sigma[1] < 0 || G.sigma[2] < 0 ||
        G.eps[0] < 0 || G.eps[1] < 0)
        throw std::runtime_error(b + ": charge > 0, energy above the rest energy, energy_spread, sigma_*, emittance_* >= 0, macroparticles >= 1");
    if ((G.eps[0] > 0 && G.sigma[1] == 0) || (G.eps[1] > 0 && G.sigma[2] == 0))
        throw std::runtime_error(b + ": an emittance needs a size (sigma_x, sigma_y > 0)");
    F.x_center = d.num(b, "z_center", Kind::Length) * k0;
    if (c.rz) { F.y_center = 0; F.z_center = 0; }
    else { F.y_center = 0.5 * c.N[1] * c.h[1]; F.z_center = 0.5 * c.N[2] * c.h[2]; }
    F.y_offset = d.num(b, "x_offset", Kind::Length, 0.0) * k0;
    F.z_offset = d.num(b, "y_offset", Kind::Length, 0.0) * k0;
    s.q = G.sign; s.m = 1; s.density = 0;
    for (int a = 0; a < 3; ++a) { s.rmin[a] = -1e300; s.rmax[a] = 1e300; }
    s.self_field = d.flag(b, "self_field", false);
    s.t_start = d.num(b, "t_start", Kind::Time, 0.0) * CL * k0;
    s.rigid_until = d.num(b, "rigid_until", Kind::Time, 0.0) * CL * k0;
    s.mobile = d.flag(b, "mobile", true);
}

void Config::check_structure() const {
    if (metal.on && solver != Solver::RipCC)
        throw std::runtime_error("metal (embedded conductor): numerics.solver = rip_cc (perfect conductor) or irip (a dense Drude medium)");
    if (metal.medium && metal.absorb && window.enabled)
        throw std::runtime_error("metal.particles = absorb with irip: not with a moving window yet (the absorbed charge is not shifted)");
    if (!quads.empty() && (mover_implicit || boost.on))
        throw std::runtime_error("quadrupole: not with numerics.mover = implicit or the boosted frame");
    for (const SpeciesConfig& s : species)
        if (!s.file.path.empty() && boost.on) throw std::runtime_error("species " + s.name + ".file: not in the boosted frame");
}

void structure_summary(FILE* f, const Config& c) {
    if (c.metal.medium)
        std::fprintf(f, "  metal: function = %s (> 0 inside); irip: a Drude medium, n = %g n_c (omega_p dt = %.3g), nu = %g omega0; "
                        "particles %s (decision 124)\n", c.metal.fn.text().c_str(), c.metal.density,
                     std::sqrt(c.metal.density) * c.dt(), c.metal.nu,
                     c.metal.absorb ? "absorbed inside, their charge stays at rest" : "pass through its fields");
    if (c.metal.on)
        std::fprintf(f, "  metal: function = %s (> 0 inside); particles %s\n", c.metal.fn.text().c_str(),
                     c.metal.absorb ? "absorbed inside (the conductor keeps their charge)" : "pass through (no field inside)");
    const double k0 = 2 * M_PI / c.lambda0, gunit = ME * CL * k0 / QE * k0;
    if (!c.quads.empty()) {
        if (c.rz) std::fprintf(f, "  quadrupoles: %zu (B_x = -G (y - y_c), B_y = -G (x - x_c), G > 0 focuses electrons in x)\n", c.quads.size());
        else std::fprintf(f, "  quadrupoles: %zu (B_y = -G (z - z_c), B_z = -G (y - y_c), G > 0 focuses electrons in y)\n", c.quads.size());
        for (size_t i = 0; i < c.quads.size() && i < 8; ++i)
            std::fprintf(f, "    %s = %.6g .. %.6g mm, G = %.6g T/m\n", c.rz ? "z" : "x", c.quads[i].x0 / k0 * 1e3,
                         c.quads[i].x1 / k0 * 1e3, c.quads[i].g * gunit);
        if (c.quads.size() > 8) std::fprintf(f, "    ...\n");
    }
    for (const SpeciesConfig& s : c.species)
        if (!s.file.path.empty())
            std::fprintf(f, "  species %s from %s (centroid at %s = %.6g um)%s\n", s.name.c_str(), s.file.path.c_str(), c.rz ? "z" : "x",
                         s.file.x_center / k0 * 1e6, s.self_field ? ", self_field" : "");
}

// ------------------------------------------------------------------------------------------------ Simulation side
void Simulation::setup_structure() {
    set_quadrupoles(cfg.quads);
    if (cfg.metal.medium && cfg.metal.absorb) {      // irip (decision 124): the node mask of the medium metal for the absorption
        metal_mask_ = std::make_unique<Metal3>(dom, cfg.metal);
        rho_abs_.resize(dom);
    }
    if (!cfg.metal.on) return;
    metal3 = std::make_unique<Metal3>(dom, cfg.metal);
    fields->metal = metal3.get();
    double v[2] = {double(metal3->owned_nodes(dom)), double(metal3->thin_edges(dom))};
    comm.allreduce_sum(v, 2);
    if (comm.rank == 0) {
        std::printf("  metal: %.0f nodes inside at the start\n", v[0]);
        if (v[1] > 0)
            std::fprintf(stderr, "warning: metal: %.0f edges of metal one cell thick along x (they are set to 0 and absorb instead "
                                 "of reflecting): make the metal at least two cells thick along x\n", v[1]);
    }
}

// the beam of species i from its file (all ranks read it in portions of 2^18 particles) or from its bunch block (all
// ranks draw the same particles in portions); each rank keeps its own particles: no rank holds the whole beam
void Simulation::load_file_species(size_t i) {
    SpeciesConfig& sc = cfg.species[i];
    std::vector<double> h[Species::NC];
    long outside = 0, total = 0;
    // RZ: inside the radial grid (the outer node) and the rank's z range (MPI along z only)
    const double rmax = cfg.rz ? rz->grid().r[size_t(dom.n[1])] : 0.0;
    const auto take = [&](const BeamParticles& B) {
    total += long(B.size());
    for (size_t p = 0; p < B.size(); ++p) {
        const double r[3] = {B.x[p], B.y[p], B.z[p]};
        bool in = true, inside_box = true;
        if (cfg.rz) {
            const double l = double(dom.wx) * dom.h[0], rr = std::sqrt(r[1] * r[1] + r[2] * r[2]);
            in = r[0] >= dom.lo(0) && r[0] < dom.hi(0) && rr < rmax;
            inside_box = r[0] >= l && r[0] < l + dom.L(0) && rr < rmax;
            if (!inside_box) ++outside;
            if (!in) continue;
            const double v[Species::NC] = {B.x[p], B.y[p], B.z[p], B.ux[p], B.uy[p], B.uz[p], B.w[p]};
            for (int c = 0; c < Species::NC; ++c) h[c].push_back(v[c]);
            continue;
        }
        for (int a = 0; a < 3; ++a) {
            if (a == 2 && dom.flat) continue;
            const double l = a == 0 ? double(dom.wx) * dom.h[0] : 0.0;   // the box (the window in x)
            in = in && r[a] >= dom.lo(a) && r[a] < dom.hi(a);
            inside_box = inside_box && r[a] >= l && r[a] < l + dom.L(a);
        }
        if (!inside_box) ++outside;
        if (!in) continue;
        const double v[Species::NC] = {B.x[p], B.y[p], dom.flat ? dom.lo(2) + 0.5 * dom.h[2] : B.z[p], B.ux[p], B.uy[p], B.uz[p], B.w[p]};
        for (int c = 0; c < Species::NC; ++c) h[c].push_back(v[c]);
    }
    };
    FileMoments M;
    std::string source;
    if (sc.file.gen.on) {
        BunchSource S(sc.file, cfg.lambda0, cfg.seed + 977 * i);
        const double centre[3] = {sc.file.x_center, sc.file.y_center + sc.file.y_offset, sc.file.z_center + sc.file.z_offset};
        MomentSums sums(centre, sc.file.gen.gamma);
        while (S.left() > 0) {
            const BeamParticles B = S.next(1L << 18);
            sums.add(B);
            take(B);
        }
        sc.q = sc.file.gen.sign;
        sc.u0[0] = sums.mean_ux();
        M = sums.result(cfg.lambda0);
        source = std::string("the bunch block (generated, ") + (sc.q < 0 ? "electrons" : "positrons") + ")";
    } else {
        BeamFileSource S(sc.file, cfg.lambda0, cfg.seed + 977 * i);   // reads the file in portions
        const double centre[3] = {sc.file.x_center, sc.file.y_center + sc.file.y_offset, sc.file.z_center + sc.file.z_offset};
        MomentSums sums(centre, S.gamma_mean());
        while (S.left() > 0) {
            const BeamParticles B = S.next(1L << 18);
            sums.add(B);
            take(B);
        }
        sc.q = S.sign;
        sc.u0[0] = sums.mean_ux();
        M = sums.result(cfg.lambda0);
        source = sc.file.path + " (" + S.format + ", species " + S.species + ", iteration " + std::to_string(S.iteration) + ")";
    }
    species[i].q = sc.q;
    if (comm.rank == 0)
        std::printf("  species %s: %ld particles of %s: Q = %.6g pC, E = %.6g MeV, dE/E = %.4g,\n"
                    "    centroid %s = %.6g %.6g %.6g um, sigma %s = %.6g %.6g %.6g um, eps_n %s = %.6g %.6g um\n",
                    sc.name.c_str(), M.n, source.c_str(), M.Q * 1e12,
                    M.ekin_mev, M.de_rel, cfg.rz ? "z x y" : "x y z", M.mean[0] * 1e6, M.mean[1] * 1e6, M.mean[2] * 1e6,
                    cfg.rz ? "z x y" : "x y z", M.sigma[0] * 1e6, M.sigma[1] * 1e6, M.sigma[2] * 1e6, cfg.rz ? "x y" : "y z",
                    M.eps_n[0] * 1e6, M.eps_n[1] * 1e6);
    species[i].append(h);
    if (comm.rank == 0 && outside > 0)
        std::fprintf(stderr, "warning: species %s: %ld of %ld particles of the %s lie outside the box and are dropped\n",
                     sc.name.c_str(), outside, total, sc.file.gen.on ? "bunch" : "file");
}

// irip (decision 124): the metal is a medium. A particle whose shape lies inside it is removed and its charge stays at rest:
// the charge density it deposited is kept in rho_abs_ (owned cells, guards reduced into their owners and zero: the checkpoint
// holds it as it is and the restart continues bit for bit), added to rho in compute_rho, so the Gauss law with the medium
// charge holds as before; the medium screens it as it would the particle. Two extra charge deposits of a species only in a
// step in which it loses particles.
static double metal_absorb_medium(Simulation& s) {
    double e = 0;
    Field& tmp = s.absorb_tmp_;
    Field& tmp2 = s.absorb_tmp2_;
    if (tmp.n(0) == 0) { tmp.resize(s.dom); tmp2.resize(s.dom); }
    const int n0 = s.dom.n[0], n1 = s.dom.n[1], n2 = s.dom.n[2];
    for (size_t i = 0; i < s.species.size(); ++i) {
        Species& sp = s.species[i];
        long m = 0;
        const double de = metal_absorb(sp, s.dom, s.metal_mask_->view(), s.cfg.order, false, &m);
        const bool charged = sp.q != 0;
        // the exchange of the guard cells below is collective: all ranks do it as soon as one of them absorbed a particle
        // (a rank that skipped it left the others waiting: the run on several ranks hung)
        double mall = double(m);
        if (charged) s.comm.allreduce_sum(&mall, 1);
        if (mall == 0) continue;
        if (charged) {                               // + the deposit with the particles, - the deposit without them
            tmp.fill(0);
            if (m > 0) deposit_rho(sp, tmp, s.dom, s.cfg.order);
        }
        if (m > 0) metal_absorb(sp, s.dom, s.metal_mask_->view(), s.cfg.order, true);   // flags again (the deposit may use the scratch), remove
        if (charged) {
            const FV A = s.rho_abs_.view(), T = tmp.view(), T2 = tmp2.view();
            if (m > 0) {
                tmp2.fill(0);
                deposit_rho(sp, tmp2, s.dom, s.cfg.order);
                par_for(long(tmp.size()), [=] HD(long q) { T.p[q] -= T2.p[q]; });
            }
            s.comm.add_guards(s.dom, {&tmp});        // the removed charge of the guard cells to its owners (also across the ranks)
            par_for(long(n0) * n1 * n2, [=] HD(long q) {
                const int i = int(q % n0), j = int((q / n0) % n1), k = int(q / (long(n0) * n1));
                A(i, j, k) += T(i, j, k);
            });
        }
        sp.lost_energy += de;
        e += de;
    }
    return e;
}

double Simulation::metal_absorb_all() {
    double e = 0;
    if (metal_mask_) return metal_absorb_medium(*this);
    if (!metal3) return 0;
    for (size_t i = 0; i < species.size(); ++i) {
        const double de = metal_absorb(species[i], dom, metal3->view(), cfg.order);
        species[i].lost_energy += de;
        e += de;
    }
    return e;
}

namespace {
// stage Migrate (before the migration): the particles that moved into the metal are absorbed
class MetalAbsorb : public Operator {
public:
    const char* name() const override { return "metal_absorb"; }
    Stage stage() const override { return Stage::Migrate; }
    void apply(Simulation& s) override { s.metal_absorb_all(); }
};
}  // namespace

std::unique_ptr<Operator> make_metal_absorb_operator() { return std::make_unique<MetalAbsorb>(); }

void Simulation::structure_operators() {
    for (const WakeConfig& w : cfg.wakes) pipeline.add(make_wake_operator(w));
    for (const SliceDiagConfig& sl : cfg.slice_diags) pipeline.add(make_slices_operator(sl));
    if (cfg.side_flux.on) pipeline.add(make_side_flux_operator(cfg.side_flux));
}
