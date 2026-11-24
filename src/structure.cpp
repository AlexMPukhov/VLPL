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

void Config::check_structure() const {
    if (metal.on && solver != Solver::RipCC) throw std::runtime_error("metal (embedded conductor): numerics.solver = rip_cc only");
    if (!quads.empty() && (mover_implicit || boost.on))
        throw std::runtime_error("quadrupole: not with numerics.mover = implicit or the boosted frame");
    for (const SpeciesConfig& s : species)
        if (!s.file.path.empty() && boost.on) throw std::runtime_error("species " + s.name + ".file: not in the boosted frame");
}

void structure_summary(FILE* f, const Config& c) {
    if (c.metal.on) std::fprintf(f, "  metal: function = %s (> 0 inside)\n", c.metal.fn.text().c_str());
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

// the beam of species i from its file (all ranks read it, each keeps its own particles), then its self-field
void Simulation::load_file_species(size_t i) {
    SpeciesConfig& sc = cfg.species[i];
    const BeamParticles B = import_beam(sc.file, cfg.lambda0, cfg.seed + 977 * i);
    sc.q = B.sign;
    species[i].q = B.sign;
    double ux = 0, W = 0;
    for (size_t p = 0; p < B.size(); ++p) { ux += B.w[p] * B.ux[p]; W += B.w[p]; }
    sc.u0[0] = ux / W;
    if (comm.rank == 0) {
        const FileMoments M = file_moments(B, cfg.lambda0);
        std::printf("  species %s: %ld particles of %s (%s, species %s, iteration %ld): Q = %.6g pC, E = %.6g MeV, dE/E = %.4g,\n"
                    "    centroid %s = %.6g %.6g %.6g um, sigma %s = %.6g %.6g %.6g um, eps_n %s = %.6g %.6g um\n",
                    sc.name.c_str(), M.n, sc.file.path.c_str(), B.format.c_str(), B.species.c_str(), B.iteration, M.Q * 1e12,
                    M.ekin_mev, M.de_rel, cfg.rz ? "z x y" : "x y z", M.mean[0] * 1e6, M.mean[1] * 1e6, M.mean[2] * 1e6,
                    cfg.rz ? "z x y" : "x y z", M.sigma[0] * 1e6, M.sigma[1] * 1e6, M.sigma[2] * 1e6, cfg.rz ? "x y" : "y z",
                    M.eps_n[0] * 1e6, M.eps_n[1] * 1e6);
    }
    std::vector<double> h[Species::NC];
    long outside = 0;
    // RZ: inside the radial grid (the outer node) and the rank's z range (MPI along z only)
    const double rmax = cfg.rz ? rz->grid().r[size_t(dom.n[1])] : 0.0;
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
    species[i].append(h);
    if (comm.rank == 0 && outside > 0)
        std::fprintf(stderr, "warning: species %s: %ld of %zu particles of the file lie outside the box and are dropped\n",
                     sc.name.c_str(), outside, B.size());
}

double Simulation::metal_absorb_all() {
    double e = 0;
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
