#include "sim.hpp"
#include "prof.hpp"
#include "env.hpp"
#include "filter.hpp"
#include "implicit_mover.hpp"
#include "fsutil.hpp"
#include "beam_field.hpp"
#include "load.hpp"
#include "atomic.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

static const double c_light = 299792458.0;

// Charge and mass of a species block, and the keys of field ionization (ionization.hpp, docs/specs/field_ionization.md):
// element (mass and NIST energies), charge_state (the charge without ionization, the initial state with it), ionization = adk.
static void species_charge(const Deck& d, const std::string& b, SpeciesConfig& s) {
    const Element* el = nullptr;
    if (d.has(b, "element")) {
        el = find_element(d.str(b, "element"));
        if (!el) throw std::runtime_error(b + ".element: unknown element " + d.str(b, "element") + " (data/ionization_energies.def)");
    }
    const std::string model = d.str(b, "ionization", "none");
    if (model != "none" && model != "adk") throw std::runtime_error(b + ".ionization: none or adk");
    IonizationConfig& I = s.ion;
    I.on = model == "adk";
    const double u_me = 1822.888486209;                  // atomic mass unit / m_e (CODATA 2018)
    s.m = el && !d.has(b, "mass") ? el->mass_u * u_me : d.num(b, "mass", Kind::None);
    const long z0 = d.integer(b, "charge_state", -1);
    if (!I.on) {
        for (const char* k : {"max_charge_state", "ionization_potentials", "ionization_l", "ionization_electrons", "ionization_current",
                              "ionization_bsi", "ionization_tl_alpha", "ionization_m"})
            if (d.has(b, k)) throw std::runtime_error(b + "." + k + ": needs ionization = adk");
        if (z0 >= 0) {
            if (d.has(b, "charge")) throw std::runtime_error(b + ": charge or charge_state, not both");
            s.q = double(z0);
        } else s.q = d.num(b, "charge", Kind::None);
        if (el) {                                      // kept for impact ionization (collisions with ionizing = true)
            I.element = el->symbol; I.ip_eV = el->ip_eV; I.l = el->l;
            I.z0 = z0 < 0 ? -1 : int(z0); I.zmax = int(el->ip_eV.size());
        }
        return;
    }
    if (d.has(b, "charge") && d.num(b, "charge", Kind::None) != 1.0)
        throw std::runtime_error(b + ".charge: with ionization the charge is charge_state times e (leave charge out)");
    s.q = 1;
    I.element = el ? el->symbol : "";
    I.z0 = z0 < 0 ? 0 : int(z0);
    if (el) { I.ip_eV = el->ip_eV; I.l = el->l; }
    auto list = [&](const char* k) {                     // "552.07, 667.05 eV": numbers, the unit eV optional
        std::string t = d.str(b, k);
        for (char& ch : t) if (ch == ',') ch = ' ';
        std::istringstream is(t);
        std::vector<double> v;
        for (std::string w; is >> w;) {
            if (w == "eV") continue;
            if (w.size() > 2 && w.compare(w.size() - 2, 2, "eV") == 0) w.resize(w.size() - 2);
            try { v.push_back(std::stod(w)); } catch (...) { throw std::runtime_error(b + "." + k + ": not a number: " + w); }
        }
        return v;
    };
    if (d.has(b, "ionization_potentials")) {
        I.ip_eV = list("ionization_potentials");
        if (!el) I.l.clear();
    }
    if (d.has(b, "ionization_l")) {
        I.l.clear();
        for (double x : list("ionization_l")) I.l.push_back(int(x));
    }
    if (I.ip_eV.empty()) throw std::runtime_error(b + ": ionization = adk needs element or ionization_potentials");
    I.l.resize(I.ip_eV.size(), 0);
    I.zmax = int(d.integer(b, "max_charge_state", long(I.ip_eV.size())));
    if (I.zmax > int(I.ip_eV.size()) || I.zmax < I.z0)
        throw std::runtime_error(b + ": charge_state <= max_charge_state <= the number of ionization potentials (" + std::to_string(I.ip_eV.size()) + ")");
    for (int k = I.z0; k < I.zmax; ++k)
        if (!(I.ip_eV[size_t(k)] > 0) || I.l[size_t(k)] < 0) throw std::runtime_error(b + ": ionization potentials > 0 and l >= 0");
    I.m = int(d.integer(b, "ionization_m", 0));
    const std::string bsi = d.str(b, "ionization_bsi", "none");
    if (bsi != "none" && bsi != "tong_lin") throw std::runtime_error(b + ".ionization_bsi: none or tong_lin");
    I.tong_lin = bsi == "tong_lin";
    I.tl_alpha = d.num(b, "ionization_tl_alpha", Kind::None, 6.0);
    I.electrons = d.str(b, "ionization_electrons");
    I.current = d.flag(b, "ionization_current", true);
}

Config Config::from_deck(const Deck& d) {
    Config c;
    c.lambda0 = d.num("grid", "lambda0", Kind::Length);
    const double k0 = 2 * M_PI / c.lambda0;
    const char* ax = "xyz";
    const std::string geom = d.str("grid", "geometry", "cartesian");
    if (geom != "cartesian" && geom != "cartesian2d" && geom != "rz") throw std::runtime_error("grid.geometry: cartesian, cartesian2d or rz");
    c.rz = geom == "rz";
    c.flat = geom == "cartesian2d";
    if (c.rz) {
        // docs/specs/rz_modes.md: z the transport axis, periodic so far (open z, laser, window: step 3)
        c.modes = int(d.integer("grid", "modes", 1));
        if (c.modes < 1) throw std::runtime_error("grid.modes must be >= 1");
        c.N[0] = int(d.integer("grid", "nz", 0)); c.N[1] = int(d.integer("grid", "nr", 0)); c.N[2] = 2 * c.modes - 1;
        c.h[0] = d.num("grid", "dz", Kind::Length) * k0; c.h[1] = d.num("grid", "dr", Kind::Length) * k0; c.h[2] = 1;
        if (c.N[0] < 1 || c.N[1] < 2) throw std::runtime_error("grid.nz >= 1 and grid.nr >= 2 (geometry = rz)");
        const std::string bz = d.str("grid", "boundary_z", "periodic");
        if (bz != "periodic" && bz != "open" && bz != "absorbing") throw std::runtime_error("grid.boundary_z: periodic, open or absorbing");
        if (bz == "absorbing") {                   // open with damping layers at both z ends, as the x layers of Cart3D (O3)
            const double w = d.num("grid", "z_absorber_width", Kind::Length, 4 * c.lambda0) * k0;
            c.x_abs_cells = d.has("grid", "z_absorber_cells") ? int(d.integer("grid", "z_absorber_cells", 0))
                                                               : int(std::ceil(w / c.h[0] - 1e-9));
            c.x_abs_R0 = d.num("grid", "z_absorber_R0", Kind::None, 1e-4);
            if (c.x_abs_cells < 2 || 3 * c.x_abs_cells > c.N[0] || !(c.x_abs_R0 > 0 && c.x_abs_R0 < 1))
                throw std::runtime_error("grid.z_absorber_cells (or z_absorber_width / dz) in [2, nz/3], grid.z_absorber_R0 in (0, 1)");
        }
        const std::string br = d.str("grid", "boundary_r", "wall");
        if (br != "wall" && br != "absorbing") throw std::runtime_error("grid.boundary_r: wall (conducting) or absorbing");
        if (br == "absorbing") {
            c.rz_abs_cells = int(d.integer("grid", "r_absorber_cells", 32));
            c.rz_abs_R0 = d.num("grid", "r_absorber_R0", Kind::None, 1e-4);
            if (c.rz_abs_cells < 2 || c.rz_abs_cells > c.N[1] / 2 || !(c.rz_abs_R0 > 0 && c.rz_abs_R0 < 1))
                throw std::runtime_error("grid.r_absorber_cells in [2, nr/2], grid.r_absorber_R0 in (0, 1)");
        }
        c.rz_stretch = d.num("grid", "stretch", Kind::None, 1.0);
        c.rz_r_uniform = d.num("grid", "r_uniform", Kind::Length, 0.0) * k0;
        c.rz_dr_max = d.num("grid", "dr_max", Kind::Length, 0.0) * k0;
        c.periodic[0] = bz == "periodic"; c.periodic[1] = c.periodic[2] = false;
        c.mpi[0] = int(d.integer("grid", "mpi_z", 0)); c.mpi[1] = c.mpi[2] = 1;
    }
    for (int a = 0; a < 3 && !c.rz; ++a) {
        const std::string s(1, ax[a]);
        if (a == 2 && c.flat) {                    // 2D (docs/specs/cart2d.md): one periodic z cell, the slab thickness dz
            if (d.integer("grid", "nz", 1) != 1 || d.integer("grid", "mpi_z", 1) > 1 || d.str("grid", "boundary_z", "periodic") != "periodic")
                throw std::runtime_error("grid.geometry = cartesian2d: no nz, mpi_z or boundary_z (one periodic z cell)");
            c.N[2] = 1; c.h[2] = d.num("grid", "dz", Kind::Length, 1e-6) * k0;
            c.periodic[2] = true; c.mpi[2] = 1;
            continue;
        }
        c.N[a] = int(d.integer("grid", "n" + s, 0));
        c.h[a] = d.num("grid", "d" + s, Kind::Length) * k0;
        const std::string b = d.str("grid", "boundary_" + s, "periodic");
        if (a == 0 && b != "periodic" && b != "open" && b != "absorbing")
            throw std::runtime_error("grid.boundary_x: periodic, open or absorbing");
        if (a > 0 && b != "periodic" && b != "pml") throw std::runtime_error("grid.boundary_" + s + ": periodic or pml");
        c.periodic[a] = b == "periodic";
        c.pml.on[a] = b == "pml";
        if (a == 0 && b == "absorbing") {          // open, with damping layers at both ends (O3, docs/specs/x_absorber.md)
            // the thickness: x_absorber_cells, or x_absorber_width (default 4 lambda0: 45 deg reflects 6e-4 in field)
            const double w = d.num("grid", "x_absorber_width", Kind::Length, 4 * c.lambda0) * k0;
            c.x_abs_cells = d.has("grid", "x_absorber_cells") ? int(d.integer("grid", "x_absorber_cells", 0))
                                                               : int(std::ceil(w / c.h[0] - 1e-9));
            c.x_abs_R0 = d.num("grid", "x_absorber_R0", Kind::None, 1e-4);
            if (c.x_abs_cells < 2 || 3 * c.x_abs_cells > c.N[0] || !(c.x_abs_R0 > 0 && c.x_abs_R0 < 1))
                throw std::runtime_error("grid.x_absorber_cells (or x_absorber_width / dx) in [2, nx/3], grid.x_absorber_R0 in (0, 1)");
        }
        c.mpi[a] = int(d.integer("grid", "mpi_" + s, 0));
        if (c.N[a] < 1) throw std::runtime_error("grid.n" + s + " must be >= 1");
    }
    c.pml.cells = int(d.integer("pml", "cells", 16));
    {                                               // decision 111 (2026-11-20); default conserve: decision 115 (2026-11-22)
        const std::string ap = d.str("grid", "absorber_particles", "conserve");
        if (ap != "remove" && ap != "conserve" && ap != "keep") throw std::runtime_error("grid.absorber_particles: remove, conserve or keep");
        c.absorber_mode = ap == "remove" ? ABSORBER_REMOVE : (ap == "conserve" ? ABSORBER_CONSERVE : ABSORBER_KEEP);
    }
    c.pml.R0 = d.num("pml", "R0", Kind::None, 1e-6);
    c.pml.order = int(d.integer("pml", "order", 3));
    c.t_end = d.num("grid", "t_end", Kind::Time) * c_light * k0;
    c.order = int(d.integer("numerics", "shape", 1));
    c.filter_passes = int(d.integer("numerics", "filter_passes", 0));
    {
        const std::string fk = d.str("numerics", "filter", "binomial");   // decision 91 (projection before)
        if (fk != "projection" && fk != "local" && fk != "binomial")
            throw std::runtime_error("numerics.filter: projection, local or binomial");
        c.filter_local = fk == "local";
        c.filter_binomial = fk == "binomial";
        c.filter_comp = d.flag("numerics", "filter_compensate", false);
        const std::string ax = d.str("numerics", "filter_axes", "all");
        if (ax != "all" && ax != "x") throw std::runtime_error("numerics.filter_axes: all or x (the streaming axis: z in RZ)");
        c.filter_xonly = ax == "x";
        c.filter_flat = int(d.integer("numerics", "filter_flat", 0));
    }
    {
        const std::string bi = d.str("numerics", "beam_init", "steady");   // decision 99
        if (bi != "steady" && bi != "gauss") throw std::runtime_error("numerics.beam_init: steady or gauss");
        c.beam_steady = bi == "steady";
    }
    {
        const std::string sv = d.str("numerics", "solver", "rip_cc");
        if (sv == "rip_cc") c.solver = Solver::RipCC;
        else if (sv == "yee") c.solver = Solver::Yee;
        else if (sv == "irip") c.solver = Solver::IRip;
        else throw std::runtime_error("numerics.solver: rip_cc, yee or irip");
        if (c.solver == Solver::Yee) c.cfl = d.num("numerics", "cfl", Kind::None, 0.98);
        else if (d.has("numerics", "cfl")) throw std::runtime_error("numerics.cfl applies to solver = yee only (RIP-CC, iRIP: c dt = dx)");
        const std::string cl = d.str("numerics", "closure", "box");
        if (cl != "box" && cl != "o5") throw std::runtime_error("numerics.closure: box or o5");
        c.o5 = cl == "o5";
        const std::string rg = d.str("numerics", "rz_gather", "plain");     // decision 106: plain by default (matched: dense targets, decision 80)
        if (rg != "plain" && rg != "matched" && rg != "energy") throw std::runtime_error("numerics.rz_gather: plain, matched or energy");
        c.rz_gather_matched = rg == "matched" || rg == "energy";
        c.rz_gather_energy = rg == "energy";
        const std::string am = d.str("numerics", "rz_axis_map", "m0");       // decision 106: the density-conserving map for m = 0 only
        if (am != "m0" && am != "all") throw std::runtime_error("numerics.rz_axis_map: m0 or all");
        c.rz_map_all = am == "all";
        const std::string rdp = d.str("numerics", "rz_deposit", "phase");   // 2026-11-13: chord = the axis-smooth splits
        if (rdp != "phase" && rdp != "chord") throw std::runtime_error("numerics.rz_deposit: phase or chord");
        c.rz_deposit_chord = rdp == "chord";
        const std::string ts = d.str("numerics", "rz_transverse_shape", "none");   // decision 79 (none: the default)
        if (ts != "gaussian" && ts != "none") throw std::runtime_error("numerics.rz_transverse_shape: gaussian or none");
        c.rz_shape_width = ts == "none" ? 0.0 : d.num("numerics", "rz_shape_width", Kind::None, 1.0);
        if (c.rz_shape_width < 0) throw std::runtime_error("numerics.rz_shape_width >= 0 (in units of dr)");
        c.rz_axis_cut = d.num("numerics", "rz_axis_cut", Kind::None, 0.0);
        const std::string rac = d.str("numerics", "rz_axis_current", "regular");   // decision 112
        if (rac != "regular" && rac != "raw") throw std::runtime_error("numerics.rz_axis_current: regular or raw");
        c.rz_axis_regular = rac == "regular";
        if (c.rz_axis_cut < 0) throw std::runtime_error("numerics.rz_axis_cut >= 0 (0: off)");
        if (c.o5 && c.solver != Solver::IRip) throw std::runtime_error("numerics.closure = o5: solver = irip only");
    }
    const std::string p = d.str("numerics", "pusher", "higuera_cary");
    if (p == "boris") c.pusher = Pusher::Boris;
    else if (p == "higuera_cary") c.pusher = Pusher::HigueraCary;
    else throw std::runtime_error("numerics.pusher: boris or higuera_cary");
    c.seed = (unsigned long)d.integer("numerics", "seed", 1);
    {
        const std::string mv = d.str("numerics", "mover", "explicit");
        if (mv != "explicit" && mv != "implicit") throw std::runtime_error("numerics.mover: explicit or implicit");
        c.mover_implicit = mv == "implicit";
        c.mover_tol = d.num("numerics", "mover_tol", Kind::None, 1e-6);
        c.mover_maxit = int(d.integer("numerics", "mover_maxit", 100));
        c.mover_anderson = int(d.integer("numerics", "mover_anderson", 5));
        c.medium_tol = d.num("numerics", "medium_tol", Kind::None, 1e-10);
        c.medium_maxit = int(d.integer("numerics", "medium_maxit", 200));
        c.medium_bound_tol = d.num("numerics", "medium_bound_tol", Kind::None, 1e-8);
        c.medium_bound_maxit = int(d.integer("numerics", "medium_bound_maxit", 30));
        c.medium_restart = int(d.integer("numerics", "medium_restart", 30));
    }
    for (const std::string& b : d.block_names())     // O7: linear media (medium.hpp, decision 84)
        if ((b == "medium" || b.rfind("medium ", 0) == 0) && d.flag(b, "enabled", true)) c.media.push_back(medium_block(d, b, k0, c.rz));
    {
        std::istringstream is(d.str("numerics", "tile", "8 8 8"));
        for (int a = 0; a < 3; ++a)
            if (!(is >> c.tile[a]) || c.tile[a] < 1) throw std::runtime_error("numerics.tile: three cell counts >= 1");
    }
    {
        const std::string se = d.str("numerics", "sort_every", "auto");
        if (se == "auto") c.sort_every = -1;
        else {
            std::istringstream is(se);
            if (!(is >> c.sort_every) || c.sort_every < 0)
                throw std::runtime_error("numerics.sort_every: auto or a number of steps >= 0");
        }
    }
    {
        const std::string dep = d.str("numerics", "deposit", "auto");
        if (dep == "auto") c.deposit = Config::Deposit::Auto;
        else if (dep == "global") c.deposit = Config::Deposit::Global;
        else if (dep == "tiled") c.deposit = Config::Deposit::Tiled;
        else if (dep == "box") c.deposit = Config::Deposit::Box;
        else throw std::runtime_error("numerics.deposit: auto, global, tiled or box");
    }
    c.dir = d.str("output", "dir", "out");
    c.every = d.integer("output", "every", 10);
    c.slice_every = d.integer("output", "slice_every", 0);
    c.slice_z = d.num("output", "slice_z", Kind::Length, -1.0 / k0) * k0;
    {
        std::istringstream is(d.str("output", "slice_fields", "Ex Ey Ez Bx By Bz"));
        for (std::string w; is >> w;) c.slice_fields.push_back(w);
    }
    c.openpmd.every = d.integer("output", "openpmd_every", 0);
    c.xflux.every = d.integer("output", "xflux_every", 0);
    c.xflux.on = d.flag("output", "xflux", false) || c.xflux.every > 0;
    // RZ: the field crossing a lab plane z = plane_z every step, Ez on the axis every axis_every steps (docs/specs/rz_metal.md)
    c.plane_z = d.num("output", "plane_z", Kind::Length, -1.0 / k0) * k0;
    c.axis_every = d.integer("output", "axis_every", 0);
    if (d.has_block("rings")) {                      // embedded metal rings in RZ (docs/specs/rz_metal.md)
        c.rings.on = true;
        c.rings.r_in = d.num("rings", "r_in", Kind::Length) * k0;
        c.rings.r_out = d.num("rings", "r_out", Kind::Length) * k0;
        c.rings.period = d.num("rings", "period", Kind::Length) * k0;
        c.rings.gap = d.num("rings", "gap", Kind::Length) * k0;
        c.rings.z_start = d.num("rings", "z_start", Kind::Length) * k0;
        c.rings.z_end = d.num("rings", "z_end", Kind::Length) * k0;
        c.rings.jacket = d.num("rings", "jacket", Kind::Length, 0.0) * k0;   // closed corrugated pipe, open end at z_end
        c.rings.absorb = metal_particles_absorb(d, "rings");
    }
    if (c.rz && d.has_block("metal")) {             // conductors depending on phi in RZ (rz_metal_phi.md; Cart3D: structure.cpp)
        std::string t = d.str("metal", "function");
        if (t.size() >= 2 && t.front() == '"' && t.back() == '"') t = t.substr(1, t.size() - 2);
        const UnitFn u = [k0](const std::string& n, double& f) { return unit_to_code(n, k0, f); };
        try { c.metal_rz.fn = Expr(t, {"x", "y", "z", "r", "phi"}, u); }
        catch (const std::exception& ex) { throw std::runtime_error(std::string("metal.function: ") + ex.what()); }
        c.metal_rz.text = t;
        const std::string pd = d.str("metal", "phi_derivative", "difference");
        if (pd != "difference" && pd != "spectral") throw std::runtime_error("metal.phi_derivative: difference or spectral");
        c.metal_rz.spectral = pd == "spectral";
        c.metal_rz.absorb = metal_particles_absorb(d, "metal");
        c.metal_rz.on = d.flag("metal", "enabled", true);   // the keys are read (and checked) also when disabled
    }
    if (c.metal_rz.on && c.rings.on) throw std::runtime_error("RZ: the metal block and the rings block cannot be combined");
    if (c.rz && c.solver == Solver::IRip && (c.rings.on || c.metal_rz.on)) {   // decision 124: the conductor as a Drude medium
        const UnitFn u = [k0](const std::string& n, double& f) { return unit_to_code(n, k0, f); };
        MediumConfig m = metal_medium(d, c.rings.on ? "rings" : "metal", c.dt());
        if (c.rings.on) {                            // the rings (and the jacket) as an expression of z, r (code units)
            const RingsConfig& R = c.rings;
            if (!(R.period > 0) || R.gap < 0 || R.gap >= R.period) throw std::runtime_error("rings: period > 0 and 0 <= gap < period");
            char t[640];
            std::snprintf(t, sizeof t,
                          "z >= %.17g && z < %.17g && ((r > %.17g && r < %.17g && (z - %.17g) - %.17g * floor((z - %.17g) / %.17g) < %.17g)"
                          " || (r >= %.17g && r < %.17g))",
                          R.z_start, R.z_end, R.r_in, R.r_out, R.z_start, R.period, R.z_start, R.period,
                          R.gap == 0 ? 2 * R.period : R.period - R.gap, R.r_out, R.r_out + R.jacket);
            m.profile = Expr(t, {"z", "r"}, u);
        } else {                                     // one coefficient for all modes: an axisymmetric function only
            try { m.profile = Expr("(" + c.metal_rz.text + ") > 0", {"z", "r"}, u); }
            catch (const std::exception& ex) {
                throw std::runtime_error(std::string("metal.function with numerics.solver = irip: a function of z and r only (the medium "
                                                     "is axisymmetric, decision 124): ") + ex.what());
            }
        }
        c.media.push_back(m);
        c.rings.on = false; c.metal_rz.on = false;
    }
    for (const std::string& b : d.block_names()) {   // dielectric regions in RZ (rz_dielectric.hpp, decision 100)
        if (b != "dielectric" && b.rfind("dielectric ", 0) != 0) continue;
        if (!d.flag(b, "enabled", true)) continue;
        DielectricConfig e;
        e.name = b == "dielectric" ? "" : b.substr(11);
        // eps: a number (a rectangle r_in .. r_out, z_start .. z_end) or an expression of x, y, z, r, phi (lab frame, with units;
        // the rectangle keys then bound it and are optional; x, y or phi: chi per angle)
        std::string et = d.str(b, "eps");
        if (et.size() >= 2 && et.front() == '"' && et.back() == '"') et = et.substr(1, et.size() - 2);
        const UnitFn u = [k0](const std::string& n, double& f) { return unit_to_code(n, k0, f); };
        if (!c.rz) {                                 // Cart3D / 2D (dielectric3d.hpp): eps a number (everywhere) or of x, y, z
            for (const char* k : {"r_in", "r_out", "z_start", "z_end", "ramp"})
                if (d.has(b, k)) throw std::runtime_error(b + "." + k + ": RZ only; in Cart3D / 2D give eps as an expression of x, y, z");
            Dielectric3Config e3;
            e3.name = e.name;
            try {
                Expr x(et, {"x", "y", "z"}, u);
                bool cst = true;
                for (const Expr::Ins& ins : x.program()) cst = cst && ins.op != Expr::VAR;
                if (cst) e3.eps = x({0.0, 0.0, 0.0});
                else e3.fn = x;
            } catch (const std::exception& ex) {
                throw std::runtime_error(b + ".eps: a number or an expression of x, y, z (" + ex.what() + ")");
            }
            c.dielectrics3.push_back(e3);
            continue;
        }
        bool constant = true;
        try {
            Expr x(et, {"x", "y", "z", "r", "phi"}, u);
            for (const Expr::Ins& ins : x.program()) {
                if (ins.op != Expr::VAR) continue;
                constant = false;
                if (ins.arg == 0 || ins.arg == 1 || ins.arg == 4) e.phi = true;
            }
            if (constant) e.eps = x({0.0, 0.0, 0.0, 0.0, 0.0});
            else e.fn = x;
        } catch (const std::exception& ex) {
            throw std::runtime_error(b + ".eps: a number or an expression of x, y, z, r, phi (" + ex.what() + ")");
        }
        const bool fn = !e.fn.empty();
        e.r_in = d.num(b, "r_in", Kind::Length, 0.0) * k0;
        e.r_out = fn && !d.has(b, "r_out") ? 1e300 : d.num(b, "r_out", Kind::Length) * k0;
        e.z_start = fn && !d.has(b, "z_start") ? -1e300 : d.num(b, "z_start", Kind::Length) * k0;
        e.z_end = fn && !d.has(b, "z_end") ? 1e300 : d.num(b, "z_end", Kind::Length) * k0;
        e.ramp = d.num(b, "ramp", Kind::Length, 0.0) * k0;
        c.dielectrics.push_back(e);
    }
    if (!c.rz && (c.rings.on || c.plane_z >= 0 || c.axis_every > 0))
        throw std::runtime_error("the rings block, output.plane_z and output.axis_every need grid.geometry = rz");
    for (const std::string& b : d.block_names()) {   // docs/specs/harmonic_focus.md
        if (b != "harmonic_focus" && b.rfind("harmonic_focus ", 0) != 0) continue;
        HFocusConfig H;
        H.name = b.size() > 15 ? b.substr(15) : "";
        H.n_min = d.num(b, "n_min", Kind::None, H.n_min);
        H.n_max = d.num(b, "n_max", Kind::None, H.n_max);
        H.order = int(d.integer(b, "filter_order", H.order));
        H.every = d.integer(b, "every", 0);
        const char* lo[3] = {c.rz ? "zmin" : "xmin", "ymin", "zmin"};
        const char* hi[3] = {c.rz ? "zmax" : "xmax", c.rz ? "rmax" : "ymax", "zmax"};
        for (int a = 0; a < (c.rz ? 2 : 3); ++a) {
            if (!(c.rz && a == 1) && d.has(b, lo[a])) H.lo[a] = d.num(b, lo[a], Kind::Length) * k0;
            if (d.has(b, hi[a])) H.hi[a] = d.num(b, hi[a], Kind::Length) * k0;
        }
        c.hfocus.push_back(H);
    }
    if (d.has("output", "openpmd_fields")) {
        c.openpmd.fields.clear();
        std::istringstream is(d.str("output", "openpmd_fields"));
        for (std::string w; is >> w;) c.openpmd.fields.push_back(w);
    }
    {
        std::istringstream is(d.str("output", "openpmd_species", ""));
        for (std::string w; is >> w;) c.openpmd.species.push_back(w);
    }
    {
        const std::string pr = d.str("output", "openpmd_precision", "single");
        if (pr != "single" && pr != "double") throw std::runtime_error("output.openpmd_precision: single or double");
        c.openpmd.single = pr == "single";
    }
    c.checkpoint.every = d.integer("checkpoint", "every", 0);
    c.checkpoint.keep = int(d.integer("checkpoint", "keep", 2));
    c.checkpoint.stop_hours = d.num("checkpoint", "stop_after_hours", Kind::None, 0.0);
    // blocks `laser` (c.laser) and `laser <name>` (c.extra_lasers), summed at the entrance plane
    auto laser_block = [&](const std::string& b) {
        LaserConfig L;
        L.enabled = d.flag(b, "enabled", true);
        L.a0 = d.num(b, "a0", Kind::None);
        L.tau = d.num(b, "duration", Kind::Time) * c_light * k0;
        L.tc = d.num(b, "delay", Kind::Time, 2 * L.tau / (c_light * k0)) * c_light * k0;
        L.w0 = d.num(b, "waist", Kind::Length) * k0;
        if (c.rz) {                              // round; centre (x, y) in the transverse plane of the particles, default the axis
            L.w0z = L.w0;
            L.yc = d.num(b, "y_center", Kind::Length, 0.0) * k0;   // off the axis: all modes (rz_laser.cpp quadrature)
            L.zc = d.num(b, "z_center", Kind::Length, 0.0) * k0;
        } else {
            L.w0z = d.num(b, "waist_z", Kind::Length, 0.0) * k0;
            L.yc = d.num(b, "y_center", Kind::Length, 0.5 * c.N[1] * c.h[1] / k0) * k0;
            L.zc = d.num(b, "z_center", Kind::Length, 0.5 * c.N[2] * c.h[2] / k0) * k0;
        }
        L.xf = d.num(b, "focus", Kind::Length, 0.0) * k0;
        L.p = int(d.integer(b, "mode_p", 0));
        L.l = int(d.integer(b, "mode_l", 0));
        L.phase = d.num(b, "phase", Kind::Angle, 0.0);
        L.tfsf_x = d.num(b, "tfsf_x", Kind::Length, -1.0 / k0) * k0;   // O1: TFSF plane (docs/specs/tfsf.md)
        L.angle = d.num(b, "angle", Kind::Angle, 0.0);
        {
            const std::string inc = d.str(b, "tfsf_incident", "grid");   // docs/specs/tfsf.md "Exact plane waves"
            if (inc != "grid" && inc != "modes") throw std::runtime_error(b + ".tfsf_incident: grid or modes");
            L.tfsf_modes = inc == "modes";
            if (L.tfsf_modes && (L.tfsf_x < 0 || c.rz || c.periodic[1] == false || c.periodic[2] == false))
                throw std::runtime_error(b + ".tfsf_incident = modes: a TFSF plane (tfsf_x) in Cart3D with periodic y and z");
        }
        {
            const std::string inj = d.str(b, "injection", "plane");   // boosted runs: the lab-fixed antenna (decision 89)
            if (inj != "plane" && inj != "antenna") throw std::runtime_error(b + ".injection: plane or antenna");
            L.antenna = inj == "antenna";
        }
        L.cut = L.tfsf_x >= 0;                   // TFSF: an envelope that is exactly zero beyond 3 sigma (laser.hpp)
        if (c.rz && L.tfsf_x >= 0 && L.angle != 0)
            throw std::runtime_error(b + ".tfsf_x in geometry = rz: normal incidence (no angle)");
        if (L.w0 <= 0 && (L.tfsf_x < 0 || L.w0z > 0 || L.p != 0 || L.l != 0 || std::abs(L.angle) >= 0.5 * M_PI))
            throw std::runtime_error(b + ".waist = 0 (a plane wave of one k_y) needs tfsf_x, no waist_z or mode_p, l and "
                                         "|angle| < 90 deg");
        if (L.tfsf_x >= 0 && std::abs(L.angle) > 0.25 * M_PI + 1e-12)   // Alexander 2026-10-31 (decision 60)
            throw std::runtime_error(b + ".angle: at most 45 deg through the TFSF plane; steeper lasers enter "
                                         "from a side boundary (block side, docs/specs/side_injection.md)");
        if (c.rz) {                              // E along x (pol 'y' of LaserConfig) or y ('z')
            const std::string pol = d.str(b, "polarization", "x");
            if (pol != "x" && pol != "y") throw std::runtime_error(b + ".polarization: x or y (geometry = rz)");
            L.pol = pol == "x" ? 'y' : 'z';
            if (L.enabled && c.periodic[0]) throw std::runtime_error("the laser enters through an open z boundary: grid.boundary_z = open");
            if (L.enabled && L.tfsf_x < 0 && c.x_abs_cells > 0)
                throw std::runtime_error(b + ": with grid.boundary_z = absorbing the laser must come in through a TFSF plane (tfsf_x)");
            return L;
        }
        const std::string pol = d.str(b, "polarization", "z");
        if (pol != "y" && pol != "z") throw std::runtime_error(b + ".polarization: y or z");
        L.pol = pol[0];
        if (L.enabled && c.periodic[0]) throw std::runtime_error("the laser enters through an open x boundary: grid.boundary_x = open");
        if (L.enabled && L.tfsf_x < 0 && c.x_abs_cells > 0)
            throw std::runtime_error(b + ": with grid.boundary_x = absorbing the laser must come in through a TFSF plane (tfsf_x)");
        return L;
    };
    // a laser block with `file` takes the pulse from a LASY envelope file (docs/specs/laser_file.md)
    auto file_block = [&](const std::string& b) {
        LaserFileConfig F;
        F.name = b;
        F.path = d.str(b, "file");
        if (F.path.size() >= 2 && F.path.front() == '"' && F.path.back() == '"') F.path = F.path.substr(1, F.path.size() - 2);
        if (!F.path.empty() && F.path[0] != '/' && !std::ifstream(F.path)) {   // relative to the deck's directory
            const size_t sl = d.file.rfind('/');
            if (sl != std::string::npos && std::ifstream(d.file.substr(0, sl + 1) + F.path)) F.path = d.file.substr(0, sl + 1) + F.path;
        }
        F.mesh = d.str(b, "file_mesh", "laserEnvelope");
        F.iteration = d.integer(b, "file_iteration", -1);
        F.delay = d.has(b, "delay") ? d.num(b, "delay", Kind::Time) * c_light * k0 : -1;
        F.t_center = d.has(b, "t_center") ? d.num(b, "t_center", Kind::Time) * c_light * k0 : 1e300;
        F.file_x = d.num(b, "file_x", Kind::Length, 0.0) * k0;
        F.yc = d.num(b, "y_center", Kind::Length, 0.5 * c.N[1] * c.h[1] / k0) * k0;
        F.zc = d.num(b, "z_center", Kind::Length, 0.5 * c.N[2] * c.h[2] / k0) * k0;
        F.a0 = d.num(b, "a0", Kind::None, 0.0);
        F.flat = c.flat;
        const std::string inj = d.str(b, "injection", "antenna");   // boosted runs: the lab-fixed antenna only (decision 97)
        if (inj != "antenna") throw std::runtime_error(b + ".injection: a laser file is emitted by the antenna in a boosted run");
        F.antenna = true;
        if (c.periodic[0]) throw std::runtime_error("the laser enters through an open x boundary: grid.boundary_x = open");
        return F;
    };
    for (const std::string& b : d.block_names()) {
        if (b != "laser" && b.rfind("laser ", 0) != 0) continue;
        if (d.has(b, "file")) {
            if (d.flag(b, "enabled", true)) c.file_lasers.push_back(file_block(b));
        } else if (b == "laser") c.laser = laser_block(b);
        else c.extra_lasers.push_back(laser_block(b));
    }
    if (c.rz && d.has_block("pml")) throw std::runtime_error("geometry = rz: no pml block yet (docs/specs/rz_modes.md, step 4)");
    if (d.has_block("window")) {
        WindowConfig& W = c.window;
        W.enabled = d.flag("window", "enabled", true);
        W.start = d.num("window", "start", Kind::Time, 0.0) * c_light * k0;
        W.v = d.num("window", "velocity", Kind::None, 1.0);
    }
    for (const std::string& b : d.block_names()) {
        if (b.rfind("bunch ", 0) == 0) {             // a Gaussian bunch without a file (src/structure.cpp)
            SpeciesConfig s;
            s.name = b.substr(6);
            for (const SpeciesConfig& o : c.species)
                if (o.name == s.name) throw std::runtime_error(b + ": a species of this name exists already");
            species_bunch_from_deck(d, b, c, s, k0);
            c.species.push_back(s);
            continue;
        }
        if (b.rfind("species ", 0) != 0) continue;
        SpeciesConfig s;
        s.name = b.substr(8);
        s.spin = spin_from_deck(d, b);               // spin = sx sy sz (qed.hpp)
        const std::string kind = d.str(b, "kind", "particle");
        if (kind != "particle" && kind != "photon") throw std::runtime_error(b + ".kind: particle or photon");
        if (kind == "photon") {                      // filled by the qed block's emission; q = 0, moved by the qed module
            s.photon = true; s.q = 0; s.m = 1; s.density = 0; s.mobile = false;
            for (int a = 0; a < 3; ++a) { s.rmin[a] = -1e300; s.rmax[a] = 1e300; }
            c.species.push_back(s);
            continue;
        }
        if (d.has(b, "file")) {                      // a beam from an openPMD file (beam_import.hpp, src/structure.cpp)
            species_file_from_deck(d, b, c, s, k0);
            c.species.push_back(s);
            continue;
        }
        species_charge(d, b, s);
        s.density = d.num(b, "density", Kind::Density);
        if (c.rz) {
            // particle axes (z, x, y); positions bounded in z and r, regular in (z, r, theta) per cell
            const char* pa[3] = {"z", "x", "y"};
            const char* pp[3] = {"ppc_z", "ppc_r", "ppc_theta"};
            for (int a = 0; a < 3; ++a) {
                s.ppc[a] = int(d.integer(b, pp[a], a == 2 ? 4 : 1));
                s.u0[a] = d.num(b, std::string("u") + pa[a], Kind::None, 0.0);
                s.uth[a] = d.num(b, std::string("uth_") + pa[a], Kind::None, 0.0);
            }
            s.rmin[0] = d.num(b, "zmin", Kind::Length, -1e300) * k0; s.rmax[0] = d.num(b, "zmax", Kind::Length, 1e300) * k0;
            s.rmin[1] = d.num(b, "rmin", Kind::Length, 0.0) * k0; s.rmax[1] = d.num(b, "rmax", Kind::Length, 1e300) * k0;
            s.rmin[2] = -1e300; s.rmax[2] = 1e300;
            s.t_start = d.num(b, "t_start", Kind::Time, 0.0) * c_light * k0;
            s.rigid_until = d.num(b, "rigid_until", Kind::Time, 0.0) * c_light * k0;
            s.mobile = d.flag(b, "mobile", true);
            s.self_field = d.flag(b, "self_field", false);   // docs/specs/rz_metal.md (beam along z, open z)
            if (d.has(b, "density_function")) {
                std::string t = d.str(b, "density_function");
                if (t.size() >= 2 && t.front() == '"' && t.back() == '"') t = t.substr(1, t.size() - 2);
                const UnitFn u = [k0](const std::string& n, double& f) { return unit_to_code(n, k0, f); };
                try { s.density_fn = Expr(t, {"z", "r", "x", "y", "phi"}, u); }
                catch (const std::exception& ex) { throw std::runtime_error(b + ".density_function: " + ex.what()); }
            }
            c.species.push_back(s);
            continue;
        }
        for (int a = 0; a < 3; ++a) {
            const std::string x(1, ax[a]);
            s.ppc[a] = int(d.integer(b, "ppc_" + x, 1));
            s.u0[a] = d.num(b, "u" + x, Kind::None, 0.0);
            s.uth[a] = d.num(b, "uth_" + x, Kind::None, 0.0);
            s.rmin[a] = d.num(b, x + "min", Kind::Length, -1e300) * k0;
            s.rmax[a] = d.num(b, x + "max", Kind::Length, 1e300) * k0;
        }
        s.slab_thickness = d.num(b, "slab_thickness", Kind::Length, 0.0) * k0;
        s.slab_x0 = d.num(b, "slab_x0", Kind::Length, 0.0) * k0;
        s.slab_y0 = d.num(b, "slab_y0", Kind::Length, 0.0) * k0;
        s.slab_angle = d.num(b, "slab_angle", Kind::Angle, 0.0);
        s.slab_ramp = d.num(b, "slab_ramp", Kind::Length, 0.0) * k0;
        s.slab_ramp_extent = d.num(b, "slab_ramp_extent", Kind::Length, 5 * s.slab_ramp / k0) * k0;
        s.slab_rear_ramp = d.num(b, "slab_rear_ramp", Kind::Length, 0.0) * k0;
        s.slab_rear_extent = d.num(b, "slab_rear_ramp_extent", Kind::Length, 5 * s.slab_rear_ramp / k0) * k0;
        s.slab_exact = d.flag(b, "slab_exact", false);
        s.self_field = d.flag(b, "self_field", false);
        s.t_start = d.num(b, "t_start", Kind::Time, 0.0) * c_light * k0;
        s.rigid_until = d.num(b, "rigid_until", Kind::Time, 0.0) * c_light * k0;
        s.mobile = d.flag(b, "mobile", true);
        if (d.has(b, "density_function")) {
            std::string t = d.str(b, "density_function");
            if (t.size() >= 2 && t.front() == '"' && t.back() == '"') t = t.substr(1, t.size() - 2);
            const UnitFn u = [k0](const std::string& n, double& f) { return unit_to_code(n, k0, f); };
            try { s.density_fn = Expr(t, {"x", "y", "z"}, u); }
            catch (const std::exception& ex) { throw std::runtime_error(b + ".density_function: " + ex.what()); }
        }
        c.species.push_back(s);
    }
    // side injection (docs/specs/side_injection.md): a TFSF tube with faces y, z = const, the incident field a plane wave
    if (d.has_block("side") && d.flag("side", "enabled", true)) {
        SideConfig& S = c.side;
        if (c.rz) throw std::runtime_error("side: Cart3D only");
        S.on = true;
        const std::string fc = d.str("side", "faces", "yz");
        if (fc != "y" && fc != "z" && fc != "yz") throw std::runtime_error("side.faces: y, z or yz");
        S.faces[1] = fc.find('y') != std::string::npos;
        S.faces[2] = fc.find('z') != std::string::npos;
        const int margin = int(d.integer("side", "margin", 2));
        for (int a = 1; a < 3; ++a) {
            if (!S.faces[a]) continue;
            const std::string s = a == 1 ? "y" : "z";
            const int L = c.pml.on[a] ? c.pml.cells : 0;   // default: margin cells inside the CPML layers (or the edges)
            S.lo[a] = d.has("side", s + "_lo") ? int(std::ceil(d.num("side", s + "_lo", Kind::Length) * k0 / c.h[a] - 1e-9))
                                              : L + margin;
            S.hi[a] = d.has("side", s + "_hi") ? int(std::floor(d.num("side", s + "_hi", Kind::Length) * k0 / c.h[a] + 1e-9))
                                              : c.N[a] - 1 - L - margin;
        }
        const std::string wave = d.str("side", "wave", "plane");
        if (wave != "plane" && wave != "flying_focus") throw std::runtime_error("side.wave: plane or flying_focus");
        S.wave = wave == "flying_focus" ? 1 : 0;
        S.a0 = d.num("side", "a0", Kind::None);
        const std::string pol = d.str("side", "polarization", "z");
        if (pol != "y" && pol != "z") throw std::runtime_error("side.polarization: y or z");
        S.pol = pol[0];
        S.r0[1] = d.num("side", "y_center", Kind::Length, 0.5 * c.N[1] * c.h[1] / k0) * k0;
        S.r0[2] = d.num("side", "z_center", Kind::Length, 0.5 * c.N[2] * c.h[2] / k0) * k0;
        if (S.wave == 0) {
            S.tau = d.num("side", "duration", Kind::Time, 0.0) * c_light * k0;
            S.tc = d.num("side", "delay", Kind::Time, 0.0) * c_light * k0;
            S.phase = d.num("side", "phase", Kind::Angle, 0.0);
            S.angle = d.num("side", "angle", Kind::Angle);
            S.elevation = d.num("side", "elevation", Kind::Angle, 0.0);
            S.r0[0] = d.num("side", "x_center", Kind::Length, 0.5 * c.N[0] * c.h[0] / k0) * k0;
        } else {                                       // the luminal Bessel pulse of GEM-PIC-RZ's flying_focus block
            S.cone = d.num("side", "cone_angle", Kind::Angle);
            if (d.has("side", "length_1e") == d.has("side", "duration"))
                throw std::runtime_error("side.wave = flying_focus: give length_1e or duration (intensity FWHM on the axis)");
            S.len = d.has("side", "length_1e") ? d.num("side", "length_1e", Kind::Length) * k0
                                               : d.num("side", "duration", Kind::Time) * c_light * k0 / std::sqrt(2 * std::log(2.0));
            S.r0[0] = d.num("side", "x_center", Kind::Length) * k0;   // the envelope centre at t = 0 (absolute x)
            S.phase = d.num("side", "cep", Kind::Angle, 0.0);
        }
        // a background plasma (decision 63): the incident wave of that uniform plasma; outside the kinetic region (or
        // everywhere: fluid = all) a linear cold fluid carries its current
        S.density = d.num("side", "background_density", Kind::None, 0.0);
        const std::string fl = d.str("side", "fluid", "shell");
        if (fl != "shell" && fl != "all") throw std::runtime_error("side.fluid: shell or all");
        S.fluid_all = fl == "all";
        S.order = c.order;
        for (int a = 1; a < 3; ++a) S.absorb[a] = c.pml.on[a] ? c.pml.cells : 0;
    }
    c.diags = beam_diags_from_deck(d, c.every);
    boost_from_deck(d, c, k0);                       // blocks boost, lab_diagnostics (boost.cpp)
    c.qed = qed_from_deck(d, c.lambda0);             // block qed (qed.cpp)
    c.collisions = collisions_from_deck(d, c);       // blocks collisions [<name>] (collisions.cpp)
    structure_from_deck(d, c, k0);                   // metal, quadrupoles, wake diagnostics (src/structure.cpp)
    d.check_unused();
    apply_boost(c);                                  // the lab deck -> the boosted run (no-op without boost)
    c.check_structure();
    c.check();
    return c;
}

double Config::dt() const {
    if (solver == Solver::Yee) return cfl / std::sqrt(1 / (h[0] * h[0]) + 1 / (h[1] * h[1]) + 1 / (h[2] * h[2]));
    return h[0];
}

// O6 (docs/specs/implicit_mover.md): what the first version of the implicit mover supports
void Config::check_mover() const {
    auto bad = [](const std::string& w) { throw std::runtime_error("numerics.mover = implicit: " + w + " (docs/specs/implicit_mover.md)"); };
    if (rz) {                                       // RZ (decision 82): shape 2 in the C1 radial map; z and radial absorbers are fine
        if (solver != Solver::IRip) bad("numerics.solver = irip");
        if (o5) bad("numerics.closure = box (o5 does not conserve the energy)");
        if (order != 2) bad("numerics.shape = 2 in geometry = rz (the C1 radial map is a shape-2 map)");
        // m >= 1 (decision 82): even m >= 2 have no smooth deposit at the axis (the fold of decision 50 gives a particle on the axis a
        // charge of mode m with its arbitrary phase) and stall regularly; m = 1 is accepted (Alexander, 2026-11-07) although rare steps
        // with a path within ~0.01 dr of the axis stay unconverged (counted in MoverStats, Gauss law still exact)
        if (modes > 2) bad("grid.modes <= 2 (m = 0, 1) in geometry = rz (decision 82)");
        for (const LaserConfig& L : all_lasers())
            if (L.enabled && L.tfsf_x >= 0) bad("no TFSF laser yet");
        if (window.enabled) bad("no moving window yet");
        if (filter_passes > 0) bad("no current filter yet (the gather would need its adjoint)");
        if (rz_shape_width > 0) bad("no rz_transverse_shape yet (the gather would need its adjoint)");
        if (!(mover_tol > 0) || mover_maxit < 1 || mover_anderson < 0) bad("mover_tol > 0, mover_maxit >= 1, mover_anderson >= 0");
        return;
    }
    if (solver != Solver::IRip) bad("numerics.solver = irip");
    if (o5) bad("numerics.closure = box (o5 does not conserve the energy)");
    if (order != 2 && order != 3) bad("numerics.shape = 2 or 3");
    if (pml.on[1] || pml.on[2]) bad("no pml layers yet");
    if (side.on) bad("no side tube yet");
    for (const LaserConfig& L : all_lasers())
        if (L.enabled && L.tfsf_x >= 0) bad("no TFSF laser yet");
    if (window.enabled) bad("no moving window yet");
    if (filter_passes > 0) bad("no current filter yet (the gather would need its adjoint)");
    if (x_abs_cells > 0) bad("no absorbing x layers yet");
    if (deposit == Deposit::Tiled || deposit == Deposit::Box) bad("numerics.deposit = auto or global");
    if (!(mover_tol > 0) || mover_maxit < 1 || mover_anderson < 0) bad("mover_tol > 0, mover_maxit >= 1, mover_anderson >= 0");
}

void Config::check() const {
    if (!labdiag.times.empty() && !boost.on) throw std::runtime_error("lab_diagnostics: needs the boost block (boosted frame)");
    if (boost.on && !(boost.beta > 0)) throw std::runtime_error("boost: apply_boost(config) before the check");
    boost_check_neutral(*this);                      // the boosted injection must be exactly neutral (decision 108)
    qed_check(*this);                                // spin, qed block, photon species (qed.cpp)
    for (const SpeciesConfig& s : species) {
        if (!s.ion.on) continue;
        if (mover_implicit) throw std::runtime_error("species " + s.name + ": field ionization not yet with numerics.mover = implicit");
        if (s.self_field) throw std::runtime_error("species " + s.name + ": field ionization not with self_field");
    }
    for (const HFocusConfig& H : hfocus) {
        if (H.order < 2 || H.order > 16 || H.order % 2) throw std::runtime_error("harmonic_focus.filter_order: even, 2 .. 16");
        if (H.n_min < 0 || H.n_max < 0 || (H.n_max > 0 && H.n_max <= H.n_min) || (H.n_min == 0 && H.n_max == 0))
            throw std::runtime_error("harmonic_focus: 0 <= n_min < n_max (n_min = 0: no high-pass, n_max = 0: no low-pass)");
        if (H.n_max * dt() >= 0.9 * M_PI)
            throw std::runtime_error("harmonic_focus.n_max must be below 0.9 of the Nyquist order pi / (omega0 dt) = " +
                                     std::to_string(M_PI / dt()));
    }
    for (auto& s : species)                          // frozen species of collisions blocks may carry momenta (collisions.hpp)
        if (!s.mobile && !(frozen(s) && !s.self_field) && (s.self_field || s.u0[0] != 0 || s.u0[1] != 0 || s.u0[2] != 0 || s.uth[0] != 0 || s.uth[1] != 0 || s.uth[2] != 0))
            throw std::runtime_error("species " + s.name + ": mobile = false needs particles at rest (no u, uth, self_field)");
    if (filter_passes < 0) throw std::runtime_error("numerics.filter_passes must be >= 0");
    if ((filter_comp || filter_xonly) && (filter_passes == 0 || filter_local || !filter_binomial))
        throw std::runtime_error("numerics.filter_compensate, filter_axes = x: with filter = binomial and filter_passes >= 1");
    if (filter_flat != 0 && (filter_flat < 2 || filter_flat > 9 || filter_comp || filter_passes == 0 || filter_local || !filter_binomial))
        throw std::runtime_error("numerics.filter_flat = L (2 .. 9): with filter = binomial, filter_passes >= 1 and without "
                                 "filter_compensate (L = 2 is the compensator)");
    if (mover_implicit) check_mover();
    if (!media.empty()) check_media();
    if (rz) { check_rz(); return; }
    if (flat) check_flat();
    const double D = h[0];
    const double cour = D * D * (1 / (h[1] * h[1]) + (flat ? 0.0 : 1 / (h[2] * h[2])));   // 2D: no z term (d/dz = 0)
    double wp2 = 0;
    for (auto& s : species)
        if (!frozen(s)) wp2 += check_density_factor(s) * (s.density * s.q2() / s.m + s.wp2_electrons());   // frozen: no current
    bool laser = !file_lasers.empty();
    for (const LaserConfig& L : all_lasers()) laser = laser || L.enabled;
    // self_field (docs/specs/beam_self_field.md): with iRIP the discrete RIP-CC equilibrium (iRIP keeps the same Gauss
    // law, so it holds to round-off; the near field is steady to O(h^2), test_beam B5); with Yee in a periodic box the
    // equilibrium of Yee's Gauss law with B = beta x E at x_{i+1/2} (decision 98)
    const bool boundary = !periodic[0] || !periodic[1] || !periodic[2] || laser || window.enabled || pml.on[1] || pml.on[2];
    if (solver == Solver::IRip) {
        // no Courant limit; the plasma limit omega_p dt < 2 of the explicit current stays (docs/specs/implicit_rip.md)
        if (wp2 * D * D >= 4 && !mover_implicit) throw std::runtime_error("unstable: omega_p dt >= 2 (iRIP)");
        // x periodic or open (laser, window), y and z periodic or pml
        if (N[0] < 3)
            throw std::runtime_error("numerics.solver = irip: nx >= 3 (docs/specs/implicit_rip.md)");
    } else if (solver == Solver::Yee) {
        // Courant limit and the leapfrog's plasma limit omega_p dt < 2 (docs/specs/yee.md)
        if (!(cfl > 0 && cfl < 1)) throw std::runtime_error("numerics.cfl must be in (0, 1) for the Yee solver");
        if (wp2 * dt() * dt() >= 4) throw std::runtime_error("unstable: omega_p dt >= 2 (Yee)");
        // the boundary physics so far is RIP-CC's (transport vector, its CPML, the window's one cell per step)
        if (boundary)
            throw std::runtime_error("numerics.solver = yee: periodic boxes only so far (no open x, pml, laser or moving "
                                     "window; docs/specs/yee.md)");
    } else if (cour + 0.25 * D * D * wp2 >= 1.0)
        throw std::runtime_error("unstable: dx^2 (1/dy^2 + 1/dz^2) + dx^2 wp^2/4 = " +
                                 std::to_string(cour + 0.25 * D * D * wp2) + " >= 1 (c dt = dx is fixed)");
    if (order < 1 || order > 3) throw std::runtime_error("numerics.shape must be 1, 2 or 3");
    if (order == 3 && window.enabled) throw std::runtime_error("numerics.shape = 3: not with a moving window yet (docs/specs/solid_target.md)");
    if (filter_passes > 0 && (solver == Solver::Yee || (window.enabled && !boost.on && window.v < 1)))
        throw std::runtime_error("numerics.filter_passes (Cart3D): rip_cc or irip; a moving window at v = c or in a boosted frame "
                                 "(docs/specs/solid_target.md, decision 107)");
    for (auto& s : species) {
        if (s.slab_ramp < 0 || s.slab_ramp_extent < 0 || s.slab_rear_ramp < 0 || s.slab_rear_extent < 0)
            throw std::runtime_error("species " + s.name + ": slab_ramp, slab_rear_ramp and their extents must be >= 0");
        if ((s.slab_ramp > 0 || s.slab_rear_ramp > 0 || s.slab_exact) && !(s.slab_thickness > 0))
            throw std::runtime_error("species " + s.name + ": slab_ramp, slab_rear_ramp and slab_exact need slab_thickness > 0");
    }
    for (const LaserConfig& L : all_lasers()) {
        if (!L.enabled) continue;
        if (L.p < 0) throw std::runtime_error("laser.mode_p must be >= 0");
        if ((L.p != 0 || L.l != 0) && L.w0z != L.w0)
            throw std::runtime_error("Laguerre-Gauss modes (mode_p, mode_l) need a round beam: waist_z = waist");
        if (L.w0 <= 0) {                         // the plane wave of one k_y = sin(angle) must be periodic in y
            const double m = std::sin(L.angle) * N[1] * h[1] / (2 * M_PI);
            if (!periodic[1] || pml.on[1] || std::abs(m - std::round(m)) > 1e-6)
                throw std::runtime_error("laser.waist = 0: periodic y with sin(angle) ny dy = m lambda0 (now m = " +
                                         std::to_string(m) + ")");
        }
    }
    if ((openpmd.every > 0 || checkpoint.every > 0 || checkpoint.stop_hours > 0) && !hdf5_available())
        throw std::runtime_error("output.openpmd_every and the checkpoint block need a build with HDF5 (make HDF5=1)");
    if (openpmd.every < 0 || checkpoint.every < 0 || checkpoint.keep < 0 || checkpoint.stop_hours < 0)
        throw std::runtime_error("output.openpmd_every, checkpoint.every, keep and stop_after_hours must be >= 0");
    for (const std::string& n : openpmd.fields)
        if (n != "E" && n != "B" && n != "J" && n != "rho" && n.rfind("rho_", 0) != 0)
            throw std::runtime_error("output.openpmd_fields: unknown field " + n + " (E B J rho rho_<species>)");
    for (const std::string& n : openpmd.species) {
        bool found = n == "all";
        for (const SpeciesConfig& sc : species) found = found || sc.name == n;
        if (!found) throw std::runtime_error("output.openpmd_species: no species " + n);
    }
    for (int a = 1; a < 3; ++a)
        if (pml.on[a] && (pml.cells < 1 || N[a] < 2 * pml.cells + 4 || !(pml.R0 > 0 && pml.R0 < 1) || pml.order < 1))
            throw std::runtime_error("pml: cells >= 1 with n >= 2 cells + 4 along each pml axis, 0 < R0 < 1, order >= 1");
    if (side.on) {
        if (solver == Solver::Yee) throw std::runtime_error("side: RIP-CC or iRIP");
        for (const LaserConfig& L : all_lasers())
            if (L.enabled && L.tfsf_x >= 0) throw std::runtime_error("side: not together with a TFSF x plane (laser.tfsf_x)");
        if (!(side.a0 > 0) || side.tau < 0) throw std::runtime_error("side: a0 > 0, duration >= 0 (0: a continuous wave)");
        if (side.wave == 1) {                          // the table holds the window frame: the window moves at c from t = 0
            if (!window.enabled || window.v != 1 || window.start != 0 || periodic[0])
                throw std::runtime_error("side.wave = flying_focus: needs the moving window at c from t = 0 (window.velocity = 1, "
                                         "start = 0, open x)");
            if (!(side.cone > 0 && side.cone < 1.2) || !(side.len > 0))
                throw std::runtime_error("side.wave = flying_focus: 0 < cone_angle < 1.2 rad, length_1e (duration) > 0");
            const double kap = 1 - std::cos(side.cone);
            if (2 * kap - kap * kap <= side.density)
                throw std::runtime_error("side.background_density: too high for this cone angle (no conical wave, k_perp^2 = "
                                         "2 kappa - kappa^2 - wp^2 < 0 at omega0)");
        }
        if (!(side.density >= 0 && side.density < 1))
            throw std::runtime_error("side.background_density: 0 <= n < 1 n_c (the incident wave of omega0 must propagate)");
        for (int a = 1; a < 3; ++a) {
            if (!side.faces[a]) continue;
            const int L = pml.on[a] ? pml.cells : 0;     // the corrected differences must not reach into a layer
            if (side.lo[a] < L + 1 || side.hi[a] > N[a] - 2 - L || side.hi[a] - side.lo[a] < 4)
                throw std::runtime_error(std::string("side: the faces along ") + (a == 1 ? "y" : "z") +
                                         " must lie at least one cell outside the CPML layers (or the edges) and >= 4 cells apart");
        }
    }
    check_window();
}

// grid.geometry = cartesian2d (docs/specs/cart2d.md): what the first version supports. Everything z-uniform: lasers
// without a z profile, no features that need z structure or have no 2D variant yet.
void Config::check_flat() const {
    const std::string g = "grid.geometry = cartesian2d: ";
    if (solver == Solver::Yee) throw std::runtime_error(g + "numerics.solver = rip_cc or irip");
    if (side.on) throw std::runtime_error(g + "no side block (its tube needs faces in z)");
    // laser files: slab files only (FileLaser checks the LASY y axis)
    for (const LaserConfig& L : all_lasers())
        if (L.enabled && (L.w0z > 0 || L.p != 0 || L.l != 0))
            throw std::runtime_error(g + "lasers uniform in z (no waist_z, mode_p, mode_l)");
    for (auto& s : species) {
        if (s.self_field) throw std::runtime_error(g + "no self_field yet (species " + s.name + ")");
        if (s.ppc[2] != 1 || s.rmin[2] > -1e299 || s.rmax[2] < 1e299)
            throw std::runtime_error(g + "species " + s.name + ": no ppc_z, zmin or zmax (one z cell)");
    }
}

void Config::check_window() const {
    if (window.enabled) {
        if (periodic[0]) throw std::runtime_error(rz ? "the moving window needs grid.boundary_z = open"
                                                     : "the moving window needs grid.boundary_x = open");
        if (!(window.v > 0 && window.v <= 1)) throw std::runtime_error("window.velocity must be in (0, 1] (units of c)");
        if (window.start < 0) throw std::runtime_error("window.start must be >= 0");
        // the laser enters through the left edge of the box, which moves with the window: the tail is
        // cut at a(tc) as the head is at a(-tc) (default delay 2 duration: 4e-3 a0)
        double tcmax = 0;
        for (const LaserConfig& L : all_lasers())
            if (L.enabled) tcmax = std::max(tcmax, L.tc);
        if (tcmax > 0 && window.start < 2 * tcmax)
            throw std::runtime_error("window.start must be >= 2 laser.delay (the pulse must have entered before "
                                     "the window moves)");
    }
}

RadialGrid Config::radial_grid() const {
    RadialGrid g = rz_radial_grid(N[1], h[1], rz_r_uniform > 0 ? rz_r_uniform : 1e300, rz_stretch, rz_dr_max > 0 ? rz_dr_max : 1e300);
    g.set_meff(modes, metal_rz.on && !metal_rz.spectral);   // the staggered angular difference of the metal block (decision 103)
    g.set_rod(modes, rz_axis_cut);
    return g;
}

// geometry = rz: periodic or open z (laser, moving window), conducting wall or absorber in r; no pml or
// self field (docs/specs/rz_modes.md)
void Config::check_rz() const {
    if (rz_stretch != 1) {
        if (!(rz_stretch > 1) || !(rz_r_uniform > 0)) throw std::runtime_error("grid.stretch > 1 needs grid.r_uniform > 0");
    }
    double wp2 = 0;
    for (auto& s : species)
        if (!frozen(s)) wp2 += check_density_factor(s) * (s.density * s.q2() / s.m + s.wp2_electrons());   // frozen: no current (collisions.hpp)
    const double D = h[0];
    if (solver == Solver::Yee) throw std::runtime_error("geometry = rz: numerics.solver = rip_cc or irip");
    if (solver == Solver::IRip) {
        if (wp2 * D * D >= 4 && !mover_implicit) throw std::runtime_error("unstable: omega_p dt >= 2 (iRIP)");
    } else {
        // explicit RIP-CC: dz^2 (lambda_max + wp^2) / 4 < 1, lambda_max of the radial operators (rz_lambda_max)
        const double c = 0.25 * D * D * (rz_lambda_max(radial_grid(), modes) + wp2);
        if (c >= 1.0)
            throw std::runtime_error("unstable: dz^2 (lambda_max + wp^2) / 4 = " + std::to_string(c) +
                                     " >= 1 (geometry = rz, rip_cc; lambda_max grows with the modes near the axis: "
                                     "use numerics.solver = irip or a smaller dz)");
    }
    if (order != 1 && order != 2) throw std::runtime_error("numerics.shape must be 1 or 2 in geometry = rz");
    for (const LaserConfig& L : all_lasers()) {
        if (!L.enabled) continue;
        if (L.p < 0) throw std::runtime_error("laser.mode_p must be >= 0");
        // a linearly polarised LG_p^l lives in the modes |l - 1| and l + 1
        if (std::abs(L.l) + 1 >= modes)
            throw std::runtime_error("geometry = rz: a laser with mode_l = " + std::to_string(L.l) + " needs grid.modes >= " +
                                     std::to_string(std::abs(L.l) + 2));
    }
    for (auto& s : species) {
        if (s.self_field && periodic[0]) throw std::runtime_error("geometry = rz: self_field needs boundary_z = open");
        // t_start (external injection into a wake, decision 98): the solve adds to the fields present, as in Cart3D
        if (s.self_field && (s.u0[1] != 0 || s.u0[2] != 0))
            throw std::runtime_error("geometry = rz: self_field for a beam along z (ux = uy = 0)");
        for (int a = 0; a < 3; ++a)
            if (s.ppc[a] < 1) throw std::runtime_error("species " + s.name + ": ppc_z, ppc_r, ppc_theta must be >= 1");
    }
    if (slice_every > 0) throw std::runtime_error("geometry = rz: no slices; field output: output.openpmd_every");
    if (rings.on && solver != Solver::RipCC) throw std::runtime_error("rings (embedded metal): numerics.solver = rip_cc only");
    if (metal_rz.on && solver != Solver::RipCC) throw std::runtime_error("metal (embedded conductors): numerics.solver = rip_cc only");
    for (const LaserConfig& L : all_lasers())
        if (metal_rz.on && L.enabled && L.tfsf_x >= 0) throw std::runtime_error("metal: no laser.tfsf_x (the incident grid has no metal)");
    if ((!dielectrics.empty() || !dielectrics3.empty()) && solver != Solver::RipCC) throw std::runtime_error("dielectric: numerics.solver = rip_cc only (iRIP: the medium block)");
    if (x_abs_cells > 0 && window.enabled) throw std::runtime_error("grid.boundary_z = absorbing: not with a moving window");
    for (const LaserConfig& L : all_lasers())
        if (L.enabled && L.tfsf_x >= 0 && window.enabled) throw std::runtime_error("geometry = rz: laser.tfsf_x not with a moving window");
    if ((checkpoint.every > 0 || checkpoint.stop_hours > 0) && !hdf5_available())
        throw std::runtime_error("the checkpoint block needs a build with HDF5 (make HDF5=1)");
    if (checkpoint.every < 0 || checkpoint.keep < 0 || checkpoint.stop_hours < 0)
        throw std::runtime_error("checkpoint.every, keep and stop_after_hours must be >= 0");
    if (openpmd.every > 0 && !hdf5_available()) throw std::runtime_error("output.openpmd_every needs a build with HDF5 (make HDF5=1)");
    if (openpmd.every < 0) throw std::runtime_error("output.openpmd_every must be >= 0");
    for (const std::string& n : openpmd.fields)
        if (n != "E" && n != "B" && n != "J" && n != "rho" && n.rfind("rho_", 0) != 0)
            throw std::runtime_error("output.openpmd_fields: unknown field " + n + " (E B J rho rho_<species>)");
    check_window();
}

void Config::summary(FILE* f, int ranks) const {
    if (rz) {
        std::fprintf(f, "%s  (RZ, %d azimuthal modes, %s solver, c dt = dz)\n", solver == Solver::IRip ? "VLPL-iRZ" : "VLPL-RZ", modes,
                     solver == Solver::IRip ? "iRIP" : "RIP-CC");
        const RadialGrid g = radial_grid();
        std::fprintf(f, "  grid nz x nr = %d x %d, dz = %.4g, dr = %.4g (1/k0), wall at r = %.4g, steps = %ld, ranks = %d\n",
                     N[0], N[1], h[0], h[1], g.R(), long(std::ceil(t_end / dt() - 0.5)), ranks);
        if (!g.uniform) {
            // the rule of docs/specs/rz_modes.md step 6: k_perp dr <~ 1 wherever radiation must leave
            const double dmax = g.dn[size_t(N[1] - 1)];
            int j1 = 0;
            while (j1 < N[1] && g.dn[size_t(j1)] * 1.0 <= 1.0) ++j1;            // k0 dr <= 1 (code units k0 = 1)
            std::fprintf(f, "  stretched radial grid: dr %.4g up to r = %.4g, then x %.4g per cell up to %.4g; cutoff k_perp = "
                            "2/dr_max = %.4g k0; k0 dr <= 1 up to r = %.4g (radiation with k_perp dr > 1 is reflected)\n",
                         h[1], rz_r_uniform, rz_stretch, dmax, 2 / dmax, g.r[size_t(j1)]);
        }
        std::fprintf(f, "  pusher %s, boundary z %s\n", pusher == Pusher::Boris ? "boris" : "higuera_cary",
                     periodic[0] ? "periodic" : "open");
        for (const LaserConfig& L : all_lasers())
            if (L.enabled)
                std::fprintf(f, "  laser: a0 = %g, pol %c, duration %.4g, delay %.4g, waist %.4g, focus %.4g, mode %d %d, "
                             "phase %.4g (1/omega0, 1/k0)\n", L.a0, L.pol == 'y' ? 'x' : 'y', L.tau, L.tc, L.w0, L.xf, L.p,
                             L.l, L.phase);
        for (auto& s : species) {
            std::fprintf(f, "  species %s: q = %g, m = %g, n = %g n_c, ppc z r theta = %d %d %d, u0 (z x y) = %g %g %g\n",
                         s.name.c_str(), s.q, s.m, s.density, s.ppc[0], s.ppc[1], s.ppc[2], s.u0[0], s.u0[1], s.u0[2]);
            if (!s.density_fn.empty()) std::fprintf(f, "    density_function(z, r, x, y, phi) = %s\n", s.density_fn.text().c_str());
            if (s.t_start > 0) std::fprintf(f, "    loaded at t = %.4g (1/omega0)\n", s.t_start);
            if (s.rigid_until > 0) std::fprintf(f, "    rigid (momenta frozen) until t = %.4g (1/omega0)\n", s.rigid_until);
        }
        for (const BeamDiagConfig& b : diags)
            std::fprintf(f, "  diagnostic %s: species %s, gamma >= %g, every %ld steps\n", b.name.c_str(), b.species.c_str(),
                         b.gamma_min, b.every);
        structure_summary(f, *this);                 // quadrupoles, beams from files (src/structure.cpp)
        return;
    }
    if (solver == Solver::Yee) std::fprintf(f, "VLPL  (Yee solver, cfl %.4g)\n", cfl);
    else if (solver == Solver::IRip) std::fprintf(f, "VLPL-iRIP  (iRIP solver, box closure, c dt = dx)\n");
    else std::fprintf(f, "VLPL-RIP  (RIP-CC solver, c dt = dx)\n");
    if (flat) std::fprintf(f, "  geometry cartesian2d: one z cell, totals per slab dz = %.4g (1/k0)\n", h[2]);
    std::fprintf(f, "  grid %d x %d x %d, h = %.4g %.4g %.4g (1/k0), dt = %.4g, steps = %ld, ranks = %d\n",
                 N[0], N[1], N[2], h[0], h[1], h[2], dt(), long(std::ceil(t_end / dt() - 0.5)), ranks);
    if (sort_every < 0) std::fprintf(f, "  tiles %d x %d x %d cells, sort: auto\n", tile[0], tile[1], tile[2]);
    else std::fprintf(f, "  tiles %d x %d x %d cells, sort every %ld steps (0: off)\n", tile[0], tile[1], tile[2], sort_every);
    std::fprintf(f, "  courant dx^2(1/dy^2+1/dz^2) = %.3f, shape order %d, pusher %s\n",
                 h[0] * h[0] * (1 / (h[1] * h[1]) + (flat ? 0.0 : 1 / (h[2] * h[2]))), order,
                 pusher == Pusher::Boris ? "boris" : "higuera_cary");
    for (const LaserConfig& L : all_lasers())
        if (L.enabled)
            std::fprintf(f, "  laser: a0 = %g, pol %c, duration %.4g, delay %.4g, waist %.4g, focus %.4g, mode %d %d, "
                         "phase %.4g (1/omega0, 1/k0)\n", L.a0, L.pol, L.tau, L.tc, L.w0, L.xf, L.p, L.l, L.phase);
    for (const LaserFileConfig& F : file_lasers)
        std::fprintf(f, "  %s: file %s (%s), file plane x = %.4g, centre y z = %.4g %.4g (1/k0)%s\n", F.name.c_str(),
                     F.path.c_str(), F.mesh.c_str(), F.file_x, F.yc, F.zc, F.a0 > 0 ? ", a0 rescaled" : "");
    if (pml.on[1] || pml.on[2])
        std::fprintf(f, "  pml: %s%s, %d cells, R0 %g, order %d\n", pml.on[1] ? "y" : "", pml.on[2] ? "z" : "", pml.cells,
                     pml.R0, pml.order);
    structure_summary(f, *this);                     // metal, quadrupoles, beams from files (src/structure.cpp)
    if (xflux.on) std::fprintf(f, "  xflux: energy through the x edges in xflux.txt%s\n", xflux.every > 0 ? ", field records in xflux/" : "");
    for (const HFocusConfig& H : hfocus)
        std::fprintf(f, "  harmonic_focus%s%s: orders %g .. %g (Butterworth, %d poles per edge), peak and fluence per node in "
                        "harmonic_focus%s%s/\n", H.name.empty() ? "" : " ", H.name.c_str(), H.n_min, H.n_max, H.order,
                     H.name.empty() ? "" : "_", H.name.c_str());
    if (side.on && side.wave == 0)
        std::fprintf(f, "  side: TFSF tube, faces%s%s (total-field nodes y %d..%d, z %d..%d), plane wave a0 %g, angle %.4g, elevation "
                        "%.4g rad, polarisation %c, duration %.4g (0: continuous), delay %.4g, through (%.4g %.4g %.4g) (1/k0)\n",
                     side.faces[1] ? " y" : "", side.faces[2] ? " z" : "", side.lo[1], side.hi[1], side.lo[2], side.hi[2],
                     side.a0, side.angle, side.elevation, side.pol, side.tau, side.tc, side.r0[0], side.r0[1], side.r0[2]);
    if (side.on && side.wave == 1)
        std::fprintf(f, "  side: TFSF tube, faces%s%s (total-field nodes y %d..%d, z %d..%d), flying focus (luminal Bessel pulse) a0 %g "
                        "on the axis, cone angle %.4g rad, 1/e half length %.4g, centre x %.4g at t = 0, axis (%.4g %.4g), "
                        "polarisation %c, cep %.4g (1/k0)\n",
                     side.faces[1] ? " y" : "", side.faces[2] ? " z" : "", side.lo[1], side.hi[1], side.lo[2], side.hi[2],
                     side.a0, side.cone, side.len, side.r0[0], side.r0[1], side.r0[2], side.pol, side.phase);
    if (side.on && side.density > 0)
        std::fprintf(f, "  side: background plasma %g n_c: the incident wave of that plasma, a cold fluid %s\n", side.density,
                     side.fluid_all ? "everywhere" : "outside the kinetic region (2 cells inside the faces)");
    if (window.enabled) std::fprintf(f, "  moving window: start %.4g (1/omega0), velocity %g c\n", window.start, window.v);
    for (const BeamDiagConfig& b : diags)
        std::fprintf(f, "  diagnostic %s: species %s, gamma >= %g, every %ld steps, %d bins to %g MeV%s\n", b.name.c_str(),
                     b.species.c_str(), b.gamma_min, b.every, b.bins, b.e_max_mev, b.particles ? ", particles" : "");
    for (auto& s : species)
    {
        std::fprintf(f, "  species %s: q = %g, m = %g, n = %g n_c, ppc = %d %d %d, u0 = %g %g %g\n", s.name.c_str(),
                     s.q, s.m, s.density, s.ppc[0], s.ppc[1], s.ppc[2], s.u0[0], s.u0[1], s.u0[2]);
        if (!s.density_fn.empty()) std::fprintf(f, "    density_function = %s\n", s.density_fn.text().c_str());
        if (s.self_field)
            std::fprintf(f, "    self_field: %s at beta = %.8g\n", beam_steady && solver == Solver::RipCC && !pml.on[1] && !pml.on[2] ?
                         "the discrete steady state of RIP-CC" : "the Gauss-law field", s.u0[0] / std::sqrt(1 + s.u0[0] * s.u0[0]));
        if (s.t_start > 0) std::fprintf(f, "    loaded at t = %.4g (1/omega0)\n", s.t_start);
        if (s.rigid_until > 0) std::fprintf(f, "    rigid (momenta frozen) until t = %.4g (1/omega0)\n", s.rigid_until);
    }
}

// species.rigid_until (decision 98): before that time the push may move the particles (with the momenta it gives them
// for this step, a change of ~E dt / u of the direction) but their momenta are restored afterwards, so the species
// moves ballistically and deposits its current while the fields relax to the solver's discrete steady state.
bool Simulation::rigid_save(size_t i) {
    if (!(t < cfg.species[i].rigid_until)) return false;
    Species& sp = species[i];
    const long n = long(sp.size());
    const int A[3] = {Species::UX, Species::UY, Species::UZ};
    for (int a = 0; a < 3; ++a) {
        if (rigid_u[a].size() < size_t(n)) rigid_u[a].resize(size_t(n));
        double* d = rigid_u[a].data();
        const double* u = sp.ptr<double>(A[a]);
        par_for(n, [=] HD(long q) { d[q] = u[q]; });
    }
    return true;
}

void Simulation::rigid_restore(size_t i) {
    Species& sp = species[i];
    const long n = long(sp.size());
    const int A[3] = {Species::UX, Species::UY, Species::UZ};
    for (int a = 0; a < 3; ++a) {
        const double* d = rigid_u[a].data();
        double* u = sp.ptr<double>(A[a]);
        par_for(n, [=] HD(long q) { u[q] = d[q]; });
    }
}

// ---------------------------------------------------------------- core operators
// A fully periodic Cart3D box with a self_field species: the uniform co-moving neutralizing background of
// remove_mean_current (decision 98)
static bool periodic_beam(const Config& c) {
    if (c.rz || !c.periodic[0] || !c.periodic[1] || !c.periodic[2]) return false;
    for (auto& s : c.species)
        if (s.self_field) return true;
    return false;
}

// Gather + push + move + Esirkepov deposit in one kernel (decision 15), then the current guards are
// added into the neighbours.
class PushDeposit : public Operator {
public:
    const char* name() const override { return "push_deposit"; }
    Stage stage() const override { return Stage::Push; }
    void setup(Simulation& s) override { overlap_ = s.push_overlap(); }
    void apply(Simulation& s) override {
        s.fields->zero_currents();
        if (overlap_) { apply_overlap(s); return; }
        for (size_t i = 0; i < s.species.size(); ++i) {
            if (!s.cfg.species[i].mobile) continue;           // static: no push, no current
            if (!s.split.empty() && s.split[i]) { qed_push(s, i); continue; }   // spin / QED: the move in stage MoveDeposit
            const bool rigid = s.rigid_save(i);                                 // species.rigid_until (decision 98)
            push_and_deposit_species(s, i, s.cfg.pusher);
            if (rigid) s.rigid_restore(i);
        }
        if (s.qed) return;                             // the rest after the move and deposit (MoveDeposit, decision 15)
        push_deposit_tail(s);
    }
private:
    int overlap_ = 0;                                  // Simulation::push_overlap
    std::vector<Buf<unsigned char>> in_;               // per mobile species: 1 = interior particle (push_and_deposit_part)
    // GPU, several ranks (Simulation::push_overlap): the boundary particles first (high-priority stream), then the interior
    // ones in a low-priority stream; meanwhile the current guards are summed into the neighbours and (overlap_ == 2,
    // the plain RIP-CC step) copied back into the guards, the exchange that starts the field step (Fields::j_ready).
    // The interior particles touch neither the guards nor the edge layers these exchanges write. Per particle the same
    // kernel; the currents differ from the one-kernel push only in the order of the atomic additions (on the GPU that
    // order is not fixed anyway).
    void apply_overlap(Simulation& s);
};

// Stage MoveDeposit (decision 15): the species pushed by qed_push move and deposit (Pusher::None), then the rest of the push
// stage. Runs only with spin / QED (qed_setup adds it).
class MoveDeposit : public Operator {
public:
    const char* name() const override { return "move_deposit"; }
    Stage stage() const override { return Stage::MoveDeposit; }
    void apply(Simulation& s) override {
        for (size_t i = 0; i < s.species.size(); ++i)
            if (s.cfg.species[i].mobile && s.split[i]) push_and_deposit_species(s, i, Pusher::None);
        push_deposit_tail(s);
    }
};

std::unique_ptr<Operator> make_move_deposit_operator(bool rz) {
    return rz ? make_rz_move_deposit_operator() : std::make_unique<MoveDeposit>();
}

void push_and_deposit_species(Simulation& s, size_t i, Pusher pusher) {
    Species& sp = s.species[i];
    if (s.tiled_deposit) s.strays += push_and_deposit_tiled(sp, *s.fields, s.dom, pusher, s.cfg.order, s.tiles);
    else if (s.box_deposit) s.strays += push_and_deposit_box(sp, *s.fields, s.dom, pusher, s.cfg.order, s.tiles);
    else push_and_deposit(sp, *s.fields, s.dom, pusher, s.cfg.order);
}

// after the deposit of all species: the ionization currents, the guards, the filter, the side tube's cold fluid
void push_deposit_tail(Simulation& s) {
    {
        ionization_currents(s);                        // the displacements of this step's new electrons (ionization.hpp)
        prof::barrier(s.comm);                         // VLPL_PROFILE_SYNC: the imbalance of the push
        { PROF_SCOPE("push.add_guards J"); s.comm.add_guards(s.dom, {&s.fields->jx, &s.fields->jy, &s.fields->jz}); }
        if (s.cfg.filter_passes > 0) s.filter_currents();          // O4 (docs/specs/solid_target.md)
        if (s.cfg.absorber_mode == ABSORBER_REMOVE && (s.cfg.pml.on[1] || s.cfg.pml.on[2])) {   // no current in the y/z PML (decision 111)
            const int L[3] = {0, s.cfg.pml.on[1] ? s.cfg.pml.cells : 0, s.cfg.pml.on[2] ? s.cfg.pml.cells : 0};
            absorber_zero_currents_cart(*s.fields, s.dom, L);
        }
        if (periodic_beam(s.cfg)) remove_mean_current(*s.fields, s.dom, s.comm);   // decision 98 (beam_field.hpp)
        Fields& f = *s.fields;
        if (f.side && f.side->fluid()) {               // the cold fluid of a background plasma (side.hpp) with E^n
            const FV E[3] = {f.ex.view(), f.ey.view(), f.ez.view()}, Jf[3] = {f.jfx.view(), f.jfy.view(), f.jfz.view()},
                     J[3] = {f.jx.view(), f.jy.view(), f.jz.view()};
            f.side->fluid_current(E, Jf, J);
        }
    }
}

void PushDeposit::apply_overlap(Simulation& s) {
        std::vector<Species*> mob;
        for (size_t i = 0; i < s.species.size(); ++i)
            if (s.cfg.species[i].mobile) mob.push_back(&s.species[i]);
        if (in_.size() < mob.size()) in_.resize(mob.size());
        for (size_t k = 0; k < mob.size(); ++k)                // before the stream order is set up: resize zeroes (default stream)
            if (in_[k].size() < mob[k]->size()) in_[k].resize(mob[k]->size() + mob[k]->size() / 4);
        // the boundary part (it marks the interior particles) and the guard exchanges in the high-priority stream (its
        // blocks go first), after the zeroed currents of the default stream; the interior part in the low-priority one,
        // after the boundary part
        const auto hi = rip_side_stream(1), lo = rip_side_stream(0);
        rip_stream_after(hi, 0);
        Fields& f = *s.fields;
        {
            StreamGuard g(hi);
            {
                PROF_SCOPE("push.boundary");
                for (size_t k = 0; k < mob.size(); ++k)
                    push_and_deposit_part(*mob[k], f, s.dom, s.cfg.pusher, s.cfg.order, 1, in_[k].data());
            }
            rip_stream_after(lo, hi);
            {
                PROF_SCOPE("push.interior launch");
                StreamGuard gl(lo);
                for (size_t k = 0; k < mob.size(); ++k)
                    push_and_deposit_part(*mob[k], f, s.dom, s.cfg.pusher, s.cfg.order, 2, in_[k].data());
            }
            { PROF_SCOPE("push.add_guards J"); s.comm.add_guards(s.dom, {&f.jx, &f.jy, &f.jz}); }
            if (overlap_ == 2) {
                PROF_SCOPE("push.xchg J");
                s.comm.exchange(s.dom, {&f.jx, &f.jy, &f.jz});
                f.j_ready = true;
            }
        }
        PROF_SCOPE("push.interior wait");
        device_sync();
}

// The overlapped push (PushDeposit::apply_overlap) for: the GPU build with several ranks, the global (atomic)
// deposit, and nothing else that writes the currents between the push and the field step (filter, ionization, side
// tube, media, boosted-frame antenna). 2: the plain RIP-CC step, whose first exchange (the currents) moves into the push.
int Simulation::push_overlap() const {
    if (!RIP_GPU || comm.size < 2 || cfg.rz || cfg.mover_implicit || tiled_deposit || box_deposit) return 0;
    if (cfg.filter_passes > 0 || ionization || medium || cfg.boost.on || (fields && fields->side) || qed_active(cfg)) return 0;
    const char* e = vlpl_getenv("VLPL_OVERLAP");
    if (e && std::atoi(e) == 0) return 0;
    return dom.solver == Solver::RipCC && fields && !fields->pml ? 2 : 1;
}

// O6 (numerics.mover = implicit): the implicit particle push and the field step in one nonlinear solve (implicit_mover.hpp)
class ImplicitStep : public Operator {
public:
    const char* name() const override { return "implicit_mover"; }
    Stage stage() const override { return Stage::Push; }
    void apply(Simulation& s) override { implicit_mover_step(s); }
};

class FieldStep : public Operator {
public:
    const char* name() const override { return "rip_cc"; }
    Stage stage() const override { return Stage::Fields; }
    void apply(Simulation& s) override { s.fields->step(s.comm, s.hooks, s.t); }
};

class Migrate : public Operator {
public:
    const char* name() const override { return "migrate"; }
    Stage stage() const override { return Stage::Migrate; }
    void apply(Simulation& s) override {
        std::vector<Species*> v;
        const int L[3] = {0, s.cfg.pml.on[1] ? s.cfg.pml.cells : 0, s.cfg.pml.on[2] ? s.cfg.pml.cells : 0};
        const bool rm = s.cfg.absorber_mode != ABSORBER_KEEP && (L[1] > 0 || L[2] > 0);
        for (auto& sp : s.species) {
            if (rm) {                                  // grid.absorber_particles = remove (absorber_particles.cpp)
                double n = 0;
                sp.lost_charge += absorber_remove_cart(sp, s.dom, L, s.cfg.order, s.cfg.absorber_mode, n);
                sp.lost_count += n;
            }
            v.push_back(&sp);
        }
        migrate(v, s.dom, s.comm);                     // all species in one message per direction
    }
};

// Stable sort by tile after the migration, every sort_every steps (decision 14).
class TileSort : public Operator {
public:
    const char* name() const override { return "tile_sort"; }
    Stage stage() const override { return Stage::Migrate; }
    void apply(Simulation& s) override {
        if ((s.istep + 1) % s.sort_every) return;
        for (auto& sp : s.species) sort_by_tile(sp, s.dom, s.tiles);
    }
};

Simulation::Simulation(const Config& c, Comm& cm) : cfg(c), comm(cm) {
    if (c.rz) { setup_rz(); return; }
    for (int a = 0; a < 3; ++a) { dom.N[a] = c.N[a]; dom.h[a] = c.h[a]; dom.periodic[a] = c.periodic[a]; }
    dom.flat = c.flat;
    dom.solver = c.solver;
    dom.o5 = c.o5;
    dom.filt_comp = c.filter_comp; dom.filt_xonly = c.filter_xonly;
    set_flat_filter(dom, c.filter_passes, c.filter_flat);
    if (c.solver == Solver::Yee) dom.tau = c.dt();      // RIP-CC keeps tau = 0: dt() = h_x exactly
    // guards: the shape-2 current reaches 2 cells beyond the cell of a particle plus its move, at most
    // dt / h cells per axis (more than one only with iRIP's h_perp < c dt); shape 3 one more, and its particles stay
    // 2 cells below an open lower edge and 1 above an open upper one (support of 2 cells, docs/specs/solid_target.md)
    const int reach = c.order == 3 ? 4 : 2;
    dom.ng = c.order == 3 ? 5 : 3;
    for (int a = 0; a < (c.flat ? 2 : 3); ++a) dom.ng = std::max(dom.ng, reach + int(std::ceil(dom.dt() / dom.h[a] - 1e-12)));
    if (c.order == 3) { dom.keep_below = 2; dom.keep_above = 1; }
    int want[3] = {c.mpi[0], c.mpi[1], c.mpi[2]};
    if (c.solver == Solver::IRip) {
        // the DFT in y, z needs whole planes: ranks along x only (docs/specs/implicit_rip.md)
        if (want[1] > 1 || want[2] > 1)
            throw std::runtime_error("numerics.solver = irip: ranks along x only (grid.mpi_y = mpi_z = 1)");
        want[1] = want[2] = 1;
    }
    comm.decompose(dom, want);
    fields = new Fields(dom);
    fields->o5 = c.o5;
    if (c.x_abs_cells > 0) {                       // O3: kappa as the RZ layer, a wave crossing it twice is damped by R0
        if (c.window.enabled || !c.file_lasers.empty())
            throw std::runtime_error("grid.boundary_x = absorbing: not with a moving window or a laser file (they enter through the left edge)");
        const double a = c.x_abs_cells * dom.h[0] / dom.dt();
        fields->xabs = {c.x_abs_cells, 1.5 * std::log(1 / c.x_abs_R0) / a};
    }
    if (c.pml.on[1] || c.pml.on[2]) {
        // iRIP: b >= 1e-4 (sigma dt <= ln 1e4) keeps the eigenbasis of its pml axes well conditioned
        pml_ = std::make_unique<Pml>(dom, c.pml, dom.dt(), c.solver == Solver::IRip ? 1e-4 : 0.0);
        fields->pml = pml_.get();
    }
    if (!c.media.empty()) setup_medium();           // O7 (after the pml: its layers must stay free of media)
    if (c.side.on) {                                   // the incident plane wave: no periodic image during the run
        double diag = 0;
        for (int a = 0; a < 3; ++a) diag += dom.L(a) * dom.L(a);
        side_ = std::make_unique<SideTfsf>(dom, c.side, c.t_end + std::sqrt(diag));
        fields->side = side_.get();
        if (side_->fluid()) { fields->jfx.resize(dom); fields->jfy.resize(dom); fields->jfz.resize(dom); }
    }
    gauss0_.resize(dom); work_.resize(dom);
    setup_structure();                                 // metal, quadrupole table (src/structure.cpp)
    if (!cfg.dielectrics3.empty()) {                   // dielectrics in Cart3D / 2D (dielectric3d.hpp, decision 100)
        if (side_) throw std::runtime_error("dielectric: not with the side tube (side_injection)");
        diel3 = std::make_unique<Dielectric3>(cfg.dielectrics3, dom);
        fields->diel = diel3.get();
        // moving window: the 32-cell front absorbing layer of decision 100 (the grid-scale branch outruns c in a dielectric)
        if (cfg.window.enabled && !dom.periodic[0] && cfg.x_abs_cells == 0) { fields->dfront = 32; fields->dfront_kappa = 1.5 * std::log(1e3) / 32; }
        if (comm.rank == 0) std::printf("%s", diel3->describe().c_str());
    }
    // the projection filter needs y, z periodic (no pml, no side tube) and ranks along x only (decision 69); otherwise the
    // local filter, whose change of J radiates at the grid scale (decision 71)
    if (c.filter_passes > 0 && !c.filter_local && !c.filter_binomial &&
        (comm.dims[1] != 1 || comm.dims[2] != 1 || !dom.periodic[1] || !dom.periodic[2] || pml_ || side_ || dom.N[0] < 3 * comm.dims[0])) {
        cfg.filter_local = true;
        if (comm.rank == 0)
            std::fprintf(stderr, "warning: numerics.filter = projection needs y, z periodic (no pml, no side tube) and ranks along "
                                 "x only (grid.mpi_y = mpi_z = 1): the local filter is used, whose change of J radiates at the grid "
                                 "scale (decision 71); set numerics.filter = local to silence this\n");
    }
    if (c.filter_passes > 0) filt_.resize(dom);
    if (c.mover_implicit) pipeline.add(std::make_unique<ImplicitStep>());   // O6: particles and fields together
    else {
        pipeline.add(std::make_unique<PushDeposit>());
        if (c.boost.on)                                 // LASY lasers of a boosted run: on every rank (the antenna crosses them all)
            for (const LaserFileConfig& F : c.file_lasers)
                file_lasers.push_back(std::make_shared<FileLaser>(F, c.lambda0, dom.lo(1), dom.hi(1), dom.lo(2), dom.hi(2)));
        if ((antenna = make_boost_antenna(*this))) pipeline.add(make_boost_antenna_operator());   // its current after the deposit
        pipeline.add(std::make_unique<FieldStep>());
    }
    if (medium && medium->bc.ion) pipeline.add(make_medium_ionize_operator());   // O7 bound: electron-ion pairs (before the migration)
    if ((metal3 || metal_mask_) && cfg.metal.absorb) pipeline.add(make_metal_absorb_operator());   // metal.particles = absorb (metal3d.hpp)
    if (diel3 && fields->dfront > 0) pipeline.add(make_front_layer_watch());   // dielectric + window (decision 100)
    pipeline.add(std::make_unique<Migrate>());
    tiles = tile_grid(dom, c.tile);
    const bool can_sort = c.sort_every != 0;          // auto (-1) sorts if the deposit needs it
    // numerics.deposit: auto = tiled on the CPU when the colouring is safe; the GPU keeps the atomic path
    if (c.deposit == Config::Deposit::Tiled) {
        if (!can_sort) throw std::runtime_error("numerics.deposit = tiled needs numerics.sort_every >= 1");
        if (!tiled_deposit_safe(tiles, dom))
            throw std::runtime_error("numerics.deposit = tiled needs tiles of >= 6 cells and dt <= h along every axis");
        tiled_deposit = true;
    } else if (c.deposit == Config::Deposit::Box) {
        // the box covers the reach [a - 2, b + 2] only if a particle moves less than one cell per step
        if (!can_sort) throw std::runtime_error("numerics.deposit = box needs numerics.sort_every >= 1");
        for (int a = 0; a < 3; ++a)
            if (dom.dt() > dom.h[a]) throw std::runtime_error("numerics.deposit = box needs dt <= h along every axis");
        box_deposit = true;
    } else if (c.deposit == Config::Deposit::Auto) {
        // only if every colour keeps all threads busy: with fewer tiles than threads in a colour the
        // others wait (32 x 16 x 16 cells, 2 tiles per colour, 20 threads: 4x slower than global)
        int threads = 1;
#ifdef _OPENMP
        threads = omp_get_max_threads();
#endif
        tiled_deposit = !RIP_GPU && !c.mover_implicit && can_sort && tiled_deposit_safe(tiles, dom) &&
                        min_tiles_per_colour(tiles) >= threads;
    }
    // numerics.sort_every = auto: sort every step only for a deposit that needs sorted particles; the
    // global deposit does not, and on the GPU the host sort cost 1.9 of 2.5 s (fourth GPU check)
    sort_every = c.sort_every >= 0 ? c.sort_every : (tiled_deposit || box_deposit ? 1 : 0);
    if (sort_every > 0) pipeline.add(std::make_unique<TileSort>());
}

Simulation::~Simulation() {
    if (modes_f_) std::fclose(modes_f_);
    if (medium_f) std::fclose(medium_f);
    delete fields; delete rz;
}

SlabProfile SpeciesConfig::slab() const {
    SlabProfile P;
    P.x0 = slab_x0; P.y0 = slab_y0; P.c = std::cos(slab_angle); P.s = std::sin(slab_angle);
    P.T = slab_thickness;
    P.Lf = slab_ramp; P.Ef = slab_ramp > 0 ? slab_ramp_extent : 0.0;
    P.Lr = slab_rear_ramp; P.Er = slab_rear_ramp > 0 ? slab_rear_extent : 0.0;
    P.exact = slab_exact || slab_ramp > 0 || slab_rear_ramp > 0;
    return P;
}

unsigned long long mix64(unsigned long long x) {               // splitmix64
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Regular positions in each owned cell; the momenta use a generator seeded by the absolute cell, so
// the state does not depend on the decomposition, nor on whether a cell was loaded at the start or
// entered with the moving window. Cold species load in kernels (load.hpp: the same particles in the
// same order).
void Simulation::load(const SpeciesConfig& sc, Species& s, int index, int i0, int i1) {
    load_cells(sc, s, index, dom.cell0(0) + i0, i1 - i0, Lattice{});
}

// The cells gi0 .. gi0 + nx - 1 of the lattice along x (boost.hpp; plain: the absolute cells of the grid) and the owned
// cells in y, z. With lat.boost the momenta of the deck are lab momenta (u0_lab, uth): each particle gets the boosted
// momentum, and the weight the density factor gamma (1 - beta v_x) of the drift u0_lab: the same for every species at a
// lattice point (neutral loading, exact Gauss law; the flux factor 1 - beta v_x of the thermal spread is left out,
// decision 87).
void Simulation::load_cells(const SpeciesConfig& sc, Species& s, int index, long gi0, int nx, const Lattice& lat) {
    if (nx <= 0) return;
    if (sc.density == 0) return;                       // an empty species (e.g. filled by ionization: medium model = bound)
    const int ppc = sc.ppc[0] * sc.ppc[1] * sc.ppc[2];
    const double w = sc.density * dom.cell_volume() / ppc;
    double fmax = 0;
    const bool hot = sc.uth[0] != 0 || sc.uth[1] != 0 || sc.uth[2] != 0;
    if (!hot) {
        ColdLoad L{};
        for (int q = 0; q < 3; ++q) {
            L.ppc[q] = sc.ppc[q]; L.h[q] = dom.h[q]; L.u0[q] = sc.u0[q];
            L.rmin[q] = sc.rmin[q]; L.rmax[q] = sc.rmax[q];
        }
        L.w = w;
        L.slab = sc.slab();
        L.ax = lat.ax; L.bx = lat.bx; L.sx = lat.sx; L.qlo = lat.qlo; L.qhi = lat.qhi;
        if (lat.boost) L.w = w * boost_momentum(lat.g, lat.b, sc.u0_lab, L.u0);
        const long g0[3] = {gi0, dom.off[1], dom.off[2]};
        const int nl[3] = {nx, dom.n[1], dom.n[2]};
        fmax = load_cold(s, L, sc.density_fn, g0, nl);
    } else {
        const SlabProfile P = sc.slab();
        const double hs[2] = {lat.ax * dom.h[0] / sc.ppc[0], dom.h[1] / sc.ppc[1]};
        std::vector<double> h[Species::NC];
        std::mt19937_64 rng;
        for (int k = 0; k < dom.n[2]; ++k)
            for (int j = 0; j < dom.n[1]; ++j)
                for (int i = 0; i < nx; ++i) {
                    const long gi = gi0 + i, gj = dom.off[1] + j, gk = dom.off[2] + k;
                    rng.seed(mix64(cfg.seed ^ mix64(index + 1000003ULL * (gi + 1000003ULL * (gj + 1000003ULL * gk)))));
                    std::normal_distribution<double> g(0.0, 1.0);       // fresh per cell (it caches a value)
                    for (int c = 0; c < sc.ppc[2]; ++c)
                        for (int b = 0; b < sc.ppc[1]; ++b)
                            for (int a = 0; a < sc.ppc[0]; ++a) {
                                const double q0 = (gi + (a + 0.5) / sc.ppc[0]) * dom.h[0];
                                const double r[3] = {lat.ax * q0 + lat.bx, (gj + (b + 0.5) / sc.ppc[1]) * dom.h[1],
                                                     (gk + (c + 0.5) / sc.ppc[2]) * dom.h[2]};
                                double u[3], wb = 1;
                                if (lat.boost) {                         // a lab momentum, boosted
                                    double ul[3];
                                    for (int q = 0; q < 3; ++q) ul[q] = sc.u0_lab[q] + sc.uth[q] * g(rng);
                                    boost_momentum(lat.g, lat.b, ul, u);   // the density factor of the drift (below): neutral loading
                                    double up0[3];
                                    wb = boost_momentum(lat.g, lat.b, sc.u0_lab, up0);
                                } else
                                    for (int q = 0; q < 3; ++q) u[q] = sc.u0[q] + sc.uth[q] * g(rng);
                                if (!(q0 >= lat.qlo && q0 < lat.qhi)) continue;
                                bool in = true;
                                for (int q = 0; q < 3; ++q) in = in && r[q] >= sc.rmin[q] && r[q] < sc.rmax[q];
                                if (!in) continue;
                                double f = 1;
                                if (P.T > 0) {                           // point test or sub-cell mean (slab.hpp)
                                    f = slab::factor(P, r, hs);
                                    if (!(f > 0)) continue;
                                }
                                if (!sc.density_fn.empty()) {
                                    const double v = sc.density_fn(r);
                                    if (!(v > 0)) continue;              // also NaN
                                    f *= v;
                                }
                                fmax = std::max(fmax, f);
                                const double v[Species::NC] = {q0 + lat.sx, r[1], r[2], u[0], u[1], u[2], w * f * wb};
                                for (int q = 0; q < Species::NC; ++q) h[q].push_back(v[q]);
                            }
                }
        s.append(h);
    }
    if (size_t(index) < peak_.size()) peak_[size_t(index)] = std::max(peak_[size_t(index)], fmax);
    // the omega_p dt check at the start uses the loaded peak (check_peak_density); later loads only warn
    if (fmax > 1 + 1e-12 && !warned_fmax_) {
        warned_fmax_ = true;
        std::fprintf(stderr, "warning: species %s: density_function reaches %.4g > 1 (the stability check of the deck used "
                             "density alone; the loaded peak is checked at the start)\n", sc.name.c_str(), fmax);
    }
}

// O4 (docs/specs/solid_target.md): omega_p dt at the largest density loaded at the start (density times the largest
// profile factor of every species over all ranks; species loaded later count with `density`). RIP-CC keeps
// dx^2 (1/dy^2 + 1/dz^2) + (omega_p dt)^2 / 4 < 1, iRIP and Yee omega_p dt < 2; above 1 a warning: the explicit current
// coupling oscillates at 2 asin(omega_p dt / 2) / dt.
void Simulation::check_peak_density() {
    double wp2 = 0;
    for (size_t i = 0; i < peak_.size(); ++i) {
        const SpeciesConfig& s = cfg.species[i];
        const double f = loaded[i] ? comm.allreduce_max(peak_[i]) : 1.0;
        if (cfg.frozen(s)) continue;                 // a frozen species of a collisions block carries no current
        wp2 += f * cfg.check_density_factor(s) * (s.density * s.q2() / s.m + s.wp2_electrons());
    }
    bool ramp = false;
    for (const SpeciesConfig& s : cfg.species) ramp = ramp || s.slab_ramp > 0 || s.slab_rear_ramp > 0;
    if (ramp && cfg.order == 1 && comm.rank == 0)     // decision 64: resonance absorption in the ramp, then shape-1 grid heating
        std::fprintf(stderr, "warning: a slab ramp with numerics.shape = 1: a resolved ramp hit by p-polarised light grew "
                             "numerically unstable with shape 1 (x600 in energy at lambda0/32); use shape 2 or 3\n");
    const double D = dom.dt(), wd = std::sqrt(wp2) * D;
    peak_wpdt = wd;
    if (!(wp2 > 0)) return;
    const double cour = cfg.solver == Solver::RipCC ? D * D * (1 / (dom.h[1] * dom.h[1]) + (dom.flat ? 0.0 : 1 / (dom.h[2] * dom.h[2]))) : 0.0;
    if (cfg.mover_implicit) return;                 // O6: no omega_p dt limit
    if (cour + 0.25 * wd * wd >= 1.0)
        throw std::runtime_error("unstable at the peak density: omega_p dt = " + std::to_string(wd) +
                                 (cfg.solver == Solver::RipCC ? ", dx^2 (1/dy^2 + 1/dz^2) + (omega_p dt)^2/4 = " +
                                  std::to_string(cour + 0.25 * wd * wd) + " >= 1" : std::string(" >= 2")) +
                                 " (sum n q^2/m = " + std::to_string(wp2) + " n_c; dx < lambda0 / (pi sqrt(n/n_c)), "
                                 "docs/specs/solid_target.md)");
    if (wd > 1 && comm.rank == 0)
        std::fprintf(stderr, "warning: omega_p dt = %.3f at the peak density: the plasma oscillation runs %.1f %% fast "
                             "(2 asin(omega_p dt / 2) / dt)\n", wd, 100 * (2 * std::asin(0.5 * wd) / wd - 1));
}

void Simulation::write_slices() {
    const int nx = dom.N[0], ny = dom.N[1];
    const int kg = cfg.slice_z < 0 ? dom.N[2] / 2 : std::min(dom.N[2] - 1, int(cfg.slice_z / dom.h[2]));
    const int kl = kg - dom.off[2];
    const bool mine = kl >= 0 && kl < dom.n[2];
    std::vector<double> buf(size_t(nx) * ny);
    const std::string sdir = cfg.dir + "/slices";
    if (comm.rank == 0) make_dirs(sdir);
    if (comm.rank == 0 && (istep == 0 || istep == restart_step)) {
        FILE* m = std::fopen((sdir + "/meta.txt").c_str(), "w");
        std::fprintf(m, "nx %d\nny %d\ndx %.10g\ndy %.10g\nz %.10g\nlambda0_m %.6g\nfields", nx, ny, dom.h[0],
                     dom.h[1], kg * dom.h[2], cfg.lambda0);
        for (auto& n : cfg.slice_fields) std::fprintf(m, " %s", n.c_str());
        std::fprintf(m, "\n");
        std::fclose(m);
    }
    const int lx = dom.n[0], ly = dom.n[1];
    Buf<double> plane(size_t(lx) * ly);
    std::vector<double> hp(size_t(lx) * ly);
    for (const std::string& name : cfg.slice_fields) {
        const Field* src = nullptr;
        double scale = 1;
        if (name == "Ex") src = &fields->ex; else if (name == "Ey") src = &fields->ey;
        else if (name == "Ez") src = &fields->ez; else if (name == "Bx") src = &fields->bx;
        else if (name == "By") src = &fields->by; else if (name == "Bz") src = &fields->bz;
        else if (name == "rho") { compute_rho(); src = &fields->rho; }
        else if (name.rfind("n_", 0) == 0) {
            size_t i = 0;
            while (i < species.size() && species[i].name != name.substr(2)) ++i;
            if (i == species.size()) throw std::runtime_error("output.slice_fields: no species " + name.substr(2));
            work_.fill(0);
            deposit_rho(species[i], work_, dom, cfg.order);
            comm.add_guards(dom, {&work_});
            src = &work_; scale = 1.0 / species[i].q;
        } else throw std::runtime_error("output.slice_fields: unknown field " + name);
        std::fill(buf.begin(), buf.end(), 0.0);
        if (mine) {
            const FV v = src->view();
            double* P = plane.data();
            par_for(long(lx) * ly, [=] HD(long q) { P[q] = scale * v(int(q % lx), int(q / lx), kl); });
            plane.to_host(hp.data(), 0, hp.size());
            for (int j = 0; j < ly; ++j)
                for (int i = 0; i < lx; ++i) buf[size_t(dom.off[1] + j) * nx + dom.off[0] + i] = hp[size_t(j) * lx + i];
        }
        comm.allreduce_sum(buf.data(), int(buf.size()));
        if (comm.rank == 0) {
            std::vector<float> fb(buf.begin(), buf.end());
            char fn[64];
            std::snprintf(fn, sizeof fn, "/%s_%06ld.f32", name.c_str(), istep);
            FILE* f = std::fopen((sdir + fn).c_str(), "wb");
            std::fwrite(fb.data(), sizeof(float), fb.size(), f);
            std::fclose(f);
        }
    }
    if (comm.rank == 0) {
        FILE* ix = std::fopen((sdir + "/index.txt").c_str(), istep == 0 ? "w" : "a");
        if (!ix) throw std::runtime_error("cannot write " + sdir + "/index.txt");
        std::fprintf(ix, "%ld %.8g %.8g\n", istep, t, dom.wx * dom.h[0]);   // x of window cell 0
        std::fclose(ix);
    }
}

void Simulation::compute_rho() {
    fields->rho.fill(0);
    for (auto& s : species) deposit_rho(s, fields->rho, dom, cfg.order);
    if (rho_abs_.n(0) > 0) {                       // the charge absorbed in a medium metal, at rest (structure.cpp, decision 124)
        const FV R = fields->rho.view(), A = rho_abs_.view();
        par_for(long(rho_abs_.size()), [=] HD(long q) { R.p[q] += A.p[q]; });
    }
    comm.add_guards(dom, {&fields->rho});
    if (cfg.filter_passes > 0) filter_rho_cart(fields->rho, filt_, dom, comm, cfg.filter_passes);   // the charge of the filtered current
    if (medium) {                                  // O7: the polarization charge -div P of the media
        comm.exchange(dom, {&medium->P[0], &medium->P[1], &medium->P[2]});
        if (medium->bound) comm.exchange(dom, {&medium->bP});
        medium_add_rho_cart(*medium, dom, fields->rho);
    }
    boost_antenna_add_rho(*this, fields->rho);    // the charge of the boosted antenna sheet (boost_antenna.cpp)
}

void Simulation::filter_currents() {
    if (cfg.filter_binomial && !cfg.filter_local) filter_currents_binomial(*fields, filt_, dom, comm, cfg.filter_passes);
    else if (cfg.filter_local) filter_currents_cart(*fields, filt_, dom, comm, cfg.filter_passes);
    else filter_currents_proj(*fields, filt_, fproj_, dom, comm, cfg.filter_passes);
}

// The steady self-field of a species marked self_field (docs/specs/beam_self_field.md) is added to
// the fields present: the solve is linear, so the Gauss law of the other charges is kept.
void Simulation::load_species(size_t i) {
    if (cfg.rz) {
        if (!cfg.species[i].file.path.empty()) load_file_species(i);   // a beam from a file (src/structure.cpp)
        else if (cfg.boost.on) load_rz_cells(cfg.species[i], species[i], int(i), dom.cell0(0), dom.n[0], boost_lattice(cfg, cfg.species[i], dom, t));
        else load_rz(cfg.species[i], species[i], int(i), 0, dom.n[0]);
        loaded[i] = 1;
        if (cfg.boost.on && cfg.species[i].boost_kind == 2 && cfg.species[i].density != 0 && cfg.species[i].file.path.empty())
            boost_beam_charge_check(*this, i);           // loaded vs the deck charge (decision 118)
        if (!cfg.species[i].self_field) return;
        work_.fill(0);                               // the charge as the Gauss check sees it (rz_rho_all), this species only
        rz_rho(species[i], *rz, work_, cfg.order);
        comm.add_guards(dom, {&work_});
        rz_smooth_rho(*rz, comm, work_);
        rz_filter_rho(*rz, comm, work_, cfg.filter_passes);
        const double u = cfg.species[i].u0[0];
        rz_add_beam_field(*rz, work_, u / std::sqrt(1 + u * u), comm);
        return;
    }
    if (!cfg.species[i].file.path.empty()) load_file_species(i);   // a beam from a file (src/structure.cpp)
    else if (cfg.boost.on) load_cells(cfg.species[i], species[i], int(i), dom.cell0(0), dom.n[0], boost_lattice(cfg, cfg.species[i], dom, t));
    else load(cfg.species[i], species[i], int(i), 0, dom.n[0]);
    loaded[i] = 1;
    if (cfg.boost.on && cfg.species[i].boost_kind == 2 && cfg.species[i].density != 0 && cfg.species[i].file.path.empty())
        boost_beam_charge_check(*this, i);               // loaded vs the deck charge (decision 118)
    if (!cfg.species[i].self_field) return;
    work_.fill(0);
    deposit_rho(species[i], work_, dom, cfg.order);
    comm.add_guards(dom, {&work_});
    const double u = cfg.species[i].u0[0];
    // the steady state of the plain RIP-CC step (periodic y, z): not with CPML layers (their own step)
    add_beam_field(*fields, work_, u / std::sqrt(1 + u * u), dom, comm, cfg.beam_steady && !cfg.pml.on[1] && !cfg.pml.on[2]);
}

// Loads the species with t_start > 0 at the first step with t_n >= t_start (stage Inject), sorted by
// tile if the deposit sorts: else its first push would take the atomic path (summation order of the
// threads, not bit-identical).
class DelayedLoad : public Operator {
public:
    const char* name() const override { return "delayed_load"; }
    Stage stage() const override { return Stage::Inject; }
    void apply(Simulation& s) override {
        for (size_t i = 0; i < s.species.size(); ++i)
            if (!s.loaded[i] && s.t >= s.cfg.species[i].t_start - 0.5 * s.dom.dt()) {
                s.load_species(i);
                if (s.sort_every > 0) sort_by_tile(s.species[i], s.dom, s.tiles);
            }
    }
};

// Warnings about the start of a moving window run (Cart3D and RZ: x or z is axis 0 of the particles).
void Simulation::warn_window_start() {
    // a later start leaves the right edge a plain open boundary until then: plasma currents in its
    // last cells break the exact Gauss law there (docs/specs/moving_window.md)
    double edge = 0;
    const double xe = (dom.N[0] - 3) * dom.h[0];
    for (auto& s : species)
        for (double x : s.get(Species::X)) edge += x >= xe;
    comm.allreduce_sum(&edge, 1);
    // a pulse that reaches the right edge before the first shift leaves its longitudinal field at
    // the last cell (no Gauss point while the edge stands still). The first shift comes after the step that ends at
    // start + h_x / v, when the head of a pulse entering at t = 0 is at c (start + h_x / v): that must not be beyond the
    // edge (decision 91: c start = L, as bench/lwfa_bubble had, cut the head and gave 1.05e-4 in the Gauss law and div B)
    bool any_laser = false;
    for (const LaserConfig& L : cfg.all_lasers()) any_laser = any_laser || L.enabled;
    if (comm.rank == 0 && any_laser && cfg.window.start + dom.h[0] / cfg.window.v > dom.L(0) * (1 + 1e-12))
        std::fprintf(stderr, "warning: the laser head reaches the right edge before the window moves (c window.start + h_x / v "
                             "> box length): the Gauss law is exact only if the window starts earlier\n");
    if (comm.rank == 0 && edge > 0 && cfg.window.start >= dom.dt())
        std::fprintf(stderr, "warning: plasma in the last 3 cells and window.start > 0: until the window moves the "
                             "right edge is a plain open boundary; unless that plasma stays cold and at rest, "
                             "the Gauss law is exact only with start = 0 or the plasma ahead of the box\n");
}

void Simulation::init(const std::string& restart) {
    restart_path = restart;
    species.resize(cfg.species.size());
    loaded.assign(species.size(), 0);
    peak_.assign(species.size(), 0.0);
    bool delayed = false;
    for (size_t i = 0; i < species.size(); ++i) {
        species[i].name = cfg.species[i].name; species[i].q = cfg.species[i].q; species[i].m = cfg.species[i].m;
    }
    collisions_register(*this);                    // id on the colliding species, before the load (collisions.hpp)
    for (size_t i = 0; i < species.size(); ++i) {
        if (cfg.species[i].ion.on || cfg.species[i].ion.states) ionization_register(species[i], cfg.species[i].ion);   // before the load: rho with Z
        qed_register(*this, i);                    // spin s0 s1 s2, id, chi; photon attributes (qed.hpp)
        if (cfg.species[i].t_start > 0) delayed = true;
        else if (restart.empty()) load_species(i);
    }
    if (delayed) pipeline.add(std::make_unique<DelayedLoad>());
    bool ionizable = false;
    for (const SpeciesConfig& sc : cfg.species) ionizable = ionizable || sc.ion.on;
    if (ionizable) pipeline.add(make_ionization_operator());   // stage Ionize (ionization.hpp)
    if (qed_active(cfg)) qed_setup(*this);         // stages Qed and MoveDeposit, the split push (qed.hpp, decision 93)
    collisions_add_operators(*this);               // stage Migrate, after the migration (collisions.hpp, decision 94)
    if (cfg.rz) {                                  // the state of a checkpoint as in Cart3D below
        if (restart.empty()) rz_gauss_residual(gauss0_);
        else if (rz->tfsf) throw std::runtime_error("geometry = rz: laser.tfsf_x (its incident grid) not with a restart yet");
        else read_checkpoint(*this, restart);
        if (cfg.boost.on && !cfg.labdiag.species.empty()) pipeline.add(make_lab_particle_operator());   // before the window, injection
        if (cfg.xflux.on) pipeline.add(make_xflux_operator());   // after the field step, before a window shift
        if (cfg.plane_z >= 0 || cfg.axis_every > 0) pipeline.add(make_rz_plane_operator());   // docs/specs/rz_metal.md
        for (const HFocusConfig& H : cfg.hfocus) pipeline.add(make_hfocus_operator(H));
        if (cfg.window.enabled) pipeline.add(make_window_operator());
        if (cfg.boost.on) pipeline.add(make_boost_inject_operator());   // after the window shifts
        if (cfg.window.enabled && restart.empty()) warn_window_start();
        for (const BeamDiagConfig& b : cfg.diags) pipeline.add(make_beam_diag_operator(b));
        structure_operators();                     // wake, beam_slices (src/structure.cpp, src/wake.cpp)
        if (cfg.boost.on && cfg.labdiag.fields && !cfg.labdiag.times.empty()) pipeline.add(make_lab_field_operator());
        pipeline.setup(*this);
        return;
    }
    if (fields->side) side_setup(restart.empty());
    if (restart.empty()) {
        if (metal3) {                              // no field inside the metal at the start; particles = absorb: no particle (metal3d.hpp)
            Fields& f = *fields;
            metal_zero_fields(dom, metal3->view(), f.ex.view(), f.ey.view(), f.ez.view(), f.bx.view(), f.by.view(), f.bz.view());
            comm.exchange(dom, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz});
            if (cfg.metal.absorb) metal_absorb_all();
        }
        if (metal_mask_) metal_absorb_all();        // irip: no particle inside the medium metal at the start; its charge stays
        check_peak_density();
        if (sort_every > 0)
            for (auto& s : species) sort_by_tile(s, dom, tiles);
        compute_rho();
        fields->gauss_residual(gauss0_);   // without self fields: -rho (0 if neutral); with them: round-off
    } else {
        // fields, particles (in their order, with the tile starts of the last sort), the window position,
        // the loaded species and the initial Gauss residual: the run continues as if never stopped
        read_checkpoint(*this, restart);
        compute_rho();
        if (comm.rank == 0 && cfg.slice_every > 0) {   // drop slice index lines written after the checkpoint
            bool fresh;
            make_dirs(cfg.dir + "/slices");
            std::fclose(open_log(cfg.dir + "/slices/index.txt", restart_step, fresh));
        }
    }
    // file lasers: built on the ranks at the open left x edge (rank 0 is one of them and reports); boosted runs: built with the
    // antenna on every rank; RZ: rz_sim.cpp
    if (!cfg.rz && !cfg.boost.on && comm.nbr[0][0] < 0)
        for (const LaserFileConfig& F : cfg.file_lasers)
            file_lasers.push_back(std::make_shared<FileLaser>(F, cfg.lambda0, dom.lo(1), dom.hi(1), dom.lo(2), dom.hi(2)));
    if (!cfg.rz)
        for (size_t i = 0; i < file_lasers.size(); ++i) {
            const LaserFileConfig& F = cfg.file_lasers[i];
            const FileLaser& L = *file_lasers[i];
            if (comm.rank == 0 && restart.empty()) {
                for (const std::string& line : L.info) std::printf("  %s: %s\n", F.name.c_str(), line.c_str());
                if (L.edge > 1e-3)
                    std::fprintf(stderr, "warning: %s: the envelope at the edge of its grid is %.2g of the peak (a pulse cut "
                                         "by the file, or wrapped around by the propagation)\n", F.name.c_str(), L.edge);
                const double ws = cfg.boost.on ? cfg.boost.start_lab : cfg.window.start;
                if (cfg.window.enabled && ws < L.t_last)
                    std::fprintf(stderr, "warning: %s enters x = 0 until (lab) t = %.4g, but the window moves from t = %.4g on and "
                                         "the injection stops then\n", F.name.c_str(), L.t_last, ws);
            }
        }
    // O1: lasers with tfsf_x come in through their TFSF plane (docs/specs/tfsf.md)
    for (const LaserConfig& L : cfg.all_lasers())
        if (L.enabled && L.tfsf_x >= 0 && !tfsf_) {
            // the incident grid holds state (no restart yet); the plane waves are closed forms of t (restarts need nothing)
            if (cfg.window.enabled || (!restart.empty() && !L.tfsf_modes))
                throw std::runtime_error("laser.tfsf_x: so far without a moving window, and restarts only with tfsf_incident = modes "
                                         "(docs/specs/tfsf.md)");
            const bool lay = cfg.pml.on[1] || cfg.pml.on[2];   // the incident grid with the same CPML layers
            double diag = 0;                                   // the plane waves: no periodic image of the pulse during the run
            for (int a = 0; a < (dom.flat ? 2 : 3); ++a) diag += dom.L(a) * dom.L(a);   // 2D: no path along z
            tfsf_ = std::make_unique<Tfsf>(dom, comm, cfg.all_lasers(), lay ? &cfg.pml : nullptr,
                                           cfg.solver == Solver::IRip ? 1e-4 : 0.0, cfg.t_end + std::sqrt(diag));
            fields->tfsf = tfsf_.get();
            if (cfg.o5) tfsf_->set_o5(comm);                  // the incident grid with the closure (decision 75)
            if (restart.empty()) tfsf_->start(*fields, comm);   // plane waves: the incident field already right of the plane
            if (fields->xabs.cells > 0) tfsf_->set_x_absorber(comm, fields->xabs);   // the incident grid with the same layers
        }
    add_laser_inflow(hooks, cfg.all_lasers(), cfg.boost.on ? std::vector<std::shared_ptr<FileLaser>>{} : file_lasers, dom, comm);
    if (cfg.boost.on && !cfg.labdiag.species.empty()) pipeline.add(make_lab_particle_operator());   // before the window, injection
    if (cfg.xflux.on) pipeline.add(make_xflux_operator());   // after the field step, before a window shift
    for (const HFocusConfig& H : cfg.hfocus) pipeline.add(make_hfocus_operator(H));
    if (cfg.window.enabled) pipeline.add(make_window_operator());
    if (cfg.boost.on) pipeline.add(make_boost_inject_operator());   // after the window shifts
    if (cfg.window.enabled && restart.empty()) warn_window_start();
    for (const BeamDiagConfig& b : cfg.diags) pipeline.add(make_beam_diag_operator(b));
    structure_operators();                         // wake, beam_slices, side_flux (src/structure.cpp)
    if (cfg.boost.on && cfg.labdiag.fields && !cfg.labdiag.times.empty()) pipeline.add(make_lab_field_operator());
    pipeline.setup(*this);
}

void Simulation::step() {
    pipeline.run(*this);
    ++istep;
    t = istep * dom.dt();
}

double Simulation::gauss_drift() {
    if (cfg.rz) { double w, g, r, b; rz_diagnostics(w, g, r, b); return g; }
    compute_rho();
    fields->gauss_residual(work_);
    if (window_frozen_rho(entry_rho_)) {             // the frozen entry columns of a window (decision 91)
        const FV O = work_.view(), R = entry_rho_.view();
        for_owned(dom, [=] HD(int i, int j, int k) { O(i, j, k) += R(i, j, k); });
    }
    const FV W = work_.view(), G = gauss0_.view();
    const int nx = dom.n[0], ny = dom.n[1];
    const int sp = fields->tfsf_point();                 // the TFSF plane: skipped (fields.cpp)
    const bool me = metal3 != nullptr;                   // the surface charge of the metal: skipped (metal3d.hpp)
    const Metal3View mv = me ? metal3->view() : Metal3View{};
    // a moving window (lab or boosted) with the current filter: the plasma leaves through the open left edge, where the filter's
    // closure differs from the interior; the offset stays frozen in the first filter_passes cells (as RZ, rz_gauss_istart)
    const int i0 = (cfg.window.enabled || cfg.boost.on) && cfg.filter_passes > 0 && !dom.periodic[0] && dom.off[0] == 0
                       ? cfg.filter_reach() : 0;
    const double m = par_max_abs(dom.owned(), [=] HD(long q) {
        const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
        if (me && mv.gauss_skip(i, j, k)) return 0.0;
        return i == sp || i < i0 ? 0.0 : W(i, j, k) - G(i, j, k);
    });
    return comm.allreduce_max(m);
}

void Simulation::diagnostics(FILE* f, bool header) {
    double wrz = 0, grz = 0, rrz = 0, brz = 0;
    if (cfg.rz) rz_diagnostics(wrz, grz, rrz, brz);
    double v[3 + 64] = {cfg.rz ? wrz : fields->energy(), 0, 0};
    const size_t ns = std::min<size_t>(species.size(), 64);
    for (size_t i = 0; i < ns; ++i) { v[1] += species[i].kinetic_energy(); v[3 + i] = double(species[i].size()); }
    comm.allreduce_sum(v, int(3 + ns));
    double g = grz, db = brz, rmax = rrz;
    if (!cfg.rz) {
        g = gauss_drift(); db = comm.allreduce_max(fields->max_div_b());
        const FV R = fields->rho.view();
        const int nx = dom.n[0], ny = dom.n[1];
        rmax = comm.allreduce_max(par_max_abs(dom.owned(), [=] HD(long q) {
            return R(int(q % nx), int((q / nx) % ny), int(q / (long(nx) * ny)));
        }));
    }
    if (medium) medium_diagnostics(*this, f != nullptr, v[0] + v[1]);   // dir/medium.txt (medium_sim.cpp; collective)
    if (ionization) ionization_diagnostics(*this, f != nullptr, v[0] + v[1]);   // dir/ionization.txt (collective)
    if (qed) qed_diagnostics(*this, f != nullptr);   // dir/qed.txt (collective)
    if (collisions) collisions_diagnostics(*this, f != nullptr);   // dir/collisions.txt (collective)
    std::vector<double> wm, gm;
    if (cfg.rz) rz_mode_diagnostics(wm, gm);
    if (!f) return;
    if (cfg.rz) {                                  // dir/rz_modes.txt: per mode, at the cadence of diag.txt
        if (!modes_f_) {
            bool fresh;
            modes_f_ = open_log(cfg.dir + "/rz_modes.txt", restart_step, fresh);
            if (fresh) {
                std::fprintf(modes_f_, "# step  t  W_0 .. W_%d (field energy per mode)  dGauss_0 .. dGauss_%d\n", cfg.modes - 1, cfg.modes - 1);
            }
        }
        std::fprintf(modes_f_, "%ld %.6e", istep, t);
        for (double x : wm) std::fprintf(modes_f_, " %.6e", x);
        for (double x : gm) std::fprintf(modes_f_, " %.3e", x);
        std::fprintf(modes_f_, "\n");
        std::fflush(modes_f_);
    }
    if (header) {
        std::fprintf(f, "# step  t  W_field  W_kin  W_total  max|dGauss|  max|rho|  max|divB|");
        for (size_t i = 0; i < ns; ++i) std::fprintf(f, "  N_%s", species[i].name.c_str());
        std::fprintf(f, "\n");
    }
    std::fprintf(f, "%ld %.6e %.12e %.12e %.12e %.3e %.3e %.3e", istep, t, v[0], v[1], v[0] + v[1], g, rmax, db);
    for (size_t i = 0; i < ns; ++i) std::fprintf(f, " %.0f", v[3 + i]);
    std::fprintf(f, "\n");
    std::fflush(f);
}

// Side injection (docs/specs/side_injection.md): particles outside the tube would feel only the scattered field, so the
// plasma must end 2 cells inside the faces (shape 2 gathers one cell further; a background plasma continues there as
// the cold fluid, decision 63); at t = 0 the incident field (and the fluid current) fills the tube;
// a warning if the incident field reaches an x end of the tube (open x), where the tube is not closed.
void Simulation::side_setup(bool fresh) {
    const SideConfig& S = cfg.side;
    double out = 0;
    for (const Species& s : species)
        for (int a = 1; a < 3; ++a) {
            if (!S.faces[a]) continue;
            const int m = cfg.order == 3 ? 3 : 2;      // shape 3 gathers 2 cells beyond its cell
            const double lo = (S.lo[a] + m) * dom.h[a], hi = (S.hi[a] - m) * dom.h[a];
            for (double y : s.get(a == 1 ? Species::Y : Species::Z)) out += y < lo || y > hi;
        }
    comm.allreduce_sum(&out, 1);
    if (out > 0)
        throw std::runtime_error("side: " + std::to_string(long(out)) + " particles outside the side tube (the plasma must end "
                                 "2 cells (shape 3: 3) inside its faces, docs/specs/side_injection.md)");
    if (fresh) {
        const FV f[6] = {fields->ex.view(), fields->ey.view(), fields->ez.view(), fields->bx.view(), fields->by.view(), fields->bz.view()};
        const FV jf[3] = {fields->jfx.view(), fields->jfy.view(), fields->jfz.view()};   // the fluid in the tube: j^{-1/2}
        fields->side->add_incident(f, t, true, fields->side->fluid() ? jf : nullptr);
        comm.exchange(dom, {&fields->ex, &fields->ey, &fields->ez, &fields->bx, &fields->by, &fields->bz});
    }
    if (fresh && S.wave == 1) {                        // the pattern must vanish at both x ends of the window
        const FV Ey = fields->ey.view(), Ez = fields->ez.view();
        const int nx = dom.n[0], ny = dom.n[1], N0 = dom.N[0], o0 = dom.off[0];
        const double m = comm.allreduce_max(par_max_abs(dom.owned(), [=] HD(long q) {
            const int i = int(q % nx), j = int((q / nx) % ny), k = int(q / (long(nx) * ny));
            return o0 + i == 0 || o0 + i == N0 - 1 ? fmax(fabs(Ey(i, j, k)), fabs(Ez(i, j, k))) : 0.0;
        }));
        if (comm.rank == 0 && m > 1e-4 * S.a0)
            std::fprintf(stderr, "warning: side flying focus: the incident field at an x end of the window is %.2g of a0 (the tube "
                                 "is open there): a longer window or the pulse further from the ends (rings run ahead by r theta / 2)\n",
                         m / S.a0);
    }
    if (comm.rank != 0 || !fresh) return;
    const SideTfsf& sd = *fields->side;
    const bool te = SideTfsf::split(dom, S);
    std::printf("  side: %zu plane-wave component%s with the discrete dispersion%s, polarisation of the modes within %.1e of "
                "the wanted one, one-step residual %.1e\n", sd.comps().size(), sd.comps().size() == 1 ? "" : "s",
                sd.fluid() ? (te ? " of the plasma (TE and TM waves)" : " of the plasma") : "", sd.polarisation_mismatch(),
                sd.mode_residual());
    if (sd.tabulated()) {
        std::printf("  side: flying focus of %d frequencies x %zu waves (azimuths%s), kappa = %.6g omega0; the corrections "
                    "tabulated at the face columns\n", sd.frequencies(), sd.comps().size() / size_t(std::max(1, sd.frequencies())),
                    te ? " x TE, TM" : "", sd.kappa());
        return;
    }
    if (dom.periodic[0]) return;
    double worst = 1;                                  // largest envelope at an x end of the tube during the run
    if (S.tau > 0) {
        worst = 0;
        const double al = 4 * std::log(2.0) / (S.tau * S.tau);
        const double n[3] = {std::cos(S.elevation) * std::cos(S.angle), std::cos(S.elevation) * std::sin(S.angle), std::sin(S.elevation)};
        const double y[2] = {(S.faces[1] ? S.lo[1] : 0) * dom.h[1], (S.faces[1] ? S.hi[1] : dom.N[1]) * dom.h[1]};
        const double z[2] = {(S.faces[2] ? S.lo[2] : 0) * dom.h[2], (S.faces[2] ? S.hi[2] : dom.N[2]) * dom.h[2]};
        for (double tt = 0; tt <= cfg.t_end; tt += dom.dt()) {
            const double sh = cfg.window.enabled ? std::max(0.0, std::floor((tt - cfg.window.start) * cfg.window.v / dom.h[0])) : 0.0;
            for (int e = 0; e < 2; ++e) {
                const double x = (sh + (e ? dom.N[0] - 1 : 0)) * dom.h[0];
                double umin = 1e300, umax = -1e300;
                for (int p = 0; p < 4; ++p) {
                    const double u = n[0] * (x - S.r0[0]) + n[1] * (y[p & 1] - S.r0[1]) + n[2] * (z[p >> 1] - S.r0[2]) - (tt - S.tc);
                    umin = std::min(umin, u); umax = std::max(umax, u);
                }
                const double u = umin <= 0 && umax >= 0 ? 0.0 : std::min(std::abs(umin), std::abs(umax));
                worst = std::max(worst, std::exp(-al * u * u));
            }
        }
    }
    if (worst > 1e-4)
        std::fprintf(stderr, "warning: side: the incident field reaches an x end of the tube (envelope %.2g of the peak there): the tube "
                             "is open at the x ends; use periodic x or a pulse that stays away from them\n", worst);
}
