// Lorentz-boosted frame: deck, the transformation of the lab inputs, the particle lattices and the injection of the
// plasma (docs/specs/boosted_frame.md, decision 87). Lab-frame diagnostics: boost_diag.cpp.
#include "boost.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "sim.hpp"

static const double c_light = 299792458.0;

void boost_from_deck(const Deck& d, Config& c, double k0) {
    if (d.has_block("boost")) {
        c.boost.gamma = d.num("boost", "gamma", Kind::None);
        c.boost.on = d.flag("boost", "enabled", true);
        if (!c.boost.on) c.boost.gamma = 1;
    }
    if (d.has_block("lab_diagnostics")) {
        const std::string b = "lab_diagnostics";
        LabDiagConfig& L = c.labdiag;
        const double t0 = d.num(b, "t_first", Kind::Time) * c_light * k0;
        const double dt = d.num(b, "t_every", Kind::Time, 0.0) * c_light * k0;
        const long n = d.integer(b, "count", 1);
        if (n < 1 || (n > 1 && !(dt > 0))) throw std::runtime_error("lab_diagnostics: count >= 1, t_every > 0 for count > 1");
        for (long k = 0; k < n; ++k) L.times.push_back(t0 + double(k) * dt);
        L.fields = d.flag(b, "fields", true);
        std::istringstream is(d.str(b, "species", ""));
        for (std::string w; is >> w;) L.species.push_back(w);
        L.gamma_min = d.num(b, "gamma_min", Kind::None, 1.0);
        L.particles = d.flag(b, "particles", true);
        L.select_uz = d.num(b, "select_uz", Kind::None, -1e300);
        L.select_r = d.has(b, "select_r") ? d.num(b, "select_r", Kind::Length) * k0 : 1e300;
        if (!c.boost.on) L = LabDiagConfig{};            // boost.enabled = false: the same deck as a lab run
    }
}

// The boosted time at which the lab time at the box's left edge reaches T (the lab plane t = T has left the box):
// the edge stands at x' = 0 until the window starts (start' = c.window.start, boosted), then moves at c.
static double boosted_time_of(const Config& c, double T) {
    const BoostConfig& B = c.boost;
    const double s0 = c.window.enabled ? c.window.start : 1e300;
    const double ts = (T - B.T0) / B.gamma;              // static edge x' = 0
    if (ts <= s0) return ts;
    return (T - B.T0 + B.gamma * B.beta * s0) / (B.gamma * (1 + B.beta));
}

void apply_boost(Config& c) {
    BoostConfig& B = c.boost;
    if (!B.on) {                                         // a lab run: the antenna is the entrance plane x = 0
        c.laser.antenna = false;
        for (LaserConfig& L : c.extra_lasers) L.antenna = false;
        return;
    }
    if (B.beta > 0) return;                              // already applied
    if (!(B.gamma > 1)) throw std::runtime_error("boost.gamma must be > 1");
    const std::string g = "boost: ";
    if (c.side.on) throw std::runtime_error(g + "no side block yet");
    if (!c.media.empty()) throw std::runtime_error(g + "no medium blocks yet");
    if (!c.file_lasers.empty() && !c.window.enabled)   // the antenna leaves the box when the window starts (decision 97)
        throw std::runtime_error(g + "laser files need a moving window (the antenna emits them at the lab plane x = 0 until it starts)");
    if (c.mover_implicit) throw std::runtime_error(g + "numerics.mover = explicit only so far");
    if (c.rings.on || c.metal_rz.on || !c.dielectrics.empty() || !c.dielectrics3.empty() || c.plane_z >= 0)
        throw std::runtime_error(g + "no rings, metal or dielectric blocks or output.plane_z (lab objects) yet");
    B.beta = std::sqrt(1 - 1 / (B.gamma * B.gamma));
    const double s = B.stretch();
    B.dx_lab = c.h[0];
    c.h[0] *= s;
    double tcmax = 0;
    for (const LaserConfig& L : c.all_lasers())
        if (L.enabled) tcmax = std::max(tcmax, L.tc);
    if (c.window.enabled) {
        if (c.window.v != 1) throw std::runtime_error(g + "window.velocity = 1 only (the window moves at c in both frames)");
        if (tcmax > 0 && c.window.start < 2 * tcmax)
            throw std::runtime_error("window.start must be >= 2 laser.delay (the pulse must have entered before the window moves)");
        B.start_lab = c.window.start;
        c.window.start *= s;                             // the end of the injection at x' = 0 maps as t - x: times s
    }
    B.t_ref = c.window.enabled ? c.window.start : 2 * tcmax * s;
    B.X0 = B.T0 = -B.gamma * B.beta * B.t_ref;
    B.t_end_lab = c.t_end;
    c.t_end = boosted_time_of(c, c.t_end) + 2 * c.dt();   // two steps more: the last lab plane has left the box
    auto lasers = [&](LaserConfig& L) {
        if (!L.enabled) return;
        if (L.tfsf_x >= 0 && !c.rz)
            throw std::runtime_error(g + "laser.tfsf_x in Cart3D not yet (its incident grid starts from the pulse at t = 0); "
                                         "use the open left edge, or geometry = rz");
        if (L.antenna && c.solver == Solver::Yee) throw std::runtime_error(g + "laser.injection = antenna: rip_cc or irip (its charge stencil)");
        if (L.antenna && L.tfsf_x >= 0) throw std::runtime_error(g + "laser.injection = antenna: without tfsf_x (the antenna is at x = 0)");
        if (L.angle != 0 || L.tfsf_modes) throw std::runtime_error(g + "lasers at normal incidence (no angle, no tfsf_incident = modes)");
        if (L.tfsf_x >= 0) L.tfsf_x *= s;                // the same cell of the box
        L.bg = B.gamma; L.bb = B.beta; L.bX0 = B.X0; L.bT0 = B.T0;
    };
    lasers(c.laser);
    for (LaserConfig& L : c.extra_lasers) lasers(L);
    for (SpeciesConfig& sc : c.species) {
        for (int a = 0; a < 3; ++a) sc.u0_lab[a] = sc.u0[a];
        const bool beam = sc.self_field || sc.u0[0] != 0;
        if (sc.photon) continue;                         // photons: empty at the start, made and moved by the qed module (decision 117)
        if (!sc.mobile) throw std::runtime_error(g + "species " + sc.name + ": mobile = false (at rest in the lab, it moves at -beta c "
                                                     "here): use a mobile species");
        double up[3];
        boost_momentum(B.gamma, B.beta, sc.u0_lab, up);
        for (int a = 0; a < 3; ++a) sc.u0[a] = up[a];
        if (!beam) {                                     // a plasma at rest in the lab
            if (sc.u0_lab[1] != 0 || sc.u0_lab[2] != 0) throw std::runtime_error(g + "species " + sc.name + ": a plasma at rest in the lab "
                                                                                    "(u = 0; beams: ux != 0 or self_field)");
            if (sc.t_start > 0) throw std::runtime_error(g + "species " + sc.name + ": t_start for beams only");
            sc.boost_kind = 1;
            continue;
        }
        if (!(sc.rmin[0] > -1e299 && sc.rmax[0] < 1e299))
            throw std::runtime_error(g + "species " + sc.name + ": a beam needs " + (c.rz ? "zmin and zmax" : "xmin and xmax"));
        sc.boost_kind = 2;
        const double gl = std::sqrt(1 + sc.u0_lab[0] * sc.u0_lab[0] + sc.u0_lab[1] * sc.u0_lab[1] + sc.u0_lab[2] * sc.u0_lab[2]);
        sc.beam_v = sc.u0_lab[0] / gl;
        sc.beam_t0 = sc.t_start;                         // lab time of the deck's beam (xmin .. xmax)
        const double xc = 0.5 * (sc.rmin[0] + sc.rmax[0]);
        sc.t_start = std::max(0.0, B.tp_of(xc, sc.beam_t0)); // loaded when its centre is at that lab event
        if (sc.density != 0 && sc.file.path.empty()) {   // decision 118: a beam the boosted lattice misses
            const double n = boost_beam_planes(c, sc);
            if (n < 1) {
                char m[400];
                std::snprintf(m, sizeof m, "species %s: the boosted lattice holds %.3g planes of the beam along the boost axis (it is "
                                           "%.3g x shorter in boosted cells): it would load almost no charge; set ppc_x >= %d (%d for 20 "
                                           "planes; decision 118)", sc.name.c_str(), n, c.h[0] * B.gamma * (1 - B.beta * sc.beam_v) / B.dx_lab,
                              boost_beam_ppc_for(c, sc, 1), boost_beam_ppc_for(c, sc, 20));
                throw std::runtime_error(g + m);
            }
        }
    }
}

double boost_beam_planes(const Config& c, const SpeciesConfig& sc) {
    const BoostConfig& B = c.boost;
    const double a = 1 / (B.gamma * (1 - B.beta * sc.beam_v));     // x' = a x_lab + b (boost_lattice)
    return (sc.rmax[0] - sc.rmin[0]) * a * sc.ppc[0] / c.h[0];
}

int boost_beam_ppc_for(const Config& c, const SpeciesConfig& sc, double planes) {
    const double n = boost_beam_planes(c, sc);
    return n > 0 ? int(std::ceil(planes * sc.ppc[0] / n - 1e-9)) : 0;
}

std::string boost_beam_warning(const Config& c) {
    std::string w;
    if (!c.boost.on) return w;
    for (const SpeciesConfig& sc : c.species) {
        if (sc.boost_kind != 2 || sc.density == 0 || !sc.file.path.empty()) continue;
        const double n = boost_beam_planes(c, sc);
        if (n >= 20) continue;
        char m[300];
        std::snprintf(m, sizeof m, "%sspecies %s: %.3g lattice planes across the beam along the boost axis: its charge is quantized "
                                   "(error ~ 1/%.3g); ppc_x >= %d gives 20 planes (decision 118)", w.empty() ? "" : "; ", sc.name.c_str(),
                      n, n, boost_beam_ppc_for(c, sc, 20));
        w += m;
    }
    return w;
}

double boost_beam_deck_charge(const Config& c, const SpeciesConfig& sc) {
    const int M = 4096;                                  // along the boost axis: midpoints, exact for the bounds
    const double x0 = sc.rmin[0], dxs = (sc.rmax[0] - sc.rmin[0]) / M;
    const bool fn = !sc.density_fn.empty();
    double sum = 0;
    if (c.rz) {                                          // exact in r (midpoints), the volume 2 pi r dr dz
        const double r0 = std::max(0.0, sc.rmin[1]), r1 = std::min(sc.rmax[1], c.N[1] * c.h[1]);
        const int MR = 1024;
        const double dr = (r1 - r0) / MR;
        for (int j = 0; j < MR && r1 > r0; ++j) {
            const double r = r0 + (j + 0.5) * dr;
            double line = 0;
            for (int i = 0; i < M; ++i) {
                const double f = fn ? sc.density_fn({x0 + (i + 0.5) * dxs, r}) : 1.0;
                if (f > 0) line += f;
            }
            sum += 2 * M_PI * r * dr * line;
        }
        return sc.density * sum * dxs;
    }
    // transversely the loader's lattice: (cell + (k + 1/2) / ppc) h inside [rmin, rmax)
    std::vector<double> ys, zs;
    for (int j = 0; j < c.N[1] * sc.ppc[1]; ++j) {
        const double y = (j + 0.5) * c.h[1] / sc.ppc[1];
        if (y >= sc.rmin[1] && y < sc.rmax[1]) ys.push_back(y);
    }
    for (int k = 0; k < c.N[2] * sc.ppc[2]; ++k) {
        const double z = (k + 0.5) * c.h[2] / sc.ppc[2];
        if (z >= sc.rmin[2] && z < sc.rmax[2]) zs.push_back(z);
    }
    if (!fn) sum = double(ys.size()) * double(zs.size()) * M;
    else
        for (double z : zs)
            for (double y : ys)
                for (int i = 0; i < M; ++i) {
                    const double f = sc.density_fn({x0 + (i + 0.5) * dxs, y, z});
                    if (f > 0) sum += f;
                }
    return sc.density * sum * dxs * (c.h[1] / sc.ppc[1]) * (c.h[2] / sc.ppc[2]);
}

double boost_beam_charge_check(Simulation& s, size_t i) {
    const SpeciesConfig& sc = s.cfg.species[i];
    const Species& sp = s.species[i];
    const double* W = sp.ptr<double>(Species::W);
    double q = par_sum(long(sp.size()), [=] HD(long p) { return W[p]; });
    s.comm.allreduce_sum(&q, 1);
    const double d = boost_beam_deck_charge(s.cfg, sc), rel = d > 0 ? q / d - 1 : 0.0;
    if (s.comm.rank == 0) {
        std::printf("  boost: beam %s loaded at boosted t = %.6g: charge (sum w) %.6g, the deck's %.6g (%+.2f %%; %.3g lattice planes "
                    "along the boost axis)\n", sc.name.c_str(), s.t, q, d, 100 * rel, boost_beam_planes(s.cfg, sc));
        if (std::abs(rel) > 0.01)
            std::fprintf(stderr, "warning: boost: beam %s: the loaded charge differs from the deck's by %+.2f %% (decision 118): raise "
                                 "ppc_x (%.3g planes along the boost axis; >= %d for 20) or check the bounds against the transverse "
                                 "cells\n", sc.name.c_str(), 100 * rel, boost_beam_planes(s.cfg, sc), boost_beam_ppc_for(s.cfg, sc, 20));
    }
    return rel;
}

// The lattice of species i loaded at the boosted time t (load_species, delayed loads).
Lattice boost_lattice(const Config& c, const SpeciesConfig& sc, const Domain& d, double t) {
    const BoostConfig& B = c.boost;
    Lattice L;
    L.boost = true; L.g = B.gamma; L.b = B.beta;
    if (sc.boost_kind == 1) {                            // plasma: q = x' + beta t = (x - X0) / gamma, comoving with it
        L.ax = B.gamma; L.bx = B.X0; L.sx = -B.beta * t;
        if (!d.periodic[0]) L.qhi = boost_inject_q(c, d, long(std::lround(t / d.dt())));
        return L;
    }
    // a beam at lab x_e at the lab time t0, moving at v: its event with boosted time t, x' = a x_e + b (ballistic)
    const double v = sc.beam_v, t0 = sc.beam_t0;
    auto xp = [&](double xe) {
        const double tl = (t / B.gamma + B.T0 + B.beta * (xe - v * t0 - B.X0)) / (1 - B.beta * v);
        const double xl = xe + v * (tl - t0);
        return B.gamma * ((xl - B.X0) - B.beta * (tl - B.T0));
    };
    const double b = xp(0.0), a = 1 / (B.gamma * (1 - B.beta * v));
    L.ax = 1 / a; L.bx = -b / a; L.sx = 0;
    return L;
}

// The lattice coordinate q = x' + beta t' up to which the plasma has entered at step n: the injection plane 3 + R cells behind
// the right edge of the window, R = cfg.filter_reach() (as the F = 3 + R frozen columns of the lab window, decisions 22, 91,
// docs/specs/moving_window.md): no current of the entering plasma reaches the last cell, and with the current filter the
// entering plasma stays out of the last R + 1 Gauss cells, where the filter's closure changes at every shift of the window
// (decision 108: with 2 passes and plasma entering 3 cells behind, warm electrons left 1e-10 in the Gauss law).
double boost_inject_q(const Config& c, const Domain& d, long n) {
    const double t = double(n) * d.dt();
    const long wx = c.window.enabled ? window_shifts(c.window, t, d.h[0]) : 0;
    return double(wx + d.N[0] - 3 - c.filter_reach()) * d.h[0] + c.boost.beta * t;
}

namespace {
class BoostInject : public Operator {
public:
    const char* name() const override { return "boost_inject"; }
    Stage stage() const override { return Stage::Fields; }
    void apply(Simulation& s) override {
        const Domain& d = s.dom;
        if (d.periodic[0] || d.off[0] + d.n[0] != d.N[0]) return;
        const double qa = boost_inject_q(s.cfg, d, s.istep), qb = boost_inject_q(s.cfg, d, s.istep + 1);
        const double t1 = double(s.istep + 1) * d.dt(), h = d.h[0];
        const long gi0 = long(std::floor(qa / h));
        const int nx = int(long(std::ceil(qb / h)) - gi0);
        for (size_t i = 0; i < s.species.size(); ++i) {
            const SpeciesConfig& sc = s.cfg.species[i];
            if (sc.boost_kind != 1 || !s.loaded[i]) continue;
            Lattice L = boost_lattice(s.cfg, sc, d, t1);
            L.qlo = qa; L.qhi = qb;
            if (s.cfg.rz) s.load_rz_cells(sc, s.species[i], int(i), gi0, nx, L);
            else s.load_cells(sc, s.species[i], int(i), gi0, nx, L);
        }
    }
};
}  // namespace

std::unique_ptr<Operator> make_boost_inject_operator() { return std::make_unique<BoostInject>(); }

bool boost_beam_cut(const Config& c, const LaserConfig& L, double& w, double& d, double& half) {
    const BoostConfig& B = c.boost;
    const double zr = 0.5 * L.w0 * L.w0;
    // the antenna (decision 89) emits at the lab plane x = 0 only; the entrance plane at lab x in [X0, 0]
    d = L.antenna ? std::abs(L.xf) : std::max(std::abs(L.xf - B.X0), std::abs(L.xf));
    w = L.w0 * std::sqrt(1 + d * d / (zr * zr));
    half = c.rz ? c.N[1] * c.h[1] : 0.5 * c.N[1] * c.h[1];
    // only when the boost makes the difference: a beam already wider than the box at its focus (a plane wave over a
    // periodic y, waist >> box) is the deck's choice, the same in the lab run
    return L.w0 > 0 && 2.5 * w > half && 2.5 * L.w0 <= half;
}

// bench/nci/boost_stability (decision 104). A plasma streaming at -beta c through the boosted grid (periodic box, warm, gamma 2 .. 20,
// h_perp = c dt / 4 .. 3 c dt, RIP-CC and iRIP, 2D, 3D and RZ; and boosted LWFAs with the window) is unstable without the binomial
// current filter: the field energy grows from the particle noise by 2 - 4 orders within 10 - 300 / omega_p, up to ~1e-2 of the
// drift energy, with ~20 % density modulation and, in RZ, 17 - 48 % excess density on the axis. Measured:
//   no filter: shape 2 stable at gamma 2 and 3, unstable from gamma 5; shape 1 unstable from gamma 3;
//   shape 1: unstable at gamma >= 5 also with one and two passes;
//   shape 2, one pass: flat for 100 / omega_p, then a slow growth (x10 - x100 by 300 / omega_p) where h_perp < c dt;
//   shape 2, two passes: flat in every case (<= 6e-6 of the drift energy);
//   filter = projection, local: smooth only the longitudinal current, unstable for a drifting plasma (decision 91).
// The plasma stays in the boosted box for about omega_p T = k_p L_lab (the lab window length; without a window
// k_p L_lab (1 + beta) / beta), its proper time.
std::string boost_nci_risk(const Config& c) {
    if (!c.boost.on) return "";
    bool plasma = false;
    double wp2 = 0;
    for (const SpeciesConfig& s : c.species)
        if (s.boost_kind == 1) { plasma = true; wp2 += s.density * s.q2() / s.m; }
    if (!plasma) return "";
    const double g = c.boost.gamma, B = c.boost.beta;
    const double T = std::sqrt(wp2) * c.N[0] * c.boost.dx_lab * (c.window.enabled ? 1.0 : (1 + B) / B);
    char b[200];
    if (c.filter_passes > 0 && (c.filter_local || !c.filter_binomial))
        return std::string("numerics.filter = ") + (c.filter_local ? "local" : "projection") +
               " smooths only the longitudinal current: unstable for a drifting plasma (decision 91); use filter = binomial";
    if (c.order < 2 && g > 2.5) {
        std::snprintf(b, sizeof b, "numerics.shape = 1 at gamma %.3g (unstable also with one and two filter passes)", g);
        return b;
    }
    if (c.filter_passes == 0 && g > 3) {
        std::snprintf(b, sizeof b, "numerics.filter_passes = 0 at gamma %.3g", g);
        return b;
    }
    if (c.filter_passes == 1 && T > 100) {
        std::snprintf(b, sizeof b, "one filter pass while the plasma stays omega_p t = %.0f in the box (> 100: slow growth, use 2 passes)", T);
        return b;
    }
    return "";
}

// The neutrality of the boosted injection (decision 108; Alexander: "the boosted injection has to be absolutely neutral. This is
// a precondition."). BoostInject creates the plasma at rest in the lab (boost_kind 1) 3 + R cells behind the right edge with no
// creation current: any net charge it creates appears with no field and the Gauss law drifts there. The injected species are
// loaded on lattices that depend only on ppc, the bounds and the slab (load_cells, load_rz_cells: no random positions), with
// the weight density x profile at each lattice point. Lattice groups (decision 114, Alexander: "each pre-ionized ion species
// brings its own electrons with the same ppc and matching weight"): the injected species with a nonzero density are grouped by
// their lattice (ppc, bounds and slab); the injection is neutral at every particle position (to round-off) iff every group is
// neutral on its own, sum_{s in group} q_s Z_s n_s f_s(x) = 0 at every point (Z_s: the initial charge state of an ionizable
// species, else 1). Groups may differ in ppc and may overlap in space (each cancels point by point, so the sum does too), e.g.
// an inner zone 3 x 2 x 8 and an outer zone 1 x 1 x 6 (two groups side by side), or He2+ with its electrons at one ppc and N3+
// with its electrons at another in the same cells. boost_net_charge measures one group (or, without a key, all injected species
// together): max |sum q Z n f| / max sum |q Z| n |f| over the lab path of the window (4000 points along x; transverse points
// across the group's bounds within the box; RZ: radii and angles). boost_check_neutral throws when a group is not neutral to
// 1e-12, naming the group's species, the residual and, for one negative species in it, the density that would make it neutral.
static double inj_charge(const SpeciesConfig& s) { return s.ion.on || s.ion.states ? s.q * std::max(0, s.ion.z0) : s.q; }
static bool injected(const SpeciesConfig& s) { return s.boost_kind == 1 && s.density != 0 && !s.photon; }

bool boost_same_lattice(const SpeciesConfig& s, const SpeciesConfig& t) {
    bool same = true;
    for (int a = 0; a < 3; ++a) same = same && s.ppc[a] == t.ppc[a] && s.rmin[a] == t.rmin[a] && s.rmax[a] == t.rmax[a];
    return same && s.slab_x0 == t.slab_x0 && s.slab_y0 == t.slab_y0 && s.slab_thickness == t.slab_thickness &&
           s.slab_angle == t.slab_angle && s.slab_ramp == t.slab_ramp && s.slab_ramp_extent == t.slab_ramp_extent &&
           s.slab_rear_ramp == t.slab_rear_ramp && s.slab_rear_extent == t.slab_rear_extent && s.slab_exact == t.slab_exact;
}

double boost_net_charge(const Config& c, double* where, const SpeciesConfig* key) {
    if (!c.boost.on) return 0;
    const BoostConfig& B = c.boost;
    const double x0 = B.X0, x1 = B.X0 + c.N[0] * B.dx_lab + B.t_end_lab;
    const double W1 = c.N[1] * c.h[1], W2 = c.N[2] * c.h[2];
    double a1 = 0, b1 = W1, a2 = 0, b2 = W2;             // transverse sample range: the box, or the group's bounds within it
    if (key) {
        a1 = std::max(a1, key->rmin[1]); b1 = std::min(b1, key->rmax[1]);
        a2 = std::max(a2, key->rmin[2]); b2 = std::min(b2, key->rmax[2]);
        if (!(b1 > a1)) return 0;
        if (!c.rz && !c.flat && !(b2 > a2)) return 0;
    }
    const int nx = 4000, ny = 9, nz = c.rz || c.flat ? 1 : 9, nphi = c.rz ? 4 : 1;
    double worst = 0, scale = 0;
    for (int i = 0; i <= nx; ++i) {
        const double x = x0 + (x1 - x0) * i / nx;
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
                for (int l = 0; l < nphi; ++l) {
                    const double y = c.rz ? a1 + (b1 - a1) * (1 - 1e-9) * j / (ny - 1) : a1 + (b1 - a1) * (j + 0.5) / ny;
                    const double z = nz > 1 ? a2 + (b2 - a2) * (k + 0.5) / nz : 0.5 * W2;
                    const double ph = 2 * M_PI * l / nphi;
                    double net = 0, tot = 0;
                    for (const SpeciesConfig& s : c.species) {
                        if (!injected(s) || (key && !boost_same_lattice(s, *key))) continue;
                        const double tr = y, t2 = z;
                        if (x < s.rmin[0] || x >= s.rmax[0] || tr < s.rmin[1] || tr >= s.rmax[1]) continue;
                        if (!c.rz && !c.flat && (t2 < s.rmin[2] || t2 >= s.rmax[2])) continue;
                        double f = 1;
                        if (!s.density_fn.empty())
                            f = c.rz ? s.density_fn({x, y, y * std::cos(ph), y * std::sin(ph), ph}) : s.density_fn({x, y, z});
                        if (!(f > 0)) continue;                  // the loaders skip f <= 0 (and NaN)
                        net += inj_charge(s) * s.density * f; tot += std::abs(inj_charge(s)) * s.density * f;
                    }
                    if (std::abs(net) > worst && where) { where[0] = x; where[1] = y; where[2] = c.rz ? ph : z; }
                    worst = std::max(worst, std::abs(net)); scale = std::max(scale, tot);
                }
    }
    return scale > 0 ? worst / scale : 0;
}

void boost_check_neutral(const Config& c) {
    if (!c.boost.on || !(c.boost.beta > 0)) return;
    std::vector<const SpeciesConfig*> keys;              // one representative per lattice group
    for (const SpeciesConfig& s : c.species) {
        if (!injected(s)) continue;
        bool known = false;
        for (const SpeciesConfig* k : keys) known = known || boost_same_lattice(s, *k);
        if (!known) keys.push_back(&s);
    }
    const double um = c.lambda0 / (2 * M_PI) * 1e6;
    for (const SpeciesConfig* key : keys) {
        std::ostringstream list;
        list.precision(10);
        double pos = 0;
        int nneg = 0;
        const SpeciesConfig* eneg = nullptr;
        for (const SpeciesConfig& s : c.species) {
            if (!injected(s) || !boost_same_lattice(s, *key)) continue;
            const double qz = inj_charge(s);
            list << " " << s.name << " (charge " << qz << ", density " << s.density << " n_c)";
            if (qz > 0) pos += qz * s.density;
            if (qz < 0) { ++nneg; eneg = &s; }
        }
        double w[3] = {0, 0, 0};
        const double r = boost_net_charge(c, w, key);
        if (r <= 1e-12) continue;
        char b[700];
        std::snprintf(b, sizeof b, "boost: the plasma injected by the boosted window is not neutral: the lattice group of %s (ppc %d %d %d) "
                                   "has max |sum q Z n| = %.3g of sum |q Z| n (at lab %s = %.6g um, %s = %.6g um); species of the group:",
                      key->name.c_str(), key->ppc[0], key->ppc[1], key->ppc[2], r, c.rz ? "z" : "x", w[0] * um, c.rz ? "r" : "y", w[1] * um);
        std::string m = b + list.str();
        if (nneg == 1 && pos > 0) {
            std::snprintf(b, sizeof b, "; with these densities the electrons (%s) are neutral at density = %.10g (now %.10g)",
                          eneg->name.c_str(), pos / std::abs(inj_charge(*eneg)), eneg->density);
            m += b;
        }
        m += ". The injection creates particles with no current, so a net charge stays at the injection plane and the Gauss law drifts. "
             "Species on one lattice (the same ppc, " + std::string(c.rz ? "zmin zmax rmin rmax" : "xmin .. zmax") + " and slab) must "
             "cancel point by point: give each pre-ionized ion species its electrons on the same lattice with exactly neutral densities "
             "and profiles (decisions 108, 114)";
        throw std::runtime_error(m);
    }
}

void boost_summary(FILE* f, const Config& c) {
    const BoostConfig& B = c.boost;
    if (!B.on) return;
    const double s = B.stretch(), nl = std::ceil(B.t_end_lab / B.dx_lab - 0.5), nb = std::ceil(c.t_end / c.dt() - 0.5);
    std::fprintf(f, "  boosted frame: gamma = %.6g, beta = %.10g; the deck is in the lab frame\n", B.gamma, B.beta);
    std::fprintf(f, "    d%s = %.4g (lab %.4g, x gamma (1 + beta) = %.4g); laser omega' = %.4g omega0; t_end lab %.6g -> boosted "
                    "%.6g (1/omega0)\n", c.rz ? "z" : "x", c.h[0], B.dx_lab, s, B.gamma * (1 - B.beta), B.t_end_lab, c.t_end);
    std::fprintf(f, "    steps %.0f (the lab run on the same lab grid: %.0f, ratio %.2f); boosted origin = lab event x = t = %.6g\n",
                 nb, nl, nl / std::max(1.0, nb), B.X0);
    if (c.window.enabled) std::fprintf(f, "    window start lab %.6g -> boosted %.6g\n", B.start_lab, c.window.start);
    for (const SpeciesConfig& sc : c.species)
        if (!sc.photon)                                  // photons: made by the qed module (decision 117)
        std::fprintf(f, "    species %s: %s, boosted u0 = %.6g %.6g %.6g%s\n", sc.name.c_str(),
                     sc.boost_kind == 1 ? "plasma at rest in the lab (injected 3 + filter reach cells behind the right edge)" : "beam (ballistic map)",
                     sc.u0[0], sc.u0[1], sc.u0[2],
                     sc.boost_kind == 2 ? (", loaded at boosted t = " + std::to_string(sc.t_start)).c_str() : "");
    // the lasers enter at lab x in [X0, 0]: their beam there (far before a distant focus) must fit the transverse box
    for (const LaserConfig& Lz : c.all_lasers()) {
        if (!Lz.enabled || !(Lz.w0 > 0)) continue;
        double w, d, half;
        const bool cut = boost_beam_cut(c, Lz, w, d, half);
        if (Lz.antenna)
            std::fprintf(f, "    laser: antenna at the lab plane x = 0 (moving at -beta c from x' = %.4g, 1/k0): its waist there %.4g "
                            "(the transverse %s %.4g)\n", boost_antenna_start(c), w, c.rz ? "wall at" : "half box", half);
        if (Lz.antenna) {   // the near field of the moving sheet (bench/antenna_nearfield, docs/specs/boosted_frame.md "Antenna")
            const double dnf = 1.1 * B.gamma * (1 + B.beta) * c.dt();
            std::fprintf(f, "      antenna near field: the emitted field reaches the lab field within 1 %% only %.4g (1/k0) = %.3g lambda0 "
                            "of lab x downstream of x = 0 (1.1 gamma (1 + beta) c dt', one boosted step after emission; weaker "
                            "nearer: 0.65 at a tenth of it, 0.9 at half)\n", dnf, dnf / (2 * M_PI));
        }
        else
            std::fprintf(f, "    laser enters at lab x in [%.4g, 0] (1/k0): its waist there up to %.4g (the transverse %s %.4g)\n", B.X0, w,
                         c.rz ? "wall at" : "half box", half);
        if (cut)
            std::fprintf(stderr, "warning: boost: the laser beam at its lab entrance events is %.3g wide (1/k0, focus %.3g away); the "
                                 "transverse box (%.3g) cuts it: %s(docs/specs/boosted_frame.md)\n",
                         w, d, half, Lz.antenna ? "widen the box or bring the focus nearer "
                                                : "laser.injection = antenna (only the lab plane x = 0), a wider box or a nearer focus ");
    }
    const std::string bw = boost_beam_warning(c);       // decision 118
    if (!bw.empty()) std::fprintf(stderr, "warning: boost: %s\n", bw.c_str());
    const std::string nci = boost_nci_risk(c);
    if (!nci.empty())
        std::fprintf(stderr, "warning: boost: %s: the plasma streaming through the grid is unstable (numerical Cherenkov: field noise "
                             "x100 - x10^4 within 10 - 100 / omega_p, density modulation, RZ: excess density on the axis; "
                             "bench/nci/boost_stability/README.md); use numerics.shape = 2 and numerics.filter_passes = 1 "
                             "(binomial; 2 when omega_p t > 100)\n",
                     nci.c_str());
    const LabDiagConfig& L = c.labdiag;
    if (!L.times.empty())
        std::fprintf(f, "    lab_diagnostics: %zu lab times %.6g .. %.6g (1/omega0)%s%s; plane spacing dt'/(gamma beta) = %.4g\n",
                     L.times.size(), L.times.front(), L.times.back(), L.fields ? ", fields" : "",
                     L.species.empty() ? "" : ", particles", c.dt() / (B.gamma * B.beta));
}
