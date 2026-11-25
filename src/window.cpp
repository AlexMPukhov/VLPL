#include "window.hpp"
#include "prof.hpp"
#include <cmath>
#include "sim.hpp"
#include "pml.hpp"
#include "filter.hpp"
#include "particles.hpp"
#include "rz.hpp"

// x is the fastest index, so f(i) = f(i + 1) over the owned cells is the whole array moved by one
// element: Field::shift_x, which moves the start of the array and copies only now and then (the
// indexed copy was 37 ms per shift on 4 threads, 2026-10-27). The y, z guards move with the data and
// stay valid; the x guards are left stale for the caller's exchange_x. left: the guard layer i = -1
// holds the old cell 0 (the left neighbour's new n-1), 0 at an open left edge.
void shift_fields_x(const Domain& d, std::vector<Field*> f, Field& tmp, bool left) {
    const int ny = d.n[1];
    const bool edge = !d.periodic[0] && d.off[0] == 0;   // nothing below the open left edge
    for (Field* F : f) {
        F->shift_x(tmp);
        if (left && edge) {
            const FV v = F->view();
            par_for(long(ny) * d.n[2], [=] HD(long q) { v(-1, int(q % ny), int(q / ny)) = 0.0; });
        }
    }
}

// iRIP (docs/specs/implicit_rip.md): the implicit solve is not causal along x, so the last cells carry a
// small precursor of Ex, Bx and the entering cell (the old zero guard) would break the Gauss law and
// div B at the point it closes. Its Ex, Bx are set from the constraints at that point (the transverse
// fields of the entering cell stay 0, the entering plasma is neutral as for RIP-CC's exact entry); with
// RIP-CC this would give 0. Only where the constraints hold: outside CPML layers (and one cell more, the
// range of the Gauss law check); inside them the entering cell stays 0 as with RIP-CC (setting it from
// the plain constraints there fed back and grew, decision 41).
static void irip_entry_cell(Fields& f, const Domain& d, Field& gauss0, const Pml* P, const Field* rhom) {
    const int m = d.n[0] - 2, ny = d.n[1];
    int lo[3] = {0, 0, 0}, hi[3] = {0, d.n[1], d.n[2]};
    for (int a = 1; a < 3; ++a)
        if (P) { lo[a] = std::max(0, P->check_lo(a) - d.off[a]); hi[a] = std::min(d.n[a], P->check_hi(a, d.N[a]) - d.off[a]); }
    const int jlo = lo[1], jhi = hi[1], klo = lo[2], khi = hi[2];
    const double D = d.h[0], iy = 1.0 / d.h[1], iz = 1.0 / d.h[2];
    const FV Ex = f.ex.view(), Ey = f.ey.view(), Ez = f.ez.view(), Bx = f.bx.view(), By = f.by.view(), Bz = f.bz.view();
    const FV G0 = gauss0.view(), RM = rhom ? rhom->view() : gauss0.view();   // O7: the medium charge -div P at the Gauss point m
    const bool med = rhom != nullptr;
    par_for(long(ny) * d.n[2], [=] HD(long q) {
        const int j = int(q % ny), k = int(q / ny);
        if (j < jlo || j >= jhi || k < klo || k >= khi) return;
        Ex(m + 1, j, k) = Ex(m, j, k) + D * (G0(m, j, k) + (med ? RM(m, j, k) : 0.0) - 0.5 * (Ey(m, j, k) - Ey(m, j - 1, k)) * iy -
                                            0.5 * (Ez(m, j, k) - Ez(m, j, k - 1)) * iz);
        Bx(m + 1, j, k) = Bx(m, j, k) - D * (0.5 * (By(m, j + 1, k) - By(m, j, k)) * iy + 0.5 * (Bz(m, j, k + 1) - Bz(m, j, k)) * iz);
    });
}

// The fields have valid guards between steps (Fields::step), so the last owned cell takes the right
// neighbour's first cell. The initial Gauss residual moves with the fields.
void Simulation::shift_window() {
    if (cfg.rz) { shift_window_rz(); return; }
    Fields& f = *fields;
    { PROF_SCOPE("window.xchg gauss0"); comm.exchange_x(dom, {&gauss0_}); }
    shift_fields_x(dom, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz, &gauss0_}, work_);
    if (f.jfx.n(0) > 0) {                          // the cold fluid of a side tube (side.hpp): its current moves too
        comm.exchange_x(dom, {&f.jfx, &f.jfy, &f.jfz});
        shift_fields_x(dom, {&f.jfx, &f.jfy, &f.jfz}, work_);
        comm.exchange_x(dom, {&f.jfx, &f.jfy, &f.jfz});
        if (dom.off[0] + dom.n[0] == dom.N[0]) {   // the entering column (and the guard beyond it) starts at rest
            const int nx = dom.n[0], sy = dom.n[1] + 2 * dom.ng, g = dom.ng;
            for (Field* J : {&f.jfx, &f.jfy, &f.jfz}) {
                const FV v = J->view();
                par_for(long(sy) * (dom.n[2] + 2 * g), [=] HD(long q) {
                    const int j = int(q % sy) - g, k = int(q / sy) - g;
                    for (int i = nx - 1; i < nx + g; ++i) v(i, j, k) = 0.0;
                });
            }
        }
    }
    if (pml_) {                                    // the CPML memory moves with the fields
        const std::vector<Field*> psi = pml_->psi_fields();
        comm.exchange_x(dom, psi);
        // the transport psi are kept at i = -1 too (the source at the left face, rip_cc_step_pml): after
        // the shift it holds the left neighbour's new cell n-1 (this rank's old cell 0), not the value of
        // one cell further left (4 ranks along x were 6 % off in a window run, decision 34)
        shift_fields_x(dom, psi, work_, true);
    }
    ++dom.wx;
    if (metal3) metal3->shift(dom, work_);          // the lab-fixed metal: the mask moves, the new column from the function
    { PROF_SCOPE("window.xchg fields"); comm.exchange_x(dom, {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz}); }   // y, z guards moved with the data
    if (diel3) {                                   // dielectric (decision 100): chi moves with the window; E_x of the entering
        diel3->shift(dom, work_);                  // column and the front layer from div D (RIP-CC)
        if (dom.solver == Solver::RipCC && dom.off[0] + dom.n[0] == dom.N[0]) dielectric3_window_entry(f, dom, gauss0_, f.dfront, *diel3);
        comm.exchange(dom, {&f.ex});
    }
    Field* rhom = nullptr;
    if (medium) {                                  // O7: the media state moves too; their charge enters the iRIP entry cell
        shift_window_medium();
        work_.fill(0);
        medium_add_rho_cart(*medium, dom, work_);  // -div P (P and bP exchanged in x by the shift; y, z guards moved with the data)
        rhom = &work_;
    }
    if (dom.solver == Solver::IRip) {
        if (dom.off[0] + dom.n[0] == dom.N[0]) irip_entry_cell(f, dom, gauss0_, pml_.get(), rhom);
        comm.exchange(dom, {&f.ex, &f.bx});                             // on every rank
    }
    if (dom.off[0] + dom.n[0] == dom.N[0] && !cfg.boost.on)   // boosted frame: the plasma enters through boost_inject
        for (size_t i = 0; i < species.size(); ++i)
            if (loaded[i]) load(cfg.species[i], species[i], int(i), dom.n[0] - 1, dom.n[0]);
    window_fold_entered();
}

// The frozen entry columns (docs/specs/moving_window.md, decision 91). 3 without the current filter: no current reaches
// the last cell. With p filter passes the closure of the filter at the right edge changes at every shift (the last Gauss
// cell becomes an inner one); T^p then differs between two shifts on the last p + 1 Gauss cells, and the charge there must be
// that of the entering plasma alone, which the field does not see either: p more frozen cells.
int Simulation::window_freeze_cells() const { return 3 + cfg.filter_reach(); }

static bool window_entry_exact(const Config& c) { return c.window.enabled && c.window.v >= 1 && !c.boost.on; }

// Cart3D: 0 outside the points of Fields::gauss_residual (the last cell at an open right edge, the CPML and x layers),
// where gauss0_ is 0 too; RZ masks in its diagnostics instead.
static void mask_gauss_points(const Fields& f, const Domain& d, Field& x) {
    int lo[3], hi[3];
    f.check_range(lo, hi);
    const int iend = !d.periodic[0] && d.off[0] + d.n[0] == d.N[0] ? d.n[0] - 1 : d.n[0];
    const FV X = x.view();
    for_owned(d, [=] HD(int i, int j, int k) {
        if (i >= iend || i < lo[0] || i >= hi[0] || j < lo[1] || j >= hi[1] || k < lo[2] || k >= hi[2]) X(i, j, k) = 0;
    });
}

// The column that leaves the frozen cells with this shift (window cell N - F - 1, loaded F shifts ago; the columns of the
// first F shifts are the plasma of the initial load, in gauss0_ already): its charge, filtered as rho, is a background
// from now on (gauss0_ -= its rho). Its cells are F + 1 cells behind the edge, out of reach of the changing closure.
void Simulation::window_fold_entered() {
    const int F = window_freeze_cells();
    if (!window_entry_exact(cfg) || dom.wx <= F) return;
    const double h = dom.h[0], x0 = double(dom.wx + dom.N[0] - F - 1) * h, x1 = x0 + h;
    work_.fill(0);
    if (cfg.rz) {
        for (size_t i = 0; i < species.size(); ++i) rz_rho_range(species[i], *rz, work_, cfg.order, x0, x1);
        comm.add_guards(dom, {&work_});
        rz_smooth_rho(*rz, comm, work_);
        rz_filter_rho(*rz, comm, work_, cfg.filter_passes);
    } else {
        for (size_t i = 0; i < species.size(); ++i) deposit_rho(species[i], work_, dom, cfg.order, x0, x1);
        comm.add_guards(dom, {&work_});
        if (cfg.filter_passes > 0) filter_rho_cart(work_, filt_, dom, comm, cfg.filter_passes);
        mask_gauss_points(*fields, dom, work_);
    }
    const FV G = gauss0_.view(), R = work_.view();
    for_owned(dom, [=] HD(int i, int j, int k) { G(i, j, k) -= R(i, j, k); });
}

// The charge of the columns still frozen, filtered as rho (min(F, wx) cells at the right edge, unmoved since their load).
bool Simulation::window_frozen_rho(Field& out) {
    const int m = int(std::min<long>(window_freeze_cells(), dom.wx));
    if (!window_entry_exact(cfg) || m <= 0) return false;
    if (out.n(0) == 0) out.resize(dom);
    const double x0 = double(dom.wx + dom.N[0] - m) * dom.h[0];
    out.fill(0);
    if (cfg.rz) {
        for (size_t i = 0; i < species.size(); ++i) rz_rho_range(species[i], *rz, out, cfg.order, x0, 1e300);
        comm.add_guards(dom, {&out});
        rz_smooth_rho(*rz, comm, out);
        rz_filter_rho(*rz, comm, out, cfg.filter_passes);
    } else {
        for (size_t i = 0; i < species.size(); ++i) deposit_rho(species[i], out, dom, cfg.order, x0, 1e300);
        comm.add_guards(dom, {&out});
        if (cfg.filter_passes > 0) filter_rho_cart(out, filt_, dom, comm, cfg.filter_passes);
        mask_gauss_points(*fields, dom, out);
    }
    return true;
}

// Shifts of the window after the step that ends at t.
long window_shifts(const WindowConfig& W, double t, double h) {
    return t > W.start ? long(std::floor((t - W.start) * W.v / h + 1e-9)) : 0;
}

// At the speed of light no particle moves towards the right edge of the window. In a step that ends
// with a shift the particles of the three entry columns stay unpushed: then no current reaches the
// last cell, whose fields are the zero guard, and the Gauss law stays exact (docs/specs/moving_window.md).
static void set_freeze(Simulation& s, double t_next) {
    const WindowConfig& W = s.cfg.window;
    // boosted frame: the plasma moves at -beta c and enters 3 + filter reach cells behind the edge (boost.cpp), nothing is frozen
    const bool shifts = W.v >= 1 && !s.cfg.boost.on && window_shifts(W, t_next, s.dom.h[0]) > s.dom.wx;
    s.dom.xfreeze = shifts ? double(s.dom.wx + s.dom.N[0] - s.window_freeze_cells()) * s.dom.h[0] : 1e300;
}

class MovingWindow : public Operator {
public:
    const char* name() const override { return "moving_window"; }
    Stage stage() const override { return Stage::Fields; }
    void setup(Simulation& s) override { set_freeze(s, double(s.istep + 1) * s.dom.dt()); }
    void apply(Simulation& s) override {
        const double t1 = double(s.istep + 1) * s.dom.dt();          // the fields are at t_{n+1} now
        const long target = window_shifts(s.cfg.window, t1, s.dom.h[0]);
        while (s.dom.wx < target) s.shift_window();
        set_freeze(s, t1 + s.dom.dt());                               // for the push of the next step
    }
};

std::unique_ptr<Operator> make_window_operator() { return std::make_unique<MovingWindow>(); }
